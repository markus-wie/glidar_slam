#include "glidar_slam/mapping/map_builder.hpp"

#include <cmath>
#include <unordered_set>
#include <utility>

#include "glidar_slam/logger/logger.hpp"
#include "glidar_slam/utils.hpp"

namespace glidar_slam::mapping {

MapBuilder::MapBuilder(std::shared_ptr<Parameters> parameters)
: parameters_(std::move(parameters)),
  occupancy_(parameters_),
  marking_(parameters_),
  texture_(parameters_)
{
}

MapBuilder::~MapBuilder()
{
  stop();
}

void MapBuilder::start()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    return;
  }
  stopping_ = false;
  running_ = true;
  worker_thread_ = std::thread(&MapBuilder::run, this);
}

void MapBuilder::stop()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    stopping_ = true;
  }
  condition_variable_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  running_ = false;
}

bool MapBuilder::submit(std::shared_ptr<const KeyFrame> keyframe)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (
    !running_ || stopping_ || !keyframe ||
    (!keyframe->local_occupancy && !keyframe->local_ground)) {
    return false;
  }

  const auto existing = pending_.find(keyframe->key);
  if (existing == pending_.end() || existing->second->revision < keyframe->revision) {
    pending_[keyframe->key] = std::move(keyframe);
  }

  condition_variable_.notify_one();
  return true;
}

bool MapBuilder::rebuild(std::vector<std::shared_ptr<const KeyFrame>> keyframes)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (!running_ || stopping_) {
    return false;
  }

  pending_.clear();
  pending_rebuild_ = std::move(keyframes);
  rebuild_requested_ = true;
  condition_variable_.notify_one();
  return true;
}

std::shared_ptr<const GlobalMapSnapshot> MapBuilder::getLatest() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return latest_;
}

void MapBuilder::run()
{
  while (true) {
    const auto process_start_time = std::chrono::steady_clock::now();

    std::shared_ptr<const KeyFrame> keyframe;
    std::vector<std::shared_ptr<const KeyFrame>> rebuild;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_variable_.wait(lock, [this] {
        return stopping_ || rebuild_requested_ || !pending_.empty();
      });
      if (stopping_ && pending_.empty() && !rebuild_requested_) {
        return;
      }

      if (rebuild_requested_) {
        rebuild = std::move(pending_rebuild_);
        rebuild_requested_ = false;
      } else if (stopping_ && pending_.empty()) {
        return;
      } else {
        auto item = pending_.begin();
        keyframe = std::move(item->second);
        pending_.erase(item);
      }
    }

    const auto prepare_end_time = std::chrono::steady_clock::now();

    if (!rebuild.empty()) {
      clearAccumulatedMap();
      for (const auto & item : rebuild) {
        if (item) {
          apply(*item);
        }
      }
    } else if (keyframe) {
      apply(*keyframe);
    }

    const auto apply_end_time = std::chrono::steady_clock::now();

    publish();

    const auto publish_end_time = std::chrono::steady_clock::now();

    if (parameters_->debug_timings) {
      const double process_duration =
        std::chrono::duration<double, std::milli>(publish_end_time - process_start_time).count();
      const auto prepare_duration =
        std::chrono::duration<double, std::milli>(prepare_end_time - process_start_time).count();
      const auto apply_duration =
        std::chrono::duration<double, std::milli>(apply_end_time - prepare_end_time).count();
      const auto publish_duration =
        std::chrono::duration<double, std::milli>(publish_end_time - apply_end_time).count();

      SAM_INFO(
        "[Map Builder] Prepare(Wait) Duration: {} ms, Apply Duration: {} ms, "
        "Publish Duration: {} ms, Total Duration: {} ms",
        prepare_duration, apply_duration, publish_duration, process_duration);
    }
  }
}

void MapBuilder::clearAccumulatedMap()
{
  occupancy_.clear();
  marking_.clear();
  texture_.clear();
}

void MapBuilder::apply(const KeyFrame & keyframe)
{
  if (keyframe.local_occupancy) {
    std::vector<GlobalOccupancyCell> global_occ_cells;
    const double res = keyframe.local_occupancy->resolution;

    for (const auto & local_cell : keyframe.local_occupancy->cells) {
      const gtsam::Point3 point = keyframe.pose.transformFrom(gtsam::Point3(
        (static_cast<double>(local_cell.x) + 0.5) * res,
        (static_cast<double>(local_cell.y) + 0.5) * res, 0.0));
      if (!std::isfinite(point.x()) || !std::isfinite(point.y())) {
        continue;
      }

      const int world_x =
        static_cast<int>(std::floor(point.x() / parameters_->mapping_occupancy_resolution));
      const int world_y =
        static_cast<int>(std::floor(point.y() / parameters_->mapping_occupancy_resolution));
      global_occ_cells.push_back({world_x, world_y, local_cell.log_odds});
    }

    occupancy_.apply(keyframe.key, keyframe.revision, global_occ_cells);
  }

  if (keyframe.local_ground) {
    std::vector<GlobalGroundCell> global_ground_cells;
    const double res = keyframe.local_ground->resolution;

    for (const auto & local_cell : keyframe.local_ground->cells) {
      const gtsam::Point3 point = keyframe.pose.transformFrom(gtsam::Point3(
        (static_cast<double>(local_cell.x) + 0.5) * res,
        (static_cast<double>(local_cell.y) + 0.5) * res, 0.0));
      if (!std::isfinite(point.x()) || !std::isfinite(point.y())) {
        continue;
      }

      const int world_x =
        static_cast<int>(std::floor(point.x() / parameters_->mapping_ground_resolution));
      const int world_y =
        static_cast<int>(std::floor(point.y() / parameters_->mapping_ground_resolution));
      global_ground_cells.push_back(
        {world_x, world_y, local_cell.log_odds, local_cell.r, local_cell.g, local_cell.b});
    }

    marking_.apply(keyframe.key, keyframe.revision, global_ground_cells);
    texture_.apply(keyframe.key, keyframe.revision, global_ground_cells);
  }
}

void MapBuilder::publish()
{
  auto snapshot = std::make_shared<GlobalMapSnapshot>();

  snapshot->occupancy = occupancy_.publish();
  snapshot->marking = marking_.publish();
  snapshot->texture = texture_.publish();

  ++generation_;
  snapshot->generation = generation_;

  std::lock_guard<std::mutex> lock(mutex_);
  latest_ = std::move(snapshot);
}

}  // namespace glidar_slam::mapping
