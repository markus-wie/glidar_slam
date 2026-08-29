#include "glidar_slam/core/map_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace glidar_slam::core {

using global_map::GroundMarkingGrid;
using global_map::GroundTextureGrid;
using global_map::LocalMapData;
using global_map::OccupancyGrid;

namespace {

std::int64_t cellKey(int x, int y)
{
  return (static_cast<std::int64_t>(x) << 32) ^ static_cast<std::uint32_t>(y);
}

int cellCoordinate(float value, double resolution)
{
  return static_cast<int>(std::floor(static_cast<double>(value) / resolution));
}

template <typename Visitor>
void raytraceLine(int x0, int y0, int x1, int y1, const Visitor & visitor)
{
  const int dx = std::abs(x1 - x0);
  const int sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0);
  const int sy = y0 < y1 ? 1 : -1;
  int error = dx + dy;
  int x = x0;
  int y = y0;

  while (true) {
    if (!visitor(x, y) || (x == x1 && y == y1)) {
      return;
    }
    const int twice_error = 2 * error;
    if (twice_error >= dy) {
      error += dy;
      x += sx;
    }
    if (twice_error <= dx) {
      error += dx;
      y += sy;
    }
  }
}

}  // namespace

MapBuilder::MapBuilder(const std::shared_ptr<Parameters> & parameters) : parameters_(parameters)
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
  if (!running_ || stopping_ || !keyframe || !keyframe->local_map) {
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
    std::shared_ptr<const KeyFrame> keyframe;
    std::vector<std::shared_ptr<const KeyFrame>> rebuild;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_variable_.wait(lock, [this] {
        return stopping_ || rebuild_requested_ || !pending_.empty();
      });
      if (stopping_ && pending_.empty()) {
        if (!rebuild_requested_) {
          return;
        }
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
    publish();
  }
}

void MapBuilder::clearAccumulatedMap()
{
  applied_.clear();
  evidence_.clear();
  marking_.clear();
  texture_.clear();
}

void MapBuilder::apply(const KeyFrame & keyframe)
{
  const auto old = applied_.find(keyframe.key);
  if (old != applied_.end()) {
    for (const auto & cell : old->second.cells) {
      const auto key = cellKey(cell.x, cell.y);
      auto evidence = evidence_.find(key);
      if (evidence != evidence_.end()) {
        evidence->second -= cell.evidence;
        if (evidence->second == 0) {
          evidence_.erase(evidence);
        }
      }
    }
    for (const auto & cell : old->second.ground_cells) {
      const auto key = cellKey(cell.x, cell.y);
      marking_[key] -= cell.log_odds;
      auto texture = texture_.find(key);
      if (texture != texture_.end() && texture->second.count > 0) {
        --texture->second.count;
        texture->second.r -= cell.r;
        texture->second.g -= cell.g;
        texture->second.b -= cell.b;
        if (texture->second.count == 0) {
          texture_.erase(texture);
        }
      }
    }
  }

  AppliedContribution applied;
  applied.revision = keyframe.revision;
  const double resolution = keyframe.local_map->resolution;
  for (const auto & local_cell : keyframe.local_map->cells) {
    const gtsam::Point3 point = keyframe.pose.transformFrom(gtsam::Point3(
      (static_cast<double>(local_cell.x) + 0.5) * resolution,
      (static_cast<double>(local_cell.y) + 0.5) * resolution, 0.0));
    if (!std::isfinite(point.x()) || !std::isfinite(point.y())) {
      continue;
    }
    const int world_x =
      static_cast<int>(std::floor(point.x() / parameters_->mapping_occupancy_resolution));
    const int world_y =
      static_cast<int>(std::floor(point.y() / parameters_->mapping_occupancy_resolution));
    const int evidence = local_cell.evidence;
    evidence_[cellKey(world_x, world_y)] += evidence;
    applied.cells.push_back({world_x, world_y, evidence});
  }

  for (const auto & local_cell : keyframe.local_map->ground_cells) {
    const double resolution = keyframe.local_map->ground_resolution;
    const gtsam::Point3 point = keyframe.pose.transformFrom(gtsam::Point3(
      (static_cast<double>(local_cell.x) + 0.5) * resolution,
      (static_cast<double>(local_cell.y) + 0.5) * resolution, 0.0));
    if (!std::isfinite(point.x()) || !std::isfinite(point.y())) {
      continue;
    }
    const int world_x =
      static_cast<int>(std::floor(point.x() / parameters_->mapping_ground_resolution));
    const int world_y =
      static_cast<int>(std::floor(point.y() / parameters_->mapping_ground_resolution));
    const auto key = cellKey(world_x, world_y);
    marking_[key] += local_cell.log_odds;
    auto & texture = texture_[key];
    ++texture.count;
    texture.r += local_cell.r;
    texture.g += local_cell.g;
    texture.b += local_cell.b;
    applied.ground_cells.push_back(
      {world_x, world_y, local_cell.log_odds, local_cell.r, local_cell.g, local_cell.b});
  }

  applied_[keyframe.key] = std::move(applied);
}

void MapBuilder::publish()
{
  if (parameters_->mapping_occupancy_resolution <= 0.0) {
    return;
  }

  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::lowest();
  int max_y = std::numeric_limits<int>::lowest();
  for (const auto & [key, value] : evidence_) {
    if (value == 0) {
      continue;
    }
    const int x = static_cast<int>(key >> 32);
    const int y = static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff));
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
  }
  auto snapshot = std::make_shared<GlobalMapSnapshot>(GlobalMapSnapshot{
    OccupancyGrid(parameters_), GroundMarkingGrid(parameters_), GroundTextureGrid(parameters_),
    generation_ + 1});
  if (min_x <= max_x && min_y <= max_y) {
    GlobalMap::Info info;
    info.resolution = parameters_->mapping_occupancy_resolution;
    info.origin_x = static_cast<double>(min_x) * info.resolution;
    info.origin_y = static_cast<double>(min_y) * info.resolution;
    info.width = static_cast<std::uint32_t>(max_x - min_x + 1);
    info.height = static_cast<std::uint32_t>(max_y - min_y + 1);
    std::vector<std::int8_t> data(
      static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height), -1);
    for (const auto & [key, value] : evidence_) {
      const int x = static_cast<int>(key >> 32);
      const int y = static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff));
      const std::size_t index = static_cast<std::size_t>(y - min_y) * info.width + (x - min_x);
      data[index] = value < 0 ? 0 : 100;
    }
    snapshot->occupancy.setGrid(info, std::move(data));
  }

  auto groundBounds = [](
                        const auto & values, int & low_x, int & low_y, int & high_x, int & high_y) {
    low_x = std::numeric_limits<int>::max();
    low_y = std::numeric_limits<int>::max();
    high_x = std::numeric_limits<int>::lowest();
    high_y = std::numeric_limits<int>::lowest();
    for (const auto & [key, value] : values) {
      if (value == 0) {
        continue;
      }
      const int x = static_cast<int>(key >> 32);
      const int y = static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff));
      low_x = std::min(low_x, x);
      low_y = std::min(low_y, y);
      high_x = std::max(high_x, x);
      high_y = std::max(high_y, y);
    }
    return low_x <= high_x && low_y <= high_y;
  };
  int ground_min_x = 0;
  int ground_min_y = 0;
  int ground_max_x = 0;
  int ground_max_y = 0;
  if (groundBounds(marking_, ground_min_x, ground_min_y, ground_max_x, ground_max_y)) {
    GlobalMap::Info info;
    info.resolution = parameters_->mapping_ground_resolution;
    info.origin_x = static_cast<double>(ground_min_x) * info.resolution;
    info.origin_y = static_cast<double>(ground_min_y) * info.resolution;
    info.width = static_cast<std::uint32_t>(ground_max_x - ground_min_x + 1);
    info.height = static_cast<std::uint32_t>(ground_max_y - ground_min_y + 1);
    const std::size_t count = static_cast<std::size_t>(info.width) * info.height;
    std::vector<std::int8_t> marking(count, 0);
    std::vector<std::uint8_t> rgb(count * 3U, 0);
    for (const auto & [key, value] : marking_) {
      const int x = static_cast<int>(key >> 32);
      const int y = static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff));
      const std::size_t index =
        static_cast<std::size_t>(y - ground_min_y) * info.width + (x - ground_min_x);
      marking[index] = value > 0.0F ? 100 : 0;
    }
    for (const auto & [key, value] : texture_) {
      const int x = static_cast<int>(key >> 32);
      const int y = static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff));
      const std::size_t index =
        static_cast<std::size_t>(y - ground_min_y) * info.width + (x - ground_min_x);
      if (value.count > 0) {
        rgb[index * 3U] = static_cast<std::uint8_t>((value.r + value.count / 2U) / value.count);
        rgb[index * 3U + 1U] =
          static_cast<std::uint8_t>((value.g + value.count / 2U) / value.count);
        rgb[index * 3U + 2U] =
          static_cast<std::uint8_t>((value.b + value.count / 2U) / value.count);
      }
    }
    snapshot->marking.setGrid(info, std::move(marking));
    snapshot->texture.setGrid(info, std::move(rgb));
  }
  ++generation_;
  snapshot->generation = generation_;
  std::lock_guard<std::mutex> lock(mutex_);
  latest_ = std::move(snapshot);
}

LocalMapData MapBuilder::buildLocalOccupancy(const PointCloudXYZConstPtr & scan, double resolution)
{
  LocalMapData result;
  result.resolution = resolution;
  if (resolution <= 0.0) {
    return result;
  }

  std::unordered_map<std::int64_t, int> evidence;
  for (const auto & point : scan->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      continue;
    }
    const int target_x = cellCoordinate(point.x, resolution);
    const int target_y = cellCoordinate(point.y, resolution);
    const int origin_x = 0;
    const int origin_y = 0;
    raytraceLine(origin_x, origin_y, target_x, target_y, [&](int x, int y) {
      if (x == target_x && y == target_y) {
        return false;
      }
      --evidence[cellKey(x, y)];
      return true;
    });
    ++evidence[cellKey(target_x, target_y)];
  }

  result.cells.reserve(evidence.size());
  for (const auto & [key, value] : evidence) {
    if (value == 0) {
      continue;
    }
    result.cells.push_back(
      {static_cast<int>(key >> 32), static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff)),
       value});
  }
  return result;
}

void MapBuilder::addLocalGroundMap(
  LocalMapData & data, const PointCloudXYZRGBAConstPtr & cloud, const Parameters & parameters)
{
  data.ground_resolution = parameters.mapping_ground_resolution;
  if (data.ground_resolution <= 0.0) {
    return;
  }
  const float hit = static_cast<float>(
    std::log(parameters.mapping_log_odds_hit / (1.0 - parameters.mapping_log_odds_hit)));
  const float miss = static_cast<float>(
    std::log(parameters.mapping_log_odds_miss / (1.0 - parameters.mapping_log_odds_miss)));
  for (const auto & point : cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      continue;
    }
    data.ground_cells.push_back(
      {static_cast<int>(std::floor(point.x / data.ground_resolution)),
       static_cast<int>(std::floor(point.y / data.ground_resolution)),
       point.a >= parameters.mapping_threshold ? hit : miss, point.r, point.g, point.b});
  }
}

}  // namespace glidar_slam::core
