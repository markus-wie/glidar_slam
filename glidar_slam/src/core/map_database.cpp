#include "glidar_slam/core/map_database.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "nanoflann.hpp"

namespace glidar_slam::core {

class MapDatabase::SpatialIndex
{
public:
  static constexpr size_t kPendingRebuildThreshold = 32;

  struct PointCloud
  {
    std::vector<std::array<double, 2>> points;

    size_t kdtree_get_point_count() const
    {
      return points.size();
    }

    double kdtree_get_pt(size_t index, size_t dimension) const
    {
      return points[index][dimension];
    }

    template <class BoundingBox>
    bool kdtree_get_bbox(BoundingBox &) const
    {
      return false;
    }
  };

  using Tree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 2>;

  SpatialIndex() : tree_(2, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10))
  {
  }

  void rebuild(const std::vector<std::shared_ptr<KeyFrame>> & keyframes)
  {
    cloud_.points.clear();
    keyframe_indices_.clear();
    cloud_.points.reserve(keyframes.size());
    keyframe_indices_.reserve(keyframes.size());

    for (size_t index = 0; index < keyframes.size(); ++index) {
      const gtsam::Point3 translation = keyframes[index]->pose.translation();
      cloud_.points.push_back({translation.x(), translation.y()});
      keyframe_indices_.push_back(index);
    }

    tree_.buildIndex();
    pending_points_.clear();
    pending_indices_.clear();
  }

  void addPending(size_t index, const KeyFrame & keyframe)
  {
    const gtsam::Point3 translation = keyframe.pose.translation();
    pending_points_.push_back({translation.x(), translation.y()});
    pending_indices_.push_back(index);
  }

  bool needsRebuild() const
  {
    return pending_points_.size() >= kPendingRebuildThreshold;
  }

  std::vector<size_t> radiusSearch(const gtsam::Pose3 & query_pose, double radius) const
  {
    std::vector<size_t> result;
    if (radius < 0.0 || cloud_.points.empty()) {
      return result;
    }

    const gtsam::Point3 & translation = query_pose.translation();
    const double query[2] = {translation.x(), translation.y()};
    std::vector<nanoflann::ResultItem<unsigned int, double>> matches;
    const auto match_count = tree_.radiusSearch(query, radius * radius, matches);
    matches.resize(match_count);
    result.reserve(matches.size());
    for (const auto & match : matches) {
      result.push_back(keyframe_indices_[match.first]);
    }

    const double radius_squared = radius * radius;
    for (size_t index = 0; index < pending_points_.size(); ++index) {
      const double dx = pending_points_[index][0] - query[0];
      const double dy = pending_points_[index][1] - query[1];
      if (dx * dx + dy * dy <= radius_squared) {
        result.push_back(pending_indices_[index]);
      }
    }

    return result;
  }

private:
  PointCloud cloud_;
  std::vector<size_t> keyframe_indices_;
  Tree tree_;
  std::vector<std::array<double, 2>> pending_points_;
  std::vector<size_t> pending_indices_;
};

MapDatabase::~MapDatabase() = default;
MapDatabase::MapDatabase() = default;

void MapDatabase::addKeyFrame(std::shared_ptr<KeyFrame> keyframe)
{
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  keyframes_.push_back(std::move(keyframe));
  if (!spatial_index_) {
    spatial_index_ = std::make_unique<SpatialIndex>();
    spatial_index_->rebuild(keyframes_);
  } else {
    spatial_index_->addPending(keyframes_.size() - 1, *keyframes_.back());
    if (spatial_index_->needsRebuild()) {
      rebuildSpatialIndex();
    }
  }
}

void MapDatabase::updatePoses(
  const gtsam::Values & optimized_values,
  const std::unordered_map<uint64_t, gtsam::Matrix66> & optimized_covariances)
{
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  for (const std::shared_ptr<KeyFrame> & keyframe : keyframes_) {
    if (optimized_values.exists(keyframe->key)) {
      keyframe->pose = optimized_values.at<gtsam::Pose3>(keyframe->key);
    }
    const auto covariance = optimized_covariances.find(keyframe->key);
    if (covariance != optimized_covariances.end()) {
      keyframe->covariance = covariance->second;
    }
  }
  rebuildSpatialIndex();
}

std::vector<std::shared_ptr<const KeyFrame>> MapDatabase::getNearbyKeyFrames(
  const gtsam::Pose3 & query_pose, double radius) const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  std::vector<std::shared_ptr<const KeyFrame>> nearby_keyframes;
  if (radius < 0.0 || !spatial_index_) {
    return nearby_keyframes;
  }

  for (const size_t index : spatial_index_->radiusSearch(query_pose, radius)) {
    nearby_keyframes.push_back(keyframes_[index]);
  }
  return nearby_keyframes;
}

std::vector<std::shared_ptr<const KeyFrame>> MapDatabase::getAllKeyFrames() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  return std::vector<std::shared_ptr<const KeyFrame>>(keyframes_.begin(), keyframes_.end());
}

std::shared_ptr<const KeyFrame> MapDatabase::getKeyFrame(uint64_t key) const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  if (key < keyframes_.size()) {
    return keyframes_[key];
  }
  return nullptr;
}

std::shared_ptr<const KeyFrame> MapDatabase::getLatestKeyFrame() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  if (!keyframes_.empty()) {
    return keyframes_.back();
  }
  return nullptr;
}

size_t MapDatabase::size() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  return keyframes_.size();
}

uint64_t MapDatabase::getNextKey() const
{
  return next_keyframe_key_.load(std::memory_order_relaxed);
}

uint64_t MapDatabase::incrementNextKey()
{
  return next_keyframe_key_.fetch_add(1, std::memory_order_relaxed);
}

void MapDatabase::rebuildSpatialIndex()
{
  if (!spatial_index_) {
    spatial_index_ = std::make_unique<SpatialIndex>();
  }
  spatial_index_->rebuild(keyframes_);
}

}  // namespace glidar_slam::core
