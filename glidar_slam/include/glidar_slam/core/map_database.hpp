#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "glidar_slam/core/key_frame.hpp"
#include "glidar_slam/core/map_builder.hpp"

namespace glidar_slam::core {

class MapDatabase
{
public:
  MapDatabase();
  ~MapDatabase();

  void addKeyFrame(std::shared_ptr<KeyFrame> keyframe);

  std::vector<std::shared_ptr<const KeyFrame>> updatePoses(
    const gtsam::Values & optimized_values,
    const std::unordered_map<uint64_t, gtsam::Matrix66> & optimized_covariances = {});

  std::shared_ptr<const KeyFrame> getSnapshot(uint64_t key) const;
  std::vector<std::shared_ptr<const KeyFrame>> getSnapshots() const;

  std::vector<std::shared_ptr<const KeyFrame>> getNearbyKeyFrames(
    const gtsam::Pose3 & query_pose, double radius) const;

  std::vector<std::shared_ptr<const KeyFrame>> getAllKeyFrames() const;

  std::shared_ptr<const KeyFrame> getKeyFrame(uint64_t key) const;
  std::shared_ptr<const KeyFrame> getLatestKeyFrame() const;

  size_t size() const;

  uint64_t getNextKey() const;
  uint64_t incrementNextKey();

private:
  class SpatialIndex;

  void rebuildSpatialIndex();

  mutable std::shared_mutex rw_mutex_;

  std::vector<std::shared_ptr<KeyFrame>> keyframes_;
  std::unique_ptr<SpatialIndex> spatial_index_;

  std::atomic<uint64_t> next_keyframe_key_{0};
};

}  // namespace glidar_slam::core
