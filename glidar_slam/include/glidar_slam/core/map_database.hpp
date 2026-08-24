#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "glidar_slam/core/key_frame.hpp"

namespace glidar_slam::core {

class MapDatabase
{
public:
  MapDatabase();
  ~MapDatabase();

  void addKeyFrame(std::shared_ptr<KeyFrame> keyframe);

  void updatePoses(const gtsam::Values & optimized_values);

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
