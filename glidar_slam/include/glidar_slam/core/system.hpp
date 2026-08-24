#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "glidar_slam/core/correlative_scan_matcher.hpp"
#include "glidar_slam/core/graph_optimizer.hpp"
#include "glidar_slam/core/key_frame.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam::core {

class SlamSystem
{
public:
  struct LaserScanOutput
  {
    std::optional<gtsam::Pose3> optimized_pose;
    std::optional<CsmResult::DebugImage> low_res_debug;
    std::optional<CsmResult::DebugImage> high_res_debug;
  };

  SlamSystem(const std::shared_ptr<Parameters> & parameters);

  LaserScanOutput handleLaserScan(
    double timestamp, const PointCloudXYZ & scan, const gtsam::Pose3 & odom_pose,
    const gtsam::Matrix66 & odom_covariance);

  PointCloudXYZ getMapCloud() const;
  std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> getTransformedKeyFrameScans() const;

  std::vector<std::shared_ptr<const KeyFrame>> getKeyFrames() const;

  gtsam::Pose3 getMapToOdom() const;

private:
  static gtsam::Pose3 projectPlanar(const gtsam::Pose3 & pose);
  static double translationDistance(const gtsam::Pose3 & lhs, const gtsam::Pose3 & rhs);
  static double yawDistance(const gtsam::Pose3 & lhs, const gtsam::Pose3 & rhs);

  void initializeIfNeeded(
    double timestamp, const PointCloudXYZ & scan, const gtsam::Pose3 & odom_pose);
  bool shouldCreateKeyFrame(const gtsam::Pose3 & current_odom_pose) const;

  void synchronizeKeyframes();

  std::shared_ptr<Parameters> parameters_;
  GraphOptimizer graph_optimizer_;

  std::vector<std::shared_ptr<KeyFrame>> keyframes_;
  uint64_t next_keyframe_key_;

  gtsam::Pose3 latest_map_to_odom_;

  mutable std::mutex latest_map_to_odom_mutex_;
  mutable std::mutex keyframes_mutex_;
};

}  // namespace glidar_slam::core
