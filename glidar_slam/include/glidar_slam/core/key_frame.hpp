#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "glidar_slam/core/ground_plane_extractor.hpp"
#include "glidar_slam/core/laser_scan.hpp"
#include "glidar_slam/core/types.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam::core {

class KeyFrame
{
public:
  KeyFrame() = default;

  KeyFrame(
    uint64_t key, double timestamp, const gtsam::Pose3 & pose, const gtsam::Pose3 & odom_pose,
    std::shared_ptr<const LaserScan> scan, std::optional<gtsam::Matrix66> covariance = std::nullopt,
    std::optional<GroundPlaneObservation> ground_observation = std::nullopt)
  : key(key),
    timestamp(timestamp),
    pose(pose),
    odom_pose(odom_pose),
    covariance(std::move(covariance)),
    scan(std::move(scan)),
    ground_observation(std::move(ground_observation))
  {
  }

  ~KeyFrame() = default;

  KeyFrame(const KeyFrame & ref) = default;
  KeyFrame(KeyFrame && ref) noexcept = default;

  KeyFrame & operator=(const KeyFrame & ref) = default;
  KeyFrame & operator=(KeyFrame && ref) noexcept = default;

  uint64_t key{0};
  double timestamp{0.0};
  gtsam::Pose3 pose;
  gtsam::Pose3 odom_pose;

  std::optional<gtsam::Matrix66> covariance;

  std::shared_ptr<const LaserScan> scan;
  std::optional<GroundPlaneObservation> ground_observation;
};

}  // namespace glidar_slam::core
