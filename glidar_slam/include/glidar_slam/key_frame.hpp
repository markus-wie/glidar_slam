#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "glidar_slam/ground_plane_extractor.hpp"
#include "glidar_slam/laser_scan.hpp"
#include "glidar_slam/mapping/local_map.hpp"
#include "glidar_slam/parameters.hpp"
#include "glidar_slam/types.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam {

class KeyFrame
{
public:
  KeyFrame() = default;

  KeyFrame(
    uint64_t key, double timestamp, const gtsam::Pose3 & pose, const gtsam::Pose3 & odom_pose,
    std::shared_ptr<const LaserScan> scan, std::optional<gtsam::Matrix66> covariance = std::nullopt,
    std::optional<GroundPlaneObservation> ground_observation = std::nullopt, uint64_t revision = 0)
  : key(key),
    revision(revision),
    timestamp(timestamp),
    pose(pose),
    odom_pose(odom_pose),
    covariance(std::move(covariance)),
    scan(std::move(scan)),
    ground_observation(std::move(ground_observation)),
    local_occupancy(std::make_shared<const mapping::LocalOccupancyMap>()),
    local_ground(std::make_shared<const mapping::LocalGroundMap>())
  {
  }

  ~KeyFrame() = default;

  KeyFrame(const KeyFrame & ref) = default;
  KeyFrame(KeyFrame && ref) noexcept = default;

  KeyFrame & operator=(const KeyFrame & ref) = default;
  KeyFrame & operator=(KeyFrame && ref) noexcept = default;

  void buildLocalMaps(const Parameters & parameters);

  uint64_t key{0};
  uint64_t revision{0};
  double timestamp{0.0};
  gtsam::Pose3 pose;
  gtsam::Pose3 odom_pose;

  std::optional<gtsam::Matrix66> covariance;

  std::shared_ptr<const LaserScan> scan;
  std::optional<GroundPlaneObservation> ground_observation;

  // Cached local maps for global map building
  std::shared_ptr<const mapping::LocalOccupancyMap> local_occupancy;
  std::shared_ptr<const mapping::LocalGroundMap> local_ground;
};

}  // namespace glidar_slam
