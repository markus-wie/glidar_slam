#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "glidar_slam/ground_plane_extractor.hpp"
#include "glidar_slam/key_frame.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam {

// This is a logical representation of a graph factor. It deliberately does not contain
// gtsam::NonlinearFactor or noise-model pointers, whose binary representation is not stable.
enum class SerializedFactorType : std::uint8_t
{
  PriorPose = 0,
  RelativePose = 1,
  GroundPlane = 2,
};

struct SerializedFactor
{
  SerializedFactorType type{SerializedFactorType::RelativePose};
  std::uint64_t from_key{0};
  std::uint64_t to_key{0};
  gtsam::Pose3 relative_pose;
  gtsam::Matrix66 covariance{gtsam::Matrix66::Zero()};

  gtsam::Vector3 observed_normal_in_base{gtsam::Vector3::UnitZ()};
  double observed_distance_to_base{0.0};
  gtsam::Vector3 reference_normal_in_map{gtsam::Vector3::UnitZ()};
  double reference_plane_offset{0.0};
  double normal_sigma{0.0};
  double distance_sigma{0.0};
};

struct SlamStateSnapshot
{
  static constexpr std::uint32_t kFormatVersion = 1;

  std::uint32_t format_version{kFormatVersion};
  std::string state_id;
  std::uint64_t next_keyframe_key{0};
  std::uint64_t optimizer_latest_key{0};
  std::uint64_t optimizer_latest_timestamp{0};
  bool optimizer_initialized{false};
  gtsam::Pose3 initial_pose;
  gtsam::Pose3 latest_pose;
  gtsam::Pose3 map_to_odom;

  std::vector<KeyFrame> keyframes;
  std::vector<SerializedFactor> factors;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> loop_closures;
};

}  // namespace glidar_slam
