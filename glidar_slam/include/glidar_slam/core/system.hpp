#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <utility>
#include <vector>

#include "glidar_slam/core/correlative_scan_matcher.hpp"
#include "glidar_slam/core/graph_optimizer.hpp"
#include "glidar_slam/core/ground_marking_matcher.hpp"
#include "glidar_slam/core/ground_plane_extractor.hpp"
#include "glidar_slam/core/key_frame.hpp"
#include "glidar_slam/core/loop_closure.hpp"
#include "glidar_slam/core/map_builder.hpp"
#include "glidar_slam/core/map_database.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/sensor_data.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam::core {

class SlamSystem
{
public:
  explicit SlamSystem(const std::shared_ptr<Parameters> & parameters);
  ~SlamSystem();

  bool process(
    double timestamp, const SensorData & sensor_data, const gtsam::Pose3 & odom_pose,
    const gtsam::Matrix66 & odom_covariance);

  std::optional<gtsam::Pose3> getLatestPose() const;
  std::optional<CsmResult::DebugImage> getLatestLowResDebug() const;
  std::optional<CsmResult::DebugImage> getLatestHighResDebug() const;
  std::optional<GroundPlaneObservation> getLatestGroundObservation() const;
  std::optional<PointCloudXYZRGBA> getLatestGroundMatchingDebug() const;

  PointCloudXYZ getMapCloud() const;
  std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> getTransformedKeyFrameScans() const;
  std::vector<pcl::PointCloud<pcl::PointXYZRGBA>> getTransformedGroundClouds() const;

  std::vector<std::shared_ptr<const KeyFrame>> getKeyFrames() const;
  std::vector<std::pair<uint64_t, uint64_t>> getLoopClosures() const;

  gtsam::Pose3 getMapToOdom() const;
  std::shared_ptr<const GlobalMapSnapshot> getLatestGlobalMap() const;

private:
  static gtsam::Pose3 projectPlanar(const gtsam::Pose3 & pose);
  static double translationDistance(const gtsam::Pose3 & lhs, const gtsam::Pose3 & rhs);
  static double yawDistance(const gtsam::Pose3 & lhs, const gtsam::Pose3 & rhs);

  static void appendVisiblePoints(
    const PointCloudXYZ & scan, const gtsam::Pose3 & pose, const Point2D & viewpoint,
    std::vector<Point2D> & output);

  bool shouldCreateKeyFrame(const gtsam::Pose3 & current_odom_pose) const;

  bool processLoopClosureProposals();

  void dispatchFindLoopClosure(const KeyFrame & latest_keyframe);

  // central parameters for all subsystems
  std::shared_ptr<Parameters> parameters_;

  // map database for managing the graph and associated data
  std::shared_ptr<MapDatabase> map_database_;

  // subsystems
  std::unique_ptr<CorrelativeScanMatcher> scan_matcher_;
  std::unique_ptr<GroundMarkingMatcher> ground_marking_matcher_;
  std::unique_ptr<GraphOptimizer> graph_optimizer_;
  std::unique_ptr<LoopClosureDetector> loop_closure_detector_;
  std::unique_ptr<MapBuilder> map_builder_;

  gtsam::Pose3 latest_map_to_odom_;
  mutable std::mutex latest_map_to_odom_mutex_;
  std::optional<gtsam::Pose3> latest_pose_;
  std::optional<CsmResult::DebugImage> latest_low_res_debug_;
  std::optional<CsmResult::DebugImage> latest_high_res_debug_;
  std::optional<GroundPlaneObservation> latest_ground_observation_;
  std::optional<PointCloudXYZRGBA> latest_ground_matching_debug_;
  mutable std::mutex latest_output_mutex_;
  std::vector<std::pair<uint64_t, uint64_t>> loop_closures_;
  mutable std::mutex loop_closures_mutex_;
  bool loop_closure_optimization_pending_{false};
};

}  // namespace glidar_slam::core
