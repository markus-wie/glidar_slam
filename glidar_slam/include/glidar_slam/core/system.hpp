#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "glidar_slam/core/graph_optimizer.hpp"
#include "glidar_slam/core/ground_plane_extractor.hpp"
#include "glidar_slam/core/key_frame.hpp"
#include "glidar_slam/core/loop_closure.hpp"
#include "glidar_slam/core/map_builder.hpp"
#include "glidar_slam/core/map_database.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/scan_matcher.hpp"
#include "glidar_slam/core/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam/core/scan_matcher/ground_marking_matcher.hpp"
#include "glidar_slam/core/sensor_data.hpp"
#include "glidar_slam/core/state_serializer.hpp"
#include "glidar_slam/core/submap_grid.hpp"
#include "glidar_slam/core/types.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam::core {

class SlamSystem
{
public:
  SlamSystem(
    const std::shared_ptr<Parameters> & parameters, std::unique_ptr<ScanMatcher> scan_matcher,
    std::unique_ptr<ScanMatcher> ground_scan_matcher,
    std::unique_ptr<ScanMatcher> loop_scan_matcher);
  ~SlamSystem();

  bool process(
    double timestamp, const SensorData & sensor_data, const gtsam::Pose3 & odom_pose,
    const gtsam::Matrix66 & odom_covariance);

  gtsam::Pose3 getLatestPose() const;
  PoseEstimate getLatestPoseAndCovariance() const;
  std::optional<CsmResult::DebugImage> getLatestLowResDebug() const;
  std::optional<CsmResult::DebugImage> getLatestHighResDebug() const;
  std::optional<GroundPlaneObservation> getLatestGroundObservation() const;
  std::optional<PointCloudXYZRGBA> getLatestGroundMatchingDebug() const;

  PointCloudXYZ getMapCloud() const;
  std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> getTransformedKeyFrameScans() const;
  std::vector<pcl::PointCloud<pcl::PointXYZRGBA>> getTransformedGroundClouds() const;

  std::vector<std::shared_ptr<const KeyFrame>> getKeyFrames() const;
  std::vector<std::pair<uint64_t, uint64_t>> getLoopClosures() const;
  std::vector<MapDatabase::GraphEdge> getEdges() const;

  gtsam::Pose3 getMapToOdom() const;
  std::shared_ptr<const GlobalMapSnapshot> getLatestGlobalMap() const;
  std::size_t getFactorCount() const;

  bool saveState(const std::filesystem::path & path, std::string * error = nullptr) const;
  bool loadState(
    const std::filesystem::path & path, const gtsam::Pose3 & initial_map_pose,
    bool use_saved_pose = false, bool localization_only = false, std::string * error = nullptr);

  bool isLocalizationMode() const;
  bool setLocalizationMode(
    bool enable, const gtsam::Pose3 & initial_map_pose, bool use_current_pose,
    std::string * error = nullptr);

private:
  bool shouldCreateKeyFrame(const gtsam::Pose3 & current_odom_pose) const;

  void handleGroundConstraint(
    std::optional<GroundPlaneObservation> & ground_observation, uint64_t next_keyframe_key);

  void handleGroundMatchingConstraint(
    const std::shared_ptr<const KeyFrame> & reference_keyframe,
    std::optional<GroundPlaneObservation> & ground_observation, const gtsam::Pose3 & current_guess,
    uint64_t next_keyframe_key);

  void rebuildSubmap();

  bool processLoopClosureProposals();

  void dispatchFindLoopClosure(const KeyFrame & latest_keyframe);

  // central parameters for all subsystems
  std::shared_ptr<Parameters> parameters_;

  // map database for managing the graph and associated data
  std::shared_ptr<MapDatabase> map_database_;

  // subsystems
  std::unique_ptr<ScanMatcher> scan_matcher_;
  std::unique_ptr<GroundMarkingMatcher> ground_marking_matcher_;
  std::unique_ptr<GraphOptimizer> graph_optimizer_;
  std::unique_ptr<LoopClosureDetector> loop_closure_detector_;
  std::unique_ptr<MapBuilder> map_builder_;
  std::unique_ptr<SubmapGrid> submap_grid_;

  gtsam::Pose3 latest_map_to_odom_;
  PoseEstimate latest_pose_;

  std::optional<CsmResult::DebugImage> latest_low_res_debug_;
  std::optional<CsmResult::DebugImage> latest_high_res_debug_;
  std::optional<GroundPlaneObservation> latest_ground_observation_;
  std::optional<PointCloudXYZRGBA> latest_ground_matching_debug_;

  bool loop_closure_optimization_pending_{false};
  bool tracking_reset_pending_{false};
  gtsam::Pose3 tracking_reset_pose_;
  bool localization_mode_{false};
  bool localization_initialized_{false};
  gtsam::Pose3 localization_submap_center_;

  mutable std::mutex latest_map_to_odom_mutex_;
  mutable std::mutex latest_output_mutex_;
  mutable std::mutex state_mutex_;
};

}  // namespace glidar_slam::core
