#include "glidar_slam/core/system.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <unordered_map>

#include "glidar_slam/core/ground_plane_extractor.hpp"
#include "glidar_slam/core/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"
#include "pcl/common/transforms.h"
#include "pcl/filters/voxel_grid.h"

namespace glidar_slam::core {

namespace {

constexpr double kUnobservablePlanarVariance = 1e6;

double elapsedMilliseconds(const std::chrono::steady_clock::time_point & start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
    .count();
}

gtsam::Pose3 initialPoseFromGroundObservation(
  const std::optional<GroundPlaneObservation> & observation)
{
  if (
    !observation || !observation->normal_in_base.allFinite() ||
    observation->normal_in_base.squaredNorm() < 1e-12F ||
    !std::isfinite(observation->distance_to_base)) {
    return gtsam::Pose3();
  }

  const Eigen::Vector3d normal_in_base = observation->normal_in_base.cast<double>().normalized();
  const Eigen::Vector3d target_normal = Eigen::Vector3d::UnitZ();
  const double alignment = normal_in_base.dot(target_normal);
  Eigen::Quaterniond map_from_base;
  if (alignment < -1.0 + 1e-12) {
    Eigen::Vector3d axis = normal_in_base.cross(Eigen::Vector3d::UnitX());
    if (axis.squaredNorm() < 1e-12) {
      axis = normal_in_base.cross(Eigen::Vector3d::UnitY());
    }
    map_from_base = Eigen::Quaterniond(0.0, axis.x(), axis.y(), axis.z()).normalized();
  } else {
    const Eigen::Vector3d rotation_axis = normal_in_base.cross(target_normal);
    map_from_base =
      Eigen::Quaterniond(1.0 + alignment, rotation_axis.x(), rotation_axis.y(), rotation_axis.z())
        .normalized();
  }
  return gtsam::Pose3(
    gtsam::Rot3::Quaternion(
      map_from_base.w(), map_from_base.x(), map_from_base.y(), map_from_base.z()),
    gtsam::Point3(0.0, 0.0, observation->distance_to_base));
}

gtsam::Matrix66 planarOdometryCovariance(const gtsam::Matrix66 & covariance)
{
  gtsam::Matrix66 result = covariance;
  const std::array<int, 3> unobservable_dimensions{0, 1, 5};
  for (const int dimension : unobservable_dimensions) {
    result.row(dimension).setZero();
    result.col(dimension).setZero();
    result(dimension, dimension) = kUnobservablePlanarVariance;
  }
  return result;
}

gtsam::Pose3 makePlanarPose(const gtsam::Pose3 & pose)
{
  const gtsam::Vector3 rpy = pose.rotation().rpy();
  const double yaw = rpy.z();
  const auto & translation = pose.translation();
  return Utils::makePlanarPose(translation.x(), translation.y(), yaw);
}

Eigen::Matrix3d transformScanMatchCovarianceToRelativeFrame(
  const Eigen::Matrix3d & covariance, const gtsam::Pose3 & reference_pose)
{
  const Eigen::Matrix2d map_to_reference =
    Eigen::Rotation2Dd(-reference_pose.rotation().yaw()).toRotationMatrix();
  Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
  transform.topLeftCorner<2, 2>() = map_to_reference;
  return transform * covariance * transform.transpose();
}

double computeTranslationDistance(const gtsam::Pose3 & lhs, const gtsam::Pose3 & rhs)
{
  const auto delta = lhs.translation() - rhs.translation();
  return std::hypot(delta.x(), delta.y());
}

double computeYawDistance(const gtsam::Pose3 & lhs, const gtsam::Pose3 & rhs)
{
  const double lhs_yaw = lhs.rotation().rpy().z();
  const double rhs_yaw = rhs.rotation().rpy().z();
  return std::abs(Utils::normalizeAngle(lhs_yaw - rhs_yaw));
}

std::vector<Point2D> transformScanPoints(const PointCloudXYZ & scan, const gtsam::Pose3 & pose)
{
  std::vector<Point2D> points;
  points.reserve(scan.size());
  for (const auto & point : scan) {
    const gtsam::Point3 world_point = pose.transformFrom(gtsam::Point3(point.x, point.y, point.z));
    if (std::isfinite(world_point.x()) && std::isfinite(world_point.y())) {
      points.push_back({world_point.x(), world_point.y()});
    }
  }
  return points;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxelizeLidarScan(const LaserScan & scan, double voxel_size)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr voxelized_cloud =
    std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(scan.points().makeShared());
  voxel_filter.setLeafSize(
    static_cast<float>(voxel_size), static_cast<float>(voxel_size), static_cast<float>(voxel_size));
  voxel_filter.filter(*voxelized_cloud);

  return voxelized_cloud;
}

std::vector<Point2D> linearInterpolateScan(const LaserScan & scan, double target_spacing = 0.05)
{
  std::vector<Point2D> continuous_points;

  const std::vector<Point2D> & scan_points = scan.points2D();

  for (size_t i = 0; i < scan_points.size() - 1; ++i) {
    const Point2D & p1 = scan_points[i];
    const Point2D & p2 = scan_points[i + 1];

    if (!std::isfinite(p1.x) || !std::isfinite(p2.x)) {
      continue;
    }

    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double segment_len = std::hypot(dx, dy);

    // If the gap is huge (e.g., transitioning from a wall to empty space), don't interpolate
    if (segment_len > 1.0) {
      continuous_points.push_back({p1.x, p1.y});
      continue;
    }

    // Step along the continuous line at exactly target_spacing intervals
    int num_steps = std::max(1, static_cast<int>(segment_len / target_spacing));
    for (int step = 0; step < num_steps; ++step) {
      double t = static_cast<double>(step) / num_steps;
      continuous_points.push_back({p1.x + t * dx, p1.y + t * dy});
    }
  }
  return continuous_points;
}

}  // namespace

SlamSystem::SlamSystem(
  const std::shared_ptr<Parameters> & parameters, std::unique_ptr<ScanMatcher> scan_matcher,
  std::unique_ptr<ScanMatcher> ground_scan_matcher, std::unique_ptr<ScanMatcher> loop_scan_matcher)
: parameters_(parameters), scan_matcher_(std::move(scan_matcher))
{
  map_database_ = std::make_shared<MapDatabase>();
  graph_optimizer_ = std::make_unique<GraphOptimizer>(parameters_);
  ground_marking_matcher_ =
    std::make_unique<GroundMarkingMatcher>(parameters_, std::move(ground_scan_matcher));

  std::vector<double> resolutions = scan_matcher_->fieldResolutions();

  submap_grid_ = std::make_unique<SubmapGrid>(resolutions);
  loop_closure_detector_ =
    std::make_unique<LoopClosureDetector>(parameters_, map_database_, std::move(loop_scan_matcher));
  loop_closure_detector_->start();
  map_builder_ = std::make_unique<MapBuilder>(parameters_);
  map_builder_->start();
}

SlamSystem::~SlamSystem()
{
  map_builder_->stop();
  loop_closure_detector_->stop();
}

bool SlamSystem::process(
  double timestamp, const SensorData & sensor_data, const gtsam::Pose3 & odom_pose,
  const gtsam::Matrix66 & odom_covariance)
{
  const auto process_start = std::chrono::steady_clock::now();
  double lidar_voxelization_ms = 0;
  double loop_proposals_ms = 0;
  double initialization_ms = 0;
  double ground_extraction_ms = 0;
  double csm_ms = 0;
  double factor_preparation_ms = 0;
  double optimization_ms = 0;
  double keyframe_ms = 0;
  double map_ms = 0;
  double submap_ms = 0;
  double update_covariance_ms = 0;
  double loop_dispatch_ms = 0;

  // const std::shared_ptr<const LaserScan> & laser_scan = sensor_data.laserScan();
  if (sensor_data.laserScan()->empty()) {
    SAM_WARN("Empty laser scan received. Did not run the system!");
    return false;
  }

  const auto voxelization_start = std::chrono::steady_clock::now();

  std::shared_ptr<const LaserScan> laser_scan;
  if (parameters_->lidar_voxelization_enable) {
    const auto scan =
      voxelizeLidarScan(*sensor_data.laserScan(), parameters_->lidar_voxelization_size);
    laser_scan = std::make_shared<LaserScan>(std::move(*scan));
  } else {
    laser_scan = sensor_data.laserScan();
  }

  std::vector<Point2D> interpolated_points =
    linearInterpolateScan(*sensor_data.laserScan(), parameters_->lidar_voxelization_size);
  laser_scan = std::make_shared<LaserScan>(std::move(interpolated_points));

  if (parameters_->debug_timings) {
    lidar_voxelization_ms = elapsedMilliseconds(voxelization_start);
  }

  const auto loop_proposals_start = std::chrono::steady_clock::now();
  processLoopClosureProposals();

  if (parameters_->debug_timings) {
    loop_proposals_ms = elapsedMilliseconds(loop_proposals_start);
  }

  const auto initialization_start = std::chrono::steady_clock::now();

  const gtsam::Pose3 latest_odom_pose = makePlanarPose(odom_pose);

  if (map_database_->size() == 0) {
    std::optional<GroundPlaneObservation> ground_observation;
    if (sensor_data.hasVisualData()) {
      ground_observation = GroundPlaneExtractor::extract(
        sensor_data.image(), sensor_data.depth(), sensor_data.cameraModel().intrinsics(),
        sensor_data.cameraModel().baseFromCamera(), *parameters_);
    }

    const gtsam::Pose3 initial_pose = initialPoseFromGroundObservation(ground_observation);

    graph_optimizer_->initialize(initial_pose, static_cast<uint64_t>(timestamp * 1e6));

    auto new_keyframe = std::make_shared<KeyFrame>(
      map_database_->incrementNextKey(), timestamp, initial_pose, latest_odom_pose, laser_scan,
      std::nullopt, ground_observation);

    auto local_map =
      MapBuilder::buildLocalOccupancy(laser_scan->points(), parameters_->occ_map_resolution);

    std::vector<Point2D> initial_points = transformScanPoints(laser_scan->points(), initial_pose);
    submap_grid_->add(initial_points, new_keyframe->key);

    if (ground_observation) {
      MapBuilder::addLocalGroundMap(local_map, ground_observation->ground_cloud, *parameters_);
    }

    new_keyframe->local_map =
      std::make_shared<const global_map::LocalMapData>(std::move(local_map));
    map_database_->addKeyFrame(new_keyframe);
    map_builder_->submit(map_database_->getSnapshot(new_keyframe->key));

    SAM_INFO("System initialized with first keyframe at timestamp: {}", timestamp);

    {
      std::lock_guard<std::mutex> lock(latest_output_mutex_);
      latest_pose_ = initial_pose;
    }
    return true;
  }

  std::shared_ptr<const KeyFrame> reference_keyframe = map_database_->getLatestKeyFrame();

  // Get relative odometry delta and apply to the reference keyframes world pose
  const gtsam::Pose3 raw_odom_delta =
    reference_keyframe->odom_pose.inverse().compose(latest_odom_pose);
  const gtsam::Pose3 current_guess = reference_keyframe->pose.compose(raw_odom_delta);

  // Check if we should spawn a new keyframe based on motion thresholds
  if (!shouldCreateKeyFrame(latest_odom_pose)) {
    {
      std::lock_guard<std::mutex> lock(latest_output_mutex_);
      latest_pose_ = current_guess;
    }
    {
      std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
      latest_map_to_odom_ = current_guess.compose(latest_odom_pose.inverse());
    }
    return false;
  }

  if (parameters_->debug_timings) {
    initialization_ms = elapsedMilliseconds(initialization_start);
  }

  std::optional<GroundPlaneObservation> ground_observation;
  if (sensor_data.hasVisualData()) {
    const auto start = std::chrono::steady_clock::now();
    ground_observation = GroundPlaneExtractor::extract(
      sensor_data.image(), sensor_data.depth(), sensor_data.cameraModel().intrinsics(),
      sensor_data.cameraModel().baseFromCamera(), *parameters_);
    if (parameters_->debug_timings) {
      ground_extraction_ms = elapsedMilliseconds(start);
    }
  }

  const auto csm_start = std::chrono::steady_clock::now();

  uint64_t next_keyframe_key = map_database_->incrementNextKey();

  // Execute the Correlative Scan Matcher
  // This yields an optimized pose and a 3x3 covariance matrix
  const CsmResult csm_result =
    scan_matcher_->match(*submap_grid_, laser_scan->points2D(), Utils::toPose2D(current_guess));

  if (parameters_->debug_timings) {
    csm_ms = elapsedMilliseconds(csm_start);
  }

  const auto factor_preperation_start = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    latest_low_res_debug_ = csm_result.low_res_debug;
    latest_high_res_debug_ = csm_result.high_res_debug;
    latest_ground_observation_ = ground_observation;
    latest_ground_matching_debug_.reset();
  }

  // Map the 3x3 CSM Covariance (x, y, yaw) to a 6x6 GTSAM Covariance Matrix
  const Eigen::Matrix3d relative_csm_covariance =
    transformScanMatchCovarianceToRelativeFrame(csm_result.covariance, reference_keyframe->pose);
  gtsam::Matrix66 csm_covariance = gtsam::Matrix66::Zero();

  // Inflate unobservable dimensions
  constexpr double INF_VAR = 1e6;

  // Diagonal of csm_result.covariance:
  // [ x-x, y-y, yaw-yaw ]
  // gtsam covariance layout is [roll, pitch, yaw, x, y, z]
  csm_covariance(0, 0) = INF_VAR;                        // roll-roll
  csm_covariance(1, 1) = INF_VAR;                        // pitch-pitch
  csm_covariance(2, 2) = relative_csm_covariance(2, 2);  // yaw-yaw
  csm_covariance(3, 3) = relative_csm_covariance(0, 0);  // x-x
  csm_covariance(4, 4) = relative_csm_covariance(1, 1);  // y-y
  csm_covariance(5, 5) = INF_VAR;                        // z-z
  csm_covariance(3, 4) = relative_csm_covariance(0, 1);  // x-y
  csm_covariance(4, 3) = relative_csm_covariance(1, 0);  // y-x
  csm_covariance(3, 2) = relative_csm_covariance(0, 2);  // x-yaw
  csm_covariance(2, 3) = relative_csm_covariance(2, 0);  // yaw-x
  csm_covariance(4, 2) = relative_csm_covariance(1, 2);  // y-yaw
  csm_covariance(2, 4) = relative_csm_covariance(2, 1);  // yaw-y

  if (parameters_->debug_timings) {
    SAM_INFO(
      "CSM Covariance (diagonal): x={}, y={}, yaw={}", csm_result.covariance(0, 0),
      csm_result.covariance(1, 1), csm_result.covariance(2, 2));
  }

  // Convert optimized absolute Pose2D back to gtsam::Pose3
  gtsam::Pose3 optimized_world_pose = Utils::toPose3(csm_result.optimized_pose);

  // Calculate the relative transform factor for the graph
  const gtsam::Pose3 csm_pose_delta =
    reference_keyframe->pose.inverse().compose(optimized_world_pose);

  const bool use_ground_constraint =
    ground_observation.has_value() && parameters_->ground_optimization_enable &&
    ground_observation->inlier_count >=
      static_cast<std::size_t>(parameters_->ground_minimum_inlier_count);

  gtsam::Vector3 ground_normal_in_base_ = gtsam::Vector3::UnitZ();
  double ground_distance_to_base_ = 0.0;
  if (ground_observation) {
    ground_normal_in_base_ = ground_observation->normal_in_base.cast<double>();
    ground_distance_to_base_ = static_cast<double>(ground_observation->distance_to_base);
  }

  // Add Odometry Factor
  graph_optimizer_->addRelativeFactor(
    reference_keyframe->key, next_keyframe_key, raw_odom_delta,
    planarOdometryCovariance(odom_covariance));

  // Add LiDAR Scan Matching Factor
  graph_optimizer_->addRelativeFactor(
    reference_keyframe->key, next_keyframe_key, csm_pose_delta, csm_covariance);

  if (
    parameters_->ground_matching_enable && reference_keyframe->ground_observation &&
    ground_observation) {
    const Pose2D relative_ground_pose_estimate =
      Utils::toPose2D(reference_keyframe->pose.inverse().compose(current_guess));

    const auto ground_match = ground_marking_matcher_->match(
      *reference_keyframe->ground_observation, *ground_observation, relative_ground_pose_estimate);

    if (ground_match && ground_match->score >= parameters_->ground_matching_minimum_score) {
      gtsam::Matrix66 ground_covariance = gtsam::Matrix66::Zero();
      ground_covariance(0, 0) = INF_VAR;
      ground_covariance(1, 1) = INF_VAR;
      ground_covariance(2, 2) = ground_match->covariance(2, 2);
      ground_covariance(3, 3) = ground_match->covariance(0, 0);
      ground_covariance(4, 4) = ground_match->covariance(1, 1);
      ground_covariance(5, 5) = INF_VAR;
      ground_covariance(3, 4) = ground_match->covariance(0, 1);
      ground_covariance(4, 3) = ground_match->covariance(1, 0);
      ground_covariance(3, 2) = ground_match->covariance(0, 2);
      ground_covariance(2, 3) = ground_match->covariance(2, 0);
      ground_covariance(4, 2) = ground_match->covariance(1, 2);
      ground_covariance(2, 4) = ground_match->covariance(2, 1);

      graph_optimizer_->addRelativeFactor(
        reference_keyframe->key, next_keyframe_key, Utils::toPose3(ground_match->optimized_pose),
        ground_covariance);

      if (parameters_->ground_matching_debug_enable) {
        const PointCloudXYZRGBA reference_debug_cloud = ground_marking_matcher_->makeDebugCloud(
          *reference_keyframe->ground_observation, *ground_observation, *ground_match,
          relative_ground_pose_estimate);

        const gtsam::Vector3 reference_rpy = reference_keyframe->pose.rotation().rpy();

        const auto translation = reference_keyframe->pose.translation();

        const Eigen::Affine3f map_from_reference =
          Eigen::Translation3f(
            static_cast<float>(translation.x()), static_cast<float>(translation.y()),
            static_cast<float>(translation.z())) *
          Eigen::AngleAxisf(static_cast<float>(reference_rpy.z()), Eigen::Vector3f::UnitZ()) *
          Eigen::AngleAxisf(static_cast<float>(reference_rpy.y()), Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(static_cast<float>(reference_rpy.x()), Eigen::Vector3f::UnitX());

        PointCloudXYZRGBA map_debug_cloud;
        pcl::transformPointCloud(reference_debug_cloud, map_debug_cloud, map_from_reference);

        {
          std::lock_guard<std::mutex> lock(latest_output_mutex_);
          latest_ground_matching_debug_ = std::move(map_debug_cloud);
          latest_low_res_debug_ = ground_match->low_res_debug;
          latest_high_res_debug_ = ground_match->high_res_debug;
        }
      }

      if (parameters_->ground_matching_debug_enable) {
        SAM_INFO(
          "Ground marking matching result: optimized_pose=({}, {}, {}), covariance=({})",
          ground_match->optimized_pose.x, ground_match->optimized_pose.y,
          ground_match->optimized_pose.yaw, ground_match->covariance.diagonal().transpose());
        SAM_INFO(
          "Ground marking factor added at keyframe {} with score {}", next_keyframe_key,
          ground_match->score);
      }
    }
  }

  if (use_ground_constraint) {
    graph_optimizer_->addGroundPlaneFactor(
      next_keyframe_key, ground_normal_in_base_, ground_distance_to_base_, gtsam::Vector3::UnitZ(),
      0.0, parameters_->ground_normal_sigma, parameters_->ground_distance_sigma);

    if (parameters_->ground_debug_enable) {
      SAM_INFO(
        "Ground plane factor added at keyframe {} with {} fresh inliers", next_keyframe_key,
        ground_observation->inlier_count);
    }
  }

  if (parameters_->debug_timings) {
    factor_preparation_ms = elapsedMilliseconds(factor_preperation_start);
  }

  const auto optimization_start = std::chrono::steady_clock::now();

  gtsam::Values updated_states = graph_optimizer_->optimize();

  if (parameters_->debug_timings) {
    optimization_ms = elapsedMilliseconds(optimization_start);
  }

  const auto keyframe_start = std::chrono::steady_clock::now();

  gtsam::Pose3 optimized_pose = graph_optimizer_->getLatestPose();

  gtsam::Matrix66 optimized_covariance =
    graph_optimizer_->getMarginalCovariance(next_keyframe_key).value_or(gtsam::Matrix66::Zero());

  if (parameters_->debug_timings) {
    SAM_INFO(
      "GTSAM Covariance (diagonal): x={}, y={}, z={}", optimized_covariance(3, 3),
      optimized_covariance(4, 4), optimized_covariance(5, 5));
  }

  const std::shared_ptr<KeyFrame> new_keyframe = std::make_shared<KeyFrame>(
    next_keyframe_key, timestamp, optimized_pose, latest_odom_pose, laser_scan,
    optimized_covariance, ground_observation);

  if (parameters_->debug_timings) {
    keyframe_ms = elapsedMilliseconds(keyframe_start);
  }

  const auto map_start = std::chrono::steady_clock::now();

  auto local_map =
    MapBuilder::buildLocalOccupancy(laser_scan->points(), parameters_->occ_map_resolution);
  if (ground_observation) {
    MapBuilder::addLocalGroundMap(local_map, ground_observation->ground_cloud, *parameters_);
  }
  new_keyframe->local_map = std::make_shared<const global_map::LocalMapData>(std::move(local_map));

  map_database_->addKeyFrame(new_keyframe);
  map_builder_->submit(map_database_->getSnapshot(next_keyframe_key));

  if (parameters_->debug_timings) {
    map_ms = elapsedMilliseconds(map_start);
  }

  const auto submap_start = std::chrono::steady_clock::now();

  std::vector<Point2D> newest_points = transformScanPoints(laser_scan->points(), optimized_pose);
  submap_grid_->add(newest_points, next_keyframe_key);
  while (submap_grid_->size() > static_cast<std::size_t>(parameters_->submap_window_size)) {
    submap_grid_->removeOldestKeyframe();
  }

  if (parameters_->debug_timings) {
    submap_ms = elapsedMilliseconds(submap_start);
  }

  const auto update_covariance_start = std::chrono::steady_clock::now();

  std::unordered_map<uint64_t, gtsam::Matrix66> optimized_covariances;
  if (loop_closure_optimization_pending_) {
    for (const auto & keyframe : map_database_->getAllKeyFrames()) {
      if (const auto covariance = graph_optimizer_->getMarginalCovariance(keyframe->key)) {
        optimized_covariances.emplace(keyframe->key, *covariance);
      }
    }
  }

  for (const auto & snapshot : map_database_->updatePoses(updated_states, optimized_covariances)) {
    map_builder_->submit(snapshot);
  }

  if (loop_closure_optimization_pending_) {
    map_builder_->rebuild(map_database_->getSnapshots());
    rebuildSubmap();
  }
  loop_closure_optimization_pending_ = false;

  {
    std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
    gtsam::Pose3 odom_to_base = latest_odom_pose.inverse();
    latest_map_to_odom_ = optimized_pose.compose(odom_to_base);
  }

  if (parameters_->debug_timings) {
    update_covariance_ms = elapsedMilliseconds(update_covariance_start);
  }

  const auto loop_dispatch_start = std::chrono::steady_clock::now();

  dispatchFindLoopClosure(*new_keyframe);

  if (parameters_->debug_timings) {
    loop_dispatch_ms = elapsedMilliseconds(loop_dispatch_start);
  }

  {
    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    latest_pose_ = optimized_pose;
  }

  if (parameters_->debug_timings) {
    SAM_INFO(
      "SLAM SYSTEM timings [ms]: lidar_voxelization={}, loop_proposals={}, initialization={}, "
      "ground_extraction={}, "
      "csm={}, factor_preparation={}, "
      "optimization={}, keyframe={}, map={}, submap={}, update_covariance={}, loop_dispatch={}, "
      "total={}",
      lidar_voxelization_ms, loop_proposals_ms, initialization_ms, ground_extraction_ms, csm_ms,
      factor_preparation_ms, optimization_ms, keyframe_ms, map_ms, submap_ms, update_covariance_ms,
      loop_dispatch_ms, elapsedMilliseconds(process_start));
  }

  return true;
}

std::optional<gtsam::Pose3> SlamSystem::getLatestPose() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_pose_;
}

std::shared_ptr<const GlobalMapSnapshot> SlamSystem::getLatestGlobalMap() const
{
  return map_builder_->getLatest();
}

std::optional<CsmResult::DebugImage> SlamSystem::getLatestLowResDebug() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_low_res_debug_;
}

std::optional<CsmResult::DebugImage> SlamSystem::getLatestHighResDebug() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_high_res_debug_;
}

std::optional<GroundPlaneObservation> SlamSystem::getLatestGroundObservation() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_ground_observation_;
}

std::optional<PointCloudXYZRGBA> SlamSystem::getLatestGroundMatchingDebug() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_ground_matching_debug_;
}

bool SlamSystem::processLoopClosureProposals()
{
  std::vector<LoopClosureProposal> proposals;

  while (const auto proposal = loop_closure_detector_->tryPopProposal()) {
    if (!proposal.has_value()) {
      break;
    }
    proposals.emplace_back(proposal.value());
  }

  for (const auto & proposal : proposals) {
    graph_optimizer_->addRelativeFactor(
      proposal.from_key, proposal.to_key, proposal.relative_pose, proposal.covariance);

    {
      std::lock_guard<std::mutex> lock(loop_closures_mutex_);
      loop_closures_.emplace_back(proposal.from_key, proposal.to_key);
    }

    SAM_INFO(
      "Loop closure accepted: from={}, to={}, score={}", proposal.from_key, proposal.to_key,
      proposal.score);
  }

  loop_closure_optimization_pending_ = loop_closure_optimization_pending_ || !proposals.empty();

  return !proposals.empty();
}

void SlamSystem::rebuildSubmap()
{
  std::vector<double> resolutions;
  resolutions = scan_matcher_->fieldResolutions();

  auto rebuilt_submap = std::make_unique<SubmapGrid>(resolutions);
  const auto keyframes = map_database_->getAllKeyFrames();
  const std::size_t window_size =
    static_cast<std::size_t>(std::max(parameters_->submap_window_size, 0));
  const std::size_t first_keyframe =
    keyframes.size() > window_size ? keyframes.size() - window_size : 0;

  for (std::size_t index = first_keyframe; index < keyframes.size(); ++index) {
    const auto & keyframe = keyframes[index];
    const std::vector<Point2D> points =
      transformScanPoints(keyframe->scan->points(), keyframe->pose);
    rebuilt_submap->add(points, keyframe->key);
  }

  submap_grid_ = std::move(rebuilt_submap);
}

void SlamSystem::dispatchFindLoopClosure(const KeyFrame & latest_keyframe)
{
  KeyFrame snapshot(latest_keyframe);
  loop_closure_detector_->submit(std::move(snapshot));
}

PointCloudXYZ SlamSystem::getMapCloud() const
{
  PointCloudXYZ map_cloud;

  for (const std::shared_ptr<const KeyFrame> & keyframe : map_database_->getAllKeyFrames()) {
    const gtsam::Vector3 rpy = keyframe->pose.rotation().rpy();
    const Eigen::AngleAxisf roll_angle(static_cast<float>(rpy.x()), Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch_angle(static_cast<float>(rpy.y()), Eigen::Vector3f::UnitY());
    const Eigen::AngleAxisf yaw_angle(static_cast<float>(rpy.z()), Eigen::Vector3f::UnitZ());

    const auto translation = keyframe->pose.translation();
    const Eigen::Affine3f transform =
      Eigen::Translation3f(
        static_cast<float>(translation.x()), static_cast<float>(translation.y()),
        static_cast<float>(translation.z())) *
      yaw_angle * pitch_angle * roll_angle;

    PointCloudXYZ transformed_scan;
    pcl::transformPointCloud(keyframe->scan->points(), transformed_scan, transform);
    map_cloud += transformed_scan;
  }

  map_cloud.width = static_cast<std::uint32_t>(map_cloud.size());
  map_cloud.height = 1;
  map_cloud.is_dense = true;
  return map_cloud;
}

std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> SlamSystem::getTransformedKeyFrameScans() const
{
  std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> out;
  const auto keyframes = map_database_->getAllKeyFrames();
  out.reserve(keyframes.size());

  for (const std::shared_ptr<const KeyFrame> & keyframe : keyframes) {
    const gtsam::Vector3 rpy = keyframe->pose.rotation().rpy();
    const Eigen::AngleAxisf roll_angle(static_cast<float>(rpy.x()), Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch_angle(static_cast<float>(rpy.y()), Eigen::Vector3f::UnitY());
    const Eigen::AngleAxisf yaw_angle(static_cast<float>(rpy.z()), Eigen::Vector3f::UnitZ());

    const auto translation = keyframe->pose.translation();
    const Eigen::Affine3f transform =
      Eigen::Translation3f(
        static_cast<float>(translation.x()), static_cast<float>(translation.y()),
        static_cast<float>(translation.z())) *
      yaw_angle * pitch_angle * roll_angle;

    PointCloudXYZ transformed_scan;
    pcl::transformPointCloud(keyframe->scan->points(), transformed_scan, transform);
    out.emplace_back(keyframe->pose, std::move(transformed_scan));
  }

  return out;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGBA>> SlamSystem::getTransformedGroundClouds() const
{
  std::vector<pcl::PointCloud<pcl::PointXYZRGBA>> clouds;

  for (const std::shared_ptr<const KeyFrame> & keyframe : map_database_->getAllKeyFrames()) {
    if (
      !keyframe->ground_observation.has_value() ||
      keyframe->ground_observation->ground_cloud.empty()) {
      continue;
    }

    const gtsam::Vector3 rpy = keyframe->pose.rotation().rpy();
    const Eigen::AngleAxisf roll_angle(static_cast<float>(rpy.x()), Eigen::Vector3f::UnitX());
    const Eigen::AngleAxisf pitch_angle(static_cast<float>(rpy.y()), Eigen::Vector3f::UnitY());
    const Eigen::AngleAxisf yaw_angle(static_cast<float>(rpy.z()), Eigen::Vector3f::UnitZ());
    const auto translation = keyframe->pose.translation();
    const Eigen::Affine3f transform =
      Eigen::Translation3f(
        static_cast<float>(translation.x()), static_cast<float>(translation.y()),
        static_cast<float>(translation.z())) *
      yaw_angle * pitch_angle * roll_angle;

    pcl::PointCloud<pcl::PointXYZRGBA> transformed_cloud;
    pcl::transformPointCloud(
      keyframe->ground_observation->ground_cloud, transformed_cloud, transform);
    clouds.push_back(std::move(transformed_cloud));
  }

  return clouds;
}

std::vector<std::shared_ptr<const KeyFrame>> SlamSystem::getKeyFrames() const
{
  return map_database_->getAllKeyFrames();
}

std::vector<std::pair<uint64_t, uint64_t>> SlamSystem::getLoopClosures() const
{
  std::lock_guard<std::mutex> lock(loop_closures_mutex_);
  return loop_closures_;
}

bool SlamSystem::shouldCreateKeyFrame(const gtsam::Pose3 & current_odom_pose) const
{
  const auto & keyframes = map_database_->getAllKeyFrames();
  if (keyframes.empty()) {
    return true;
  }

  std::shared_ptr<const KeyFrame> reference_keyframe = keyframes.back();

  double trans_dist = computeTranslationDistance(reference_keyframe->odom_pose, current_odom_pose);
  bool should_create_trans = trans_dist > parameters_->minimum_travel_distance;

  double yaw_dist = computeYawDistance(reference_keyframe->odom_pose, current_odom_pose);
  bool should_create_rot = yaw_dist > parameters_->minimum_travel_heading;

  SAM_DEBUG(
    "Keyframe (id: {}) check: translation distance = {}, rotation distance = {}, "
    "should create translation keyframe: {}, should create rotation keyframe: {}",
    reference_keyframe->key, trans_dist, yaw_dist, should_create_trans, should_create_rot);

  return should_create_trans || should_create_rot;
}

gtsam::Pose3 SlamSystem::getMapToOdom() const
{
  std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
  return latest_map_to_odom_;
}

}  // namespace glidar_slam::core
