#include "glidar_slam/system.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "glidar_slam/ground_plane_extractor.hpp"
#include "glidar_slam/logger/logger.hpp"
#include "glidar_slam/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam/state_serializer.hpp"
#include "glidar_slam/utils.hpp"
#include "pcl/common/transforms.h"

namespace glidar_slam {

namespace {

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

gtsam::Matrix66 planarOdometryCovariance(
  const gtsam::Matrix66 & covariance, const Parameters & parameters)
{
  gtsam::Matrix66 result = covariance;
  const std::array<int, 3> unobservable_dimensions{0, 1, 5};
  for (const int dimension : unobservable_dimensions) {
    result.row(dimension).setZero();
    result.col(dimension).setZero();
    result(dimension, dimension) = parameters.unobservable_variance;
  }
  return result;
}

gtsam::Pose3 makePlanarPose(const gtsam::Pose3 & pose)
{
  const gtsam::Vector3 rpy = pose.rotation().rpy();
  const double yaw = rpy.z();
  const auto & translation = pose.translation();
  return utils::makePlanarPose(translation.x(), translation.y(), yaw);
}

gtsam::Matrix66 gtsamToEigenCovariance(
  const Eigen::Matrix3d & covariance, double unobservable_variance)
{
  gtsam::Matrix66 pose_covariance = gtsam::Matrix66::Zero();
  pose_covariance(0, 0) = unobservable_variance;
  pose_covariance(1, 1) = unobservable_variance;
  pose_covariance(2, 2) = covariance(2, 2);
  pose_covariance(3, 3) = covariance(0, 0);
  pose_covariance(4, 4) = covariance(1, 1);
  pose_covariance(5, 5) = unobservable_variance;
  pose_covariance(3, 4) = covariance(0, 1);
  pose_covariance(4, 3) = covariance(1, 0);
  pose_covariance(3, 2) = covariance(0, 2);
  pose_covariance(2, 3) = covariance(2, 0);
  pose_covariance(4, 2) = covariance(1, 2);
  pose_covariance(2, 4) = covariance(2, 1);
  return pose_covariance;
}

Eigen::Matrix3d eigenToGtsamCovariance(const gtsam::Matrix66 & cov6d)
{
  Eigen::Matrix3d cov3d;

  // Map indices: x (3), y (4), yaw (2)
  cov3d << cov6d(3, 3), cov6d(3, 4), cov6d(3, 2), cov6d(4, 3), cov6d(4, 4), cov6d(4, 2),
    cov6d(2, 3), cov6d(2, 4), cov6d(2, 2);

  return cov3d;
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
  return std::abs(utils::normalizeAngle(lhs_yaw - rhs_yaw));
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
  map_builder_ = std::make_unique<mapping::MapBuilder>(parameters_);
  map_builder_->start();

  const bool localization_startup = parameters_->mode == Parameters::Mode::Localization;
  if (!parameters_->map_load_path.empty()) {
    if (!std::filesystem::exists(parameters_->map_load_path)) {
      if (localization_startup) {
        throw std::runtime_error(
          "Startup localization requires a valid map at '" + parameters_->map_load_path + "'");
      }
      SAM_WARN(
        "Startup map path '{}' does not exist; starting with an empty map",
        parameters_->map_load_path.c_str());
    } else {
      std::string error;
      if (!loadState(
            parameters_->map_load_path, utils::toPose3(parameters_->initial_pose),
            !parameters_->initial_pose_use_provided, localization_startup, &error)) {
        if (localization_startup) {
          throw std::runtime_error(
            "Failed to load startup map '" + parameters_->map_load_path + "': " + error);
        }
        SAM_WARN(
          "Failed to load startup map '{}'; starting with an empty map: {}",
          parameters_->map_load_path.c_str(), error.c_str());
      }
    }
  } else if (localization_startup) {
    throw std::runtime_error("Startup localization requires startup.map_load_path");
  }
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
  std::unique_lock<std::mutex> state_lock(state_mutex_, std::try_to_lock);
  if (!state_lock.owns_lock()) {
    SAM_WARN("System is currently locked (likely saving). Dropping incoming frame.");
    return false;
  }

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
  double update_poses_ms = 0;
  double rebuild_submap_ms = 0;
  double loop_dispatch_ms = 0;

  if (sensor_data.laserScan()->empty()) {
    SAM_WARN("Empty laser scan received. Did not run the system!");
    return false;
  }

  const auto voxelization_start = std::chrono::steady_clock::now();

  std::shared_ptr<const LaserScan> laser_scan;
  if (parameters_->scan_voxelization_enable) {
    const PointCloudXYZPtr scan =
      utils::voxelize(sensor_data.laserScan()->points(), parameters_->scan_voxelization_size);
    laser_scan = std::make_shared<LaserScan>(scan);
  } else {
    laser_scan = sensor_data.laserScan();
  }

  if (parameters_->scan_densification_enable) {
    std::vector<Point2D> interpolated_points =
      linearInterpolateScan(*laser_scan, parameters_->scan_voxelization_size);
    laser_scan = std::make_shared<LaserScan>(std::move(interpolated_points));
  }

  if (parameters_->debug_timings) {
    lidar_voxelization_ms = elapsedMilliseconds(voxelization_start);
  }

  const auto loop_proposals_start = std::chrono::steady_clock::now();
  if (!localization_mode_) {
    processLoopClosureProposals();
  }

  if (parameters_->debug_timings) {
    loop_proposals_ms = elapsedMilliseconds(loop_proposals_start);
  }

  const auto initialization_start = std::chrono::steady_clock::now();

  const gtsam::Pose3 latest_odom_pose = makePlanarPose(odom_pose);

  if (localization_mode_) {
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    if (!localization_initialized_) {
      {
        std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
        latest_map_to_odom_ = latest_pose_.pose.compose(latest_odom_pose.inverse());

        Pose2D initial_pose2d = utils::toPose2D(latest_pose_.pose);
        Eigen::Matrix3d initial_cov = Eigen::Matrix3d::Identity() * 1e-6;
        ekf_.initialize(initial_pose2d, initial_cov);

        SAM_INFO(
          "latest_pose: ({}, {}, {}), latest_odom_pose: ({}, {}, {}), latest_map_to_odom: ({}, {}, "
          "{})",
          latest_pose_.pose.translation().x(), latest_pose_.pose.translation().y(),
          latest_pose_.pose.rotation().rpy().z(), latest_odom_pose.translation().x(),
          latest_odom_pose.translation().y(), latest_odom_pose.rotation().rpy().z(),
          latest_map_to_odom_.translation().x(), latest_map_to_odom_.translation().y(),
          latest_map_to_odom_.rotation().rpy().z());
      }

      submap_grid_->reset();
      auto nearby_keyframes = map_database_->getNearbyKeyFrames(latest_pose_.pose, 2);

      for (const auto & kf : nearby_keyframes) {
        std::vector<Point2D> pts = utils::transformScanPoints(kf->scan->points2D(), kf->pose);
        submap_grid_->add(pts, kf->key);
      }

      localization_submap_center_ = latest_pose_.pose;
      localization_initialized_ = true;
    }

    const gtsam::Pose3 odom_delta = previous_odom_pose_.inverse().compose(latest_odom_pose);

    Pose2D odom_delta_2d = utils::toPose2D(odom_delta);
    previous_odom_pose_ = latest_odom_pose;

    Eigen::Matrix3d odom_cov = eigenToGtsamCovariance(odom_covariance);

    ekf_.predict(odom_delta_2d, odom_cov);

    gtsam::Pose3 current_guess = utils::toPose3(ekf_.getState());

    const CsmResult result =
      scan_matcher_->match(*submap_grid_, laser_scan->points2D(), utils::toPose2D(current_guess));

    std::chrono::steady_clock::time_point matching_time = std::chrono::steady_clock::now();

    if (!std::isfinite(result.score) || result.score < parameters_->localization_minimum_score) {
      SAM_WARN(
        "Localization failed: score {} below threshold {}. Localization lost.", result.score,
        parameters_->localization_minimum_score);
      return false;
    }

    ekf_.update(result.optimized_pose, result.covariance);

    gtsam::Pose3 optimized_pose = utils::toPose3(ekf_.getState());

    {
      std::lock_guard<std::mutex> lock(latest_output_mutex_);
      latest_pose_.pose = optimized_pose;

      latest_pose_.covariance =
        gtsamToEigenCovariance(ekf_.getCovariance(), parameters_->unobservable_variance);
      latest_low_res_debug_ = result.low_res_debug;
      latest_high_res_debug_ = result.high_res_debug;
    }
    {
      std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
      latest_map_to_odom_ = optimized_pose.compose(latest_odom_pose.inverse());
    }

    const double distance_moved =
      (optimized_pose.translation() - localization_submap_center_.translation()).norm();
    const double shift_threshold = parameters_->minimum_travel_distance;

    if (distance_moved > shift_threshold) {
      submap_grid_->reset();
      auto nearby_keyframes = map_database_->getNearbyKeyFrames(optimized_pose, 2);
      for (const auto & kf : nearby_keyframes) {
        std::vector<Point2D> pts = utils::transformScanPoints(kf->scan->points2D(), kf->pose);
        submap_grid_->add(pts, kf->key);
      }
      localization_submap_center_ = optimized_pose;
    }

    std::chrono::steady_clock::time_point rebuild_time = std::chrono::steady_clock::now();

    if (parameters_->debug_timings) {
      const double matching_ms =
        std::chrono::duration<double, std::milli>(matching_time - start_time).count();
      const double rebuild_ms =
        std::chrono::duration<double, std::milli>(rebuild_time - matching_time).count();
      const double total_ms =
        std::chrono::duration<double, std::milli>(rebuild_time - start_time).count();

      SAM_INFO(
        "Localization timing [ms]: matching={}, rebuild={}, total={}", matching_ms, rebuild_ms,
        total_ms);
    }

    return true;
  }

  const bool is_initialization = map_database_->size() == 0;
  if (is_initialization || tracking_reset_pending_) {
    tracking_reset_pending_ = false;

    std::optional<GroundPlaneObservation> ground_observation;
    if (sensor_data.hasVisualData()) {
      GroundPlaneExtractor::Result result = GroundPlaneExtractor::extract(
        sensor_data.image(), sensor_data.depth(), sensor_data.cameraModel().intrinsics(),
        sensor_data.cameraModel().baseFromCamera(), *parameters_);

      if (!result.observation) {
        SAM_WARN("Ground plane extraction failed: {}", result.failure_reason);
      }

      ground_observation = result.observation;
      latest_ground_extraction_debug_ = result.debug_image;
    }

    gtsam::Pose3 root_pose;
    uint64_t next_key = map_database_->getNextKey();

    if (is_initialization) {
      // Standard Initialization (Strong Prior)
      root_pose = initialPoseFromGroundObservation(ground_observation);
      graph_optimizer_->initialize(root_pose, static_cast<uint64_t>(timestamp * 1e6));
    } else {
      // Tracking Reset (Weak Prior)
      root_pose = tracking_reset_pose_;

      SAM_INFO(
        "System tracking reset at timestamp: {} with weak prior pose ({}, {}, {})", timestamp,
        root_pose.translation().x(), root_pose.translation().y(), root_pose.rotation().rpy().z());

      // Inflate prior covariance massively so it anchors the floating
      // graph to prevent ISAM2 crashes, but yields immediately when loop closure arrives.
      gtsam::Matrix66 prior_sigmas = gtsam::Matrix66::Zero();
      prior_sigmas.diagonal() << 1e4, 1e4, 1e4, 1e4, 1e4, 1e4;
      graph_optimizer_->addPriorFactor(next_key, root_pose, prior_sigmas);

      handleGroundConstraint(ground_observation, next_key);

      graph_optimizer_->optimize();
    }

    SAM_INFO(
      "latest_odom_pose: ({}, {}, {})", latest_odom_pose.translation().x(),
      latest_odom_pose.translation().y(), latest_odom_pose.rotation().rpy().z());

    gtsam::Matrix66 prior_sigmas = gtsam::Matrix66::Zero();
    prior_sigmas.diagonal() << 1e4, 1e4, 1e4, 1e4, 1e4, 1e4;

    auto new_keyframe = std::make_shared<KeyFrame>(
      next_key, timestamp, root_pose, latest_odom_pose, laser_scan, prior_sigmas,
      ground_observation);

    new_keyframe->local_occupancy = std::make_shared<const mapping::LocalOccupancyMap>(
      mapping::buildOccupancy(laser_scan->points(), *parameters_));

    if (ground_observation) {
      new_keyframe->local_ground = std::make_shared<const mapping::LocalGroundMap>(
        mapping::buildGround(ground_observation->ground_cloud, *parameters_));
    }

    map_database_->addKeyFrame(new_keyframe);
    map_builder_->submit(new_keyframe);

    {
      std::lock_guard<std::mutex> lock(latest_output_mutex_);
      latest_pose_.pose = root_pose;
      latest_pose_.covariance = unobservablePoseCovariance();
    }
    {
      std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
      latest_map_to_odom_ = root_pose.compose(latest_odom_pose.inverse());
    }

    if (is_initialization) {
      std::vector<Point2D> initial_points =
        utils::transformScanPoints(laser_scan->points2D(), root_pose);
      submap_grid_->add(initial_points, new_keyframe->key);

      SAM_INFO("System initialized with first keyframe at timestamp: {}", timestamp);
    } else {
      submap_grid_->reset();

      std::vector<Point2D> initial_points =
        utils::transformScanPoints(laser_scan->points2D(), root_pose);
      submap_grid_->add(initial_points, new_keyframe->key);

      // Dispatch background loop closure to tie this weak-prior branch to the historical map.
      dispatchFindLoopClosure(*new_keyframe);

      SAM_INFO("System tracking reset at timestamp: {}", timestamp);
    }

    return true;
  }

  std::shared_ptr<const KeyFrame> reference_keyframe = map_database_->getLatestKeyFrame();

  // Get delta from the last keyframes odometry pose
  const gtsam::Pose3 odom_delta = reference_keyframe->odom_pose.inverse().compose(latest_odom_pose);
  const gtsam::Pose3 current_guess = reference_keyframe->pose.compose(odom_delta);

  // Check if we should spawn a new keyframe based on motion thresholds
  if (!shouldCreateKeyFrame(latest_odom_pose)) {
    {
      std::lock_guard<std::mutex> lock(latest_output_mutex_);
      latest_pose_.pose = current_guess;
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

    GroundPlaneExtractor::Result result = GroundPlaneExtractor::extract(
      sensor_data.image(), sensor_data.depth(), sensor_data.cameraModel().intrinsics(),
      sensor_data.cameraModel().baseFromCamera(), *parameters_);

    if (!result.observation) {
      SAM_WARN("Ground plane extraction failed: {}", result.failure_reason);
    }

    ground_observation = result.observation;
    latest_ground_extraction_debug_ = result.debug_image;

    if (parameters_->debug_timings) {
      ground_extraction_ms = elapsedMilliseconds(start);
    }
  }

  const auto csm_start = std::chrono::steady_clock::now();

  uint64_t next_keyframe_key = map_database_->getNextKey();

  // Execute the Correlative Scan Matcher
  const CsmResult csm_result =
    scan_matcher_->match(*submap_grid_, laser_scan->points2D(), utils::toPose2D(current_guess));

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
    latest_ground_extraction_debug_.reset();
  }

  // Map the 3x3 CSM Covariance (x, y, yaw) to a 6x6 GTSAM Covariance Matrix
  const Eigen::Matrix3d relative_csm_covariance =
    transformScanMatchCovarianceToRelativeFrame(csm_result.covariance, reference_keyframe->pose);
  const gtsam::Matrix66 csm_covariance =
    gtsamToEigenCovariance(relative_csm_covariance, parameters_->unobservable_variance);

  if (parameters_->debug_timings) {
    SAM_INFO(
      "CSM Covariance (diagonal): x={}, y={}, yaw={}", csm_result.covariance(0, 0),
      csm_result.covariance(1, 1), csm_result.covariance(2, 2));
  }

  // Convert optimized absolute Pose2D back to gtsam::Pose3
  gtsam::Pose3 optimized_world_pose = utils::toPose3(csm_result.optimized_pose);

  // Calculate the relative transform factor for the graph
  const gtsam::Pose3 csm_pose_delta =
    reference_keyframe->pose.inverse().compose(optimized_world_pose);

  // Add Odometry Factor
  graph_optimizer_->addRelativeFactor(
    reference_keyframe->key, next_keyframe_key, odom_delta,
    planarOdometryCovariance(odom_covariance, *parameters_));

  // Add LiDAR Scan Matching Factor
  graph_optimizer_->addRelativeFactor(
    reference_keyframe->key, next_keyframe_key, csm_pose_delta, csm_covariance);

  handleGroundMatchingConstraint(
    reference_keyframe, ground_observation, current_guess, next_keyframe_key);

  handleGroundConstraint(ground_observation, next_keyframe_key);

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

  new_keyframe->buildLocalMaps(*parameters_);

  map_database_->addKeyFrame(new_keyframe);
  map_database_->addEdge(
    reference_keyframe->key, new_keyframe->key, MapDatabase::EdgeType::Neighbor);

  map_builder_->submit(map_database_->getKeyFrame(next_keyframe_key));

  if (parameters_->debug_timings) {
    map_ms = elapsedMilliseconds(map_start);
  }

  const auto submap_start = std::chrono::steady_clock::now();

  std::vector<Point2D> newest_points =
    utils::transformScanPoints(laser_scan->points2D(), optimized_pose);
  submap_grid_->add(newest_points, next_keyframe_key);
  while (submap_grid_->size() > static_cast<std::size_t>(parameters_->submap_window_size)) {
    submap_grid_->removeOldestKeyframe();
  }

  if (parameters_->debug_timings) {
    submap_ms = elapsedMilliseconds(submap_start);
  }

  const auto update_covariance_start = std::chrono::steady_clock::now();

  std::unordered_map<uint64_t, gtsam::Matrix66> optimized_covariances;
  // TODO: This can get really costly.. Should at least be in some background thread
  // The reason we need is for the loop closure proximity checker to correctly calculate the
  // Mahalanobis distance between two keyframes.
  if (loop_closure_optimization_pending_) {
    for (const auto & keyframe : map_database_->getAllKeyFrames()) {
      if (const auto covariance = graph_optimizer_->getMarginalCovariance(keyframe->key)) {
        optimized_covariances.emplace(keyframe->key, *covariance);
      }
    }
  }

  if (parameters_->debug_timings) {
    update_covariance_ms = elapsedMilliseconds(update_covariance_start);
  }

  const auto update_poses_start = std::chrono::steady_clock::now();

  for (const auto & keyframe : map_database_->updatePoses(updated_states, optimized_covariances)) {
    map_builder_->submit(keyframe);
  }

  if (parameters_->debug_timings) {
    update_poses_ms = elapsedMilliseconds(update_poses_start);
  }

  const auto rebuild_submap_start = std::chrono::steady_clock::now();

  if (loop_closure_optimization_pending_) {
    map_builder_->rebuild(map_database_->getAllKeyFrames());
    rebuildSubmap();
  }
  loop_closure_optimization_pending_ = false;

  if (parameters_->debug_timings) {
    rebuild_submap_ms = elapsedMilliseconds(rebuild_submap_start);
  }

  {
    std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
    gtsam::Pose3 odom_to_base = latest_odom_pose.inverse();
    latest_map_to_odom_ = optimized_pose.compose(odom_to_base);
  }

  const auto loop_dispatch_start = std::chrono::steady_clock::now();

  dispatchFindLoopClosure(*new_keyframe);

  if (parameters_->debug_timings) {
    loop_dispatch_ms = elapsedMilliseconds(loop_dispatch_start);
  }

  {
    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    latest_pose_.pose = optimized_pose;
    latest_pose_.covariance = optimized_covariance;
  }

  if (parameters_->debug_timings) {
    SAM_INFO(
      "SLAM SYSTEM timings [ms]: lidar_voxelization={}, loop_proposals={}, initialization={}, "
      "ground_extraction={}, "
      "csm={}, factor_preparation={}, "
      "optimization={}, keyframe={}, map={}, submap={}, update_covariance={}, update_poses={}, "
      "rebuild_submap={}, "
      "loop_dispatch={}, "
      "total={}",
      lidar_voxelization_ms, loop_proposals_ms, initialization_ms, ground_extraction_ms, csm_ms,
      factor_preparation_ms, optimization_ms, keyframe_ms, map_ms, submap_ms, update_covariance_ms,
      update_poses_ms, rebuild_submap_ms, loop_dispatch_ms, elapsedMilliseconds(process_start));
  }

  return true;
}

PoseEstimate SlamSystem::getLatestPoseAndCovariance() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_pose_;
}

std::shared_ptr<const mapping::GlobalMapSnapshot> SlamSystem::getLatestGlobalMap() const
{
  return map_builder_->getLatest();
}

bool SlamSystem::saveState(const std::filesystem::path & path, std::string * error) const
{
  std::lock_guard<std::mutex> state_lock(state_mutex_);

  SAM_INFO("Saving SLAM state snapshot to '{}'", path.string());

  SlamStateSnapshot snapshot;

  snapshot.state_id = std::to_string(
    static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
  snapshot.next_keyframe_key = map_database_->getNextKey();
  snapshot.optimizer_initialized = graph_optimizer_->isInitialized();
  snapshot.optimizer_latest_key = graph_optimizer_->getLatestKey();
  snapshot.optimizer_latest_timestamp = graph_optimizer_->getLatestTimestamp();
  snapshot.latest_pose = graph_optimizer_->getLatestPose();
  snapshot.factors = graph_optimizer_->getSerializedFactors();
  snapshot.loop_closures = map_database_->getLoopClosures();

  const auto keyframes = map_database_->getAllKeyFrames();
  snapshot.keyframes.reserve(keyframes.size());

  for (const auto & keyframe : keyframes) {
    snapshot.keyframes.push_back(*keyframe);
  }

  if (!snapshot.keyframes.empty()) {
    snapshot.initial_pose = snapshot.keyframes.front().pose;
    // Keep the saved transform consistent with the pose/odometry pair that defines the
    // latest map state. This avoids restoring a stale map-to-odom transform.
    const auto & latest_keyframe = snapshot.keyframes.back();
    snapshot.latest_pose = latest_keyframe.pose;
  }

  snapshot.map_to_odom = getMapToOdom();

  bool success = StateSerializer::save(path, snapshot, error);

  if (success) {
    SAM_INFO(
      "SLAM state snapshot successfully saved ({} keyframes, {} factors, {} loop closures)",
      snapshot.keyframes.size(), snapshot.factors.size(), snapshot.loop_closures.size());
  } else {
    SAM_ERROR("Failed to save SLAM state snapshot: {}", error ? *error : "unknown error");
  }

  return success;
}

bool SlamSystem::loadState(
  const std::filesystem::path & path, const gtsam::Pose3 & initial_map_pose, bool use_saved_pose,
  bool localization_only, std::string * error)
{
  std::lock_guard<std::mutex> state_lock(state_mutex_);

  SAM_INFO("Loading SLAM state snapshot from '{}'", path.string());

  SlamStateSnapshot snapshot;
  if (!StateSerializer::load(path, snapshot, error)) {
    return false;
  }

  // No worker may inspect the database or enqueue proposals while the graph and keyframes are
  // being replaced
  loop_closure_detector_->stop();
  map_builder_->stop();

  bool map_database_restore_success =
    map_database_->restore(snapshot.keyframes, snapshot.next_keyframe_key);

  bool graph_optimizer_restore_success = graph_optimizer_->restore(snapshot);

  if (!map_database_restore_success || !graph_optimizer_restore_success) {
    map_builder_->start();
    loop_closure_detector_->start();
    if (error) {
      *error = "state snapshot is inconsistent";
    }
    return false;
  }

  for (const auto & [from_key, to_key] : snapshot.loop_closures) {
    map_database_->addEdge(from_key, to_key, MapDatabase::EdgeType::LoopClosure);
  }

  map_builder_ = std::make_unique<mapping::MapBuilder>(parameters_);
  map_builder_->start();
  map_builder_->rebuild(map_database_->getAllKeyFrames());

  const gtsam::Pose3 starting_pose = use_saved_pose ? snapshot.latest_pose : initial_map_pose;

  SAM_INFO(
    "Loaded state snapshot with {} keyframes, next key {}, latest pose ({}, {}, {})",
    snapshot.keyframes.size(), snapshot.next_keyframe_key, starting_pose.translation().x(),
    starting_pose.translation().y(), starting_pose.rotation().rpy().z());

  {
    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    latest_pose_.pose = starting_pose;
    latest_pose_.covariance = unobservablePoseCovariance();
    latest_low_res_debug_.reset();
    latest_high_res_debug_.reset();
    latest_ground_observation_.reset();
    latest_ground_matching_debug_.reset();
    latest_ground_extraction_debug_.reset();
  }

  {
    std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
    latest_map_to_odom_ = starting_pose;
  }

  submap_grid_ = std::make_unique<SubmapGrid>(scan_matcher_->fieldResolutions());

  loop_closure_optimization_pending_ = false;
  localization_mode_ = localization_only;
  localization_initialized_ = false;

  if (localization_mode_) {
    tracking_reset_pending_ = false;
    SAM_INFO("Map loaded successfully in LOCALIZATION ONLY mode.");
  } else if (use_saved_pose) {
    // 1. Clean continuous resume: Link directly to the last keyframe
    tracking_reset_pending_ = false;
    loop_closure_detector_->start();
    SAM_INFO("Map loaded successfully. Resuming continuous SLAM session.");
  } else {
    // 2. Teleport / Relocalization: Needs proximity match
    tracking_reset_pending_ = true;
    tracking_reset_pose_ = starting_pose;
    loop_closure_detector_->start();
    SAM_INFO("Map loaded successfully. Tracking reset pending at custom pose.");
  }

  return true;
}

bool SlamSystem::isLocalizationMode() const
{
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  return localization_mode_;
}

bool SlamSystem::setLocalizationMode(
  bool enable, const gtsam::Pose3 & initial_map_pose, bool use_current_pose, std::string * error)
{
  std::lock_guard<std::mutex> state_lock(state_mutex_);

  if (enable) {
    if (localization_mode_) {
      if (error) {
        *error = "localization mode is already active";
      }
      return false;
    }
    if (map_database_->size() == 0) {
      if (error) {
        *error = "cannot enter localization mode without a loaded map";
      }
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(latest_output_mutex_);
      latest_pose_.pose = use_current_pose ? latest_pose_.pose : initial_map_pose;
      latest_pose_.covariance = unobservablePoseCovariance();
    }

    localization_initialized_ = false;
    localization_mode_ = true;

    tracking_reset_pending_ = false;
    loop_closure_detector_->stop();

    return true;
  }

  if (!localization_mode_) {
    if (error) {
      *error = "localization mode is not active";
    }
    return false;
  }

  localization_mode_ = false;
  localization_initialized_ = false;

  tracking_reset_pending_ = true;
  {
    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    tracking_reset_pose_ = latest_pose_.pose;
  }

  loop_closure_detector_->start();

  return true;
}

void SlamSystem::rebuildGlobalMap() const
{
  SAM_INFO("Rebuilding global map requested...");
  map_database_->rebuildLocalMaps(*parameters_);
  map_builder_->rebuild(map_database_->getAllKeyFrames());
}

std::size_t SlamSystem::getFactorCount() const
{
  return graph_optimizer_->getSerializedFactors().size();
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

PointCloudXYZRGBAConstPtr SlamSystem::getLatestGroundMatchingDebug() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_ground_matching_debug_;
}

std::optional<cv::Mat> SlamSystem::getLatestGroundExtractionDebug() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_ground_extraction_debug_;
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

    map_database_->addEdge(proposal.from_key, proposal.to_key, MapDatabase::EdgeType::LoopClosure);

    SAM_INFO(
      "Loop closure accepted: from={}, to={}, score={}", proposal.from_key, proposal.to_key,
      proposal.score);
  }

  loop_closure_optimization_pending_ = loop_closure_optimization_pending_ || !proposals.empty();

  return !proposals.empty();
}

void SlamSystem::rebuildSubmap()
{
  auto start_time = std::chrono::steady_clock::now();

  std::vector<double> resolutions;
  resolutions = scan_matcher_->fieldResolutions();

  auto fieldResolutions_time = std::chrono::steady_clock::now();

  auto rebuilt_submap = std::make_unique<SubmapGrid>(resolutions);

  const auto latest_keyframe = map_database_->getLatestKeyFrame();

  auto getkeyframes_time = std::chrono::steady_clock::now();

  const std::vector<std::shared_ptr<const KeyFrame>> keyframes =
    map_database_->getKeyFrameWindow(latest_keyframe->key, parameters_->submap_window_size);

  for (const std::shared_ptr<const KeyFrame> & keyframe : keyframes) {
    const std::vector<Point2D> points =
      utils::transformScanPoints(keyframe->scan->points2D(), keyframe->pose);
    rebuilt_submap->add(points, keyframe->key);
  }

  const auto end_time = std::chrono::steady_clock::now();

  auto field_resolution_duration =
    std::chrono::duration<double, std::milli>(fieldResolutions_time - start_time).count();
  auto get_keyframes_duration =
    std::chrono::duration<double, std::milli>(getkeyframes_time - fieldResolutions_time).count();
  auto rebuild_duration =
    std::chrono::duration<double, std::milli>(end_time - getkeyframes_time).count();

  if (parameters_->debug_timings) {
    SAM_INFO(
      "Rebuilt submap in {} ms (fieldResolutions: {} ms, getKeyFrames: {} ms, rebuild: {} ms)",
      field_resolution_duration + get_keyframes_duration + rebuild_duration,
      field_resolution_duration, get_keyframes_duration, rebuild_duration);
  }

  submap_grid_ = std::move(rebuilt_submap);
}

void SlamSystem::dispatchFindLoopClosure(const KeyFrame & latest_keyframe)
{
  KeyFrame snapshot(latest_keyframe);
  loop_closure_detector_->submit(std::move(snapshot));
}

gtsam::Matrix66 SlamSystem::unobservablePoseCovariance()
{
  gtsam::Matrix66 covariance = gtsam::Matrix66::Zero();
  covariance.diagonal().setConstant(parameters_->unobservable_variance);
  return covariance;
}

void SlamSystem::handleGroundConstraint(
  std::optional<GroundPlaneObservation> & ground_observation, uint64_t next_keyframe_key)
{
  // arbitrarily chosen threshold under which a realistic ground plane might not be observable
  constexpr std::size_t MINIMUM_INLIER_COUNT = 50;

  const bool use_ground_constraint = ground_observation.has_value() &&
                                     parameters_->ground_optimization_enable &&
                                     ground_observation->inlier_count >= MINIMUM_INLIER_COUNT;

  if (!use_ground_constraint) {
    return;
  }

  gtsam::Vector3 ground_normal_in_base = ground_observation->normal_in_base.cast<double>();
  double ground_distance_to_base = static_cast<double>(ground_observation->distance_to_base);

  graph_optimizer_->addGroundPlaneFactor(
    next_keyframe_key, ground_normal_in_base, ground_distance_to_base, gtsam::Vector3::UnitZ(), 0.0,
    parameters_->ground_normal_sigma, parameters_->ground_distance_sigma);

  if (parameters_->ground_debug_enable) {
    SAM_INFO(
      "Ground plane factor added at keyframe {} with {} fresh inliers. Ground normal: ({}, {}, "
      "{}), Ground distance: {}",
      next_keyframe_key, ground_observation->inlier_count, ground_normal_in_base.x(),
      ground_normal_in_base.y(), ground_normal_in_base.z(), ground_distance_to_base);
  }
}

void SlamSystem::handleGroundMatchingConstraint(
  const std::shared_ptr<const KeyFrame> & reference_keyframe,
  std::optional<GroundPlaneObservation> & ground_observation, const gtsam::Pose3 & current_guess,
  uint64_t next_keyframe_key)
{
  const bool perform_ground_matching = parameters_->ground_matching_enable &&
                                       reference_keyframe->ground_observation && ground_observation;
  if (!perform_ground_matching) {
    return;
  }

  const double INF_VAR = parameters_->unobservable_variance;
  const Pose2D relative_ground_pose_estimate =
    utils::toPose2D(reference_keyframe->pose.inverse().compose(current_guess));

  const auto ground_match = ground_marking_matcher_->match(
    *reference_keyframe->ground_observation, *ground_observation, relative_ground_pose_estimate);

  if (!ground_match || ground_match->score < parameters_->ground_matching_minimum_score) {
    return;
  }

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
    reference_keyframe->key, next_keyframe_key, utils::toPose3(ground_match->optimized_pose),
    ground_covariance);

  if (parameters_->ground_matching_debug_enable) {
    const PointCloudXYZRGBAPtr reference_debug_cloud = ground_marking_matcher_->makeDebugCloud(
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

    PointCloudXYZRGBAPtr map_debug_cloud = std::make_shared<PointCloudXYZRGBA>();
    pcl::transformPointCloud(*reference_debug_cloud, *map_debug_cloud, map_from_reference);

    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    latest_ground_matching_debug_ = std::move(map_debug_cloud);
    latest_low_res_debug_ = ground_match->low_res_debug;
    latest_high_res_debug_ = ground_match->high_res_debug;

    SAM_INFO(
      "Ground marking matching result: optimized_pose=({}, {}, {}), covariance=({})",
      ground_match->optimized_pose.x, ground_match->optimized_pose.y,
      ground_match->optimized_pose.yaw, ground_match->covariance.diagonal().transpose());
    SAM_INFO(
      "Ground marking factor added at keyframe {} with score {}", next_keyframe_key,
      ground_match->score);
  }
}

std::vector<std::shared_ptr<const KeyFrame>> SlamSystem::getKeyFrames() const
{
  return map_database_->getAllKeyFrames();
}

std::vector<std::pair<uint64_t, uint64_t>> SlamSystem::getLoopClosures() const
{
  return map_database_->getLoopClosures();
}

std::vector<MapDatabase::GraphEdge> SlamSystem::getEdges() const
{
  return map_database_->getEdges();
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

}  // namespace glidar_slam
