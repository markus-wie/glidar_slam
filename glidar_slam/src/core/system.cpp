#include "glidar_slam/core/system.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <unordered_map>

#include "glidar_slam/core/correlative_scan_matcher.hpp"
#include "glidar_slam/core/ground_plane_extractor.hpp"
#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"
#include "pcl/common/transforms.h"

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

void collectVisiblePoints(
  const PointCloudXYZ & scan, const gtsam::Pose3 & pose, const Point2D & viewpoint,
  std::vector<Point2D> & output)
{
  const size_t world_points_start = output.size();
  output.reserve(output.size() + 2 * scan.size());
  for (const auto & point : scan) {
    const gtsam::Point3 world_point = pose.transformFrom(gtsam::Point3(point.x, point.y, point.z));
    if (std::isfinite(world_point.x()) && std::isfinite(world_point.y())) {
      output.push_back({world_point.x(), world_point.y()});
    }
  }

  const size_t world_points_end = output.size();
  if (world_points_start == world_points_end) {
    return;
  }

  // TODO: check what it does exactly. basically voxilization and a trick to filter "ghost" points
  constexpr double min_square_distance = 0.1 * 0.1;
  size_t trailing_point = world_points_start;
  Point2D first_point{};
  bool first_time = true;

  for (size_t point = world_points_start; point < world_points_end; ++point) {
    const Point2D current_point = output[point];
    if (first_time) {
      first_point = current_point;
      first_time = false;
    }

    const double delta_x = first_point.x - current_point.x;
    const double delta_y = first_point.y - current_point.y;
    if (delta_x * delta_x + delta_y * delta_y <= min_square_distance) {
      continue;
    }

    // Keep only the contiguous side of the scan that faces the matcher viewpoint.
    const double a = viewpoint.y - first_point.y;
    const double b = first_point.x - viewpoint.x;
    const double c = first_point.y * viewpoint.x - first_point.x * viewpoint.y;
    const double side = current_point.x * a + current_point.y * b + c;
    first_point = current_point;

    if (side < 0.0) {
      trailing_point = point;
      continue;
    }

    for (size_t segment_point = trailing_point; segment_point < point; ++segment_point) {
      output.push_back(output[segment_point]);
    }
    trailing_point = point;
  }
}

}  // namespace

SlamSystem::SlamSystem(const std::shared_ptr<Parameters> & parameters) : parameters_(parameters)
{
  map_database_ = std::make_shared<MapDatabase>();
  graph_optimizer_ = std::make_unique<GraphOptimizer>(parameters_);
  scan_matcher_ = std::make_unique<CorrelativeScanMatcher>(parameters_);
  ground_marking_matcher_ = std::make_unique<GroundMarkingMatcher>(parameters_);
  loop_closure_detector_ = std::make_unique<LoopClosureDetector>(parameters_, map_database_);
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
  double ground_extraction_ms = 0;
  double loop_proposals_ms = 0;
  double submap_ms = 0;
  double csm_ms = 0;
  double optimization_ms = 0;
  double loop_dispatch_ms = 0;

  const std::shared_ptr<const LaserScan> & laser_scan = sensor_data.laserScan();
  if (laser_scan->empty()) {
    SAM_WARN("Empty laser scan received. Did not run the system!");
    return false;
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

  const auto loop_proposals_start = std::chrono::steady_clock::now();
  processLoopClosureProposals();

  if (parameters_->debug_timings) {
    loop_proposals_ms = elapsedMilliseconds(loop_proposals_start);
  }

  const gtsam::Pose3 latest_odom_pose = makePlanarPose(odom_pose);

  if (map_database_->size() == 0) {
    const gtsam::Pose3 initial_pose = initialPoseFromGroundObservation(ground_observation);
    graph_optimizer_->initialize(initial_pose, static_cast<uint64_t>(timestamp * 1e6));
    auto new_keyframe = std::make_shared<KeyFrame>(
      map_database_->incrementNextKey(), timestamp, initial_pose, latest_odom_pose, laser_scan,
      std::nullopt, ground_observation);
    auto local_map = buildLocalOccupancy(laser_scan->points(), parameters_->occ_map_resolution);
    if (ground_observation) {
      addLocalGroundMap(local_map, ground_observation->ground_cloud, *parameters_);
    }
    new_keyframe->local_map =
      std::make_shared<const global_map::LocalMapData>(std::move(local_map));
    const uint64_t initial_key = new_keyframe->key;
    map_database_->addKeyFrame(new_keyframe);
    map_builder_->submit(map_database_->getSnapshot(initial_key));

    SAM_INFO("System initialized with first keyframe at timestamp: {}", timestamp);
  }

  std::shared_ptr<const KeyFrame> reference_keyframe = map_database_->getLatestKeyFrame();

  // Get relative odometry delta and apply to the reference keyframes world pose
  const gtsam::Pose3 raw_odom_delta =
    reference_keyframe->odom_pose.inverse().compose(latest_odom_pose);
  const gtsam::Pose3 current_guess = reference_keyframe->pose.compose(raw_odom_delta);

  // Check if we should spawn a new keyframe based on motion thresholds
  if (!shouldCreateKeyFrame(latest_odom_pose)) {
    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    latest_pose_ = current_guess;
    return false;
  }

  uint64_t next_keyframe_key = map_database_->incrementNextKey();

  // Construct the Submap Reference Scan
  // Aggregate the points of the last N keyframes transformed into the world frame
  const auto submap_start = std::chrono::steady_clock::now();
  std::vector<Point2D> reference_points;

  std::vector<std::shared_ptr<const KeyFrame>> keyframes = map_database_->getAllKeyFrames();

  size_t start_idx = static_cast<int>(keyframes.size()) > parameters_->submap_window_size
                       ? static_cast<int>(keyframes.size()) - parameters_->submap_window_size
                       : 0;

  const Point2D viewpoint{current_guess.x(), current_guess.y()};

  for (size_t i = start_idx; i < keyframes.size(); ++i) {
    const KeyFrame & kf = *keyframes.at(i);
    collectVisiblePoints(kf.scan->points(), kf.pose, viewpoint, reference_points);
  }
  if (parameters_->debug_timings) {
    submap_ms = elapsedMilliseconds(submap_start);
  }

  // Execute the Correlative Scan Matcher
  // This yields an optimized world pose and a 3x3 covariance matrix
  const auto csm_start = std::chrono::steady_clock::now();
  const CsmResult csm_result =
    scan_matcher_->match(reference_points, laser_scan->points2D(), Utils::toPose2D(current_guess));
  if (parameters_->debug_timings) {
    csm_ms = elapsedMilliseconds(csm_start);
  }

  {
    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    latest_low_res_debug_ = csm_result.low_res_debug;
    latest_high_res_debug_ = csm_result.high_res_debug;
    latest_ground_observation_ = ground_observation;
    latest_ground_matching_debug_.reset();
  }

  // Map the 3x3 CSM Covariance (x, y, yaw) to a 6x6 GTSAM Covariance Matrix
  gtsam::Matrix66 csm_covariance = gtsam::Matrix66::Zero();

  // Inflate unobservable dimensions
  constexpr double INF_VAR = 1e6;

  // Diagonal of csm_result.covariance:
  // [ x-x, y-y, yaw-yaw ]
  // gtsam covariance layout is [roll, pitch, yaw, x, y, z]
  csm_covariance(0, 0) = INF_VAR;                      // roll-roll
  csm_covariance(1, 1) = INF_VAR;                      // pitch-pitch
  csm_covariance(2, 2) = csm_result.covariance(2, 2);  // yaw-yaw
  csm_covariance(3, 3) = csm_result.covariance(0, 0);  // x-x
  csm_covariance(4, 4) = csm_result.covariance(1, 1);  // y-y
  csm_covariance(5, 5) = INF_VAR;                      // z-z
  csm_covariance(3, 4) = csm_result.covariance(0, 1);  // x-y
  csm_covariance(4, 3) = csm_result.covariance(1, 0);  // y-x
  csm_covariance(3, 2) = csm_result.covariance(0, 2);  // x-yaw
  csm_covariance(2, 3) = csm_result.covariance(2, 0);  // yaw-x
  csm_covariance(4, 2) = csm_result.covariance(1, 2);  // y-yaw
  csm_covariance(2, 4) = csm_result.covariance(2, 1);  // yaw-y

  if (parameters_->csm_debug_enable) {
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

  const auto optimization_start = std::chrono::steady_clock::now();
  gtsam::Values updated_states = graph_optimizer_->optimize();
  if (parameters_->debug_timings) {
    optimization_ms = elapsedMilliseconds(optimization_start);
  }

  gtsam::Pose3 optimized_pose = graph_optimizer_->getLatestPose();

  gtsam::Matrix66 optimized_covariance =
    graph_optimizer_->getMarginalCovariance(next_keyframe_key).value_or(gtsam::Matrix66::Zero());

  if (parameters_->csm_debug_enable) {
    SAM_INFO(
      "GTSAM Covariance (diagonal): x={}, y={}, z={}", optimized_covariance(3, 3),
      optimized_covariance(4, 4), optimized_covariance(5, 5));
  }

  const std::shared_ptr<KeyFrame> new_keyframe = std::make_shared<KeyFrame>(
    next_keyframe_key, timestamp, optimized_pose, latest_odom_pose, laser_scan,
    optimized_covariance, ground_observation);

  auto local_map = buildLocalOccupancy(laser_scan->points(), parameters_->occ_map_resolution);
  if (ground_observation) {
    addLocalGroundMap(local_map, ground_observation->ground_cloud, *parameters_);
  }
  new_keyframe->local_map = std::make_shared<const global_map::LocalMapData>(std::move(local_map));

  map_database_->addKeyFrame(new_keyframe);
  map_builder_->submit(map_database_->getSnapshot(next_keyframe_key));

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
  }
  loop_closure_optimization_pending_ = false;

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
    latest_pose_ = optimized_pose;
  }

  if (parameters_->debug_timings) {
    SAM_INFO(
      "SLAM timing [ms]: ground_extraction={}, loop_proposals={}, submap={}, "
      "csm={}, optimization={}, loop_dispatch={}, total={}",
      ground_extraction_ms, loop_proposals_ms, submap_ms, csm_ms, optimization_ms, loop_dispatch_ms,
      elapsedMilliseconds(process_start));
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
