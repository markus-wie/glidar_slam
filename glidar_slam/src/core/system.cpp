#include "glidar_slam/core/system.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "glidar_slam/core/ground_plane_extractor.hpp"
#include "glidar_slam/core/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam/core/state_serializer.hpp"
#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"
#include "pcl/common/transforms.h"
#include "pcl/filters/voxel_grid.h"

namespace glidar_slam::core {

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
  if (parameters_->lidar_voxelization_enable) {
    const auto scan =
      voxelizeLidarScan(*sensor_data.laserScan(), parameters_->lidar_voxelization_size);
    laser_scan = std::make_shared<LaserScan>(std::move(*scan));
  } else {
    laser_scan = sensor_data.laserScan();
  }

  std::vector<Point2D> interpolated_points =
    linearInterpolateScan(*laser_scan, parameters_->lidar_voxelization_size);
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

  const bool is_initialization = map_database_->size() == 0;
  if (is_initialization || tracking_reset_pending_) {
    tracking_reset_pending_ = false;

    std::optional<GroundPlaneObservation> ground_observation;
    if (sensor_data.hasVisualData()) {
      ground_observation = GroundPlaneExtractor::extract(
        sensor_data.image(), sensor_data.depth(), sensor_data.cameraModel().intrinsics(),
        sensor_data.cameraModel().baseFromCamera(), *parameters_);
    }

    gtsam::Pose3 root_pose;
    uint64_t next_key = map_database_->getNextKey();

    if (is_initialization) {
      // 1A. Standard Initialization (Strong Prior)
      root_pose = initialPoseFromGroundObservation(ground_observation);
      graph_optimizer_->initialize(root_pose, static_cast<uint64_t>(timestamp * 1e6));
    } else {
      // 1B. Tracking Reset (Weak Prior)
      root_pose = tracking_reset_pose_;

      SAM_INFO(
        "System tracking reset at timestamp: {} with weak prior pose ({}, {}, {})", timestamp,
        root_pose.translation().x(), root_pose.translation().y(), root_pose.rotation().rpy().z());

      // Inflate prior covariance massively (e.g., 10,000 variance) so it anchors the floating
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

    // 2. Create the unified KeyFrame
    gtsam::Matrix66 prior_sigmas = gtsam::Matrix66::Zero();
    prior_sigmas.diagonal() << 1e4, 1e4, 1e4, 1e4, 1e4, 1e4;

    auto new_keyframe = std::make_shared<KeyFrame>(
      next_key, timestamp, root_pose, latest_odom_pose, laser_scan, prior_sigmas,
      ground_observation);

    auto local_map =
      MapBuilder::buildLocalOccupancy(laser_scan->points(), parameters_->occ_map_resolution);

    if (ground_observation) {
      MapBuilder::addLocalGroundMap(local_map, ground_observation->ground_cloud, *parameters_);
    }

    new_keyframe->local_map =
      std::make_shared<const global_map::LocalMapData>(std::move(local_map));

    // 3. Pure Vertex Insertion (No edges! It is a new trajectory branch)
    map_database_->addKeyFrame(new_keyframe);
    map_builder_->submit(new_keyframe);

    // 4. Update internal states and Submap Grid
    {
      std::lock_guard<std::mutex> lock(latest_output_mutex_);
      latest_pose_ = root_pose;
    }
    {
      std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
      latest_map_to_odom_ = root_pose.compose(latest_odom_pose.inverse());
    }

    if (is_initialization) {
      std::vector<Point2D> initial_points =
        Utils::transformScanPoints(laser_scan->points2D(), root_pose);
      submap_grid_->add(initial_points, new_keyframe->key);

      SAM_INFO("System initialized with first keyframe at timestamp: {}", timestamp);
    } else {
      submap_grid_->reset();

      std::vector<Point2D> initial_points =
        Utils::transformScanPoints(laser_scan->points2D(), root_pose);
      submap_grid_->add(initial_points, new_keyframe->key);

      // Dispatch background loop closure to tie this weak-prior branch to the historical map.
      dispatchFindLoopClosure(*new_keyframe);

      SAM_INFO("System tracking reset at timestamp: {}", timestamp);
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

  uint64_t next_keyframe_key = map_database_->getNextKey();

  // Execute the Correlative Scan Matcher
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
  const double INF_VAR = parameters_->unobservable_variance;

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

  // Add Odometry Factor
  graph_optimizer_->addRelativeFactor(
    reference_keyframe->key, next_keyframe_key, raw_odom_delta,
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

  auto local_map =
    MapBuilder::buildLocalOccupancy(laser_scan->points(), parameters_->occ_map_resolution);
  if (ground_observation) {
    MapBuilder::addLocalGroundMap(local_map, ground_observation->ground_cloud, *parameters_);
  }
  new_keyframe->local_map = std::make_shared<const global_map::LocalMapData>(std::move(local_map));

  map_database_->addKeyFrame(new_keyframe);
  map_database_->addEdge(
    reference_keyframe->key, new_keyframe->key, MapDatabase::EdgeType::Neighbor);

  map_builder_->submit(map_database_->getKeyFrame(next_keyframe_key));

  if (parameters_->debug_timings) {
    map_ms = elapsedMilliseconds(map_start);
  }

  const auto submap_start = std::chrono::steady_clock::now();

  std::vector<Point2D> newest_points =
    Utils::transformScanPoints(laser_scan->points2D(), optimized_pose);
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
    latest_pose_ = optimized_pose;
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

gtsam::Pose3 SlamSystem::getLatestPose() const
{
  std::lock_guard<std::mutex> lock(latest_output_mutex_);
  return latest_pose_;
}

std::shared_ptr<const GlobalMapSnapshot> SlamSystem::getLatestGlobalMap() const
{
  return map_builder_->getLatest();
}

bool SlamSystem::saveState(const std::filesystem::path & path, std::string * error) const
{
  std::lock_guard<std::mutex> state_lock(state_mutex_);

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

  return StateSerializer::save(path, snapshot, error);
}

bool SlamSystem::loadState(
  const std::filesystem::path & path, const gtsam::Pose3 & initial_map_pose,
  bool use_saved_pose = false, std::string * error = nullptr)
{
  std::lock_guard<std::mutex> state_lock(state_mutex_);

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

  map_builder_ = std::make_unique<MapBuilder>(parameters_);
  map_builder_->start();
  map_builder_->rebuild(map_database_->getAllKeyFrames());

  const gtsam::Pose3 starting_pose = use_saved_pose ? snapshot.latest_pose : initial_map_pose;

  SAM_INFO(
    "Loaded state snapshot with {} keyframes, next key {}, latest pose ({}, {}, {})",
    snapshot.keyframes.size(), snapshot.next_keyframe_key, starting_pose.translation().x(),
    starting_pose.translation().y(), starting_pose.rotation().rpy().z());

  {
    std::lock_guard<std::mutex> lock(latest_output_mutex_);
    latest_pose_ = starting_pose;
    latest_low_res_debug_.reset();
    latest_high_res_debug_.reset();
    latest_ground_observation_.reset();
    latest_ground_matching_debug_.reset();
  }

  {
    std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
    latest_map_to_odom_ = starting_pose;
  }

  submap_grid_ = std::make_unique<SubmapGrid>(scan_matcher_->fieldResolutions());

  loop_closure_detector_->start();

  loop_closure_optimization_pending_ = false;
  tracking_reset_pending_ = true;
  tracking_reset_pose_ = starting_pose;

  return true;
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
      Utils::transformScanPoints(keyframe->scan->points2D(), keyframe->pose);
    rebuilt_submap->add(points, keyframe->key);
  }

  const auto end_time = std::chrono::steady_clock::now();

  auto field_resolution_duration =
    std::chrono::duration<double, std::milli>(fieldResolutions_time - start_time).count();
  auto get_keyframes_duration =
    std::chrono::duration<double, std::milli>(getkeyframes_time - fieldResolutions_time).count();
  auto rebuild_duration =
    std::chrono::duration<double, std::milli>(end_time - getkeyframes_time).count();

  SAM_INFO(
    "Rebuilt submap in {} ms (fieldResolutions: {} ms, getKeyFrames: {} ms, rebuild: {} ms)",
    field_resolution_duration + get_keyframes_duration + rebuild_duration,
    field_resolution_duration, get_keyframes_duration, rebuild_duration);

  submap_grid_ = std::move(rebuilt_submap);
}

void SlamSystem::dispatchFindLoopClosure(const KeyFrame & latest_keyframe)
{
  KeyFrame snapshot(latest_keyframe);
  loop_closure_detector_->submit(std::move(snapshot));
}

void SlamSystem::handleGroundConstraint(
  std::optional<GroundPlaneObservation> & ground_observation, uint64_t next_keyframe_key)
{
  const bool use_ground_constraint =
    ground_observation.has_value() && parameters_->ground_optimization_enable &&
    ground_observation->inlier_count >=
      static_cast<std::size_t>(parameters_->ground_minimum_inlier_count);

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
      "Ground plane factor added at keyframe {} with {} fresh inliers", next_keyframe_key,
      ground_observation->inlier_count);
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
    Utils::toPose2D(reference_keyframe->pose.inverse().compose(current_guess));

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

}  // namespace glidar_slam::core
