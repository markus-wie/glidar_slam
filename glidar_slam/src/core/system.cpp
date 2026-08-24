#include "glidar_slam/core/system.hpp"

#include <algorithm>
#include <cmath>

#include "glidar_slam/core/correlative_scan_matcher.hpp"
#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"
#include "pcl/common/transforms.h"

namespace glidar_slam::core {

namespace {

void appendKartoVisiblePoints(
  const PointCloudXYZ & scan, const gtsam::Pose3 & pose, const Point2D & viewpoint,
  std::vector<Point2D> & output)
{
  std::vector<Point2D> world_points;
  world_points.reserve(scan.size());
  for (const auto & point : scan) {
    const gtsam::Point3 world_point = pose.transformFrom(gtsam::Point3(point.x, point.y, point.z));
    if (std::isfinite(world_point.x()) && std::isfinite(world_point.y())) {
      world_points.push_back({world_point.x(), world_point.y()});
    }
  }

  if (world_points.empty()) {
    return;
  }

  output.insert(output.end(), world_points.begin(), world_points.end());

  constexpr double min_square_distance = 0.1 * 0.1;
  auto trailing_point = world_points.begin();
  Point2D first_point{};
  bool first_time = true;

  for (auto point = world_points.begin(); point != world_points.end(); ++point) {
    const Point2D current_point = *point;
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

    output.insert(output.end(), trailing_point, point);
    trailing_point = point;
  }
}

}  // namespace

SlamSystem::SlamSystem(const std::shared_ptr<Parameters> & parameters)
: parameters_(parameters), graph_optimizer_(parameters), next_keyframe_key_(0)
{
}

SlamSystem::LaserScanOutput SlamSystem::handleLaserScan(
  double timestamp, const PointCloudXYZ & scan, const gtsam::Pose3 & odom_pose,
  const gtsam::Matrix66 & odom_covariance)
{
  if (scan.empty()) {
    return {};
  }

  const gtsam::Pose3 latest_odom_pose = projectPlanar(odom_pose);

  initializeIfNeeded(timestamp, scan, latest_odom_pose);
  if (keyframes_.empty()) {
    return {};
  }

  std::shared_ptr<KeyFrame> reference_keyframe = keyframes_.back();

  // Get relative odometry delta and apply to the reference keyframes world pose
  const gtsam::Pose3 raw_odom_delta =
    reference_keyframe->odom_pose.inverse().compose(latest_odom_pose);
  const gtsam::Pose3 current_guess = reference_keyframe->pose.compose(raw_odom_delta);

  // Check if we should spawn a new keyframe based on motion thresholds
  if (!shouldCreateKeyFrame(latest_odom_pose)) {
    return {current_guess, std::nullopt, std::nullopt};
  }

  // Construct the Current Scan (with world_pose acting as the initial guess)
  LaserScan current_csm_scan;
  current_csm_scan.id = next_keyframe_key_;
  current_csm_scan.timestamp = timestamp;
  current_csm_scan.odom_pose = Utils::toPose2D(current_guess);
  current_csm_scan.world_pose = Utils::toPose2D(current_guess);

  for (const auto & pt : scan) {
    current_csm_scan.points.push_back({pt.x, pt.y});
  }

  // Construct the Submap Reference Scan
  // Aggregate the points of the last N keyframes transformed into the world frame
  LaserScan submap_reference;
  submap_reference.id = reference_keyframe->key;
  submap_reference.timestamp = reference_keyframe->timestamp;
  submap_reference.world_pose = Utils::toPose2D(reference_keyframe->pose);

  size_t start_idx = static_cast<int>(keyframes_.size()) > parameters_->submap_window_size
                       ? static_cast<int>(keyframes_.size()) - parameters_->submap_window_size
                       : 0;

  const Point2D viewpoint{current_guess.x(), current_guess.y()};
  {
    std::lock_guard<std::mutex> lock(keyframes_mutex_);

    for (size_t i = start_idx; i < keyframes_.size(); ++i) {
      const KeyFrame & kf = *keyframes_.at(i);
      appendKartoVisiblePoints(kf.scan, kf.pose, viewpoint, submap_reference.points);
    }
  }

  // Execute the Correlative Scan Matcher
  // This yields an optimized world pose and a 3x3 covariance matrix
  CsmResult csm_result =
    CorrelativeScanMatcher::match(submap_reference, current_csm_scan, *parameters_);

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

  SAM_INFO(
    "CSM Covariance (diagonal): x={}, y={}, yaw={}", csm_result.covariance(0, 0),
    csm_result.covariance(1, 1), csm_result.covariance(2, 2));

  // Convert optimized absolute Pose2D back to gtsam::Pose3
  gtsam::Pose3 optimized_world_pose = Utils::toPose3(csm_result.optimized_pose);

  // Calculate the relative transform factor for the graph
  // Factor = T_world_ref.inverse() * T_world_optimized_curr
  const gtsam::Pose3 csm_pose_delta =
    reference_keyframe->pose.inverse().compose(optimized_world_pose);

  // Add the Wheel Odometry Factor
  graph_optimizer_.addRelativeFactor(
    reference_keyframe->key, next_keyframe_key_, raw_odom_delta, odom_covariance);

  // Add the Scan Matching Factor (in parallel)
  graph_optimizer_.addRelativeFactor(
    reference_keyframe->key, next_keyframe_key_, csm_pose_delta, csm_covariance);

  gtsam::Values updated_states = graph_optimizer_.optimize();

  synchronizeKeyframes();

  gtsam::Pose3 optimized_pose = graph_optimizer_.getLatestPose();

  gtsam::Matrix66 optimized_covariance =
    graph_optimizer_.getMarginalCovariance(next_keyframe_key_).value_or(gtsam::Matrix66::Zero());

  SAM_INFO(
    "Covariance (diagonal) of latest pose: roll={}, pitch={}, yaw={}, x={}, y={}, z={}",
    optimized_covariance(0, 0), optimized_covariance(1, 1), optimized_covariance(2, 2),
    optimized_covariance(3, 3), optimized_covariance(4, 4), optimized_covariance(5, 5));

  {
    std::lock_guard<std::mutex> lock(keyframes_mutex_);
    keyframes_.emplace_back(std::make_shared<KeyFrame>(
      next_keyframe_key_, timestamp, optimized_pose, latest_odom_pose, scan, optimized_covariance));
  }
  ++next_keyframe_key_;

  {
    std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
    gtsam::Pose3 odom_to_base = latest_odom_pose.inverse();
    latest_map_to_odom_ = optimized_pose.compose(odom_to_base);
  }

  return {optimized_pose, csm_result.low_res_debug, csm_result.high_res_debug};
}

void SlamSystem::synchronizeKeyframes()
{
  std::lock_guard<std::mutex> lock(keyframes_mutex_);

  const gtsam::Values & current_estimates = graph_optimizer_.getCurrentEstimates();

  for (const std::shared_ptr<KeyFrame> & keyframe : keyframes_) {
    if (current_estimates.exists(keyframe->key)) {
      keyframe->pose = current_estimates.at<gtsam::Pose3>(keyframe->key);
    }
  }

  if (!keyframes_.empty()) {
    std::lock_guard<std::mutex> lock(latest_map_to_odom_mutex_);
    const KeyFrame & latest_kf = *keyframes_.back();
    // Correction = T_map * T_odom^-1
    latest_map_to_odom_ = latest_kf.pose.compose(latest_kf.odom_pose.inverse());
  }
}

PointCloudXYZ SlamSystem::getMapCloud() const
{
  PointCloudXYZ map_cloud;

  for (const std::shared_ptr<KeyFrame> & keyframe : keyframes_) {
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
    pcl::transformPointCloud(keyframe->scan, transformed_scan, transform);
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
  out.reserve(keyframes_.size());

  for (const std::shared_ptr<KeyFrame> & keyframe : keyframes_) {
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
    pcl::transformPointCloud(keyframe->scan, transformed_scan, transform);
    out.emplace_back(keyframe->pose, std::move(transformed_scan));
  }

  return out;
}

std::vector<std::shared_ptr<const KeyFrame>> SlamSystem::getKeyFrames() const
{
  std::lock_guard lock(keyframes_mutex_);
  std::vector<std::shared_ptr<const KeyFrame>> keyframes_copy(keyframes_.begin(), keyframes_.end());
  return keyframes_copy;
}

gtsam::Pose3 SlamSystem::projectPlanar(const gtsam::Pose3 & pose)
{
  const gtsam::Vector3 rpy = pose.rotation().rpy();
  const double yaw = rpy.z();
  const auto & translation = pose.translation();
  return Utils::makePlanarPose(translation.x(), translation.y(), yaw);
}

double SlamSystem::translationDistance(const gtsam::Pose3 & lhs, const gtsam::Pose3 & rhs)
{
  const auto delta = lhs.translation() - rhs.translation();
  return std::hypot(delta.x(), delta.y());
}

double SlamSystem::yawDistance(const gtsam::Pose3 & lhs, const gtsam::Pose3 & rhs)
{
  const double lhs_yaw = lhs.rotation().rpy().z();
  const double rhs_yaw = rhs.rotation().rpy().z();
  return std::abs(Utils::normalizeAngle(lhs_yaw - rhs_yaw));
}

void SlamSystem::initializeIfNeeded(
  double timestamp, const PointCloudXYZ & scan, const gtsam::Pose3 & odom_pose)
{
  if (!keyframes_.empty()) {
    return;
  }

  const gtsam::Pose3 initial_pose = gtsam::Pose3();
  graph_optimizer_.initialize(initial_pose, static_cast<uint64_t>(timestamp * 1e6));
  keyframes_.emplace_back(std::make_shared<KeyFrame>(0, timestamp, initial_pose, odom_pose, scan));
  next_keyframe_key_ = 1;

  SAM_INFO("System initialized with first keyframe at timestamp: {}", timestamp);
}

bool SlamSystem::shouldCreateKeyFrame(const gtsam::Pose3 & current_odom_pose) const
{
  if (keyframes_.empty()) {
    return true;
  }

  std::shared_ptr<KeyFrame> reference_keyframe = keyframes_.back();

  double trans_dist = translationDistance(reference_keyframe->odom_pose, current_odom_pose);
  bool should_create_trans = trans_dist > parameters_->minimum_travel_distance;

  double yaw_dist = yawDistance(reference_keyframe->odom_pose, current_odom_pose);
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
