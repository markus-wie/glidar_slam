#include "glidar_slam/ros2/ros2_slam_wrapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cv_bridge/cv_bridge.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "glidar_slam/ros2/utils.hpp"
#include "opencv2/imgproc.hpp"
#include "pcl/common/transforms.h"
#include "pcl/point_types.h"
#include "tf2/LinearMath/Quaternion.h"
#include "visualization_msgs/msg/marker.hpp"

namespace glidar_slam::ros2 {

using glidar_slam::core::KeyFrame;
using glidar_slam::core::OccupancyGrid;
using glidar_slam::core::Parameters;
using glidar_slam::core::PointCloudXYZ;
using glidar_slam::core::SlamSystem;

namespace {

sensor_msgs::msg::Image toHeatmapRosImage(
  const glidar_slam::core::CsmResult::DebugImage & debug, const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  sensor_msgs::msg::Image image;
  image.header.stamp = stamp;
  image.header.frame_id = frame_id;

  if (debug.width <= 0 || debug.height <= 0 || debug.pixels.empty()) {
    image.height = 0;
    image.width = 0;
    image.encoding = "rgb8";
    image.is_bigendian = false;
    image.step = 0;
    return image;
  }

  cv::Mat gray(debug.height, debug.width, CV_8UC1);
  std::memcpy(gray.data, debug.pixels.data(), debug.pixels.size());

  cv::Mat bgr;
  cv::applyColorMap(gray, bgr, cv::COLORMAP_TURBO);

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  cv_bridge::CvImage cv_image;
  cv_image.header = image.header;
  cv_image.encoding = "rgb8";
  cv_image.image = rgb;
  return *cv_image.toImageMsg();
}

PointCloudXYZ transformPointCloud(
  const PointCloudXYZ & input, const geometry_msgs::msg::TransformStamped & transform_stamped)
{
  const auto & t = transform_stamped.transform.translation;
  const auto & q = transform_stamped.transform.rotation;

  const Eigen::Quaternionf rotation(q.w, q.x, q.y, q.z);
  const Eigen::Translation3f translation(
    static_cast<float>(t.x), static_cast<float>(t.y), static_cast<float>(t.z));
  const Eigen::Affine3f transform = translation * rotation;

  PointCloudXYZ output;
  pcl::transformPointCloud(input, output, transform);
  return output;
}

}  // namespace

Ros2SlamWrapper::Ros2SlamWrapper(const rclcpp::NodeOptions & options) : Node("glidar_slam", options)
{
  parameters_ = std::make_shared<Parameters>();
  slam_system_ = std::make_unique<SlamSystem>(parameters_);
  occ_grid_ = std::make_unique<OccupancyGrid>(parameters_);
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*(tf_buffer_));
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  // Parameters
  parameters_->odom_frame = this->declare_parameter<std::string>("odom_frame", "odom");
  parameters_->map_frame = this->declare_parameter<std::string>("map_frame", "map");
  parameters_->scan_frame = this->declare_parameter<std::string>("scan_frame", "laser");
  parameters_->base_frame = this->declare_parameter<std::string>("base_frame", "base_link");

  parameters_->scan_topic = this->declare_parameter<std::string>("scan_topic", "/scan");
  parameters_->tf_topic = this->declare_parameter<std::string>("tf_topic", "/tf");
  parameters_->odom_topic = this->declare_parameter<std::string>("odom_topic", "/odom");
  parameters_->csm_debug_low_topic = this->declare_parameter<std::string>(
    "csm_debug_low_topic", "glidar_slam/debug/csm_likelihood_low");
  parameters_->csm_debug_high_topic = this->declare_parameter<std::string>(
    "csm_debug_high_topic", "glidar_slam/debug/csm_likelihood_high");

  parameters_->odom_covariance_diagonal = this->declare_parameter<std::vector<double>>(
    "odom_covariance_diagonal", std::vector<double>{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0});

  parameters_->minimum_travel_distance =
    this->declare_parameter<double>("minimum_travel_distance", 0.5);
  parameters_->minimum_travel_heading =
    this->declare_parameter<double>("minimum_travel_heading", 0.5);
  parameters_->enable_csm_debug_images =
    this->declare_parameter<bool>("enable_csm_debug_images", false);

  parameters_->occ_map_resolution = this->declare_parameter<double>("occ_map_resolution", 0.05);
  parameters_->occ_map_padding = this->declare_parameter<int>("occ_map_padding", 2);

  parameters_->submap_window_size = this->declare_parameter<int>("submap_window_size", 5);

  processCSMParameters();

  // Subscribers
  odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
    parameters_->odom_topic, rclcpp::SensorDataQoS(),
    std::bind(&Ros2SlamWrapper::odomCallback, this, std::placeholders::_1));

  scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    parameters_->scan_topic, rclcpp::SensorDataQoS(),
    std::bind(&Ros2SlamWrapper::scanCallback, this, std::placeholders::_1));

  // Publishers
  marker_array_publisher_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("graph_visualization", 10);
  occupancy_grid_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map", 10);
  csm_debug_low_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>(parameters_->csm_debug_low_topic, 10);
  csm_debug_high_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>(parameters_->csm_debug_high_topic, 10);

  // Timer
  transform_broadcast_callback_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  transform_broadcast_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100), std::bind(&Ros2SlamWrapper::publishMapToOdom, this),
    transform_broadcast_callback_group_);
}

void Ros2SlamWrapper::processCSMParameters()
{
  const auto stage_resolutions =
    this->declare_parameter<std::vector<double>>("csm_stage_resolutions", std::vector<double>{});
  const auto stage_translation_steps = this->declare_parameter<std::vector<double>>(
    "csm_stage_translation_steps", std::vector<double>{});
  const auto stage_angular_steps =
    this->declare_parameter<std::vector<double>>("csm_stage_angular_steps", std::vector<double>{});
  const auto stage_windows_x =
    this->declare_parameter<std::vector<double>>("csm_stage_window_x", std::vector<double>{});
  const auto stage_windows_y =
    this->declare_parameter<std::vector<double>>("csm_stage_window_y", std::vector<double>{});
  const auto stage_windows_yaw =
    this->declare_parameter<std::vector<double>>("csm_stage_window_yaw", std::vector<double>{});

  const bool has_stage_configuration = !stage_resolutions.empty() ||
                                       !stage_translation_steps.empty() ||
                                       !stage_angular_steps.empty() || !stage_windows_x.empty() ||
                                       !stage_windows_y.empty() || !stage_windows_yaw.empty();
  if (has_stage_configuration) {
    const std::size_t stage_count = stage_resolutions.size();
    if (
      stage_count == 0 || stage_translation_steps.size() != stage_count ||
      stage_angular_steps.size() != stage_count || stage_windows_x.size() != stage_count ||
      stage_windows_y.size() != stage_count || stage_windows_yaw.size() != stage_count) {
      throw std::runtime_error(
        "All CSM stage parameter arrays must be non-empty and equal in length");
    }

    constexpr auto valid_positive = [](double value) {
      return std::isfinite(value) && value > 0.0;
    };
    constexpr auto valid_window = [](double value) {
      return std::isfinite(value) && value >= 0.0;
    };
    for (std::size_t index = 0; index < stage_count; ++index) {
      if (
        !valid_positive(stage_resolutions[index]) ||
        !valid_positive(stage_translation_steps[index]) ||
        !valid_positive(stage_angular_steps[index]) || !valid_window(stage_windows_x[index]) ||
        !valid_window(stage_windows_y[index]) || !valid_window(stage_windows_yaw[index])) {
        throw std::runtime_error(
          "CSM stage values must be finite; steps and resolutions positive; windows non-negative");
      }
      glidar_slam::core::CsmSearchStage stage{};
      stage.field_resolution = stage_resolutions[index];
      stage.translation_step = stage_translation_steps[index];
      stage.angular_step = stage_angular_steps[index];
      stage.window_x = stage_windows_x[index];
      stage.window_y = stage_windows_y[index];
      stage.window_yaw = stage_windows_yaw[index];
      parameters_->csm_search_stages.push_back(stage);
    }
  }

  parameters_->csm_smear_deviation = this->declare_parameter<double>("csm_smear_deviation", 0.1);
  parameters_->csm_use_penalty = this->declare_parameter<bool>("csm_use_penalty", true);
  parameters_->csm_distance_variance_penalty =
    this->declare_parameter<double>("csm_distance_variance_penalty", 0.5);
  parameters_->csm_angle_variance_penalty =
    this->declare_parameter<double>("csm_angle_variance_penalty", 1.0);
  parameters_->csm_minimum_distance_penalty =
    this->declare_parameter<double>("csm_minimum_distance_penalty", 0.5);
  parameters_->csm_minimum_angle_penalty =
    this->declare_parameter<double>("csm_minimum_angle_penalty", 0.9);

  if (
    !std::isfinite(parameters_->csm_smear_deviation) || parameters_->csm_smear_deviation <= 0.0 ||
    !std::isfinite(parameters_->csm_distance_variance_penalty) ||
    parameters_->csm_distance_variance_penalty <= 0.0 ||
    !std::isfinite(parameters_->csm_angle_variance_penalty) ||
    parameters_->csm_angle_variance_penalty <= 0.0 ||
    !std::isfinite(parameters_->csm_minimum_distance_penalty) ||
    parameters_->csm_minimum_distance_penalty < 0.0 ||
    parameters_->csm_minimum_distance_penalty > 1.0 ||
    !std::isfinite(parameters_->csm_minimum_angle_penalty) ||
    parameters_->csm_minimum_angle_penalty < 0.0 || parameters_->csm_minimum_angle_penalty > 1.0) {
    throw std::runtime_error("CSM penalty parameters are outside their valid ranges");
  }
}

void Ros2SlamWrapper::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr & msg)
{
  if (
    std::find(
      parameters_->odom_covariance_diagonal.begin(), parameters_->odom_covariance_diagonal.end(),
      -1.0) == parameters_->odom_covariance_diagonal.end()) {
    // gtsam covariance in tangent space is [roll, pitch, yaw, x, y, z] (see gtsam::Pose3)
    latest_odom_covariance_(0, 0) = msg->pose.covariance[21];
    latest_odom_covariance_(1, 1) = msg->pose.covariance[28];
    latest_odom_covariance_(2, 2) = msg->pose.covariance[35];
    latest_odom_covariance_(3, 3) = msg->pose.covariance[0];
    latest_odom_covariance_(4, 4) = msg->pose.covariance[7];
    latest_odom_covariance_(5, 5) = msg->pose.covariance[14];
  } else {
    // the ros2 parameter keeps the ros2 convention which is [x, y, z, roll, pitch, yaw]
    latest_odom_covariance_(0, 0) = parameters_->odom_covariance_diagonal[5];
    latest_odom_covariance_(1, 1) = parameters_->odom_covariance_diagonal[4];
    latest_odom_covariance_(2, 2) = parameters_->odom_covariance_diagonal[3];
    latest_odom_covariance_(3, 3) = parameters_->odom_covariance_diagonal[2];
    latest_odom_covariance_(4, 4) = parameters_->odom_covariance_diagonal[1];
    latest_odom_covariance_(5, 5) = parameters_->odom_covariance_diagonal[0];
  }
}

void Ros2SlamWrapper::scanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr & msg)
{
  const PointCloudXYZ scan_cloud = laserScanToPointCloud(*msg);

  // Get the latest odometry pose from tf.
  gtsam::Pose3 odom_pose;
  try {
    geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
      parameters_->odom_frame, parameters_->base_frame, msg->header.stamp);
    odom_pose = gtsam::Pose3(
      gtsam::Rot3::Quaternion(
        transform_stamped.transform.rotation.w, transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z),
      gtsam::Point3(
        transform_stamped.transform.translation.x, transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(this->get_logger(), "Could not get odometry pose from tf: %s", ex.what());
    return;
  }

  // Move scan points into the same body frame as odometry.
  PointCloudXYZ cloud = scan_cloud;
  try {
    const std::string scan_frame =
      msg->header.frame_id.empty() ? parameters_->scan_frame : msg->header.frame_id;
    const geometry_msgs::msg::TransformStamped base_from_scan =
      tf_buffer_->lookupTransform(parameters_->base_frame, scan_frame, msg->header.stamp);
    cloud = transformPointCloud(scan_cloud, base_from_scan);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(
      this->get_logger(), "Could not transform scan frame '%s' to base frame '%s': %s",
      msg->header.frame_id.c_str(), parameters_->base_frame.c_str(), ex.what());
    return;
  }

  const SlamSystem::LaserScanOutput slam_output = slam_system_->handleLaserScan(
    rclcpp::Time(msg->header.stamp).seconds(), cloud, odom_pose, latest_odom_covariance_);

  if (!slam_output.optimized_pose.has_value()) {
    RCLCPP_WARN(
      this->get_logger(),
      "SLAM system did not return an optimized pose. This should never happen.");
    return;
  }

  latest_map_to_base_ = slam_output.optimized_pose.value();

  const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> scans_transformed =
    slam_system_->getTransformedKeyFrameScans();

  std::vector<std::shared_ptr<const KeyFrame>> keyframes = slam_system_->getKeyFrames();

  publishGraph(keyframes);
  publishOccupancyGrid(scans_transformed, rclcpp::Time(msg->header.stamp));
  publishDebugImage(slam_output, rclcpp::Time(msg->header.stamp));
}

PointCloudXYZ Ros2SlamWrapper::laserScanToPointCloud(const sensor_msgs::msg::LaserScan & msg)
{
  PointCloudXYZ cloud;
  cloud.reserve(msg.ranges.size());

  double angle = msg.angle_min;
  for (const auto range : msg.ranges) {
    if (std::isfinite(range) && range >= msg.range_min && range <= msg.range_max) {
      pcl::PointXYZ point;
      point.x = static_cast<float>(range * std::cos(angle));
      point.y = static_cast<float>(range * std::sin(angle));
      point.z = 0.0f;
      cloud.push_back(point);
    }
    angle += msg.angle_increment;
  }

  cloud.width = static_cast<std::uint32_t>(cloud.size());
  cloud.height = 1;
  cloud.is_dense = true;
  return cloud;
}

geometry_msgs::msg::TransformStamped Ros2SlamWrapper::poseToTransformStamped(
  const gtsam::Pose3 & map_to_odom, const std::string & parent_frame,
  const std::string & child_frame, const rclcpp::Time & stamp)
{
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = parent_frame;
  transform.child_frame_id = child_frame;

  const auto & translation = map_to_odom.translation();
  const gtsam::Vector3 rpy = map_to_odom.rotation().rpy();

  transform.transform.translation.x = translation.x();
  transform.transform.translation.y = translation.y();
  transform.transform.translation.z = translation.z();

  tf2::Quaternion quaternion;
  quaternion.setRPY(rpy.x(), rpy.y(), rpy.z());
  transform.transform.rotation.x = quaternion.x();
  transform.transform.rotation.y = quaternion.y();
  transform.transform.rotation.z = quaternion.z();
  transform.transform.rotation.w = quaternion.w();
  return transform;
}

geometry_msgs::msg::PoseStamped Ros2SlamWrapper::poseToPoseStamped(
  const gtsam::Pose3 & pose, const std::string & frame, const rclcpp::Time & stamp)
{
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header.stamp = stamp;
  pose_msg.header.frame_id = frame;

  const auto & translation = pose.translation();
  const gtsam::Vector3 rpy = pose.rotation().rpy();

  pose_msg.pose.position.x = translation.x();
  pose_msg.pose.position.y = translation.y();
  pose_msg.pose.position.z = translation.z();

  tf2::Quaternion quaternion;
  quaternion.setRPY(rpy.x(), rpy.y(), rpy.z());
  pose_msg.pose.orientation.x = quaternion.x();
  pose_msg.pose.orientation.y = quaternion.y();
  pose_msg.pose.orientation.z = quaternion.z();
  pose_msg.pose.orientation.w = quaternion.w();
  return pose_msg;
}

namespace {

visualization_msgs::msg::Marker keyframeToMarker(
  const KeyFrame & keyframe, const std::string & frame)
{
  visualization_msgs::msg::Marker marker;
  marker.header.stamp = rclcpp::Time(keyframe.timestamp);
  marker.header.frame_id = frame;
  marker.ns = "slam_graph";
  marker.id = keyframe.key;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;

  const auto & translation = keyframe.pose.translation();
  const gtsam::Vector3 rpy = keyframe.pose.rotation().rpy();

  marker.pose.position.x = translation.x();
  marker.pose.position.y = translation.y();
  marker.pose.position.z = translation.z();

  tf2::Quaternion quaternion;
  quaternion.setRPY(rpy.x(), rpy.y(), rpy.z());
  marker.pose.orientation.x = quaternion.x();
  marker.pose.orientation.y = quaternion.y();
  marker.pose.orientation.z = quaternion.z();
  marker.pose.orientation.w = quaternion.w();

  marker.scale.x = 0.1;
  marker.scale.y = 0.1;
  marker.scale.z = 0.1;

  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0f;

  return marker;
}

}  // namespace

void Ros2SlamWrapper::publishGraph(const std::vector<std::shared_ptr<const KeyFrame>> & keyframes)
{
  visualization_msgs::msg::MarkerArray marker_array_msg;

  for (const auto & keyframe : keyframes) {
    marker_array_msg.markers.push_back(keyframeToMarker(*keyframe, parameters_->map_frame));
  }

  marker_array_publisher_->publish(marker_array_msg);
}

void Ros2SlamWrapper::publishOccupancyGrid(
  const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> & scans_transformed,
  const rclcpp::Time & stamp)
{
  occ_grid_->buildFromScans(scans_transformed);

  nav_msgs::msg::OccupancyGrid occupancy_grid_msg =
    Utils::toRosMessage(*occ_grid_, parameters_->map_frame, stamp);

  occupancy_grid_publisher_->publish(occupancy_grid_msg);
}

void Ros2SlamWrapper::publishMapToOdom()
{
  const gtsam::Pose3 map_to_odom = slam_system_->getMapToOdom();
  tf_broadcaster_->sendTransform(poseToTransformStamped(
    map_to_odom, parameters_->map_frame, parameters_->odom_frame, this->now()));
}

void Ros2SlamWrapper::publishDebugImage(
  const SlamSystem::LaserScanOutput & output, const rclcpp::Time & stamp)
{
  if (!parameters_->enable_csm_debug_images) {
    return;
  }

  if (output.low_res_debug.has_value() && !output.low_res_debug->pixels.empty()) {
    csm_debug_low_publisher_->publish(
      toHeatmapRosImage(output.low_res_debug.value(), parameters_->map_frame, stamp));
  }

  if (output.high_res_debug.has_value() && !output.high_res_debug->pixels.empty()) {
    csm_debug_high_publisher_->publish(
      toHeatmapRosImage(output.high_res_debug.value(), parameters_->map_frame, stamp));
  }
}

}  // namespace glidar_slam::ros2
