#include "glidar_slam/ros2/ros2_slam_wrapper.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Eigen/Geometry"
#include "cv_bridge/cv_bridge.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "glidar_slam/core/camera_model.hpp"
#include "glidar_slam/core/sensor_data.hpp"
#include "glidar_slam/ros2/utils.hpp"
#include "opencv2/imgproc.hpp"
#include "pcl/common/transforms.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "sensor_msgs/image_encodings.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "visualization_msgs/msg/marker.hpp"

namespace glidar_slam::ros2 {

using glidar_slam::core::KeyFrame;
using glidar_slam::core::LaserScan;
using glidar_slam::core::OccupancyGrid;
using glidar_slam::core::Parameters;
using glidar_slam::core::PointCloudXYZ;
using glidar_slam::core::SlamSystem;

Ros2SlamWrapper::Ros2SlamWrapper(const rclcpp::NodeOptions & options) : Node("glidar_slam", options)
{
  parameters_ = std::make_shared<Parameters>();
  slam_system_ = std::make_unique<SlamSystem>(parameters_);
  occ_grid_ = std::make_unique<OccupancyGrid>(parameters_);
  ground_marking_grid_ = std::make_unique<glidar_slam::core::GroundMarkingGrid>(parameters_);
  latest_odom_covariance_.setIdentity();
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*(tf_buffer_));
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  // Parameters
  parameters_->odom_frame = this->declare_parameter<std::string>("odom_frame", "odom");
  parameters_->map_frame = this->declare_parameter<std::string>("map_frame", "map");
  parameters_->scan_frame = this->declare_parameter<std::string>("scan_frame", "laser");
  parameters_->camera_frame = this->declare_parameter<std::string>("camera_frame", "camera");
  parameters_->base_frame = this->declare_parameter<std::string>("base_frame", "base_link");

  parameters_->scan_topic = this->declare_parameter<std::string>("scan_topic", "/scan");
  parameters_->tf_topic = this->declare_parameter<std::string>("tf_topic", "/tf");
  parameters_->odom_topic = this->declare_parameter<std::string>("odom_topic", "/odom");
  parameters_->csm_debug_low_topic = this->declare_parameter<std::string>(
    "csm_debug_low_topic", "glidar_slam/debug/csm_likelihood_low");
  parameters_->csm_debug_high_topic = this->declare_parameter<std::string>(
    "csm_debug_high_topic", "glidar_slam/debug/csm_likelihood_high");
  parameters_->color_image_topic =
    this->declare_parameter<std::string>("color_image_topic", "/camera/color/image_raw");
  parameters_->aligned_depth_image_topic = this->declare_parameter<std::string>(
    "aligned_depth_image_topic", "/camera/aligned_depth/image_raw");
  parameters_->color_camera_info_topic =
    this->declare_parameter<std::string>("color_camera_info_topic", "/camera/color/camera_info");

  parameters_->odom_covariance_diagonal = this->declare_parameter<std::vector<double>>(
    "odom_covariance_diagonal", std::vector<double>{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0});

  parameters_->minimum_travel_distance =
    this->declare_parameter<double>("minimum_travel_distance", 0.5);
  parameters_->minimum_travel_heading =
    this->declare_parameter<double>("minimum_travel_heading", 0.5);

  parameters_->occ_map_resolution = this->declare_parameter<double>("occ_map_resolution", 0.05);
  parameters_->occ_map_padding = this->declare_parameter<int>("occ_map_padding", 2);

  parameters_->submap_window_size = this->declare_parameter<int>("submap_window_size", 5);
  parameters_->ground_debug_enable = this->declare_parameter<bool>("ground_debug_enable", false);
  parameters_->ground_debug_cloud_topic = this->declare_parameter<std::string>(
    "ground_debug_cloud_topic", "glidar_slam/debug/ground_inliers");
  parameters_->ground_debug_marker_topic = this->declare_parameter<std::string>(
    "ground_debug_marker_topic", "glidar_slam/debug/ground_plane");
  parameters_->ground_debug_image_topic = this->declare_parameter<std::string>(
    "ground_debug_image_topic", "glidar_slam/debug/ground_overlay");
  parameters_->ground_debug_plane_size =
    this->declare_parameter<double>("ground_debug_plane_size", 4.0);
  parameters_->ground_optimization_enable =
    this->declare_parameter<bool>("ground_optimization_enable", true);
  parameters_->ground_observation_max_age_sec =
    this->declare_parameter<double>("ground_observation_max_age_sec", 0.1);
  parameters_->ground_minimum_inlier_count =
    this->declare_parameter<int>("ground_minimum_inlier_count", 100);
  parameters_->ground_normal_sigma = this->declare_parameter<double>("ground_normal_sigma", 0.05);
  parameters_->ground_distance_sigma =
    this->declare_parameter<double>("ground_distance_sigma", 0.03);
  parameters_->ground_fallback_variance =
    this->declare_parameter<double>("ground_fallback_variance", 1.0);
  const std::string ground_roi_ratios =
    this->declare_parameter<std::string>("ground_roi_ratios", "0.0 0.0 0.2 0.4");
  if (!parseGroundRoiRatios(ground_roi_ratios, parameters_->ground_roi_ratios)) {
    throw std::runtime_error(
      "ground_roi_ratios must be four values [left right top bottom] in [0,1] with opposing "
      "sides summing to less than one");
  }
  parameters_->ground_extraction_pixel_stride =
    this->declare_parameter<int>("ground_extraction_pixel_stride", 2);
  parameters_->ground_extraction_voxel_size =
    this->declare_parameter<double>("ground_extraction_voxel_size", 0.05);
  parameters_->ground_extraction_max_distance =
    this->declare_parameter<double>("ground_extraction_max_distance", 4.0);
  parameters_->ground_extraction_distance_threshold =
    this->declare_parameter<double>("ground_extraction_distance_threshold", 0.05);
  parameters_->ground_marking_map_enable =
    this->declare_parameter<bool>("ground_marking_map_enable", true);
  parameters_->ground_marking_map_topic =
    this->declare_parameter<std::string>("ground_marking_map_topic", "ground_markings_map");
  parameters_->ground_marking_map_resolution =
    this->declare_parameter<double>("ground_marking_map_resolution", 0.02);
  parameters_->ground_marking_map_padding =
    this->declare_parameter<int>("ground_marking_map_padding", 2);
  parameters_->ground_marking_white_threshold =
    this->declare_parameter<int>("ground_marking_white_threshold", 200);

  parameters_->loop_debug_enable = this->declare_parameter<bool>("loop_debug_enable", false);
  parameters_->loop_input_queue_capacity =
    static_cast<std::size_t>(this->declare_parameter<int>("loop_input_queue_capacity", 2));
  parameters_->loop_output_queue_capacity =
    static_cast<std::size_t>(this->declare_parameter<int>("loop_output_queue_capacity", 8));
  parameters_->loop_minimum_key_separation =
    static_cast<uint64_t>(this->declare_parameter<int>("loop_minimum_key_separation", 20));
  parameters_->loop_maximum_distance =
    this->declare_parameter<double>("loop_maximum_distance", 2.0);
  parameters_->loop_maximum_yaw_difference =
    this->declare_parameter<double>("loop_maximum_yaw_difference", 1.0);
  parameters_->loop_mahalanobis_threshold =
    this->declare_parameter<double>("loop_mahalanobis_threshold", 3.0);
  parameters_->loop_minimum_xy_variance =
    this->declare_parameter<double>("loop_minimum_xy_variance", 0.01);
  parameters_->loop_minimum_score = this->declare_parameter<double>("loop_minimum_score", 0.5);
  parameters_->loop_maximum_consistency_error =
    this->declare_parameter<double>("loop_maximum_consistency_error", 0.5);
  parameters_->debug_timings = this->declare_parameter<bool>("debug_timings", false);

  processCSMParameters();

  // Callback groups
  odom_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  camera_callback_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  transform_broadcast_callback_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Parameter callback
  parameter_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&Ros2SlamWrapper::onParametersChanged, this, std::placeholders::_1));

  // Subscription options
  rclcpp::SubscriptionOptions odom_subscription_options;
  odom_subscription_options.callback_group = odom_callback_group_;
  rclcpp::SubscriptionOptions camera_subscription_options;
  camera_subscription_options.callback_group = camera_callback_group_;

  // Subscribers
  odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
    parameters_->odom_topic, rclcpp::SensorDataQoS(),
    std::bind(&Ros2SlamWrapper::odomCallback, this, std::placeholders::_1),
    odom_subscription_options);

  scan_subscriber_.subscribe(this, parameters_->scan_topic, 5);
  color_image_subscriber_.subscribe(this, parameters_->color_image_topic, 5);
  aligned_depth_image_subscriber_.subscribe(this, parameters_->aligned_depth_image_topic, 5);
  color_camera_info_subscriber_.subscribe(this, parameters_->color_camera_info_topic, 5);

  ground_synchronizer_ = std::make_shared<message_filters::Synchronizer<GroundSyncPolicy>>(
    GroundSyncPolicy(50), scan_subscriber_, color_image_subscriber_,
    aligned_depth_image_subscriber_, color_camera_info_subscriber_);
  ground_synchronizer_->registerCallback(std::bind(
    &Ros2SlamWrapper::ScanRGBDCallback, this, std::placeholders::_1, std::placeholders::_2,
    std::placeholders::_3, std::placeholders::_4));
  rclcpp::Duration max_delay = rclcpp::Duration::from_seconds(0.05);
  ground_synchronizer_->setMaxIntervalDuration(max_delay);

  // Publishers
  marker_array_publisher_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("graph_visualization", 10);
  occupancy_grid_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map", 10);
  ground_marking_grid_publisher_ =
    this->create_publisher<nav_msgs::msg::OccupancyGrid>(parameters_->ground_marking_map_topic, 10);
  ground_debug_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    parameters_->ground_debug_cloud_topic, 10);
  ground_debug_marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    parameters_->ground_debug_marker_topic, 10);
  ground_debug_image_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>(parameters_->ground_debug_image_topic, 10);
  csm_debug_low_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>(parameters_->csm_debug_low_topic, 10);
  csm_debug_high_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>(parameters_->csm_debug_high_topic, 10);

  // Timer
  transform_broadcast_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100), std::bind(&Ros2SlamWrapper::publishMapToOdom, this),
    transform_broadcast_callback_group_);
}

void Ros2SlamWrapper::processCSMParameters()
{
  parameters_->csm_debug_enable = this->declare_parameter<bool>("csm_debug_enable", false);

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
  parameters_->csm_use_distance_transform =
    this->declare_parameter<bool>("csm_use_distance_transform", false);
  parameters_->csm_use_tbb = this->declare_parameter<bool>("csm_use_tbb", false);
  parameters_->csm_use_penalty = this->declare_parameter<bool>("csm_use_penalty", true);
  parameters_->csm_distance_penalty_std_dev =
    this->declare_parameter<double>("csm_distance_penalty_std_dev", 0.5);
  parameters_->csm_angle_penalty_std_dev =
    this->declare_parameter<double>("csm_angle_penalty_std_dev", 1.0);

  if (
    !std::isfinite(parameters_->csm_smear_deviation) || parameters_->csm_smear_deviation <= 0.0 ||
    !std::isfinite(parameters_->csm_distance_penalty_std_dev) ||
    parameters_->csm_distance_penalty_std_dev <= 0.0 ||
    !std::isfinite(parameters_->csm_angle_penalty_std_dev) ||
    parameters_->csm_angle_penalty_std_dev <= 0.0) {
    throw std::runtime_error("CSM penalty parameters are outside their valid ranges");
  }
}

rcl_interfaces::msg::SetParametersResult Ros2SlamWrapper::onParametersChanged(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  Parameters updated = *parameters_;
  for (const auto & parameter : parameters) {
    const std::string & name = parameter.get_name();
    if (name == "odom_covariance_diagonal") {
      updated.odom_covariance_diagonal = parameter.as_double_array();
      if (updated.odom_covariance_diagonal.size() != 6) {
        result.successful = false;
        result.reason = "odom_covariance_diagonal must contain six values";
        return result;
      }
    } else if (name == "minimum_travel_distance") {
      updated.minimum_travel_distance = parameter.as_double();
    } else if (name == "minimum_travel_heading") {
      updated.minimum_travel_heading = parameter.as_double();
    } else if (name == "occ_map_resolution") {
      updated.occ_map_resolution = parameter.as_double();
    } else if (name == "occ_map_padding") {
      updated.occ_map_padding = static_cast<int>(parameter.as_int());
    } else if (name == "submap_window_size") {
      updated.submap_window_size = static_cast<int>(parameter.as_int());
    } else if (name == "ground_debug_enable") {
      updated.ground_debug_enable = parameter.as_bool();
    } else if (name == "ground_debug_plane_size") {
      updated.ground_debug_plane_size = parameter.as_double();
    } else if (name == "ground_optimization_enable") {
      updated.ground_optimization_enable = parameter.as_bool();
    } else if (name == "ground_observation_max_age_sec") {
      updated.ground_observation_max_age_sec = parameter.as_double();
    } else if (name == "ground_minimum_inlier_count") {
      updated.ground_minimum_inlier_count = static_cast<int>(parameter.as_int());
    } else if (name == "ground_normal_sigma") {
      updated.ground_normal_sigma = parameter.as_double();
    } else if (name == "ground_distance_sigma") {
      updated.ground_distance_sigma = parameter.as_double();
    } else if (name == "ground_fallback_variance") {
      updated.ground_fallback_variance = parameter.as_double();
    } else if (name == "ground_roi_ratios") {
      if (!parseGroundRoiRatios(parameter.as_string(), updated.ground_roi_ratios)) {
        result.successful = false;
        result.reason = "ground_roi_ratios has an invalid value";
        return result;
      }
    } else if (name == "ground_extraction_pixel_stride") {
      updated.ground_extraction_pixel_stride = static_cast<int>(parameter.as_int());
    } else if (name == "ground_extraction_voxel_size") {
      updated.ground_extraction_voxel_size = parameter.as_double();
    } else if (name == "ground_extraction_max_distance") {
      updated.ground_extraction_max_distance = parameter.as_double();
    } else if (name == "ground_extraction_distance_threshold") {
      updated.ground_extraction_distance_threshold = parameter.as_double();
    } else if (name == "ground_marking_map_enable") {
      updated.ground_marking_map_enable = parameter.as_bool();
    } else if (name == "ground_marking_map_resolution") {
      updated.ground_marking_map_resolution = parameter.as_double();
    } else if (name == "ground_marking_map_padding") {
      updated.ground_marking_map_padding = static_cast<int>(parameter.as_int());
    } else if (name == "ground_marking_white_threshold") {
      updated.ground_marking_white_threshold = static_cast<int>(parameter.as_int());
    } else if (name == "loop_debug_enable") {
      updated.loop_debug_enable = parameter.as_bool();
    } else if (name == "loop_minimum_key_separation") {
      updated.loop_minimum_key_separation = static_cast<uint64_t>(parameter.as_int());
    } else if (name == "loop_maximum_distance") {
      updated.loop_maximum_distance = parameter.as_double();
    } else if (name == "loop_maximum_yaw_difference") {
      updated.loop_maximum_yaw_difference = parameter.as_double();
    } else if (name == "loop_mahalanobis_threshold") {
      updated.loop_mahalanobis_threshold = parameter.as_double();
    } else if (name == "loop_minimum_xy_variance") {
      updated.loop_minimum_xy_variance = parameter.as_double();
    } else if (name == "loop_minimum_score") {
      updated.loop_minimum_score = parameter.as_double();
    } else if (name == "loop_maximum_consistency_error") {
      updated.loop_maximum_consistency_error = parameter.as_double();
    } else if (name == "debug_timings") {
      updated.debug_timings = parameter.as_bool();
    } else if (name == "csm_debug_enable") {
      updated.csm_debug_enable = parameter.as_bool();
    } else if (name == "csm_smear_deviation") {
      updated.csm_smear_deviation = parameter.as_double();
    } else if (name == "csm_use_distance_transform") {
      updated.csm_use_distance_transform = parameter.as_bool();
    } else if (name == "csm_use_tbb") {
      updated.csm_use_tbb = parameter.as_bool();
    } else if (name == "csm_use_penalty") {
      updated.csm_use_penalty = parameter.as_bool();
    } else if (name == "csm_distance_penalty_std_dev") {
      updated.csm_distance_penalty_std_dev = parameter.as_double();
    } else if (name == "csm_angle_penalty_std_dev") {
      updated.csm_angle_penalty_std_dev = parameter.as_double();
    }
  }

  *parameters_ = std::move(updated);

  return result;
}

void Ros2SlamWrapper::ScanRGBDCallback(
  const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & color_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  cv_bridge::CvImageConstPtr cv_ptr_color;
  cv_bridge::CvImageConstPtr cv_ptr_depth;
  try {
    cv_ptr_color = cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8);
    cv_ptr_depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding);
  } catch (const cv_bridge::Exception & exception) {
    RCLCPP_WARN(this->get_logger(), "Cannot decode synchronized RGB-D input: %s", exception.what());
  }

  // Get the latest odometry pose from tf.
  gtsam::Pose3 odom_pose;
  try {
    geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
      parameters_->odom_frame, parameters_->base_frame, scan_msg->header.stamp,
      rclcpp::Duration::from_seconds(0.05));
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

  Eigen::Affine3f base_from_scan_eigen = Eigen::Affine3f::Identity();
  try {
    const std::string scan_frame =
      scan_msg->header.frame_id.empty() ? parameters_->scan_frame : scan_msg->header.frame_id;
    const geometry_msgs::msg::TransformStamped base_from_scan =
      tf_buffer_->lookupTransform(parameters_->base_frame, scan_frame, scan_msg->header.stamp);
    const auto & translation = base_from_scan.transform.translation;
    const auto & rotation = base_from_scan.transform.rotation;
    base_from_scan_eigen.translate(Eigen::Vector3f(
      static_cast<float>(translation.x), static_cast<float>(translation.y),
      static_cast<float>(translation.z)));
    base_from_scan_eigen.rotate(Eigen::Quaternionf(
      static_cast<float>(rotation.w), static_cast<float>(rotation.x),
      static_cast<float>(rotation.y), static_cast<float>(rotation.z)));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(
      this->get_logger(), "Could not transform scan frame '%s' to base frame '%s': %s",
      scan_msg->header.frame_id.c_str(), parameters_->base_frame.c_str(), ex.what());
    return;
  }

  const rclcpp::Time scan_stamp(scan_msg->header.stamp);

  std::vector<glidar_slam::core::Point2D> scan_points;
  scan_points.reserve(scan_msg->ranges.size());
  float angle = scan_msg->angle_min;
  for (const float range : scan_msg->ranges) {
    if (std::isfinite(range) && range >= scan_msg->range_min && range <= scan_msg->range_max) {
      const Eigen::Vector3f point =
        base_from_scan_eigen *
        Eigen::Vector3f(range * std::cos(angle), range * std::sin(angle), 0.0F);
      scan_points.push_back({point.x(), point.y()});
    }
    angle += scan_msg->angle_increment;
  }
  const std::shared_ptr<const LaserScan> laser_scan =
    std::make_shared<LaserScan>(std::move(scan_points));

  glidar_slam::core::SensorData sensor_data(laser_scan);
  Eigen::Affine3f base_from_camera_eigen = Eigen::Affine3f::Identity();

  geometry_msgs::msg::TransformStamped base_from_camera;
  try {
    base_from_camera = tf_buffer_->lookupTransform(
      parameters_->base_frame, color_msg->header.frame_id, color_msg->header.stamp);
  } catch (const tf2::TransformException & exception) {
    RCLCPP_WARN(
      this->get_logger(), "Cannot transform RGB-D camera frame to base frame: %s",
      exception.what());
  }

  const auto & translation = base_from_camera.transform.translation;
  const auto & rotation = base_from_camera.transform.rotation;

  base_from_camera_eigen.translate(Eigen::Vector3f(
    static_cast<float>(translation.x), static_cast<float>(translation.y),
    static_cast<float>(translation.z)));
  base_from_camera_eigen.rotate(Eigen::Quaternionf(
    static_cast<float>(rotation.w), static_cast<float>(rotation.x), static_cast<float>(rotation.y),
    static_cast<float>(rotation.z)));

  const glidar_slam::core::CameraModel camera_model(
    cv::Size(static_cast<int>(camera_info_msg->width), static_cast<int>(camera_info_msg->height)),
    static_cast<float>(camera_info_msg->k[0]), static_cast<float>(camera_info_msg->k[4]),
    static_cast<float>(camera_info_msg->k[2]), static_cast<float>(camera_info_msg->k[5]),
    base_from_camera_eigen);

  sensor_data = glidar_slam::core::SensorData(
    cv_ptr_color->image, cv_ptr_depth->image, laser_scan, camera_model);

  if (!slam_system_->process(
        scan_stamp.seconds(), sensor_data, odom_pose, latest_odom_covariance_)) {
    RCLCPP_WARN(this->get_logger(), "SLAM system could not process the lidar scan.");
    return;
  }

  const std::optional<gtsam::Pose3> optimized_pose = slam_system_->getLatestPose();
  if (!optimized_pose) {
    RCLCPP_WARN(
      this->get_logger(),
      "SLAM system did not return an optimized pose. This should never happen.");
    return;
  }

  const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> scans_transformed =
    slam_system_->getTransformedKeyFrameScans();

  const auto publish_start = std::chrono::steady_clock::now();

  std::vector<std::shared_ptr<const KeyFrame>> keyframes = slam_system_->getKeyFrames();

  publishGraph(keyframes);
  publishOccupancyGrid(scans_transformed, rclcpp::Time(scan_msg->header.stamp));
  publishGroundMarkingGrid(rclcpp::Time(scan_msg->header.stamp));
  publishDebugImage(rclcpp::Time(scan_msg->header.stamp));

  if (parameters_->ground_debug_enable) {
    const auto observation = slam_system_->getLatestGroundObservation();

    if (!observation) {
      return;
    }

    const glidar_slam::core::CameraIntrinsics intrinsics{
      camera_info_msg->k[0], camera_info_msg->k[4], camera_info_msg->k[2], camera_info_msg->k[5]};
    publishGroundDebugImage(*observation, cv_ptr_color->image, intrinsics, base_from_camera_eigen);
    publishGroundDebug(*observation, color_msg->header.stamp);
  }

  const double elapsed_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - publish_start)
      .count();

  if (parameters_->debug_timings) {
    RCLCPP_INFO(
      this->get_logger(), "Publishing graph, occupancy grid, and ground map took %.2f ms",
      elapsed_ms);
  }
}

void Ros2SlamWrapper::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr & msg)
{
  latest_odom_covariance_.setZero();
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

void Ros2SlamWrapper::publishGroundDebug(
  const glidar_slam::core::GroundPlaneObservation & observation, const rclcpp::Time & stamp) const
{
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(observation.binary_ground_cloud, cloud_msg);
  cloud_msg.header.stamp = stamp;
  cloud_msg.header.frame_id = parameters_->base_frame;
  ground_debug_cloud_publisher_->publish(cloud_msg);

  visualization_msgs::msg::MarkerArray marker_array;

  marker_array.markers.push_back(makeGroundPlaneMarker(
    observation, parameters_->base_frame, stamp, parameters_->ground_debug_plane_size));

  marker_array.markers.push_back(
    makeGroundNormalMarker(observation, parameters_->base_frame, stamp));

  ground_debug_marker_publisher_->publish(marker_array);

  const double inlier_ratio = observation.point_count == 0
                                ? 0.0
                                : static_cast<double>(observation.inlier_count) /
                                    static_cast<double>(observation.point_count);
  const double tilt_degrees =
    std::acos(std::clamp(static_cast<double>(observation.normal_in_base.z()), -1.0, 1.0)) * 180.0 /
    M_PI;
  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 2000,
    "Ground debug: points=%zu inliers=%zu ratio=%.3f normal=[%.3f, %.3f, %.3f] "
    "plane_coefficient=%.3f tilt=%.2f deg",
    observation.point_count, observation.inlier_count, inlier_ratio, observation.normal_in_base.x(),
    observation.normal_in_base.y(), observation.normal_in_base.z(), observation.distance_to_base,
    tilt_degrees);
}

void Ros2SlamWrapper::clearGroundDebug(const rclcpp::Time & stamp) const
{
  if (!parameters_->ground_debug_enable) {
    return;
  }

  pcl::PointCloud<pcl::PointXYZRGB> empty_cloud;
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(empty_cloud, cloud_msg);
  cloud_msg.header.stamp = stamp;
  cloud_msg.header.frame_id = parameters_->base_frame;
  ground_debug_cloud_publisher_->publish(cloud_msg);

  visualization_msgs::msg::MarkerArray marker_array;
  marker_array.markers.push_back(makeGroundDeleteMarker(0, parameters_->base_frame));
  marker_array.markers.push_back(makeGroundDeleteMarker(1, parameters_->base_frame));
  ground_debug_marker_publisher_->publish(marker_array);

  sensor_msgs::msg::Image empty_image;
  empty_image.header.stamp = stamp;
  empty_image.header.frame_id = parameters_->base_frame;
  empty_image.encoding = sensor_msgs::image_encodings::BGR8;
  ground_debug_image_publisher_->publish(empty_image);
}

void Ros2SlamWrapper::publishGroundDebugImage(
  const glidar_slam::core::GroundPlaneObservation & observation, const cv::Mat & color,
  const glidar_slam::core::CameraIntrinsics & intrinsics, const Eigen::Affine3f & base_from_camera)
{
  cv::Mat overlay = color.clone();
  const int left = std::clamp(
    static_cast<int>(std::floor(parameters_->ground_roi_ratios[0] * overlay.cols)), 0,
    overlay.cols);
  const int right = std::clamp(
    static_cast<int>(std::ceil((1.0F - parameters_->ground_roi_ratios[1]) * overlay.cols)), 0,
    overlay.cols);
  const int top = std::clamp(
    static_cast<int>(std::floor(parameters_->ground_roi_ratios[2] * overlay.rows)), 0,
    overlay.rows);
  const int bottom = std::clamp(
    static_cast<int>(std::ceil((1.0F - parameters_->ground_roi_ratios[3]) * overlay.rows)), 0,
    overlay.rows);
  if (right > left && bottom > top) {
    cv::rectangle(
      overlay, cv::Point(left, top), cv::Point(right - 1, bottom - 1), cv::Scalar(255, 0, 255), 2);
  }

  const Eigen::Affine3f camera_from_base = base_from_camera.inverse();

  for (const auto & point : observation.binary_ground_cloud) {
    const Eigen::Vector3f point_in_camera =
      camera_from_base * Eigen::Vector3f(point.x, point.y, point.z);
    if (!point_in_camera.allFinite() || point_in_camera.z() <= 0.0F) {
      continue;
    }
    const int column = static_cast<int>(
      std::lround(intrinsics.fx * point_in_camera.x() / point_in_camera.z() + intrinsics.cx));
    const int row = static_cast<int>(
      std::lround(intrinsics.fy * point_in_camera.y() / point_in_camera.z() + intrinsics.cy));
    if (column >= 0 && column < overlay.cols && row >= 0 && row < overlay.rows) {
      cv::circle(overlay, cv::Point(column, row), 2, cv::Scalar(0, 255, 0), cv::FILLED);
    }
  }

  cv_bridge::CvImage cv_image;
  cv_image.header.stamp = this->now();
  cv_image.header.frame_id = parameters_->base_frame;
  cv_image.encoding = sensor_msgs::image_encodings::BGR8;
  cv_image.image = overlay;
  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  cv_image.toImageMsg(*msg);

  ground_debug_image_publisher_->publish(std::move(msg));
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

void Ros2SlamWrapper::publishGroundMarkingGrid(const rclcpp::Time & stamp)
{
  if (!parameters_->ground_marking_map_enable) {
    return;
  }

  const auto ground_clouds = slam_system_->getTransformedGroundMarkingClouds();
  if (!ground_marking_grid_->buildFromGroundClouds(ground_clouds)) {
    return;
  }

  ground_marking_grid_publisher_->publish(
    Utils::toRosMessage(*ground_marking_grid_, parameters_->map_frame, stamp));
}

void Ros2SlamWrapper::publishMapToOdom()
{
  const gtsam::Pose3 map_to_odom = slam_system_->getMapToOdom();
  tf_broadcaster_->sendTransform(poseToTransformStamped(
    map_to_odom, parameters_->map_frame, parameters_->odom_frame,
    this->now() + rclcpp::Duration::from_seconds(0.6)));
}

void Ros2SlamWrapper::publishDebugImage(const rclcpp::Time & stamp)
{
  if (!parameters_->csm_debug_enable) {
    return;
  }

  const auto low_res_debug = slam_system_->getLatestLowResDebug();
  const auto high_res_debug = slam_system_->getLatestHighResDebug();
  if (low_res_debug && !low_res_debug->pixels.empty()) {
    csm_debug_low_publisher_->publish(
      toHeatmapRosImage(low_res_debug.value(), parameters_->map_frame, stamp));
  }

  if (high_res_debug && !high_res_debug->pixels.empty()) {
    csm_debug_high_publisher_->publish(
      toHeatmapRosImage(high_res_debug.value(), parameters_->map_frame, stamp));
  }
}

sensor_msgs::msg::Image Ros2SlamWrapper::toHeatmapRosImage(
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

PointCloudXYZ Ros2SlamWrapper::transformPointCloud(
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

bool Ros2SlamWrapper::parseGroundRoiRatios(const std::string & value, std::vector<float> & ratios)
{
  std::istringstream stream(value);
  std::vector<float> parsed(4, 0.0F);
  for (float & ratio : parsed) {
    if (!(stream >> ratio) || ratio < 0.0F || ratio > 1.0F) {
      return false;
    }
  }
  if (stream >> std::ws && !stream.eof()) {
    return false;
  }
  if (parsed[0] + parsed[1] >= 1.0F || parsed[2] + parsed[3] >= 1.0F) {
    return false;
  }
  ratios = std::move(parsed);
  return true;
}

visualization_msgs::msg::Marker Ros2SlamWrapper::makeGroundPlaneMarker(
  const glidar_slam::core::GroundPlaneObservation & observation, const std::string & frame,
  const rclcpp::Time & stamp, double plane_size)
{
  visualization_msgs::msg::Marker marker;
  marker.header.stamp = stamp;
  marker.header.frame_id = frame;
  marker.ns = "ground_plane_debug";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::CUBE;
  marker.action = visualization_msgs::msg::Marker::ADD;

  Eigen::Vector3f normal = observation.normal_in_base;
  if (!normal.allFinite() || normal.squaredNorm() < 1e-12F) {
    normal = Eigen::Vector3f::UnitZ();
  } else {
    normal.normalize();
  }
  const Eigen::Vector3f plane_point = -observation.distance_to_base * normal;
  const Eigen::Vector3f rotation_axis = Eigen::Vector3f::UnitZ().cross(normal);
  Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();
  if (rotation_axis.squaredNorm() < 1e-12F) {
    if (normal.z() < 0.0F) {
      orientation =
        Eigen::Quaternionf(Eigen::AngleAxisf(static_cast<float>(M_PI), Eigen::Vector3f::UnitX()));
    }
  } else {
    orientation = Eigen::Quaternionf(Eigen::AngleAxisf(
      std::acos(std::clamp(normal.z(), -1.0F, 1.0F)), rotation_axis.normalized()));
  }

  marker.pose.position.x = plane_point.x();
  marker.pose.position.y = plane_point.y();
  marker.pose.position.z = plane_point.z();
  marker.pose.orientation.x = orientation.x();
  marker.pose.orientation.y = orientation.y();
  marker.pose.orientation.z = orientation.z();
  marker.pose.orientation.w = orientation.w();
  marker.scale.x = plane_size;
  marker.scale.y = plane_size;
  marker.scale.z = 0.01;
  marker.color.r = 0.1f;
  marker.color.g = 1.0f;
  marker.color.b = 0.1f;
  marker.color.a = 0.35f;
  return marker;
}

visualization_msgs::msg::Marker Ros2SlamWrapper::makeGroundNormalMarker(
  const glidar_slam::core::GroundPlaneObservation & observation, const std::string & frame,
  const rclcpp::Time & stamp)
{
  visualization_msgs::msg::Marker marker;
  marker.header.stamp = stamp;
  marker.header.frame_id = frame;
  marker.ns = "ground_plane_debug";
  marker.id = 1;
  marker.type = visualization_msgs::msg::Marker::ARROW;
  marker.action = visualization_msgs::msg::Marker::ADD;

  Eigen::Vector3f normal = observation.normal_in_base;
  if (!normal.allFinite() || normal.squaredNorm() < 1e-12F) {
    normal = Eigen::Vector3f::UnitZ();
  } else {
    normal.normalize();
  }
  const Eigen::Vector3f start = -observation.distance_to_base * normal;
  const Eigen::Vector3f end = start + 0.5F * normal;

  geometry_msgs::msg::Point start_point;
  start_point.x = start.x();
  start_point.y = start.y();
  start_point.z = start.z();
  geometry_msgs::msg::Point end_point;
  end_point.x = end.x();
  end_point.y = end.y();
  end_point.z = end.z();
  marker.points = {start_point, end_point};
  marker.scale.x = 0.03;
  marker.scale.y = 0.06;
  marker.scale.z = 0.09;
  marker.color.r = 1.0F;
  marker.color.g = 0.1F;
  marker.color.b = 0.1F;
  marker.color.a = 1.0F;
  return marker;
}

visualization_msgs::msg::Marker Ros2SlamWrapper::makeGroundDeleteMarker(
  int id, const std::string & frame)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame;
  marker.ns = "ground_plane_debug";
  marker.id = id;
  marker.action = visualization_msgs::msg::Marker::DELETE;
  return marker;
}

}  // namespace glidar_slam::ros2
