#include "glidar_slam_ros/ros2_slam_wrapper.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "Eigen/Eigenvalues"
#include "Eigen/Geometry"
#include "cv_bridge/cv_bridge.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "glidar_slam/camera_model.hpp"
#include "glidar_slam/sensor_data.hpp"
#include "glidar_slam_ros/utils.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "pcl/common/transforms.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "sensor_msgs/image_encodings.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "visualization_msgs/msg/marker.hpp"

namespace glidar_slam_ros {

using glidar_slam::KeyFrame;
using glidar_slam::LaserScan;
using glidar_slam::Parameters;
using glidar_slam::PointCloudXYZ;
using glidar_slam::SlamSystem;
using GraphEdge = glidar_slam::MapDatabase::GraphEdge;

Ros2SlamWrapper::Ros2SlamWrapper(const rclcpp::NodeOptions & options) : Node("glidar_slam", options)
{
  parameters_ = std::make_shared<Parameters>();
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

  parameters_->color_image_topic =
    this->declare_parameter<std::string>("color_image_topic", "/camera/color/image_raw");
  parameters_->aligned_depth_image_topic = this->declare_parameter<std::string>(
    "aligned_depth_image_topic", "/camera/aligned_depth/image_raw");
  parameters_->color_camera_info_topic =
    this->declare_parameter<std::string>("color_camera_info_topic", "/camera/color/camera_info");

  parameters_->approx_sync_max_interval =
    this->declare_parameter<double>("approx_sync_max_interval", 0.1);

  parameters_->odom_covariance_diagonal = this->declare_parameter<std::vector<double>>(
    "odom_covariance_diagonal", std::vector<double>{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0});
  if (parameters_->odom_covariance_diagonal.size() != 6) {
    throw std::runtime_error("odom_covariance_diagonal must contain six values");
  }

  parameters_->minimum_travel_distance =
    this->declare_parameter<double>("minimum_travel_distance", 0.5);
  parameters_->minimum_travel_heading =
    this->declare_parameter<double>("minimum_travel_heading", 0.5);
  parameters_->unobservable_variance =
    this->declare_parameter<double>("unobservable_variance", 1e6);
  parameters_->submap_window_size = this->declare_parameter<int>("submap_window_size", 5);

  parameters_->mode =
    parseStartupMode(this->declare_parameter<std::string>("startup.mode", "mapping"));
  parameters_->map_load_path = this->declare_parameter<std::string>("startup.map_load_path", "");
  parameters_->initial_pose_use_provided =
    this->declare_parameter<bool>("startup.initial_pose.use_provided", false);
  parameters_->initial_pose = this->declare_parameter<std::vector<double>>(
    "startup.initial_pose.pose", std::vector<double>{0.0, 0.0, 0.0});
  if (parameters_->initial_pose.size() != 3) {
    throw std::runtime_error("startup.initial_pose.pose must contain [x, y, yaw]");
  }
  parameters_->localization_minimum_score =
    this->declare_parameter<double>("localization_minimum_score", 0.5);

  parameters_->debug_visualize_covariances =
    this->declare_parameter<bool>("debug_visualize_covariances", false);
  parameters_->debug_timings = this->declare_parameter<bool>("debug_timings", false);
  parameters_->ground_debug_enable = this->declare_parameter<bool>("ground_debug_enable", false);
  parameters_->ground_matching_debug_enable =
    this->declare_parameter<bool>("ground_matching_debug_enable", false);
  parameters_->loop_debug_enable = this->declare_parameter<bool>("loop_debug_enable", false);

  parameters_->scan_voxelization_enable =
    this->declare_parameter<bool>("scan.voxelization_enable", true);
  parameters_->scan_voxelization_size =
    this->declare_parameter<double>("scan.voxelization_size", 0.2);
  parameters_->scan_densification_enable =
    this->declare_parameter<bool>("scan.densification_enable", true);

  parameters_->mapping_occupancy_resolution =
    this->declare_parameter<double>("mapping.occupancy.resolution", 0.05);
  parameters_->mapping_ground_resolution =
    this->declare_parameter<double>("mapping.ground.resolution", 0.02);
  parameters_->mapping_ground_enable_texture_mapping =
    this->declare_parameter<bool>("mapping.ground.enable_texture_mapping", true);
  parameters_->mapping_ground_enable_ground_marking_mapping =
    this->declare_parameter<bool>("mapping.ground.enable_ground_marking_mapping", false);
  parameters_->mapping_occupancy_threshold =
    this->declare_parameter<double>("mapping.occupancy.threshold", 0.5);
  parameters_->mapping_prob_hit = this->declare_parameter<double>("mapping.prob_hit", 0.8);
  parameters_->mapping_prob_miss = this->declare_parameter<double>("mapping.prob_miss", 0.35);
  parameters_->mapping_prob_cap_min =
    this->declare_parameter<double>("mapping.prob_cap_min", 0.1192);
  parameters_->mapping_prob_cap_max =
    this->declare_parameter<double>("mapping.prob_cap_max", 0.971);

  parameters_->ground_optimization_enable =
    this->declare_parameter<bool>("ground_optimization_enable", true);
  parameters_->ground_normal_sigma = this->declare_parameter<double>("ground_normal_sigma", 0.05);
  parameters_->ground_distance_sigma =
    this->declare_parameter<double>("ground_distance_sigma", 0.03);

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
  parameters_->ground_extraction_adaptive_threshold_block_size =
    this->declare_parameter<int>("ground_extraction_adaptive_threshold_block_size", 101);
  parameters_->ground_extraction_adaptive_threshold_C =
    this->declare_parameter<double>("ground_extraction_adaptive_threshold_C", -65.0);

  parameters_->ground_matching_enable =
    this->declare_parameter<bool>("ground_matching_enable", false);
  parameters_->ground_matching_minimum_score =
    this->declare_parameter<double>("ground_matching_minimum_score", 0.5);
  parameters_->ground_matching_max_distance =
    this->declare_parameter<double>("ground_matching_max_distance", 2.0);

  parameters_->loop_minimum_key_separation =
    static_cast<uint64_t>(this->declare_parameter<int>("loop_minimum_key_separation", 20));
  parameters_->loop_maximum_yaw_difference =
    this->declare_parameter<double>("loop_maximum_yaw_difference", 1.0);
  parameters_->loop_mahalanobis_threshold =
    this->declare_parameter<double>("loop_mahalanobis_threshold", 3.0);
  parameters_->loop_minimum_score = this->declare_parameter<double>("loop_minimum_score", 0.5);

  parameters_->isam_relinearizeThreshold =
    this->declare_parameter<double>("isam_relinearizeThreshold", 0.1);
  parameters_->isam_relinearizeSkip = this->declare_parameter<int>("isam_relinearizeSkip", 10);

  scan_matcher_loader_ = std::make_unique<pluginlib::ClassLoader<ScanMatcherInterface>>(
    "glidar_slam_ros", "glidar_slam_ros::ScanMatcherInterface");

  // Callback groups
  odom_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  camera_callback_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  map_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  state_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
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

  save_state_service_ = this->create_service<glidar_slam_msgs::srv::SaveSlamState>(
    "save_state",
    std::bind(
      &Ros2SlamWrapper::saveStateCallback, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(), state_callback_group_);
  save_maps_service_ = this->create_service<glidar_slam_msgs::srv::SaveMaps>(
    "save_maps",
    std::bind(
      &Ros2SlamWrapper::saveMapsCallback, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(), state_callback_group_);

  load_state_service_ = this->create_service<glidar_slam_msgs::srv::LoadSlamState>(
    "load_state",
    std::bind(
      &Ros2SlamWrapper::loadStateCallback, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(), state_callback_group_);
  set_localization_mode_service_ = this->create_service<glidar_slam_msgs::srv::SetLocalizationMode>(
    "set_localization_mode",
    std::bind(
      &Ros2SlamWrapper::setLocalizationModeCallback, this, std::placeholders::_1,
      std::placeholders::_2),
    rclcpp::ServicesQoS(), state_callback_group_);

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
  ground_synchronizer_->setMaxIntervalDuration(
    rclcpp::Duration::from_seconds(parameters_->approx_sync_max_interval));

  // Publishers
  marker_array_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "glidar_slam/graph_visualization", 10);
  occupancy_grid_publisher_ =
    this->create_publisher<nav_msgs::msg::OccupancyGrid>("glidar_slam/map", 10);
  ground_marking_grid_publisher_ =
    this->create_publisher<nav_msgs::msg::OccupancyGrid>("glidar_slam/ground_markings_map", 10);
  ground_texture_image_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>("glidar_slam/ground_texture_map", 10);
  ground_debug_cloud_publisher_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("glidar_slam/debug/ground_inliers", 10);
  ground_initial_debug_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "glidar_slam/debug/ground_initial_inliers", 10);
  ground_matching_debug_publisher_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("glidar_slam/debug/ground_matching", 10);
  ground_debug_marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "glidar_slam/debug/ground_plane", 10);
  ground_debug_image_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>("glidar_slam/debug/ground_overlay", 10);
  csm_debug_low_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>("glidar_slam/debug/csm_likelihood_low", 10);
  csm_debug_high_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>("glidar_slam/debug/csm_likelihood_high", 10);
  ground_extraction_debug_publisher_ =
    this->create_publisher<sensor_msgs::msg::Image>("glidar_slam/debug/ground_extraction", 10);
  estimated_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "glidar_slam/estimated_pose", 10);

  // Timer
  transform_broadcast_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100), std::bind(&Ros2SlamWrapper::publishMapToOdom, this),
    transform_broadcast_callback_group_);
  map_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(1000), std::bind(&Ros2SlamWrapper::publishMapsTimerCallback, this),
    map_callback_group_);

  // SLAM System
  slam_system_ = std::make_unique<SlamSystem>(
    parameters_, loadScanMatcher("csm_local_tracker"),
    loadScanMatcher("csm_ground_marking_detection"), loadScanMatcher("csm_loop_closure_detection"));
}

void Ros2SlamWrapper::saveStateCallback(
  std::shared_ptr<glidar_slam_msgs::srv::SaveSlamState::Request> request,
  std::shared_ptr<glidar_slam_msgs::srv::SaveSlamState::Response> response)
{
  if (request->path.empty()) {
    response->success = false;
    response->message = "Path is empty";
    return;
  }

  std::string error;
  response->success = slam_system_->saveState(request->path, &error);
  response->message = response->success ? "SLAM state saved" : error;

  response->keyframe_count = slam_system_->getKeyFrames().size();
  response->factor_count = slam_system_->getFactorCount();
  response->localization_active = slam_system_->isLocalizationMode();
}

void Ros2SlamWrapper::saveMapsCallback(
  std::shared_ptr<glidar_slam_msgs::srv::SaveMaps::Request> request,
  std::shared_ptr<glidar_slam_msgs::srv::SaveMaps::Response> response)
{
  // agreed upon value for unknown occupancy for better visual differentiation
  constexpr int UNKNOWN_OCCUPANCY = 205;

  if (request->base_filepath.empty() || (!request->save_occupancy && !request->save_texture)) {
    response->success = false;
    response->message = "base_filepath must be non-empty and at least one map must be selected";
    return;
  }

  const auto snapshot = slam_system_->getLatestGlobalMap();
  if (!snapshot) {
    response->success = false;
    response->message = "No global map is available";
    return;
  }

  try {
    const auto & occupancy_info = snapshot->occupancy->getInfo();
    if (request->save_occupancy) {
      const std::string pgm_path = request->base_filepath + ".pgm";
      const std::string yaml_path = request->base_filepath + ".yaml";
      std::ofstream pgm(pgm_path, std::ios::binary);
      if (!pgm) {
        throw std::runtime_error("Cannot open " + pgm_path);
      }
      pgm << "P5\n" << occupancy_info.width << " " << occupancy_info.height << "\n255\n";
      const auto & data = snapshot->occupancy->getData();
      for (std::size_t y = 0; y < occupancy_info.height; ++y) {
        for (std::size_t x = 0; x < occupancy_info.width; ++x) {
          const std::size_t map_y = occupancy_info.height - 1 - y;
          const std::size_t map_x = occupancy_info.width - 1 - x;
          const std::size_t index = map_y * occupancy_info.width + map_x;

          const auto value = data[index];
          const std::uint8_t pixel =
            value < 0
              ? UNKNOWN_OCCUPANCY
              : static_cast<std::uint8_t>(255U - (static_cast<unsigned int>(value) * 255U / 100U));
          pgm.put(static_cast<char>(pixel));
        }
      }

      std::ofstream yaml(yaml_path);
      if (!yaml) {
        throw std::runtime_error("Cannot open " + yaml_path);
      }
      yaml << "image: " << pgm_path.substr(pgm_path.find_last_of('/') + 1) << "\n"
           << "mode: trinary\nresolution: " << occupancy_info.resolution << "\norigin: ["
           << occupancy_info.origin_x << ", " << occupancy_info.origin_y
           << ", 0.0]\nnegate: 0\noccupied_thresh: 0.6\nfree_thresh: 0.6\nthreshold: 128\n";
    }

    if (request->save_texture) {
      const auto & info = snapshot->texture->getInfo();
      const auto & rgb = snapshot->texture->getRgbData();
      if (rgb.size() != static_cast<std::size_t>(info.width) * info.height * 3U) {
        throw std::runtime_error("Texture map data size does not match its metadata");
      }
      cv::Mat image(info.height, info.width, CV_8UC4);
      for (uint32_t r = 0; r < info.height; ++r) {
        for (uint32_t c = 0; c < info.width; ++c) {
          const std::size_t map_y = info.height - 1 - r;
          const std::size_t map_x = info.width - 1 - c;
          const std::size_t index = map_y * info.width + map_x;

          auto & pixel = image.at<cv::Vec4b>(r, c);
          pixel[0] = rgb[index * 3U + 2U];
          pixel[1] = rgb[index * 3U + 1U];
          pixel[2] = rgb[index * 3U];
          pixel[3] = (pixel[0] || pixel[1] || pixel[2]) ? 255U : 0U;
        }
      }
      const std::string png_path = request->base_filepath + "_texture.png";
      if (!cv::imwrite(png_path, image)) {
        throw std::runtime_error("Cannot write " + png_path);
      }
      std::ofstream yaml(request->base_filepath + "_texture.yaml");
      if (!yaml) {
        throw std::runtime_error("Cannot open texture YAML");
      }
      yaml << "image: " << png_path.substr(png_path.find_last_of('/') + 1)
           << "\nmode: trinary\nresolution: " << info.resolution << "\norigin: [" << info.origin_x
           << ", " << info.origin_y << ", 0.0]\nnegate: 0\n";
    }
  } catch (const std::exception & exception) {
    response->success = false;
    response->message = exception.what();
    return;
  }
  response->success = true;
  response->message = "Maps saved";
}

void Ros2SlamWrapper::loadStateCallback(
  std::shared_ptr<glidar_slam_msgs::srv::LoadSlamState::Request> request,
  std::shared_ptr<glidar_slam_msgs::srv::LoadSlamState::Response> response)
{
  if (request->path.empty()) {
    response->success = false;
    response->message = "Path is empty";
    return;
  }

  std::string error;

  const gtsam::Pose3 initial_map_pose(
    gtsam::Rot3::Quaternion(
      request->initial_map_pose.orientation.w, request->initial_map_pose.orientation.x,
      request->initial_map_pose.orientation.y, request->initial_map_pose.orientation.z),
    gtsam::Point3(
      request->initial_map_pose.position.x, request->initial_map_pose.position.y,
      request->initial_map_pose.position.z));

  Parameters::Mode mode{Parameters::Mode::Mapping};
  try {
    mode = parseStartupMode(request->mode);
  } catch (const std::exception & exception) {
    response->success = false;
    response->message = exception.what();
    response->localization_active = slam_system_->isLocalizationMode();
    return;
  }

  response->success = slam_system_->loadState(
    request->path, initial_map_pose, request->use_saved_pose,
    mode == Parameters::Mode::Localization, &error);

  response->message = response->success ? "SLAM state loaded" : error;

  const std::vector<std::shared_ptr<const KeyFrame>> & keyframes = slam_system_->getKeyFrames();

  response->keyframe_count = slam_system_->getKeyFrames().size();
  response->factor_count = slam_system_->getFactorCount();
  response->localization_active = slam_system_->isLocalizationMode();

  if (response->success) {
    publishGraph(keyframes, slam_system_->getEdges());
  }
}

void Ros2SlamWrapper::setLocalizationModeCallback(
  std::shared_ptr<glidar_slam_msgs::srv::SetLocalizationMode::Request> request,
  std::shared_ptr<glidar_slam_msgs::srv::SetLocalizationMode::Response> response)
{
  const gtsam::Pose3 initial_map_pose(
    gtsam::Rot3::Quaternion(
      request->initial_map_pose.orientation.w, request->initial_map_pose.orientation.x,
      request->initial_map_pose.orientation.y, request->initial_map_pose.orientation.z),
    gtsam::Point3(
      request->initial_map_pose.position.x, request->initial_map_pose.position.y,
      request->initial_map_pose.position.z));

  std::string error;
  response->success = slam_system_->setLocalizationMode(
    request->enable, initial_map_pose, request->use_current_pose, &error);
  response->message = response->success ? "localization mode updated" : error;
  response->localization_active = slam_system_->isLocalizationMode();
}

std::unique_ptr<ScanMatcherInterface> Ros2SlamWrapper::loadScanMatcher(const std::string & name)
{
  const std::string plugin = this->declare_parameter<std::string>(
    name + ".plugin", "glidar_slam/CorrelativeScanMatcherPlugin");
  try {
    auto matcher =
      std::unique_ptr<ScanMatcherInterface>(scan_matcher_loader_->createUnmanagedInstance(plugin));
    matcher->initialize(this, name);
    return matcher;
  } catch (const std::exception & exception) {
    throw std::runtime_error(
      "Failed to load scan matcher '" + plugin + "' for " + name + ": " + exception.what());
  }
}

rcl_interfaces::msg::SetParametersResult Ros2SlamWrapper::onParametersChanged(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  bool rebuild_global_map = false;

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
    } else if (name == "unobservable_variance") {
      updated.unobservable_variance = parameter.as_double();
    } else if (name == "debug_visualize_covariances") {
      updated.debug_visualize_covariances = parameter.as_bool();
    } else if (name == "scan.voxelization_enable") {
      updated.scan_voxelization_enable = parameter.as_bool();
    } else if (name == "scan.voxelization_size") {
      updated.scan_voxelization_size = parameter.as_double();
    } else if (name == "scan.densification_enable") {
      updated.scan_densification_enable = parameter.as_bool();
    } else if (name == "submap_window_size") {
      updated.submap_window_size = static_cast<int>(parameter.as_int());
    } else if (name == "ground_debug_enable") {
      updated.ground_debug_enable = parameter.as_bool();
    } else if (name == "ground_optimization_enable") {
      updated.ground_optimization_enable = parameter.as_bool();
    } else if (name == "ground_normal_sigma") {
      updated.ground_normal_sigma = parameter.as_double();
    } else if (name == "ground_distance_sigma") {
      updated.ground_distance_sigma = parameter.as_double();
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
    } else if (name == "ground_extraction_adaptive_threshold_block_size") {
      updated.ground_extraction_adaptive_threshold_block_size =
        static_cast<int>(parameter.as_int());
    } else if (name == "ground_extraction_adaptive_threshold_C") {
      updated.ground_extraction_adaptive_threshold_C = parameter.as_double();
    } else if (name == "mapping.occupancy.resolution") {
      updated.mapping_occupancy_resolution = parameter.as_double();
      rebuild_global_map = true;
    } else if (name == "mapping.ground.enable_texture_mapping") {
      updated.mapping_ground_enable_texture_mapping = parameter.as_bool();
      rebuild_global_map = true;
    } else if (name == "mapping.ground.enable_ground_marking_mapping") {
      updated.mapping_ground_enable_ground_marking_mapping = parameter.as_bool();
      rebuild_global_map = true;
    } else if (name == "mapping.ground.resolution") {
      updated.mapping_ground_resolution = parameter.as_double();
      rebuild_global_map = true;
    } else if (name == "mapping.occupancy_threshold") {
      updated.mapping_occupancy_threshold = parameter.as_double();
      rebuild_global_map = true;
    } else if (name == "mapping.prob_hit") {
      updated.mapping_prob_hit = parameter.as_double();
      rebuild_global_map = true;
    } else if (name == "mapping.prob_miss") {
      updated.mapping_prob_miss = parameter.as_double();
      rebuild_global_map = true;
    } else if (name == "mapping.prob_cap_min") {
      updated.mapping_prob_cap_min = parameter.as_double();
      rebuild_global_map = true;
    } else if (name == "mapping.prob_cap_max") {
      updated.mapping_prob_cap_max = parameter.as_double();
      rebuild_global_map = true;
    } else if (name == "ground_matching_enable") {
      updated.ground_matching_enable = parameter.as_bool();
    } else if (name == "ground_matching_minimum_score") {
      updated.ground_matching_minimum_score = parameter.as_double();
    } else if (name == "ground_matching_max_distance") {
      updated.ground_matching_max_distance = parameter.as_double();
    } else if (name == "ground_matching_debug_enable") {
      updated.ground_matching_debug_enable = parameter.as_bool();
    } else if (name == "loop_debug_enable") {
      updated.loop_debug_enable = parameter.as_bool();
    } else if (name == "loop_minimum_key_separation") {
      updated.loop_minimum_key_separation = static_cast<uint64_t>(parameter.as_int());
    } else if (name == "loop_maximum_yaw_difference") {
      updated.loop_maximum_yaw_difference = parameter.as_double();
    } else if (name == "loop_mahalanobis_threshold") {
      updated.loop_mahalanobis_threshold = parameter.as_double();
    } else if (name == "loop_minimum_score") {
      updated.loop_minimum_score = parameter.as_double();
    } else if (name == "localization_minimum_score") {
      updated.localization_minimum_score = parameter.as_double();
    } else if (name == "debug_timings") {
      updated.debug_timings = parameter.as_bool();
    }
  }

  if (rebuild_global_map) {
    slam_system_->rebuildGlobalMap();
  }

  if (
    !std::isfinite(updated.mapping_prob_hit) || updated.mapping_prob_hit <= 0.0 ||
    updated.mapping_prob_hit >= 1.0 || !std::isfinite(updated.mapping_prob_miss) ||
    updated.mapping_prob_miss <= 0.0 || updated.mapping_prob_miss >= 1.0 ||
    !std::isfinite(updated.mapping_prob_cap_min) || updated.mapping_prob_cap_min <= 0.0 ||
    !std::isfinite(updated.mapping_prob_cap_max) || updated.mapping_prob_cap_max <= 0.0 ||
    !std::isfinite(updated.ground_matching_minimum_score) ||
    updated.ground_matching_minimum_score < 0.0) {
    result.successful = false;
    result.reason = "ground marking log-odds, observation age, or matching parameters are invalid";
    return result;
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
    return;
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

  std::vector<glidar_slam::Point2D> scan_points;
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

  glidar_slam::SensorData sensor_data(laser_scan);
  Eigen::Affine3f base_from_camera_eigen = Eigen::Affine3f::Identity();

  geometry_msgs::msg::TransformStamped base_from_camera;
  try {
    base_from_camera = tf_buffer_->lookupTransform(
      parameters_->base_frame, color_msg->header.frame_id, color_msg->header.stamp);
  } catch (const tf2::TransformException & exception) {
    RCLCPP_WARN(
      this->get_logger(), "Cannot transform RGB-D camera frame to base frame: %s",
      exception.what());
    return;
  }

  const auto & translation = base_from_camera.transform.translation;
  const auto & rotation = base_from_camera.transform.rotation;

  base_from_camera_eigen.translate(Eigen::Vector3f(
    static_cast<float>(translation.x), static_cast<float>(translation.y),
    static_cast<float>(translation.z)));
  base_from_camera_eigen.rotate(Eigen::Quaternionf(
    static_cast<float>(rotation.w), static_cast<float>(rotation.x), static_cast<float>(rotation.y),
    static_cast<float>(rotation.z)));

  const glidar_slam::CameraModel camera_model(
    cv::Size(static_cast<int>(camera_info_msg->width), static_cast<int>(camera_info_msg->height)),
    static_cast<float>(camera_info_msg->k[0]), static_cast<float>(camera_info_msg->k[4]),
    static_cast<float>(camera_info_msg->k[2]), static_cast<float>(camera_info_msg->k[5]),
    base_from_camera_eigen);

  sensor_data =
    glidar_slam::SensorData(cv_ptr_color->image, cv_ptr_depth->image, laser_scan, camera_model);

  if (!slam_system_->process(
        scan_stamp.seconds(), sensor_data, odom_pose, latest_odom_covariance_)) {
    return;
  }

  publishPoseEstimate(scan_stamp);

  std::vector<std::shared_ptr<const KeyFrame>> keyframes = slam_system_->getKeyFrames();

  const auto publish_start = std::chrono::steady_clock::now();

  publishGraph(keyframes, slam_system_->getEdges());
  publishDebugImage(rclcpp::Time(scan_msg->header.stamp));
  publishGroundMatchingDebug(rclcpp::Time(scan_msg->header.stamp));

  if (parameters_->ground_debug_enable) {
    const auto observation = slam_system_->getLatestGroundObservation();

    if (!observation) {
      return;
    }

    const glidar_slam::CameraIntrinsics intrinsics{
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

void Ros2SlamWrapper::publishPoseEstimate(const rclcpp::Time & stamp)
{
  const glidar_slam::PoseEstimate estimate = slam_system_->getLatestPoseAndCovariance();

  geometry_msgs::msg::PoseWithCovarianceStamped message;
  message.header.stamp = stamp;
  message.header.frame_id = parameters_->map_frame;

  const auto translation = estimate.pose.translation();
  const auto quaternion = estimate.pose.rotation().toQuaternion();
  message.pose.pose.position.x = translation.x();
  message.pose.pose.position.y = translation.y();
  message.pose.pose.position.z = translation.z();
  message.pose.pose.orientation.w = quaternion.w();
  message.pose.pose.orientation.x = quaternion.x();
  message.pose.pose.orientation.y = quaternion.y();
  message.pose.pose.orientation.z = quaternion.z();

  constexpr std::array<int, 6> gtsam_to_ros{{3, 4, 5, 0, 1, 2}};
  for (std::size_t row = 0; row < gtsam_to_ros.size(); ++row) {
    for (std::size_t column = 0; column < gtsam_to_ros.size(); ++column) {
      message.pose.covariance[row * gtsam_to_ros.size() + column] =
        estimate.covariance(gtsam_to_ros[row], gtsam_to_ros[column]);
    }
  }

  estimated_pose_pub_->publish(message);
}

void Ros2SlamWrapper::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr & msg)
{
  auto sanitizeOdometryVariance = [this](double variance) {
    return std::isfinite(variance) && variance > 0.0 ? variance
                                                     : parameters_->unobservable_variance;
  };

  latest_odom_covariance_.setZero();
  if (
    std::find(
      parameters_->odom_covariance_diagonal.begin(), parameters_->odom_covariance_diagonal.end(),
      -1.0) != parameters_->odom_covariance_diagonal.end()) {
    // gtsam covariance in tangent space is [roll, pitch, yaw, x, y, z] (see gtsam::Pose3)
    // We care about relative covariance, not the total of the odometries pose, that's why we use
    // the twist covariance.
    latest_odom_covariance_(0, 0) = sanitizeOdometryVariance(msg->twist.covariance[21]);
    latest_odom_covariance_(1, 1) = sanitizeOdometryVariance(msg->twist.covariance[28]);
    latest_odom_covariance_(2, 2) = sanitizeOdometryVariance(msg->twist.covariance[35]);
    latest_odom_covariance_(3, 3) = sanitizeOdometryVariance(msg->twist.covariance[0]);
    latest_odom_covariance_(4, 4) = sanitizeOdometryVariance(msg->twist.covariance[7]);
    latest_odom_covariance_(5, 5) = sanitizeOdometryVariance(msg->twist.covariance[14]);
  } else {
    // the ros2 parameter keeps the ros2 convention which is [x, y, z, roll, pitch, yaw]
    latest_odom_covariance_(0, 0) = parameters_->odom_covariance_diagonal[3];
    latest_odom_covariance_(1, 1) = parameters_->odom_covariance_diagonal[4];
    latest_odom_covariance_(2, 2) = parameters_->odom_covariance_diagonal[5];
    latest_odom_covariance_(3, 3) = parameters_->odom_covariance_diagonal[0];
    latest_odom_covariance_(4, 4) = parameters_->odom_covariance_diagonal[1];
    latest_odom_covariance_(5, 5) = parameters_->odom_covariance_diagonal[2];
  }
}

void Ros2SlamWrapper::publishGroundDebug(
  const glidar_slam::GroundPlaneObservation & observation, const rclcpp::Time & stamp) const
{
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*observation.ground_cloud, cloud_msg);
  cloud_msg.header.stamp = stamp;
  cloud_msg.header.frame_id = parameters_->base_frame;
  ground_debug_cloud_publisher_->publish(cloud_msg);

  sensor_msgs::msg::PointCloud2 initial_cloud_msg;
  pcl::toROSMsg(*observation.ground_cloud_sparse, initial_cloud_msg);
  initial_cloud_msg.header.stamp = stamp;
  initial_cloud_msg.header.frame_id = parameters_->base_frame;
  ground_initial_debug_cloud_publisher_->publish(initial_cloud_msg);

  visualization_msgs::msg::MarkerArray marker_array;

  marker_array.markers.push_back(
    makeGroundPlaneMarker(observation, parameters_->base_frame, stamp));

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

void Ros2SlamWrapper::publishGroundMatchingDebug(const rclcpp::Time & stamp)
{
  if (!parameters_->ground_matching_debug_enable) {
    return;
  }

  const auto debug_cloud = slam_system_->getLatestGroundMatchingDebug();
  if (!debug_cloud) {
    return;
  }

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*debug_cloud, cloud_msg);
  cloud_msg.header.stamp = stamp;
  cloud_msg.header.frame_id = parameters_->map_frame;
  ground_matching_debug_publisher_->publish(cloud_msg);
}

void Ros2SlamWrapper::publishGroundDebugImage(
  const glidar_slam::GroundPlaneObservation & observation, const cv::Mat & color,
  const glidar_slam::CameraIntrinsics & intrinsics, const Eigen::Affine3f & base_from_camera)
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

  for (const auto & point : *observation.ground_cloud) {
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

visualization_msgs::msg::Marker Ros2SlamWrapper::keyframeToMarker(
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

std::optional<visualization_msgs::msg::Marker> Ros2SlamWrapper::keyframeCovarianceToMarker(
  const KeyFrame & keyframe, const std::string & frame)
{
  if (!keyframe.covariance) {
    return std::nullopt;
  }

  Eigen::Matrix2d local_covariance;
  local_covariance << (*keyframe.covariance)(3, 3),
    0.5 * ((*keyframe.covariance)(3, 4) + (*keyframe.covariance)(4, 3)),
    0.5 * ((*keyframe.covariance)(3, 4) + (*keyframe.covariance)(4, 3)),
    (*keyframe.covariance)(4, 4);
  if (!local_covariance.allFinite()) {
    return std::nullopt;
  }

  const Eigen::Matrix2d local_to_map =
    Eigen::Rotation2Dd(keyframe.pose.rotation().yaw()).toRotationMatrix();
  const Eigen::Matrix2d covariance = local_to_map * local_covariance * local_to_map.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
  if (solver.info() != Eigen::Success || (solver.eigenvalues().array() <= 0.0).any()) {
    return std::nullopt;
  }

  const auto & translation = keyframe.pose.translation();
  const Eigen::Vector2d axis = solver.eigenvectors().col(1);
  const double major_axis = std::sqrt(solver.eigenvalues()(1));
  const double minor_axis = std::sqrt(solver.eigenvalues()(0));

  visualization_msgs::msg::Marker marker;
  marker.header.stamp = rclcpp::Time(keyframe.timestamp);
  marker.header.frame_id = frame;
  marker.ns = "slam_graph_covariance";
  marker.id = keyframe.key;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.02;
  marker.color.r = 1.0f;
  marker.color.g = 0.1f;
  marker.color.b = 0.1f;
  marker.color.a = 0.8f;

  constexpr int point_count = 36;
  marker.points.reserve(point_count + 1);
  const Eigen::Vector2d perpendicular_axis(-axis.y(), axis.x());
  for (int point_index = 0; point_index <= point_count; ++point_index) {
    const double angle = 2.0 * M_PI * static_cast<double>(point_index) / point_count;
    const Eigen::Vector2d point = Eigen::Vector2d(translation.x(), translation.y()) +
                                  major_axis * std::cos(angle) * axis +
                                  minor_axis * std::sin(angle) * perpendicular_axis;

    geometry_msgs::msg::Point marker_point;
    marker_point.x = point.x();
    marker_point.y = point.y();
    marker_point.z = translation.z();
    marker.points.push_back(marker_point);
  }

  return marker;
}

visualization_msgs::msg::Marker Ros2SlamWrapper::graphEdgesToMarker(
  const std::vector<std::shared_ptr<const KeyFrame>> & keyframes,
  const std::vector<GraphEdge> & edges, const std::string & frame)
{
  std::unordered_map<uint64_t, gtsam::Pose3> poses;
  poses.reserve(keyframes.size());
  for (const auto & keyframe : keyframes) {
    poses.emplace(keyframe->key, keyframe->pose);
  }

  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame;
  marker.ns = "slam_graph_edges";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.04;
  marker.color.r = 0.1f;
  marker.color.g = 0.8f;
  marker.color.b = 1.0f;
  marker.color.a = 1.0f;

  for (const auto & edge : edges) {
    const auto from_key = edge.from_key;
    const auto to_key = edge.to_key;
    const auto from_pose = poses.find(from_key);
    const auto to_pose = poses.find(to_key);
    if (from_pose == poses.end() || to_pose == poses.end()) {
      continue;
    }

    geometry_msgs::msg::Point from_point;
    from_point.x = from_pose->second.x();
    from_point.y = from_pose->second.y();
    from_point.z = from_pose->second.z();
    marker.points.push_back(from_point);
    std_msgs::msg::ColorRGBA edge_color;
    edge_color.r = edge.type == "LoopClosure" ? 1.0F : 0.1F;
    edge_color.g = edge.type == "LoopClosure" ? 0.65F : 0.8F;
    edge_color.b = edge.type == "LoopClosure" ? 0.0F : 1.0F;
    edge_color.a = 1.0F;
    marker.colors.push_back(edge_color);

    geometry_msgs::msg::Point to_point;
    to_point.x = to_pose->second.x();
    to_point.y = to_pose->second.y();
    to_point.z = to_pose->second.z();
    marker.points.push_back(to_point);
    marker.colors.push_back(edge_color);
  }

  if (marker.points.empty()) {
    marker.action = visualization_msgs::msg::Marker::DELETE;
  }
  return marker;
}

void Ros2SlamWrapper::publishGraph(
  const std::vector<std::shared_ptr<const KeyFrame>> & keyframes,
  const std::vector<GraphEdge> & edges)
{
  visualization_msgs::msg::MarkerArray marker_array_msg;

  for (std::size_t index = 0; index < keyframes.size(); ++index) {
    const auto & keyframe = keyframes[index];
    marker_array_msg.markers.push_back(keyframeToMarker(*keyframe, parameters_->map_frame));
    if (parameters_->debug_visualize_covariances || index + 1 == keyframes.size()) {
      if (
        const auto covariance_marker =
          keyframeCovarianceToMarker(*keyframe, parameters_->map_frame)) {
        marker_array_msg.markers.push_back(*covariance_marker);
      }
    } else {
      visualization_msgs::msg::Marker delete_marker;
      delete_marker.header.frame_id = parameters_->map_frame;
      delete_marker.ns = "slam_graph_covariance";
      delete_marker.id = keyframe->key;
      delete_marker.action = visualization_msgs::msg::Marker::DELETE;
      marker_array_msg.markers.push_back(delete_marker);
    }
  }

  marker_array_msg.markers.push_back(graphEdgesToMarker(keyframes, edges, parameters_->map_frame));

  marker_array_publisher_->publish(marker_array_msg);
}

void Ros2SlamWrapper::publishOccupancyGrid(
  const std::shared_ptr<const glidar_slam::mapping::GlobalMapSnapshot> & map_snapshot,
  const rclcpp::Time & stamp)
{
  if (!map_snapshot || !map_snapshot->occupancy) {
    return;
  }
  occupancy_grid_publisher_->publish(
    Utils::toRosMessage(*map_snapshot->occupancy, parameters_->map_frame, stamp));
}

void Ros2SlamWrapper::publishMapsTimerCallback()
{
  const rclcpp::Time stamp = this->now();
  const auto map_snapshot = slam_system_->getLatestGlobalMap();
  publishOccupancyGrid(map_snapshot, stamp);
  publishGroundMap(map_snapshot, stamp);
}

void Ros2SlamWrapper::publishGroundMap(
  const std::shared_ptr<const glidar_slam::mapping::GlobalMapSnapshot> & map_snapshot,
  const rclcpp::Time & stamp)
{
  if (!map_snapshot) {
    return;
  }

  if (parameters_->mapping_ground_enable_texture_mapping && map_snapshot->texture) {
    ground_texture_image_publisher_->publish(
      Utils::toRosImage(*map_snapshot->texture, parameters_->map_frame, stamp));
  }

  if (parameters_->mapping_ground_enable_ground_marking_mapping && map_snapshot->marking) {
    ground_marking_grid_publisher_->publish(
      Utils::toRosMessage(*map_snapshot->marking, parameters_->map_frame, stamp));
  }
}

void Ros2SlamWrapper::publishMapToOdom()
{
  const gtsam::Pose3 map_to_odom = slam_system_->getMapToOdom();
  tf_broadcaster_->sendTransform(poseToTransformStamped(
    map_to_odom, parameters_->map_frame, parameters_->odom_frame,
    this->now() + rclcpp::Duration::from_seconds(0.6)));
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

void Ros2SlamWrapper::publishDebugImage(const rclcpp::Time & stamp)
{
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

  const std::optional<cv::Mat> ground_debug = slam_system_->getLatestGroundExtractionDebug();
  if (ground_debug) {
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = stamp;
    cv_image.header.frame_id = parameters_->camera_frame;
    // binary image
    cv_image.encoding = sensor_msgs::image_encodings::MONO8;
    cv_image.image = ground_debug.value();
    ground_extraction_debug_publisher_->publish(*cv_image.toImageMsg());
  }
}

sensor_msgs::msg::Image Ros2SlamWrapper::toHeatmapRosImage(
  const glidar_slam::CsmResult::DebugImage & debug, const std::string & frame_id,
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

Parameters::Mode Ros2SlamWrapper::parseStartupMode(const std::string & value)
{
  if (value == "mapping") {
    return Parameters::Mode::Mapping;
  }
  if (value == "localization") {
    return Parameters::Mode::Localization;
  }
  throw std::runtime_error("startup.mode must be either 'mapping' or 'localization'");
}

visualization_msgs::msg::Marker Ros2SlamWrapper::makeGroundPlaneMarker(
  const glidar_slam::GroundPlaneObservation & observation, const std::string & frame,
  const rclcpp::Time & stamp)
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
  marker.scale.x = 5.0;
  marker.scale.y = 5.0;
  marker.scale.z = 0.01;
  marker.color.r = 0.1f;
  marker.color.g = 1.0f;
  marker.color.b = 0.1f;
  marker.color.a = 0.35f;
  return marker;
}

visualization_msgs::msg::Marker Ros2SlamWrapper::makeGroundNormalMarker(
  const glidar_slam::GroundPlaneObservation & observation, const std::string & frame,
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

}  // namespace glidar_slam_ros
