#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/system.hpp"
#include "glidar_slam/ros2/scan_matcher_interface.hpp"
#include "glidar_slam_msgs/srv/load_slam_state.hpp"
#include "glidar_slam_msgs/srv/save_slam_state.hpp"
#include "glidar_slam_msgs/srv/set_localization_mode.hpp"
#include "gtsam/geometry/Pose3.h"
#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/synchronizer.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker_array.hpp"

namespace glidar_slam::ros2 {

using glidar_slam::core::KeyFrame;
using glidar_slam::core::Parameters;
using glidar_slam::core::PointCloudXYZ;
using glidar_slam::core::SlamSystem;
using GraphEdge = glidar_slam::core::MapDatabase::GraphEdge;

class Ros2SlamWrapper : public rclcpp::Node
{
public:
  explicit Ros2SlamWrapper(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr & msg);

  void ScanRGBDCallback(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & color_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  void saveStateCallback(
    std::shared_ptr<glidar_slam_msgs::srv::SaveSlamState::Request> request,
    std::shared_ptr<glidar_slam_msgs::srv::SaveSlamState::Response> response);

  void loadStateCallback(
    std::shared_ptr<glidar_slam_msgs::srv::LoadSlamState::Request> request,
    std::shared_ptr<glidar_slam_msgs::srv::LoadSlamState::Response> response);

  void setLocalizationModeCallback(
    std::shared_ptr<glidar_slam_msgs::srv::SetLocalizationMode::Request> request,
    std::shared_ptr<glidar_slam_msgs::srv::SetLocalizationMode::Response> response);

  void publishGraph(
    const std::vector<std::shared_ptr<const KeyFrame>> & keyframes,
    const std::vector<GraphEdge> & edges);
  void publishMapToOdom();
  void publishPoseEstimate(const rclcpp::Time & stamp);
  void publishMapsTimerCallback();
  void publishOccupancyGrid(
    const std::shared_ptr<const glidar_slam::core::GlobalMapSnapshot> & snapshot,
    const rclcpp::Time & stamp);
  void publishGroundMap(
    const std::shared_ptr<const glidar_slam::core::GlobalMapSnapshot> & snapshot,
    const rclcpp::Time & stamp);
  void publishGroundDebug(
    const glidar_slam::core::GroundPlaneObservation & observation,
    const rclcpp::Time & stamp) const;
  void publishGroundMatchingDebug(const rclcpp::Time & stamp);
  void publishGroundDebugImage(
    const glidar_slam::core::GroundPlaneObservation & observation, const cv::Mat & color,
    const glidar_slam::core::CameraIntrinsics & intrinsics,
    const Eigen::Affine3f & base_from_camera);
  void publishDebugImage(const rclcpp::Time & stamp);
  rcl_interfaces::msg::SetParametersResult onParametersChanged(
    const std::vector<rclcpp::Parameter> & parameters);

  bool buildOccupancyGrid(
    const PointCloudXYZ & map_cloud, nav_msgs::msg::OccupancyGrid & grid,
    const rclcpp::Time & stamp) const;
  bool buildOccupancyGrid(
    const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> & scans_transformed,
    nav_msgs::msg::OccupancyGrid & grid, const rclcpp::Time & stamp) const;

  std::unique_ptr<ScanMatcherInterface> loadScanMatcher(const std::string & name);

  static geometry_msgs::msg::TransformStamped poseToTransformStamped(
    const gtsam::Pose3 & map_to_odom, const std::string & parent_frame,
    const std::string & child_frame, const rclcpp::Time & stamp);

  static sensor_msgs::msg::Image toHeatmapRosImage(
    const glidar_slam::core::CsmResult::DebugImage & debug, const std::string & frame_id,
    const rclcpp::Time & stamp);

  static bool parseGroundRoiRatios(const std::string & value, std::vector<float> & ratios);

  static visualization_msgs::msg::Marker makeGroundPlaneMarker(
    const glidar_slam::core::GroundPlaneObservation & observation, const std::string & frame,
    const rclcpp::Time & stamp, double plane_size);

  static visualization_msgs::msg::Marker makeGroundNormalMarker(
    const glidar_slam::core::GroundPlaneObservation & observation, const std::string & frame,
    const rclcpp::Time & stamp);

  static sensor_msgs::msg::Image::UniquePtr makeGroundDebugImage(
    const cv::Mat & color, const glidar_slam::core::CameraIntrinsics & intrinsics,
    const Eigen::Affine3f & base_from_camera, const std::vector<float> & roi_ratios,
    const glidar_slam::core::GroundPlaneObservation & observation, const rclcpp::Time & stamp,
    const std::string & frame);

  static visualization_msgs::msg::Marker graphEdgesToMarker(
    const std::vector<std::shared_ptr<const KeyFrame>> & keyframes,
    const std::vector<GraphEdge> & edges, const std::string & frame);

  static std::optional<visualization_msgs::msg::Marker> keyframeCovarianceToMarker(
    const KeyFrame & keyframe, const std::string & frame);

  static visualization_msgs::msg::Marker keyframeToMarker(
    const KeyFrame & keyframe, const std::string & frame);

  // Modules
  std::unique_ptr<pluginlib::ClassLoader<ScanMatcherInterface>> scan_matcher_loader_;
  std::unique_ptr<SlamSystem> slam_system_;

  // ROS2 interfaces
  // TF2
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // ROS2 callback groups
  rclcpp::CallbackGroup::SharedPtr transform_broadcast_callback_group_;
  rclcpp::CallbackGroup::SharedPtr odom_callback_group_;
  rclcpp::CallbackGroup::SharedPtr scan_callback_group_;
  rclcpp::CallbackGroup::SharedPtr camera_callback_group_;
  rclcpp::CallbackGroup::SharedPtr map_callback_group_;
  rclcpp::CallbackGroup::SharedPtr state_callback_group_;

  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
  rclcpp::Service<glidar_slam_msgs::srv::SaveSlamState>::SharedPtr save_state_service_;
  rclcpp::Service<glidar_slam_msgs::srv::LoadSlamState>::SharedPtr load_state_service_;
  rclcpp::Service<glidar_slam_msgs::srv::SetLocalizationMode>::SharedPtr
    set_localization_mode_service_;

  using GroundSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::LaserScan, sensor_msgs::msg::Image, sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;
  message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_subscriber_;
  message_filters::Subscriber<sensor_msgs::msg::Image> color_image_subscriber_;
  message_filters::Subscriber<sensor_msgs::msg::Image> aligned_depth_image_subscriber_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> color_camera_info_subscriber_;
  std::shared_ptr<message_filters::Synchronizer<GroundSyncPolicy>> ground_synchronizer_;

  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr ground_marking_grid_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ground_texture_image_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr ground_texture_coverage_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_debug_cloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_initial_debug_cloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_matching_debug_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ground_debug_marker_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ground_debug_image_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr csm_debug_low_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr csm_debug_high_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr estimated_pose_pub_;

  rclcpp::TimerBase::SharedPtr transform_broadcast_timer_;
  rclcpp::TimerBase::SharedPtr map_timer_;

  // Variables
  gtsam::Matrix66 latest_odom_covariance_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_color_msg;
  sensor_msgs::msg::Image::ConstSharedPtr latest_depth_msg;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_camera_info_msg;

  // ROS2 Parameters
  std::shared_ptr<Parameters> parameters_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
};

}  // namespace glidar_slam::ros2
