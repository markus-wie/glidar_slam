#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "glidar_slam/core/ground_marking_grid.hpp"
#include "glidar_slam/core/occupancy_grid.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/system.hpp"
#include "gtsam/geometry/Pose3.h"
#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/synchronizer.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
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
using glidar_slam::core::OccupancyGrid;
using glidar_slam::core::Parameters;
using glidar_slam::core::PointCloudXYZ;
using glidar_slam::core::SlamSystem;

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

  static geometry_msgs::msg::TransformStamped poseToTransformStamped(
    const gtsam::Pose3 & map_to_odom, const std::string & parent_frame,
    const std::string & child_frame, const rclcpp::Time & stamp);
  static geometry_msgs::msg::PoseStamped poseToPoseStamped(
    const gtsam::Pose3 & pose, const std::string & frame, const rclcpp::Time & stamp);
  void publishGraph(
    const std::vector<std::shared_ptr<const KeyFrame>> & keyframes,
    const std::vector<std::pair<uint64_t, uint64_t>> & loop_closures);
  void publishMapToOdom();
  void publishOccupancyGrid(
    const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> & scans_transformed,
    const rclcpp::Time & stamp);
  void publishGroundMarkingGrid(const rclcpp::Time & stamp);
  void publishGroundDebug(
    const glidar_slam::core::GroundPlaneObservation & observation,
    const rclcpp::Time & stamp) const;
  void clearGroundDebug(const rclcpp::Time & stamp) const;
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

  void processCSMParameters();

  static sensor_msgs::msg::Image toHeatmapRosImage(
    const glidar_slam::core::CsmResult::DebugImage & debug, const std::string & frame_id,
    const rclcpp::Time & stamp);

  static PointCloudXYZ transformPointCloud(
    const PointCloudXYZ & input, const geometry_msgs::msg::TransformStamped & transform_stamped);

  static bool parseGroundRoiRatios(const std::string & value, std::vector<float> & ratios);

  static visualization_msgs::msg::Marker makeGroundPlaneMarker(
    const glidar_slam::core::GroundPlaneObservation & observation, const std::string & frame,
    const rclcpp::Time & stamp, double plane_size);

  static visualization_msgs::msg::Marker makeGroundNormalMarker(
    const glidar_slam::core::GroundPlaneObservation & observation, const std::string & frame,
    const rclcpp::Time & stamp);

  static visualization_msgs::msg::Marker makeGroundDeleteMarker(int id, const std::string & frame);

  static sensor_msgs::msg::Image::UniquePtr makeGroundDebugImage(
    const cv::Mat & color, const glidar_slam::core::CameraIntrinsics & intrinsics,
    const Eigen::Affine3f & base_from_camera, const std::vector<float> & roi_ratios,
    const glidar_slam::core::GroundPlaneObservation & observation, const rclcpp::Time & stamp,
    const std::string & frame);

  // Modules
  std::unique_ptr<SlamSystem> slam_system_;
  std::unique_ptr<OccupancyGrid> occ_grid_;
  std::unique_ptr<glidar_slam::core::GroundMarkingGrid> ground_marking_grid_;

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

  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;

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
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_debug_cloud_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ground_debug_marker_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ground_debug_image_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr csm_debug_low_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr csm_debug_high_publisher_;

  rclcpp::TimerBase::SharedPtr transform_broadcast_timer_;

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
