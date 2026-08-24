#pragma once

#include <optional>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "glidar_slam/core/occupancy_grid.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/system.hpp"
#include "gtsam/geometry/Pose3.h"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
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
  void scanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr & msg);

  static PointCloudXYZ laserScanToPointCloud(const sensor_msgs::msg::LaserScan & msg);
  static geometry_msgs::msg::TransformStamped poseToTransformStamped(
    const gtsam::Pose3 & map_to_odom, const std::string & parent_frame,
    const std::string & child_frame, const rclcpp::Time & stamp);
  static geometry_msgs::msg::PoseStamped poseToPoseStamped(
    const gtsam::Pose3 & pose, const std::string & frame, const rclcpp::Time & stamp);

  void publishGraph(const std::vector<std::shared_ptr<const KeyFrame>> & keyframes);
  void publishMapToOdom();
  void publishOccupancyGrid(
    const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> & scans_transformed,
    const rclcpp::Time & stamp);
  void publishDebugImage(const SlamSystem::LaserScanOutput & output, const rclcpp::Time & stamp);

  bool buildOccupancyGrid(
    const PointCloudXYZ & map_cloud, nav_msgs::msg::OccupancyGrid & grid,
    const rclcpp::Time & stamp) const;
  bool buildOccupancyGrid(
    const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> & scans_transformed,
    nav_msgs::msg::OccupancyGrid & grid, const rclcpp::Time & stamp) const;

  void processCSMParameters();

  // Modules
  std::unique_ptr<SlamSystem> slam_system_;
  std::unique_ptr<OccupancyGrid> occ_grid_;

  // ROS2 interfaces
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
  rclcpp::CallbackGroup::SharedPtr transform_broadcast_callback_group_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr csm_debug_low_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr csm_debug_high_publisher_;

  rclcpp::TimerBase::SharedPtr transform_broadcast_timer_;

  // Variables
  gtsam::Matrix66 latest_odom_covariance_;
  gtsam::Pose3 latest_map_to_base_;

  // ROS2 Parameters
  std::shared_ptr<Parameters> parameters_;
};

}  // namespace glidar_slam::ros2
