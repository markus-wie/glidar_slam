#pragma once

#include <string>

#include "glidar_slam/mapping/global_map.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace glidar_slam_ros {

class Utils
{
public:
  Utils() = delete;

  static nav_msgs::msg::OccupancyGrid toRosMessage(
    const glidar_slam::mapping::OccupancyGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
  static nav_msgs::msg::OccupancyGrid toRosMessage(
    const glidar_slam::mapping::GroundMarkingGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
  static sensor_msgs::msg::Image toRosImage(
    const glidar_slam::mapping::GroundTextureGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
};

}  // namespace glidar_slam_ros
