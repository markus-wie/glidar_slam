#pragma once

#include <string>

#include "glidar_slam/core/ground_marking_grid.hpp"
#include "glidar_slam/core/ground_texture_grid.hpp"
#include "glidar_slam/core/occupancy_grid.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace glidar_slam::ros2 {

class Utils
{
public:
  Utils() = delete;

  static nav_msgs::msg::OccupancyGrid toRosMessage(
    const glidar_slam::core::OccupancyGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
  static nav_msgs::msg::OccupancyGrid toRosMessage(
    const glidar_slam::core::GroundMarkingGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
  static sensor_msgs::msg::Image toRosImage(
    const glidar_slam::core::GroundTextureGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
  static nav_msgs::msg::OccupancyGrid toCoverageMessage(
    const glidar_slam::core::GroundTextureGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
};

}  // namespace glidar_slam::ros2
