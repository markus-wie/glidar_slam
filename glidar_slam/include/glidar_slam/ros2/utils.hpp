#pragma once

#include <string>

#include "glidar_slam/core/global_map/ground_marking_grid.hpp"
#include "glidar_slam/core/global_map/ground_texture_grid.hpp"
#include "glidar_slam/core/global_map/occupancy_grid.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace glidar_slam::ros2 {

class Utils
{
public:
  Utils() = delete;

  static nav_msgs::msg::OccupancyGrid toRosMessage(
    const glidar_slam::core::global_map::OccupancyGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
  static nav_msgs::msg::OccupancyGrid toRosMessage(
    const glidar_slam::core::global_map::GroundMarkingGrid & core_grid,
    const std::string & frame_id, const rclcpp::Time & stamp);
  static sensor_msgs::msg::Image toRosImage(
    const glidar_slam::core::global_map::GroundTextureGrid & core_grid,
    const std::string & frame_id, const rclcpp::Time & stamp);
  static nav_msgs::msg::OccupancyGrid toCoverageMessage(
    const glidar_slam::core::global_map::GroundTextureGrid & core_grid,
    const std::string & frame_id, const rclcpp::Time & stamp);
};

}  // namespace glidar_slam::ros2
