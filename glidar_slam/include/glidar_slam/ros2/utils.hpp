#pragma once

#include "glidar_slam/core/occupancy_grid.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/time.hpp"

namespace glidar_slam::ros2 {

class Utils
{
public:
  Utils() = delete;

  static nav_msgs::msg::OccupancyGrid toRosMessage(
    const glidar_slam::core::OccupancyGrid & core_grid, const std::string & frame_id,
    const rclcpp::Time & stamp);
};

}  // namespace glidar_slam::ros2
