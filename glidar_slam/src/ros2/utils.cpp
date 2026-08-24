#include "glidar_slam/ros2/utils.hpp"

namespace glidar_slam::ros2 {
nav_msgs::msg::OccupancyGrid Utils::toRosMessage(
  const glidar_slam::core::OccupancyGrid & core_grid, const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  nav_msgs::msg::OccupancyGrid msg;
  const auto & info = core_grid.getInfo();

  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;

  msg.info.resolution = static_cast<float>(info.resolution);
  msg.info.width = info.width;
  msg.info.height = info.height;
  msg.info.origin.position.x = info.origin_x;
  msg.info.origin.position.y = info.origin_y;
  msg.info.origin.position.z = 0.0;

  // Neutral quaternion
  msg.info.origin.orientation.x = 0.0;
  msg.info.origin.orientation.y = 0.0;
  msg.info.origin.orientation.z = 0.0;
  msg.info.origin.orientation.w = 1.0;

  // Direct copy of the underlying int8_t vector
  msg.data = core_grid.getData();

  return msg;
}
}  // namespace glidar_slam::ros2