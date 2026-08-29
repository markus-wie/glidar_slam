#include "glidar_slam/ros2/utils.hpp"

namespace glidar_slam::ros2 {
namespace {

void setOccupancyGridMetadata(
  nav_msgs::msg::OccupancyGrid & msg, const glidar_slam::core::mapping::GlobalMap::Info & info,
  const std::string & frame_id, const rclcpp::Time & stamp)
{
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.info.resolution = static_cast<float>(info.resolution);
  msg.info.width = info.width;
  msg.info.height = info.height;
  msg.info.origin.position.x = info.origin_x;
  msg.info.origin.position.y = info.origin_y;
  msg.info.origin.position.z = 0.0;
  msg.info.origin.orientation.w = 1.0;
}

template <typename Grid>
nav_msgs::msg::OccupancyGrid toRosMessage(
  const Grid & core_grid, const std::string & frame_id, const rclcpp::Time & stamp)
{
  nav_msgs::msg::OccupancyGrid msg;
  setOccupancyGridMetadata(msg, core_grid.getInfo(), frame_id, stamp);
  msg.data = core_grid.getData();

  return msg;
}

}  // namespace

nav_msgs::msg::OccupancyGrid Utils::toRosMessage(
  const glidar_slam::core::mapping::OccupancyGrid & core_grid, const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  return glidar_slam::ros2::toRosMessage(core_grid, frame_id, stamp);
}

nav_msgs::msg::OccupancyGrid Utils::toRosMessage(
  const glidar_slam::core::mapping::GroundMarkingGrid & core_grid, const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  return glidar_slam::ros2::toRosMessage(core_grid, frame_id, stamp);
}

sensor_msgs::msg::Image Utils::toRosImage(
  const glidar_slam::core::mapping::GroundTextureGrid & core_grid, const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  sensor_msgs::msg::Image msg;
  const auto & info = core_grid.getInfo();
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.height = info.height;
  msg.width = info.width;
  msg.encoding = "rgb8";
  msg.is_bigendian = false;
  msg.step = info.width * 3U;
  msg.data = core_grid.getRgbData();
  return msg;
}

}  // namespace glidar_slam::ros2
