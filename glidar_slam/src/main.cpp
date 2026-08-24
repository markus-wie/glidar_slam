#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "glidar_slam/ros2/ros2_slam_wrapper.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<glidar_slam::ros2::Ros2SlamWrapper>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
