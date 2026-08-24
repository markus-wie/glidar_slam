#pragma once

#include <string>

#include "glidar_slam/core/scan_matcher.hpp"
#include "rclcpp/rclcpp.hpp"

namespace glidar_slam::ros2 {

class ScanMatcherInterface : public core::ScanMatcher
{
public:
  ~ScanMatcherInterface() override = default;

  virtual void initialize(rclcpp::Node * node, const std::string & plugin_name) = 0;
};

}  // namespace glidar_slam::ros2
