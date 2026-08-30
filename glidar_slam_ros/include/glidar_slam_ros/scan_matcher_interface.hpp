#pragma once

#include <string>

#include "glidar_slam/scan_matcher.hpp"
#include "rclcpp/rclcpp.hpp"

namespace glidar_slam_ros {

class ScanMatcherInterface : public glidar_slam::ScanMatcher
{
public:
  ~ScanMatcherInterface() override = default;

  virtual void initialize(rclcpp::Node * node, const std::string & plugin_name) = 0;
};

}  // namespace glidar_slam_ros
