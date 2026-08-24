#pragma once

#include <memory>
#include <string>
#include <vector>

#include "glidar_slam/core/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam/ros2/scan_matcher_interface.hpp"

namespace glidar_slam::ros2 {

class CorrelativeScanMatcherPlugin final : public ScanMatcherInterface
{
public:
  void initialize(rclcpp::Node * node, const std::string & plugin_name) override;

  core::CsmResult match(
    const core::SubmapGrid & submap_grid, const std::vector<core::Point2D> & current_points,
    const core::Pose2D & pose_estimate) const override;

  std::vector<double> fieldResolutions() const override;

private:
  std::unique_ptr<core::CorrelativeScanMatcher> matcher_;
};

}  // namespace glidar_slam::ros2
