#pragma once

#include <memory>
#include <string>
#include <vector>

#include "glidar_slam/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam_ros/scan_matcher_interface.hpp"

namespace glidar_slam_ros {

class CorrelativeScanMatcherPlugin final : public ScanMatcherInterface
{
public:
  void initialize(rclcpp::Node * node, const std::string & plugin_name) override;

  glidar_slam::CsmResult match(
    const glidar_slam::SubmapGrid & submap_grid,
    const std::vector<glidar_slam::Point2D> & current_points,
    const glidar_slam::Pose2D & pose_estimate) const override;

  std::vector<double> fieldResolutions() const override;

private:
  std::unique_ptr<glidar_slam::CorrelativeScanMatcher> matcher_;
};

}  // namespace glidar_slam_ros
