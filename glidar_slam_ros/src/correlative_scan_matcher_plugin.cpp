#include "glidar_slam_ros/correlative_scan_matcher_plugin.hpp"

#include <cmath>
#include <stdexcept>

#include "pluginlib/class_list_macros.hpp"

namespace glidar_slam_ros {

void CorrelativeScanMatcherPlugin::initialize(rclcpp::Node * node, const std::string & name)
{
  if (node == nullptr) {
    throw std::invalid_argument("Scan matcher plugin node must not be null");
  }

  auto parameters = std::make_shared<glidar_slam::Parameters>();
  double unobservable_variance = node->get_parameter_or("unobservable_variance", 1e6);
  bool debug_timings = node->get_parameter_or("debug_timings", false);

  parameters->unobservable_variance =
    node->declare_parameter<double>(name + ".unobservable_variance", unobservable_variance);
  parameters->debug_timings = node->declare_parameter<bool>(name + ".debug_timings", debug_timings);
  parameters->csm_smear_deviation =
    node->declare_parameter<double>(name + ".csm_smear_deviation", 0.1);
  parameters->csm_use_laplace_kernel =
    node->declare_parameter<bool>(name + ".csm_use_laplace_kernel", false);
  parameters->csm_use_distance_transform =
    node->declare_parameter<bool>(name + ".csm_use_distance_transform", false);
  parameters->csm_use_tbb = node->declare_parameter<bool>(name + ".csm_use_tbb", false);
  parameters->csm_debug_enable = node->declare_parameter<bool>(name + ".csm_debug_enable", false);

  const auto resolutions = node->declare_parameter<std::vector<double>>(
    name + ".csm_stage_resolutions", std::vector<double>{0.05, 0.04, 0.02});
  const auto translation_steps = node->declare_parameter<std::vector<double>>(
    name + ".csm_stage_translation_steps", std::vector<double>{0.1, 0.01, 0.001});
  const auto angular_steps = node->declare_parameter<std::vector<double>>(
    name + ".csm_stage_angular_steps", std::vector<double>{0.1, 0.01, 0.001});
  const auto windows_x = node->declare_parameter<std::vector<double>>(
    name + ".csm_stage_window_x", std::vector<double>{2.0, 0.15, 0.015});
  const auto windows_y = node->declare_parameter<std::vector<double>>(
    name + ".csm_stage_window_y", std::vector<double>{2.0, 0.15, 0.015});
  const auto windows_yaw = node->declare_parameter<std::vector<double>>(
    name + ".csm_stage_window_yaw", std::vector<double>{0.5, 0.15, 0.015});

  const std::size_t count = resolutions.size();
  if (
    count == 0 || translation_steps.size() != count || angular_steps.size() != count ||
    windows_x.size() != count || windows_y.size() != count || windows_yaw.size() != count) {
    throw std::runtime_error("CSM plugin stage arrays must be non-empty and have equal lengths");
  }

  parameters->csm_search_stages.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (
      !std::isfinite(resolutions[i]) || resolutions[i] <= 0.0 ||
      !std::isfinite(translation_steps[i]) || translation_steps[i] <= 0.0 ||
      !std::isfinite(angular_steps[i]) || angular_steps[i] <= 0.0 || !std::isfinite(windows_x[i]) ||
      windows_x[i] < 0.0 || !std::isfinite(windows_y[i]) || windows_y[i] < 0.0 ||
      !std::isfinite(windows_yaw[i]) || windows_yaw[i] < 0.0) {
      throw std::runtime_error("CSM plugin stage values are outside their valid ranges");
    }
    parameters->csm_search_stages.push_back(
      {resolutions[i], translation_steps[i], angular_steps[i], windows_x[i], windows_y[i],
       windows_yaw[i]});
  }
  if (!std::isfinite(parameters->csm_smear_deviation) || parameters->csm_smear_deviation <= 0.0) {
    throw std::runtime_error("CSM plugin smear deviation must be finite and positive");
  }

  matcher_ = std::make_unique<glidar_slam::CorrelativeScanMatcher>(parameters);
}

glidar_slam::CsmResult CorrelativeScanMatcherPlugin::match(
  const glidar_slam::SubmapGrid & submap_grid,
  const std::vector<glidar_slam::Point2D> & current_points,
  const glidar_slam::Pose2D & pose_estimate) const
{
  if (!matcher_) {
    throw std::logic_error("Scan matcher plugin was not initialized");
  }
  return matcher_->match(submap_grid, current_points, pose_estimate);
}

std::vector<double> CorrelativeScanMatcherPlugin::fieldResolutions() const
{
  if (!matcher_) {
    throw std::logic_error("Scan matcher plugin was not initialized");
  }
  return matcher_->fieldResolutions();
}

}  // namespace glidar_slam_ros

PLUGINLIB_EXPORT_CLASS(
  glidar_slam_ros::CorrelativeScanMatcherPlugin, glidar_slam_ros::ScanMatcherInterface)
