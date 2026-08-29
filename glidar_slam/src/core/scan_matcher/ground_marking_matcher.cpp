#include "glidar_slam/core/scan_matcher/ground_marking_matcher.hpp"

#include <algorithm>
#include <cmath>

#include "glidar_slam/core/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam/logger/logger.hpp"

namespace glidar_slam::core {

GroundMarkingMatcher::GroundMarkingMatcher(
  const std::shared_ptr<Parameters> & parameters, std::unique_ptr<ScanMatcher> scan_matcher)
: parameters_(parameters), scan_matcher_(std::move(scan_matcher))
{
}

std::vector<Point2D> GroundMarkingMatcher::extractMarkingPoints(
  const PointCloudXYZRGBAConstPtr & cloud) const
{
  const int max_depth_sq =
    parameters_->ground_matching_max_distance * parameters_->ground_matching_max_distance;

  std::vector<Point2D> points;
  points.reserve(cloud->size());
  for (const auto & point : *cloud) {
    if (point.a < parameters_->mapping_threshold) {
      continue;
    }

    if ((point.x * point.x + point.y * point.y + point.z * point.z) > max_depth_sq) {
      continue;
    }

    points.push_back({point.x, point.y});
  }
  return points;
}

PointCloudXYZRGBAPtr GroundMarkingMatcher::toPCL(const std::vector<Point2D> & points) const
{
  PointCloudXYZRGBAPtr cloud = std::make_shared<PointCloudXYZRGBA>();
  cloud->reserve(points.size());
  for (const auto & point : points) {
    pcl::PointXYZRGBA pcl_point;
    pcl_point.x = static_cast<float>(point.x);
    pcl_point.y = static_cast<float>(point.y);
    pcl_point.z = 0.0F;
    pcl_point.r = 255;
    pcl_point.g = 255;
    pcl_point.b = 255;
    pcl_point.a = 255;
    cloud->push_back(pcl_point);
  }
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

PointCloudXYZRGBAPtr GroundMarkingMatcher::makeDebugCloud(
  const GroundPlaneObservation & reference_observation,
  const GroundPlaneObservation & current_observation, const CsmResult & result,
  const Pose2D & relative_pose) const
{
  const auto reference_points = extractMarkingPoints(reference_observation.ground_cloud);
  const auto current_points = extractMarkingPoints(current_observation.ground_cloud);
  return makeDebugCloud(reference_points, current_points, result, relative_pose);
}

PointCloudXYZRGBAPtr GroundMarkingMatcher::makeDebugCloud(
  const std::vector<Point2D> & reference_points, const std::vector<Point2D> & current_points,
  const CsmResult & result, const Pose2D & relative_pose) const
{
  const std::vector<Point2D> filtered_current_points =
    filterCurrentPoints(reference_points, current_points, relative_pose);

  PointCloudXYZRGBAPtr debug_cloud = std::make_shared<PointCloudXYZRGBA>();
  debug_cloud->reserve(reference_points.size() + 2 * filtered_current_points.size());
  for (const auto & point : reference_points) {
    pcl::PointXYZRGBA debug_point;
    debug_point.x = static_cast<float>(point.x);
    debug_point.y = static_cast<float>(point.y);
    debug_point.z = 0.0F;
    debug_point.r = 255;
    debug_point.g = 60;
    debug_point.b = 60;
    debug_point.a = 255;
    debug_cloud->push_back(debug_point);
  }

  const double cosine = std::cos(result.optimized_pose.yaw);
  const double sine = std::sin(result.optimized_pose.yaw);
  for (const auto & point : filtered_current_points) {
    pcl::PointXYZRGBA debug_point;
    debug_point.x = static_cast<float>(result.optimized_pose.x + cosine * point.x - sine * point.y);
    debug_point.y = static_cast<float>(result.optimized_pose.y + sine * point.x + cosine * point.y);
    debug_point.z = 0.0F;
    debug_point.r = 60;
    debug_point.g = 255;
    debug_point.b = 60;
    debug_point.a = 255;
    debug_cloud->push_back(debug_point);
  }

  const double odometry_cosine = std::cos(relative_pose.yaw);
  const double odometry_sine = std::sin(relative_pose.yaw);
  for (const auto & point : filtered_current_points) {
    pcl::PointXYZRGBA debug_point;
    debug_point.x =
      static_cast<float>(relative_pose.x + odometry_cosine * point.x - odometry_sine * point.y);
    debug_point.y =
      static_cast<float>(relative_pose.y + odometry_sine * point.x + odometry_cosine * point.y);
    debug_point.z = 0.0F;
    debug_point.r = 60;
    debug_point.g = 60;
    debug_point.b = 255;
    debug_point.a = 255;
    debug_cloud->push_back(debug_point);
  }
  debug_cloud->width = static_cast<std::uint32_t>(debug_cloud->size());
  debug_cloud->height = 1;
  debug_cloud->is_dense = true;
  return debug_cloud;
}

std::optional<CsmResult> GroundMarkingMatcher::match(
  const GroundPlaneObservation & reference_observation,
  const GroundPlaneObservation & current_observation, const Pose2D & relative_pose) const
{
  const std::vector<Point2D> reference_points =
    extractMarkingPoints(reference_observation.ground_cloud);
  const std::vector<Point2D> current_points =
    extractMarkingPoints(current_observation.ground_cloud);

  const std::vector<Point2D> filtered_current_points =
    filterCurrentPoints(reference_points, current_points, relative_pose);

  std::vector<double> resolutions;
  resolutions = scan_matcher_->fieldResolutions();
  SubmapGrid submap_grid(resolutions);
  submap_grid.add(reference_points, 0);
  return scan_matcher_->match(submap_grid, filtered_current_points, relative_pose);
}

std::vector<Point2D> GroundMarkingMatcher::filterCurrentPoints(
  const std::vector<Point2D> & reference_points, const std::vector<Point2D> & current_points,
  const Pose2D & relative_pose) const
{
  // arbitrarily chosen threshold under which matching should not be attempted
  constexpr std::size_t MINIMUM_MARKING_COUNT = 25;

  if (
    reference_points.size() < MINIMUM_MARKING_COUNT ||
    current_points.size() < MINIMUM_MARKING_COUNT) {
    return {};
  }

  // Calculate reference bounding box
  double min_x = reference_points.front().x;
  double max_x = reference_points.front().x;
  double min_y = reference_points.front().y;
  double max_y = reference_points.front().y;

  for (const auto & p : reference_points) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }

  // Expand by tolerance
  const double tolerance = 0.0;

  min_x -= tolerance;
  max_x += tolerance;
  min_y -= tolerance;
  max_y += tolerance;

  // Filter current points using the relative pose estimate
  const double c = std::cos(relative_pose.yaw);
  const double s = std::sin(relative_pose.yaw);

  std::vector<Point2D> filtered_current_points;
  filtered_current_points.reserve(current_points.size());

  for (const auto & p : current_points) {
    // Project current point into reference frame
    const double ref_x = relative_pose.x + c * p.x - s * p.y;
    const double ref_y = relative_pose.y + s * p.x + c * p.y;

    if (ref_x >= min_x && ref_x <= max_x && ref_y >= min_y && ref_y <= max_y) {
      filtered_current_points.push_back(p);  // Keep the original local point
    }
  }

  // Ensure we still have enough points after cropping
  if (filtered_current_points.size() < MINIMUM_MARKING_COUNT) {
    return {};
  }

  return filtered_current_points;
}

}  // namespace glidar_slam::core
