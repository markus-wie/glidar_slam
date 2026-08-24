#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "glidar_slam/core/correlative_scan_matcher.hpp"
#include "glidar_slam/core/ground_plane_extractor.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::core {

class GroundMarkingMatcher
{
public:
  explicit GroundMarkingMatcher(const std::shared_ptr<Parameters> & parameters);

  std::vector<Point2D> extractMarkingPoints(const pcl::PointCloud<pcl::PointXYZRGBA> & cloud) const;
  pcl::PointCloud<pcl::PointXYZRGBA> toPCL(const std::vector<Point2D> & points) const;

  PointCloudXYZRGBA makeDebugCloud(
    const GroundPlaneObservation & reference_observation,
    const GroundPlaneObservation & current_observation, const CsmResult & result,
    const Pose2D & relative_pose) const;

  PointCloudXYZRGBA makeDebugCloud(
    const std::vector<Point2D> & reference_points, const std::vector<Point2D> & current_points,
    const CsmResult & result, const Pose2D & relative_pose) const;

  std::optional<CsmResult> match(
    const GroundPlaneObservation & reference_observation,
    const GroundPlaneObservation & current_observation, const Pose2D & relative_pose) const;

private:
  std::vector<Point2D> filterCurrentPoints(
    const std::vector<Point2D> & reference_points, const std::vector<Point2D> & current_points,
    const Pose2D & relative_pose) const;

  std::shared_ptr<Parameters> parameters_;
  std::shared_ptr<Parameters> scan_matcher_parameters_;
  std::unique_ptr<CorrelativeScanMatcher> scan_matcher_;
};

}  // namespace glidar_slam::core
