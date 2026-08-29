#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "glidar_slam/ground_plane_extractor.hpp"
#include "glidar_slam/parameters.hpp"
#include "glidar_slam/scan_matcher.hpp"
#include "glidar_slam/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam/types.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam {

class GroundMarkingMatcher
{
public:
  GroundMarkingMatcher(
    const std::shared_ptr<Parameters> & parameters, std::unique_ptr<ScanMatcher> scan_matcher);

  std::vector<Point2D> extractMarkingPoints(const PointCloudXYZRGBAConstPtr & cloud) const;
  PointCloudXYZRGBAPtr toPCL(const std::vector<Point2D> & points) const;

  PointCloudXYZRGBAPtr makeDebugCloud(
    const GroundPlaneObservation & reference_observation,
    const GroundPlaneObservation & current_observation, const CsmResult & result,
    const Pose2D & relative_pose) const;

  PointCloudXYZRGBAPtr makeDebugCloud(
    const std::vector<Point2D> & reference_points, const std::vector<Point2D> & current_points,
    const CsmResult & result, const Pose2D & relative_pose) const;

  std::optional<CsmResult> match(
    const GroundPlaneObservation & reference_observation,
    const GroundPlaneObservation & current_observation, const Pose2D & relative_pose) const;

  std::optional<CsmResult> match(
    const SubmapGrid & submap, const GroundPlaneObservation & current_observation,
    const Pose2D & pose_estimate) const;

  std::vector<double> fieldResolutions() const;

private:
  std::vector<Point2D> filterCurrentPoints(
    const std::vector<Point2D> & reference_points, const std::vector<Point2D> & current_points,
    const Pose2D & relative_pose) const;

  std::shared_ptr<Parameters> parameters_;
  std::unique_ptr<ScanMatcher> scan_matcher_;
};

}  // namespace glidar_slam
