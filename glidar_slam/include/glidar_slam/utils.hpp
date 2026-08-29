#pragma once

#include <vector>

#include "glidar_slam/types.hpp"
#include "gtsam/geometry/Pose3.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::utils {

gtsam::Pose3 makePlanarPose(double x, double y, double yaw);

double normalizeAngle(double angle);

Pose2D toPose2D(const gtsam::Pose3 & pose);

gtsam::Pose3 toPose3(const Pose2D & pose2d);

gtsam::Pose3 toPose3(const std::vector<double> & pose);

bool doubleEqual(double a, double b);

std::vector<Point2D> transformScanPoints(
  const std::vector<Point2D> & scan, const gtsam::Pose3 & pose);

PointCloudXYZPtr transformScanPoints(const PointCloudXYZConstPtr & scan, const gtsam::Pose3 & pose);

PointCloudXYZPtr voxelize(const PointCloudXYZConstPtr & pcl, double voxel_size);

float probFromLogOdds(float log_odds);
float logOddsFromProb(float prob);

std::vector<Point2D> voxelize(const std::vector<Point2D> & input, double voxel_size);

std::vector<Point2D> densify(
  const std::vector<Point2D> & input, double target_spacing, double max_gap);

}  // namespace glidar_slam::utils
