#include "glidar_slam/core/utils.hpp"

#include <vector>

#include "glidar_slam/core/types.hpp"
#include "gtsam/geometry/Pose3.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

// #include "pcl/common/transforms.h"

namespace glidar_slam::core::utils {

gtsam::Pose3 makePlanarPose(double x, double y, double yaw)
{
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, yaw), gtsam::Point3(x, y, 0.0));
}

double normalizeAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

Pose2D toPose2D(const gtsam::Pose3 & pose)
{
  return {pose.x(), pose.y(), pose.rotation().yaw()};
}

gtsam::Pose3 toPose3(const Pose2D & pose2d)
{
  return makePlanarPose(pose2d.x, pose2d.y, pose2d.yaw);
}

gtsam::Pose3 toPose3(const std::vector<double> & pose)
{
  return gtsam::Pose3(gtsam::Rot3::Yaw(pose[2]), gtsam::Point3(pose[0], pose[1], 0.0));
}

bool doubleEqual(double a, double b)
{
  constexpr double TOLERANCE = 1e-06;
  double delta = a - b;
  return delta < 0.0 ? delta >= -TOLERANCE : delta <= TOLERANCE;
}

std::vector<Point2D> transformScanPoints(
  const std::vector<Point2D> & scan, const gtsam::Pose3 & pose)
{
  std::vector<Point2D> points;
  points.reserve(scan.size());

  const Eigen::Matrix4f T = pose.matrix().cast<float>();

  for (const auto & point : scan) {
    const Eigen::Vector4f world_point = T * Eigen::Vector4f(point.x, point.y, 0.0f, 1.0f);

    if (std::isfinite(world_point.x()) && std::isfinite(world_point.y())) {
      points.push_back({world_point.x(), world_point.y()});
    }
  }

  return points;
}

PointCloudXYZPtr transformScanPoints(const PointCloudXYZConstPtr & scan, const gtsam::Pose3 & pose)
{
  PointCloudXYZPtr points = std::make_shared<PointCloudXYZ>();
  points->reserve(scan->size());
  points->header = scan->header;

  const Eigen::Matrix4f T = pose.matrix().cast<float>();

  for (const auto & point : *scan) {
    const Eigen::Vector4f world_point = T * point.getVector4fMap();

    if (
      std::isfinite(world_point.x()) && std::isfinite(world_point.y()) &&
      std::isfinite(world_point.z())) {
      points->push_back(pcl::PointXYZ(world_point.x(), world_point.y(), world_point.z()));
    }
  }

  return points;
}

PointCloudXYZPtr voxelize(const PointCloudXYZConstPtr & pcl, double voxel_size)
{
  PointCloudXYZPtr voxelized_cloud = std::make_shared<PointCloudXYZ>();
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(pcl);
  voxel_filter.setLeafSize(
    static_cast<float>(voxel_size), static_cast<float>(voxel_size), static_cast<float>(voxel_size));
  voxel_filter.filter(*voxelized_cloud);

  return voxelized_cloud;
}

float probFromLogOdds(float log_odds)
{
  return 1.0f - 1.0f / (1.0f + std::exp(log_odds));
}

float logOddsFromProb(float prob)
{
  return std::log(prob / (1.0f - prob));
}

}  // namespace glidar_slam::core::utils
