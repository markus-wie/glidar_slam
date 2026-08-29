#include "glidar_slam/utils.hpp"

#include <vector>

#include "glidar_slam/types.hpp"
#include "gtsam/geometry/Pose3.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

// #include "pcl/common/transforms.h"

namespace glidar_slam::utils {

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

std::vector<Point2D> voxelize(const std::vector<Point2D> & input, double voxel_size)
{
  if (input.empty()) {
    return {};
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->reserve(input.size());
  for (const auto & p : input) {
    cloud->push_back(pcl::PointXYZ(p.x, p.y, 0.0f));
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr voxelized_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(
    static_cast<float>(voxel_size), static_cast<float>(voxel_size), static_cast<float>(voxel_size));
  voxel_filter.filter(*voxelized_cloud);

  std::vector<Point2D> output;
  output.reserve(voxelized_cloud->size());
  for (const auto & p : *voxelized_cloud) {
    output.push_back({p.x, p.y});
  }
  return output;
}

std::vector<Point2D> densify(
  const std::vector<Point2D> & input, double target_spacing, double max_gap)
{
  if (input.empty() || target_spacing <= 0.0) {
    return input;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->reserve(input.size());
  for (const auto & p : input) {
    cloud->push_back(pcl::PointXYZ(p.x, p.y, 0.0f));
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud);

  std::vector<Point2D> output = input;  // Keep original points

  for (size_t i = 0; i < cloud->size(); ++i) {
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    // Find neighbors within the max gap
    if (
      kdtree.radiusSearch((*cloud)[i], max_gap, pointIdxRadiusSearch, pointRadiusSquaredDistance) >
      0) {
      for (size_t j = 1; j < pointIdxRadiusSearch.size(); ++j) {  // Start at 1 to skip self
        int neighbor_idx = pointIdxRadiusSearch[j];

        // Only process pairs once (j > i) to prevent duplicate interpolation
        if (neighbor_idx < static_cast<int>(i)) {
          continue;
        }

        double dist = std::sqrt(pointRadiusSquaredDistance[j]);

        // Only densify if the gap is larger than our target spacing
        if (dist > target_spacing) {
          int num_inserts = static_cast<int>(dist / target_spacing);
          double dx = (input[neighbor_idx].x - input[i].x) / (num_inserts + 1);
          double dy = (input[neighbor_idx].y - input[i].y) / (num_inserts + 1);

          for (int k = 1; k <= num_inserts; ++k) {
            output.push_back({input[i].x + k * dx, input[i].y + k * dy});
          }
        }
      }
    }
  }

  return output;
}

}  // namespace glidar_slam::utils
