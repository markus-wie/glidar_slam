#pragma once

#include <functional>

#include "gtsam/geometry/Pose3.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::core {

struct TimedPose
{
  double timestamp{0.0};
  gtsam::Pose3 pose;
};

struct Point2D
{
  double x{0.0};
  double y{0.0};
};

struct Point3D
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct PoseEstimate
{
  gtsam::Pose3 pose;
  gtsam::Matrix66 covariance;
};

using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using PointCloudXYZConstPtr = pcl::PointCloud<pcl::PointXYZ>::ConstPtr;
using PointCloudXYZRGBA = pcl::PointCloud<pcl::PointXYZRGBA>;
using PointCloudXYZRGBAPtr = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr;
using PointCloudXYZRGBAConstPtr = pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr;

}  // namespace glidar_slam::core
