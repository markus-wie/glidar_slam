#pragma once

#include <cstdint>

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
  double x, y;
};

struct Pose2D
{
  double x, y;
  double yaw;
};

using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZPtr = PointCloudXYZ::Ptr;
using PointCloudXYZConstPtr = PointCloudXYZ::ConstPtr;

}  // namespace glidar_slam::core
