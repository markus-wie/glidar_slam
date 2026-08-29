#pragma once

#include <vector>

#include "glidar_slam/core/types.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam::core {

class Utils
{
public:
  Utils() = delete;

  static gtsam::Pose3 makePlanarPose(double x, double y, double yaw)
  {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, yaw), gtsam::Point3(x, y, 0.0));
  }

  static double normalizeAngle(double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  static Pose2D toPose2D(const gtsam::Pose3 & pose)
  {
    return {pose.x(), pose.y(), pose.rotation().yaw()};
  }

  static gtsam::Pose3 toPose3(const Pose2D & pose2d)
  {
    return makePlanarPose(pose2d.x, pose2d.y, pose2d.yaw);
  }

  static bool DoubleEqual(double a, double b)
  {
    constexpr double TOLERANCE = 1e-06;
    double delta = a - b;
    return delta < 0.0 ? delta >= -TOLERANCE : delta <= TOLERANCE;
  }

  static std::vector<Point2D> transformScanPoints(
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

  static PointCloudXYZ transformScanPoints(const PointCloudXYZ & scan, const gtsam::Pose3 & pose)
  {
    PointCloudXYZ points;
    points.reserve(scan.size());

    points.header = scan.header;

    const Eigen::Matrix4f T = pose.matrix().cast<float>();

    for (const auto & point : scan) {
      const Eigen::Vector4f world_point = T * point.getVector4fMap();

      if (
        std::isfinite(world_point.x()) && std::isfinite(world_point.y()) &&
        std::isfinite(world_point.z())) {
        points.push_back(pcl::PointXYZ(world_point.x(), world_point.y(), world_point.z()));
      }
    }

    return points;
  }
};

}  // namespace glidar_slam::core
