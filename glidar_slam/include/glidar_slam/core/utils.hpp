#pragma once

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
};

}  // namespace glidar_slam::core
