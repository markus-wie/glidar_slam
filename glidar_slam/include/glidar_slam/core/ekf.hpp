#pragma once

#include <gtsam/geometry/Pose2.h>

#include <Eigen/Dense>

#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core {

class Pose2EKF
{
public:
  Pose2EKF();

  void initialize(const Pose2D & initial_pose, const Eigen::Matrix3d & initial_cov);

  void predict(const Pose2D & odom_delta, const Eigen::Matrix3d & odom_cov);

  void update(const Pose2D & measurement, const Eigen::Matrix3d & measurement_cov);

  Pose2D getState() const;
  Eigen::Matrix3d getCovariance() const;
  bool isInitialized() const;

private:
  bool initialized_;
  Pose2D state_;
  Eigen::Matrix3d cov_;
};

}  // namespace glidar_slam::core
