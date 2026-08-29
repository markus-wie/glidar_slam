#include "glidar_slam/core/ekf.hpp"

#include "Eigen/Dense"
#include "glidar_slam/core/utils.hpp"

namespace glidar_slam::core {

Pose2EKF::Pose2EKF() : initialized_(false)
{
}

void Pose2EKF::initialize(const Pose2D & initial_pose, const Eigen::Matrix3d & initial_cov)
{
  state_ = initial_pose;
  cov_ = initial_cov;
  initialized_ = true;
}

void Pose2EKF::predict(const Pose2D & odom_delta, const Eigen::Matrix3d & odom_cov)
{
  if (!initialized_) {
    return;
  }

  const double dx = odom_delta.x;
  const double dy = odom_delta.y;
  const double theta = state_.yaw;

  // predict state
  state_.x += std::cos(theta) * dx - std::sin(theta) * dy;
  state_.y += std::sin(theta) * dx + std::cos(theta) * dy;
  state_.yaw = utils::normalizeAngle(state_.yaw + odom_delta.yaw);

  // predict covariance
  // calculate the jacobian of the motion model with respect to the current state
  Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
  F(0, 2) = -dx * std::sin(theta) - dy * std::cos(theta);
  F(1, 2) = dx * std::cos(theta) - dy * std::sin(theta);

  // calculate the jacobian of the motion model with respect to the control input
  Eigen::Matrix3d V = Eigen::Matrix3d::Identity();
  V(0, 0) = std::cos(theta);
  V(0, 1) = -std::sin(theta);
  V(1, 0) = std::sin(theta);
  V(1, 1) = std::cos(theta);

  // covariance prediction: P = F*P*F^T + V*R*V^T
  cov_ = F * cov_ * F.transpose() + V * odom_cov * V.transpose();
}

void Pose2EKF::update(const Pose2D & measurement, const Eigen::Matrix3d & measurement_cov)
{
  if (!initialized_) {
    return;
  }

  // because our measurement is a direct observation of the state, the observation matrix H is the
  // identity

  Eigen::Vector3d y;
  y(0) = measurement.x - state_.x;
  y(1) = measurement.y - state_.y;
  y(2) = utils::normalizeAngle(measurement.yaw - state_.yaw);

  Eigen::Matrix3d S = cov_ + measurement_cov;
  Eigen::Matrix3d K = cov_ * S.inverse();

  Eigen::Vector3d correction = K * y;
  state_.x += correction(0);
  state_.y += correction(1);
  state_.yaw = utils::normalizeAngle(state_.yaw + correction(2));

  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  cov_ = (I - K) * cov_;

  cov_ = 0.5 * (cov_ + cov_.transpose());
}

Pose2D Pose2EKF::getState() const
{
  return state_;
}

Eigen::Matrix3d Pose2EKF::getCovariance() const
{
  return cov_;
}

bool Pose2EKF::isInitialized() const
{
  return initialized_;
}

}  // namespace glidar_slam::core
