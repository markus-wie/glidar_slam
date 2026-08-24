#include "glidar_slam/core/graph_optimizer.hpp"

#include <algorithm>
#include <cmath>

#include "glidar_slam/logger/logger.hpp"
#include "gtsam/inference/Symbol.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"

namespace glidar_slam::core {

GraphOptimizer::GraphOptimizer(std::shared_ptr<Parameters> params)
: params_(params), isam_(isam_params_), latest_key_(0), latest_timestamp_(0), initialized_(false)
{
  isam_params_.relinearizeThreshold = params->isam_relinearizeThreshold;
  isam_params_.relinearizeSkip = params->isam_relinearizeSkip;
  isam_ = gtsam::ISAM2(isam_params_);
}

void GraphOptimizer::initialize(const gtsam::Pose3 & initial_pose, uint64_t timestamp)
{
  pending_factors_.resize(0);
  pending_values_.clear();
  current_estimates_.clear();
  isam_ = gtsam::ISAM2(isam_params_);
  latest_key_ = 0;
  latest_timestamp_ = timestamp;
  initialized_ = true;

  auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());

  pending_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(0, initial_pose, prior_noise));
  pending_values_.insert(0, initial_pose);
}

uint64_t GraphOptimizer::addRelativeFactor(
  uint64_t from_key, uint64_t to_key, const gtsam::Pose3 & relative_pose,
  const gtsam::Matrix66 & covariance)
{
  if (!initialized_) {
    SAM_WARN("Optimizer not initialized. Discarding factor.");
    initialize(gtsam::Pose3(), 0);
  }

  pending_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
    from_key, to_key, relative_pose, covarianceFromMatrix(covariance)));
  if (!pending_values_.exists(to_key)) {
    const gtsam::Pose3 from_pose = current_estimates_.exists(from_key)
                                     ? current_estimates_.at<gtsam::Pose3>(from_key)
                                     : getLatestPose();
    const gtsam::Pose3 to_pose = from_pose.compose(relative_pose);
    pending_values_.insert(to_key, to_pose);
  }

  latest_key_ = to_key;
  return to_key;
}

gtsam::Values GraphOptimizer::optimize()
{
  std::lock_guard<std::mutex> lock(isam_mutex_);

  if (!initialized_) {
    return current_estimates_;
  }

  if (!pending_factors_.empty() || !pending_values_.empty()) {
    isam_.update(pending_factors_, pending_values_);
    pending_factors_.resize(0);
    pending_values_.clear();

    current_estimates_ = isam_.calculateEstimate();
  }

  return current_estimates_;
}

std::optional<gtsam::Matrix66> GraphOptimizer::getMarginalCovariance(gtsam::Key key)
{
  std::lock_guard<std::mutex> lock(isam_mutex_);

  try {
    return isam_.marginalCovariance(key);
  } catch (const gtsam::ValuesKeyDoesNotExist & e) {
    return std::nullopt;
  }
}

gtsam::Pose3 GraphOptimizer::getLatestPose() const
{
  if (!initialized_ || !current_estimates_.exists(latest_key_)) {
    return gtsam::Pose3();
  }

  return current_estimates_.at<gtsam::Pose3>(latest_key_);
}

bool GraphOptimizer::isInitialized() const
{
  return initialized_;
}

const gtsam::Values & GraphOptimizer::getCurrentEstimates() const
{
  return current_estimates_;
}

gtsam::Pose3 GraphOptimizer::projectPlanar(const gtsam::Pose3 & pose)
{
  const gtsam::Vector3 rpy = pose.rotation().rpy();
  const double yaw = rpy.z();
  const auto & translation = pose.translation();
  auto planar_pose = gtsam::Pose3(
    gtsam::Rot3::RzRyRx(0.0, 0.0, yaw), gtsam::Point3(translation.x(), translation.y(), 0.0));
  return planar_pose;
}

gtsam::noiseModel::Gaussian::shared_ptr GraphOptimizer::covarianceFromMatrix(
  const gtsam::Matrix66 & covariance)
{
  gtsam::Matrix66 sanitized = covariance;
  for (int i = 0; i < 6; ++i) {
    sanitized(i, i) = std::max(sanitized(i, i), 1e-8);
  }
  return gtsam::noiseModel::Gaussian::Covariance(sanitized);
}

}  // namespace glidar_slam::core
