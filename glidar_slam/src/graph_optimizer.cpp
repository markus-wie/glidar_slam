#include "glidar_slam/graph_optimizer.hpp"

#include <algorithm>
#include <cmath>

#include "Eigen/Eigenvalues"
#include "glidar_slam/logger/logger.hpp"
#include "gtsam/inference/Symbol.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"

namespace glidar_slam {

namespace {

gtsam::noiseModel::Gaussian::shared_ptr makeCovarianceNoiseModel(
  const gtsam::Matrix66 & covariance, double unobservable_variance)
{
  constexpr double minimum_variance = 1e-8;

  if (!covariance.allFinite()) {
    return gtsam::noiseModel::Gaussian::Covariance(
      gtsam::Matrix66::Identity() * unobservable_variance);
  }

  gtsam::Matrix66 symmetric = 0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<gtsam::Matrix66> solver(symmetric);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return gtsam::noiseModel::Gaussian::Covariance(
      gtsam::Matrix66::Identity() * unobservable_variance);
  }
  if (solver.eigenvalues().minCoeff() < -minimum_variance) {
    return gtsam::noiseModel::Gaussian::Covariance(
      gtsam::Matrix66::Identity() * unobservable_variance);
  }

  const gtsam::Matrix66 sanitized = solver.eigenvectors() *
                                    solver.eigenvalues().cwiseMax(minimum_variance).asDiagonal() *
                                    solver.eigenvectors().transpose();
  return gtsam::noiseModel::Gaussian::Covariance(sanitized);
}

}  // namespace

GraphOptimizer::GraphOptimizer(const std::shared_ptr<Parameters> & params)
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

void GraphOptimizer::addPriorFactor(uint64_t key, const gtsam::Pose3 & pose)
{
  const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());
  pending_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(key, pose, prior_noise));
  pending_values_.insert(key, pose);

  latest_key_ = key;
}

void GraphOptimizer::addPriorFactor(
  uint64_t key, const gtsam::Pose3 & pose, const gtsam::Matrix66 & covariance)
{
  pending_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(
    key, pose, makeCovarianceNoiseModel(covariance, params_->unobservable_variance)));
  pending_values_.insert(key, pose);
  latest_key_ = key;
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
    from_key, to_key, relative_pose,
    makeCovarianceNoiseModel(covariance, params_->unobservable_variance)));

  if (!pending_values_.exists(to_key) && !current_estimates_.exists(to_key)) {
    const gtsam::Pose3 from_pose = current_estimates_.exists(from_key)
                                     ? current_estimates_.at<gtsam::Pose3>(from_key)
                                     : getLatestPose();
    const gtsam::Pose3 to_pose = from_pose.compose(relative_pose);
    pending_values_.insert(to_key, to_pose);
  }

  latest_key_ = to_key;
  return to_key;
}

void GraphOptimizer::addGroundPlaneFactor(
  uint64_t key, const gtsam::Vector3 & observed_normal_in_base, double observed_distance_to_base,
  const gtsam::Vector3 & reference_normal_in_map, double reference_plane_offset,
  double normal_sigma, double distance_sigma)
{
  const auto noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(3) << normal_sigma, normal_sigma, distance_sigma).finished());

  pending_factors_.add(std::make_shared<GroundPlaneFactor>(
    key, observed_normal_in_base, observed_distance_to_base, reference_normal_in_map,
    reference_plane_offset, noise));
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

std::vector<SerializedFactor> GraphOptimizer::getSerializedFactors() const
{
  std::lock_guard<std::mutex> lock(isam_mutex_);

  std::vector<SerializedFactor> serialized_factors;

  const gtsam::NonlinearFactorGraph & graph = isam_.getFactorsUnsafe();

  for (const auto & factor : graph) {
    if (!factor) {
      continue;
    }

    SerializedFactor sf;

    if (auto prior_factor = std::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Pose3>>(factor)) {
      sf.type = SerializedFactorType::PriorPose;
      sf.to_key = prior_factor->key();
      sf.relative_pose = prior_factor->prior();

      auto gaussian =
        std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(prior_factor->noiseModel());
      sf.covariance = gaussian ? gaussian->covariance() : gtsam::Matrix66::Zero();

    } else if (
      auto between_factor = std::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor)) {
      sf.type = SerializedFactorType::RelativePose;
      sf.from_key = between_factor->front();
      sf.to_key = between_factor->back();
      sf.relative_pose = between_factor->measured();

      auto gaussian =
        std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(between_factor->noiseModel());
      sf.covariance = gaussian ? gaussian->covariance() : gtsam::Matrix66::Zero();

    } else if (auto ground_factor = std::dynamic_pointer_cast<GroundPlaneFactor>(factor)) {
      sf.type = SerializedFactorType::GroundPlane;
      sf.to_key = ground_factor->key();
      sf.observed_normal_in_base = ground_factor->getObservedNormalInBase();
      sf.observed_distance_to_base = ground_factor->getObservedDistanceToBase();
      sf.reference_normal_in_map = ground_factor->getReferenceNormalInMap();
      sf.reference_plane_offset = ground_factor->getReferencePlaneOffset();
      sf.normal_sigma = ground_factor->getNormalSigma();
      sf.distance_sigma = ground_factor->getDistanceSigma();
    }

    serialized_factors.push_back(sf);
  }

  return serialized_factors;
}

uint64_t GraphOptimizer::getLatestKey() const
{
  std::unique_lock lock(isam_mutex_);
  return latest_key_;
}

uint64_t GraphOptimizer::getLatestTimestamp() const
{
  std::unique_lock lock(isam_mutex_);
  return latest_timestamp_;
}

bool GraphOptimizer::restore(const SlamStateSnapshot & snapshot)
{
  pending_factors_.resize(0);
  pending_values_.clear();
  current_estimates_.clear();
  isam_ = gtsam::ISAM2(isam_params_);
  latest_key_ = 0;
  latest_timestamp_ = snapshot.optimizer_latest_timestamp;
  initialized_ = snapshot.optimizer_initialized;

  if (!initialized_) {
    return snapshot.factors.empty();
  }

  for (const auto & factor : snapshot.factors) {
    if (factor.type == SerializedFactorType::PriorPose) {
      const auto prior_noise =
        makeCovarianceNoiseModel(factor.covariance, params_->unobservable_variance);
      pending_factors_.add(
        gtsam::PriorFactor<gtsam::Pose3>(factor.to_key, factor.relative_pose, prior_noise));
      pending_values_.insert(factor.to_key, factor.relative_pose);
      latest_key_ = factor.to_key;
      continue;
    }
    if (factor.type == SerializedFactorType::RelativePose) {
      addRelativeFactor(factor.from_key, factor.to_key, factor.relative_pose, factor.covariance);
    } else if (factor.type == SerializedFactorType::GroundPlane) {
      addGroundPlaneFactor(
        factor.to_key, factor.observed_normal_in_base, factor.observed_distance_to_base,
        factor.reference_normal_in_map, factor.reference_plane_offset, factor.normal_sigma,
        factor.distance_sigma);
    } else {
      return false;
    }
  }

  // Seed iSAM2 with the saved optimized poses. Replaying the graph alone can choose different
  // linearization points after loading and then move restored keyframes on the first update.
  for (const auto & kf : snapshot.keyframes) {
    const auto key = static_cast<gtsam::Key>(kf.key);
    if (pending_values_.exists(key)) {
      pending_values_.update(key, kf.pose);
    } else {
      pending_values_.insert(key, kf.pose);
    }
  }

  optimize();
  latest_key_ = snapshot.optimizer_latest_key;
  latest_timestamp_ = snapshot.optimizer_latest_timestamp;
  return true;
}

}  // namespace glidar_slam
