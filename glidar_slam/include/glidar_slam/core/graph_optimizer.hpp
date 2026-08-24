#pragma once

#include <cstdint>
#include <mutex>
#include <optional>

#include "glidar_slam/core/parameters.hpp"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/nonlinear/ISAM2.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"

namespace glidar_slam::core {

class GraphOptimizer
{
public:
  GraphOptimizer(const std::shared_ptr<Parameters> & params);

  void initialize(const gtsam::Pose3 & initial_pose, uint64_t timestamp);

  uint64_t addRelativeFactor(
    uint64_t from_key, uint64_t to_key, const gtsam::Pose3 & relative_pose,
    const gtsam::Matrix66 & covariance);

  gtsam::Values optimize();

  std::optional<gtsam::Matrix66> getMarginalCovariance(gtsam::Key key);

  gtsam::Pose3 getLatestPose() const;

  bool isInitialized() const;

  const gtsam::Values & getCurrentEstimates() const;

private:
  static gtsam::Pose3 projectPlanar(const gtsam::Pose3 & pose);
  static gtsam::noiseModel::Gaussian::shared_ptr covarianceFromMatrix(
    const gtsam::Matrix66 & covariance);

  std::shared_ptr<Parameters> params_;

  gtsam::ISAM2Params isam_params_;
  gtsam::ISAM2 isam_;
  gtsam::NonlinearFactorGraph pending_factors_;
  gtsam::Values pending_values_;
  gtsam::Values current_estimates_;
  uint64_t latest_key_;
  uint64_t latest_timestamp_;
  bool initialized_;

  std::mutex isam_mutex_;
};

}  // namespace glidar_slam::core
