#pragma once

#include "gtsam/geometry/Pose3.h"
#include "gtsam/nonlinear/NonlinearFactor.h"

namespace glidar_slam {

class GroundPlaneFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  GroundPlaneFactor(
    gtsam::Key key, const gtsam::Vector3 & observed_normal_in_base,
    double observed_distance_to_base, const gtsam::Vector3 & reference_normal_in_map,
    double reference_plane_offset, const gtsam::SharedNoiseModel & model);

  gtsam::Vector evaluateError(
    const gtsam::Pose3 & pose, gtsam::OptionalMatrixType H = nullptr) const override;

  gtsam::Vector3 getObservedNormalInBase() const;
  double getObservedDistanceToBase() const;
  gtsam::Vector3 getReferenceNormalInMap() const;
  double getReferencePlaneOffset() const;
  double getNormalSigma() const;
  double getDistanceSigma() const;

private:
  gtsam::Vector error(const gtsam::Pose3 & pose) const;

  gtsam::Vector3 observed_normal_in_base_;
  double observed_distance_to_base_;
  gtsam::Vector3 reference_normal_in_map_;
  double reference_plane_offset_;

  double normal_sigma_{0.0};
  double distance_sigma_{0.0};

  gtsam::Vector3 tangent_x_;
  gtsam::Vector3 tangent_y_;
};

}  // namespace glidar_slam
