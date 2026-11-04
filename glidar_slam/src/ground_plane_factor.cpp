#include "glidar_slam/ground_plane_factor.hpp"

#include <cmath>

#include "gtsam/base/numericalDerivative.h"

namespace glidar_slam {

GroundPlaneFactor::GroundPlaneFactor(
  gtsam::Key key, const gtsam::Vector3 & observed_normal_in_base,
  const gtsam::Vector3 & observed_centroid_in_base, const gtsam::Vector3 & reference_normal_in_map,
  double reference_plane_offset, const gtsam::SharedNoiseModel & model)
: NoiseModelFactor1<gtsam::Pose3>(model, key),
  observed_normal_in_base_(observed_normal_in_base.normalized()),
  observed_centroid_in_base_(observed_centroid_in_base),
  reference_normal_in_map_(reference_normal_in_map.normalized()),
  reference_plane_offset_(reference_plane_offset)
{
  // The sigmas are kept for serialization, so look through a robust wrapper.
  gtsam::SharedNoiseModel base_model = model;
  if (const auto robust_model = std::dynamic_pointer_cast<gtsam::noiseModel::Robust>(model)) {
    base_model = robust_model->noise();
  }
  auto gaussian_model = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(base_model);
  if (gaussian_model) {
    gtsam::Vector sigmas = gaussian_model->sigmas();
    normal_sigma_ = sigmas(0);
    distance_sigma_ = sigmas(2);
  }
}

gtsam::Vector GroundPlaneFactor::evaluateError(
  const gtsam::Pose3 & pose, gtsam::OptionalMatrixType H) const
{
  if (H) {
    *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
      [this](const gtsam::Pose3 & value) {
        return error(value);
      },
      pose);
  }
  return error(pose);
}

gtsam::Vector GroundPlaneFactor::error(const gtsam::Pose3 & pose) const
{
  const gtsam::Vector3 normal_in_map = pose.rotation().matrix() * observed_normal_in_base_;
  // The height is measured at the ground anchor point of base_link. A plane distance measured at
  // the base origin would change with a camera tilt through the lever arm to the camera.
  const gtsam::Point3 centroid_in_map = pose.transformFrom(observed_centroid_in_base_);
  const double height_error =
    reference_normal_in_map_.dot(centroid_in_map) + reference_plane_offset_;

  // The angle-based chart is monotonic up to 180 degrees. A sin(angle) projection onto the tangent
  // plane would turn into a force towards an upside-down pose beyond 90 degrees.
  const gtsam::Vector2 normal_error =
    gtsam::Unit3(reference_normal_in_map_).localCoordinates(gtsam::Unit3(normal_in_map));

  gtsam::Vector3 error;
  error << normal_error.x(), normal_error.y(), height_error;

  return error;
}

gtsam::Vector3 GroundPlaneFactor::getObservedNormalInBase() const
{
  return observed_normal_in_base_;
}

gtsam::Vector3 GroundPlaneFactor::getReferenceNormalInMap() const
{
  return reference_normal_in_map_;
}

gtsam::Vector3 GroundPlaneFactor::getObservedCentroidInBase() const
{
  return observed_centroid_in_base_;
}

double GroundPlaneFactor::getReferencePlaneOffset() const
{
  return reference_plane_offset_;
}

double GroundPlaneFactor::getNormalSigma() const
{
  return normal_sigma_;
}

double GroundPlaneFactor::getDistanceSigma() const
{
  return distance_sigma_;
}

}  // namespace glidar_slam
