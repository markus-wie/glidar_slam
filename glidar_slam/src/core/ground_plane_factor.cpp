#include "glidar_slam/core/ground_plane_factor.hpp"

#include <cmath>

#include "gtsam/base/numericalDerivative.h"

namespace glidar_slam::core {

GroundPlaneFactor::GroundPlaneFactor(
  gtsam::Key key, const gtsam::Vector3 & observed_normal_in_base, double observed_distance_to_base,
  const gtsam::Vector3 & reference_normal_in_map, double reference_plane_offset,
  const gtsam::SharedNoiseModel & model)
: NoiseModelFactor1<gtsam::Pose3>(model, key),
  observed_normal_in_base_(observed_normal_in_base.normalized()),
  observed_distance_to_base_(observed_distance_to_base),
  reference_normal_in_map_(reference_normal_in_map.normalized()),
  reference_plane_offset_(reference_plane_offset)
{
  const gtsam::Vector3 basis_axis =
    std::abs(reference_normal_in_map_.dot(gtsam::Vector3::UnitZ())) < 0.9 ? gtsam::Vector3::UnitZ()
                                                                          : gtsam::Vector3::UnitX();
  tangent_x_ = reference_normal_in_map_.cross(basis_axis).normalized();
  tangent_y_ = reference_normal_in_map_.cross(tangent_x_).normalized();
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
  const double plane_offset = observed_distance_to_base_ - normal_in_map.dot(pose.translation());

  gtsam::Vector2 normal_err_2d =
    gtsam::Unit3(reference_normal_in_map_).localCoordinates(gtsam::Unit3(normal_in_map));

  gtsam::Vector3 error;
  error << normal_err_2d.x(), normal_err_2d.y(), plane_offset - reference_plane_offset_;

  return error;
}

}  // namespace glidar_slam::core
