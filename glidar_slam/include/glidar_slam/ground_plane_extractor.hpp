#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "glidar_slam/camera_model.hpp"
#include "glidar_slam/parameters.hpp"
#include "glidar_slam/types.hpp"
#include "opencv2/core/mat.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam {

struct GroundPlaneObservation
{
  Eigen::Vector3f normal_in_base;
  float distance_to_base{0.0f};
  // Ground anchor point of base_link in the base frame, i.e. -base_height * normal_in_base. The
  // base height is the camera height above the plane (invariant to a camera rotation about its
  // own center) minus the camera mounting height, so the height measurement is not distorted by
  // the lever arm between base_link and the camera.
  Eigen::Vector3f centroid_in_base{Eigen::Vector3f::Zero()};
  std::size_t point_count{0};
  std::size_t inlier_count{0};

  /**
   * Transform that levels base-frame points: it rotates the observed ground normal onto +Z about
   * the camera position and shifts z so that the ground lies at z = 0. `ground_cloud` is expressed
   * in this leveled frame; use `level_from_base.inverse()` to get back to the base frame, e.g. for
   * projecting it into the camera image.
   */
  Eigen::Affine3f level_from_base{Eigen::Affine3f::Identity()};

  /**
   * Dense ground cloud in the leveled base frame
   *
   * Contains the dense 3D point cloud of the ground plane inliers. The points are expressed in the
   * base frame and then transformed by `level_from_base`, so that the observed ground normal
   * coincides with the base z axis. The cloud can therefore be projected to 2D by dropping z.
   * Keeps the original RGB color from the camera and the binarized lane-marking evidence in the
   * alpha channel.
   */
  PointCloudXYZRGBAPtr ground_cloud{new PointCloudXYZRGBA()};

  /**
   * Sparse ground cloud in base frame (not leveled, for debugging the plane fit)
   *
   * Originates from the initial sparse extraction for the plane fitting step.
   *
   * Contains the sparse 3D point cloud of the ground plane inliers in the base frame.
   * Keeps only the XYZ coordinates, without color or alpha channel.
   */
  PointCloudXYZPtr ground_cloud_sparse{new PointCloudXYZ()};
};

class GroundPlaneExtractor
{
public:
  struct Result
  {
    std::optional<GroundPlaneObservation> observation;
    std::string failure_reason;
    cv::Mat debug_image;
  };

  static Result extract(
    const cv::Mat & bgr_image, const cv::Mat & depth_image, const CameraIntrinsics & intrinsics,
    const Eigen::Affine3f & base_from_camera, const Parameters & parameters);
};

}  // namespace glidar_slam
