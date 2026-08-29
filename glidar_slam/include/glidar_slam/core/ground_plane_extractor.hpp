#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "glidar_slam/core/camera_model.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"
#include "opencv2/core/mat.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::core {

struct GroundPlaneObservation
{
  Eigen::Vector3f normal_in_base;
  float distance_to_base{0.0f};
  std::size_t point_count{0};
  std::size_t inlier_count{0};

  /**
   * Dense ground cloud in base frame
   *
   * Contains the dense 3D point cloud of the ground plane inliers in the base frame.
   * Keeps the original RGB color from the camera and the binarized lane-marking evidence in the
   * alpha channel.
   */
  PointCloudXYZRGBAPtr ground_cloud{new PointCloudXYZRGBA()};

  /**
   * Sparse ground cloud in base frame
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

}  // namespace glidar_slam::core
