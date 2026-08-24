#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "glidar_slam/core/camera_model.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "opencv2/core/mat.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::core {

struct GroundPlaneObservation
{
  Eigen::Vector3f normal_in_base;
  float distance_to_base{0.0F};
  std::size_t point_count{0};
  std::size_t inlier_count{0};
  pcl::PointCloud<pcl::PointXYZRGB> binary_ground_cloud;
};

class GroundPlaneExtractor
{
public:
  static std::optional<GroundPlaneObservation> extract(
    const cv::Mat & bgr_image, const cv::Mat & depth_image, const CameraIntrinsics & intrinsics,
    const Eigen::Affine3f & base_from_camera, const Parameters & parameters,
    std::string * failure_reason = nullptr);

private:
  static float depthAtMeters(const cv::Mat & depth_image, int row, int col);
};

}  // namespace glidar_slam::core
