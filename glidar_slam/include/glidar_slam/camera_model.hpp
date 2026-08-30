#pragma once

#include "Eigen/Geometry"
#include "opencv2/core/types.hpp"

namespace glidar_slam {

struct CameraIntrinsics
{
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
};

class CameraModel
{
public:
  CameraModel() = default;

  CameraModel(
    const cv::Size & image_size, float fx, float fy, float cx, float cy,
    const Eigen::Affine3f & base_from_camera = Eigen::Affine3f::Identity());

  bool valid() const;
  CameraIntrinsics intrinsics() const;
  const Eigen::Affine3f & baseFromCamera() const;

private:
  cv::Size image_size_;
  float fx_{0.0F};
  float fy_{0.0F};
  float cx_{0.0F};
  float cy_{0.0F};
  Eigen::Affine3f base_from_camera_;
};

}  // namespace glidar_slam
