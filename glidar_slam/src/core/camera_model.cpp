#include "glidar_slam/core/camera_model.hpp"

namespace glidar_slam::core {

CameraModel::CameraModel(
  const cv::Size & image_size, float fx, float fy, float cx, float cy,
  const Eigen::Affine3f & base_from_camera)
: image_size_(image_size), fx_(fx), fy_(fy), cx_(cx), cy_(cy), base_from_camera_(base_from_camera)
{
}

bool CameraModel::valid() const
{
  return image_size_.width > 0 && image_size_.height > 0 && fx_ > 0.0F && fy_ > 0.0F;
}

CameraIntrinsics CameraModel::intrinsics() const
{
  return {fx_, fy_, cx_, cy_};
}

const Eigen::Affine3f & CameraModel::baseFromCamera() const
{
  return base_from_camera_;
}

}  // namespace glidar_slam::core
