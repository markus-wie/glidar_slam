#include "glidar_slam/sensor_data.hpp"

#include <utility>

namespace glidar_slam {

SensorData::SensorData(std::shared_ptr<const LaserScan> scan) : laser_scan_(std::move(scan))
{
}

SensorData::SensorData(
  const cv::Mat & image, const cv::Mat & depth, std::shared_ptr<const LaserScan> scan,
  const CameraModel & camera_model)
: image_(image), depth_(depth), laser_scan_(std::move(scan)), camera_model_(camera_model)
{
}

const cv::Mat & SensorData::image() const
{
  return image_;
}

const cv::Mat & SensorData::depth() const
{
  return depth_;
}

const std::shared_ptr<const LaserScan> & SensorData::laserScan() const
{
  return laser_scan_;
}

const CameraModel & SensorData::cameraModel() const
{
  return camera_model_;
}

bool SensorData::hasVisualData() const
{
  return !image_.empty() && !depth_.empty() && camera_model_.valid();
}

}  // namespace glidar_slam
