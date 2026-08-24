#pragma once

#include "glidar_slam/core/camera_model.hpp"
#include "glidar_slam/core/laser_scan.hpp"
#include "opencv2/core/mat.hpp"

namespace glidar_slam::core {

class SensorData
{
public:
  SensorData() = default;

  explicit SensorData(std::shared_ptr<const LaserScan> scan);

  SensorData(
    const cv::Mat & image, const cv::Mat & depth, std::shared_ptr<const LaserScan> scan,
    const CameraModel & camera_model);

  const cv::Mat & image() const;
  const cv::Mat & depth() const;
  const std::shared_ptr<const LaserScan> & laserScan() const;
  const CameraModel & cameraModel() const;
  bool hasVisualData() const;

private:
  cv::Mat image_;
  cv::Mat depth_;
  std::shared_ptr<const LaserScan> laser_scan_;
  CameraModel camera_model_;
};

}  // namespace glidar_slam::core
