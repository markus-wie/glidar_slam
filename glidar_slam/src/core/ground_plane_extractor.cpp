#include "glidar_slam/core/ground_plane_extractor.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>

#include "glidar_slam/core/parameters.hpp"
#include "opencv2/imgproc.hpp"
#include "pcl/features/normal_3d_omp.h"
#include "pcl/filters/passthrough.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/segmentation/sac_segmentation.h"

namespace glidar_slam::core {

std::optional<GroundPlaneObservation> GroundPlaneExtractor::extract(
  const cv::Mat & bgr_image, const cv::Mat & depth_image, const CameraIntrinsics & intrinsics,
  const Eigen::Affine3f & base_from_camera, const Parameters & parameters,
  std::string * failure_reason)
{
  const std::vector<float> & roi_ratios = parameters.ground_roi_ratios;

  const auto reject = [failure_reason](const std::string & reason) {
    if (failure_reason != nullptr) {
      *failure_reason = reason;
    }
    return std::optional<GroundPlaneObservation>{};
  };

  if (
    bgr_image.empty() || depth_image.empty() || bgr_image.size() != depth_image.size() ||
    intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0) {
    return reject("invalid image dimensions or camera intrinsics");
  }
  if (depth_image.type() != CV_16UC1 && depth_image.type() != CV_32FC1) {
    return reject("unsupported depth image type " + std::to_string(depth_image.type()));
  }
  if (
    roi_ratios.size() != 4 || roi_ratios[0] < 0.0F || roi_ratios[1] < 0.0F ||
    roi_ratios[2] < 0.0F || roi_ratios[3] < 0.0F || roi_ratios[0] + roi_ratios[1] >= 1.0F ||
    roi_ratios[2] + roi_ratios[3] >= 1.0F) {
    return reject("invalid ground ROI ratios");
  }

  cv::Mat grayscale;
  cv::Mat binary;
  cv::cvtColor(bgr_image, grayscale, cv::COLOR_BGR2GRAY);
  cv::adaptiveThreshold(
    grayscale, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 101, -65.0);

  const int max_depth_sq =
    parameters.ground_extraction_max_distance * parameters.ground_extraction_max_distance;

  const int first_col = static_cast<int>(std::floor(roi_ratios[0] * depth_image.cols));
  const int last_col = static_cast<int>(std::ceil((1.0F - roi_ratios[1]) * depth_image.cols));
  const int first_row = static_cast<int>(std::floor(roi_ratios[2] * depth_image.rows));
  const int last_row = static_cast<int>(std::ceil((1.0F - roi_ratios[3]) * depth_image.rows));

  pcl::PointCloud<pcl::PointXYZRGBA> cloud;
  cloud.reserve(
    static_cast<std::size_t>((last_row - first_row) / parameters.ground_extraction_pixel_stride) *
    static_cast<std::size_t>((last_col - first_col) / parameters.ground_extraction_pixel_stride));

  for (int row = first_row; row < last_row; row += parameters.ground_extraction_pixel_stride) {
    for (int col = first_col; col < last_col; col += parameters.ground_extraction_pixel_stride) {
      const float z = depthAtMeters(depth_image, row, col);

      // arbitrary setting to omit obvious outliers
      if (z < 0.01f || z > 20.0f) {
        continue;
      }

      const float x =
        static_cast<float>((static_cast<double>(col) - intrinsics.cx) * z / intrinsics.fx);
      const float y =
        static_cast<float>((static_cast<double>(row) - intrinsics.cy) * z / intrinsics.fy);

      if ((x * x + y * y + z * z) > max_depth_sq) {
        continue;
      }

      const Eigen::Vector3f point_in_camera(x, y, z);
      const Eigen::Vector3f point_in_base = base_from_camera * point_in_camera;
      pcl::PointXYZRGBA point;
      point.x = point_in_base.x();
      point.y = point_in_base.y();
      point.z = point_in_base.z();
      const cv::Vec3b & color = bgr_image.at<cv::Vec3b>(row, col);
      point.r = color[2];
      point.g = color[1];
      point.b = color[0];
      point.a = binary.at<std::uint8_t>(row, col);
      cloud.push_back(point);
    }
  }

  if (cloud.empty()) {
    return reject("no point were able to be extracted");
  }

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filtered_cloud = cloud.makeShared();

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr voxelized_cloud =
    std::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();
  pcl::VoxelGrid<pcl::PointXYZRGBA> voxel_filter;
  voxel_filter.setInputCloud(filtered_cloud);
  voxel_filter.setLeafSize(
    static_cast<float>(parameters.ground_extraction_voxel_size),
    static_cast<float>(parameters.ground_extraction_voxel_size),
    static_cast<float>(parameters.ground_extraction_voxel_size));
  voxel_filter.filter(*voxelized_cloud);

  // Standard Ransac using constrained perpendicular plane. Needs fine tuning of the distance
  // threshold
  pcl::SACSegmentation<pcl::PointXYZRGBA> segmentation;
  segmentation.setOptimizeCoefficients(true);
  segmentation.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  segmentation.setMethodType(pcl::SAC_MSAC);
  segmentation.setAxis(Eigen::Vector3f::UnitZ());
  segmentation.setEpsAngle(static_cast<float>(15.0 * M_PI / 180.0));
  segmentation.setMaxIterations(1000);
  segmentation.setDistanceThreshold(
    static_cast<float>(parameters.ground_extraction_distance_threshold));
  segmentation.setInputCloud(voxelized_cloud);

  pcl::PointIndices inliers;
  pcl::ModelCoefficients coefficients;
  segmentation.segment(inliers, coefficients);
  if (inliers.indices.empty() || coefficients.values.size() != 4) {
    return reject(
      "RANSAC found no ground plane (input_points=" + std::to_string(voxelized_cloud->size()) +
      ")");
  }

  Eigen::Vector3f normal(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
  if (!normal.allFinite() || normal.norm() < 1e-6f) {
    return reject("RANSAC produced an invalid ground-plane normal");
  }

  normal.normalize();
  float distance = coefficients.values[3];
  if (normal.z() < 0.0F) {
    normal = -normal;
    distance = -distance;
  }

  GroundPlaneObservation observation;
  observation.normal_in_base = normal;
  observation.distance_to_base = distance;
  observation.point_count = voxelized_cloud->size();
  observation.inlier_count = inliers.indices.size();
  pcl::copyPointCloud(*voxelized_cloud, inliers.indices, observation.ground_cloud);
  observation.ground_cloud.width = static_cast<std::uint32_t>(observation.ground_cloud.size());
  observation.ground_cloud.height = 1;
  observation.ground_cloud.is_dense = true;
  return observation;
}

float GroundPlaneExtractor::depthAtMeters(const cv::Mat & depth_image, int row, int col)
{
  if (depth_image.type() == CV_16UC1) {
    return static_cast<float>(depth_image.at<std::uint16_t>(row, col)) * 0.001f;
  }
  if (depth_image.type() == CV_32FC1) {
    return depth_image.at<float>(row, col);
  }
  return -1.0f;
}

}  // namespace glidar_slam::core
