#include "glidar_slam/core/ground_plane_extractor.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>

#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/logger/logger.hpp"
#include "opencv2/imgproc.hpp"
#include "pcl/filters/voxel_grid.h"
#include "pcl/segmentation/sac_segmentation.h"

namespace glidar_slam::core {

std::optional<GroundPlaneObservation> GroundPlaneExtractor::extract(
  const cv::Mat & bgr_image, const cv::Mat & depth_image, const CameraIntrinsics & intrinsics,
  const Eigen::Affine3f & base_from_camera, const Parameters & parameters,
  std::string * failure_reason)
{
  double start_timestamp = 0.0;
  double initialization_timestamp = 0.0;
  double first_extraction_timestamp = 0.0;
  double segmentation_timestamp = 0.0;
  double dense_extraction_timestamp = 0.0;

  const auto timestamp = []() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
  };
  start_timestamp = timestamp();

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

  const int max_depth_sq =
    parameters.ground_extraction_max_distance * parameters.ground_extraction_max_distance;

  const int first_col = static_cast<int>(std::floor(roi_ratios[0] * depth_image.cols));
  const int last_col = static_cast<int>(std::ceil((1.0F - roi_ratios[1]) * depth_image.cols));
  const int first_row = static_cast<int>(std::floor(roi_ratios[2] * depth_image.rows));  // top
  const int last_row =
    static_cast<int>(std::ceil((1.0F - roi_ratios[3]) * depth_image.rows));  // bot

  // Precompute unprojection rays to eliminate divisions and subtractions inside the loops
  std::vector<float> ray_x(last_col);
  for (int col = first_col; col < last_col; ++col) {
    ray_x[col] = static_cast<float>((static_cast<double>(col) - intrinsics.cx) / intrinsics.fx);
  }
  std::vector<float> ray_y(last_row);
  for (int row = first_row; row < last_row; ++row) {
    ray_y[row] = static_cast<float>((static_cast<double>(row) - intrinsics.cy) / intrinsics.fy);
  }

  // binarize the roi
  const cv::Rect roi(first_col, first_row, last_col - first_col, last_row - first_row);
  cv::Mat grayscale;
  cv::Mat binary;
  cv::cvtColor(bgr_image(roi), grayscale, cv::COLOR_BGR2GRAY);
  cv::adaptiveThreshold(
    grayscale, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 101, -65.0);

  const int stride = parameters.ground_extraction_pixel_stride;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud =
    std::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();
  cloud->reserve(
    (static_cast<std::size_t>(last_row - first_row) / stride + 1) *
    (static_cast<std::size_t>(last_col - first_col) / stride + 1));

  const bool depth_is_u16 = depth_image.type() == CV_16UC1;
  initialization_timestamp = timestamp();

  for (int row = first_row; row < last_row; row += stride) {
    const cv::Vec3b * bgr_row = bgr_image.ptr<cv::Vec3b>(row);
    const std::uint8_t * binary_row = binary.ptr<std::uint8_t>(row - first_row);
    const std::uint16_t * depth_row_u16 =
      depth_is_u16 ? depth_image.ptr<std::uint16_t>(row) : nullptr;
    const float * depth_row_f32 = depth_is_u16 ? nullptr : depth_image.ptr<float>(row);

    const float ry = ray_y[row];

    for (int col = first_col; col < last_col; col += stride) {
      const float z =
        depth_is_u16 ? static_cast<float>(depth_row_u16[col]) * 0.001f : depth_row_f32[col];

      // arbitrary setting to omit obvious outliers
      if (z < 0.01f || z > 20.0f) {
        continue;
      }

      const float x = ray_x[col] * z;
      const float y = ry * z;

      if ((x * x + y * y + z * z) > max_depth_sq) {
        continue;
      }

      const Eigen::Vector3f point_in_camera(x, y, z);
      const Eigen::Vector3f point_in_base = base_from_camera * point_in_camera;
      pcl::PointXYZRGBA point;
      point.x = point_in_base.x();
      point.y = point_in_base.y();
      point.z = point_in_base.z();
      const cv::Vec3b & color = bgr_row[col];
      point.r = color[2];
      point.g = color[1];
      point.b = color[0];
      point.a = binary_row[col - first_col];
      cloud->push_back(point);
    }
  }

  first_extraction_timestamp = timestamp();

  if (cloud->empty()) {
    return reject("no point were able to be extracted");
  }

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr voxelized_cloud =
    std::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();
  pcl::VoxelGrid<pcl::PointXYZRGBA> voxel_filter;
  voxel_filter.setInputCloud(cloud);
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
  segmentation_timestamp = timestamp();
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

  if (parameters.ground_debug_enable) {
    observation.pcl.reserve(inliers.indices.size());
    for (const int index : inliers.indices) {
      const pcl::PointXYZRGBA & point = voxelized_cloud->points.at(static_cast<std::size_t>(index));
      observation.pcl.push_back(pcl::PointXYZ{point.x, point.y, point.z});
    }
    observation.pcl.width = static_cast<std::uint32_t>(observation.pcl.size());
    observation.pcl.height = 1;
    observation.pcl.is_dense = true;
  }

  // Dense Extraction
  // transform the plane equation to the camera frame
  const Eigen::Vector3f n_cam = base_from_camera.linear().transpose() * normal;
  const float d_cam = normal.dot(base_from_camera.translation()) + distance;
  const float dist_thresh = static_cast<float>(parameters.ground_extraction_distance_threshold);

  // Pre-multiply ray directions with camera normal components to strip away operations from the
  // inner loop
  std::vector<float> col_dot(last_col);
  for (int col = first_col; col < last_col; ++col) {
    col_dot[col] = n_cam.x() * ray_x[col];
  }

  std::vector<float> row_dot(last_row);
  for (int row = first_row; row < last_row; ++row) {
    row_dot[row] = n_cam.y() * ray_y[row] + n_cam.z();
  }

  int dynamic_first_row = first_row;

  if (std::abs(n_cam.y()) > 1e-4f) {
    const float ray_x_left = ray_x[first_col];
    const float ray_x_right = ray_x[last_col - 1];

    const float ray_y_left = -(n_cam.x() * ray_x_left + n_cam.z()) / n_cam.y();
    const float ray_y_right = -(n_cam.x() * ray_x_right + n_cam.z()) / n_cam.y();

    const int horizon_row_left = static_cast<int>(intrinsics.cy + intrinsics.fy * ray_y_left);
    const int horizon_row_right = static_cast<int>(intrinsics.cy + intrinsics.fy * ray_y_right);

    // The ground plane is entirely below the highest point of the horizon line
    const int min_horizon_row = std::min(horizon_row_left, horizon_row_right);

    // We can safely clamp our starting row to this horizon bound
    dynamic_first_row = std::max(first_row, min_horizon_row);
    // Ensure we don't overshoot if the entire ROI is above the horizon
    dynamic_first_row = std::min(dynamic_first_row, last_row);
  }

  SAM_INFO(
    "Ground plane extraction: dense extraction starting at row {} (ROI rows {}-{})",
    dynamic_first_row, first_row, last_row);

  // Reserve a generous heuristic size to minimize reallocations
  observation.ground_cloud.reserve((last_row - dynamic_first_row) * (last_col - first_col) / 2);

  for (int row = dynamic_first_row; row < last_row; ++row) {
    const cv::Vec3b * bgr_row = bgr_image.ptr<cv::Vec3b>(row);
    const std::uint8_t * binary_row = binary.ptr<std::uint8_t>(row - first_row);
    const std::uint16_t * depth_row_u16 =
      depth_is_u16 ? depth_image.ptr<std::uint16_t>(row) : nullptr;
    const float * depth_row_f32 = depth_is_u16 ? nullptr : depth_image.ptr<float>(row);

    const float r_dot = row_dot[row];
    const float ry = ray_y[row];

    for (int col = first_col; col < last_col; ++col) {
      const float z =
        depth_is_u16 ? static_cast<float>(depth_row_u16[col]) * 0.001f : depth_row_f32[col];

      if (z < 0.01f || z > 20.0f) {
        continue;
      }

      // Early rejection distance check: 1 add, 1 mult, 1 add, 1 abs.
      const float pt_distance = (col_dot[col] + r_dot) * z + d_cam;
      if (std::abs(pt_distance) > dist_thresh) {
        continue;
      }

      const float x = ray_x[col] * z;
      const float y = ry * z;

      // Validate threshold against max distance
      if ((x * x + y * y + z * z) > max_depth_sq) {
        continue;
      }

      // We only run the expensive matrix transform on points proven to be inliers
      const Eigen::Vector3f point_in_base = base_from_camera * Eigen::Vector3f(x, y, z);

      pcl::PointXYZRGBA point;
      point.x = point_in_base.x();
      point.y = point_in_base.y();
      point.z = point_in_base.z();
      const cv::Vec3b & color = bgr_row[col];
      point.r = color[2];
      point.g = color[1];
      point.b = color[0];
      point.a = binary_row[col - first_col];
      observation.ground_cloud.push_back(point);
    }
  }

  dense_extraction_timestamp = timestamp();

  observation.point_count = (last_row - dynamic_first_row) * (last_col - first_col);
  observation.inlier_count = observation.ground_cloud.size();
  observation.ground_cloud.width = static_cast<std::uint32_t>(observation.ground_cloud.size());
  observation.ground_cloud.height = 1;
  observation.ground_cloud.is_dense = true;

  if (parameters.debug_timings) {
    SAM_INFO(
      "Ground plane extraction timing [ms]: initialization={}, first_extraction={}, "
      "segmentation={}, "
      "dense_extraction={}",
      (initialization_timestamp - start_timestamp) * 1000.0,
      (first_extraction_timestamp - initialization_timestamp) * 1000.0,
      (segmentation_timestamp - first_extraction_timestamp) * 1000.0,
      (dense_extraction_timestamp - segmentation_timestamp) * 1000.0);
  }

  return observation;
}

}  // namespace glidar_slam::core
