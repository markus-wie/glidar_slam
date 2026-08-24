#include "glidar_slam/core/ground_marking_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace glidar_slam::core {

GroundMarkingGrid::GroundMarkingGrid(const std::shared_ptr<Parameters> & parameters)
: parameters_(parameters)
{
  info_.resolution = parameters_->ground_marking_map_resolution;
}

bool GroundMarkingGrid::isWhiteMarking(const pcl::PointXYZRGB & point) const
{
  return point.r >= parameters_->ground_marking_white_threshold &&
         point.g >= parameters_->ground_marking_white_threshold &&
         point.b >= parameters_->ground_marking_white_threshold;
}

bool GroundMarkingGrid::buildFromGroundClouds(
  const std::vector<pcl::PointCloud<pcl::PointXYZRGB>> & clouds)
{
  if (clouds.empty() || parameters_->ground_marking_map_resolution <= 0.0) {
    return false;
  }

  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  std::size_t marking_count = 0;

  for (const auto & cloud : clouds) {
    for (const auto & point : cloud.points) {
      if (!isWhiteMarking(point) || !std::isfinite(point.x) || !std::isfinite(point.y)) {
        continue;
      }
      min_x = std::min(min_x, point.x);
      min_y = std::min(min_y, point.y);
      max_x = std::max(max_x, point.x);
      max_y = std::max(max_y, point.y);
      ++marking_count;
    }
  }

  if (marking_count == 0) {
    return false;
  }

  info_.resolution = parameters_->ground_marking_map_resolution;
  const double padding =
    static_cast<double>(parameters_->ground_marking_map_padding) * info_.resolution;
  info_.origin_x = static_cast<double>(min_x) - padding;
  info_.origin_y = static_cast<double>(min_y) - padding;
  info_.width = static_cast<uint32_t>(
    std::ceil((static_cast<double>(max_x - min_x) + 2.0 * padding) / info_.resolution));
  info_.height = static_cast<uint32_t>(
    std::ceil((static_cast<double>(max_y - min_y) + 2.0 * padding) / info_.resolution));
  if (info_.width == 0U || info_.height == 0U) {
    return false;
  }

  data_.assign(static_cast<size_t>(info_.width) * static_cast<size_t>(info_.height), -1);
  for (const auto & cloud : clouds) {
    for (const auto & point : cloud.points) {
      if (!isWhiteMarking(point) || !std::isfinite(point.x) || !std::isfinite(point.y)) {
        continue;
      }
      const int cell_x =
        static_cast<int>(std::floor((point.x - info_.origin_x) / info_.resolution));
      const int cell_y =
        static_cast<int>(std::floor((point.y - info_.origin_y) / info_.resolution));
      if (
        cell_x < 0 || cell_y < 0 || cell_x >= static_cast<int>(info_.width) ||
        cell_y >= static_cast<int>(info_.height)) {
        continue;
      }
      data_
        [static_cast<size_t>(cell_y) * static_cast<size_t>(info_.width) +
         static_cast<size_t>(cell_x)] = 100;
    }
  }

  return true;
}

}  // namespace glidar_slam::core
