#include "glidar_slam/core/ground_texture_grid.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace glidar_slam::core {

GroundTextureGrid::GroundTextureGrid(const std::shared_ptr<Parameters> & parameters)
: parameters_(parameters)
{
  info_.resolution = parameters_->ground_map_resolution;
}

bool GroundTextureGrid::buildFromGroundClouds(
  const std::vector<pcl::PointCloud<pcl::PointXYZRGBA>> & clouds)
{
  if (clouds.empty() || parameters_->ground_map_resolution <= 0.0) {
    return false;
  }

  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  std::size_t point_count = 0;

  for (const auto & cloud : clouds) {
    for (const auto & point : cloud.points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        continue;
      }
      min_x = std::min(min_x, point.x);
      min_y = std::min(min_y, point.y);
      max_x = std::max(max_x, point.x);
      max_y = std::max(max_y, point.y);
      ++point_count;
    }
  }

  if (point_count == 0) {
    return false;
  }

  info_.resolution = parameters_->ground_map_resolution;
  const double padding = static_cast<double>(parameters_->ground_map_padding) * info_.resolution;
  info_.origin_x = static_cast<double>(min_x) - padding;
  info_.origin_y = static_cast<double>(min_y) - padding;
  info_.width = static_cast<std::uint32_t>(
    std::ceil((static_cast<double>(max_x - min_x) + 2.0 * padding) / info_.resolution));
  info_.height = static_cast<std::uint32_t>(
    std::ceil((static_cast<double>(max_y - min_y) + 2.0 * padding) / info_.resolution));
  if (info_.width == 0U || info_.height == 0U) {
    return false;
  }

  const std::size_t cell_count =
    static_cast<std::size_t>(info_.width) * static_cast<std::size_t>(info_.height);
  std::vector<std::array<std::uint64_t, 3>> color_sums(cell_count);
  std::vector<std::uint64_t> sample_counts(cell_count, 0U);
  coverage_data_.assign(cell_count, -1);

  for (const auto & cloud : clouds) {
    for (const auto & point : cloud.points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
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

      const std::size_t index =
        static_cast<std::size_t>(cell_y) * static_cast<std::size_t>(info_.width) +
        static_cast<std::size_t>(cell_x);
      color_sums[index][0] += point.r;
      color_sums[index][1] += point.g;
      color_sums[index][2] += point.b;
      ++sample_counts[index];
      coverage_data_[index] = 100;
    }
  }

  rgb_data_.assign(cell_count * 3U, 0U);
  for (std::size_t index = 0; index < cell_count; ++index) {
    if (sample_counts[index] == 0U) {
      continue;
    }
    rgb_data_[index * 3U] = static_cast<std::uint8_t>(
      (color_sums[index][0] + sample_counts[index] / 2U) / sample_counts[index]);
    rgb_data_[index * 3U + 1U] = static_cast<std::uint8_t>(
      (color_sums[index][1] + sample_counts[index] / 2U) / sample_counts[index]);
    rgb_data_[index * 3U + 2U] = static_cast<std::uint8_t>(
      (color_sums[index][2] + sample_counts[index] / 2U) / sample_counts[index]);
  }

  return true;
}

}  // namespace glidar_slam::core
