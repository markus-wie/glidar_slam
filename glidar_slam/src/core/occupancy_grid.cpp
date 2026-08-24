#include "glidar_slam/core/occupancy_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace glidar_slam::core {

OccupancyGrid::OccupancyGrid(const std::shared_ptr<Parameters> & parameters)
: parameters_(parameters)
{
  info_.resolution = parameters_->occ_map_resolution;
}

template <typename Visitor>
void OccupancyGrid::raytraceLine(int x0, int y0, int x1, int y1, Visitor && visit) const
{
  const int dx = std::abs(x1 - x0);
  const int sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0);
  const int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  int x = x0;
  int y = y0;

  while (true) {
    if (!visit(x, y)) {
      break;
    }
    if (x == x1 && y == y1) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
}

bool OccupancyGrid::buildFromScans(
  const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> & scans_transformed)
{
  if (scans_transformed.empty() || parameters_->occ_map_resolution <= 0.0) {
    return false;
  }

  info_.resolution = parameters_->occ_map_resolution;

  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();

  for (const auto & [pose, cloud] : scans_transformed) {
    const auto & translation = pose.translation();
    min_x = std::min(min_x, static_cast<float>(translation.x()));
    min_y = std::min(min_y, static_cast<float>(translation.y()));
    max_x = std::max(max_x, static_cast<float>(translation.x()));
    max_y = std::max(max_y, static_cast<float>(translation.y()));

    for (const auto & point : cloud.points) {
      min_x = std::min(min_x, point.x);
      min_y = std::min(min_y, point.y);
      max_x = std::max(max_x, point.x);
      max_y = std::max(max_y, point.y);
    }
  }

  const double padding = static_cast<double>(parameters_->occ_map_padding) * info_.resolution;
  info_.origin_x = static_cast<double>(min_x) - padding;
  info_.origin_y = static_cast<double>(min_y) - padding;
  const double width_m = static_cast<double>(max_x - min_x) + 2.0 * padding;
  const double height_m = static_cast<double>(max_y - min_y) + 2.0 * padding;

  info_.width = static_cast<uint32_t>(std::ceil(width_m / info_.resolution));
  info_.height = static_cast<uint32_t>(std::ceil(height_m / info_.resolution));

  if (info_.width == 0U || info_.height == 0U) {
    return false;
  }

  data_.assign(static_cast<size_t>(info_.width) * static_cast<size_t>(info_.height), -1);

  for (const auto & [pose, cloud] : scans_transformed) {
    const auto & translation = pose.translation();
    const int ox =
      static_cast<int>(std::floor((translation.x() - info_.origin_x) / info_.resolution));
    const int oy =
      static_cast<int>(std::floor((translation.y() - info_.origin_y) / info_.resolution));

    if (
      ox < 0 || oy < 0 || ox >= static_cast<int>(info_.width) ||
      oy >= static_cast<int>(info_.height)) {
      continue;
    }

    for (const auto & point : cloud.points) {
      const int tx = static_cast<int>(std::floor((point.x - info_.origin_x) / info_.resolution));
      const int ty = static_cast<int>(std::floor((point.y - info_.origin_y) / info_.resolution));

      if (
        tx < 0 || ty < 0 || tx >= static_cast<int>(info_.width) ||
        ty >= static_cast<int>(info_.height)) {
        continue;
      }

      raytraceLine(ox, oy, tx, ty, [&](int cx, int cy) {
        if (cx == tx && cy == ty) {
          return false;
        }
        const size_t idx = indexOf(cx, cy);
        if (data_[idx] != 100) {
          data_[idx] = 0;
        }
        return true;
      });

      data_[indexOf(tx, ty)] = 100;
    }
  }

  return true;
}

}  // namespace glidar_slam::core
