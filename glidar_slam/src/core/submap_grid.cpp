#include "glidar_slam/core/submap_grid.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "glidar_slam/logger/logger.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace glidar_slam::core {

namespace {

int cellCoordinate(double value, double resolution)
{
  return static_cast<int>(std::floor(value / resolution));
}

void buildDistanceTransformField(
  LikelihoodField & field, const SubmapGrid::HitCountGrid & grid, double smear_deviation)
{
  const int width = field.width;
  const int height = field.height;

  cv::Mat binary_map(height, width, CV_8UC1, cv::Scalar(255));

  const int origin_cx = static_cast<int>(std::round(field.origin_x / field.resolution));
  const int origin_cy = static_cast<int>(std::round(field.origin_y / field.resolution));

  for (const auto & cell : grid.active_cells) {
    const int local_x = cell.x - origin_cx;
    const int local_y = cell.y - origin_cy;

    if (local_x >= 0 && local_x < width && local_y >= 0 && local_y < height) {
      binary_map.at<uint8_t>(local_y, local_x) = 0;
    }
  }

  cv::Mat dist_map;
  cv::distanceTransform(binary_map, dist_map, cv::DIST_L2, cv::DIST_MASK_PRECISE);

  const double max_distance_sq = std::pow(smear_deviation * 2.0, 2);
  const double denominator = 2.0 * std::pow(smear_deviation, 2);
  const double resolution = field.resolution;

  field.data.assign(static_cast<std::size_t>(width) * height, 0.0F);

  for (int y = 0; y < height; ++y) {
    const float * dist_row = dist_map.ptr<float>(y);
    const int row_offset = y * width;

    for (int x = 0; x < width; ++x) {
      // OpenCV returns linear distance in pixels. Multiply by resolution and square it.
      const double pixel_dist = static_cast<double>(dist_row[x]);
      const double distance_sq = (pixel_dist * resolution) * (pixel_dist * resolution);

      if (distance_sq <= max_distance_sq) {
        field.data[row_offset + x] = static_cast<float>(std::exp(-distance_sq / denominator));
      }
    }
  }
}
}  // namespace

SubmapGrid::SubmapGrid(const std::vector<double> & resolutions)
{
  grids_.reserve(resolutions.size());
  for (const double resolution : resolutions) {
    if (!std::isfinite(resolution) || resolution <= 0.0) {
      throw std::invalid_argument("SubmapGrid resolution must be positive and finite");
    }
    HitCountGrid grid;
    grid.resolution = resolution;
    grids_.emplace(grid.resolution, std::move(grid));
  }
}

void SubmapGrid::add(const std::vector<Point2D> & points, uint64_t keyframe_id)
{
  for (auto & [resolution, grid] : grids_) {
    for (const auto & point : points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        continue;
      }

      GridIndex idx{
        cellCoordinate(point.x, grid.resolution), cellCoordinate(point.y, grid.resolution)};

      const int flat_idx = grid.flatIdx(idx.x, idx.y);

      if (grid.counts[flat_idx] == 0) {
        grid.active_index[flat_idx] = static_cast<int>(grid.active_cells.size());
        grid.active_cells.push_back(idx);
      }
      grid.counts[flat_idx]++;
    }
  }

  active_keyframes_.push_back(keyframe_id);
  cached_world_points_.emplace(keyframe_id, points);
}

void SubmapGrid::remove(uint64_t keyframe_id)
{
  auto cache_it = cached_world_points_.find(keyframe_id);
  if (cache_it == cached_world_points_.end()) {
    return;
  }

  const auto & points = cache_it->second;

  for (auto & [resolution, grid] : grids_) {
    for (const auto & point : points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        continue;
      }

      GridIndex idx{
        cellCoordinate(point.x, grid.resolution), cellCoordinate(point.y, grid.resolution)};

      const int flat_idx = grid.flatIdx(idx.x, idx.y);

      if (grid.counts[flat_idx] > 0) {
        grid.counts[flat_idx]--;

        if (grid.counts[flat_idx] == 0) {
          const int array_pos = grid.active_index[flat_idx];
          const GridIndex last_idx = grid.active_cells.back();
          const int last_flat_idx = grid.flatIdx(last_idx.x, last_idx.y);

          grid.active_cells[array_pos] = last_idx;
          grid.active_index[last_flat_idx] = array_pos;
          grid.active_cells.pop_back();
        }
      }
    }
  }

  cached_world_points_.erase(cache_it);

  auto kf_it = std::find(active_keyframes_.begin(), active_keyframes_.end(), keyframe_id);
  if (kf_it != active_keyframes_.end()) {
    active_keyframes_.erase(kf_it);
  }
}

void SubmapGrid::removeOldestKeyframe()
{
  if (!active_keyframes_.empty()) {
    remove(active_keyframes_.front());
  }
}

size_t SubmapGrid::size() const
{
  return active_keyframes_.size();
}

LikelihoodField SubmapGrid::getLikelihoodField(
  double resolution, double smear_deviation, bool use_distance_transform, bool debug_timings) const
{
  if (grids_.find(resolution) == grids_.end()) {
    throw std::invalid_argument("Requested resolution is not configured in SubmapGrid");
  }

  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point preparation_time;
  std::chrono::steady_clock::time_point kernel_time;
  std::chrono::steady_clock::time_point splatting_time;

  start_time = std::chrono::steady_clock::now();

  const HitCountGrid & grid = grids_.at(resolution);

  LikelihoodField field;
  field.resolution = resolution;

  if (grid.active_cells.empty()) {
    return field;
  }

  const double max_smear_distance = smear_deviation * 2.0;
  const int rad = std::ceil(max_smear_distance / resolution);

  int min_x = grid.active_cells[0].x;
  int max_x = grid.active_cells[0].x;
  int min_y = grid.active_cells[0].y;
  int max_y = grid.active_cells[0].y;

  for (const auto & cell : grid.active_cells) {
    min_x = std::min(min_x, cell.x);
    max_x = std::max(max_x, cell.x);
    min_y = std::min(min_y, cell.y);
    max_y = std::max(max_y, cell.y);
  }

  // Size the field, explicitly adding 'rad' padding to all sides
  // so the Gaussian tails are not cut off at the edges of the map
  field.origin_x = (min_x - rad) * resolution;
  field.origin_y = (min_y - rad) * resolution;
  field.width = (max_x - min_x) + 2 * rad + 1;
  field.height = (max_y - min_y) + 2 * rad + 1;
  field.max_x_index = field.width - 1;
  field.max_y_index = field.height - 1;
  field.data.clear();
  field.data.resize(static_cast<std::size_t>(field.width) * field.height, 0.0F);

  preparation_time = std::chrono::steady_clock::now();

  if (use_distance_transform) {
    buildDistanceTransformField(field, grid, smear_deviation);
    return field;
  }

  // Precompute the splat kernel
  const double denom = 2.0 * smear_deviation * smear_deviation;
  const double resolution_sq = resolution * resolution;
  const double max_smear_distance_sq = max_smear_distance * max_smear_distance;
  const int kernel_width = 2 * rad + 1;

  struct KernelOffset
  {
    int dx;
    int dy;
    float probability;
  };

  std::vector<KernelOffset> kernel_offsets;
  kernel_offsets.reserve(static_cast<std::size_t>(kernel_width) * kernel_width);
  for (int dy = -rad; dy <= rad; ++dy) {
    for (int dx = -rad; dx <= rad; ++dx) {
      const double dist_sq = (dx * dx + dy * dy) * resolution_sq;
      if (dist_sq <= max_smear_distance_sq) {
        kernel_offsets.push_back({dx, dy, static_cast<float>(std::exp(-dist_sq / denom))});
      }
    }
  }

  kernel_time = std::chrono::steady_clock::now();

  // Splat the kernel onto the field using ONLY the active cells
  const int field_width = field.width;
  const int x_offset_map = min_x - rad;
  const int y_offset_map = min_y - rad;

  for (const auto & offset : kernel_offsets) {
    const int dx = offset.dx;
    const int dy = offset.dy;
    const float prob = offset.probability;

    // Loop over the unique occupied cells inside
    for (const auto & cell : grid.active_cells) {
      const int cx = cell.x - x_offset_map;
      const int cy = cell.y - y_offset_map;

      const int nx = cx + dx;
      const int ny = cy + dy;

      const int out_idx = ny * field_width + nx;

      field.data[out_idx] = (field.data[out_idx] > prob) ? field.data[out_idx] : prob;
    }
  }

  splatting_time = std::chrono::steady_clock::now();

  if (debug_timings) {
    const double preparation_ms =
      std::chrono::duration<double, std::milli>(preparation_time - start_time).count();
    const double kernel_ms =
      std::chrono::duration<double, std::milli>(kernel_time - preparation_time).count();
    const double splatting_ms =
      std::chrono::duration<double, std::milli>(splatting_time - kernel_time).count();
    const double total_ms =
      std::chrono::duration<double, std::milli>(splatting_time - start_time).count();

    SAM_INFO(
      "SubmapGrid likelihood field generation [ms]: preparation={}, kernel={}, splatting={}, "
      "total={}",
      preparation_ms, kernel_ms, splatting_ms, total_ms);
  }

  return field;
}

}  // namespace glidar_slam::core
