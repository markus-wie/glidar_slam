#include "glidar_slam/submap_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "glidar_slam/logger/logger.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace glidar_slam {

namespace {

constexpr double MAXIMUM_POINT_COORDINATE = 2000.0;

bool isUsablePoint(const Point2D & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::abs(point.x) < MAXIMUM_POINT_COORDINATE &&
         std::abs(point.y) < MAXIMUM_POINT_COORDINATE;
}

int cellCoordinate(double value, double resolution)
{
  return static_cast<int>(std::floor(value / resolution));
}

constexpr std::int64_t MAXIMUM_GRID_CELLS = 25'000'000;

bool fitsGridLimit(std::int64_t width, std::int64_t height)
{
  return width * height <= MAXIMUM_GRID_CELLS;
}

bool resizeGrid(
  SubmapGrid::HitCountGrid & grid, int required_min_x, int required_max_x, int required_min_y,
  int required_max_y)
{
  const bool has_storage = grid.width > 0 && grid.height > 0;
  if (
    has_storage && required_min_x >= grid.origin_x && required_max_x < grid.origin_x + grid.width &&
    required_min_y >= grid.origin_y && required_max_y < grid.origin_y + grid.height) {
    return true;
  }

  std::int64_t new_min_x = required_min_x;
  std::int64_t new_max_x = required_max_x;
  std::int64_t new_min_y = required_min_y;
  std::int64_t new_max_y = required_max_y;

  if (has_storage) {
    new_min_x = std::min<std::int64_t>(new_min_x, grid.origin_x);
    new_max_x = std::max<std::int64_t>(new_max_x, grid.origin_x + grid.width - 1);
    new_min_y = std::min<std::int64_t>(new_min_y, grid.origin_y);
    new_max_y = std::max<std::int64_t>(new_max_y, grid.origin_y + grid.height - 1);

    if (!fitsGridLimit(new_max_x - new_min_x + 1, new_max_y - new_min_y + 1)) {
      // The storage only ever grows and also covers the area the sliding window has left. Forget
      // that area before giving up, only cells that still hold hits have to be kept.
      new_min_x = required_min_x;
      new_max_x = required_max_x;
      new_min_y = required_min_y;
      new_max_y = required_max_y;
      for (const auto & cell : grid.active_cells) {
        new_min_x = std::min<std::int64_t>(new_min_x, cell.x);
        new_max_x = std::max<std::int64_t>(new_max_x, cell.x);
        new_min_y = std::min<std::int64_t>(new_min_y, cell.y);
        new_max_y = std::max<std::int64_t>(new_max_y, cell.y);
      }
    }
  }

  const std::int64_t required_width = new_max_x - new_min_x + 1;
  const std::int64_t required_height = new_max_y - new_min_y + 1;
  if (!fitsGridLimit(required_width, required_height)) {
    return false;
  }

  // Leave room to grow, so that not every new keyframe causes a reallocation.
  std::int64_t new_width = required_width;
  std::int64_t new_height = required_height;
  if (has_storage) {
    new_width = std::max<std::int64_t>(required_width, grid.width * 2LL);
    new_height = std::max<std::int64_t>(required_height, grid.height * 2LL);
    if (!fitsGridLimit(new_width, new_height)) {
      new_width = required_width * 5 / 4;
      new_height = required_height * 5 / 4;
    }
    if (!fitsGridLimit(new_width, new_height)) {
      new_width = required_width;
      new_height = required_height;
    }
  }
  new_min_x -= (new_width - required_width) / 2;
  new_min_y -= (new_height - required_height) / 2;

  std::vector<int> new_counts(static_cast<std::size_t>(new_width * new_height), 0);
  std::vector<int> new_active_index(static_cast<std::size_t>(new_width * new_height), 0);

  // Only active cells hold hits, all other cells are zero.
  for (const auto & cell : grid.active_cells) {
    const int old_index = grid.flatIdx(cell.x, cell.y);
    const std::int64_t new_index = (cell.y - new_min_y) * new_width + (cell.x - new_min_x);
    new_counts[static_cast<std::size_t>(new_index)] = grid.counts[old_index];
    new_active_index[static_cast<std::size_t>(new_index)] = grid.active_index[old_index];
  }

  grid.origin_x = static_cast<int>(new_min_x);
  grid.origin_y = static_cast<int>(new_min_y);
  grid.width = static_cast<int>(new_width);
  grid.height = static_cast<int>(new_height);
  grid.counts = std::move(new_counts);
  grid.active_index = std::move(new_active_index);

  return true;
}

void buildDistanceTransformField(
  LikelihoodField & field, const SubmapGrid::HitCountGrid & grid, double smear_deviation,
  bool use_laplace_kernel)
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
        const double distance = pixel_dist * resolution;
        const double exponent = use_laplace_kernel
                                  ? -distance / smear_deviation
                                  : -distance_sq / (2.0 * std::pow(smear_deviation, 2));
        field.data[row_offset + x] = static_cast<float>(std::exp(exponent));
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
  // Resize every resolution first so that a keyframe is either added to all grids or to none.
  // remove() relies on this to undo exactly what add() did.
  for (auto & [resolution, grid] : grids_) {
    int min_x = 0;
    int max_x = 0;
    int min_y = 0;
    int max_y = 0;
    bool has_finite_point = false;

    for (const auto & point : points) {
      if (!isUsablePoint(point)) {
        continue;
      }

      const GridIndex idx{
        cellCoordinate(point.x, grid.resolution), cellCoordinate(point.y, grid.resolution)};
      if (!has_finite_point) {
        min_x = max_x = idx.x;
        min_y = max_y = idx.y;
        has_finite_point = true;
      } else {
        min_x = std::min(min_x, idx.x);
        max_x = std::max(max_x, idx.x);
        min_y = std::min(min_y, idx.y);
        max_y = std::max(max_y, idx.y);
      }
    }

    if (!has_finite_point) {
      continue;
    }
    if (!resizeGrid(grid, min_x, max_x, min_y, max_y)) {
      SAM_WARN(
        "Submap grid would exceed the size limit ({} to {}, {} to {} cells), skipping keyframe {}.",
        min_x, max_x, min_y, max_y, keyframe_id);
      return;
    }
  }

  for (auto & [resolution, grid] : grids_) {
    for (const auto & point : points) {
      if (!isUsablePoint(point)) {
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

  {
    std::lock_guard<std::mutex> lock(field_cache_mutex_);
    field_cache_.clear();
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
      if (!isUsablePoint(point)) {
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

  {
    std::lock_guard<std::mutex> lock(field_cache_mutex_);
    field_cache_.clear();
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

void SubmapGrid::reset()
{
  {
    std::lock_guard<std::mutex> lock(field_cache_mutex_);
    field_cache_.clear();
  }

  active_keyframes_.clear();
  cached_world_points_.clear();

  for (auto & [resolution, grid] : grids_) {
    grid.counts.clear();
    grid.active_index.clear();
    grid.active_cells.clear();
    grid.origin_x = 0;
    grid.origin_y = 0;
    grid.width = 0;
    grid.height = 0;
  }
}

size_t SubmapGrid::size() const
{
  return active_keyframes_.size();
}

std::shared_ptr<LikelihoodField> SubmapGrid::getLikelihoodField(
  double resolution, double smear_deviation, bool use_distance_transform, bool use_laplace_kernel,
  bool debug_timings) const
{
  if (grids_.find(resolution) == grids_.end()) {
    throw std::invalid_argument("Requested resolution is not configured in SubmapGrid");
  }

  {
    std::lock_guard<std::mutex> lock(field_cache_mutex_);
    auto it = field_cache_.find(resolution);
    if (it != field_cache_.end()) {
      return it->second;
    }
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
    std::lock_guard<std::mutex> lock(field_cache_mutex_);
    field_cache_.clear();
    return std::make_shared<LikelihoodField>(field);
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
    buildDistanceTransformField(field, grid, smear_deviation, use_laplace_kernel);

    std::lock_guard<std::mutex> lock(field_cache_mutex_);
    field_cache_.insert({resolution, std::make_shared<LikelihoodField>(field)});

    return field_cache_.at(resolution);
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
        const double distance = std::sqrt(dist_sq);
        const double exponent = use_laplace_kernel ? -distance / smear_deviation : -dist_sq / denom;
        kernel_offsets.push_back({dx, dy, static_cast<float>(std::exp(exponent))});
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

  std::lock_guard<std::mutex> lock(field_cache_mutex_);
  field_cache_.insert({resolution, std::make_shared<LikelihoodField>(field)});

  return field_cache_.at(resolution);
}

}  // namespace glidar_slam
