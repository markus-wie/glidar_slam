#include "glidar_slam/core/mapping/local_map.hpp"

#include <unordered_set>

#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"
#include "glidar_slam/core/utils.hpp"

namespace glidar_slam::core::mapping {

namespace {
std::int64_t cellKey(int x, int y)
{
  return (static_cast<std::int64_t>(x) << 32) ^ static_cast<std::uint32_t>(y);
}

int cellCoordinate(float value, double resolution)
{
  return static_cast<int>(std::floor(static_cast<double>(value) / resolution));
}

}  // namespace

LocalOccupancyMap buildOccupancy(const PointCloudXYZConstPtr & scan, const Parameters & parameters)
{
  LocalOccupancyMap result;
  result.resolution = parameters.mapping_occupancy_resolution;
  if (result.resolution <= 0.0) {
    return result;
  }

  const float hit_log_odds =
    utils::logOddsFromProb(static_cast<float>(parameters.mapping_prob_hit));
  const float miss_log_odds =
    utils::logOddsFromProb(static_cast<float>(parameters.mapping_prob_miss));

  std::unordered_set<std::int64_t> hits;
  std::unordered_set<std::int64_t> misses;

  const float res = result.resolution;
  const int origin_x = cellCoordinate(0.0f, res);
  const int origin_y = cellCoordinate(0.0f, res);

  for (const auto & point : scan->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      continue;
    }

    const int target_x = cellCoordinate(point.x, res);
    const int target_y = cellCoordinate(point.y, res);

    hits.insert(cellKey(target_x, target_y));

    // Continuous DDA Raytracing from (0.0, 0.0) to (point.x, point.y)
    int x = origin_x;
    int y = origin_y;

    const float dx = point.x;
    const float dy = point.y;

    const int stepX = (dx > 0.0f) ? 1 : ((dx < 0.0f) ? -1 : 0);
    const int stepY = (dy > 0.0f) ? 1 : ((dy < 0.0f) ? -1 : 0);

    // Distance to the first grid boundary
    const float next_bnd_x = (x + (stepX > 0 ? 1 : 0)) * res;
    const float next_bnd_y = (y + (stepY > 0 ? 1 : 0)) * res;

    // tMax: T-parameter value to reach the next voxel boundary
    float tMaxX = (stepX != 0) ? (next_bnd_x / dx) : std::numeric_limits<float>::infinity();
    float tMaxY = (stepY != 0) ? (next_bnd_y / dy) : std::numeric_limits<float>::infinity();

    // tDelta: T-parameter interval to cross exactly one voxel
    const float tDeltaX =
      (stepX != 0) ? (res / std::abs(dx)) : std::numeric_limits<float>::infinity();
    const float tDeltaY =
      (stepY != 0) ? (res / std::abs(dy)) : std::numeric_limits<float>::infinity();

    while (x != target_x || y != target_y) {
      if (tMaxX < tMaxY) {
        tMaxX += tDeltaX;
        x += stepX;
      } else {
        tMaxY += tDeltaY;
        y += stepY;
      }

      if (x == target_x && y == target_y) {
        break;
      }
      misses.insert(cellKey(x, y));
    }
  }

  // Deduplicate: Hits take priority over misses in the same scan
  for (const auto & hit_key : hits) {
    misses.erase(hit_key);
  }

  // Populate result
  result.cells.reserve(hits.size() + misses.size());

  auto push_evidence = [&result](std::int64_t key, float log_odds) {
    if (std::abs(log_odds) < 1e-5f) {
      return;
    }
    result.cells.push_back(
      {static_cast<int>(key >> 32), static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff)),
       log_odds});
  };

  for (const auto & key : hits) {
    push_evidence(key, hit_log_odds);
  }
  for (const auto & key : misses) {
    push_evidence(key, miss_log_odds);
  }

  return result;
}

LocalGroundMap buildGround(const PointCloudXYZRGBAConstPtr & cloud, const Parameters & parameters)
{
  LocalGroundMap result;
  result.resolution = parameters.mapping_ground_resolution;
  if (result.resolution <= 0.0) {
    return result;
  }

  const float hit =
    static_cast<float>(std::log(parameters.mapping_prob_hit / (1.0 - parameters.mapping_prob_hit)));
  const float miss = static_cast<float>(
    std::log(parameters.mapping_prob_miss / (1.0 - parameters.mapping_prob_miss)));

  result.cells.reserve(cloud->points.size());

  const int threshold = static_cast<int>(parameters.mapping_occupancy_threshold * 255);

  for (const auto & point : cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      continue;
    }

    result.cells.push_back(
      {static_cast<int>(std::floor(point.x / result.resolution)),
       static_cast<int>(std::floor(point.y / result.resolution)), point.a >= threshold ? hit : miss,
       point.r, point.g, point.b});
  }

  return result;
}

}  // namespace glidar_slam::core::mapping
