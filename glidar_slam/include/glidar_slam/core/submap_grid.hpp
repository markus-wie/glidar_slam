#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "glidar_slam/core/likelihood_field.hpp"
#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core {

class SubmapGrid
{
public:
  explicit SubmapGrid(const std::vector<double> & resolutions);

  void add(const std::vector<Point2D> & points, uint64_t keyframe_id);
  void remove(uint64_t keyframe_id);

  void removeOldestKeyframe();

  size_t size() const;

  std::shared_ptr<LikelihoodField> getLikelihoodField(
    double resolution, double smear_deviation, bool use_distance_transform, bool use_laplace_kernel,
    bool debug_timings = false) const;

  struct GridIndex
  {
    int x{0};
    int y{0};
    bool operator==(const GridIndex & other) const
    {
      return x == other.x && y == other.y;
    }
  };

  struct GridIndexHash
  {
    std::size_t operator()(const GridIndex & idx) const
    {
      uint64_t packed = (static_cast<uint64_t>(static_cast<uint32_t>(idx.x)) << 32) |
                        (static_cast<uint64_t>(static_cast<uint32_t>(idx.y)));
      return std::hash<uint64_t>{}(packed);
    }
  };

  struct CellState
  {
    int count;
    int active_index;
  };

  struct HitCountGrid
  {
    double resolution{0.0};

    int origin_x{0};
    int origin_y{0};
    int width{0};
    int height{0};

    std::vector<int> counts;
    std::vector<int> active_index;
    std::vector<GridIndex> active_cells;  // Dynamic sparse set

    // O(1) bounds-relative indexing.
    int flatIdx(int x, int y) const
    {
      return (y - origin_y) * width + (x - origin_x);
    }
  };

private:
  std::unordered_map<double, HitCountGrid> grids_;

  mutable std::unordered_map<double, std::shared_ptr<LikelihoodField>> field_cache_;
  mutable std::mutex field_cache_mutex_;

  std::deque<uint64_t> active_keyframes_;
  std::unordered_map<uint64_t, std::vector<Point2D>> cached_world_points_;
};

}  // namespace glidar_slam::core
