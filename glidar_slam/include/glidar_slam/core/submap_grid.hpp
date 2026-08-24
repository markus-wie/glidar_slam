#pragma once

#include <deque>
#include <unordered_map>

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

  LikelihoodField getLikelihoodField(
    double resolution, double smear_deviation, bool use_distance_transform,
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

    // 2048 x 2048 cells = ~100x100 meters at 5cm resolution.
    static constexpr int BITS = 11;
    static constexpr int WIDTH = 1 << BITS;
    static constexpr int MASK = WIDTH - 1;

    std::vector<int> counts;              // Fixed size
    std::vector<int> active_index;        // Fixed size
    std::vector<GridIndex> active_cells;  // Dynamic sparse set

    // Initialize fixed arrays to 0 once at startup
    HitCountGrid() : counts(WIDTH * WIDTH, 0), active_index(WIDTH * WIDTH, 0)
    {
    }

    // O(1) Bitwise Toroidal Indexing
    inline int flatIdx(int x, int y) const
    {
      return ((y & MASK) << BITS) | (x & MASK);
    }
  };

private:
  std::unordered_map<double, HitCountGrid> grids_;

  std::deque<uint64_t> active_keyframes_;
  std::unordered_map<uint64_t, std::vector<Point2D>> cached_world_points_;
};

}  // namespace glidar_slam::core