#pragma once

#include <cstdint>
#include <vector>

#include "glidar_slam/parameters.hpp"
#include "glidar_slam/types.hpp"

namespace glidar_slam::mapping {

struct LocalOccupancyMap
{
  struct Cell
  {
    int x{0};
    int y{0};
    float log_odds{0.0f};
  };

  double resolution{0.0};
  std::vector<Cell> cells;
};

struct LocalGroundMap
{
  struct Cell
  {
    int x{0};
    int y{0};
    float log_odds{0.0f};
    std::uint8_t r{0};
    std::uint8_t g{0};
    std::uint8_t b{0};
  };

  double resolution{0.0};
  std::vector<Cell> cells;
};

LocalOccupancyMap buildOccupancy(const PointCloudXYZConstPtr & scan, const Parameters & parameters);

LocalGroundMap buildGround(const PointCloudXYZRGBAConstPtr & cloud, const Parameters & parameters);

}  // namespace glidar_slam::mapping
