#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core::global_map {

struct OccupancyCellContribution
{
  int x{0};
  int y{0};
  int evidence{0};
};

struct GroundCell
{
  int x{0};
  int y{0};
  float log_odds{0.0F};
  std::uint8_t r{0};
  std::uint8_t g{0};
  std::uint8_t b{0};
};

struct LocalMapData
{
  double resolution{0.05};
  std::vector<OccupancyCellContribution> cells;
  double ground_resolution{0.02};
  std::vector<GroundCell> ground_cells;
};

}  // namespace glidar_slam::core::global_map
