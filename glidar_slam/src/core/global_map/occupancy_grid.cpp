#include "glidar_slam/core/global_map/occupancy_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace glidar_slam::core::global_map {

OccupancyGrid::OccupancyGrid(const std::shared_ptr<Parameters> & parameters)
: GlobalMap(parameters->occ_map_resolution), parameters_(parameters)
{
}

}  // namespace glidar_slam::core::global_map
