#include "glidar_slam/core/global_map/ground_marking_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace glidar_slam::core::global_map {

GroundMarkingGrid::GroundMarkingGrid(const std::shared_ptr<Parameters> & parameters)
: GlobalMap(parameters->mapping_ground_resolution), parameters_(parameters)
{
}

}  // namespace glidar_slam::core::global_map
