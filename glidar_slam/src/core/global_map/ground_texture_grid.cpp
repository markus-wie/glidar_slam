#include "glidar_slam/core/global_map/ground_texture_grid.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace glidar_slam::core::global_map {

GroundTextureGrid::GroundTextureGrid(const std::shared_ptr<Parameters> & parameters)
: GlobalMap(parameters->ground_map_resolution), parameters_(parameters)
{
}

}  // namespace glidar_slam::core::global_map
