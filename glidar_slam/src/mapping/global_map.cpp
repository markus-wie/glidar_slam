#include "glidar_slam/mapping/global_map.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "glidar_slam/parameters.hpp"

namespace glidar_slam::mapping {

GlobalMap::GlobalMap(Info info) : info_(info)
{
}

OccupancyGrid::OccupancyGrid(Info info, std::vector<int8_t> data)
: GlobalMap(info), data_(std::move(data))
{
}

GroundMarkingGrid::GroundMarkingGrid(Info info, std::vector<int8_t> data)
: GlobalMap(info), data_(std::move(data))
{
}

GroundTextureGrid::GroundTextureGrid(Info info, std::vector<uint8_t> rgb_data)
: GlobalMap(info), rgb_data_(std::move(rgb_data))
{
}

}  // namespace glidar_slam::mapping
