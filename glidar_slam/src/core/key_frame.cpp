#include "glidar_slam/core/key_frame.hpp"

#include "glidar_slam/core/mapping/local_map.hpp"

namespace glidar_slam::core {

void KeyFrame::buildLocalMaps(const Parameters & parameters)
{
  if (scan) {
    local_occupancy = std::make_shared<const mapping::LocalOccupancyMap>(
      mapping::buildOccupancy(scan->points(), parameters));
  }

  if (ground_observation) {
    local_ground = std::make_shared<const mapping::LocalGroundMap>(
      mapping::buildGround(ground_observation->ground_cloud, parameters));
  }
}

}  // namespace glidar_slam::core
