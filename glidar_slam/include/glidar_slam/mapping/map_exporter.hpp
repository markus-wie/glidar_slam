#pragma once

#include <string>

#include "glidar_slam/mapping/map_builder.hpp"

namespace glidar_slam::mapping {

/**
 * Writes layers of a global map snapshot to disk.
 *
 * The occupancy grid is written to `<base_filepath>.pgm` and `<base_filepath>.yaml`, the ground
 * texture to `<base_filepath>_texture.png` and `<base_filepath>_texture.yaml`.
 *
 * A layer of a snapshot can be missing, because it is still empty, disabled or exceeds the size
 * limit of the global map. Nothing is written if a requested layer is missing.
 *
 * @return True on success, otherwise `error` describes the failure.
 */
bool saveGlobalMaps(
  const GlobalMapSnapshot & snapshot, const std::string & base_filepath, bool save_occupancy,
  bool save_texture, std::string * error = nullptr);

}  // namespace glidar_slam::mapping
