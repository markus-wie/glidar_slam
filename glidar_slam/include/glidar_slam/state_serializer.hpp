#pragma once

#include <filesystem>
#include <string>

#include "glidar_slam/state_snapshot.hpp"

namespace glidar_slam {

class StateSerializer
{
public:
  static bool save(
    const std::filesystem::path & path, const SlamStateSnapshot & snapshot,
    std::string * error = nullptr);
  static bool load(
    const std::filesystem::path & path, SlamStateSnapshot & snapshot,
    std::string * error = nullptr);
};

}  // namespace glidar_slam
