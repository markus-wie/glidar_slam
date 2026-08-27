#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "glidar_slam/core/global_map.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam::core::global_map {

class OccupancyGrid : public GlobalMap
{
public:
  using Info = glidar_slam::core::GlobalMap::Info;

  explicit OccupancyGrid(const std::shared_ptr<Parameters> & parameters);

  const std::vector<int8_t> & getData() const
  {
    return data_;
  }

private:
  std::vector<int8_t> data_;

  std::shared_ptr<Parameters> parameters_;
};

}  // namespace glidar_slam::core::global_map
