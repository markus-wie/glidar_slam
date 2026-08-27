#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "glidar_slam/core/global_map.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::core::global_map {

class GroundMarkingGrid : public GlobalMap
{
public:
  explicit GroundMarkingGrid(const std::shared_ptr<Parameters> & parameters);

  const std::vector<int8_t> & getData() const
  {
    return data_;
  }

private:
  std::vector<int8_t> data_;
  std::shared_ptr<Parameters> parameters_;
};

}  // namespace glidar_slam::core::global_map
