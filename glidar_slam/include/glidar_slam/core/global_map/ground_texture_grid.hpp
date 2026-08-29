#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "glidar_slam/core/global_map.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::core::global_map {

class GroundTextureGrid : public GlobalMap
{
public:
  explicit GroundTextureGrid(const std::shared_ptr<Parameters> & parameters);

  const std::vector<std::uint8_t> & getRgbData() const
  {
    return rgb_data_;
  }

  void setGrid(const Info & info, std::vector<std::uint8_t> rgb_data)
  {
    setInfo(info);
    rgb_data_ = std::move(rgb_data);
  }

private:
  std::vector<std::uint8_t> rgb_data_;
  std::shared_ptr<Parameters> parameters_;
};

}  // namespace glidar_slam::core::global_map
