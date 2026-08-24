#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "glidar_slam/core/global_map.hpp"
#include "glidar_slam/core/parameters.hpp"
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

  const std::vector<std::int8_t> & getCoverageData() const
  {
    return coverage_data_;
  }

  void setGrid(
    const Info & info, std::vector<std::uint8_t> rgb_data, std::vector<std::int8_t> coverage_data)
  {
    setInfo(info);
    rgb_data_ = std::move(rgb_data);
    coverage_data_ = std::move(coverage_data);
  }

private:
  std::vector<std::uint8_t> rgb_data_;
  std::vector<std::int8_t> coverage_data_;
  std::shared_ptr<Parameters> parameters_;
};

}  // namespace glidar_slam::core::global_map
