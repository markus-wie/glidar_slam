#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "glidar_slam/core/occupancy_grid.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::core {

class GroundTextureGrid
{
public:
  explicit GroundTextureGrid(const std::shared_ptr<Parameters> & parameters);

  bool buildFromGroundClouds(const std::vector<pcl::PointCloud<pcl::PointXYZRGBA>> & clouds);

  const OccupancyGrid::Info & getInfo() const
  {
    return info_;
  }

  const std::vector<std::uint8_t> & getRgbData() const
  {
    return rgb_data_;
  }

  const std::vector<std::int8_t> & getCoverageData() const
  {
    return coverage_data_;
  }

private:
  OccupancyGrid::Info info_;
  std::vector<std::uint8_t> rgb_data_;
  std::vector<std::int8_t> coverage_data_;
  std::shared_ptr<Parameters> parameters_;
};

}  // namespace glidar_slam::core
