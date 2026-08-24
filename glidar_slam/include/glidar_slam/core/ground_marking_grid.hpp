#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "glidar_slam/core/occupancy_grid.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace glidar_slam::core {

class GroundMarkingGrid
{
public:
  explicit GroundMarkingGrid(const std::shared_ptr<Parameters> & parameters);

  bool buildFromGroundClouds(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>> & clouds);

  const OccupancyGrid::Info & getInfo() const
  {
    return info_;
  }
  const std::vector<int8_t> & getData() const
  {
    return data_;
  }

private:
  bool isWhiteMarking(const pcl::PointXYZRGB & point) const;

  OccupancyGrid::Info info_;
  std::vector<int8_t> data_;
  std::shared_ptr<Parameters> parameters_;
};

}  // namespace glidar_slam::core
