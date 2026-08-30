#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "glidar_slam/types.hpp"

namespace glidar_slam {
class LaserScan
{
public:
  LaserScan() = default;

  explicit LaserScan(PointCloudXYZPtr points);
  explicit LaserScan(std::vector<Point2D> points);

  bool empty() const;
  PointCloudXYZConstPtr points() const;
  const std::vector<Point2D> & points2D() const;

private:
  mutable PointCloudXYZPtr points_;
  mutable std::optional<std::vector<Point2D>> points2d_;
};
}  // namespace glidar_slam
