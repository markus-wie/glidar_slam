#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core {
class LaserScan
{
public:
  LaserScan() = default;

  explicit LaserScan(PointCloudXYZ points);
  explicit LaserScan(std::vector<Point2D> points);

  bool empty() const;
  const PointCloudXYZ & points() const;
  const std::vector<Point2D> & points2D() const;

private:
  mutable std::optional<PointCloudXYZ> points_;
  mutable std::optional<std::vector<Point2D>> points2d_;
};
}  // namespace glidar_slam::core
