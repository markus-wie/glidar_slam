#include "glidar_slam/core/laser_scan.hpp"

namespace glidar_slam::core {

LaserScan::LaserScan(PointCloudXYZ points) : points_(std::move(points))
{
}

LaserScan::LaserScan(std::vector<Point2D> points) : points2d_(std::move(points))
{
}

bool LaserScan::empty() const
{
  return points2D().empty();
}

const PointCloudXYZ & LaserScan::points() const
{
  if (!points_.has_value()) {
    if (!points2d_.has_value()) {
      points_ = PointCloudXYZ();
      return *points_;
    }

    PointCloudXYZ points;
    points.reserve(points2d_->size());
    for (const Point2D & point : *points2d_) {
      points.push_back({static_cast<float>(point.x), static_cast<float>(point.y), 0.0F});
    }
    points.width = static_cast<std::uint32_t>(points.size());
    points.height = 1;
    points.is_dense = true;
    points_ = std::move(points);
  }
  return *points_;
}

const std::vector<Point2D> & LaserScan::points2D() const
{
  if (!points2d_.has_value()) {
    if (!points_.has_value()) {
      points2d_ = std::vector<Point2D>();
      return *points2d_;
    }

    std::vector<Point2D> points;
    points.reserve(points_->size());
    for (const auto & point : *points_) {
      points.push_back({point.x, point.y});
    }
    points2d_ = std::move(points);
  }
  return *points2d_;
}

}  // namespace glidar_slam::core
