#include "glidar_slam/core/laser_scan.hpp"

#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core {

LaserScan::LaserScan(PointCloudXYZPtr points) : points_(std::move(points))
{
}

LaserScan::LaserScan(std::vector<Point2D> points) : points2d_(std::move(points))
{
}

bool LaserScan::empty() const
{
  return points2D().empty();
}

PointCloudXYZConstPtr LaserScan::points() const
{
  if (!points_) {
    if (!points2d_.has_value()) {
      points_ = std::make_shared<PointCloudXYZ>();
      return points_;
    }

    PointCloudXYZPtr points = std::make_shared<PointCloudXYZ>();
    points->reserve(points2d_->size());
    for (const Point2D & point : *points2d_) {
      points->push_back({static_cast<float>(point.x), static_cast<float>(point.y), 0.0F});
    }
    points->width = static_cast<std::uint32_t>(points->size());
    points->height = 1;
    points->is_dense = true;
    points_ = std::move(points);
  }
  return points_;
}

const std::vector<Point2D> & LaserScan::points2D() const
{
  if (!points2d_.has_value()) {
    if (!points_) {
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
