#pragma once

#include <vector>

#include "glidar_slam/submap_grid.hpp"
#include "glidar_slam/types.hpp"

namespace glidar_slam {

struct CsmResult;

class ScanMatcher
{
public:
  virtual ~ScanMatcher() = default;

  virtual CsmResult match(
    const SubmapGrid & submap_grid, const std::vector<Point2D> & current_points,
    const Pose2D & pose_estimate) const = 0;

  virtual std::vector<double> fieldResolutions() const = 0;
};

}  // namespace glidar_slam
