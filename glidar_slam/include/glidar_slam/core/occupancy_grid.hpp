#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam::core {

class OccupancyGrid
{
public:
  struct Info
  {
    uint32_t width{0};
    uint32_t height{0};
    double resolution{0.05};
    double origin_x{0.0};
    double origin_y{0.0};
  };

  explicit OccupancyGrid(const std::shared_ptr<Parameters> & parameters);

  /// @brief Builds the grid from transformed scans. Returns false if scans are empty.
  bool buildFromScans(
    const std::vector<std::pair<gtsam::Pose3, PointCloudXYZ>> & scans_transformed);

  const Info & getInfo() const
  {
    return info_;
  }
  const std::vector<int8_t> & getData() const
  {
    return data_;
  }

private:
  Info info_;
  std::vector<int8_t> data_;

  std::shared_ptr<Parameters> parameters_;

  size_t indexOf(int cx, int cy) const
  {
    return static_cast<size_t>(cy) * static_cast<size_t>(info_.width) + static_cast<size_t>(cx);
  }

  template <typename Visitor>
  void raytraceLine(int x0, int y0, int x1, int y1, Visitor && visit) const;
};

}  // namespace glidar_slam::core