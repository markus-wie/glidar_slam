#pragma once

#include <vector>

namespace glidar_slam {

class LikelihoodField
{
public:
  double getScore(double x, double y) const;

  std::vector<float> data;
  double origin_x{0.0};
  double origin_y{0.0};
  int width{0};
  int height{0};
  int max_x_index{0};
  int max_y_index{0};
  double resolution{0.0};
};

}  // namespace glidar_slam
