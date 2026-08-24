#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core {
struct LaserScan
{
  uint64_t id;
  double timestamp;
  std::vector<Point2D> points;
  Pose2D odom_pose;
  Pose2D world_pose;
};

struct CsmResult
{
  Pose2D optimized_pose;
  double score;
  Eigen::Matrix3d covariance;
  struct DebugImage
  {
    std::vector<std::uint8_t> pixels;
    double origin_x = 0.0;
    double origin_y = 0.0;
    double resolution = 0.0;
    int width = 0;
    int height = 0;
  };

  DebugImage low_res_debug;
  DebugImage high_res_debug;
};

struct LikelihoodField
{
  std::vector<double> data;
  double origin_x;
  double origin_y;
  int width;
  int height;
  double resolution;

  // Bilinear interpolation for sub-grid scoring
  double getScore(double x, double y) const;
};

class CorrelativeScanMatcher
{
public:
  static CsmResult match(
    const LaserScan & reference, const LaserScan & current, const Parameters & params);

private:
  struct SearchResponse
  {
    Pose2D pose{0.0, 0.0, 0.0};
    double score{0.0};
  };

  struct SearchResult
  {
    Pose2D center;
    Pose2D best_pose;
    double best_score{0.0};
    std::vector<SearchResponse> responses;
  };

  static LikelihoodField buildField(
    const std::vector<Point2D> & points, double resolution, const Parameters & params);

  static SearchResult searchSpace(
    const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & center,
    const CsmSearchStage & stage, const Parameters & params);

  static double evaluatePose(
    const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & pose);

  static Eigen::Matrix3d computeCovariance(
    const std::vector<SearchResult> & results, const std::vector<CsmSearchStage> & stages);
};

}  // namespace glidar_slam::core
