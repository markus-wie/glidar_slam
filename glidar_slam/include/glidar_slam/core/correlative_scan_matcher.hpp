#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core {
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

  static CsmResult::DebugImage toDebugImage(const LikelihoodField & field);
};

class CorrelativeScanMatcher
{
public:
  explicit CorrelativeScanMatcher(const std::shared_ptr<Parameters> & params);

  CsmResult match(
    const std::vector<Point2D> & reference_points, const std::vector<Point2D> & current_points,
    const Pose2D & pose_estimate) const;

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

  LikelihoodField buildField(const std::vector<Point2D> & points, double resolution) const;

  SearchResult searchSpace(
    const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & center,
    const CsmSearchStage & stage) const;

  double evaluatePose(
    const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & pose) const;

  Eigen::Matrix3d computeCovariance(
    const std::vector<SearchResult> & results, const std::vector<CsmSearchStage> & stages) const;

  std::shared_ptr<Parameters> params_;
};

}  // namespace glidar_slam::core
