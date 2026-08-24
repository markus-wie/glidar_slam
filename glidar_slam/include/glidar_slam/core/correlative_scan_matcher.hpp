#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "glidar_slam/core/likelihood_field.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/submap_grid.hpp"
#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core {

constexpr double MAX_VARIANCE = 500.0;

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
    const SubmapGrid & submap_grid, const std::vector<Point2D> & current_points,
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
    std::size_t candidate_count{0};
    std::vector<SearchResponse> responses;
  };

  struct SearchGrid
  {
    std::vector<double> yaw_positions;
    std::vector<double> x_positions;
    std::vector<double> y_positions;
    std::size_t candidates_per_yaw{0};
  };

  SearchResult searchSpace(
    const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & center,
    const CsmSearchStage & stage) const;

  void evaluateYawSlice(
    const std::vector<Point2D> & points, const LikelihoodField & field, const SearchGrid & grid,
    std::size_t yaw_index, SearchResult & result) const;

  LikelihoodField buildField(const std::vector<Point2D> & points, double resolution) const;

  void buildDistanceTransformField(
    LikelihoodField & field, const std::vector<Point2D> & points) const;

  std::shared_ptr<const std::vector<LikelihoodField>> getLikelihoodFields(
    const SubmapGrid & submap_grid, const std::vector<CsmSearchStage> & stages) const;

  static void selectBestPose(SearchResult & result);

  static Eigen::Matrix3d computeCovariance(
    const std::vector<SearchResult> & results, const std::vector<CsmSearchStage> & stages);

  std::shared_ptr<Parameters> params_;
};

}  // namespace glidar_slam::core
