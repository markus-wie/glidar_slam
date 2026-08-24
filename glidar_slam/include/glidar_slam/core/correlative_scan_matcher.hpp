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
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/types.hpp"

namespace glidar_slam::core {
struct LikelihoodField
{
  std::vector<float> data;
  double origin_x;
  double origin_y;
  int width;
  int height;
  int max_x_index;
  int max_y_index;
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

  LikelihoodField buildField(const std::vector<Point2D> & points, double resolution) const;

  void buildDistanceTransformField(
    LikelihoodField & field, const std::vector<Point2D> & points) const;

  std::shared_ptr<const std::vector<LikelihoodField>> getLikelihoodFields(
    const std::vector<Point2D> & reference_points,
    const std::vector<CsmSearchStage> & stages) const;

  static bool referencePointsEqual(
    const std::vector<Point2D> & lhs, const std::vector<Point2D> & rhs);

  static double maximumSearchWindow(const std::vector<CsmSearchStage> & stages);

  static std::vector<double> makeSearchPositions(double center, double window, double step);

  SearchResult searchSpace(
    const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & center,
    const CsmSearchStage & stage) const;

  void evaluateYawSlice(
    const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & center,
    const SearchGrid & grid, std::size_t yaw_index, SearchResult & result) const;

  double scoreCandidate(
    const std::vector<Point2D> & points, const LikelihoodField & field, double x, double y,
    const Pose2D & center, double angle_penalty) const;

  static void selectBestPose(SearchResult & result);

  static double evaluatePose(
    const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & pose);

  static Eigen::Matrix3d computeCovariance(
    const std::vector<SearchResult> & results, const std::vector<CsmSearchStage> & stages);

  std::shared_ptr<Parameters> params_;

  mutable std::vector<Point2D> cached_reference_points_;
  mutable std::shared_ptr<const std::vector<LikelihoodField>> cached_fields_;
};

}  // namespace glidar_slam::core
