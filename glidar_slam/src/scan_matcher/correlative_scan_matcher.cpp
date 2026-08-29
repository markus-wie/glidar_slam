#include "glidar_slam/scan_matcher/correlative_scan_matcher.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <fstream>
#include <limits>
#include <numeric>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "glidar_slam/logger/logger.hpp"
#include "glidar_slam/utils.hpp"
#include "tbb/parallel_for.h"

namespace glidar_slam {

namespace {

std::vector<double> makeSearchPositions(double center, double window, double step)
{
  const int half_steps = static_cast<int>(std::ceil((window / 2.0) / step));

  std::vector<double> positions;
  // +1 for the center position
  positions.reserve(2 * half_steps + 1);

  for (int i = -half_steps; i <= half_steps; ++i) {
    positions.push_back(center + i * step);
  }

  return positions;
}

double evaluatePoseScore(
  const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & pose)
{
  double score = 0.0;
  const double c = std::cos(pose.yaw);
  const double s = std::sin(pose.yaw);

  for (const auto & point : points) {
    const double eval_score =
      field.getScore(pose.x + c * point.x - s * point.y, pose.y + s * point.x + c * point.y);
    if (eval_score < 0.0) {
      continue;
    }
    score += eval_score;
  }

  return score / static_cast<double>(points.size());
}

}  // namespace

CorrelativeScanMatcher::CorrelativeScanMatcher(const std::shared_ptr<Parameters> & params)
: params_(params)
{
}

std::vector<double> CorrelativeScanMatcher::fieldResolutions() const
{
  std::vector<double> resolutions;
  resolutions.reserve(params_->csm_search_stages.size());
  for (const auto & stage : params_->csm_search_stages) {
    resolutions.push_back(stage.field_resolution);
  }
  return resolutions;
}

double LikelihoodField::getScore(double x, double y) const
{
  const double px = (x - origin_x) / resolution;
  const double py = (y - origin_y) / resolution;

  const int x0 = static_cast<int>(std::floor(px));
  const int y0 = static_cast<int>(std::floor(py));

  if (x0 < 0 || x0 >= max_x_index || y0 < 0 || y0 >= max_y_index) {
    return -1.0;
  }

  double dx = px - x0;
  double dy = py - y0;

  double s00 = data[y0 * width + x0];
  double s10 = data[y0 * width + (x0 + 1)];
  double s01 = data[(y0 + 1) * width + x0];
  double s11 = data[(y0 + 1) * width + (x0 + 1)];

  return (1.0 - dx) * (1.0 - dy) * s00 + dx * (1.0 - dy) * s10 + (1.0 - dx) * dy * s01 +
         dx * dy * s11;
}

CsmResult CorrelativeScanMatcher::match(
  const SubmapGrid & submap_grid, const std::vector<Point2D> & current_points,
  const Pose2D & pose_estimate) const
{
  const auto timing_start = std::chrono::steady_clock::now();
  CsmResult result{};
  result.optimized_pose = pose_estimate;
  result.score = 0.0;
  result.covariance = Eigen::Matrix3d::Identity();

  const std::vector<CsmSearchStage> & stages = params_->csm_search_stages;

  if (stages.empty() || current_points.empty()) {
    result.covariance(0, 0) = params_->unobservable_variance;
    result.covariance(1, 1) = params_->unobservable_variance;
    result.covariance(2, 2) = params_->unobservable_variance;
    return result;
  }

  const std::shared_ptr<const std::vector<std::shared_ptr<LikelihoodField>>> fields =
    getLikelihoodFields(submap_grid, stages);

  if (params_->debug_timings) {
    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - timing_start)
        .count();
    SAM_INFO("CSM timing: field construction={} ms", elapsed_ms);
  }

  const auto initial_score_start = std::chrono::steady_clock::now();
  const double initial_score = evaluatePoseScore(current_points, *fields->front(), pose_estimate);
  Pose2D best_pose = pose_estimate;
  if (params_->debug_timings) {
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - initial_score_start)
                                .count();
    SAM_INFO("CSM timing: initial score={} ms", elapsed_ms);
  }

  std::vector<SearchResult> results;
  results.reserve(stages.size());

  if (params_->csm_debug_enable) {
    SAM_INFO(
      "CSM pose estimate: x={}, y={}, yaw={}, score={}", pose_estimate.x, pose_estimate.y,
      pose_estimate.yaw, initial_score);
  }

  for (std::size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
    const auto & stage = stages[stage_index];

    const auto stage_start = std::chrono::steady_clock::now();
    SearchResult stage_result =
      searchSpace(current_points, *fields->at(stage_index), best_pose, stage);

    if (params_->debug_timings) {
      const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start)
          .count();
      SAM_INFO("CSM timing: search stage {}={} ms", stage_index, elapsed_ms);
    }

    best_pose = stage_result.best_pose;

    results.push_back(stage_result);

    if (params_->csm_debug_enable) {
      SAM_INFO(
        "CSM stage {}: center=({}, {}, {}), result=({}, {}, {}), score={}, candidates={}, "
        "delta_x={}, delta_y={}, delta_yaw={}",
        stage_index, stage_result.center.x, stage_result.center.y, stage_result.center.yaw,
        stage_result.best_pose.x, stage_result.best_pose.y, stage_result.best_pose.yaw,
        stage_result.best_score, stage_result.candidate_count, best_pose.x - pose_estimate.x,
        best_pose.y - pose_estimate.y, utils::normalizeAngle(best_pose.yaw - pose_estimate.yaw));
    }
  }

  Eigen::Matrix3d cov = computeCovariance(results, stages);

  if (params_->csm_debug_enable) {
    SAM_INFO(
      "CSM covariance (of best result): x={}, y={}, yaw={}", cov(0, 0), cov(1, 1), cov(2, 2));
  }

  result.optimized_pose = best_pose;
  result.score = results.back().best_score;
  result.covariance = cov;

  const auto debug_image_start = std::chrono::steady_clock::now();

  if (params_->csm_debug_enable) {
    result.low_res_debug = CsmResult::toDebugImage(*fields->front());
    result.high_res_debug = CsmResult::toDebugImage(*fields->back());
  }

  if (params_->debug_timings) {
    const double image_elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - debug_image_start)
                                      .count();
    const double total_elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - timing_start)
        .count();
    SAM_INFO("CSM timing: debug images={} ms, total={} ms", image_elapsed_ms, total_elapsed_ms);
  }

  return result;
}

std::shared_ptr<const std::vector<std::shared_ptr<LikelihoodField>>>
CorrelativeScanMatcher::getLikelihoodFields(
  const SubmapGrid & submap_grid, const std::vector<CsmSearchStage> & stages) const
{
  auto fields = std::make_shared<std::vector<std::shared_ptr<LikelihoodField>>>();
  fields->reserve(stages.size());
  for (const auto & stage : stages) {
    fields->push_back(submap_grid.getLikelihoodField(
      stage.field_resolution, params_->csm_smear_deviation, params_->csm_use_distance_transform,
      params_->csm_use_laplace_kernel, params_->debug_timings));
  }
  return fields;
}

CorrelativeScanMatcher::SearchResult CorrelativeScanMatcher::searchSpace(
  const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & center,
  const CsmSearchStage & stage) const
{
  SearchGrid grid;
  grid.yaw_positions = makeSearchPositions(center.yaw, stage.window_yaw, stage.angular_step);
  grid.x_positions = makeSearchPositions(center.x, stage.window_x, stage.translation_step);
  grid.y_positions = makeSearchPositions(center.y, stage.window_y, stage.translation_step);
  grid.candidates_per_yaw = grid.x_positions.size() * grid.y_positions.size();

  SearchResult result;
  result.center = center;
  result.best_pose = center;
  result.best_score = -std::numeric_limits<double>::infinity();
  result.candidate_count = grid.yaw_positions.size() * grid.candidates_per_yaw;
  result.responses.resize(result.candidate_count);

  if (points.empty()) {
    return result;
  }

  if (params_->csm_use_tbb) {
    tbb::parallel_for(std::size_t{0}, grid.yaw_positions.size(), [&](std::size_t yaw_index) {
      evaluateYawSlice(points, field, grid, yaw_index, result);
    });
  } else {
    for (std::size_t yaw_index = 0; yaw_index < grid.yaw_positions.size(); ++yaw_index) {
      evaluateYawSlice(points, field, grid, yaw_index, result);
    }
  }

  selectBestPose(result);
  return result;
}

void CorrelativeScanMatcher::evaluateYawSlice(
  const std::vector<Point2D> & points, const LikelihoodField & field, const SearchGrid & grid,
  std::size_t yaw_index, SearchResult & result) const
{
  const double yaw = grid.yaw_positions[yaw_index];
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);

  std::vector<Point2D> rotated_points(points.size());
  for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
    rotated_points[point_index].x = c * points[point_index].x - s * points[point_index].y;
    rotated_points[point_index].y = s * points[point_index].x + c * points[point_index].y;
  }

  // Cache field parameters locally to avoid multiple pointer-chasing steps in hot loop
  const double origin_x = field.origin_x;
  const double origin_y = field.origin_y;
  const double inv_res = 1.0 / field.resolution;
  const int field_width = field.width;
  const int max_x = field.max_x_index;
  const int max_y = field.max_y_index;
  const float * field_ptr = field.data.data();

  const std::size_t yaw_offset = yaw_index * grid.candidates_per_yaw;
  const std::size_t y_size = grid.y_positions.size();

  for (std::size_t x_index = 0; x_index < grid.x_positions.size(); ++x_index) {
    const double candidate_x = grid.x_positions[x_index];

    for (std::size_t y_index = 0; y_index < y_size; ++y_index) {
      const double candidate_y = grid.y_positions[y_index];

      double total_score = 0.0;

      for (const auto & point : rotated_points) {
        const double map_x = candidate_x + point.x;
        const double map_y = candidate_y + point.y;

        const double px = (map_x - origin_x) * inv_res;
        const double py = (map_y - origin_y) * inv_res;

        const int x0 = static_cast<int>(std::floor(px));
        const int y0 = static_cast<int>(std::floor(py));

        if (x0 >= 0 && x0 < max_x && y0 >= 0 && y0 < max_y) {
          const double dx = px - x0;
          const double dy = py - y0;

          const int idx0 = y0 * field_width + x0;
          const int idx1 = idx0 + field_width;

          const float s00 = field_ptr[idx0];
          const float s10 = field_ptr[idx0 + 1];
          const float s01 = field_ptr[idx1];
          const float s11 = field_ptr[idx1 + 1];

          total_score += (1.0 - dx) * (1.0 - dy) * s00 + dx * (1.0 - dy) * s10 +
                         (1.0 - dx) * dy * s01 + dx * dy * s11;
        }
      }

      double final_score = total_score / static_cast<double>(points.size());

      const std::size_t response_index = yaw_offset + x_index * y_size + y_index;
      result.responses[response_index] = {{candidate_x, candidate_y, yaw}, final_score};
    }
  }
}

void CorrelativeScanMatcher::selectBestPose(SearchResult & result)
{
  for (const auto & response : result.responses) {
    if (response.score > result.best_score) {
      result.best_score = response.score;
      result.best_pose = response.pose;
    }
  }

  // average all poses with same highest score
  double theta_x = 0.0;
  double theta_y = 0.0;
  double theta_sin = 0.0;
  double theta_cos = 0.0;
  std::size_t theta_count = 0;
  for (const auto & response : result.responses) {
    if (utils::doubleEqual(response.score, result.best_score)) {
      theta_x += response.pose.x;
      theta_y += response.pose.y;
      theta_sin += std::sin(response.pose.yaw);
      theta_cos += std::cos(response.pose.yaw);
      ++theta_count;
    }
  }

  if (theta_count > 0) {
    result.best_pose.x = theta_x / static_cast<double>(theta_count);
    result.best_pose.y = theta_y / static_cast<double>(theta_count);
    result.best_pose.yaw = std::atan2(theta_sin, theta_cos);
  }
}

Eigen::Matrix3d CorrelativeScanMatcher::computeCovariance(
  const std::vector<SearchResult> & results, const std::vector<CsmSearchStage> & stages) const
{
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

  if (results.empty() || stages.empty() || results.size() != stages.size()) {
    covariance.diagonal().setConstant(params_->unobservable_variance);
    return covariance;
  }

  // Lambda to compute the 2x2 spatial covariance for a single stage
  auto computeStageCovXY =
    [&](const SearchResult & result, const CsmSearchStage & stage) -> Eigen::Matrix2d {
    using Cell = std::pair<int, int>;
    std::map<Cell, std::pair<double, double>> cell_positions;
    std::map<Cell, double> cell_responses;

    const double threshold = result.best_score - 0.1;
    for (const auto & response : result.responses) {
      if (response.score < threshold) {
        continue;
      }

      const Cell cell{
        static_cast<int>(std::lround((response.pose.x - result.center.x) / stage.translation_step)),
        static_cast<int>(
          std::lround((response.pose.y - result.center.y) / stage.translation_step))};

      const auto existing = cell_responses.find(cell);
      if (existing == cell_responses.end() || response.score > existing->second) {
        cell_responses[cell] = response.score;
        cell_positions[cell] = {response.pose.x, response.pose.y};
      }
    }

    double norm = 0.0;
    double var_xx = 0.0;
    double var_yy = 0.0;
    double var_xy = 0.0;
    constexpr double sharpness = 20.0;

    for (const auto & [cell, score] : cell_responses) {
      const double dx = cell_positions[cell].first - result.best_pose.x;
      const double dy = cell_positions[cell].second - result.best_pose.y;

      // Basically, the score represents a fitness measure, rather than a joint probability of how
      // the points fit the likelihood field. If it were a true joint probability distribution, we
      // could take the score as the weight directly, but it is an arithmetic mean of individual
      // point likelihoods, so we can't directly incorporate it into a variance formulation.
      // An approximation can be achieved using a Boltzmann-like weighting scheme. (score is the
      // energy function, sharpness the inverse temperature)
      const double weight = std::exp(sharpness * (score - result.best_score));

      norm += weight;
      var_xx += dx * dx * weight;
      var_yy += dy * dy * weight;
      var_xy += dx * dy * weight;
    }

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    if (norm > 1e-9 && std::isfinite(norm)) {
      const double multiplier = 1.0 / std::max(result.best_score, 1e-9);
      cov(0, 0) =
        std::max(var_xx / norm * multiplier, 0.1 * stage.translation_step * stage.translation_step);
      cov(1, 1) =
        std::max(var_yy / norm * multiplier, 0.1 * stage.translation_step * stage.translation_step);
      cov(0, 1) = cov(1, 0) = var_xy / norm * multiplier;
    } else {
      cov(0, 0) = cov(1, 1) = params_->unobservable_variance;
    }
    return cov;
  };

  // Compute covariances for all stages independently
  std::vector<Eigen::Matrix2d> stage_covs;
  stage_covs.reserve(stages.size());
  for (size_t i = 0; i < stages.size(); ++i) {
    if (
      results[i].responses.empty() || !std::isfinite(results[i].best_score) ||
      results[i].best_score <= 1e-9) {
      stage_covs.push_back(Eigen::Matrix2d::Identity() * params_->unobservable_variance);
    } else {
      stage_covs.push_back(computeStageCovXY(results[i], stages[i]));
    }
  }

  // Start with the precision of the finest stage
  Eigen::Matrix2d final_cov_xy = stage_covs.back();

  // Evaluate the coarse stage for unobservability thresholding
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> macro_solver(stage_covs.front());

  constexpr double threshold_strictness = 0.5;
  const double saturation_threshold =
    threshold_strictness * (stages.front().window_x * stages.front().window_x) / 12.0;

  for (int i = 0; i < 2; ++i) {
    if (macro_solver.eigenvalues()(i) > saturation_threshold) {
      // Inflate only along the direction that exceeded the threshold
      Eigen::Vector2d dir = macro_solver.eigenvectors().col(i);
      final_cov_xy += dir * dir.transpose() * params_->unobservable_variance;
    }
  }

  covariance.topLeftCorner<2, 2>() = final_cov_xy;

  // Calculate Yaw Covariance (Retained from the fine stage)
  const auto & fine_result = results.back();
  const auto & fine_stage = stages.back();
  const double fine_threshold = fine_result.best_score - 0.1;
  double angular_norm = 0.0;
  double angular_variance = 0.0;
  const double cell_tolerance = 0.5 * fine_stage.translation_step + 1e-9;

  for (const auto & response : fine_result.responses) {
    if (
      response.score < fine_threshold ||
      std::abs(response.pose.x - fine_result.best_pose.x) > cell_tolerance ||
      std::abs(response.pose.y - fine_result.best_pose.y) > cell_tolerance) {
      continue;
    }
    const double angle_delta = utils::normalizeAngle(response.pose.yaw - fine_result.best_pose.yaw);
    angular_norm += response.score;
    angular_variance += angle_delta * angle_delta * response.score;
  }

  if (angular_norm > 1e-9 && std::isfinite(angular_norm)) {
    covariance(2, 2) =
      std::max(angular_variance / angular_norm, fine_stage.angular_step * fine_stage.angular_step);
  } else {
    covariance(2, 2) = 1000.0 * fine_stage.angular_step * fine_stage.angular_step;
  }

  covariance = 0.5 * (covariance + covariance.transpose());

  return covariance;
}

CsmResult::DebugImage CsmResult::toDebugImage(const LikelihoodField & field)
{
  CsmResult::DebugImage out;
  out.origin_x = field.origin_x;
  out.origin_y = field.origin_y;
  out.resolution = field.resolution;
  out.width = field.width;
  out.height = field.height;

  if (field.width <= 0 || field.height <= 0 || field.data.empty()) {
    return out;
  }

  const auto min_max = std::minmax_element(field.data.begin(), field.data.end());
  const double min_val = *min_max.first;
  const double max_val = *min_max.second;
  const double span = max_val - min_val;

  out.pixels.resize(field.data.size(), 0);

  if (span <= std::numeric_limits<double>::epsilon()) {
    return out;
  }

  for (int y = 0; y < field.height; ++y) {
    for (int x = 0; x < field.width; ++x) {
      const int src_idx = y * field.width + x;

      const int dst_idx = (field.height - 1 - y) * field.width + x;

      const double normalized = (field.data[src_idx] - min_val) / span;
      const double scaled = std::clamp(normalized, 0.0, 1.0) * 255.0;
      out.pixels[dst_idx] = static_cast<std::uint8_t>(std::lround(scaled));
    }
  }

  return out;
}

}  // namespace glidar_slam
