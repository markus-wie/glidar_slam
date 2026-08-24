#include "glidar_slam/core/correlative_scan_matcher.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"
#include "tbb/parallel_for.h"

namespace glidar_slam::core {

CorrelativeScanMatcher::CorrelativeScanMatcher(const std::shared_ptr<Parameters> & params)
: params_(params)
{
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
  const std::vector<Point2D> & reference_points, const std::vector<Point2D> & current_points,
  const Pose2D & pose_estimate) const
{
  const auto timing_start = std::chrono::steady_clock::now();
  CsmResult result{};
  result.optimized_pose = pose_estimate;
  result.score = 0.0;
  result.covariance = Eigen::Matrix3d::Identity();

  const std::vector<CsmSearchStage> & stages = params_->csm_search_stages;

  if (stages.empty() || reference_points.empty() || current_points.empty()) {
    result.covariance(0, 0) = 500.0;
    result.covariance(1, 1) = 500.0;
    result.covariance(2, 2) = 4.0;
    return result;
  }

  const std::shared_ptr<const std::vector<LikelihoodField>> fields =
    getLikelihoodFields(reference_points, stages);

  if (params_->debug_timings) {
    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - timing_start)
        .count();
    SAM_INFO("CSM timing: field construction={} ms", elapsed_ms);
  }

  const auto initial_score_start = std::chrono::steady_clock::now();
  const double initial_score = evaluatePose(current_points, fields->front(), pose_estimate);
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
      searchSpace(current_points, fields->at(stage_index), best_pose, stage);
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
        "steps=({}, {}, {}), windows=({}, {}, {})",
        stage_index, stage_result.center.x, stage_result.center.y, stage_result.center.yaw,
        stage_result.best_pose.x, stage_result.best_pose.y, stage_result.best_pose.yaw,
        stage_result.best_score, stage_result.candidate_count, stage.translation_step,
        stage.translation_step, stage.angular_step, stage.window_x, stage.window_y,
        stage.window_yaw);
    }
  }

  const auto covariance_start = std::chrono::steady_clock::now();
  Eigen::Matrix3d cov = computeCovariance(results, stages);
  if (params_->debug_timings) {
    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - covariance_start)
        .count();
    SAM_INFO("CSM timing: covariance={} ms", elapsed_ms);
  }

  if (params_->csm_debug_enable) {
    SAM_INFO(
      "CSM final pose: x={}, y={}, yaw={}, score={}, delta_x={}, delta_y={}, delta_yaw={}",
      best_pose.x, best_pose.y, best_pose.yaw, results.back().best_score,
      best_pose.x - pose_estimate.x, best_pose.y - pose_estimate.y,
      Utils::normalizeAngle(best_pose.yaw - pose_estimate.yaw));
  }

  result.optimized_pose = best_pose;
  result.score = results.back().best_score;
  result.covariance = cov;
  const auto debug_image_start = std::chrono::steady_clock::now();
  if (params_->csm_debug_enable) {
    result.low_res_debug = CsmResult::toDebugImage(fields->front());
    result.high_res_debug = CsmResult::toDebugImage(fields->back());
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

std::shared_ptr<const std::vector<LikelihoodField>> CorrelativeScanMatcher::getLikelihoodFields(
  const std::vector<Point2D> & reference_points, const std::vector<CsmSearchStage> & stages) const
{
  const bool cache_matches =
    cached_fields_ && referencePointsEqual(cached_reference_points_, reference_points);

  if (cache_matches) {
    return cached_fields_;
  }

  auto fields = std::make_shared<std::vector<LikelihoodField>>();
  fields->reserve(stages.size());
  for (const auto & stage : stages) {
    fields->push_back(buildField(reference_points, stage.field_resolution));
  }

  cached_reference_points_ = reference_points;

  cached_fields_ = std::move(fields);
  return cached_fields_;
}

bool CorrelativeScanMatcher::referencePointsEqual(
  const std::vector<Point2D> & lhs, const std::vector<Point2D> & rhs)
{
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index].x != rhs[index].x || lhs[index].y != rhs[index].y) {
      return false;
    }
  }
  return true;
}

double CorrelativeScanMatcher::maximumSearchWindow(const std::vector<CsmSearchStage> & stages)
{
  double maximum_window = 0.0;
  for (const auto & stage : stages) {
    maximum_window = std::max({maximum_window, stage.window_x * 0.5, stage.window_y * 0.5});
  }
  return maximum_window;
}

LikelihoodField CorrelativeScanMatcher::buildField(
  const std::vector<Point2D> & points, double resolution) const
{
  LikelihoodField field;
  field.resolution = resolution;

  if (points.empty()) {
    return field;
  }

  // define the max smear distance by a multiple of the deviation
  const double max_smear_distance = params_->csm_smear_deviation * 2.0;
  const double max_search_window = maximumSearchWindow(params_->csm_search_stages);
  const double padding = 0;
  const double max_smear_distance_sq = max_smear_distance * max_smear_distance;

  double min_x = points[0].x;
  double max_x = points[0].x;
  double min_y = points[0].y;
  double max_y = points[0].y;
  for (const auto & p : points) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }

  // Pad grid by max search distance
  field.origin_x = min_x - padding;
  field.origin_y = min_y - padding;
  field.width =
    std::max(2, static_cast<int>(std::ceil((max_x - min_x + 2 * padding) / resolution)) + 1);
  field.height =
    std::max(2, static_cast<int>(std::ceil((max_y - min_y + 2 * padding) / resolution)) + 1);
  field.max_x_index = field.width - 1;
  field.max_y_index = field.height - 1;
  field.data.assign(
    static_cast<std::size_t>(field.width) * static_cast<std::size_t>(field.height), 0.0F);

  // Splat points onto grid
  const int rad = std::ceil(max_smear_distance / resolution);
  const double denom = 2.0 * params_->csm_smear_deviation * params_->csm_smear_deviation;
  const double resolution_sq = resolution * resolution;
  const int kernel_width = 2 * rad + 1;

  struct KernelOffset
  {
    int dx;
    int dy;
    float probability;
  };

  std::vector<KernelOffset> kernel_offsets;
  kernel_offsets.reserve(static_cast<std::size_t>(kernel_width) * kernel_width);
  for (int dy = -rad; dy <= rad; ++dy) {
    for (int dx = -rad; dx <= rad; ++dx) {
      const double dist_sq = (dx * dx + dy * dy) * resolution_sq;
      if (dist_sq <= max_smear_distance_sq) {
        kernel_offsets.push_back({dx, dy, static_cast<float>(std::exp(-dist_sq / denom))});
      }
    }
  }

  for (const auto & p : points) {
    const int cx = static_cast<int>((p.x - field.origin_x) / field.resolution);
    const int cy = static_cast<int>((p.y - field.origin_y) / field.resolution);

    for (const auto & offset : kernel_offsets) {
      const int nx = cx + offset.dx;
      const int ny = cy + offset.dy;
      if (nx >= 0 && nx < field.width && ny >= 0 && ny < field.height) {
        const int idx = ny * field.width + nx;
        field.data[idx] = std::max<float>(field.data[idx], offset.probability);
      }
    }
  }
  return field;
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
      evaluateYawSlice(points, field, center, grid, yaw_index, result);
    });
  } else {
    for (std::size_t yaw_index = 0; yaw_index < grid.yaw_positions.size(); ++yaw_index) {
      evaluateYawSlice(points, field, center, grid, yaw_index, result);
    }
  }

  selectBestPose(result);
  return result;
}

std::vector<double> CorrelativeScanMatcher::makeSearchPositions(
  double center, double window, double step)
{
  const int sample_count = std::max(1, static_cast<int>(std::ceil(window / step)));
  const double start = center - window * 0.5;
  std::vector<double> positions;
  positions.reserve(static_cast<std::size_t>(sample_count));
  for (int index = 0; index < sample_count; ++index) {
    positions.push_back(start + index * step);
  }
  return positions;
}

void CorrelativeScanMatcher::evaluateYawSlice(
  const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & center,
  const SearchGrid & grid, std::size_t yaw_index, SearchResult & result) const
{
  const double yaw = grid.yaw_positions[yaw_index];
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double dyaw = Utils::normalizeAngle(yaw - center.yaw);
  const double angle_variance =
    params_->csm_angle_penalty_std_dev * params_->csm_angle_penalty_std_dev;
  const double angle_penalty =
    std::clamp(1.0 - 0.2 * (dyaw * dyaw) / std::max(angle_variance, 1e-12), 0.0, 1.0);

  std::vector<Point2D> rotated_points(points.size());
  for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
    rotated_points[point_index].x = c * points[point_index].x - s * points[point_index].y;
    rotated_points[point_index].y = s * points[point_index].x + c * points[point_index].y;
  }

  const std::size_t yaw_offset = yaw_index * grid.candidates_per_yaw;
  for (std::size_t x_index = 0; x_index < grid.x_positions.size(); ++x_index) {
    for (std::size_t y_index = 0; y_index < grid.y_positions.size(); ++y_index) {
      const std::size_t response_index = yaw_offset + x_index * grid.y_positions.size() + y_index;
      const double score = scoreCandidate(
        rotated_points, field, grid.x_positions[x_index], grid.y_positions[y_index], center,
        angle_penalty);
      result.responses[response_index] = {
        {grid.x_positions[x_index], grid.y_positions[y_index], yaw}, score};
    }
  }
}

double CorrelativeScanMatcher::scoreCandidate(
  const std::vector<Point2D> & points, const LikelihoodField & field, double x, double y,
  const Pose2D & center, double angle_penalty) const
{
  double score = 0.0;

  for (const auto & point : points) {
    const double eval_score = field.getScore(x + point.x, y + point.y);
    if (eval_score < 0.0) {
      continue;
    }
    score += eval_score;
  }

  score /= static_cast<double>(points.size());

  if (params_->csm_use_penalty && score > 0.0) {
    const double dx = x - center.x;
    const double dy = y - center.y;
    // Basically a first order approximation of the exp function, but multiplied by 0.2 to make it
    // less aggressive. Acts as a gaussian penalty on large deviations from the initial guess
    const double distance_variance =
      params_->csm_distance_penalty_std_dev * params_->csm_distance_penalty_std_dev;
    const double distance_penalty =
      std::clamp(1.0 - 0.2 * (dx * dx + dy * dy) / std::max(distance_variance, 1e-12), 0.0, 1.0);
    score *= distance_penalty * angle_penalty;
  }
  return score;
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
    if (Utils::DoubleEqual(response.score, result.best_score)) {
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

double CorrelativeScanMatcher::evaluatePose(
  const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & pose) const
{
  double score = 0.0;
  double c = std::cos(pose.yaw);
  double s = std::sin(pose.yaw);

  int valid_points = 0;
  for (const auto & p : points) {
    double eval_score = field.getScore(pose.x + c * p.x - s * p.y, pose.y + s * p.x + c * p.y);
    if (eval_score < 0.0) {
      continue;
    }
    score += eval_score;
    ++valid_points;
  }

  if (valid_points == 0) {
    return -1.0;
  }

  const double normalized_score = score / static_cast<double>(valid_points);

  return normalized_score;
}

Eigen::Matrix3d CorrelativeScanMatcher::computeCovariance(
  const std::vector<SearchResult> & results, const std::vector<CsmSearchStage> & stages)
{
  constexpr double max_variance = 25.0;

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

  if (results.empty() || stages.empty() || results.size() != stages.size()) {
    covariance(0, 0) = max_variance;
    covariance(1, 1) = max_variance;
    covariance(2, 2) = max_variance;
    return covariance;
  }

  // use the coarsest stage for x/y translation variance
  const auto & coarse_result = results.front();
  const auto & coarse_stage = stages.front();

  // use the finest stage for yaw rotational variance
  const auto & fine_result = results.back();
  const auto & fine_stage = stages.back();

  covariance(2, 2) = 4.0 * fine_stage.angular_step * fine_stage.angular_step;

  if (
    coarse_result.responses.empty() || !std::isfinite(coarse_result.best_score) ||
    coarse_result.best_score <= 1e-9) {
    covariance(0, 0) = max_variance;
    covariance(1, 1) = max_variance;
    covariance(2, 2) = 1000.0 * fine_stage.angular_step * fine_stage.angular_step;
    return covariance;
  }

  using Cell = std::pair<int, int>;
  std::map<Cell, std::pair<double, double>> cell_positions;
  std::map<Cell, double> cell_responses;
  const double response_threshold = coarse_result.best_score - 0.1;
  for (const auto & response : coarse_result.responses) {
    if (response.score < response_threshold) {
      continue;
    }
    const Cell cell{
      static_cast<int>(
        std::lround((response.pose.x - coarse_result.center.x) / coarse_stage.translation_step)),
      static_cast<int>(
        std::lround((response.pose.y - coarse_result.center.y) / coarse_stage.translation_step))};
    const auto existing_response = cell_responses.find(cell);
    if (existing_response == cell_responses.end() || response.score > existing_response->second) {
      cell_responses[cell] = response.score;
      cell_positions[cell] = {response.pose.x, response.pose.y};
    }
  }

  double norm = 0.0;
  double variance_xx = 0.0;
  double variance_xy = 0.0;
  double variance_yy = 0.0;

  // Scaling factor to convert [0, 1] scores into sharp probability weights
  constexpr double sharpness = 50.0;

  for (const auto & [cell, response_score] : cell_responses) {
    const auto position = cell_positions[cell];
    const double dx = position.first - coarse_result.best_pose.x;
    const double dy = position.second - coarse_result.best_pose.y;

    // Softmax-style weighting
    const double weight = std::exp(sharpness * (response_score - coarse_result.best_score));

    norm += weight;
    variance_xx += dx * dx * weight;
    variance_xy += dx * dy * weight;
    variance_yy += dy * dy * weight;
  }

  if (norm > 1e-9 && std::isfinite(norm)) {
    const double multiplier = 1.0 / std::max(coarse_result.best_score, 1e-9);

    Eigen::Matrix2d cov_xy;
    // Calculate raw sample variance in meters squared (removed the old multiplier hack)
    cov_xy(0, 0) = std::max(
      variance_xx / norm * multiplier,
      0.1 * coarse_stage.translation_step * coarse_stage.translation_step);
    cov_xy(0, 1) = cov_xy(1, 0) = variance_xy / norm * multiplier;
    cov_xy(1, 1) = std::max(
      variance_yy / norm * multiplier,
      0.1 * coarse_stage.translation_step * coarse_stage.translation_step);

    // Eigendecomposition to identify flat ridges (unobservable directions)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov_xy);
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors();

    // Theoretical maximum variance for a uniform distribution on window [-W, W] is W^2 / 3
    // Using window_x (or average search window) as bounding box reference
    const double max_search_variance = (coarse_stage.window_x * coarse_stage.window_x) / 12.0;
    const double saturation_threshold = 0.8 * max_search_variance;

    for (int i = 0; i < 2; ++i) {
      if (eigenvalues(i) > saturation_threshold) {
        eigenvalues(i) = max_variance;  // Inflate only the unconstrained principal axis
      }
    }

    // Reconstruct the covariance matrix
    cov_xy = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

    covariance(0, 0) = cov_xy(0, 0);
    covariance(0, 1) = covariance(1, 0) = cov_xy(0, 1);
    covariance(1, 1) = cov_xy(1, 1);
  } else {
    covariance(0, 0) = max_variance;
    covariance(1, 1) = max_variance;
  }

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
    const double angle_delta = Utils::normalizeAngle(response.pose.yaw - fine_result.best_pose.yaw);
    angular_norm += response.score;
    angular_variance += angle_delta * angle_delta * response.score;
  }

  if (angular_norm > 1e-9 && std::isfinite(angular_norm)) {
    covariance(2, 2) = angular_variance / angular_norm;
    covariance(2, 2) =
      std::max(covariance(2, 2), fine_stage.angular_step * fine_stage.angular_step);
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

  for (size_t i = 0; i < field.data.size(); ++i) {
    const double normalized = (field.data[i] - min_val) / span;
    const double scaled = std::clamp(normalized, 0.0, 1.0) * 255.0;
    out.pixels[i] = static_cast<std::uint8_t>(std::lround(scaled));
  }

  return out;
}

}  // namespace glidar_slam::core
