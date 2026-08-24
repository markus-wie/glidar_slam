#include "glidar_slam/core/correlative_scan_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"

namespace glidar_slam::core {

CorrelativeScanMatcher::CorrelativeScanMatcher(const std::shared_ptr<Parameters> & params)
: params_(params)
{
}

double LikelihoodField::getScore(double x, double y) const
{
  double px = (x - origin_x) / resolution;
  double py = (y - origin_y) / resolution;

  const int x0 = static_cast<int>(std::floor(px));
  const int y0 = static_cast<int>(std::floor(py));

  unsigned int ux = static_cast<unsigned int>(x0);
  unsigned int uy = static_cast<unsigned int>(y0);
  unsigned int w_limit = static_cast<unsigned int>(width - 1);
  unsigned int h_limit = static_cast<unsigned int>(height - 1);

  if ((ux >= w_limit) | (uy >= h_limit)) {
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
  const LaserScan & reference, const LaserScan & current) const
{
  CsmResult result{};
  result.optimized_pose = current.world_pose;
  result.score = 0.0;
  result.covariance = Eigen::Matrix3d::Identity();

  const std::vector<CsmSearchStage> & stages = params_->csm_search_stages;

  if (stages.empty() || reference.points.empty() || current.points.empty()) {
    result.covariance(0, 0) = 500.0;
    result.covariance(1, 1) = 500.0;
    result.covariance(2, 2) = 4.0;
    return result;
  }

  std::vector<LikelihoodField> fields;
  fields.reserve(stages.size());

  for (const auto & stage : stages) {
    fields.push_back(buildField(reference.points, stage.field_resolution));
  }

  const Pose2D initial_guess = current.world_pose;
  const double initial_score = evaluatePose(current.points, fields.front(), initial_guess);
  Pose2D best_pose = initial_guess;

  std::vector<SearchResult> results;
  results.reserve(stages.size());

  if (params_->csm_debug_enable) {
    SAM_INFO(
      "CSM initial guess: x={}, y={}, yaw={}, score={}", initial_guess.x, initial_guess.y,
      initial_guess.yaw, initial_score);
  }

  for (std::size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
    const auto & stage = stages[stage_index];

    SearchResult stage_result = searchSpace(current.points, fields[stage_index], best_pose, stage);

    best_pose = stage_result.best_pose;

    results.push_back(stage_result);

    if (params_->csm_debug_enable) {
      SAM_INFO(
        "CSM stage {}: center=({}, {}, {}), result=({}, {}, {}), score={}, candidates={}, "
        "steps=({}, {}, {}), windows=({}, {}, {})",
        stage_index, best_pose.x, best_pose.y, best_pose.yaw, stage_result.best_pose.x,
        stage_result.best_pose.y, stage_result.best_pose.yaw, stage_result.best_score,
        stage_result.responses.size(), stage.translation_step, stage.translation_step,
        stage.angular_step, stage.window_x, stage.window_y, stage.window_yaw);
    }
  }

  Eigen::Matrix3d cov = computeCovariance(results, stages);

  if (params_->csm_debug_enable) {
    SAM_INFO(
      "CSM final pose: x={}, y={}, yaw={}, score={}, delta_x={}, delta_y={}, delta_yaw={}",
      best_pose.x, best_pose.y, best_pose.yaw, results.back().best_score,
      best_pose.x - initial_guess.x, best_pose.y - initial_guess.y,
      Utils::normalizeAngle(best_pose.yaw - initial_guess.yaw));
  }

  result.optimized_pose = best_pose;
  result.score = results.back().best_score;
  result.covariance = cov;
  result.low_res_debug = CsmResult::toDebugImage(fields.front());
  result.high_res_debug = CsmResult::toDebugImage(fields.back());
  return result;
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
  double max_search_window = 0.0;
  for (const auto & stage : params_->csm_search_stages) {
    max_search_window = std::max({max_search_window, stage.window_x, stage.window_y});
  }
  const double padding = max_smear_distance + max_search_window;
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
  field.data.assign(
    static_cast<std::size_t>(field.width) * static_cast<std::size_t>(field.height), 0.0);

  // Splat points onto grid
  const int rad = std::ceil(max_smear_distance / resolution);
  const double denom = 2.0 * params_->csm_smear_deviation * params_->csm_smear_deviation;

  for (const auto & p : points) {
    const int cx = static_cast<int>((p.x - field.origin_x) / resolution);
    const int cy = static_cast<int>((p.y - field.origin_y) / resolution);

    for (int dy = -rad; dy <= rad; ++dy) {
      for (int dx = -rad; dx <= rad; ++dx) {
        const int nx = cx + dx;
        const int ny = cy + dy;
        if (nx >= 0 && nx < field.width && ny >= 0 && ny < field.height) {
          const double dist_sq = (dx * dx + dy * dy) * resolution * resolution;
          if (dist_sq <= max_smear_distance_sq) {
            const double prob = std::exp(-dist_sq / denom);
            const int idx = ny * field.width + nx;
            field.data[idx] = std::max(field.data[idx], prob);
          }
        }
      }
    }
  }
  return field;
}

CorrelativeScanMatcher::SearchResult CorrelativeScanMatcher::searchSpace(
  const std::vector<Point2D> & points, const LikelihoodField & field, const Pose2D & center,
  const CsmSearchStage & stage) const
{
  SearchResult result;
  result.center = center;
  result.best_pose = center;
  result.best_score = 0.0;

  if (points.empty()) {
    return result;
  }

  const auto sample_count = [](double window, double step) {
    return static_cast<int>(std::ceil((window) / step));
  };
  const int x_count = sample_count(stage.window_x, stage.translation_step);
  const int y_count = sample_count(stage.window_y, stage.translation_step);
  const int yaw_count = sample_count(stage.window_yaw, stage.angular_step);
  const int pose_count = x_count * y_count * yaw_count;

  const double yaw_start = center.yaw - (stage.window_yaw * 0.5);
  const double yaw_end = center.yaw + (stage.window_yaw * 0.5);
  const double x_start = center.x - (stage.window_x * 0.5);
  const double x_end = center.x + (stage.window_x * 0.5);
  const double y_start = center.y - (stage.window_y * 0.5);
  const double y_end = center.y + (stage.window_y * 0.5);

  std::vector<Point2D> rotated_points(points.size());

  result.responses.reserve(static_cast<std::size_t>(pose_count));
  result.best_score = -std::numeric_limits<double>::infinity();

  for (int yaw_index = 0; yaw_index < yaw_count; ++yaw_index) {
    const double yaw = std::min(yaw_start + yaw_index * stage.angular_step, yaw_end);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);

    for (std::size_t i = 0; i < points.size(); ++i) {
      rotated_points[i].x = c * points[i].x - s * points[i].y;
      rotated_points[i].y = s * points[i].x + c * points[i].y;
    }

    for (int x_index = 0; x_index < x_count; ++x_index) {
      const double x = std::min(x_start + x_index * stage.translation_step, x_end);
      for (int y_index = 0; y_index < y_count; ++y_index) {
        const double y = std::min(y_start + y_index * stage.translation_step, y_end);
        double score = 0.0;
        int valid_points = 0;
        for (std::size_t i = 0; i < points.size(); ++i) {
          double eval_score = field.getScore(x + rotated_points[i].x, y + rotated_points[i].y);
          if (eval_score < 0.0) {
            continue;
          }
          score += eval_score;
          ++valid_points;
        }

        score /= static_cast<double>(valid_points);

        if (params_->csm_use_penalty && score > 0.0) {
          const double dx = x - center.x;
          const double dy = y - center.y;
          const double dyaw = Utils::normalizeAngle(yaw - center.yaw);
          const double distance_variance =
            params_->csm_distance_variance_penalty * params_->csm_distance_variance_penalty;
          const double angle_variance =
            params_->csm_angle_variance_penalty * params_->csm_angle_variance_penalty;
          const double distance_penalty = std::clamp(
            1.0 - 0.2 * (dx * dx + dy * dy) / std::max(distance_variance, 1e-12),
            std::clamp(params_->csm_minimum_distance_penalty, 0.0, 1.0), 1.0);
          const double angle_penalty = std::clamp(
            1.0 - 0.2 * (dyaw * dyaw) / std::max(angle_variance, 1e-12),
            std::clamp(params_->csm_minimum_angle_penalty, 0.0, 1.0), 1.0);
          score *= distance_penalty * angle_penalty;
        }

        result.responses.push_back({{x, y, yaw}, score});
        result.best_score = std::max(score, result.best_score);
      }
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

  return result;
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

  double normalized_score = score / static_cast<double>(valid_points);

  return normalized_score;
}

Eigen::Matrix3d CorrelativeScanMatcher::computeCovariance(
  const std::vector<SearchResult> & results, const std::vector<CsmSearchStage> & stages) const
{
  constexpr double max_variance = 500.0;

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

  if (results.empty() || stages.empty() || results.size() != stages.size()) {
    covariance(0, 0) = max_variance;
    covariance(1, 1) = max_variance;
    covariance(2, 2) = 1000.0;
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
  for (const auto & [cell, response] : cell_responses) {
    const auto position = cell_positions[cell];
    const double dx = position.first - coarse_result.best_pose.x;
    const double dy = position.second - coarse_result.best_pose.y;
    norm += response;
    variance_xx += dx * dx * response;
    variance_xy += dx * dy * response;
    variance_yy += dy * dy * response;
  }

  if (norm > 1e-9 && std::isfinite(norm)) {
    const double multiplier = 1.0 / std::max(coarse_result.best_score, 1e-9);
    covariance(0, 0) = std::max(
      variance_xx / norm * multiplier,
      0.1 * coarse_stage.translation_step * coarse_stage.translation_step);
    covariance(0, 1) = covariance(1, 0) = variance_xy / norm * multiplier;
    covariance(1, 1) = std::max(
      variance_yy / norm * multiplier,
      0.1 * coarse_stage.translation_step * coarse_stage.translation_step);
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
