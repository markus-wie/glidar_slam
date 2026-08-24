#include "glidar_slam/core/loop_closure.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"

namespace glidar_slam::core {

LoopClosureDetector::LoopClosureDetector(
  const std::shared_ptr<Parameters> & parameters, const std::shared_ptr<MapDatabase> & map_database,
  std::unique_ptr<ScanMatcher> scan_matcher)
: parameters_(parameters), map_database_(map_database), scan_matcher_(std::move(scan_matcher))
{
  if (!map_database_) {
    throw std::invalid_argument("Map database must not be null");
  }
  if (!parameters_) {
    throw std::invalid_argument("Parameters must not be null");
  }
}

LoopClosureDetector::~LoopClosureDetector()
{
  stop();
}

std::vector<LoopClosureProposal> LoopClosureDetector::findClosures(const KeyFrame & query)
{
  std::vector<LoopClosureProposal> proposals;

  double search_radius = parameters_->loop_maximum_distance;  // Fallback

  if (query.covariance) {
    // Extract X and Y variance/covariance (GTSAM Pose3 translation indices are 3, 4)
    const double var_x = (*query.covariance)(3, 3);
    const double var_y = (*query.covariance)(4, 4);
    const double cov_xy = (*query.covariance)(3, 4);

    // Compute max eigenvalue of the 2x2 XY covariance matrix
    const double trace = var_x + var_y;
    const double det = var_x * var_y - cov_xy * cov_xy;
    const double max_eigenvalue = (trace + std::sqrt(trace * trace - 4.0 * det)) / 2.0;

    // Radius = Mahalanobis threshold * std::sqrt(max_eigenvalue).
    search_radius = parameters_->loop_mahalanobis_threshold * std::sqrt(max_eigenvalue);
  }

  if (parameters_->loop_debug_enable) {
    SAM_INFO("Loop search radius: query={}, search_radius={}", query.key, search_radius);
  }

  // TODO: Consider using marginal covariance of the latest pose to determine the search radius for
  // nearby keyframes instead of a fixed distance threshold.
  std::vector<std::shared_ptr<const KeyFrame>> nearby_keyframes =
    map_database_->getNearbyKeyFrames(query.pose, search_radius);

  std::vector<std::shared_ptr<const KeyFrame>> candidates;
  for (const std::shared_ptr<const KeyFrame> & candidate : nearby_keyframes) {
    if (!isCandidate(query, *candidate)) {
      continue;
    }
    candidates.push_back(candidate);
  }

  std::vector<double> resolutions;
  resolutions = scan_matcher_->fieldResolutions();

  for (const std::shared_ptr<const KeyFrame> & candidate : candidates) {
    const std::vector<Point2D> & reference_points = candidate->scan->points2D();
    const std::vector<Point2D> & current_points = query.scan->points2D();

    const Pose2D pose_estimate = Utils::toPose2D(candidate->pose.inverse().compose(query.pose));

    SubmapGrid submap_grid(resolutions);
    submap_grid.add(reference_points, candidate->key);
    const CsmResult result = scan_matcher_->match(submap_grid, current_points, pose_estimate);

    if (result.score < parameters_->loop_minimum_score) {
      if (parameters_->loop_debug_enable) {
        SAM_INFO(
          "Loop candidate rejected: query={}, candidate={}, score={}, minimum_score={}", query.key,
          candidate->key, result.score, parameters_->loop_minimum_score);
      }
      continue;
    }

    const double consistency_error = 0.0;

    LoopClosureProposal proposal;
    proposal.from_key = candidate->key;
    proposal.to_key = query.key;
    proposal.relative_pose = Utils::toPose3(result.optimized_pose);
    proposal.covariance = gtsam::Matrix66::Zero();
    proposal.covariance(0, 0) = 1e6;
    proposal.covariance(1, 1) = 1e6;
    proposal.covariance(2, 2) = result.covariance(2, 2);
    proposal.covariance(3, 3) = result.covariance(0, 0);
    proposal.covariance(4, 4) = result.covariance(1, 1);
    proposal.covariance(5, 5) = 1e6;
    proposal.covariance(3, 4) = result.covariance(0, 1);
    proposal.covariance(4, 3) = result.covariance(1, 0);
    proposal.covariance(2, 3) = result.covariance(2, 0);
    proposal.covariance(3, 2) = result.covariance(0, 2);
    proposal.covariance(2, 4) = result.covariance(2, 1);
    proposal.covariance(4, 2) = result.covariance(1, 2);
    proposal.score = result.score;
    proposal.consistency_error = consistency_error;
    proposals.push_back(std::move(proposal));

    if (parameters_->loop_debug_enable) {
      SAM_INFO(
        "Loop candidate matched: query={}, candidate={}, score={}, pose_estimate=({}, {}, {}), "
        "relative_pose=({}, {}, {})",
        query.key, candidate->key, result.score, pose_estimate.x, pose_estimate.y,
        pose_estimate.yaw, result.optimized_pose.x, result.optimized_pose.y,
        result.optimized_pose.yaw);
    }
  }

  if (parameters_->loop_debug_enable) {
    SAM_INFO(
      "Loop query processed: query={}, nearby_keyframes={}, candidates={}, proposals={}", query.key,
      nearby_keyframes.size(), candidates.size(), proposals.size());
  }

  if (proposals.empty()) {
    return proposals;
  }

  LoopClosureProposal best_proposal;
  double best_score = -std::numeric_limits<double>::infinity();
  for (const LoopClosureProposal & proposal : proposals) {
    if (proposal.score > best_score) {
      best_score = proposal.score;
      best_proposal = proposal;
    }
  }
  proposals.clear();
  proposals.push_back(best_proposal);

  return proposals;
}

bool LoopClosureDetector::isCandidate(const KeyFrame & query, const KeyFrame & candidate) const
{
  if (
    query.key <= candidate.key ||
    query.key - candidate.key < parameters_->loop_minimum_key_separation) {
    return false;
  }

  // The mahalanobis distance is meant to act as a distance measure between a probability
  // distribution and a point. Here, however, we want to check the "distance" between two
  // probability distributions. To do that with the help of the mahalanobis distance, we first
  // compute the distribution of the difference between the two poses. In the best case, we want the
  // candidate to overlay perfectly with the query, which would result in a difference of zero.
  // Since that is never exactly the case, we need to define a measure, in this case between the
  // constructed distribution and the zero vector. On that, we can take the mahalanobis distance.

  // Compute relative pose and Jacobians
  gtsam::Matrix H_candidate;
  gtsam::Matrix H_query;
  const gtsam::Pose3 relative_pose = candidate.pose.between(query.pose, H_candidate, H_query);

  // Hard yaw check is good for early rejection
  if (std::abs(relative_pose.rotation().yaw()) > parameters_->loop_maximum_yaw_difference) {
    return false;
  }

  // Compute relative covariance (summing marginals projected via Jacobians)
  gtsam::Matrix66 cov_candidate = candidate.covariance.value_or(gtsam::Matrix66::Identity() * 1e-4);
  gtsam::Matrix66 cov_query = query.covariance.value_or(gtsam::Matrix66::Identity() * 1e-4);

  gtsam::Matrix66 relative_cov_6d = H_candidate * cov_candidate * H_candidate.transpose() +
                                    H_query * cov_query * H_query.transpose();

  // Extract the 3x3 SE(2) block. GTSAM tangent space is [rx, ry, rz, tx, ty, tz]
  // We want indices 2 (yaw), 3 (x), 4 (y)
  gtsam::Matrix33 cov_se2;
  std::vector<int> idx = {2, 3, 4};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cov_se2(i, j) = relative_cov_6d(idx[i], idx[j]);
    }
  }

  // Add minimum variance floor to ensure the matrix is invertible
  cov_se2(0, 0) += 1e-4;  // Yaw minimum variance
  cov_se2(1, 1) += 1e-4;
  cov_se2(2, 2) += 1e-4;

  // Extract the 3-DOF error vector via Logmap to respect manifold geometry
  const gtsam::Vector6 log_error = gtsam::Pose3::Logmap(relative_pose);
  gtsam::Vector3 error_se2;
  error_se2 << log_error(2), log_error(3), log_error(4);

  // Compute exact Mahalanobis distance
  const double mahalanobis_squared = error_se2.transpose() * cov_se2.inverse() * error_se2;
  const double mahalanobis_limit =
    parameters_->loop_mahalanobis_threshold * parameters_->loop_mahalanobis_threshold;

  if (parameters_->loop_debug_enable) {
    SAM_INFO(
      "Loop proximity gate: query={}, candidate={}, relative_pose=({}, {}, {}), error=({}, {}, "
      "{}), mahalanobis_squared={}, limit={}, accepted={}",
      query.key, candidate.key, relative_pose.x(), relative_pose.y(),
      relative_pose.rotation().yaw(), error_se2(0), error_se2(1), error_se2(2), mahalanobis_squared,
      mahalanobis_limit, mahalanobis_squared <= mahalanobis_limit);
  }

  return mahalanobis_squared <= mahalanobis_limit;
}

void LoopClosureDetector::start()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    return;
  }

  stopping_ = false;
  running_ = true;
  worker_thread_ = std::thread(&LoopClosureDetector::run, this);
  SAM_INFO("Loop closure worker started");
}

void LoopClosureDetector::stop()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    stopping_ = true;
  }
  condition_variable_.notify_all();

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }

  running_ = false;
  SAM_INFO("Loop closure worker stopped");
}

bool LoopClosureDetector::submit(KeyFrame snapshot)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_ || stopping_) {
    return false;
  }

  if (input_queue_.size() >= parameters_->loop_input_queue_capacity) {
    SAM_INFO("Loop input queue full; dropping oldest snapshot");
    input_queue_.pop_front();
  }
  input_queue_.push_back(std::move(snapshot));
  condition_variable_.notify_one();
  return true;
}

std::optional<LoopClosureProposal> LoopClosureDetector::tryPopProposal()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (output_queue_.empty()) {
    return std::nullopt;
  }

  LoopClosureProposal proposal = std::move(output_queue_.front());
  output_queue_.pop_front();
  return proposal;
}

void LoopClosureDetector::run()
{
  while (true) {
    KeyFrame query;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_variable_.wait(lock, [this] {
        return stopping_ || !input_queue_.empty();
      });

      if (stopping_ && input_queue_.empty()) {
        return;
      }

      query = std::move(input_queue_.front());
      input_queue_.pop_front();
    }

    const auto search_start = std::chrono::steady_clock::now();

    std::vector<LoopClosureProposal> proposals = findClosures(query);
    if (parameters_->debug_timings) {
      const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - search_start)
          .count();
      SAM_INFO(
        "Loop closure timing [ms]: query_key={}, search={}, proposals={}", query.key, elapsed_ms,
        proposals.size());
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto & proposal : proposals) {
        if (output_queue_.size() >= parameters_->loop_output_queue_capacity) {
          SAM_INFO("Loop output queue full; dropping oldest proposal");
          output_queue_.pop_front();
        }
        output_queue_.push_back(std::move(proposal));
      }
    }
  }
}

}  // namespace glidar_slam::core
