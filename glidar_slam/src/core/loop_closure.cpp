#include "glidar_slam/core/loop_closure.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"

namespace glidar_slam::core {

LoopClosureDetector::LoopClosureDetector(
  const std::shared_ptr<Parameters> & parameters, const std::shared_ptr<MapDatabase> & map_database)
: parameters_(parameters), map_database_(map_database)
{
  if (!map_database_) {
    throw std::invalid_argument("Map database must not be null");
  }
  if (!parameters_) {
    throw std::invalid_argument("Parameters must not be null");
  }
  scan_matcher_ = std::make_unique<CorrelativeScanMatcher>(parameters_);
}

LoopClosureDetector::~LoopClosureDetector()
{
  stop();
}

std::vector<LoopClosureProposal> LoopClosureDetector::findClosures(const KeyFrame & query)
{
  std::vector<LoopClosureProposal> proposals;

  // TODO: Consider using marginal covariance of the latest pose to determine the search radius for
  // nearby keyframes instead of a fixed distance threshold.
  std::vector<std::shared_ptr<const KeyFrame>> nearby_keyframes =
    map_database_->getNearbyKeyFrames(query.pose, parameters_->loop_maximum_distance);

  std::vector<std::shared_ptr<const KeyFrame>> candidates;
  for (const std::shared_ptr<const KeyFrame> & candidate : nearby_keyframes) {
    if (!isCandidate(query, *candidate)) {
      continue;
    }
    candidates.push_back(candidate);
  }

  for (const std::shared_ptr<const KeyFrame> & candidate : candidates) {
    const std::vector<Point2D> & reference_points = candidate->scan->points2D();
    const std::vector<Point2D> & current_points = query.scan->points2D();

    const Pose2D pose_estimate = Utils::toPose2D(candidate->pose.inverse().compose(query.pose));
    const CsmResult result = scan_matcher_->match(reference_points, current_points, pose_estimate);

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

  // TODO: First checking a hard distance threshold before computing the Mahalanobis distance is not
  // really a good idea. This works now since the problem is small, but should be revisited later.
  // Same for how nearby keyframes are retrieved in the first place.

  const gtsam::Pose3 relative_pose = candidate.pose.inverse().compose(query.pose);
  const double dx = relative_pose.x();
  const double dy = relative_pose.y();
  const double distance_squared = dx * dx + dy * dy;
  if (distance_squared > parameters_->loop_maximum_distance * parameters_->loop_maximum_distance) {
    return false;
  }

  const double yaw_difference = std::abs(relative_pose.rotation().yaw());
  if (yaw_difference > parameters_->loop_maximum_yaw_difference) {
    return false;
  }

  double variance_x = parameters_->loop_minimum_xy_variance;
  double variance_y = parameters_->loop_minimum_xy_variance;
  if (candidate.covariance) {
    variance_x += (*candidate.covariance)(3, 3);
    variance_y += (*candidate.covariance)(4, 4);
  }
  if (query.covariance) {
    variance_x += (*query.covariance)(3, 3);
    variance_y += (*query.covariance)(4, 4);
  }

  const double mahalanobis_squared = dx * dx / variance_x + dy * dy / variance_y;
  const double mahalanobis_limit =
    parameters_->loop_mahalanobis_threshold * parameters_->loop_mahalanobis_threshold;
  const bool is_candidate = mahalanobis_squared <= mahalanobis_limit;
  if (parameters_->loop_debug_enable) {
    SAM_INFO(
      "Loop proximity gate: query={}, candidate={}, distance=({}, {}), distance_norm={}, "
      "yaw={}, variance=({}, {}), mahalanobis_squared={}, limit={}, accepted={}",
      query.key, candidate.key, dx, dy, std::hypot(dx, dy), yaw_difference, variance_x, variance_y,
      mahalanobis_squared, mahalanobis_limit, is_candidate);
  }
  return is_candidate;
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

    std::vector<LoopClosureProposal> proposals = findClosures(query);

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
