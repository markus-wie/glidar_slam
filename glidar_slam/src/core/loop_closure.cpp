#include "glidar_slam/core/loop_closure.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "Eigen/Eigenvalues"
#include "glidar_slam/core/scan_matcher/correlative_scan_matcher.hpp"
#include "glidar_slam/core/utils.hpp"
#include "glidar_slam/logger/logger.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

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
  double search_radius = 1.0;  // Fallback

  if (query.covariance) {
    // Extract X and Y variance/covariance (GTSAM Pose3 translation indices are 3, 4)
    const double var_x = (*query.covariance)(3, 3);
    const double var_y = (*query.covariance)(4, 4);
    const double cov_xy = (*query.covariance)(3, 4);

    // Compute max eigenvalue of the 2x2 XY covariance matrix
    const double trace = var_x + var_y;
    const double det = var_x * var_y - cov_xy * cov_xy;
    const double max_eigenvalue = (trace + std::sqrt(trace * trace - 4.0 * det)) / 2.0;

    search_radius = parameters_->loop_mahalanobis_threshold * std::sqrt(max_eigenvalue);
  }

  if (parameters_->loop_debug_enable) {
    SAM_INFO("Loop search radius: query={}, search_radius={}", query.key, search_radius);
  }

  std::vector<std::shared_ptr<const KeyFrame>> nearby_keyframes =
    map_database_->getNearbyKeyFrames(query.pose, search_radius);

  std::shared_ptr<const KeyFrame> best_candidate;
  double best_mahalanobis_squared = std::numeric_limits<double>::infinity();

  for (const std::shared_ptr<const KeyFrame> & candidate : nearby_keyframes) {
    double mahalanobis_squared = std::numeric_limits<double>::infinity();

    const size_t order_distance =
      map_database_->getKeyFrameOrderDistance(candidate->key, query.key);

    if (order_distance < parameters_->loop_minimum_key_separation) {
      if (parameters_->loop_debug_enable) {
        SAM_INFO(
          "Loop candidate rejected: query={}, candidate={}, order_distance={}, "
          "minimum_key_separation={}",
          query.key, candidate->key, order_distance, parameters_->loop_minimum_key_separation);
      }
      continue;
    }

    if (!isCandidate(query, *candidate, mahalanobis_squared)) {
      continue;
    }

    if (parameters_->loop_debug_enable) {
      SAM_INFO(
        "Loop closure candidate: query={}, candidate={}, order_distance={}, mahalanobis_squared={}",
        query.key, candidate->key, order_distance, mahalanobis_squared);
    }

    if (mahalanobis_squared < best_mahalanobis_squared) {
      best_mahalanobis_squared = mahalanobis_squared;
      best_candidate = candidate;
    }
  }

  std::vector<double> resolutions;
  resolutions = scan_matcher_->fieldResolutions();

  LoopClosureProposal proposal;

  if (best_candidate) {
    const std::shared_ptr<const KeyFrame> & candidate = best_candidate;
    const std::vector<Point2D> & candidate_points = candidate->scan->points2D();

    SubmapGrid submap_grid(resolutions);

    const std::size_t requested_window_size = parameters_->submap_window_size;

    const std::size_t past_window_size = (requested_window_size / 2) + 1;

    auto past_keyframes = map_database_->getKeyFrameWindow(candidate->key, past_window_size);

    const std::size_t future_window_size =
      requested_window_size / 2 + (past_window_size - past_keyframes.size());

    const auto future_keyframes =
      map_database_->getKeyFrameWindowAfter(candidate->key, future_window_size);

    past_keyframes.insert(past_keyframes.end(), future_keyframes.begin(), future_keyframes.end());

    SAM_INFO(
      "Loop closure submap window: candidate={}, window_size={}, actual_size={}", candidate->key,
      requested_window_size, past_keyframes.size());

    for (const auto & neighbor : past_keyframes) {
      std::vector<Point2D> points;
      if (neighbor->key == candidate->key) {
        points = candidate_points;
      } else {
        const gtsam::Pose3 neighbor_in_candidate =
          candidate->pose.inverse().compose(neighbor->pose);
        points = Utils::transformScanPoints(neighbor->scan->points2D(), neighbor_in_candidate);
      }
      submap_grid.add(points, neighbor->key);
    }

    const Pose2D pose_estimate = Utils::toPose2D(candidate->pose.between(
      query.pose));  // Utils::toPose2D(query.pose.inverse().compose(candidate->pose));

    const CsmResult result =
      scan_matcher_->match(submap_grid, query.scan->points2D(), pose_estimate);

    // Can be used to debug the likelihood field that is built here.
    // const std::string debug_image_path = "/workspaces/ros2/debug_out";
    // static std::atomic_uint64_t debug_image_number{0};
    // const uint64_t image_number = debug_image_number.fetch_add(1, std::memory_order_relaxed);
    // const auto save_debug_image = [&](const CsmResult::DebugImage & image, const char * name) {
    //   if (image.width <= 0 || image.height <= 0 || image.pixels.empty()) {
    //     return;
    //   }

    //   cv::Mat image_mat(image.height, image.width, CV_8UC1);
    //   std::copy(image.pixels.begin(), image.pixels.end(), image_mat.data);

    //   cv::Mat bgr;
    //   cv::applyColorMap(image_mat, bgr, cv::COLORMAP_TURBO);

    //   cv::Mat rgb;
    //   cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    //   const std::string filename =
    //     debug_image_path + "/loop_closure_" + name + "_" + std::to_string(image_number) + ".png";
    //   if (!cv::imwrite(filename, rgb)) {
    //     SAM_WARN("Failed to write loop-closure debug image: {}", filename);
    //   }
    // };

    // save_debug_image(result.low_res_debug, "low");
    // save_debug_image(result.high_res_debug, "high");
    // SAM_INFO(
    //   "Loop-closure debug image number: {}. Amount of keyframes added to submap: {}",
    //   image_number, submap_grid.size());

    if (result.score < parameters_->loop_minimum_score) {
      if (parameters_->loop_debug_enable) {
        SAM_INFO(
          "Loop candidate rejected: query={}, candidate={}, score={}, minimum_score={}", query.key,
          candidate->key, result.score, parameters_->loop_minimum_score);
      }
    } else {
      Pose2D constrained_pose = result.optimized_pose;

      proposal.from_key = candidate->key;
      proposal.to_key = query.key;

      proposal.relative_pose = Utils::toPose3(constrained_pose);
      proposal.covariance = gtsam::Matrix66::Zero();

      proposal.covariance(0, 0) = parameters_->unobservable_variance;
      proposal.covariance(1, 1) = parameters_->unobservable_variance;
      proposal.covariance(2, 2) = result.covariance(2, 2);
      proposal.covariance(3, 3) = result.covariance(0, 0);
      proposal.covariance(4, 4) = result.covariance(1, 1);
      proposal.covariance(5, 5) = parameters_->unobservable_variance;
      proposal.covariance(3, 4) = result.covariance(0, 1);
      proposal.covariance(4, 3) = result.covariance(1, 0);
      proposal.covariance(3, 2) = result.covariance(0, 2);
      proposal.covariance(2, 3) = result.covariance(2, 0);
      proposal.covariance(4, 2) = result.covariance(1, 2);
      proposal.covariance(2, 4) = result.covariance(2, 1);

      proposal.score = result.score;

      if (parameters_->loop_debug_enable) {
        SAM_INFO(
          "Loop candidate matched: query={}, candidate={}, score={}, pose_estimate=({}, {}, {}), "
          "relative_pose=({}, {}, {}), csm_covariance=({}, {}, {})",
          query.key, candidate->key, result.score, pose_estimate.x, pose_estimate.y,
          pose_estimate.yaw, constrained_pose.x, constrained_pose.y, constrained_pose.yaw,
          result.covariance(0, 0), result.covariance(1, 1), result.covariance(2, 2));
      }
    }
  }

  if (parameters_->loop_debug_enable) {
    SAM_INFO(
      "Loop query processed: query={}, nearby_keyframes={}", query.key, nearby_keyframes.size());
  }

  std::vector<LoopClosureProposal> proposals;
  if (best_candidate && proposal.score >= parameters_->loop_minimum_score) {
    proposals.push_back(proposal);
  }

  return proposals;
}

bool LoopClosureDetector::isCandidate(
  const KeyFrame & query, const KeyFrame & candidate, double & mahalanobis_squared) const
{
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
    if (parameters_->loop_debug_enable) {
      SAM_INFO(
        "Loop candidate rejected: query={}, candidate={}, yaw_difference={}, "
        "maximum_yaw_difference={}",
        query.key, candidate.key, std::abs(relative_pose.rotation().yaw()),
        parameters_->loop_maximum_yaw_difference);
    }
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
  mahalanobis_squared = error_se2.transpose() * cov_se2.inverse() * error_se2;
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

  bool rejected = mahalanobis_squared > mahalanobis_limit;

  if (rejected && parameters_->loop_debug_enable) {
    SAM_INFO(
      "Loop candidate rejected: query={}, candidate={}, mahalanobis_squared={}, "
      "mahalanobis_limit={}",
      query.key, candidate.key, mahalanobis_squared, mahalanobis_limit);
  }

  return !rejected;
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
    // Proposals and queries belong to the old graph. Keeping them across a state load can add
    // constraints computed from poses that no longer exist in the restored state.
    input_queue_.clear();
    output_queue_.clear();
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

      if (stopping_) {
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
      SAM_INFO("Loop closure timing [ms]: total={}", elapsed_ms);
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
