#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "glidar_slam/key_frame.hpp"
#include "glidar_slam/map_database.hpp"
#include "glidar_slam/parameters.hpp"
#include "glidar_slam/scan_matcher.hpp"
#include "glidar_slam/scan_matcher/correlative_scan_matcher.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam {

struct LoopClosureProposal
{
  uint64_t from_key{0};
  uint64_t to_key{0};
  gtsam::Pose3 relative_pose;
  gtsam::Matrix66 covariance{gtsam::Matrix66::Zero()};
  double score{0.0};
  double overlap{0.0};
};

class LoopClosureDetector
{
public:
  LoopClosureDetector(
    const std::shared_ptr<Parameters> & parameters,
    const std::shared_ptr<MapDatabase> & map_database, std::unique_ptr<ScanMatcher> scan_matcher);
  ~LoopClosureDetector();

  void start();
  void stop();
  bool submit(KeyFrame snapshot);
  std::optional<LoopClosureProposal> tryPopProposal();

private:
  std::vector<LoopClosureProposal> findClosures(const KeyFrame & query);
  bool isCandidate(
    const KeyFrame & query, const KeyFrame & candidate, double & mahalanobis_squared) const;

  void run();

  std::shared_ptr<Parameters> parameters_;
  std::shared_ptr<MapDatabase> map_database_;
  std::unique_ptr<ScanMatcher> scan_matcher_;

  std::deque<KeyFrame> input_queue_;
  std::deque<LoopClosureProposal> output_queue_;

  std::mutex mutex_;
  std::condition_variable condition_variable_;
  bool running_{false};
  bool stopping_{false};
  std::thread worker_thread_;
};

}  // namespace glidar_slam
