#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "glidar_slam/core/key_frame.hpp"
#include "glidar_slam/core/mapping/accumulators.hpp"
#include "glidar_slam/core/parameters.hpp"

namespace glidar_slam::core::mapping {

struct GlobalMapSnapshot
{
  std::optional<OccupancyGrid> occupancy;
  std::optional<GroundMarkingGrid> marking;
  std::optional<GroundTextureGrid> texture;
  std::uint64_t generation{0};
};

class MapBuilder
{
public:
  explicit MapBuilder(std::shared_ptr<Parameters> parameters);
  ~MapBuilder();

  void start();
  void stop();
  bool submit(std::shared_ptr<const KeyFrame> keyframe);
  bool rebuild(std::vector<std::shared_ptr<const KeyFrame>> keyframes);
  std::shared_ptr<const GlobalMapSnapshot> getLatest() const;

private:
  void run();
  void apply(const KeyFrame & keyframe);
  void clearAccumulatedMap();
  void publish();

  std::shared_ptr<Parameters> parameters_;
  mutable std::mutex mutex_;
  std::condition_variable condition_variable_;

  std::unordered_map<std::uint64_t, std::shared_ptr<const KeyFrame>> pending_;
  std::vector<std::shared_ptr<const KeyFrame>> pending_rebuild_;

  OccupancyAccumulator occupancy_;
  GroundMarkingAccumulator marking_;
  GroundTextureAccumulator texture_;

  std::shared_ptr<const GlobalMapSnapshot> latest_;
  std::thread worker_thread_;
  bool running_{false};
  bool stopping_{false};
  bool rebuild_requested_{false};
  std::uint64_t generation_{0};
};

}  // namespace glidar_slam::core::mapping
