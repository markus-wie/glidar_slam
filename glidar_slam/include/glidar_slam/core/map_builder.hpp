#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "glidar_slam/core/global_map/ground_marking_grid.hpp"
#include "glidar_slam/core/global_map/ground_texture_grid.hpp"
#include "glidar_slam/core/global_map/map_data.hpp"
#include "glidar_slam/core/global_map/occupancy_grid.hpp"
#include "glidar_slam/core/key_frame.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "gtsam/geometry/Pose3.h"

namespace glidar_slam::core {

struct GlobalMapSnapshot
{
  global_map::OccupancyGrid occupancy;
  global_map::GroundMarkingGrid marking;
  global_map::GroundTextureGrid texture;
  std::uint64_t generation{0};
};

class MapBuilder
{
public:
  explicit MapBuilder(const std::shared_ptr<Parameters> & parameters);
  ~MapBuilder();

  void start();
  void stop();
  bool submit(std::shared_ptr<const KeyFrame> keyframe);
  bool rebuild(std::vector<std::shared_ptr<const KeyFrame>> keyframes);
  std::shared_ptr<const GlobalMapSnapshot> getLatest() const;

private:
  struct AppliedContribution
  {
    std::uint64_t revision{0};
    std::vector<global_map::OccupancyCellContribution> cells;
    std::vector<global_map::GroundCell> ground_cells;
  };

  struct TextureValue
  {
    std::uint64_t count{0};
    std::uint64_t r{0};
    std::uint64_t g{0};
    std::uint64_t b{0};
  };

  void run();
  void apply(const KeyFrame & keyframe);
  void clearAccumulatedMap();
  void publish();

  std::shared_ptr<Parameters> parameters_;
  mutable std::mutex mutex_;
  std::condition_variable condition_variable_;
  std::unordered_map<std::uint64_t, std::shared_ptr<const KeyFrame>> pending_;
  std::vector<std::shared_ptr<const KeyFrame>> pending_rebuild_;
  std::unordered_map<std::uint64_t, AppliedContribution> applied_;
  std::unordered_map<std::int64_t, int> evidence_;
  std::unordered_map<std::int64_t, float> marking_;
  std::unordered_map<std::int64_t, TextureValue> texture_;
  std::shared_ptr<const GlobalMapSnapshot> latest_;
  std::thread worker_thread_;
  bool running_{false};
  bool stopping_{false};
  bool rebuild_requested_{false};
  std::uint64_t generation_{0};
};

global_map::LocalMapData buildLocalOccupancy(const PointCloudXYZ & scan, double resolution);
void addLocalGroundMap(
  global_map::LocalMapData & data, const PointCloudXYZRGBA & cloud, const Parameters & parameters);

}  // namespace glidar_slam::core
