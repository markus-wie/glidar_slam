#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glidar_slam/core/key_frame.hpp"
#include "glidar_slam/core/parameters.hpp"
#include "glidar_slam/core/state_snapshot.hpp"

namespace glidar_slam::core {

class MapDatabase
{
public:
  MapDatabase();
  ~MapDatabase();

  enum class EdgeType : uint8_t
  {
    Neighbor,
    LoopClosure
  };

  struct GraphEdge
  {
    uint64_t from_key;
    uint64_t to_key;
    std::string type;
  };

  void addKeyFrame(std::shared_ptr<KeyFrame> keyframe);

  void addEdge(uint64_t from_key, uint64_t to_key, EdgeType type = EdgeType::Neighbor);

  std::vector<std::shared_ptr<const KeyFrame>> updatePoses(
    const gtsam::Values & optimized_values,
    const std::unordered_map<uint64_t, gtsam::Matrix66> & optimized_covariances = {});

  std::vector<std::shared_ptr<const KeyFrame>> getNearbyKeyFrames(
    const gtsam::Pose3 & query_pose, double radius) const;

  std::shared_ptr<const KeyFrame> getClosestKeyFrame(const gtsam::Pose3 & query_pose) const;

  std::vector<std::shared_ptr<const KeyFrame>> getAllKeyFrames() const;

  std::vector<std::pair<uint64_t, uint64_t>> getLoopClosures() const;
  std::vector<GraphEdge> getEdges() const;

  std::vector<std::shared_ptr<const KeyFrame>> getKeyFrameWindow(
    uint64_t base_key, size_t window_size) const;

  std::vector<std::shared_ptr<const KeyFrame>> getKeyFrameWindowAfter(
    uint64_t base_key, size_t window_size) const;

  size_t getKeyFrameOrderDistance(uint64_t earlier_key, uint64_t later_key) const;

  std::shared_ptr<const KeyFrame> getKeyFrame(uint64_t key) const;
  std::shared_ptr<const KeyFrame> getLatestKeyFrame() const;

  size_t size() const;

  uint64_t getNextKey() const;

  bool restore(const std::vector<KeyFrame> & keyframes, uint64_t next_key);

  void rebuildSpatialIndex();

  void rebuildLocalMaps(const Parameters & parameters);

private:
  struct Edge
  {
    uint64_t target_key;
    EdgeType type;
  };

  /**
   * Unified Node structure to hold keyframe, its metadata, and topological edges.
   */
  struct Node
  {
    std::shared_ptr<KeyFrame> keyframe;
    size_t order_index{0};
    std::vector<Edge> incoming_edges;  // Used for reverse topological traversal
    std::vector<Edge> outgoing_edges;  // Used for forward traversal / loop closure extraction
  };

  uint64_t incrementNextKey();

  void addEdgeInternal(uint64_t from_key, uint64_t to_key, EdgeType type);

  std::unordered_map<uint64_t, Node> nodes_;
  std::vector<uint64_t> keyframe_order_;

  class SpatialIndex;

  mutable std::shared_mutex rw_mutex_;

  std::unique_ptr<SpatialIndex> spatial_index_;

  std::atomic<uint64_t> next_keyframe_key_{0};
};

}  // namespace glidar_slam::core
