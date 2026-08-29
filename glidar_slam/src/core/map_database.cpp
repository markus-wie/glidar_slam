#include "glidar_slam/core/map_database.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nanoflann.hpp"

namespace glidar_slam::core {

class MapDatabase::SpatialIndex
{
public:
  static constexpr size_t kPendingRebuildThreshold = 32;

  struct PointCloud
  {
    size_t kdtree_get_point_count() const
    {
      return points.size();
    }

    double kdtree_get_pt(size_t index, size_t dimension) const
    {
      return points[index][dimension];
    }

    template <class BoundingBox>
    bool kdtree_get_bbox(BoundingBox & bounding_box) const
    {
      static_cast<void>(bounding_box);
      return false;
    }

  private:
    std::vector<std::array<double, 2>> points;

    friend class MapDatabase::SpatialIndex;
  };

  using Tree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 2>;

  SpatialIndex() : tree_(2, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10))
  {
  }

  void rebuild(const std::unordered_map<uint64_t, Node> & nodes)
  {
    cloud_.points.clear();
    keyframe_indices_.clear();
    cloud_.points.reserve(nodes.size());
    keyframe_indices_.reserve(nodes.size());

    // UPDATED: Extract the pose from the Node's keyframe
    for (const auto & [key, node] : nodes) {
      const gtsam::Point3 translation = node.keyframe->pose.translation();
      cloud_.points.push_back({translation.x(), translation.y()});
      keyframe_indices_.push_back(key);
    }

    tree_.buildIndex();
    pending_points_.clear();
    pending_indices_.clear();
  }

  void addPending(uint64_t key, const KeyFrame & keyframe)
  {
    const gtsam::Point3 translation = keyframe.pose.translation();
    pending_points_.push_back({translation.x(), translation.y()});
    pending_indices_.push_back(key);
  }

  bool needsRebuild() const
  {
    return pending_points_.size() >= kPendingRebuildThreshold;
  }

  std::vector<uint64_t> radiusSearch(const gtsam::Pose3 & query_pose, double radius) const
  {
    std::vector<uint64_t> result;
    if (radius < 0.0 || cloud_.points.empty()) {
      return result;
    }

    const gtsam::Point3 & translation = query_pose.translation();
    const double query[2] = {translation.x(), translation.y()};
    std::vector<nanoflann::ResultItem<unsigned int, double>> matches;
    const auto match_count = tree_.radiusSearch(query, radius * radius, matches);
    matches.resize(match_count);
    result.reserve(matches.size());
    for (const auto & match : matches) {
      result.push_back(keyframe_indices_[match.first]);
    }

    const double radius_squared = radius * radius;
    for (size_t index = 0; index < pending_points_.size(); ++index) {
      const double dx = pending_points_[index][0] - query[0];
      const double dy = pending_points_[index][1] - query[1];
      if (dx * dx + dy * dy <= radius_squared) {
        result.push_back(pending_indices_[index]);
      }
    }

    return result;
  }

private:
  PointCloud cloud_;
  std::vector<uint64_t> keyframe_indices_;
  Tree tree_;
  std::vector<std::array<double, 2>> pending_points_;
  std::vector<uint64_t> pending_indices_;
};

MapDatabase::~MapDatabase() = default;
MapDatabase::MapDatabase() = default;

void MapDatabase::addEdgeInternal(uint64_t from_key, uint64_t to_key, EdgeType type)
{
  auto from_it = nodes_.find(from_key);
  auto to_it = nodes_.find(to_key);
  if (from_it == nodes_.end() || to_it == nodes_.end()) {
    return;
  }

  auto & from_node = from_it->second;
  auto & to_node = to_it->second;

  const auto has_outgoing_edge = [to_key, type](const Edge & edge) {
    return edge.target_key == to_key && edge.type == type;
  };
  if (
    std::find_if(
      from_node.outgoing_edges.begin(), from_node.outgoing_edges.end(), has_outgoing_edge) ==
    from_node.outgoing_edges.end()) {
    from_node.outgoing_edges.push_back(Edge{to_key, type});
  }

  const auto has_incoming_edge = [from_key, type](const Edge & edge) {
    return edge.target_key == from_key && edge.type == type;
  };
  if (
    std::find_if(to_node.incoming_edges.begin(), to_node.incoming_edges.end(), has_incoming_edge) ==
    to_node.incoming_edges.end()) {
    to_node.incoming_edges.push_back(Edge{from_key, type});
  }
}

void MapDatabase::addKeyFrame(std::shared_ptr<KeyFrame> keyframe)
{
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  if (!keyframe || nodes_.find(keyframe->key) != nodes_.end()) {
    throw std::invalid_argument("Keyframe must be non-null and have a unique key");
  }

  const uint64_t key = keyframe->key;
  const size_t order_idx = keyframe_order_.size();

  Node node;
  node.keyframe = keyframe;
  node.order_index = order_idx;

  nodes_.emplace(key, std::move(node));
  keyframe_order_.push_back(key);

  // Spatial Index maintenance
  if (!spatial_index_) {
    spatial_index_ = std::make_unique<SpatialIndex>();
  }
  spatial_index_->addPending(key, *keyframe);
  if (spatial_index_->needsRebuild()) {
    rebuildSpatialIndex();
  }
}

void MapDatabase::addEdge(uint64_t from_key, uint64_t to_key, EdgeType type)
{
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  addEdgeInternal(from_key, to_key, type);
}

std::vector<std::shared_ptr<const KeyFrame>> MapDatabase::updatePoses(
  const gtsam::Values & optimized_values,
  const std::unordered_map<uint64_t, gtsam::Matrix66> & optimized_covariances)
{
  std::unique_lock<std::shared_mutex> lock(rw_mutex_);
  std::vector<std::shared_ptr<const KeyFrame>> changed;

  for (const uint64_t key : keyframe_order_) {
    Node & node = nodes_.at(key);
    const std::shared_ptr<KeyFrame> & keyframe = node.keyframe;

    if (optimized_values.exists(key)) {
      const gtsam::Pose3 optimized_pose = optimized_values.at<gtsam::Pose3>(key);
      if (!keyframe->pose.matrix().isApprox(optimized_pose.matrix(), 1e-12)) {
        keyframe->pose = optimized_pose;
        ++keyframe->revision;
        changed.push_back(keyframe);
      }
    }

    const auto covariance_it = optimized_covariances.find(key);
    if (covariance_it != optimized_covariances.end()) {
      keyframe->covariance = covariance_it->second;
    }
  }

  rebuildSpatialIndex();
  return changed;
}

std::shared_ptr<const KeyFrame> MapDatabase::getSnapshot(uint64_t key) const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);

  const auto it = nodes_.find(key);
  if (it != nodes_.end()) {
    const Node & node = it->second;
    return node.keyframe;
  }

  return {};
}

std::vector<std::shared_ptr<const KeyFrame>> MapDatabase::getSnapshots() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);

  std::vector<std::shared_ptr<const KeyFrame>> snapshots;
  snapshots.reserve(keyframe_order_.size());

  for (const uint64_t key : keyframe_order_) {
    const Node & node = nodes_.at(key);
    snapshots.push_back(node.keyframe);
  }

  return snapshots;
}

std::vector<std::shared_ptr<const KeyFrame>> MapDatabase::getNearbyKeyFrames(
  const gtsam::Pose3 & query_pose, double radius) const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  std::vector<std::shared_ptr<const KeyFrame>> nearby_keyframes;
  if (radius < 0.0 || !spatial_index_) {
    return nearby_keyframes;
  }

  for (const uint64_t key : spatial_index_->radiusSearch(query_pose, radius)) {
    nearby_keyframes.push_back(nodes_.at(key).keyframe);
  }
  return nearby_keyframes;
}

std::vector<std::shared_ptr<const KeyFrame>> MapDatabase::getAllKeyFrames() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  std::vector<std::shared_ptr<const KeyFrame>> result;
  result.reserve(keyframe_order_.size());

  for (const uint64_t key : keyframe_order_) {
    result.push_back(nodes_.at(key).keyframe);
  }
  return result;
}

std::vector<std::pair<uint64_t, uint64_t>> MapDatabase::getLoopClosures() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  std::vector<std::pair<uint64_t, uint64_t>> loop_closures;
  for (const auto & [key, node] : nodes_) {
    for (const auto & edge : node.outgoing_edges) {
      if (edge.type == EdgeType::LoopClosure) {
        loop_closures.emplace_back(key, edge.target_key);
      }
    }
  }
  return loop_closures;
}

std::vector<MapDatabase::GraphEdge> MapDatabase::getEdges() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  std::vector<GraphEdge> edges;
  for (const auto & [key, node] : nodes_) {
    for (const auto & edge : node.outgoing_edges) {
      edges.push_back(
        {key, edge.target_key, edge.type == EdgeType::Neighbor ? "Neighbor" : "LoopClosure"});
    }
  }
  return edges;
}

std::vector<std::shared_ptr<const KeyFrame>> MapDatabase::getKeyFrameWindow(
  uint64_t base_key, size_t window_size) const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  if (window_size == 0 || nodes_.find(base_key) == nodes_.end()) {
    return {};
  }

  std::vector<std::shared_ptr<const KeyFrame>> result;
  result.reserve(window_size);

  std::queue<uint64_t> queue;
  std::unordered_set<uint64_t> visited;

  queue.push(base_key);
  visited.insert(base_key);

  while (!queue.empty() && result.size() < window_size) {
    const uint64_t current_key = queue.front();
    queue.pop();

    auto it = nodes_.find(current_key);
    if (it == nodes_.end()) {
      continue;
    }

    result.push_back(it->second.keyframe);
    if (result.size() == window_size) {
      break;
    }

    std::vector<uint64_t> neighbors;
    neighbors.reserve(it->second.incoming_edges.size());
    for (const auto & edge : it->second.incoming_edges) {
      if (visited.find(edge.target_key) == visited.end()) {
        neighbors.push_back(edge.target_key);
      }
    }

    std::sort(neighbors.begin(), neighbors.end(), [this](uint64_t a, uint64_t b) {
      auto it_a = nodes_.find(a);
      auto it_b = nodes_.find(b);
      const size_t order_a = (it_a != nodes_.end()) ? it_a->second.order_index : 0;
      const size_t order_b = (it_b != nodes_.end()) ? it_b->second.order_index : 0;
      return order_a > order_b;
    });

    for (const uint64_t neighbor_key : neighbors) {
      if (visited.insert(neighbor_key).second) {
        queue.push(neighbor_key);
      }
    }
  }

  std::reverse(result.begin(), result.end());
  return result;
}

std::vector<std::shared_ptr<const KeyFrame>> MapDatabase::getKeyFrameWindowAfter(
  uint64_t base_key, size_t window_size) const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  auto it = nodes_.find(base_key);
  if (it == nodes_.end() || window_size == 0) {
    return {};
  }

  const size_t start_idx = it->second.order_index + 1;
  const size_t end_idx = std::min(keyframe_order_.size(), start_idx + window_size);

  std::vector<std::shared_ptr<const KeyFrame>> result;
  result.reserve(end_idx - start_idx);
  for (size_t i = start_idx; i < end_idx; ++i) {
    result.push_back(nodes_.at(keyframe_order_[i]).keyframe);
  }
  return result;
}

size_t MapDatabase::getKeyFrameOrderDistance(uint64_t earlier_key, uint64_t later_key) const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  auto it_early = nodes_.find(earlier_key);
  auto it_late = nodes_.find(later_key);

  if (
    it_early == nodes_.end() || it_late == nodes_.end() ||
    it_early->second.order_index >= it_late->second.order_index) {
    return 0;
  }
  return static_cast<size_t>(it_late->second.order_index - it_early->second.order_index);
}

std::shared_ptr<const KeyFrame> MapDatabase::getKeyFrame(uint64_t key) const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  const auto keyframe = nodes_.find(key);
  if (keyframe != nodes_.end()) {
    return keyframe->second.keyframe;
  }
  return nullptr;
}

std::shared_ptr<const KeyFrame> MapDatabase::getLatestKeyFrame() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  if (!keyframe_order_.empty()) {
    return nodes_.at(keyframe_order_.back()).keyframe;
  }
  return nullptr;
}

size_t MapDatabase::size() const
{
  std::shared_lock<std::shared_mutex> lock(rw_mutex_);
  return keyframe_order_.size();
}

uint64_t MapDatabase::getNextKey() const
{
  return next_keyframe_key_.load(std::memory_order_relaxed);
}

uint64_t MapDatabase::incrementNextKey()
{
  return next_keyframe_key_.fetch_add(1, std::memory_order_relaxed);
}

void MapDatabase::rebuildSpatialIndex()
{
  if (!spatial_index_) {
    spatial_index_ = std::make_unique<SpatialIndex>();
  }
  spatial_index_->rebuild(nodes_);
}

}  // namespace glidar_slam::core
