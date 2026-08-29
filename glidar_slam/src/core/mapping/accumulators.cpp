#include "glidar_slam/core/mapping/accumulators.hpp"

#include <algorithm>
#include <limits>

namespace glidar_slam::core::mapping {

namespace {
template <typename MapType>
bool getBounds(const MapType & values, int & min_x, int & min_y, int & max_x, int & max_y)
{
  min_x = std::numeric_limits<int>::max();
  min_y = std::numeric_limits<int>::max();
  max_x = std::numeric_limits<int>::lowest();
  max_y = std::numeric_limits<int>::lowest();
  bool has_data = false;

  for (const auto & [key, value] : values) {
    if constexpr (std::is_same_v<MapType, std::unordered_map<std::int64_t, float>>) {
      if (std::abs(value) < 1e-5f) {
        continue;
      }
    } else {
      if (value.count == 0) {
        continue;
      }
    }
    const int x = static_cast<int>(key >> 32);
    const int y = static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff));
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
    has_data = true;
  }
  return has_data;
}
}  // namespace

OccupancyAccumulator::OccupancyAccumulator(std::shared_ptr<Parameters> parameters)
: parameters_(std::move(parameters))
{
}

void OccupancyAccumulator::addObservation(const GlobalOccupancyCell & cell)
{
  evidence_[cellKey(cell.x, cell.y)] += cell.log_odds;
}

void OccupancyAccumulator::removeObservation(const GlobalOccupancyCell & cell)
{
  const auto key = cellKey(cell.x, cell.y);
  auto it = evidence_.find(key);
  if (it != evidence_.end()) {
    it->second -= cell.log_odds;
    if (std::abs(it->second) < 1e-5f) {
      evidence_.erase(it);
    }
  }
}

void OccupancyAccumulator::clearState()
{
  evidence_.clear();
}

std::optional<OccupancyGrid> OccupancyAccumulator::publish() const
{
  if (parameters_->mapping_occupancy_resolution <= 0.0) {
    return std::nullopt;
  }

  int min_x{0};
  int min_y{0};
  int max_x{0};
  int max_y{0};
  if (!getBounds(evidence_, min_x, min_y, max_x, max_y)) {
    return std::nullopt;
  }

  GlobalMap::Info info;
  info.resolution = parameters_->mapping_occupancy_resolution;
  info.origin_x = static_cast<double>(min_x) * info.resolution;
  info.origin_y = static_cast<double>(min_y) * info.resolution;
  info.width = static_cast<std::uint32_t>(max_x - min_x + 1);
  info.height = static_cast<std::uint32_t>(max_y - min_y + 1);

  std::vector<std::int8_t> data(static_cast<std::size_t>(info.width) * info.height, -1);
  for (const auto & [key, value] : evidence_) {
    if (std::abs(value) < 1e-5f) {
      continue;
    }

    const auto [x, y] = cellCoordinates(key);
    const std::size_t index = static_cast<std::size_t>(y - min_y) * info.width + (x - min_x);

    data[index] = value < 0.0f ? 0 : 100;
  }

  return std::make_optional(OccupancyGrid(info, std::move(data)));
}

GroundMarkingAccumulator::GroundMarkingAccumulator(std::shared_ptr<Parameters> parameters)
: parameters_(std::move(parameters))
{
}

void GroundMarkingAccumulator::addObservation(const GlobalGroundCell & cell)
{
  marking_[cellKey(cell.x, cell.y)] += cell.log_odds;
}

void GroundMarkingAccumulator::removeObservation(const GlobalGroundCell & cell)
{
  const auto key = cellKey(cell.x, cell.y);
  auto it = marking_.find(key);
  if (it != marking_.end()) {
    it->second -= cell.log_odds;
    if (std::abs(it->second) < 1e-5f) {
      marking_.erase(it);
    }
  }
}

void GroundMarkingAccumulator::clearState()
{
  marking_.clear();
}

std::optional<GroundMarkingGrid> GroundMarkingAccumulator::publish() const
{
  if (parameters_->mapping_ground_resolution <= 0.0) {
    return std::nullopt;
  }

  int min_x{0};
  int min_y{0};
  int max_x{0};
  int max_y{0};
  if (!getBounds(marking_, min_x, min_y, max_x, max_y)) {
    return std::nullopt;
  }

  GlobalMap::Info info;
  info.resolution = parameters_->mapping_ground_resolution;
  info.origin_x = static_cast<double>(min_x) * info.resolution;
  info.origin_y = static_cast<double>(min_y) * info.resolution;
  info.width = static_cast<std::uint32_t>(max_x - min_x + 1);
  info.height = static_cast<std::uint32_t>(max_y - min_y + 1);

  std::vector<std::int8_t> data(static_cast<std::size_t>(info.width) * info.height, 0);
  for (const auto & [key, value] : marking_) {
    if (std::abs(value) < 1e-5f) {
      continue;
    }

    const auto [x, y] = cellCoordinates(key);
    const std::size_t index = static_cast<std::size_t>(y - min_y) * info.width + (x - min_x);

    data[index] = value > 0.0f ? 100 : 0;
  }

  return std::make_optional(GroundMarkingGrid(info, std::move(data)));
}

GroundTextureAccumulator::GroundTextureAccumulator(std::shared_ptr<Parameters> parameters)
: parameters_(std::move(parameters))
{
}

void GroundTextureAccumulator::addObservation(const GlobalGroundCell & cell)
{
  auto & tex = texture_[cellKey(cell.x, cell.y)];
  ++tex.count;
  tex.r += cell.r;
  tex.g += cell.g;
  tex.b += cell.b;
}

void GroundTextureAccumulator::removeObservation(const GlobalGroundCell & cell)
{
  auto it = texture_.find(cellKey(cell.x, cell.y));
  if (it != texture_.end() && it->second.count > 0) {
    --it->second.count;
    it->second.r -= cell.r;
    it->second.g -= cell.g;
    it->second.b -= cell.b;
    if (it->second.count == 0) {
      texture_.erase(it);
    }
  }
}

void GroundTextureAccumulator::clearState()
{
  texture_.clear();
}

std::optional<GroundTextureGrid> GroundTextureAccumulator::publish() const
{
  if (parameters_->mapping_ground_resolution <= 0.0) {
    return std::nullopt;
  }

  int min_x{0};
  int min_y{0};
  int max_x{0};
  int max_y{0};
  if (!getBounds(texture_, min_x, min_y, max_x, max_y)) {
    return std::nullopt;
  }

  GlobalMap::Info info;
  info.resolution = parameters_->mapping_ground_resolution;
  info.origin_x = static_cast<double>(min_x) * info.resolution;
  info.origin_y = static_cast<double>(min_y) * info.resolution;
  info.width = static_cast<std::uint32_t>(max_x - min_x + 1);
  info.height = static_cast<std::uint32_t>(max_y - min_y + 1);

  std::vector<std::uint8_t> data(static_cast<std::size_t>(info.width) * info.height * 3U, 0);
  for (const auto & [key, value] : texture_) {
    if (value.count == 0) {
      continue;
    }

    const auto [x, y] = cellCoordinates(key);
    const std::size_t index = static_cast<std::size_t>(y - min_y) * info.width + (x - min_x);

    data[index * 3U] = static_cast<std::uint8_t>((value.r + value.count / 2U) / value.count);
    data[index * 3U + 1U] = static_cast<std::uint8_t>((value.g + value.count / 2U) / value.count);
    data[index * 3U + 2U] = static_cast<std::uint8_t>((value.b + value.count / 2U) / value.count);
  }

  return std::make_optional(GroundTextureGrid(info, std::move(data)));
}

}  // namespace glidar_slam::core::mapping
