#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glidar_slam/mapping/global_map.hpp"
#include "glidar_slam/mapping/local_map.hpp"
#include "glidar_slam/parameters.hpp"

namespace glidar_slam::mapping {

struct GlobalOccupancyCell
{
  int x{0};
  int y{0};
  float log_odds{0.0f};
};

struct GlobalGroundCell
{
  int x{0};
  int y{0};
  float log_odds{0.0f};
  std::uint8_t r{0};
  std::uint8_t g{0};
  std::uint8_t b{0};
};

template <typename TCell, typename TGridOutput>
class BaseAccumulator
{
public:
  void apply(std::uint64_t key, std::uint64_t revision, const std::vector<TCell> & local_cells)
  {
    if (auto it = history_.find(key); it != history_.end()) {
      if (it->second.revision >= revision) {
        return;
      }
      for (const auto & cell : it->second.cells) {
        removeObservation(cell);
      }
    }

    history_[key] = {revision, local_cells};
    for (const auto & cell : local_cells) {
      addObservation(cell);
    }
  }

  void clear()
  {
    history_.clear();
    clearState();
  }

  virtual std::optional<TGridOutput> publish() const = 0;

protected:
  virtual ~BaseAccumulator() = default;

  virtual void addObservation(const TCell & cell) = 0;
  virtual void removeObservation(const TCell & cell) = 0;
  virtual void clearState() = 0;

  /**
   * @brief Computes a unique key for a cell based on its x and y coordinates.
   *
   * This function combines two 32-bit integers representing the x and y coordinates into a single
   * 64-bit integer. This is done to leverage the efficient hashing and lookup capabilities of
   * unordered maps for 64-bit keys.
   *
   * @param x The x coordinate of the cell.
   * @param y The y coordinate of the cell.
   * @return A 64-bit integer key representing the cell.
   */
  static std::int64_t cellKey(int x, int y)
  {
    return (static_cast<std::int64_t>(x) << 32) ^ static_cast<std::uint32_t>(y);
  }

  /**
   * @brief Extracts the x and y coordinates from a 64-bit cell key.
   *
   * This function reverses the process performed by cellKey, retrieving the original x and y
   * coordinates from the combined 64-bit key.
   *
   * @param key The 64-bit integer key representing the cell.
   * @return A pair of integers representing the x and y coordinates of the cell.
   */
  static std::pair<int, int> cellCoordinates(std::int64_t key)
  {
    const int x = static_cast<int>(key >> 32);
    const int y = static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff));
    return {x, y};
  }

private:
  struct Transaction
  {
    std::uint64_t revision{0};
    std::vector<TCell> cells;
  };
  std::unordered_map<std::uint64_t, Transaction> history_;
};

class OccupancyAccumulator : public BaseAccumulator<GlobalOccupancyCell, OccupancyGrid>
{
public:
  explicit OccupancyAccumulator(std::shared_ptr<Parameters> parameters);
  std::optional<OccupancyGrid> publish() const override;

protected:
  void addObservation(const GlobalOccupancyCell & cell) override;
  void removeObservation(const GlobalOccupancyCell & cell) override;
  void clearState() override;

private:
  std::shared_ptr<Parameters> parameters_;
  std::unordered_map<std::int64_t, float> evidence_;
};

class GroundMarkingAccumulator : public BaseAccumulator<GlobalGroundCell, GroundMarkingGrid>
{
public:
  explicit GroundMarkingAccumulator(std::shared_ptr<Parameters> parameters);
  std::optional<GroundMarkingGrid> publish() const override;

protected:
  void addObservation(const GlobalGroundCell & cell) override;
  void removeObservation(const GlobalGroundCell & cell) override;
  void clearState() override;

private:
  std::shared_ptr<Parameters> parameters_;
  std::unordered_map<std::int64_t, float> marking_;
};

class GroundTextureAccumulator : public BaseAccumulator<GlobalGroundCell, GroundTextureGrid>
{
public:
  explicit GroundTextureAccumulator(std::shared_ptr<Parameters> parameters);
  std::optional<GroundTextureGrid> publish() const override;

protected:
  void addObservation(const GlobalGroundCell & cell) override;
  void removeObservation(const GlobalGroundCell & cell) override;
  void clearState() override;

private:
  struct TextureValue
  {
    std::uint64_t count{0};
    std::uint64_t r{0};
    std::uint64_t g{0};
    std::uint64_t b{0};
  };

  std::shared_ptr<Parameters> parameters_;
  std::unordered_map<std::int64_t, TextureValue> texture_;
};

}  // namespace glidar_slam::mapping
