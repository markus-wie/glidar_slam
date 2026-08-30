#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "glidar_slam/parameters.hpp"

namespace glidar_slam::mapping {

class GlobalMap
{
public:
  struct Info
  {
    uint32_t width{0};
    uint32_t height{0};
    double resolution{0.05};
    double origin_x{0.0};
    double origin_y{0.0};
  };

  explicit GlobalMap(Info info);

  virtual ~GlobalMap() = default;

  const Info & getInfo() const
  {
    return info_;
  }

protected:
  void setInfo(const Info & info)
  {
    info_ = info;
  }

private:
  Info info_;
};

class OccupancyGrid : public GlobalMap
{
public:
  OccupancyGrid(Info info, std::vector<int8_t> data);

  const std::vector<int8_t> & getData() const
  {
    return data_;
  }

private:
  std::vector<int8_t> data_;
};

class GroundMarkingGrid : public GlobalMap
{
public:
  GroundMarkingGrid(Info info, std::vector<int8_t> data);

  const std::vector<int8_t> & getData() const
  {
    return data_;
  }

private:
  std::vector<int8_t> data_;
};

class GroundTextureGrid : public GlobalMap
{
public:
  GroundTextureGrid(Info info, std::vector<uint8_t> rgb_data);

  const std::vector<uint8_t> & getRgbData() const
  {
    return rgb_data_;
  }

private:
  std::vector<uint8_t> rgb_data_;
};

}  // namespace glidar_slam::mapping
