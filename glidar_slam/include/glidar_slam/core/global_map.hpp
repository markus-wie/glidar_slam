#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace glidar_slam::core {

class GlobalMap
{
public:
  struct Info
  {
    std::uint32_t width{0};
    std::uint32_t height{0};
    double resolution{0.05};
    double origin_x{0.0};
    double origin_y{0.0};
  };

  explicit GlobalMap(double resolution = 0.05)
  {
    info_.resolution = resolution;
  }

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

}  // namespace glidar_slam::core
