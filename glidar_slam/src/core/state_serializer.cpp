#include "glidar_slam/core/state_serializer.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace glidar_slam::core {
namespace {

constexpr std::array<char, 8> kMagic{'S', 'A', 'M', 'S', 'L', 'A', 'M', '\0'};
constexpr std::uint64_t kMaxCollectionSize = 10'000'000;

template <typename T>
void write(std::ostream & stream, const T & value)
{
  std::array<char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  stream.write(bytes.data(), sizeof(T));
  if (!stream) {
    throw std::runtime_error("failed to write state file");
  }
}

template <typename T>
void read(std::istream & stream, T & value)
{
  std::array<char, sizeof(T)> bytes{};
  stream.read(bytes.data(), sizeof(T));
  if (!stream) {
    throw std::runtime_error("truncated state file");
  }
  std::memcpy(&value, bytes.data(), sizeof(T));
}

void writeString(std::ostream & stream, const std::string & value)
{
  write(stream, static_cast<std::uint64_t>(value.size()));
  stream.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!stream) {
    throw std::runtime_error("failed to write state string");
  }
}

std::string readString(std::istream & stream)
{
  std::uint64_t size = 0;
  read(stream, size);
  if (size > kMaxCollectionSize) {
    throw std::runtime_error("state string is too large");
  }
  std::string value(size, '\0');
  stream.read(value.data(), static_cast<std::streamsize>(size));
  if (!stream) {
    throw std::runtime_error("truncated state string");
  }
  return value;
}

void writePose(std::ostream & stream, const gtsam::Pose3 & pose)
{
  const auto rotation = pose.rotation().matrix();
  const auto & translation = pose.translation();
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      write(stream, rotation(row, column));
    }
  }
  for (int index = 0; index < 3; ++index) {
    write(stream, translation(index));
  }
}

gtsam::Pose3 readPose(std::istream & stream)
{
  gtsam::Matrix33 rotation;
  gtsam::Vector3 translation;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      read(stream, rotation(row, column));
    }
  }
  for (int index = 0; index < 3; ++index) {
    read(stream, translation(index));
  }
  return gtsam::Pose3(gtsam::Rot3(rotation), gtsam::Point3(translation));
}

template <int Rows, int Columns>
void writeMatrix(std::ostream & stream, const Eigen::Matrix<double, Rows, Columns> & matrix)
{
  for (int row = 0; row < Rows; ++row) {
    for (int column = 0; column < Columns; ++column) {
      write(stream, matrix(row, column));
    }
  }
}

template <int Rows, int Columns>
void readMatrix(std::istream & stream, Eigen::Matrix<double, Rows, Columns> & matrix)
{
  for (int row = 0; row < Rows; ++row) {
    for (int column = 0; column < Columns; ++column) {
      read(stream, matrix(row, column));
    }
  }
}

void writePointCloudXYZ(std::ostream & stream, const PointCloudXYZ & cloud)
{
  write(stream, static_cast<std::uint64_t>(cloud.size()));
  for (const auto & point : cloud) {
    write(stream, point.x);
    write(stream, point.y);
    write(stream, point.z);
  }
}

void writePointCloudRGBA(std::ostream & stream, const PointCloudXYZRGBA & cloud)
{
  write(stream, static_cast<std::uint64_t>(cloud.size()));
  for (const auto & point : cloud) {
    write(stream, point.x);
    write(stream, point.y);
    write(stream, point.z);
    write(stream, point.r);
    write(stream, point.g);
    write(stream, point.b);
    write(stream, point.a);
  }
}

std::uint64_t readCount(std::istream & stream, const char * what)
{
  std::uint64_t count = 0;
  read(stream, count);
  if (count > kMaxCollectionSize) {
    throw std::runtime_error(std::string(what) + " is too large");
  }
  return count;
}

PointCloudXYZ readPointCloudXYZ(std::istream & stream)
{
  PointCloudXYZ cloud;
  const auto count = readCount(stream, "point cloud");
  cloud.resize(static_cast<std::size_t>(count));
  for (auto & point : cloud) {
    read(stream, point.x);
    read(stream, point.y);
    read(stream, point.z);
  }
  return cloud;
}

PointCloudXYZRGBA readPointCloudRGBA(std::istream & stream)
{
  PointCloudXYZRGBA cloud;
  const auto count = readCount(stream, "colored point cloud");
  cloud.resize(static_cast<std::size_t>(count));
  for (auto & point : cloud) {
    read(stream, point.x);
    read(stream, point.y);
    read(stream, point.z);
    read(stream, point.r);
    read(stream, point.g);
    read(stream, point.b);
    read(stream, point.a);
  }
  return cloud;
}

void writeLocalMap(std::ostream & stream, const global_map::LocalMapData & map)
{
  write(stream, map.resolution);
  write(stream, static_cast<std::uint64_t>(map.cells.size()));
  for (const auto & cell : map.cells) {
    write(stream, cell.x);
    write(stream, cell.y);
    write(stream, cell.evidence);
  }
  write(stream, map.ground_resolution);
  write(stream, static_cast<std::uint64_t>(map.ground_cells.size()));
  for (const auto & cell : map.ground_cells) {
    write(stream, cell.x);
    write(stream, cell.y);
    write(stream, cell.log_odds);
    write(stream, cell.r);
    write(stream, cell.g);
    write(stream, cell.b);
  }
}

global_map::LocalMapData readLocalMap(std::istream & stream)
{
  global_map::LocalMapData map;
  read(stream, map.resolution);
  const auto cell_count = readCount(stream, "local map cell count");
  map.cells.resize(static_cast<std::size_t>(cell_count));
  for (auto & cell : map.cells) {
    read(stream, cell.x);
    read(stream, cell.y);
    read(stream, cell.evidence);
  }
  read(stream, map.ground_resolution);
  const auto ground_cell_count = readCount(stream, "local ground map cell count");
  map.ground_cells.resize(static_cast<std::size_t>(ground_cell_count));
  for (auto & cell : map.ground_cells) {
    read(stream, cell.x);
    read(stream, cell.y);
    read(stream, cell.log_odds);
    read(stream, cell.r);
    read(stream, cell.g);
    read(stream, cell.b);
  }
  return map;
}

void writeGroundObservation(std::ostream & stream, const GroundPlaneObservation & observation)
{
  for (int index = 0; index < 3; ++index) {
    write(stream, observation.normal_in_base(index));
  }
  write(stream, observation.distance_to_base);
  write(stream, static_cast<std::uint64_t>(observation.point_count));
  write(stream, static_cast<std::uint64_t>(observation.inlier_count));
  writePointCloudRGBA(stream, observation.ground_cloud);
  writePointCloudXYZ(stream, observation.pcl);
}

GroundPlaneObservation readGroundObservation(std::istream & stream)
{
  GroundPlaneObservation observation;
  for (int index = 0; index < 3; ++index) {
    read(stream, observation.normal_in_base(index));
  }
  read(stream, observation.distance_to_base);
  const auto point_count = readCount(stream, "ground point count");
  const auto inlier_count = readCount(stream, "ground inlier count");
  observation.point_count = static_cast<std::size_t>(point_count);
  observation.inlier_count = static_cast<std::size_t>(inlier_count);
  observation.ground_cloud = readPointCloudRGBA(stream);
  observation.pcl = readPointCloudXYZ(stream);
  return observation;
}

void setError(std::string * error, const std::string & message)
{
  if (error) {
    *error = message;
  }
}

}  // namespace

bool StateSerializer::save(
  const std::filesystem::path & path, const SlamStateSnapshot & snapshot, std::string * error)
{
  try {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
      throw std::runtime_error("cannot open state file for writing");
    }
    stream.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write(stream, snapshot.format_version);
    writeString(stream, snapshot.state_id);
    write(stream, snapshot.next_keyframe_key);
    write(stream, snapshot.optimizer_latest_key);
    write(stream, snapshot.optimizer_latest_timestamp);
    write(stream, snapshot.optimizer_initialized);
    writePose(stream, snapshot.initial_pose);
    writePose(stream, snapshot.latest_pose);
    writePose(stream, snapshot.map_to_odom);
    write(stream, static_cast<std::uint64_t>(snapshot.keyframes.size()));
    for (const auto & keyframe : snapshot.keyframes) {
      write(stream, keyframe.revision);
      write(stream, keyframe.key);
      write(stream, keyframe.timestamp);
      writePose(stream, keyframe.pose);
      writePose(stream, keyframe.odom_pose);
      write(stream, keyframe.covariance.has_value());
      if (keyframe.covariance) {
        writeMatrix(stream, *keyframe.covariance);
      }
      write(stream, static_cast<std::uint64_t>(keyframe.scan->points2D().size()));
      for (const auto & point : keyframe.scan->points2D()) {
        write(stream, point.x);
        write(stream, point.y);
      }
      write(stream, keyframe.ground_observation.has_value());
      if (keyframe.ground_observation) {
        writeGroundObservation(stream, *keyframe.ground_observation);
      }
      writeLocalMap(stream, *keyframe.local_map);
    }
    write(stream, static_cast<std::uint64_t>(snapshot.factors.size()));
    for (const auto & factor : snapshot.factors) {
      write(stream, factor.type);
      write(stream, factor.from_key);
      write(stream, factor.to_key);
      writePose(stream, factor.relative_pose);
      writeMatrix(stream, factor.covariance);
      for (int index = 0; index < 3; ++index) {
        write(stream, factor.observed_normal_in_base(index));
      }
      write(stream, factor.observed_distance_to_base);
      for (int index = 0; index < 3; ++index) {
        write(stream, factor.reference_normal_in_map(index));
      }
      write(stream, factor.reference_plane_offset);
      write(stream, factor.normal_sigma);
      write(stream, factor.distance_sigma);
    }
    write(stream, static_cast<std::uint64_t>(snapshot.loop_closures.size()));
    for (const auto & closure : snapshot.loop_closures) {
      write(stream, closure.first);
      write(stream, closure.second);
    }
    return true;
  } catch (const std::exception & exception) {
    setError(error, exception.what());
    return false;
  }
}

bool StateSerializer::load(
  const std::filesystem::path & path, SlamStateSnapshot & snapshot, std::string * error)
{
  // Deserialization is intentionally staged in a temporary object so a failed read cannot
  // partially mutate the caller's active snapshot.
  try {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
      throw std::runtime_error("cannot open state file for reading");
    }
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != kMagic) {
      throw std::runtime_error("invalid state file magic");
    }
    SlamStateSnapshot loaded;
    read(stream, loaded.format_version);
    if (loaded.format_version != SlamStateSnapshot::kFormatVersion) {
      throw std::runtime_error("unsupported state file version");
    }
    loaded.state_id = readString(stream);
    read(stream, loaded.next_keyframe_key);
    read(stream, loaded.optimizer_latest_key);
    read(stream, loaded.optimizer_latest_timestamp);
    read(stream, loaded.optimizer_initialized);
    loaded.initial_pose = readPose(stream);
    loaded.latest_pose = readPose(stream);
    loaded.map_to_odom = readPose(stream);
    const auto keyframe_count = readCount(stream, "keyframe count");
    loaded.keyframes.reserve(static_cast<std::size_t>(keyframe_count));
    for (std::uint64_t index = 0; index < keyframe_count; ++index) {
      KeyFrame keyframe;
      bool has_covariance = false;
      bool has_ground_observation = false;
      read(stream, keyframe.revision);
      read(stream, keyframe.key);
      read(stream, keyframe.timestamp);
      keyframe.pose = readPose(stream);
      keyframe.odom_pose = readPose(stream);
      read(stream, has_covariance);
      if (has_covariance) {
        gtsam::Matrix66 covariance;
        readMatrix(stream, covariance);
        keyframe.covariance = covariance;
      }
      const auto scan_count = readCount(stream, "scan point count");
      std::vector<Point2D> points(static_cast<std::size_t>(scan_count));
      for (auto & point : points) {
        read(stream, point.x);
        read(stream, point.y);
      }
      keyframe.scan = std::make_shared<const LaserScan>(std::move(points));
      read(stream, has_ground_observation);
      if (has_ground_observation) {
        keyframe.ground_observation = readGroundObservation(stream);
      }
      keyframe.local_map = std::make_shared<const global_map::LocalMapData>(readLocalMap(stream));
      loaded.keyframes.push_back(std::move(keyframe));
    }
    const auto factor_count = readCount(stream, "factor count");
    loaded.factors.resize(static_cast<std::size_t>(factor_count));
    for (auto & factor : loaded.factors) {
      read(stream, factor.type);
      read(stream, factor.from_key);
      read(stream, factor.to_key);
      factor.relative_pose = readPose(stream);
      readMatrix(stream, factor.covariance);
      for (int index = 0; index < 3; ++index) {
        read(stream, factor.observed_normal_in_base(index));
      }
      read(stream, factor.observed_distance_to_base);
      for (int index = 0; index < 3; ++index) {
        read(stream, factor.reference_normal_in_map(index));
      }
      read(stream, factor.reference_plane_offset);
      read(stream, factor.normal_sigma);
      read(stream, factor.distance_sigma);
    }
    const auto closure_count = readCount(stream, "loop closure count");
    loaded.loop_closures.resize(static_cast<std::size_t>(closure_count));
    for (auto & closure : loaded.loop_closures) {
      read(stream, closure.first);
      read(stream, closure.second);
    }
    snapshot = std::move(loaded);
    return true;
  } catch (const std::exception & exception) {
    setError(error, exception.what());
    return false;
  }
}

}  // namespace glidar_slam::core
