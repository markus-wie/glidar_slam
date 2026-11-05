#include <memory>
#include <string>
#include <vector>

#include "glidar_slam/state_serializer.hpp"
#include "gtest/gtest.h"
#include "gtsam/geometry/Pose3.h"
#include "test_helpers.hpp"

namespace glidar_slam {
namespace {

using test::readBytes;
using test::TemporaryDirectory;
using test::writeBytes;

gtsam::Matrix66 makeCovariance()
{
  gtsam::Matrix66 covariance = gtsam::Matrix66::Identity() * 0.01;
  covariance(0, 1) = 0.001;
  covariance(1, 0) = 0.001;
  return covariance;
}

GroundPlaneObservation makeGroundObservation()
{
  GroundPlaneObservation observation;
  observation.normal_in_base = Eigen::Vector3f(0.01F, -0.02F, 0.99F);
  observation.distance_to_base = 0.31F;
  observation.centroid_in_base = Eigen::Vector3f(0.0F, 0.0F, -0.3F);
  observation.point_count = 120;
  observation.inlier_count = 100;

  pcl::PointXYZRGBA colored_point;
  colored_point.x = 1.0F;
  colored_point.y = -2.0F;
  colored_point.z = 0.25F;
  colored_point.r = 10;
  colored_point.g = 20;
  colored_point.b = 30;
  colored_point.a = 255;
  observation.ground_cloud->push_back(colored_point);
  observation.ground_cloud_sparse->push_back(pcl::PointXYZ(0.5F, 0.25F, -0.125F));
  return observation;
}

// A state that uses every part of the format: both variants of all optional keyframe members and
// all factor types.
SlamStateSnapshot makeSnapshot()
{
  const gtsam::Pose3 pose_a(gtsam::Rot3::Ypr(0.3, 0.02, -0.01), gtsam::Point3(1.0, 2.0, 0.3));
  const gtsam::Pose3 pose_b(gtsam::Rot3::Ypr(-1.2, 0.0, 0.0), gtsam::Point3(-3.0, 0.5, 0.0));

  SlamStateSnapshot snapshot;
  snapshot.state_id = "test_state";
  snapshot.next_keyframe_key = 2;
  snapshot.optimizer_latest_key = 1;
  snapshot.optimizer_latest_timestamp = 13000000;
  snapshot.optimizer_initialized = true;
  snapshot.initial_pose = pose_a;
  snapshot.latest_pose = pose_b;
  snapshot.map_to_odom = pose_a.compose(pose_b);

  KeyFrame complete_keyframe(
    0, 12.5, pose_a, pose_b,
    std::make_shared<const LaserScan>(std::vector<Point2D>{{1.0, 2.0}, {-3.5, 0.25}}),
    makeCovariance(), makeGroundObservation(), 3);
  complete_keyframe.local_occupancy = std::make_shared<const mapping::LocalOccupancyMap>(
    mapping::LocalOccupancyMap{0.05, {{1, -2, 0.5F}, {3, 4, -0.25F}}});
  complete_keyframe.local_ground = std::make_shared<const mapping::LocalGroundMap>(
    mapping::LocalGroundMap{0.02, {{5, -6, 0.75F, 10, 20, 30}}});
  snapshot.keyframes.push_back(complete_keyframe);

  KeyFrame minimal_keyframe(
    1, 13.0, pose_b, pose_a, std::make_shared<const LaserScan>(std::vector<Point2D>{{0.5, 0.5}}));
  minimal_keyframe.local_occupancy = nullptr;
  minimal_keyframe.local_ground = nullptr;
  snapshot.keyframes.push_back(minimal_keyframe);

  SerializedFactor prior;
  prior.type = SerializedFactorType::PriorPose;
  prior.to_key = 0;
  prior.relative_pose = pose_a;
  prior.covariance = makeCovariance();
  snapshot.factors.push_back(prior);

  SerializedFactor relative;
  relative.type = SerializedFactorType::RelativePose;
  relative.from_key = 0;
  relative.to_key = 1;
  relative.relative_pose = pose_a.between(pose_b);
  relative.covariance = makeCovariance();
  snapshot.factors.push_back(relative);

  SerializedFactor loop_closure = relative;
  loop_closure.robust = true;
  snapshot.factors.push_back(loop_closure);

  SerializedFactor ground;
  ground.type = SerializedFactorType::GroundPlane;
  ground.to_key = 1;
  ground.observed_normal_in_base = gtsam::Vector3(0.01, -0.02, 0.99);
  ground.observed_centroid_in_base = gtsam::Vector3(0.0, 0.0, -0.3);
  ground.reference_normal_in_map = gtsam::Vector3::UnitZ();
  ground.reference_plane_offset = 0.125;
  ground.normal_sigma = 0.05;
  ground.distance_sigma = 0.03;
  snapshot.factors.push_back(ground);

  snapshot.loop_closures.emplace_back(0, 1);
  return snapshot;
}

// Saving a loaded state has to reproduce the file. This covers the whole format without a check per
// field, so a new field that is saved and loaded consistently needs no change here.
TEST(StateSerializer, SaveLoadSaveIsByteIdentical)
{
  const TemporaryDirectory directory;
  const auto first_path = directory.file("first.bin");
  const auto second_path = directory.file("second.bin");
  std::string error;

  ASSERT_TRUE(StateSerializer::save(first_path, makeSnapshot(), &error)) << error;

  SlamStateSnapshot loaded;
  ASSERT_TRUE(StateSerializer::load(first_path, loaded, &error)) << error;
  ASSERT_EQ(loaded.keyframes.size(), 2U);
  ASSERT_EQ(loaded.factors.size(), 4U);

  ASSERT_TRUE(StateSerializer::save(second_path, loaded, &error)) << error;

  const std::string first_bytes = readBytes(first_path);
  const std::string second_bytes = readBytes(second_path);
  EXPECT_FALSE(first_bytes.empty());
  EXPECT_TRUE(first_bytes == second_bytes)
    << "The state changed by saving and loading it (" << first_bytes.size() << " and "
    << second_bytes.size() << " bytes)";
}

TEST(StateSerializer, TruncatedFilesAreRejectedWithoutTouchingTheTarget)
{
  const TemporaryDirectory directory;
  const auto complete_path = directory.file("complete.bin");
  const auto truncated_path = directory.file("truncated.bin");
  ASSERT_TRUE(StateSerializer::save(complete_path, makeSnapshot()));
  const std::string bytes = readBytes(complete_path);

  SlamStateSnapshot target;
  target.state_id = "untouched";

  for (std::size_t size = 0; size < bytes.size(); ++size) {
    writeBytes(truncated_path, bytes.substr(0, size));

    std::string error;
    ASSERT_FALSE(StateSerializer::load(truncated_path, target, &error)) << "size " << size;
    ASSERT_FALSE(error.empty()) << "size " << size;
    ASSERT_EQ(target.state_id, "untouched") << "size " << size;
  }
}

TEST(StateSerializer, ForeignFilesAreRejected)
{
  const TemporaryDirectory directory;
  const auto path = directory.file("state.bin");
  ASSERT_TRUE(StateSerializer::save(path, makeSnapshot()));
  const std::string bytes = readBytes(path);

  // The file starts with the 8 byte magic, followed by the format version.
  for (const std::size_t corrupted_byte : {std::size_t{0}, std::size_t{8}}) {
    std::string corrupted = bytes;
    corrupted[corrupted_byte] = static_cast<char>(corrupted[corrupted_byte] ^ 0xFF);
    writeBytes(path, corrupted);

    SlamStateSnapshot target;
    std::string error;
    EXPECT_FALSE(StateSerializer::load(path, target, &error)) << "byte " << corrupted_byte;
    EXPECT_FALSE(error.empty()) << "byte " << corrupted_byte;
  }
}

TEST(StateSerializer, MissingPathsAreReported)
{
  const TemporaryDirectory directory;
  const auto path = directory.file("missing_directory/state.bin");
  std::string error;

  SlamStateSnapshot target;
  EXPECT_FALSE(StateSerializer::load(path, target, &error));
  EXPECT_FALSE(error.empty());

  error.clear();
  EXPECT_FALSE(StateSerializer::save(path, makeSnapshot(), &error));
  EXPECT_FALSE(error.empty());
}

// The last block of a state file only reaches the disk when the stream is flushed. A full disk at
// that point must not be reported as a successful save.
TEST(StateSerializer, FailedWritesAreReported)
{
  const std::filesystem::path full_device{"/dev/full"};
  if (!std::filesystem::exists(full_device)) {
    GTEST_SKIP() << "This system has no /dev/full";
  }

  std::string error;
  EXPECT_FALSE(StateSerializer::save(full_device, makeSnapshot(), &error));
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace glidar_slam
