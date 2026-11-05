#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glidar_slam/logger/logger.hpp"
#include "glidar_slam/state_serializer.hpp"
#include "glidar_slam/system.hpp"
#include "gtest/gtest.h"
#include "gtsam/geometry/Pose3.h"
#include "opencv2/core.hpp"
#include "test_helpers.hpp"

namespace glidar_slam {
namespace {

using test::TemporaryDirectory;

// The robot drives along the x axis of a walled room. Every step creates a keyframe, too few for a
// loop closure, which keeps the runs deterministic.
constexpr double STEP_LENGTH = 0.6;
constexpr double ROOM_MIN_X = -4.0;
constexpr double ROOM_MAX_X = 8.0;
constexpr double ROOM_HALF_WIDTH = 4.0;
constexpr int RAY_COUNT = 360;
constexpr int STEPS_BEFORE_SAVING = 6;

// The camera looks straight down from this height above the ground.
constexpr float CAMERA_HEIGHT_ABOVE_BASE = 0.9F;
constexpr float CAMERA_HEIGHT_ABOVE_GROUND = 1.0F;

// Factors of a keyframe with a scan match and a ground observation: odometry, scan match, ground
// plane.
constexpr std::size_t FACTORS_PER_KEYFRAME = 3;

std::shared_ptr<const LaserScan> makeRoomScan(double x)
{
  std::vector<Point2D> points;
  points.reserve(RAY_COUNT);
  for (int ray = 0; ray < RAY_COUNT; ++ray) {
    const double angle = 2.0 * M_PI * (ray + 0.5) / RAY_COUNT;
    const double direction_x = std::cos(angle);
    const double direction_y = std::sin(angle);
    const double range_x = ((direction_x > 0.0 ? ROOM_MAX_X : ROOM_MIN_X) - x) / direction_x;
    const double range_y = (direction_y > 0.0 ? ROOM_HALF_WIDTH : -ROOM_HALF_WIDTH) / direction_y;
    const double range = std::min(range_x, range_y);
    points.push_back({range * direction_x, range * direction_y});
  }
  return std::make_shared<const LaserScan>(std::move(points));
}

SensorData makeSensorData(double x)
{
  const cv::Size image_size(160, 120);
  const cv::Mat image(image_size, CV_8UC3, cv::Scalar(128, 128, 128));
  const cv::Mat depth(image_size, CV_32FC1, cv::Scalar(CAMERA_HEIGHT_ABOVE_GROUND));

  Eigen::Affine3f base_from_camera = Eigen::Affine3f::Identity();
  base_from_camera.translation() = Eigen::Vector3f(0.0F, 0.0F, CAMERA_HEIGHT_ABOVE_BASE);
  base_from_camera.linear() =
    Eigen::AngleAxisf(static_cast<float>(M_PI), Eigen::Vector3f::UnitX()).toRotationMatrix();

  const CameraModel camera(image_size, 100.0F, 100.0F, 80.0F, 60.0F, base_from_camera);
  return SensorData(image, depth, makeRoomScan(x), camera);
}

class SlamSystemState : public ::testing::Test
{
protected:
  SlamSystemState()
  {
    log::set_log_level(log::LogLevel::Error);

    parameters_->csm_search_stages = {
      {0.1, 0.05, 0.02, 0.2, 0.2, 0.06},
      {0.05, 0.01, 0.005, 0.05, 0.05, 0.02},
    };
    // The score of a good match depends on the scan and the smearing. A failed match scores zero.
    parameters_->tracking_minimum_score = 0.2;
  }

  std::unique_ptr<SlamSystem> makeSystem() const
  {
    return std::make_unique<SlamSystem>(
      parameters_, std::make_unique<CorrelativeScanMatcher>(parameters_),
      std::make_unique<CorrelativeScanMatcher>(parameters_),
      std::make_unique<CorrelativeScanMatcher>(parameters_));
  }

  bool driveStep(SlamSystem & system, int step) const
  {
    const double x = step * STEP_LENGTH;
    return system.process(
      static_cast<double>(step), makeSensorData(x),
      gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(x, 0.0, 0.0)), gtsam::Matrix66::Identity() * 1e-3);
  }

  std::unique_ptr<SlamSystem> makeDrivenSystem() const
  {
    auto system = makeSystem();
    for (int step = 0; step < STEPS_BEFORE_SAVING; ++step) {
      EXPECT_TRUE(driveStep(*system, step)) << "step " << step;
    }
    return system;
  }

  std::shared_ptr<Parameters> parameters_{std::make_shared<Parameters>()};
  const TemporaryDirectory directory_;
};

TEST_F(SlamSystemState, SavedStateCanBeLoaded)
{
  const auto original = makeDrivenSystem();
  const auto original_keyframes = original->getKeyFrames();
  ASSERT_EQ(original_keyframes.size(), static_cast<std::size_t>(STEPS_BEFORE_SAVING));
  ASSERT_TRUE(original_keyframes.back()->ground_observation.has_value());

  const auto path = directory_.file("state.bin");
  std::string error;
  ASSERT_TRUE(original->saveState(path, &error)) << error;

  const auto loaded = makeSystem();
  ASSERT_TRUE(loaded->loadState(path, gtsam::Pose3(), true, false, &error)) << error;

  EXPECT_EQ(loaded->getFactorCount(), original->getFactorCount());
  const auto loaded_keyframes = loaded->getKeyFrames();
  ASSERT_EQ(loaded_keyframes.size(), original_keyframes.size());
  for (std::size_t index = 0; index < loaded_keyframes.size(); ++index) {
    EXPECT_EQ(loaded_keyframes[index]->key, original_keyframes[index]->key);
    EXPECT_TRUE(loaded_keyframes[index]->pose.equals(original_keyframes[index]->pose, 1e-9))
      << "keyframe " << index;
    EXPECT_TRUE(loaded_keyframes[index]->ground_observation.has_value()) << "keyframe " << index;
  }
}

TEST_F(SlamSystemState, LoadedStateCanBeSavedAndLoadedAgain)
{
  const auto original = makeDrivenSystem();
  const auto first_path = directory_.file("first.bin");
  const auto second_path = directory_.file("second.bin");
  std::string error;
  ASSERT_TRUE(original->saveState(first_path, &error)) << error;

  const auto first_load = makeSystem();
  ASSERT_TRUE(first_load->loadState(first_path, gtsam::Pose3(), true, false, &error)) << error;
  ASSERT_TRUE(first_load->saveState(second_path, &error)) << error;

  const auto second_load = makeSystem();
  ASSERT_TRUE(second_load->loadState(second_path, gtsam::Pose3(), true, false, &error)) << error;
  EXPECT_EQ(second_load->getFactorCount(), original->getFactorCount());
  EXPECT_EQ(second_load->getKeyFrames().size(), original->getKeyFrames().size());
}

// The next keyframe is matched against the submap, so the submap has to be restored with the state.
// With an empty submap the keyframe would silently get no scan matching factor.
TEST_F(SlamSystemState, ResumedSystemContinuesLikeTheOriginal)
{
  const auto original = makeDrivenSystem();
  const auto path = directory_.file("state.bin");
  std::string error;
  ASSERT_TRUE(original->saveState(path, &error)) << error;

  const auto resumed = makeSystem();
  ASSERT_TRUE(resumed->loadState(path, gtsam::Pose3(), true, false, &error)) << error;

  const std::size_t factors_before_original = original->getFactorCount();
  const std::size_t factors_before_resumed = resumed->getFactorCount();

  ASSERT_TRUE(driveStep(*original, STEPS_BEFORE_SAVING));
  ASSERT_TRUE(driveStep(*resumed, STEPS_BEFORE_SAVING));

  EXPECT_EQ(original->getFactorCount() - factors_before_original, FACTORS_PER_KEYFRAME);
  EXPECT_EQ(resumed->getFactorCount() - factors_before_resumed, FACTORS_PER_KEYFRAME);
  EXPECT_EQ(resumed->getKeyFrames().size(), original->getKeyFrames().size());
  EXPECT_TRUE(resumed->getLatestPoseAndCovariance().pose.equals(
    original->getLatestPoseAndCovariance().pose, 0.02));
}

// A second prior factor enters the graph here, which also has to survive saving and loading.
TEST_F(SlamSystemState, TrackingResetCanBeSavedAndLoaded)
{
  const auto original = makeDrivenSystem();
  const auto first_path = directory_.file("first.bin");
  const auto second_path = directory_.file("second.bin");
  std::string error;
  ASSERT_TRUE(original->saveState(first_path, &error)) << error;

  const auto reset = makeSystem();
  const gtsam::Pose3 reset_pose(gtsam::Rot3(), gtsam::Point3(1.0, 0.5, 0.0));
  ASSERT_TRUE(reset->loadState(first_path, reset_pose, false, false, &error)) << error;
  ASSERT_TRUE(driveStep(*reset, STEPS_BEFORE_SAVING));
  ASSERT_EQ(reset->getKeyFrames().size(), static_cast<std::size_t>(STEPS_BEFORE_SAVING + 1));
  ASSERT_TRUE(reset->saveState(second_path, &error)) << error;

  const auto loaded = makeSystem();
  ASSERT_TRUE(loaded->loadState(second_path, gtsam::Pose3(), true, false, &error)) << error;
  EXPECT_EQ(loaded->getFactorCount(), reset->getFactorCount());
  EXPECT_EQ(loaded->getKeyFrames().size(), reset->getKeyFrames().size());
}

TEST_F(SlamSystemState, LocalizationWorksOnALoadedState)
{
  const auto original = makeDrivenSystem();
  const auto path = directory_.file("state.bin");
  std::string error;
  ASSERT_TRUE(original->saveState(path, &error)) << error;

  const auto localization = makeSystem();
  ASSERT_TRUE(localization->loadState(path, gtsam::Pose3(), true, true, &error)) << error;
  EXPECT_TRUE(localization->isLocalizationMode());

  const auto keyframes_before = localization->getKeyFrames().size();
  EXPECT_TRUE(driveStep(*localization, STEPS_BEFORE_SAVING - 1));
  EXPECT_EQ(localization->getKeyFrames().size(), keyframes_before);
}

TEST_F(SlamSystemState, EmptyStateCanBeSavedAndLoaded)
{
  const auto empty = makeSystem();
  const auto path = directory_.file("state.bin");
  std::string error;
  ASSERT_TRUE(empty->saveState(path, &error)) << error;

  const auto loaded = makeSystem();
  ASSERT_TRUE(loaded->loadState(path, gtsam::Pose3(), true, false, &error)) << error;
  EXPECT_TRUE(loaded->getKeyFrames().empty());

  EXPECT_TRUE(driveStep(*loaded, 0));
  EXPECT_EQ(loaded->getKeyFrames().size(), 1U);
}

TEST_F(SlamSystemState, InvalidFileIsReportedAndKeepsTheRunningState)
{
  const auto system = makeDrivenSystem();
  const auto path = directory_.file("invalid.bin");
  test::writeBytes(path, "this is not a state file");

  std::string error;
  EXPECT_FALSE(system->loadState(path, gtsam::Pose3(), true, false, &error));
  EXPECT_FALSE(error.empty());

  EXPECT_EQ(system->getKeyFrames().size(), static_cast<std::size_t>(STEPS_BEFORE_SAVING));
  EXPECT_TRUE(driveStep(*system, STEPS_BEFORE_SAVING));
}

}  // namespace
}  // namespace glidar_slam
