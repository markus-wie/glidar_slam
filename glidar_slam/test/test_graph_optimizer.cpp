#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "glidar_slam/graph_optimizer.hpp"
#include "glidar_slam/state_serializer.hpp"
#include "gtest/gtest.h"
#include "gtsam/geometry/Pose3.h"
#include "test_helpers.hpp"

namespace glidar_slam {
namespace {

using test::TemporaryDirectory;

constexpr double BASE_HEIGHT = 0.3;

gtsam::Pose3 poseAt(double x)
{
  return gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(x, 0.0, BASE_HEIGHT));
}

gtsam::Pose3 forward(double distance)
{
  return gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(distance, 0.0, 0.0));
}

// Tangent order of a Pose3: roll, pitch, yaw, x, y, z.
gtsam::Matrix66 relativeCovariance()
{
  gtsam::Matrix66 covariance = gtsam::Matrix66::Zero();
  covariance.diagonal() << 9e-4, 9e-4, 1e-2, 1e-2, 1e-2, 9e-4;
  return covariance;
}

void addGroundFactor(GraphOptimizer & optimizer, std::uint64_t key)
{
  optimizer.addGroundPlaneFactor(
    key, gtsam::Vector3::UnitZ(), gtsam::Vector3(0.0, 0.0, -BASE_HEIGHT), gtsam::Vector3::UnitZ(),
    0.0, 0.05, 0.03);
}

// All factor types the system creates, with the prior of a first keyframe with a ground observation
// and a robust loop closure.
void buildGraph(GraphOptimizer & optimizer)
{
  optimizer.initialize(poseAt(0.0), 1000, /*constrain_vertical_dofs=*/false);
  addGroundFactor(optimizer, 0);
  optimizer.optimize();

  optimizer.addRelativeFactor(0, 1, forward(1.0), relativeCovariance());
  addGroundFactor(optimizer, 1);
  optimizer.optimize();

  optimizer.addRelativeFactor(1, 2, forward(1.0), relativeCovariance());
  optimizer.optimize();

  optimizer.addRelativeFactor(0, 2, forward(2.0), relativeCovariance(), /*robust=*/true);
  optimizer.optimize();
}

SlamStateSnapshot makeSnapshot(const GraphOptimizer & optimizer)
{
  SlamStateSnapshot snapshot;
  snapshot.optimizer_initialized = optimizer.isInitialized();
  snapshot.optimizer_latest_key = optimizer.getLatestKey();
  snapshot.optimizer_latest_timestamp = optimizer.getLatestTimestamp();
  snapshot.factors = optimizer.getSerializedFactors();
  for (std::uint64_t key = 0; key <= optimizer.getLatestKey(); ++key) {
    snapshot.keyframes.emplace_back(
      key, static_cast<double>(key), optimizer.getCurrentEstimates().at<gtsam::Pose3>(key),
      gtsam::Pose3(), std::make_shared<const LaserScan>(std::vector<Point2D>{}));
  }
  return snapshot;
}

// Takes the same path as loading a state in the system: through the binary file into the optimizer.
void saveAndRestore(const GraphOptimizer & source, GraphOptimizer & target)
{
  const TemporaryDirectory directory;
  const auto path = directory.file("state.bin");
  std::string error;
  ASSERT_TRUE(StateSerializer::save(path, makeSnapshot(source), &error)) << error;

  SlamStateSnapshot loaded;
  ASSERT_TRUE(StateSerializer::load(path, loaded, &error)) << error;
  ASSERT_TRUE(target.restore(loaded));
}

// Compared entry by entry, because the entries of a prior differ by orders of magnitude and a
// matrix norm would only see the largest ones.
void expectSameVariances(const gtsam::Matrix66 & actual, const gtsam::Matrix66 & expected)
{
  for (int index = 0; index < 6; ++index) {
    ASSERT_GT(expected(index, index), 0.0) << "dimension " << index;
    EXPECT_NEAR(actual(index, index) / expected(index, index), 1.0, 1e-6)
      << "dimension " << index << ": " << actual(index, index) << " vs " << expected(index, index);
  }
}

class GraphOptimizerTest : public ::testing::Test
{
protected:
  std::shared_ptr<Parameters> parameters_{std::make_shared<Parameters>()};
  GraphOptimizer optimizer_{parameters_};
  GraphOptimizer restored_{parameters_};
};

TEST_F(GraphOptimizerTest, RestoreKeepsAllFactors)
{
  buildGraph(optimizer_);
  saveAndRestore(optimizer_, restored_);

  const auto expected = optimizer_.getSerializedFactors();
  const auto actual = restored_.getSerializedFactors();
  ASSERT_EQ(actual.size(), expected.size());

  for (std::size_t index = 0; index < expected.size(); ++index) {
    SCOPED_TRACE("factor " + std::to_string(index));
    EXPECT_EQ(actual[index].type, expected[index].type);
    EXPECT_EQ(actual[index].from_key, expected[index].from_key);
    EXPECT_EQ(actual[index].to_key, expected[index].to_key);
    EXPECT_EQ(actual[index].robust, expected[index].robust);
    EXPECT_TRUE(actual[index].relative_pose.equals(expected[index].relative_pose, 1e-9));
    EXPECT_DOUBLE_EQ(actual[index].normal_sigma, expected[index].normal_sigma);
    EXPECT_DOUBLE_EQ(actual[index].distance_sigma, expected[index].distance_sigma);
    if (expected[index].type != SerializedFactorType::GroundPlane) {
      expectSameVariances(actual[index].covariance, expected[index].covariance);
    }
  }
}

TEST_F(GraphOptimizerTest, RestoreKeepsTheEstimateAndCanContinue)
{
  buildGraph(optimizer_);
  saveAndRestore(optimizer_, restored_);

  EXPECT_EQ(restored_.getLatestKey(), optimizer_.getLatestKey());
  for (std::uint64_t key = 0; key <= optimizer_.getLatestKey(); ++key) {
    EXPECT_TRUE(restored_.getCurrentEstimates().at<gtsam::Pose3>(key).equals(
      optimizer_.getCurrentEstimates().at<gtsam::Pose3>(key), 1e-3))
      << "key " << key;
  }

  restored_.addRelativeFactor(2, 3, forward(1.0), relativeCovariance());
  restored_.optimize();
  EXPECT_EQ(restored_.getLatestKey(), 3U);
  EXPECT_NEAR(restored_.getLatestPose().x(), 3.0, 1e-3);
}

// A failed cast of the robust noise model used to save a zero covariance, which pins all dimensions
// of the factor after loading.
TEST_F(GraphOptimizerTest, RobustFactorsAreSavedWithTheirCovariance)
{
  buildGraph(optimizer_);

  const auto factors = optimizer_.getSerializedFactors();
  const auto loop_closure =
    std::find_if(factors.begin(), factors.end(), [](const SerializedFactor & factor) {
      return factor.robust;
    });
  ASSERT_NE(loop_closure, factors.end());
  expectSameVariances(loop_closure->covariance, relativeCovariance());
}

// The query keyframe of a loop closure is older than the latest keyframe if newer ones were added
// while the asynchronous search was running.
TEST_F(GraphOptimizerTest, LoopClosureDoesNotMoveTheLatestKeyBackwards)
{
  buildGraph(optimizer_);
  ASSERT_EQ(optimizer_.getLatestKey(), 2U);

  optimizer_.addRelativeFactor(0, 1, forward(1.0), relativeCovariance(), /*robust=*/true);
  optimizer_.optimize();

  EXPECT_EQ(optimizer_.getLatestKey(), 2U);
  EXPECT_NEAR(optimizer_.getLatestPose().x(), 2.0, 1e-3);
}

}  // namespace
}  // namespace glidar_slam
