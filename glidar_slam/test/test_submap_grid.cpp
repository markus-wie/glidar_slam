#include <limits>
#include <vector>

#include "glidar_slam/likelihood_field.hpp"
#include "glidar_slam/submap_grid.hpp"
#include "gtest/gtest.h"

namespace glidar_slam {
namespace {

constexpr double FINE_RESOLUTION = 0.05;
constexpr double COARSE_RESOLUTION = 0.4;

// The limit of 25M cells corresponds to 250 m x 250 m at the fine and 2 km x 2 km at the coarse
// resolution.
std::vector<Point2D> pointsAt(double first, double second)
{
  return {{first, first}, {second, second}};
}

std::shared_ptr<LikelihoodField> fieldOf(const SubmapGrid & grid, double resolution)
{
  return grid.getLikelihoodField(resolution, 0.1, false, false);
}

TEST(SubmapGrid, UnusablePointsAreIgnoredWhenAddingAndRemoving)
{
  SubmapGrid grid({FINE_RESOLUTION});

  grid.add(
    {{1.0, 1.0},
     {1e7, 0.0},
     {0.0, -3000.0},
     {std::numeric_limits<double>::quiet_NaN(), 0.0},
     {0.0, std::numeric_limits<double>::infinity()}},
    0);

  ASSERT_EQ(grid.size(), 1U);
  EXPECT_GT(fieldOf(grid, FINE_RESOLUTION)->getScore(1.0, 1.0), 0.0);
  // Only the usable point determines the size of the field.
  EXPECT_LT(fieldOf(grid, FINE_RESOLUTION)->width, 20);

  grid.remove(0);
  EXPECT_EQ(grid.size(), 0U);
  EXPECT_TRUE(fieldOf(grid, FINE_RESOLUTION)->data.empty());
}

// The grid only fits at the coarse resolution. A keyframe that is added to the coarse grid only
// could never be removed correctly.
TEST(SubmapGrid, KeyframeIsAddedToAllResolutionsOrToNone)
{
  SubmapGrid grid({FINE_RESOLUTION, COARSE_RESOLUTION});

  grid.add(pointsAt(0.0, 500.0), 0);

  EXPECT_EQ(grid.size(), 0U);
  EXPECT_TRUE(fieldOf(grid, FINE_RESOLUTION)->data.empty());
  EXPECT_TRUE(fieldOf(grid, COARSE_RESOLUTION)->data.empty());

  // The rejected keyframe is unknown to the grid, and the grid stays usable.
  grid.remove(0);
  grid.add(pointsAt(0.0, 1.0), 1);
  EXPECT_EQ(grid.size(), 1U);
  EXPECT_GT(fieldOf(grid, FINE_RESOLUTION)->getScore(1.0, 1.0), 0.0);
  EXPECT_GT(fieldOf(grid, COARSE_RESOLUTION)->getScore(1.0, 1.0), 0.0);
}

TEST(SubmapGrid, KeyframeIsRefusedIfTheCellsWithHitsExceedTheLimit)
{
  SubmapGrid grid({FINE_RESOLUTION});

  grid.add(pointsAt(0.0, 100.0), 0);
  grid.add(pointsAt(1000.0, 1100.0), 1);

  EXPECT_EQ(grid.size(), 1U);
  EXPECT_GT(fieldOf(grid, FINE_RESOLUTION)->getScore(100.0, 100.0), 0.0);

  grid.remove(1);
  EXPECT_EQ(grid.size(), 1U);
  grid.remove(0);
  EXPECT_EQ(grid.size(), 0U);
}

// The grid storage only ever grows. Driving far enough would hit the limit although the window of
// active keyframes easily fits, unless the area without hits is dropped.
TEST(SubmapGrid, AreaWithoutHitsIsDroppedAtTheLimit)
{
  SubmapGrid grid({FINE_RESOLUTION});

  grid.add(pointsAt(0.0, 100.0), 0);
  grid.remove(0);
  grid.add(pointsAt(1000.0, 1100.0), 1);

  ASSERT_EQ(grid.size(), 1U);
  EXPECT_GT(fieldOf(grid, FINE_RESOLUTION)->getScore(1100.0, 1100.0), 0.0);

  grid.remove(1);
  EXPECT_EQ(grid.size(), 0U);
}

// All hits have to survive when the storage is reallocated and moved.
TEST(SubmapGrid, HitsSurviveGridGrowth)
{
  SubmapGrid grid({FINE_RESOLUTION});

  grid.add(pointsAt(1.0, 2.0), 0);
  grid.add(pointsAt(-30.0, 40.0), 1);

  ASSERT_EQ(grid.size(), 2U);
  for (const double coordinate : {1.0, 2.0, -30.0, 40.0}) {
    EXPECT_GT(fieldOf(grid, FINE_RESOLUTION)->getScore(coordinate, coordinate), 0.0)
      << "coordinate " << coordinate;
  }

  grid.removeOldestKeyframe();
  EXPECT_EQ(grid.size(), 1U);
  EXPECT_LE(fieldOf(grid, FINE_RESOLUTION)->getScore(1.0, 1.0), 0.0);
  EXPECT_GT(fieldOf(grid, FINE_RESOLUTION)->getScore(40.0, 40.0), 0.0);
}

}  // namespace
}  // namespace glidar_slam
