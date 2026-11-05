#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "glidar_slam/mapping/accumulators.hpp"
#include "glidar_slam/mapping/map_exporter.hpp"
#include "gtest/gtest.h"
#include "test_helpers.hpp"

namespace glidar_slam::mapping {
namespace {

using test::readBytes;
using test::TemporaryDirectory;

constexpr int MAXIMUM_CELLS_PER_SIDE = 20000;

OccupancyGrid makeOccupancyGrid()
{
  GlobalMap::Info info;
  info.width = 2;
  info.height = 2;
  // The first row of the data is the bottom row of the map.
  return OccupancyGrid(info, {-1, 0, 100, 50});
}

GroundTextureGrid makeTextureGrid()
{
  GlobalMap::Info info;
  info.width = 1;
  info.height = 1;
  return GroundTextureGrid(info, {10, 20, 30});
}

std::vector<GlobalOccupancyCell> occupancyCellsUpTo(int x)
{
  return {{0, 0, 1.0F}, {x, 0, 1.0F}};
}

std::vector<GlobalGroundCell> groundCellsUpTo(int x)
{
  return {{0, 0, 1.0F, 10, 20, 30}, {x, 0, 1.0F, 10, 20, 30}};
}

TEST(GlobalMapLimit, MapsUpToTheLimitArePublished)
{
  const auto parameters = std::make_shared<Parameters>();

  OccupancyAccumulator occupancy(parameters);
  occupancy.apply(0, 1, occupancyCellsUpTo(MAXIMUM_CELLS_PER_SIDE - 1));
  const auto occupancy_grid = occupancy.publish();
  ASSERT_TRUE(occupancy_grid.has_value());
  EXPECT_EQ(occupancy_grid->getInfo().width, static_cast<std::uint32_t>(MAXIMUM_CELLS_PER_SIDE));

  GroundTextureAccumulator texture(parameters);
  texture.apply(0, 1, groundCellsUpTo(MAXIMUM_CELLS_PER_SIDE - 1));
  EXPECT_TRUE(texture.publish().has_value());

  GroundMarkingAccumulator marking(parameters);
  marking.apply(0, 1, groundCellsUpTo(MAXIMUM_CELLS_PER_SIDE - 1));
  EXPECT_TRUE(marking.publish().has_value());
}

TEST(GlobalMapLimit, LargerMapsAreNotPublished)
{
  const auto parameters = std::make_shared<Parameters>();

  OccupancyAccumulator occupancy(parameters);
  occupancy.apply(0, 1, occupancyCellsUpTo(MAXIMUM_CELLS_PER_SIDE));
  EXPECT_FALSE(occupancy.publish().has_value());

  GroundTextureAccumulator texture(parameters);
  texture.apply(0, 1, groundCellsUpTo(MAXIMUM_CELLS_PER_SIDE));
  EXPECT_FALSE(texture.publish().has_value());

  GroundMarkingAccumulator marking(parameters);
  marking.apply(0, 1, groundCellsUpTo(MAXIMUM_CELLS_PER_SIDE));
  EXPECT_FALSE(marking.publish().has_value());
}

TEST(MapExporter, WritesAllRequestedLayers)
{
  const TemporaryDirectory directory;
  const std::string base_path = directory.file("map").string();

  GlobalMapSnapshot snapshot;
  snapshot.occupancy = makeOccupancyGrid();
  snapshot.texture = makeTextureGrid();

  std::string error;
  ASSERT_TRUE(saveGlobalMaps(snapshot, base_path, true, true, &error)) << error;

  // The top row of the image is the last row of the map.
  const std::string expected_pgm =
    std::string("P5\n2 2\n255\n") + '\x00' + '\x80' + '\xCD' + '\xFF';
  EXPECT_EQ(readBytes(base_path + ".pgm"), expected_pgm);
  EXPECT_NE(readBytes(base_path + ".yaml").find("image: map.pgm"), std::string::npos);
  EXPECT_FALSE(readBytes(base_path + "_texture.png").empty());
  EXPECT_NE(
    readBytes(base_path + "_texture.yaml").find("image: map_texture.png"), std::string::npos);
}

// This used to dereference the empty occupancy layer, although it was not requested.
TEST(MapExporter, TextureCanBeSavedWithoutAnOccupancyLayer)
{
  const TemporaryDirectory directory;
  const std::string base_path = directory.file("map").string();

  GlobalMapSnapshot snapshot;
  snapshot.texture = makeTextureGrid();

  std::string error;
  ASSERT_TRUE(saveGlobalMaps(snapshot, base_path, false, true, &error)) << error;
  EXPECT_TRUE(std::filesystem::exists(base_path + "_texture.png"));
  EXPECT_FALSE(std::filesystem::exists(base_path + ".pgm"));
}

TEST(MapExporter, MissingOccupancyLayerIsReported)
{
  const TemporaryDirectory directory;
  const std::string base_path = directory.file("map").string();

  GlobalMapSnapshot snapshot;
  snapshot.texture = makeTextureGrid();

  std::string error;
  EXPECT_FALSE(saveGlobalMaps(snapshot, base_path, true, true, &error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(std::filesystem::exists(base_path + "_texture.png"));
}

// This is the case of a LiDAR-only run, which has an occupancy grid but no texture.
TEST(MapExporter, MissingTextureLayerLeavesNoPartialExport)
{
  const TemporaryDirectory directory;
  const std::string base_path = directory.file("map").string();

  GlobalMapSnapshot snapshot;
  snapshot.occupancy = makeOccupancyGrid();

  std::string error;
  EXPECT_FALSE(saveGlobalMaps(snapshot, base_path, true, true, &error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(std::filesystem::exists(base_path + ".pgm"));
}

// An oversized map leaves the snapshot without that layer, which then must not reach the export.
TEST(MapExporter, OversizedMapIsReported)
{
  const TemporaryDirectory directory;
  const std::string base_path = directory.file("map").string();

  OccupancyAccumulator occupancy(std::make_shared<Parameters>());
  occupancy.apply(0, 1, occupancyCellsUpTo(MAXIMUM_CELLS_PER_SIDE));

  GlobalMapSnapshot snapshot;
  snapshot.occupancy = occupancy.publish();

  std::string error;
  EXPECT_FALSE(saveGlobalMaps(snapshot, base_path, true, false, &error));
  EXPECT_FALSE(error.empty());
}

TEST(MapExporter, UnwritablePathIsReported)
{
  const TemporaryDirectory directory;
  const std::string base_path = directory.file("missing_directory/map").string();

  GlobalMapSnapshot snapshot;
  snapshot.occupancy = makeOccupancyGrid();

  std::string error;
  EXPECT_FALSE(saveGlobalMaps(snapshot, base_path, true, false, &error));
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace glidar_slam::mapping
