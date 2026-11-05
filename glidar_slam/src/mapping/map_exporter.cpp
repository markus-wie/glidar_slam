#include "glidar_slam/mapping/map_exporter.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

namespace glidar_slam::mapping {

namespace {

// agreed upon value for unknown occupancy for better visual differentiation
constexpr int UNKNOWN_OCCUPANCY = 205;

void saveOccupancy(const OccupancyGrid & grid, const std::string & base_filepath)
{
  const auto & info = grid.getInfo();
  const auto & data = grid.getData();
  if (data.size() != static_cast<std::size_t>(info.width) * info.height) {
    throw std::runtime_error("Occupancy map data size does not match its metadata");
  }

  const std::string pgm_path = base_filepath + ".pgm";
  const std::string yaml_path = base_filepath + ".yaml";
  std::ofstream pgm(pgm_path, std::ios::binary);
  if (!pgm) {
    throw std::runtime_error("Cannot open " + pgm_path);
  }
  pgm << "P5\n" << info.width << " " << info.height << "\n255\n";
  for (std::size_t y = 0; y < info.height; ++y) {
    for (std::size_t x = 0; x < info.width; ++x) {
      const std::size_t map_y = info.height - 1 - y;
      const std::size_t index = map_y * info.width + x;

      const auto value = data[index];
      const std::uint8_t pixel =
        value < 0
          ? UNKNOWN_OCCUPANCY
          : static_cast<std::uint8_t>(255U - (static_cast<unsigned int>(value) * 255U / 100U));
      pgm.put(static_cast<char>(pixel));
    }
  }
  pgm.flush();
  if (!pgm) {
    throw std::runtime_error("Cannot write " + pgm_path);
  }

  std::ofstream yaml(yaml_path);
  if (!yaml) {
    throw std::runtime_error("Cannot open " + yaml_path);
  }
  yaml << "image: " << pgm_path.substr(pgm_path.find_last_of('/') + 1) << "\n"
       << "mode: trinary\nresolution: " << info.resolution << "\norigin: [" << info.origin_x << ", "
       << info.origin_y
       << ", 0.0]\nnegate: 0\noccupied_thresh: 0.6\nfree_thresh: 0.6\nthreshold: 128\n";
  yaml.flush();
  if (!yaml) {
    throw std::runtime_error("Cannot write " + yaml_path);
  }
}

void saveTexture(const GroundTextureGrid & grid, const std::string & base_filepath)
{
  const auto & info = grid.getInfo();
  const auto & rgb = grid.getRgbData();
  if (rgb.size() != static_cast<std::size_t>(info.width) * info.height * 3U) {
    throw std::runtime_error("Texture map data size does not match its metadata");
  }

  cv::Mat image(info.height, info.width, CV_8UC4);
  for (uint32_t r = 0; r < info.height; ++r) {
    for (uint32_t c = 0; c < info.width; ++c) {
      const std::size_t map_y = info.height - 1 - r;
      const std::size_t index = map_y * info.width + c;

      auto & pixel = image.at<cv::Vec4b>(r, c);
      pixel[0] = rgb[index * 3U + 2U];
      pixel[1] = rgb[index * 3U + 1U];
      pixel[2] = rgb[index * 3U];
      pixel[3] = (pixel[0] || pixel[1] || pixel[2]) ? 255U : 0U;
    }
  }

  const std::string png_path = base_filepath + "_texture.png";
  if (!cv::imwrite(png_path, image)) {
    throw std::runtime_error("Cannot write " + png_path);
  }

  const std::string yaml_path = base_filepath + "_texture.yaml";
  std::ofstream yaml(yaml_path);
  if (!yaml) {
    throw std::runtime_error("Cannot open " + yaml_path);
  }
  yaml << "image: " << png_path.substr(png_path.find_last_of('/') + 1)
       << "\nmode: trinary\nresolution: " << info.resolution << "\norigin: [" << info.origin_x
       << ", " << info.origin_y << ", 0.0]\nnegate: 0\n";
  yaml.flush();
  if (!yaml) {
    throw std::runtime_error("Cannot write " + yaml_path);
  }
}

}  // namespace

bool saveGlobalMaps(
  const GlobalMapSnapshot & snapshot, const std::string & base_filepath, bool save_occupancy,
  bool save_texture, std::string * error)
{
  try {
    // Checked up front, so that a missing layer does not leave a partial export behind.
    if (save_occupancy && !snapshot.occupancy) {
      throw std::runtime_error("Occupancy map is not available, it is empty or too large");
    }
    if (save_texture && !snapshot.texture) {
      throw std::runtime_error("Texture map is not available, it is empty, disabled or too large");
    }

    if (save_occupancy) {
      saveOccupancy(*snapshot.occupancy, base_filepath);
    }
    if (save_texture) {
      saveTexture(*snapshot.texture, base_filepath);
    }
  } catch (const std::exception & exception) {
    if (error) {
      *error = exception.what();
    }
    return false;
  }
  return true;
}

}  // namespace glidar_slam::mapping
