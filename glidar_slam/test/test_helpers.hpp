#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>

namespace glidar_slam::test {

// A directory for the files of one test, it is removed again at the end of the test.
class TemporaryDirectory
{
public:
  TemporaryDirectory()
  {
    std::string name =
      (std::filesystem::temp_directory_path() / "glidar_slam_test_XXXXXX").string();
    if (mkdtemp(name.data()) == nullptr) {
      throw std::runtime_error("Cannot create a temporary directory");
    }
    path_ = name;
  }

  ~TemporaryDirectory()
  {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory & operator=(const TemporaryDirectory &) = delete;

  std::filesystem::path file(const std::string & name) const
  {
    return path_ / name;
  }

private:
  std::filesystem::path path_;
};

inline std::string readBytes(const std::filesystem::path & path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Cannot open " + path.string());
  }
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

inline void writeBytes(const std::filesystem::path & path, const std::string & bytes)
{
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace glidar_slam::test
