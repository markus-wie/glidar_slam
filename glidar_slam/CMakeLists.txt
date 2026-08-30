cmake_minimum_required(VERSION 3.27)
set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
project(glidar_slam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
find_package(PCL REQUIRED COMPONENTS common registration segmentation)
find_package(Eigen3 REQUIRED)
find_package(nanoflann REQUIRED)
find_package(TBB REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME} SHARED
  src/scan_matcher/correlative_scan_matcher.cpp
  src/scan_matcher/ground_marking_matcher.cpp
  src/mapping/accumulators.cpp
  src/mapping/global_map.cpp
  src/mapping/local_map.cpp
  src/mapping/map_builder.cpp
  src/utils.cpp
  src/submap_grid.cpp
  src/graph_optimizer.cpp
  src/ground_plane_extractor.cpp
  src/ground_plane_factor.cpp
  src/loop_closure.cpp
  src/system.cpp
  src/key_frame.cpp
  src/map_database.cpp
  src/camera_model.cpp
  src/laser_scan.cpp
  src/sensor_data.cpp
  src/state_serializer.cpp
  src/ekf.cpp
)

target_precompile_headers(${PROJECT_NAME} PUBLIC
  <Eigen/Dense>
  <Eigen/Geometry>
  <pcl/point_types.h>
  <pcl/point_cloud.h>
  <pcl/registration/icp.h>
  <pcl/common/transforms.h>
  <gtsam/geometry/Pose3.h>
  <gtsam/nonlinear/ISAM2.h>
  <gtsam/nonlinear/NonlinearFactorGraph.h>
  <gtsam/nonlinear/Values.h>
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PUBLIC ${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}
  gtsam
  ${PCL_LIBRARIES}
  ${Eigen3_LIBRARIES}
  nanoflann::nanoflann
  ${OpenCV_LIBS}
)

target_link_libraries(${PROJECT_NAME} TBB::tbb)

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV GTSAM PCL Eigen3 nanoflann TBB)

install(
  DIRECTORY include/
  DESTINATION include
)

install(TARGETS
  ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
)

ament_package()
