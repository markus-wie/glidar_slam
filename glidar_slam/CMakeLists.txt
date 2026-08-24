cmake_minimum_required(VERSION 3.27)
set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
project(glidar_slam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(GTSAM REQUIRED)
find_package(PCL REQUIRED COMPONENTS common registration segmentation)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nanoflann REQUIRED)
find_package(TBB REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}_core SHARED
  src/core/correlative_scan_matcher.cpp
  src/core/graph_optimizer.cpp
  src/core/ground_plane_extractor.cpp
  src/core/ground_plane_factor.cpp
  src/core/ground_marking_grid.cpp
  src/core/loop_closure.cpp
  src/core/system.cpp
  src/core/occupancy_grid.cpp
  src/core/map_database.cpp
  src/core/camera_model.cpp
  src/core/laser_scan.cpp
  src/core/sensor_data.cpp
)

target_precompile_headers(${PROJECT_NAME}_core PUBLIC
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

target_include_directories(${PROJECT_NAME}_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(${PROJECT_NAME}_core SYSTEM PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

target_compile_features(${PROJECT_NAME}_core PUBLIC cxx_std_17)
target_compile_definitions(${PROJECT_NAME}_core PUBLIC ${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_core
  gtsam
  ${PCL_LIBRARIES}
  ${Eigen3_LIBRARIES}
  nanoflann::nanoflann
  ${OpenCV_LIBS}
)

target_link_libraries(${PROJECT_NAME}_core TBB::tbb)

add_library(${PROJECT_NAME}_ros2 SHARED
  src/ros2/ros2_slam_wrapper.cpp
  src/ros2/utils.cpp
)

target_precompile_headers(${PROJECT_NAME}_ros2 PUBLIC
  <rclcpp/rclcpp.hpp>
  <tf2_ros/buffer.h>
  <tf2_ros/transform_broadcaster.h>
  <tf2_ros/transform_listener.h>
)

target_include_directories(${PROJECT_NAME}_ros2
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(${PROJECT_NAME}_ros2 SYSTEM PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)


target_compile_features(${PROJECT_NAME}_ros2 PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME}_ros2
  rclcpp::rclcpp
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  tf2_ros::tf2_ros
  cv_bridge::cv_bridge
  message_filters::message_filters
  pcl_conversions::pcl_conversions
  ${OpenCV_LIBS}
  gtsam
  ${PCL_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_ros2
  ${PROJECT_NAME}_core
  gtsam
)

add_executable(${PROJECT_NAME}_node src/main.cpp)
target_compile_features(${PROJECT_NAME}_node PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_node
  rclcpp::rclcpp
  ${PROJECT_NAME}_ros2
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(TARGETS
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_ros2
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
