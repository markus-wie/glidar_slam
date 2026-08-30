cmake_minimum_required(VERSION 3.27)
set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
project(glidar_slam_ros)

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
find_package(pluginlib REQUIRED)
find_package(GTSAM REQUIRED)
find_package(PCL REQUIRED COMPONENTS common registration segmentation)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glidar_slam REQUIRED)
find_package(glidar_slam_msgs REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}_wrapper SHARED
  src/ros2_slam_wrapper.cpp
  src/utils.cpp
)

add_library(${PROJECT_NAME}_scan_matcher_plugins SHARED
  src/correlative_scan_matcher_plugin.cpp
)

target_compile_features(${PROJECT_NAME}_scan_matcher_plugins PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_scan_matcher_plugins
  pluginlib::pluginlib
  rclcpp::rclcpp
  glidar_slam::glidar_slam
)

target_precompile_headers(${PROJECT_NAME}_wrapper PUBLIC
  <rclcpp/rclcpp.hpp>
  <tf2_ros/buffer.h>
  <tf2_ros/transform_broadcaster.h>
  <tf2_ros/transform_listener.h>
)

target_include_directories(${PROJECT_NAME}_wrapper
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(${PROJECT_NAME}_wrapper SYSTEM PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

target_compile_features(${PROJECT_NAME}_wrapper PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME}_wrapper
  pluginlib::pluginlib
  rclcpp::rclcpp
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  ${glidar_slam_msgs_TARGETS}
  tf2_ros::tf2_ros
  cv_bridge::cv_bridge
  message_filters::message_filters
  pcl_conversions::pcl_conversions
  ${OpenCV_LIBS}
  gtsam
  ${PCL_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_wrapper
  glidar_slam::glidar_slam
  gtsam
)

add_executable(glidar_slam src/main.cpp)
target_compile_features(glidar_slam PUBLIC cxx_std_17)
target_link_libraries(glidar_slam
  rclcpp::rclcpp
  ${PROJECT_NAME}_wrapper
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(TARGETS
  ${PROJECT_NAME}_wrapper
  ${PROJECT_NAME}_scan_matcher_plugins
  glidar_slam
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES scan_matcher_plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(
  glidar_slam_ros scan_matcher_plugins.xml
)

ament_package()