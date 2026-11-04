cmake_minimum_required(VERSION 3.27)
project(glidar_slam_eval)

# Fix for the CMP0144 warning from PCL/FLANN
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(lidar_fusion src/lidar_fusion_node.cpp)
target_compile_features(lidar_fusion PUBLIC cxx_std_17)
target_link_libraries(lidar_fusion PUBLIC
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs_library
  tf2::tf2
  tf2_ros::tf2_ros
  message_filters::message_filters
)

install(TARGETS lidar_fusion DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  scripts/evaluate_trajectory.py
  scripts/plot_trajectory.py
  scripts/rgbd_bridge_node.py
  scripts/sim_pose_recorder.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/evaluate_trajectory.py
  scripts/plot_trajectory.py
  scripts/rgbd_bridge_node.py
  scripts/sim_pose_recorder.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()