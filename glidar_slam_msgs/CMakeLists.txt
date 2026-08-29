cmake_minimum_required(VERSION 3.18)
project(glidar_slam_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

set(services
  "srv/SaveSlamState.srv"
  "srv/SaveMaps.srv"
  "srv/LoadSlamState.srv"
  "srv/SetLocalizationMode.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${services}
  DEPENDENCIES geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime rosidl_default_generators geometry_msgs)

ament_package()
