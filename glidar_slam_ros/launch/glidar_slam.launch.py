from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value=["INFO"],
        description="Logging level",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Enable ros2 simulation time",
    )

    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    log_level = LaunchConfiguration("log_level", default="INFO")

    glidar_slam_dir = get_package_share_directory("glidar_slam_ros")

    config_file = os.path.join(glidar_slam_dir, "config", "glidar_slam_params.yaml")

    return LaunchDescription(
        [
            log_level_arg,
            use_sim_time_arg,
            Node(
                package="glidar_slam_ros",
                executable="glidar_slam",
                name="glidar_slam",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "use_sim_time": use_sim_time,
                    },
                ],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    ["glidar_slam:=", log_level],
                ],
            ),
        ]
    )
