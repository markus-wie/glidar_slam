from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    glidar_slam_share = get_package_share_directory("glidar_slam_ros")
    glidar_slam_config = os.path.join(
        glidar_slam_share, "config", "glidar_slam_params.yaml"
    )

    default_odom_topic = "/dingo_velocity_controller/odom"
    default_scan_topic = "/scan"

    scan_topic = LaunchConfiguration("scan_topic", default=default_scan_topic)
    odom_topic = LaunchConfiguration("odom_topic", default=default_odom_topic)

    slam_node = Node(
        package="glidar_slam_ros",
        executable="glidar_slam",
        name="glidar_slam",
        output="screen",
        parameters=[
            glidar_slam_config,
            {
                "use_sim_time": True,
                "scan_topic": scan_topic,
                "odom_topic": odom_topic,
            },
        ],
        prefix=["xterm -e gdb -ex run --args"],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("scan_topic", default_value=default_scan_topic),
            DeclareLaunchArgument("odom_topic", default_value=default_odom_topic),
            slam_node,
        ]
    )
