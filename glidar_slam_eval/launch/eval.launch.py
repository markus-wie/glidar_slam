from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from datetime import datetime
import os


def generate_launch_description():
    package_share = get_package_share_directory("glidar_slam_eval")
    glidar_slam_share = get_package_share_directory("glidar_slam_ros")
    glidar_slam_config = os.path.join(
        glidar_slam_share, "config", "glidar_slam_params.yaml"
    )
    metrics_script = os.path.join(package_share, "scripts", "evaluate_trajectory.py")
    plot_script = os.path.join(package_share, "scripts", "plot_trajectory.py")

    # default_bag_path = "/workspaces/glidar_slam_demo/bagfiles/mapping_ros2"
    default_bag_path = (
        "/workspaces/glidar_slam_demo/bagfiles/rosbag2_2026_10_01-16_25_46"
    )
    default_ground_truth_path = "/workspaces/glidar_slam_demo/bagfiles/gt_poses.txt"
    evaluation_root = "/workspaces/glidar_slam_demo/eval"
    os.makedirs(evaluation_root, exist_ok=True)
    run_directory = os.path.join(
        evaluation_root, datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    os.makedirs(run_directory, exist_ok=False)
    default_output_dir = run_directory
    default_map_base_path = os.path.join(default_output_dir, "occupancy_map")
    default_scan_topic = "/scan"
    default_front_scan_topic = "/laser_scan_front/scan"
    default_rear_scan_topic = "/laser_scan_rear/scan"
    default_odom_topic = "/dingo_velocity_controller/odom"
    default_trajectory_path = os.path.join(default_output_dir, "estimated.tum")
    default_playback_rate = "2.0"
    default_alignment_mode = "se2"

    bag_path = LaunchConfiguration("bag_path", default=default_bag_path)
    ground_truth_path = LaunchConfiguration(
        "ground_truth_path", default=default_ground_truth_path
    )
    output_dir = LaunchConfiguration("output_dir", default=default_output_dir)
    map_base_path = LaunchConfiguration("map_base_path", default=default_map_base_path)
    scan_topic = LaunchConfiguration("scan_topic", default=default_scan_topic)
    front_scan_topic = LaunchConfiguration(
        "front_scan_topic", default=default_front_scan_topic
    )
    rear_scan_topic = LaunchConfiguration(
        "rear_scan_topic", default=default_rear_scan_topic
    )
    scan_sync_max_interval = LaunchConfiguration(
        "scan_sync_max_interval", default="0.1"
    )
    odom_topic = LaunchConfiguration("odom_topic", default=default_odom_topic)
    trajectory_path = LaunchConfiguration(
        "trajectory_path", default=default_trajectory_path
    )
    playback_rate = LaunchConfiguration("playback_rate", default=default_playback_rate)
    alignment_mode = LaunchConfiguration(
        "alignment_mode", default=default_alignment_mode
    )
    plot_output_path = PathJoinSubstitution([output_dir, "trajectory_comparison.png"])

    bag_player = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "--clock",
            "--rate",
            playback_rate,
        ],
        output="log",
    )
    metric_process = ExecuteProcess(
        cmd=[
            "python3",
            metrics_script,
            "--ground-truth",
            ground_truth_path,
            "--estimate",
            trajectory_path,
            "--output-dir",
            output_dir,
            "--alignment-mode",
            alignment_mode,
        ],
        output="screen",
    )
    plot_process = ExecuteProcess(
        cmd=[
            "python3",
            plot_script,
            trajectory_path,
            ground_truth_path,
            "--output",
            plot_output_path,
            "--alignment-mode",
            alignment_mode,
        ],
        output="screen",
    )
    save_maps_request = PythonExpression(
        [
            "'{base_filepath: ' + '",
            map_base_path,
            "' + ', save_occupancy: true, save_texture: false}'",
        ]
    )
    save_maps_process = ExecuteProcess(
        cmd=[
            "ros2",
            "service",
            "call",
            "/glidar_slam/save_maps",
            "glidar_slam_msgs/srv/SaveMaps",
            save_maps_request,
        ],
        output="screen",
    )

    fusion_node = Node(
        package="glidar_slam_eval",
        executable="lidar_fusion",
        name="lidar_fusion",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "front_scan_topic": front_scan_topic,
                "rear_scan_topic": rear_scan_topic,
                "output_topic": scan_topic,
                "sync_max_interval": scan_sync_max_interval,
            }
        ],
    )

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
    )

    bridge_node = Node(
        package="glidar_slam_eval",
        executable="rgbd_bridge_node.py",
        name="rgbd_bridge",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "scan_topic": scan_topic,
                "estimated_pose_topic": "glidar_slam/estimated_pose",
                "trajectory_path": trajectory_path,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag_path", default_value=default_bag_path),
            DeclareLaunchArgument(
                "ground_truth_path", default_value=default_ground_truth_path
            ),
            DeclareLaunchArgument("output_dir", default_value=default_output_dir),
            DeclareLaunchArgument("map_base_path", default_value=default_map_base_path),
            DeclareLaunchArgument("scan_topic", default_value=default_scan_topic),
            DeclareLaunchArgument(
                "front_scan_topic", default_value=default_front_scan_topic
            ),
            DeclareLaunchArgument(
                "rear_scan_topic", default_value=default_rear_scan_topic
            ),
            DeclareLaunchArgument("scan_sync_max_interval", default_value="0.1"),
            DeclareLaunchArgument("odom_topic", default_value=default_odom_topic),
            DeclareLaunchArgument(
                "trajectory_path", default_value=default_trajectory_path
            ),
            DeclareLaunchArgument("playback_rate", default_value=default_playback_rate),
            DeclareLaunchArgument(
                "alignment_mode", default_value=default_alignment_mode
            ),
            fusion_node,
            slam_node,
            bridge_node,
            bag_player,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=bag_player,
                    on_exit=[TimerAction(period=1.5, actions=[save_maps_process])],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=save_maps_process,
                    on_exit=[TimerAction(period=1.0, actions=[metric_process])],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(target_action=metric_process, on_exit=[plot_process])
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=plot_process,
                    on_exit=[EmitEvent(event=Shutdown())],
                )
            ),
        ]
    )
