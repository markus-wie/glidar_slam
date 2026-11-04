from datetime import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Evaluate GLiDAR-SLAM in Gazebo against the simulator ground truth."""
    package_share = get_package_share_directory("glidar_slam_eval")
    bringup_share = get_package_share_directory("demo_bringup")
    metrics_script = os.path.join(package_share, "scripts", "evaluate_trajectory.py")
    plot_script = os.path.join(package_share, "scripts", "plot_trajectory.py")

    evaluation_root = "/workspaces/glidar_slam_demo/eval"
    os.makedirs(evaluation_root, exist_ok=True)
    run_directory = os.path.join(
        evaluation_root, "sim_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    os.makedirs(run_directory, exist_ok=False)

    output_dir = LaunchConfiguration("output_dir")
    map_base_path = PathJoinSubstitution([output_dir, "occupancy_map"])
    ground_truth_path = PathJoinSubstitution([output_dir, "ground_truth.tum"])
    trajectory_path = PathJoinSubstitution([output_dir, "estimated.tum"])
    plot_output_path = PathJoinSubstitution([output_dir, "trajectory_comparison.png"])
    alignment_mode = LaunchConfiguration("alignment_mode")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "bringup_sim_launch.py")
        ),
        launch_arguments={
            "scenario": LaunchConfiguration("scenario"),
            "car_type": LaunchConfiguration("car_type"),
            "course_type": LaunchConfiguration("course_type"),
            "gui": LaunchConfiguration("gui"),
            "racing": LaunchConfiguration("racing"),
        }.items(),
    )

    recorder = Node(
        package="glidar_slam_eval",
        executable="sim_pose_recorder.py",
        name="sim_pose_recorder",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "ground_truth_topic": LaunchConfiguration("ground_truth_topic"),
                "ground_truth_static_topic": LaunchConfiguration(
                    "ground_truth_static_topic"
                ),
                "ground_truth_frame": LaunchConfiguration("ground_truth_frame"),
                "estimated_pose_topic": LaunchConfiguration("estimated_pose_topic"),
                "ground_truth_path": ground_truth_path,
                "trajectory_path": trajectory_path,
                "map_base_path": map_base_path,
                "duration": ParameterValue(
                    LaunchConfiguration("duration"), value_type=float
                ),
            }
        ],
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

    return LaunchDescription(
        [
            DeclareLaunchArgument("output_dir", default_value=run_directory),
            DeclareLaunchArgument(
                "duration",
                default_value="60.0",
                description="Simulation seconds to record after the first estimated pose",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="Start Gazebo, RViz and RQT GUIs",
            ),
            DeclareLaunchArgument(
                "racing",
                default_value="true",
                description="Enable lane following so the vehicle drives",
            ),
            DeclareLaunchArgument("scenario", default_value="racing"),
            DeclareLaunchArgument("car_type", default_value="car_rigid"),
            DeclareLaunchArgument(
                "course_type", default_value="course_short_complex_corridor"
            ),
            DeclareLaunchArgument("ground_truth_topic", default_value="/gz/tf"),
            DeclareLaunchArgument(
                "ground_truth_static_topic", default_value="/gz/tf_static"
            ),
            DeclareLaunchArgument(
                "ground_truth_frame",
                default_value="car",
                description="Frame whose pose in the Gazebo world is the ground truth",
            ),
            DeclareLaunchArgument(
                "estimated_pose_topic", default_value="/glidar_slam/estimated_pose"
            ),
            DeclareLaunchArgument("alignment_mode", default_value="se2"),
            simulation,
            recorder,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=recorder,
                    on_exit=[metric_process],
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
