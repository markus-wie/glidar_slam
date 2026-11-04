#!/usr/bin/env python3
"""Record Gazebo ground truth and export the final GLiDAR-SLAM trajectory.

The ground truth comes from the Gazebo PosePublisher plugin of the car model, which is bridged to
ROS as tf2_msgs/TFMessage on /gz/tf (dynamic) and /gz/tf_static (static). The transforms are
chained from the world root down to the requested frame, so the recorded ground truth is the pose
of that frame (default: car) in the Gazebo world, sampled at every dynamic TF update.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from glidar_slam_msgs.srv import SaveMaps, SaveTrajectory
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from tf2_msgs.msg import TFMessage


def frame_name(name: str) -> str:
    """Strip Gazebo scoping ("car/base_link" -> "base_link") and a leading slash."""
    return name.strip("/").split("/")[-1]


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return (x, y, z, w)."""
    trace = np.trace(rotation)
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([x, y, z, w])
    quaternion /= np.linalg.norm(quaternion)
    return tuple(quaternion)


def transform_matrix(transform) -> np.ndarray:
    matrix = np.eye(4)
    rotation = transform.rotation
    matrix[:3, :3] = quaternion_to_matrix(
        rotation.x, rotation.y, rotation.z, rotation.w
    )
    translation = transform.translation
    matrix[:3, 3] = (translation.x, translation.y, translation.z)
    return matrix


class SimPoseRecorder(Node):
    def __init__(self) -> None:
        super().__init__("sim_pose_recorder")

        self.declare_parameter("ground_truth_topic", "/gz/tf")
        self.declare_parameter("ground_truth_static_topic", "/gz/tf_static")
        self.declare_parameter("ground_truth_frame", "car")
        self.declare_parameter("estimated_pose_topic", "/glidar_slam/estimated_pose")
        self.declare_parameter(
            "ground_truth_path", "/tmp/glidar_slam_eval/ground_truth.tum"
        )
        self.declare_parameter("trajectory_path", "/tmp/glidar_slam_eval/estimated.tum")
        self.declare_parameter("map_base_path", "")
        # Simulation seconds to record after the first estimated pose. 0 records until shutdown.
        self.declare_parameter("duration", 0.0)

        self.target_frame = frame_name(self.get_parameter("ground_truth_frame").value)
        self.duration = float(self.get_parameter("duration").value)
        ground_truth_path = Path(self.get_parameter("ground_truth_path").value)
        trajectory_path_str = str(self.get_parameter("trajectory_path").value)
        self.trajectory_path = trajectory_path_str
        self.map_base_path = str(self.get_parameter("map_base_path").value)

        ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
        if self.trajectory_path:
            Path(self.trajectory_path).parent.mkdir(parents=True, exist_ok=True)
        self.ground_truth_file = ground_truth_path.open("w", encoding="ascii")
        self.ground_truth_path = ground_truth_path

        # Pre-initialize service clients so there is zero DDS discovery delay at cutoff
        self.save_traj_client = (
            self.create_client(SaveTrajectory, "/glidar_slam/save_trajectory")
            if self.trajectory_path
            else None
        )
        self.save_maps_client = (
            self.create_client(SaveMaps, "/glidar_slam/save_maps")
            if self.map_base_path
            else None
        )

        # child frame -> (parent frame, 4x4 parent_from_child)
        self.static_edges: dict[str, tuple[str, np.ndarray]] = {}
        self.dynamic_edges: dict[str, tuple[str, np.ndarray]] = {}

        self.ground_truth_count = 0
        self.last_ground_truth_stamp = -1.0
        self.last_estimate_stamp = -1.0
        self.first_estimate_stamp: Optional[float] = None
        self.warned_missing_frame = False
        self.duration_reached = False

        dynamic_qos = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE)
        static_qos = QoSProfile(
            depth=100,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            TFMessage,
            self.get_parameter("ground_truth_static_topic").value,
            self._static_callback,
            static_qos,
        )
        self.create_subscription(
            TFMessage,
            self.get_parameter("ground_truth_topic").value,
            self._dynamic_callback,
            dynamic_qos,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("estimated_pose_topic").value,
            self._estimate_callback,
            dynamic_qos,
        )

        self.get_logger().info(
            f"Recording ground truth of '{self.target_frame}' -> '{ground_truth_path}' "
            f"(will export SLAM trajectory -> '{self.trajectory_path}')"
        )

    @staticmethod
    def _store_edges(message: TFMessage, edges: dict) -> None:
        for transform in message.transforms:
            child = frame_name(transform.child_frame_id)
            parent = frame_name(transform.header.frame_id)
            if not child or child == parent:
                continue
            edges[child] = (parent, transform_matrix(transform.transform))

    def _static_callback(self, message: TFMessage) -> None:
        self._store_edges(message, self.static_edges)

    def _lookup_in_root(self, frame: str) -> Optional[np.ndarray]:
        """Chain parent_from_child transforms from `frame` up to the root of the tree."""
        pose = np.eye(4)
        visited = set()
        while True:
            if frame in visited:
                return None
            visited.add(frame)
            edge = self.dynamic_edges.get(frame) or self.static_edges.get(frame)
            if edge is None:
                return pose if len(visited) > 1 else None
            parent, parent_from_child = edge
            pose = parent_from_child @ pose
            frame = parent

    def _dynamic_callback(self, message: TFMessage) -> None:
        if not message.transforms:
            return
        self._store_edges(message, self.dynamic_edges)

        stamp = max(
            transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            for transform in message.transforms
        )
        if stamp <= self.last_ground_truth_stamp:
            return

        pose = self._lookup_in_root(self.target_frame)
        if pose is None:
            if not self.warned_missing_frame:
                self.get_logger().warn(
                    f"Frame '{self.target_frame}' is not part of the Gazebo TF tree yet."
                )
                self.warned_missing_frame = True
            return

        self.last_ground_truth_stamp = stamp
        qx, qy, qz, qw = matrix_to_quaternion(pose[:3, :3])
        self.ground_truth_file.write(
            f"{stamp:.9f} {pose[0, 3]:.17g} {pose[1, 3]:.17g} {pose[2, 3]:.17g} "
            f"{qx:.17g} {qy:.17g} {qz:.17g} {qw:.17g}\n"
        )
        self.ground_truth_count += 1

    def _estimate_callback(self, message: PoseWithCovarianceStamped) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if stamp <= self.last_estimate_stamp:
            return
        self.last_estimate_stamp = stamp
        if self.first_estimate_stamp is None:
            self.first_estimate_stamp = stamp

        if (
            not self.duration_reached
            and self.duration > 0.0
            and stamp - self.first_estimate_stamp >= self.duration
        ):
            self.get_logger().info(
                f"Recorded {self.duration} s of simulation (t={stamp:.3f}s), exporting SLAM outputs..."
            )
            self.duration_reached = True

    def _report_save_result(self, future, description: str) -> None:
        response = future.result() if future.done() else None
        if response is None:
            self.get_logger().error(f"Saving {description} timed out")
        elif not response.success:
            self.get_logger().error(
                f"Saving {description} failed: {response.message}"
            )
        else:
            self.get_logger().info(f"Saved {description}")

    def export_slam_outputs(self) -> None:
        """Synchronously call SaveTrajectory and SaveMaps before closing."""
        if self.save_traj_client is not None:
            if self.save_traj_client.wait_for_service(timeout_sec=5.0):
                req = SaveTrajectory.Request()
                req.filepath = self.trajectory_path
                future = self.save_traj_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
                self._report_save_result(
                    future, f"SLAM trajectory to '{self.trajectory_path}'"
                )
            else:
                self.get_logger().error(
                    "/glidar_slam/save_trajectory service not available"
                )

        if self.save_maps_client is not None:
            if self.save_maps_client.wait_for_service(timeout_sec=5.0):
                req = SaveMaps.Request()
                req.base_filepath = self.map_base_path
                req.save_occupancy = True
                req.save_texture = False
                future = self.save_maps_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
                self._report_save_result(
                    future, f"occupancy map to '{self.map_base_path}'"
                )
            else:
                self.get_logger().error("/glidar_slam/save_maps service not available")

    def close(self) -> None:
        if not self.ground_truth_file.closed:
            self.ground_truth_file.flush()
            self.ground_truth_file.close()
        if self.ground_truth_count == 0:
            self.get_logger().error(
                "Recorded no ground truth poses. Does the car model publish its pose to "
                "the ground truth topic?"
            )
            return
        self.get_logger().info(
            f"Recorded {self.ground_truth_count} ground truth poses to "
            f"{self.ground_truth_path}"
        )


def main() -> None:
    rclpy.init()
    node = SimPoseRecorder()
    try:
        while rclpy.ok() and not node.duration_reached:
            rclpy.spin_once(node, timeout_sec=0.1)
        if rclpy.ok() and node.duration_reached:
            node.export_slam_outputs()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
