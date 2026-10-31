#!/usr/bin/env python3
"""Synthesize synchronized RGB-D ground plane streams and log estimated poses for GLiDAR-SLAM wrapper."""

from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSProfile,
)
from sensor_msgs.msg import CameraInfo, Image, LaserScan


class RgbdBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_bridge")

        self.declare_parameter(
            "scan_topic",
            "/laser_scan_front/scan",
            ParameterDescriptor(description="Input LaserScan topic from bag"),
        )
        self.declare_parameter(
            "color_image_topic",
            "/camera/color/image_raw",
            ParameterDescriptor(description="Output color image topic"),
        )
        self.declare_parameter(
            "aligned_depth_image_topic",
            "/camera/aligned_depth/image_raw",
            ParameterDescriptor(description="Output aligned depth image topic"),
        )
        self.declare_parameter(
            "color_camera_info_topic",
            "/camera/color/camera_info",
            ParameterDescriptor(description="Output camera info topic"),
        )
        self.declare_parameter(
            "camera_frame_id",
            "camera_front_color_optical_frame",
            ParameterDescriptor(description="Optical frame of the synthesized camera"),
        )
        self.declare_parameter(
            "estimated_pose_topic",
            "glidar_slam/estimated_pose",
            ParameterDescriptor(description="Estimated pose topic to record"),
        )
        self.declare_parameter(
            "trajectory_path",
            "/tmp/glidar_slam_eval/estimated.tum",
            ParameterDescriptor(description="Output TUM trajectory path"),
        )
        self.declare_parameter(
            "image_width",
            640,
            ParameterDescriptor(description="Synthetic image width"),
        )
        self.declare_parameter(
            "image_height",
            480,
            ParameterDescriptor(description="Synthetic image height"),
        )

        self.scan_topic = self.get_parameter("scan_topic").value
        self.color_image_topic = self.get_parameter("color_image_topic").value
        self.aligned_depth_image_topic = self.get_parameter(
            "aligned_depth_image_topic"
        ).value
        self.color_camera_info_topic = self.get_parameter(
            "color_camera_info_topic"
        ).value
        self.camera_frame_id = self.get_parameter("camera_frame_id").value
        self.estimated_pose_topic = self.get_parameter("estimated_pose_topic").value
        self.trajectory_path = Path(self.get_parameter("trajectory_path").value)
        self.width = int(self.get_parameter("image_width").value)
        self.height = int(self.get_parameter("image_height").value)

        self.fx = float(self.width) / 2.0
        self.fy = float(self.width) / 2.0
        self.cx = float(self.width) / 2.0
        self.cy = float(self.height) / 2.0

        # Precompute synthetic images
        self._precompute_images()

        # Output trajectory file
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self.trajectory_file = self.trajectory_path.open("w", encoding="ascii")
        self.logged_pose_count = 0
        self.last_timestamp = -1.0

        # Reliable QoS for image streams to match glidar_slam subscriber expectations
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.color_pub = self.create_publisher(Image, self.color_image_topic, image_qos)
        self.depth_pub = self.create_publisher(
            Image, self.aligned_depth_image_topic, image_qos
        )
        self.info_pub = self.create_publisher(
            CameraInfo, self.color_camera_info_topic, image_qos
        )

        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, scan_qos
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.estimated_pose_topic,
            self._pose_callback,
            10,
        )

        self.get_logger().info(
            f"RGB-D Bridge started: bridging '{self.scan_topic}' -> RGB-D ({self.camera_frame_id}); "
            f"recording '{self.estimated_pose_topic}' -> '{self.trajectory_path}'"
        )

    def _precompute_images(self) -> None:
        # Rotation matrix from camera optical frame to base_link
        # (X_opt: right, Y_opt: down, Z_opt: forward; +Y_opt points into ground -Z_base)
        r_base_cam = np.array(
            [
                [0.00379393, -0.04094911, 0.99915403],
                [-0.99995874, 0.00809198, 0.00412863],
                [-0.00825419, -0.99912847, -0.04091672],
            ],
            dtype=np.float32,
        )
        # Translation from base_link to camera optical frame (camera height ~0.66m)
        t_base_cam = np.array([0.09436128, -0.02190898, 0.65973093], dtype=np.float32)

        u = np.arange(self.width, dtype=np.float32) + 0.5
        v = np.arange(self.height, dtype=np.float32) + 0.5
        uu, vv = np.meshgrid(u, v)

        ray_x = (uu - self.cx) / self.fx
        ray_y = (vv - self.cy) / self.fy

        denom = r_base_cam[2, 0] * ray_x + r_base_cam[2, 1] * ray_y + r_base_cam[2, 2]
        depth_array = np.zeros((self.height, self.width), dtype=np.float32)

        # For rays pointing towards the ground (denom < -0.05), depth = -T_z / denom
        ground_mask = denom < -0.05
        depth_array[ground_mask] = -t_base_cam[2] / denom[ground_mask]
        depth_array[(depth_array < 0.1) | (depth_array > 4.5)] = 0.0

        self.depth_bytes = depth_array.tobytes()

        # BGR image with subtle contrast pattern for adaptive thresholding
        bgr_array = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        bgr_array[::4, ::4] = 64
        bgr_array[2::4, 2::4] = 192
        self.bgr_bytes = bgr_array.tobytes()

        # Precompute CameraInfo
        self.camera_info = CameraInfo()
        self.camera_info.width = self.width
        self.camera_info.height = self.height
        self.camera_info.distortion_model = "plumb_bob"
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_info.k = [
            self.fx,
            0.0,
            self.cx,
            0.0,
            self.fy,
            self.cy,
            0.0,
            0.0,
            1.0,
        ]
        self.camera_info.r = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        self.camera_info.p = [
            self.fx,
            0.0,
            self.cx,
            0.0,
            0.0,
            self.fy,
            self.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

    def _scan_callback(self, scan_msg: LaserScan) -> None:
        stamp = scan_msg.header.stamp

        # CameraInfo
        info_msg = CameraInfo()
        info_msg.header.stamp = stamp
        info_msg.header.frame_id = self.camera_frame_id
        info_msg.width = self.camera_info.width
        info_msg.height = self.camera_info.height
        info_msg.distortion_model = self.camera_info.distortion_model
        info_msg.d = list(self.camera_info.d)
        info_msg.k = list(self.camera_info.k)
        info_msg.r = list(self.camera_info.r)
        info_msg.p = list(self.camera_info.p)

        # Color image
        color_msg = Image()
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = self.camera_frame_id
        color_msg.height = self.height
        color_msg.width = self.width
        color_msg.encoding = "bgr8"
        color_msg.is_bigendian = 0
        color_msg.step = self.width * 3
        color_msg.data = self.bgr_bytes

        # Depth image (32FC1)
        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.camera_frame_id
        depth_msg.height = self.height
        depth_msg.width = self.width
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = self.width * 4
        depth_msg.data = self.depth_bytes

        # Publish all synchronously
        self.info_pub.publish(info_msg)
        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)

    def _pose_callback(self, pose_msg: PoseWithCovarianceStamped) -> None:
        stamp = pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec * 1e-9
        if stamp <= self.last_timestamp:
            return
        self.last_timestamp = stamp
        pos = pose_msg.pose.pose.position
        ori = pose_msg.pose.pose.orientation

        self.trajectory_file.write(
            f"{stamp:.9f} {pos.x:.17g} {pos.y:.17g} {pos.z:.17g} "
            f"{ori.x:.17g} {ori.y:.17g} {ori.z:.17g} {ori.w:.17g}\n"
        )
        self.trajectory_file.flush()
        self.logged_pose_count += 1

    def destroy_node(self) -> None:
        if hasattr(self, "trajectory_file") and not self.trajectory_file.closed:
            self.trajectory_file.close()
            self.get_logger().info(
                f"Trajectory file closed: {self.logged_pose_count} poses recorded to {self.trajectory_path}"
            )
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RgbdBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
