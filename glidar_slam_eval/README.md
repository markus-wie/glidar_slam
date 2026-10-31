# GLiDAR-SLAM Evaluation Adapter

This package provides an evaluation adapter for the GLiDAR-SLAM system, allowing for the assessment of SLAM performance.
It replays [LILocBench](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/) rosbags through `glidar_slam_ros` and computes APE/RPE metrics against the ground truth.

Usage, launch arguments, metrics and caveats are documented in the [Evaluation section of the main README](../README.md#evaluation-glidar_slam_eval).

```bash
ros2 launch glidar_slam_eval eval.launch.py bag_path:=/path/to/rosbag ground_truth_path:=/path/to/gt_poses.txt
```
