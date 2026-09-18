
<p align="center">
  <h1 align="center">GLiDAR-SLAM: Ground Constrained 2D LiDAR SLAM</h1>
  <p align="center">
    <strong>Markus Wiechers</strong>
  </p>
</p>
<p align="center">
  <a href="">
    <img src="./media/glidar_slam.gif" width="80%">
  </a>
</p>

---

GLiDAR-SLAM is a ground-constrained 2D LiDAR SLAM system implementing a core library and a ros2 wrapper.
It combines 2D LiDAR correlative scan matching with optional RGB-D ground-plane, ground-texture, and ground-marking information formulated as a graph-SLAM problem utilizing a factor graph.

A demo environment is available at [glidar_slam_demo](https://github.com/markus-wie/glidar_slam_demo).

## Prerequisites

- Ubuntu 22.04 LTS or later
- ROS 2 Kilted (tested only on Kilted, might work on other distros as well)
- Installed dependencies via rosdep.

## Build

Build the GLiDAR-SLAM packages in Release mode from the workspace root:

```bash
# Clone the repo
cd glidar_slam
# Install rosdep dependencies
rosdep update
rosdep install --from-paths glidar_slam glidar_slam_ros glidar_slam_msgs --ignore-src -r -y
# Build
colcon build --packages-up-to glidar_slam --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Run

Launch the ROS 2 wrapper with its default configuration:

```bash
source install/setup.bash
ros2 launch glidar_slam_ros glidar_slam.launch.py
```

The launch file accepts `log_level` and `use_sim_time` arguments. The default
configuration is installed with the package and can be found in
`glidar_slam_ros/config/glidar_slam_params.yaml`:

```bash
source install/setup.bash
ros2 launch glidar_slam_ros glidar_slam.launch.py \
  log_level:=info \
  use_sim_time:=true
```

The default topics are:

| Data | Topic |
| --- | --- |
| Transform Tree | `/tf` |
| Odometry | `/odom` |
| LiDAR | `/scan` |
| Color image | `/camera/color/image_raw` |
| Aligned depth image | `/camera/aligned_depth/image_raw` |
| Camera calibration | `/camera/color/camera_info` |

Topic names, frame names, startup mode, matcher settings, and mapping settings are configurable in the parameter file.

## Project Structure

The subproject contains three ROS 2 packages:

- **`glidar_slam`**: ROS-independent C++17 SLAM library. It contains sensor representations, scan matcher, keyframes, maps, graph optimization, loop closure, ground-plane processing, and state serialization.
- **`glidar_slam_ros`**: ROS 2 node and adapter. It converts ROS messages to core-library data, loads scan-matcher plugins, publishes maps and poses, and exposes runtime services.
- **`glidar_slam_msgs`**: Custom service definitions for saving/loading SLAM state and maps, and switching localization mode.

## SLAM Pipeline

At a surface level, the pipeline is:

1. **Sensor synchronization and preprocessing**. LiDAR scans are optionally voxelized and densified into a more uniform 2D point cloud. RGB-D data is synchronized with scans when ground processing is enabled.
2. **Motion estimation**. The odometry is sampled from the tf tree and supplies a coarse motion estimate which is refined by subsequent submap based scan matching.
3. **Local scan matching**. A correlative scan matcher aligns each scan against a sliding local submap. It uses a multi-resolution likelihood-field search and returns a planar pose estimate and covariance. (TODO: use branch and bound)
4. **Ground-plane constraint**. RGB-D points are fit to a ground plane. The resulting normal and height from `base_link` are added as a ground factor to the factor graph, constraining vertical and attitude-related motion. This feature is optional; the primary LiDAR trajectory remains planar.
5. **Keyframes and local maps**. A keyframe is created after configurable translation or heading motion. Keyframes retain scans, poses, odometry, covariance, and ground observations. A bounded window of recent keyframes forms the local matching submap.
6. **Pose-graph optimization**. GTSAM/iSAM2 combines the initial prior, consecutive scan-matching constraints, odometry constraints, optional ground factors, and loop closure constraints into an optimized 6-DoF trajectory.
7. **Loop closure**. A detector running in a separate thread selects older keyframes in vicinity (and with fitting topological context) and verifies them with scan matching. Accepted matches add relative constraints to the graph.
8. **Map construction**. Keyframe scans are ray-traced into local occupancy grid maps that are accumulated into a global occupancy grid map in a background thread. Optional layers store RGB ground texture and ground-marking evidence at their own resolutions, that themselves are also accumulated into a texture map.

## Outputs

The ROS wrapper publishes:

- `glidar_slam/estimated_pose` (`geometry_msgs/msg/PoseWithCovarianceStamped`)
- `glidar_slam/map` (`nav_msgs/msg/OccupancyGrid`)
- `glidar_slam/ground_markings_map` (`nav_msgs/msg/OccupancyGrid`)
- `glidar_slam/ground_texture_map` (`sensor_msgs/msg/Image`)
- `glidar_slam/graph_visualization` (`visualization_msgs/msg/MarkerArray`)
- The `map -> odom` TF transform

Additional debug images, point clouds, and matcher visualizations are
available when enabled in the parameter file.

## Modes
GLiDAR-SLAM provides both **mapping** and **localization** modes.

In the **mapping** mode, new keyframes and map data are added continuously. 

In the **localization** mode, you can either load a map at startup or switch to the localization mode at runtime using the exposed service.
Global relocalization is currently not supported, so you need to provide an initial accurate pose when switching to localization mode.
Nearby keyframes (by topology) are used to construct a submap against which new scans are matched.
The pose is piped into an EKF for smooth state estimation of the localized pose which is subsequently published as `glidar_slam/estimated_pose` and `map -> odom` TF transform.

## Services and Modes

The node provides these services:

- `/glidar_slam/save_state`: serialize keyframes and pose-graph state to a binary file.
- `/glidar_slam/load_state`: restore a saved graph, optionally using a supplied pose.
- `/glidar_slam/save_maps`: export occupancy, texture, and marking maps.
- `/glidar_slam/set_localization_mode`: switch between mapping and localization.

The service names are relative to the node namespace, so the examples below assume the default node name and namespace.
Paths are resolved by the running ROS2 process and should point to a writable location.

### Save SLAM State

Save the current keyframes and factor graph to a binary file:

```bash
ros2 service call /glidar_slam/save_state \
  glidar_slam_msgs/srv/SaveSlamState \
  "{ \
    path: '/tmp/glidar_mapping.bin' \
  }"
```

### Load SLAM State

Load a saved state and continue in mapping mode. 
With `use_saved_pose: true`, the pose stored in the state file is used:

```bash
ros2 service call /glidar_slam/load_state \
  glidar_slam_msgs/srv/LoadSlamState \
  "{ \
    path: '/tmp/glidar_mapping.bin', \
    mode: 'mapping', \
    use_saved_pose: true, \
    initial_map_pose: \
    { \
      position: {x: 0.0, y: 0.0, z: 0.0}, \
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    } \
  }"
```

To load the same state directly into localization mode, change `mode` to `localization`.
When `use_saved_pose` is false, the supplied `initial_map_pose` is used as the map-frame pose instead.

### Save Maps

Export the occupancy map and the RGB ground-texture map using a common base filepath.
Occupancy output is written as: `<base_filepath>.pgm` and `<base_filepath>.yaml`
Texture output is written as: `<base_filepath>_texture.png` and `<base_filepath>_texture.yaml`:

```bash
ros2 service call /glidar_slam/save_maps \
  glidar_slam_msgs/srv/SaveMaps \
  "{ \
    base_filepath: '/tmp/glidar_map', \
    save_occupancy: true, \
    save_texture: true \
  }"
```

At least one of `save_occupancy` or `save_texture` must be true, and a global map must already be available.

### Set Localization Mode

Switch an existing mapping session into localization mode while using the node's current estimated pose as the initial map pose:

```bash
ros2 service call /glidar_slam/set_localization_mode \
  glidar_slam_msgs/srv/SetLocalizationMode \
  "{ \
    enable: true, \
    use_current_pose: true, \
    initial_map_pose: { \
      position: {x: 0.0, y: 0.0, z: 0.0}, \
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0} \
    } \
  }"
```

To provide an explicit initial pose instead, set `use_current_pose: false` and replace `initial_map_pose`.
Disable localization and resume mapping with:

```bash
ros2 service call /glidar_slam/set_localization_mode \
  glidar_slam_msgs/srv/SetLocalizationMode \
  "{ \
    enable: false, \
    use_current_pose: true, \
    initial_map_pose: { \
      position: {x: 0.0, y: 0.0, z: 0.0}, \
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0} \
    } \
  }"
```

## Configuration

The main configuration file is `glidar_slam_ros/config/glidar_slam_params.yaml`. 
Important parameter groups include:

- `startup`: mapping/localization mode, saved-state path, and initial pose.
- `scan`: topic, voxelization, and densification settings.
- `mapping`: occupancy and ground-map resolutions and update probabilities.
- `ground_plane`: RGB-D region of interest and plane-fitting settings.
- `scan_matching`: coarse-to-fine search windows, resolutions, and thresholds.
- `loop_closure`: candidate separation, geometric filtering, and score thresholds.

The ROS wrapper loads scan matchers through `pluginlib`. 
The included correlative matcher is configured for local tracking, ground-marking detection, and loop-closure use cases with different tuning parameters.

## Scope and Limitations

- GLiDAR-SLAM is primarily a planar 2D LiDAR SLAM system.
- It does not provide a general 3D reconstruction pipeline: RGB-D data is used for optional ground constraints only. 
- Successful startup also depends on receiving valid odometry and sensor data, and localization mode requires a previously saved state. 
- Exact matcher score semantics and the binary state format are internal implementation details rather than stable external interfaces.
- The serialization is a dumb binary format. Any changes to the internal data structures will break compatibility.
