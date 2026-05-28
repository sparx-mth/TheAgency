# RGBD Full Pipeline — ROS 2 Mapping Stack

Monocular RGB → metric depth → 3D occupancy → 2D projected map, with AprilTag-based live pose updates.

## Architecture

```
                 ┌─────────────────────────────────────────────────────┐
  mock: images   │                   PIPELINE NODES                    │
  live: /xtend/rgb ──► FrameSourceNode ──► /rgbd/rgb ──► DepthProcessorNode ──► /rgbd/pointcloud
                 │                                    └──► /rgbd/depth_m        │
                 │                                                               ▼
                 │                                              octomap_server_node
                 │                                               /occupied_cells_vis_array
                 │                                               /projected_map  (2D)
                 │
                 │    /rgbd/rgb ──► apriltag_triangulation_node
                 │                       │ (pupil_apriltags + solvePnP)
                 │                       ▼
                 │              /{xtend_ns}/april_tag_pose  (PoseStamped)
                 │                       │
                 │                       ▼
                 │              pose_to_tf_node ──► TF: map → xtend_camera  (dynamic)
                 │
                 │    static_transform_publisher ──► TF: map → xtend_camera  (fallback, z=1.0m)
                 └─────────────────────────────────────────────────────┘
```

## TF Tree

```
map
 └── xtend_camera          ← dynamic (from pose_to_tf_node when tag visible)
                           ← static fallback at z=1.0m, roll=-π/2, yaw=-π/2
```

`xtend_camera` is the **optical frame**: X=right, Y=down, Z=forward (depth).
The static TF encodes this convention. `pose_to_tf_node` corrects the AprilTag body-frame pose to optical convention via a fixed quaternion composition.

## Nodes

| # | Node | Executable | In | Out |
|---|------|------------|----|-----|
| 1 | `frame_source_node` | `ros_py310` | disk images / `/{ns}/rgb` | `/rgbd/rgb`, `/rgbd/camera_info` |
| 2 | `depth_processor_node` | `ros_py310` | `/rgbd/rgb` | `/rgbd/depth_m`, `/rgbd/pointcloud` |
| 3 | `octomap_server_node` | system | `/rgbd/pointcloud` + TF | `/occupied_cells_vis_array`, `/projected_map` |
| 4 | `static_transform_publisher` | system | — | TF `map→xtend_camera` (fallback) |
| 5 | `apriltag_triangulation_node` | `ros_py310` | `/rgbd/rgb` | `/{ns}/april_tag_pose` |
| 6 | `pose_to_tf_node` | `ros_py310` | `/{ns}/april_tag_pose` | TF `map→xtend_camera` (dynamic) |

## Launch

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=$PYTHONPATH:/home/daphnaa/GIT/TheAgency

# Mock mode (replay saved images)
setsid ros2 launch sparx_agency/tasks/mapping/ros2/rgbd_full_pipeline.launch.py \
  mode:=mock \
  > /tmp/pipeline.log 2>&1 &
echo $! > /tmp/pipeline.pid

# Live mode (XTEND drone streaming)
setsid ros2 launch sparx_agency/tasks/mapping/ros2/rgbd_full_pipeline.launch.py \
  mode:=live \
  xtend_ns:=xtend \
  > /tmp/pipeline.log 2>&1 &
echo $! > /tmp/pipeline.pid
```

In **live mode** the apriltag node subscribes to `/rgbd/rgb` (the frame_source relay).
To bypass the relay and connect directly to the drone camera:
```bash
# edit the apriltag ExecuteProcess cmd in launch file:
"--image_topic", "/{xtend_ns}/rgb"
```

## Kill

```bash
pkill -9 -f "depth_processor_node|frame_source_node|octomap_server|static_transform_publisher|apriltag_triangulation|pose_to_tf_node"
```

## RViz

```bash
rviz2 -d sparx_agency/tasks/mapping/ros2/rgbd_mapping.rviz
```

Displays: PointCloud2 (`/rgbd/pointcloud`), Octomap 3D (`/occupied_cells_vis_array`), OccupancyGrid 2D (`/projected_map`), TF frames.

## Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `xtend_ns` | `xtend` | Drone topic namespace prefix |
| `mode` | `mock` | `mock` (images from disk) or `live` (drone stream) |
| `rgb_dir` | `~/Documents/xtend_da3_takes/…/rgb` | Mock image directory |
| `publish_hz` | `3.0` | Mock frame rate (Hz) — keep ≤ depth model throughput |
| `engine_path` | DA3METRIC-LARGE fp16 engine | TensorRT engine path |
| `config_yaml` | XTEND 720×420 calibration | Camera intrinsics YAML |
| `max_depth_m` | `8.0` | Depth clip + octomap max range (m) |
| `octomap_resolution` | `0.05` | Voxel size (m) |
| `prob_hit` | `0.70` | P(occupied \| ray hit endpoint) |
| `prob_miss` | `0.40` | P(occupied \| ray passes through) |
| `occupancy_thres` | `0.50` | Voxel occupied threshold |
| `min_z` | `0.1` | Min Z for 2D projection (clips floor) |
| `max_z` | `5.0` | Max Z for 2D projection |

## AprilTag Localization

Tags are defined in `sparx_agency/tasks/localization/config/tag_map_path_ALL.yaml` (world XYZ + RPY, tag family `tag36h11`, default size 0.13 m).

The node outputs `world_T_cam` in ROS body convention. `pose_to_tf_node` converts to optical frame convention before broadcasting `map → xtend_camera` TF, which octomap uses to correctly raycast each point cloud at the drone's current pose.

When no tag is visible, TF2 falls back to the static transform after ~10 s.

## Dependencies

```bash
sudo apt install ros-humble-octomap-server
# pupil_apriltags must be available in ros_py310:
/home/daphnaa/venvs/ros_py310/bin/pip install pupil-apriltags
```