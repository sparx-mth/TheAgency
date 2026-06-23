# XTEND Dome Capture Demo

360° room sweep with RGB + depth capture, AprilTag localization, and offline room mapping.

## Setup

**Jetson:** `192.0.0.89`  
**Repo:** `/home/user/agency_ws`  
**SSH:**
```bash
ssh user@192.0.0.89
```

Each terminal needs this preamble:
```bash
source /opt/ros/humble/setup.bash
source /home/user/agency_ws/venv/bin/activate
cd /home/user/agency_ws
```

---

## Run order

### Terminal 1 — XTEND bridge + frame publisher

Connects to the XTEND WebSocket, saves 504×294 JPEG frames to `/tmp/xtend_frames`, publishes each path on `/xtend/rgb_frame_path`. Also publishes `/xtend/bearing` and `/xtend/local_telemetry`.

```bash
python3 sparx_agency/robots/XTEND/online_nav_bridge_dir_publisher.py \
  --frequency 10.0 \
  --out-dir /tmp/xtend_frames \
  --path-topic /xtend/rgb_frame_path \
  --preprocess-mode resize \
  --output-width 504 \
  --output-height 294
```

---

### Terminal 2 — Depth processor (DA3 Metric Large)

Reads `/xtend/rgb_frame_path`, runs DA3 TRT, saves depth NPY files to `/tmp/xtend_depth`, publishes paths on `/xtend/depth_frame_path`.

```bash
python3 sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
  --ros-args \
  -p frame_path_topic:=/xtend/rgb_frame_path \
  -p depth_topic:=/xtend/depth_m \
  -p engine_path:=/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-392x504.depth_only.engine \
  -p config_yaml:=/home/user/agency_ws/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_392_crop_resize.yaml \
  -p camera_info_mode:=base \
  -p model_type:=large_metric \
  -p apply_metric_focal_scaling:=true \
  -p metric_scale_divisor:=300.0 \
  -p clip_min_m:=0.2 \
  -p clip_max_m:=5.0 \
  -p depth_encoding:=32FC1 \
  -p depth_path_topic:=/xtend/depth_frame_path \
  -p depth_dir:=/tmp/xtend_depth \
  -p max_depth_kept:=300 \
  -p publish_cloud:=true \
  -p pointcloud_topic:=/xtend/pointcloud
```

---

### Terminal 3 — Localization node (AprilTag)

Reads `/xtend/rgb_frame_path`, detects tag36h11 AprilTags, publishes pose on `/xtend/localization` (PoseStamped).

```bash
python3 -m sparx_agency.tasks.localization.ros2.localization_node \
  --ros-args \
  -p provider_type:=apriltag \
  -p frame_path_topic:=/xtend/rgb_frame_path \
  -p tag_map_path:=/home/user/agency_ws/sparx_agency/tasks/localization/config/tag_map_path_ALL.yaml \
  -p camera_calib_path:=/home/user/agency_ws/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
  -p tag_size_m:=0.13
```

---

### Terminal 4 — Dome main (run after drone is ready)

Arm, takeoff, rotate 360° in 90° chunks (guided by `/xtend/localization`), capture frames to disk.

```bash
python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_dome_main.py \
  --pose-topic /xtend/localization \
  --depth-topic /xtend/depth_frame_path \
  --out-dir /home/user/jetson-containers/data/R1 \
  --capture-interval-sec 1.0 \
  --yaw-bucket-deg 30.0
```

Output lands in:
```
/home/user/jetson-containers/data/captures/<YYYYmmdd_HHMMSS>/
  R2_20260127_122951.jpg
  R2_20260127_122951.json   ← pose sidecar {x, y, z, yaw}
  R2_20260127_122951.npy    ← depth from DA3
  ...
/home/user/jetson-containers/data/captures/latest  → symlink to session
```

---

## Offline processing (after the sweep)

### 1. Run NanoOWL detections

Run your NanoOWL service against the session's JPEGs.  
Output: one detection JSON per frame (list of `{label, bbox, score}`).

### 2. Run room mapper

```bash
python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/room_mapper/run_room_mapper.py \
  --data-dir /home/user/Documents/xtend_dome_capture/latest \
  --labels /path/to/detections.json
```

Output: `trajectory.npy`, `occupancy_2d.npy`, `map_with_objects.png` in the session dir.

---

## Useful checks

```bash
# Verify topics are publishing
ros2 topic hz /xtend/rgb_frame_path
ros2 topic hz /xtend/depth_frame_path
ros2 topic hz /xtend/localization

# Watch localization source (apriltag / none)
ros2 topic echo /xtend/localization_source

# Browse latest captures
ls -lh /home/user/Documents/xtend_dome_capture/latest/
```

## Tips

- Wait for Terminal 3 to print a localization message before starting the dome script — the 360° rotation waits up to 5 s per 90° chunk for a pose, but a cold start with no tags visible causes fallback to XTEND bearing.
- The launcher UI (`xtend_pipeline_launcher_ui_mapping.py`) can start terminals 1–3 with one click per step if you prefer a GUI over raw SSH.
- Pass `--depth-topic ""` to skip depth capture (RGB + pose only).