# ROBOTICAN Dome Capture (sensing bridge + 360° sweep)

ROBOTICAN/Rooster equivalent of `sparx_agency/demos/Demo_No4_XTEND_MapRoom`'s
dome sweep: arm, takeoff, rotate 360°, capture RGB + AprilTag pose + DA3
depth to disk per frame, land. Output layout is identical to the XTEND
version, so `room_mapper/run_room_mapper.py` works unchanged on either
robot's captures.

This covers only the sensing bridge + sweep capture. FALCON/InternNav
navigation on ROBOTICAN is a separate, not-yet-built follow-up — see the
"Out of scope" note in this repo's ROBOTICAN sensing-bridge plan.

## One-time: camera calibration

Each ROBOTICAN camera needs its own calibration — calibrate the real drone
camera directly, don't reuse or derive it from XTEND's intrinsics.
`calibrate_camera.py` already exists for this (interactive chessboard
capture against the live stream) and calibrates at the drone's native
resolution:

```bash
python3 sparx_agency/robots/ROBOTICAN/calibrate_camera.py \
  --host-ip <this machine's IP> --drone-id R1
# -> sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml
```

The frame publisher below does no crop/resize — it publishes frames at that
same native resolution — so this calibration file is used as-is (`camera_info_mode:=base`)
for both `depth_processor_node.py` and `localization_node.py`. The DA3
TensorRT engine must be exported/sized to match this same resolution;
`depth_processor_node.py --engine-path` should point at that engine, not
XTEND's 504x294 one.

## Run order

Each terminal needs the ROS2 environment sourced first (see the repo
`README.md`/`run_ui.sh` for the exact Rooster/Sphera env preamble). This
works identically against the Sphera simulator or the real drone — nothing
here branches on which one is running underneath.

### Terminal 1 — Command gateway

The single owner of this drone's FCU (arm/disarm/takeoff/land/video). Every
other process — including `rooster_dome_main.py` below — only ever talks to
it over `/R1/cmd_nav` / `/R1/rooster_status`.

```bash
python3 -m sparx_agency.robots.ROBOTICAN.adapters.rooster_command_unit \
  --ros-args -p rooster_id:=R1
```

### Terminal 2 — Frame publisher

Decodes the drone's UDP/RTP-H264 stream, saves each frame at its native
resolution (no crop/resize) as a JPEG to `/tmp/rooster_frames`, publishes
each path on `/R1/rgb_frame_path`. Frames are always written to disk *and*
their path published — required for both this capture flow and any future
FALCON bridging.

```bash
python3 sparx_agency/robots/ROBOTICAN/rooster_frame_dir_publisher.py \
  --rooster-id R1 \
  --out-dir /tmp/rooster_frames \
  --port 5001
```

`--port` must match `rooster_command_unit`'s `video_port` parameter (default
5001 on both sides). The drone only starts streaming once `rooster_dome_main.py`
sends a `video_on` command in Terminal 4 below — this terminal will show no
frames until then.

### Terminal 3 — DA3 depth + AprilTag localization

Reuses the exact same generic nodes XTEND uses, just repointed at ROBOTICAN's
topics, calibration, and a DA3 TensorRT engine sized/exported for ROBOTICAN's
own frame dimensions (running on the PC's GPU against the Sphera simulator,
not XTEND's Jetson-targeted 504x294 engine):

```bash
python3 sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
  --ros-args \
  -p frame_path_topic:=/R1/rgb_frame_path \
  -p depth_path_topic:=/R1/depth_frame_path \
  -p depth_dir:=/tmp/rooster_depth \
  -p engine_path:=<path to a DA3 engine exported for ROBOTICAN's native resolution> \
  -p config_yaml:=sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml \
  -p camera_info_mode:=base \
  -p model_type:=large_metric \
  -p depth_encoding:=32FC1
```

```bash
python3 -m sparx_agency.tasks.localization.ros2.localization_node \
  --ros-args \
  -p provider_type:=apriltag \
  -p frame_path_topic:=/R1/rgb_frame_path \
  -p camera_calib_path:=sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml \
  -p tag_map_path:=sparx_agency/tasks/localization/config/tag_map_path.yaml \
  -p tag_size_m:=0.13
```

`localization_node.py` always publishes on the fixed `/xtend/localization` /
`/xtend/localization_source` topic names — remap them to ROBOTICAN's
namespace:

```bash
  --ros-args -r /xtend/localization:=/R1/localization -r /xtend/localization_source:=/R1/localization_source
```

**Verify `tag_map_path.yaml` matches the physical AprilTag placement in the
current room before trusting localization output** — it can't be checked
from code.

### Terminal 4 — Dome main

Run once the drone is ready and Terminal 3 is publishing localization.

```bash
python3 sparx_agency/robots/ROBOTICAN/rooster_dome_main.py \
  --rooster-id R1 \
  --pose-topic /R1/localization \
  --out-dir ~/rooster_dome_capture \
  --capture-interval-sec 1.0 \
  --yaw-bucket-deg 30.0
```

This turns on the video stream, arms, takes off, rotates 360° in 90° chunks
(guided by `/R1/localization`, falling back to a rough time-based blind turn
if no pose arrives — tune `--blind-turn-deg-per-sec` for your drone/room if
you ever see that fallback fire), captures frames, then lands, disarms, and
turns the video stream back off — with a `finally`-guaranteed land+disarm
safety net on Ctrl-C or SIGTERM.

Output lands in:
```
~/rooster_dome_capture/<YYYYmmdd_HHMMSS>/
  R1_20260708_143012.jpg
  R1_20260708_143012.json   <- pose sidecar {x, y, z, yaw}
  R1_20260708_143012.npy    <- depth from DA3
  ...
~/rooster_dome_capture/latest  -> symlink to session
```

## Offline processing

Same as XTEND — run NanoOWL against the session's JPEGs, then:

```bash
python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/room_mapper/run_room_mapper.py \
  --data-dir ~/rooster_dome_capture/latest \
  --labels /path/to/detections.json
```

## Useful checks

```bash
ros2 topic hz /R1/rgb_frame_path
ros2 topic hz /R1/depth_frame_path
ros2 topic hz /R1/localization
ros2 topic echo /R1/rooster_status
```
