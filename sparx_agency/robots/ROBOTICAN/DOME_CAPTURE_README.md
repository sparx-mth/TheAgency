# ROBOTICAN Dome Capture (sensing bridge + 360° sweep)

ROBOTICAN/Rooster equivalent of `sparx_agency/demos/Demo_No4_XTEND_MapRoom`'s
dome sweep: arm, takeoff, rotate 360°, capture RGB + AprilTag pose + DA3
depth to disk per frame, land. Output layout is identical to the XTEND
version, so `room_mapper/run_room_mapper.py` works unchanged on either
robot's captures.

This covers only the sensing bridge + sweep capture. FALCON/InternNav
navigation on ROBOTICAN is a separate, not-yet-built follow-up.

**Confirmed working end-to-end 2026-07-13** against live Sphera: gateway
(arm/takeoff/hover/rotate/land), frame publisher, DA3 depth, ground-truth
localization, and the full autonomous dome-sweep mission all validated on
real Sphera video/telemetry — see `NEXT_SESSION_PLAN.md` for exactly what
was confirmed and what's still open.

## Offline replay (no Sphera needed)

`rooster_offline_frame_dir_publisher.py` replays a previously-captured
session (produced by `rooster_dome_main.py`) back onto the exact same
topics the live pipeline uses (`/R1/rgb_frame_path`, `/R1/depth_frame_path`,
`/R1/localization`) — for testing downstream consumers (FALCON, etc.)
without a live drone/Sphera connection at all:
```bash
./sparx_agency/robots/ROBOTICAN/run_rooster_offline_replay.sh \
  --session-dir ~/rooster_dome_capture/latest \
  --rooster-id R1 --rate 2.0 [--loop]
```
RGB/depth files are copied into `--rgb-out-dir`/`--depth-out-dir` (default
`/tmp/rooster_frames`/`/tmp/rooster_depth` — the same paths the live
pipeline uses), so a consumer mounting those fixed directories doesn't need
to know whether it's live or replayed.

## Where things run (read this first)

Not everything runs on the host. `rooster_command_unit.py` needs the custom
Rooster ROS2 interfaces (`fcu_driver_interfaces`, `rooster_handler_interfaces`,
`rooster_manager_interfaces`, `video_handler_interfaces`), which are only
built for ROS2 **Foxy** inside the `it` Docker container
(`sphera-backend:rooster-with-sparx`) — this host only has Jazzy, and
`~/rqs_iai_ws` here is a stale/incompatible Foxy-era build. Everything else
(frame publisher, DA3, localization, dome main, `ui.py`) only needs plain
`rclpy`/`std_msgs`/`geometry_msgs`, so it runs on the host.

`sparx_agency` is bind-mounted read-write into `it` at
`/home/rooster/sparx_agency` (same files as this repo — edits here are
live there immediately, no rebuild/copy step).

| Process | Where | Why |
|---|---|---|
| `rooster_command_unit.py` | **container `it`** | needs Foxy-built custom Rooster interfaces |
| `rooster_frame_dir_publisher.py` | host | only `std_msgs`, `rclpy`, `gi`/GStreamer |
| `depth_processor_node.py` | host | runs DA3 on the PC's GPU |
| `localization_node.py` | host | only `std_msgs`/`geometry_msgs` |
| `ui.py` | host | only `std_msgs`, `rclpy` |
| `rooster_dome_main.py` | host | only `std_msgs`/`geometry_msgs` |

**CycloneDDS network interface — pragmatic fix in place as of 2026-07-13,
temporary.** All three Sphera/Rooster containers (`it`, `R1`,
`drone_simulator`) run with `NetworkMode: host`, so they share this
machine's network namespace directly — there's no container/host boundary
here. This machine has two active NICs: `enx00e04c680602` (192.168.131.x,
the network the Sphera vendor says Rooster traffic *should* use) and
`enp129s0` (172.16.17.x, general/internet network). Without `CYCLONEDDS_URI`
explicitly set, CycloneDDS silently ignores its config and picks a network
interface "arbitrarily" — which crashed the older-CycloneDDS containers
with a SIGSEGV in their discovery-packet parser (`ddsi_plist_init_frommsg`)
as soon as they saw a mismatched/cross-interface participant.

`R1` is spawned **internally by the Sphera app itself** with no
`CYCLONEDDS_URI`/config at all, and always "arbitrarily" lands on
`enp129s0` — confirmed not fixable from our side (no compose file, script,
or env var controls R1's launch). Rather than keep fighting that, everyone
else is currently pointed at `enp129s0`/`172.16.17.10` to match it:
`/home/user1/rqs_iai_ws/src/cyclonedds.xml` (the single shared config file,
bind-mounted as `/etc/cyclonedds.xml` into both `it` and `drone_simulator`,
and loaded via `CYCLONEDDS_URI` by every host-side `run_*.sh` wrapper) has
`<NetworkInterfaceAddress>172.16.17.10</NetworkInterfaceAddress>`. Verified
working: `rooster_command_unit.py` starts without crashing and
`/R1/rooster_status` carries real telemetry.

**This contradicts the vendor's stated 192.168.131.x requirement** and is
only in place to unblock testing — the user is asking the vendor how they
intend R1's spawned-container networking to be configured. Once that's
answered, change `NetworkInterfaceAddress` back to `192.168.131.20`
(or `enx00e04c680602`) in that one shared file.

## One-time: camera calibration

Each ROBOTICAN camera needs its own calibration — calibrate the real drone
camera directly, don't reuse or derive it from XTEND's intrinsics.
`calibrate_camera.py` already exists for this (interactive chessboard
capture against the live stream) and calibrates at the drone's native
resolution:

```bash
./sparx_agency/robots/ROBOTICAN/run_calibrate_camera.sh \
  --host-ip <this machine's IP> --drone-id R1
# -> sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml
```

For the Sphera simulator specifically, a starting-point calibration already
exists at `sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml`
— an ideal-pinhole approximation for the wide FOV (130°x90°) already assumed
elsewhere in this codebase (`rooster_video_adapter.py`'s `intrinsic_from_fov`),
**not verified against Sphera's actual camera sensor definition**. Replace it
if you know the real FOV/intrinsics Sphera uses for R1's camera.

The frame publisher below does no crop/resize — it publishes frames at the
drone's native resolution — so this calibration file is used as-is
(`camera_info_mode:=base`) for both `depth_processor_node.py` and
`localization_node.py`.

## Run order

All host commands below assume this preamble (or use the `run_*.sh`
wrappers, which already do this — **always use the `.sh` wrapper over the
`.py` file directly**; running the `.py` file straight skips the ROS
environment and will crash with either a SIGSEGV in `gi`/GStreamer or an
`rmw_cyclonedds_cpp` load error):
```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///etc/cyclonedds.xml
```

### Terminal 1 — Command gateway (inside container `it`)

The single owner of this drone's FCU (arm/disarm/takeoff/land/video). Every
other process — including `rooster_dome_main.py` and `ui.py` — only ever
talks to it over `/R1/cmd_nav` / `/R1/rooster_status`.

```bash
docker exec -it it bash
source /opt/ros/foxy/setup.bash
source /home/rooster/workspace/install/setup.bash
export CYCLONEDDS_URI=file:///etc/cyclonedds.xml
cd /home/rooster
python3 -m sparx_agency.robots.ROBOTICAN.adapters.rooster_command_unit \
  --ros-args -p rooster_id:=R1 -p climb_z:=700.0 -p hover_z:=550.0 -p climb_duration_sec:=5.0
```

**Climb tuning (Sphera sim, confirmed 2026-07-13)**: the code defaults
(`climb_z=600`, `climb_duration_sec=3.0`) aren't enough thrust/time for this
drone's simulated mass to actually leave the ground in Sphera — it silently
"completes" the climb sequence (RoosterUnit's climb is open-loop, no
altitude feedback) while never actually gaining altitude. `climb_z=700`,
`climb_duration_sec=5.0` reliably gets it airborne; `hover_z=550` (the
existing default) then holds a stable hover once already airborne — values
between these (600, 450) either kept slowly climbing or sank. These are
**simulator-specific empirical values, not code defaults** (a real drone's
thrust characteristics likely differ) — pass them as `-p` overrides here,
don't bake them into `RoosterUnit`'s defaults.

Verify from the host: `ros2 topic echo /R1/rooster_status --once` should
print real `armed`/`airborne`/`battery_pct`/`video_on` values (not just
defaults) — if `ros2 topic info /R1/rooster_status` shows
`Publisher count: 0`, this isn't actually running.

### Terminal 2 — Frame publisher (host)

Decodes the drone's UDP/RTP-H264 stream, saves each frame at its native
resolution (no crop/resize) as a JPEG to `/tmp/rooster_frames`, publishes
each path on `/R1/rgb_frame_path`. Frames are always written to disk *and*
their path published — required for both this capture flow and any future
FALCON bridging.

```bash
./sparx_agency/robots/ROBOTICAN/run_rooster_frame_dir_publisher.sh \
  --rooster-id R1 \
  --out-dir /tmp/rooster_frames \
  --port 5001
```

`--port` must match `rooster_command_unit`'s `video_port` parameter (default
5001 on both sides). Nothing arrives here until something sends `video_on`
over `/R1/cmd_nav` (Terminal 4's "VIDEO STREAM" button, or Terminal 5).

### Terminal 3 — DA3 depth + AprilTag localization (host)

Reuses the exact same generic nodes XTEND uses, just repointed at
ROBOTICAN's topics, calibration, and a DA3 TensorRT engine sized/exported
for ROBOTICAN's own frame dimensions (running on the PC's GPU against the
Sphera simulator, not XTEND's Jetson-targeted 504x294 engine):

```bash
/home/user1/GIT/TheAgency/venv/bin/python \
  /home/user1/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
  --ros-args \
  -p frame_path_topic:=/R1/rgb_frame_path \
  -p depth_path_topic:=/R1/depth_frame_path \
  -p depth_dir:=/tmp/rooster_depth \
  -p engine_path:=/home/user1/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_546x364.engine \
  -p config_yaml:=/home/user1/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml \
  -p camera_info_mode:=base \
  -p model_type:=large_metric \
  -p depth_encoding:=32FC1
```

The engine's 546x364 input (a multiple of 14, the ViT patch size) has the
exact same 3:2 aspect ratio as the native 540x360 stream, so
`DA3TensorRTModel`'s internal resize is a clean uniform scale — no
crop/letterbox needed on the publisher side.

```bash
/home/user1/GIT/TheAgency/venv/bin/python -m sparx_agency.tasks.localization.ros2.localization_node \
  --ros-args \
  -p provider_type:=apriltag \
  -p frame_path_topic:=/R1/rgb_frame_path \
  -p camera_calib_path:=/home/user1/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml \
  -p tag_map_path:=/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/config/tag_map_path.yaml \
  -p tag_size_m:=0.13 \
  -r /xtend/localization:=/R1/localization -r /xtend/localization_source:=/R1/localization_source
```

`localization_node.py` always publishes on the fixed `/xtend/localization` /
`/xtend/localization_source` topic names internally — the `-r` remaps above
are required to get them onto ROBOTICAN's `/R1/...` namespace.

**Verify `tag_map_path.yaml` matches the physical/simulated AprilTag
placement before trusting localization output** — it can't be checked from
code.

### Terminal 4 — `ui.py` (host, optional but recommended for manual control)

```bash
./sparx_agency/robots/ROBOTICAN/run_ui.sh
```

Use ARM/TAKEOFF/LAND/DISARM and the **"VIDEO STREAM"** button freely — it
only sends `video_on`/`video_off` over `cmd_nav`. Do **not** click **"LOCAL
PREVIEW"** while capturing — it opens a second UDP listener on the same
video port as Terminal 2 and splits the stream between the two, so neither
gets a complete, reliable feed.

### Terminal 5 — Dome main (host)

Run once Terminal 3 is publishing localization. This sends its own
`video_on`/`video_off` (you don't need to click `ui.py`'s button first).

```bash
./sparx_agency/robots/ROBOTICAN/run_rooster_dome_main.sh \
  --rooster-id R1 \
  --pose-topic /R1/localization \
  --out-dir ~/rooster_dome_capture \
  --capture-interval-sec 1.0 \
  --yaw-bucket-deg 30.0
```

Arms, takes off, rotates 360° in 90° chunks (guided by `/R1/localization`,
falling back to a rough time-based blind turn if no pose arrives — tune
`--blind-turn-deg-per-sec` for your drone/room if you ever see that fallback
fire), captures frames, then lands, disarms, and turns the video stream back
off — with a `finally`-guaranteed land+disarm safety net on Ctrl-C or
SIGTERM.

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
/home/user1/GIT/TheAgency/venv/bin/python \
  sparx_agency/demos/Demo_No4_XTEND_MapRoom/room_mapper/run_room_mapper.py \
  --data-dir ~/rooster_dome_capture/latest \
  --labels /path/to/detections.json
```

## Useful checks

```bash
ros2 topic list                       # /R1/cmd_nav and /R1/rooster_status must
                                       # exist WITH a publisher (see Terminal 1)
ros2 topic info /R1/rooster_status    # Publisher count must be >= 1
ros2 topic hz /R1/rgb_frame_path
ros2 topic hz /R1/depth_frame_path
ros2 topic hz /R1/localization
ros2 topic echo /R1/rooster_status --once
```

The `Failed to parse type hash ... USER_DATA '(null)'` warnings from
`rmw_cyclonedds_cpp` are harmless discovery-time noise (seen whenever the
host's Jazzy DDS participant discovers Sphera's Foxy-built topics) — not an
error, safe to ignore.
