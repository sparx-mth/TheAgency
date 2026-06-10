# FALCON Docker

Self-contained Docker build that runs the [FALCON](https://github.com/HKUST-Aerial-Robotics/FALCON)
aerial exploration / planning stack (ROS 1 Noetic) plus our `falcon_adapter` package.

The single purpose of this folder is to **build the FALCON image and run it** — on a
PC (x86_64 + Gazebo) or on a Jetson (aarch64, real drone). The same image serves both;
the launch file you run inside the container picks which.

> FALCON itself is **not** vendored here — the Dockerfile clones it
> (`HKUST-Aerial-Robotics/FALCON`, branch `ros1-noetic`) at build time and then applies
> the local patches in `patches/`.

## Layout

```
falcon/
├── Dockerfile            # builds the FALCON + adapter image
├── docker-compose.yml    # build profiles: falcon-hospital (x86), falcon-jetson (aarch64)
├── entrypoint.sh         # sources ROS + the catkin workspace
├── run_falcon.sh         # runs the container for any env: ./run_falcon.sh <env>
├── maps/                 # map configs, selected by name at run time
│   ├── office.yaml       #   real-drone environment
│   ├── hospital.yaml     #   baked into the image (the build needs one map present)
│   └── bookstore.yaml, playground.yaml, small_house.yaml, small_warehouse.yaml
├── patches/              # post-clone fixes applied to FALCON during the build
│   ├── fix_falcon_system_info.sh   # nvidia-smi parse crash on Jetson
│   ├── fix_falcon_cost_check.sh    # glog FATAL on tiny A* cost
│   ├── fix_falcon_depth_overflow.sh# depthToPointcloud buffer overflow
│   ├── fix_falcon_sop.sh           # SOP timeout 1s -> 10s (rebuild)
│   └── ignore_cuda_pkgs.sh         # CATKIN_IGNORE CUDA / sim-only packages
├── bridge/              # ROS1<->ROS2 bridge (parameter_bridge, QoS-aware) — see bridge/README.md
│   ├── Dockerfile           #   builds ros1_bridge:noetic-foxy
│   ├── bridge.yaml          #   bridged topics + per-topic QoS (sensor streams = best_effort)
│   ├── run_bridge.sh        #   build-if-missing + run
│   └── verify_bridge.sh     #   topic-flow health check
└── adapter/              # the falcon_adapter catkin package (the FALCON task's ROS1 nodes)
    ├── scripts/          #   FALCON adapter nodes (rospy) — import the algorithms from core
    │   ├── falcon_adapter_node.py  # drone pose+depth -> FALCON topics + TF (core dead-reckoning + depth noise)
    │   ├── sensor_gate_node.py     # rotation-aware freezable pose+depth gate (core.mapping.depth_fusion_gate)
    │   ├── bev_publisher_node.py   # FALCON voxel clouds -> 2D OccupancyGrid (core.mapping.bev)
    │   ├── mapping_sync_node.py    # depth<->pose pairing + localization gate + authoritative rotation freeze (core.localization + core.mapping.depth_fusion_gate)
    │   ├── astar_planner_node.py   # 2D BEV -> smoothed waypoints (core.planning.planners.astar)
    │   ├── navdp_click_node.py     # click an RGB pixel -> NavDP point-goal policy -> world Path (core.planning.navdp); A* replacement
    │   ├── waypoint_follower_node.py # waypoints -> /cmd_vel, X+YAW only (core.planning.trackers.waypoint_follower)
    │   ├── bev_click_goal_node.py  # matplotlib BEV viewer + click-to-goal
    │   ├── pose_adapter_node.py    # real-drone localization (PoseStamped/Odometry) -> bare Pose
    │   ├── sim_adapter_node.py     # Gazebo sjtu_drone -> XTEND topic/camera emulation (core.common.intrinsic_remap + wall-clock restamp)
    │   └── cloud_utils.py          # PointCloud2 -> (N,3) helper (imported, not a node)
    └── launch/
        ├── nav_stack.launch    # shared nav core (Gazebo sim)
        └── real_drone.launch   # real drone — includes nav_stack.launch + pose/depth bridge
```

These nodes are **FALCON-specific adapters** (FALCON topics, `/map_config`,
frame conventions), so the thin ROS1 entrypoints live with FALCON (they are the
only thing the Docker build context can bake into the image). The **pure,
ROS-free algorithms** they call live in `core/`, each in its own domain:

- `core/mapping/bev`, `core/mapping/depth_noise`
- `core/mapping/sensor_freeze_policy` + `core/mapping/depth_fusion_gate`
  (the mapping "don't fuse the map while rotating, and drop the stale in-flight
  frame on resume" decision). Applied at the node that forms the (depth, pose)
  pair fed to the voxels: `mapping_sync_node` is the AUTHORITATIVE voxel freeze
  on the real-drone path (it pairs gated depth with an un-gated pose, so the
  freeze must live there, in the capture clock); `sensor_gate_node` applies the
  same gate to the pure-Gazebo `falcon_adapter` path (where pose+depth are both
  gated and co-frozen).
- `core/localization` (incl. `se3`, `temporal_transform_buffer`,
  `dead_reckoning_noise`)
- `core/planning/planners/astar`, `core/planning/trackers/waypoint_follower`
- `core/planning/navdp` (point-goal geometry + NavDP HTTP client used by
  `navdp_click_node`)
- `core/common/intrinsic_remap` (resample a render to a target camera's
  intrinsics — sim_adapter uses it to hit the XTEND's anisotropic fx≠fy;
  `principal_point_crop` is the older crop-only special case)

`run_falcon.sh` mounts the repo read-only at `/opt/sparx_agency` with
`PYTHONPATH=/opt` so `import sparx_agency.core...` resolves; `cloud_utils` is
imported as a sibling. Adding a node = drop it in `scripts/`, list it in
`adapter/CMakeLists.txt`, and add its name to the mount loop in `run_falcon.sh`.

## Build

x86_64 (with Gazebo + Open3D):

```bash
cd sparx_agency/tasks/planning/falcon
docker compose build falcon-hospital      # -> image falcon-ros:noetic
```

Jetson AGX (real drone only, no sim / Open3D):

```bash
docker compose build falcon-jetson        # -> image falcon-ros:jetson
```

The Dockerfile is identical for both; only `BASE_IMAGE` and `WITH_SIM` differ
(see `docker-compose.yml`). `WITH_SIM=1` builds Open3D + the simulator packages;
`WITH_SIM=0` skips them and `CATKIN_IGNORE`s the sim-only / CUDA packages.

## Run

```bash
./run_falcon.sh <env> [docker CMD ...]
```

`<env>` selects `maps/<env>.yaml` and mounts it into the container. With no extra
command it drops you into a bash shell. The script auto-detects the arch, picks the
matching image tag, wires up the NVIDIA GPU flag, and mounts the adapter `scripts/`
and `launch/` so host-side edits take effect without a rebuild.

Example — real drone on the `office` map:

```bash
./run_falcon.sh office
# inside the container:
source /catkin_ws/devel/setup.bash
roslaunch falcon_adapter real_drone.launch map_name:=office
```

Example — Gazebo sim:

```bash
./run_falcon.sh hospital
# inside the container:
roslaunch falcon_adapter nav_stack.launch map_name:=hospital
```

## NavDP click-to-go (replacing A*)

[NavDP](https://github.com/InternRobotics/NavDP) is a point-goal navigation
policy. With `use_navdp:=true`, `navdp_click_node` runs **instead of**
`astar_planner`: instead of A* searching the BEV grid to a clicked *map* goal, you
click a pixel in the live **camera** image, NavDP returns a body-frame trajectory,
and the node anchors it at the current pose and publishes it as a world-frame
`nav_msgs/Path` on the same `/path/waypoints` topic. So `waypoint_follower` flies
it and `bev_click_goal` draws it on the BEV — both unchanged. A new inference runs
only when you click again and press ENTER; until then the follower keeps flying
the last published path.

```bash
# 0. Start the NavDP HTTP server on the GPU box (default 127.0.0.1:8888) — see
#    the NavDP repo's eval_*_wheeled.py. navdp_click_node only POSTs to it.
# 1. Real XTEND (bring up the bridge so /xtend/rgb + /xtend/depth_m flow):
roslaunch falcon_adapter real_drone.launch map_name:=office use_navdp:=true
# or, to auto-sync intrinsics from the bridged camera_info:
roslaunch falcon_adapter real_drone.launch use_navdp:=true \
    navdp_camera_info_topic:=/xtend/camera_info
# 2. Gazebo sim, raw camera (no sim_adapter) — point it at the sim camera,
#    intrinsics, and the bare-Pose ground truth:
roslaunch falcon_adapter nav_stack.launch use_navdp:=true \
    navdp_rgb_topic:=/simple_drone/front/image_raw \
    navdp_depth_topic:=/simple_drone/front_depth/depth/image_raw \
    navdp_pose_topic:=/simple_drone/gt_pose navdp_pose_type:=pose \
    navdp_fx:=320.0 navdp_fy:=320.0 navdp_cx:=320.5 navdp_cy:=240.5 \
    navdp_img_width:=640 navdp_img_height:=480
# 3. Standalone sidecar (e.g. rosbag playback through the bridge): the node
#    defaults already match — /xtend/rgb + /xtend/depth_m, raw-K 504x294
#    intrinsics, and pose from /xtend/april_tag_pose (PoseStamped):
rosrun falcon_adapter navdp_click_node.py
```

In the **NavDP click** window: LEFT-CLICK the RGB panel (a readout shows the goal
in metres), then ENTER/SPACE to send it and publish the path; `r` clears, `q`
quits.

> **Intrinsics must match the `/xtend/depth_m` stream NavDP receives** — the
> SAME stream FALCON's mapping uses, so the `navdp_*` intrinsic args default to
> the shared `cam_*` (raw K on the real drone, `sim_adapter`'s P-target in sim).
> You normally pass nothing; override per camera, or use a
> `navdp_camera_info_topic` (K; must describe the live stream). Wrong intrinsics
> distort both the pixel→goal mapping and the on-image overlay. `/xtend/rgb` and
> `/xtend/depth_m` must share one resolution — the node fails loud otherwise.
>
> **Pose source** — the path is anchored at the drone pose. By default navdp reads
> `/xtend/april_tag_pose` (`PoseStamped`), present in bag playback, on the real
> drone, and from `sim_adapter`; set `pose_type:=pose` to read a bare `Pose`
> (e.g. the nav stack's `/gt_pose` or Gazebo's `/simple_drone/gt_pose`). The
> pose's `z` is the altitude the overlay projects the trajectory onto.
>
> The node imports `cv2`, and at runtime `requests` + `Pillow` (lazy, only when it
> calls the server). Install them in the container if missing
> (`pip install requests pillow opencv-python`), as with `bev_click_goal`'s
> matplotlib.

## Talking to a ROS2 sim / drone — the bridge

FALCON is ROS1; the SJTU Gazebo sim and the real XTEND are ROS2. The
`bridge/` subdir holds the ROS1↔ROS2 bridge that connects them — lift it
alongside the FALCON container:

```bash
./run_falcon.sh hospital          # ROS1 FALCON (its roslaunch provides roscore)
bridge/run_bridge.sh              # ROS1<->ROS2 bridge (separate terminal)
# then start the ROS2 sim / power the drone, and: bridge/verify_bridge.sh
```

The bridge does **all** ROS1↔ROS2 message passing (the adapters never bridge);
its `bridge.yaml` is pinned to exactly this stack's topics. See
[`bridge/README.md`](bridge/README.md).

## Notes

- The container needs the repo importable: `run_falcon.sh` mounts
  `…/sparx_agency` at `/opt/sparx_agency` and sets `PYTHONPATH=/opt` (ROS's
  setup.bash then prepends its own paths). Override the host location with
  `SPARX_PARENT=/path/that/contains/sparx_agency ./run_falcon.sh office`.
- Only the two launch files we use are kept (`nav_stack.launch`, `real_drone.launch`);
  the original `gazebo_exploration.launch`, `playback_exploration.launch`, and
  `visual_servoing.launch` were dropped.
- The import chain for the nodes (`core.mapping.bev`,
  `core.planning.planners.astar`, `core.planning.trackers.waypoint_follower`,
  `core.localization`) is numpy-only (the follower is pure-stdlib) — no
  torch/scipy/OMPL/ROS pulled in at import time — so they load in Noetic.
