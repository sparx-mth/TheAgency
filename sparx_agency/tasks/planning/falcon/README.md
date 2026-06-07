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
    │   ├── sensor_gate_node.py     # freezable pose+depth pass-through (core.planning.sensor_freeze_policy)
    │   ├── bev_publisher_node.py   # FALCON voxel clouds -> 2D OccupancyGrid (core.mapping.bev)
    │   ├── mapping_sync_node.py    # depth<->pose pairing + gate (core.localization)
    │   ├── astar_planner_node.py   # 2D BEV -> smoothed waypoints (core.planning.planners.astar)
    │   ├── waypoint_follower_node.py # waypoints -> /cmd_vel, X+YAW only (core.planning.trackers.waypoint_follower)
    │   ├── bev_click_goal_node.py  # matplotlib BEV viewer + click-to-goal
    │   ├── pose_adapter_node.py    # real-drone localization (PoseStamped/Odometry) -> bare Pose
    │   ├── sim_adapter_node.py     # Gazebo sjtu_drone -> XTEND topic/camera emulation (core.common crop)
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
- `core/localization` (incl. `se3`, `temporal_transform_buffer`,
  `dead_reckoning_noise`)
- `core/planning/planners/astar`, `core/planning/trackers/waypoint_follower`,
  `core/planning/sensor_freeze_policy` (the planner's "don't fuse the map
  while rotating" decision)
- `core/common/principal_point_crop` (a general image utility)

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
