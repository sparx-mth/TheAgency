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
└── adapter/              # the falcon_adapter catkin package (the FALCON task's ROS1 nodes)
    ├── scripts/          #   FALCON adapter nodes (rospy) — import the algorithms from core
    │   ├── bev_publisher_node.py   # FALCON voxel clouds -> 2D OccupancyGrid (core.mapping.bev)
    │   ├── mapping_sync_node.py    # depth<->pose pairing + gate (core.localization)
    │   ├── bev_click_goal_node.py  # matplotlib BEV viewer + click-to-goal
    │   └── cloud_utils.py          # PointCloud2 -> (N,3) helper (imported, not a node)
    └── launch/
        ├── nav_stack.launch    # shared nav core (Gazebo sim)
        └── real_drone.launch   # real drone — includes nav_stack.launch + pose/depth bridge
```

These nodes are **FALCON-specific adapters** (FALCON topics, `/map_config`,
frame conventions), so they live with FALCON. The **reusable, ROS-free**
algorithms they call live in `core/` (`core/mapping/bev`, `core/localization`).
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

## Notes

- The container needs the repo importable: `run_falcon.sh` mounts
  `…/sparx_agency` at `/opt/sparx_agency` and sets `PYTHONPATH=/opt` (ROS's
  setup.bash then prepends its own paths). Override the host location with
  `SPARX_PARENT=/path/that/contains/sparx_agency ./run_falcon.sh office`.
- Only the two launch files we use are kept (`nav_stack.launch`, `real_drone.launch`);
  the original `gazebo_exploration.launch`, `playback_exploration.launch`, and
  `visual_servoing.launch` were dropped.
- The import chain for the nodes (`core.mapping.bev`, `core.localization`) is
  numpy-only — no torch/scipy/ROS in any `__init__.py` — so they load in Noetic.
