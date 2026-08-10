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
├── docker-compose.yml    # build profiles: falcon-pc (x86), falcon-jetson (aarch64)
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
    │                     #   nav_mode:=astar/object approach — see adapter/README_click_to_fly.md
    │                     #   nav_mode:=exploration — see adapter/README_exploration.md
    ├── scripts/          #   FALCON adapter nodes (rospy) — import the algorithms from core
    │   ├── falcon_adapter_node.py  # drone pose+depth -> FALCON topics + TF (core dead-reckoning + depth noise)
    │   ├── sensor_gate_node.py     # rotation-aware freezable pose+depth gate (core.mapping.depth_fusion_gate)
    │   ├── bev_publisher_node.py   # FALCON voxel clouds -> 2D OccupancyGrid (core.mapping.bev)
    │   ├── mapping_sync_node.py    # depth<->pose pairing + localization gate + authoritative rotation freeze (core.localization + core.mapping.depth_fusion_gate)
    │   ├── astar_planner_node.py   # 2D BEV -> smoothed waypoints (core.planning.planners.astar)
    │   ├── navdp_click_node.py     # click an RGB pixel -> NavDP point-goal policy -> world Path (core.planning.vlas.navdp); A* replacement
    │   ├── combination_planner_node.py # nav_mode:=combination — A* global route + NavDP local legs (farthest visible A* waypoint -> NavDP -> fly to midpoint -> re-infer); core.planning.vlas.navdp
    │   ├── hybrid_planner_node.py  # nav_mode:=hybrid (DEFAULT) — A* on easy legs, NavDP only for hard turns/doorways (core.planning.replanning.route_difficulty)
    │   ├── waypoint_follower_node.py # waypoints -> /cmd_vel, X+YAW only (core.planning.trackers.waypoint_follower)
    │   ├── bev_click_goal_node.py  # matplotlib BEV viewer + click-to-goal + the drone-thinking log panel
    │   ├── pose_adapter_node.py    # real-drone localization (PoseStamped/Odometry) -> bare Pose
    │   ├── sim_adapter_node.py     # Gazebo sjtu_drone -> XTEND topic/camera emulation (core.common.intrinsic_remap + wall-clock restamp)
    │   ├── object_approach_node.py # detections -> track + visual servo + SEARCH/SCAN/APPROACH/HOVER_LOCK/RECOVER (core.planning.visual_servo)
    │   ├── falcon_exploration_follower_node.py # nav_mode:=exploration — tracks traj_server's /planning/pos_cmd -> /cmd_vel_raw (core.planning.trackers.reference_tracker_3d)
    │   ├── target_lock_viewer_node.py # on-screen live target-lock HUD (subscribes the overlay Image)
    │   ├── cloud_utils.py          # PointCloud2 -> (N,3) helper (imported, not a node)
    │   └── thinking.py             # Thinker: narrate a node's decisions to /nav/thinking (imported, not a node)
    └── launch/
        ├── nav_stack.launch    # shared nav core (Gazebo sim)
        ├── real_drone.launch   # real drone — includes nav_stack.launch + pose/depth bridge
        ├── object_approach.launch          # detector + servo + HUD, ALONGSIDE a running nav stack
        └── real_drone_object_approach.launch # ONE launch: real_drone + object_approach + BEV window
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
- `core/planning/vlas/navdp` (point-goal geometry + NavDP HTTP client used by
  `navdp_click_node`)
- `core/mapping/tracking` + `core/planning/visual_servo` (the object-approach
  mission: acquisition gate, robust Median-Flow tracker (or detector-only, via
  `lock_mode`), visual servo, force shaping, scan sweep, re-search, and the mission
  state machine) — see [`OBJECT_APPROACH.md`](OBJECT_APPROACH.md)
- `core/common/intrinsic_remap` (resample a render to a target camera's
  intrinsics — sim_adapter uses it to hit the XTEND's anisotropic fx≠fy;
  `principal_point_crop` is the older crop-only special case)

`run_falcon.sh` mounts the repo read-only at `/opt/sparx_agency` with
`PYTHONPATH=/opt` so `import sparx_agency.core...` resolves; `cloud_utils` is
imported as a sibling. Adding a node = drop it in `scripts/`, list it in
`adapter/CMakeLists.txt`, and add its name to the mount loop in `run_falcon.sh`.
(A helper module the nodes *import* — `cloud_utils`, `pure_pursuit_follower`,
`thinking` — skips CMakeLists but **must** still be in the `run_falcon.sh` mount
loop, or every node importing it dies on `ImportError` inside the container.)

## Debugging the drone's thinking

Every nav node narrates its own decisions — why it stopped, which waypoint it is
flying at, what it is replanning around, which object it is homing on, why it
gave up — as first-person lines on one shared topic, `/nav/thinking`. The BEV
viewer opens a **second window, "drone thinking"**, with a rolling log of them —
separate from the map so the map keeps its whole canvas, and so the log can be
moved, resized or closed on its own (closing it leaves the map and the drone
running):

```
 +12.9s  waypoint_follower   Aligning to waypoint 1/4 (x=-1.20, y=-1.60)
 +15.6s  waypoint_follower   Flying forward to waypoint 1/4 (x=-1.20, y=-1.60)
 +21.3s  waypoint_follower   Reached waypoint 1, heading for waypoint 2
 +24.7s  astar_planner       Replanning: obstacle on the route
 +28.2s  object_approach     Lost the refrigerator from frame -- searching
 +31.0s  hybrid_planner      No route from A* and NavDP returned nothing -- I am stuck
```

Warnings are orange, unresolved decisions red, and a line thought repeatedly
collapses to `... (x3)` instead of flushing the reasoning that explains it off
the top. Watch it live without the GUI with
`rostopic echo /nav/thinking`.

To narrate from a new node:

```python
from thinking import Thinker
self.thinker = Thinker("my_node")          # in __init__, after rospy.init_node
self.thinker.say("Stopping to turn", category="nav")
```

`say()` is **edge-triggered**: it drops a line whose text has not changed since
the last call on the same slot, so calling it every control tick emits once per
decision. That is the whole point — narrate decisions, not telemetry. A line
whose numbers change every tick defeats the gate and buries the log; that belongs
in the viewer's HUD. See `scripts/thinking.py` for the full contract
(`key` slots, `repeat_after_s`, `forget()`), and `core/common/thought_message.py`
for the wire format.

Knobs: `~thinking:=false` silences one node, `~thinking_echo:=false` stops it
mirroring to rosout, and on the viewer `~thinking_lines` sets the panel height
while `~thinking_topic:=''` drops the panel and gives the whole window to the map.

## Build

x86_64 (with Gazebo + Open3D):

```bash
cd sparx_agency/tasks/planning/falcon
docker compose build falcon-pc      # -> image falcon-ros:noetic
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

### Image transport: frame-path vs topic

RGB/depth can arrive two ways; pick with `image_transport` (default
`frame_path`). The launch arg and the bridge config **must match**.

**1. frame-path (default — real XTEND):** the drone writes `.jpg`/`.npy` to disk
and publishes tiny `std_msgs/String` path messages the nodes load.

```bash
# container:
roslaunch falcon_adapter real_drone.launch map_name:=office
# host (bridge):
cd bridge && ./run_bridge.sh
```

**2. topic (Gazebo sim or old bag — raw images on the wire):**

```bash
# container:
roslaunch falcon_adapter real_drone.launch map_name:=office image_transport:=topic real_pose_topic:=/xtend/april_tag_pose
#   add real_pose_topic:=/xtend/april_tag_pose for bags recorded before the
#   /xtend/localization rename
# host (bridge):
cd bridge && BRIDGE_CFG=bridge_topic.yaml ./run_bridge.sh
```

## Visualizing & interacting (RViz, BEV, NavDP viewer)

These run against the live stack, so start the FALCON container first
(`./run_falcon.sh <env>`) and leave that shell open.

### Open RViz

Now in a new host terminal:

```bash
docker exec -it falcon bash
export DISPLAY=:0
source /catkin_ws/devel/setup.bash
roslaunch exploration_manager rviz.launch
```

This loads a pre-configured RViz with the BEV map, planned path, and odometry
already wired up.

### Open the 2D Map (BEV click-to-goal)

`real_drone.launch` **auto-starts** this viewer by default (`bev_viewer:=false` to run
headless), so you usually do not launch it by hand. It shows the map, the routes, and a
**system-status HUD** (top-left): `mode:` (the nav_mode) + `planner:` (who is driving
right now — `A*` or `NavDP (<reason>)`, from `/nav/status`), `motion:` (forward flight
vs rotation-in-place, from `/cmd_vel`), and the last **A\* replan event** (first route /
obstacle reroute / boxed-in STOP / shorter-route, from `/path/astar_event`). By default
only three overlays are drawn — **[a]** A\* global route (teal), **[1]** NavDP leg (blue),
**[4]** flown path (green) — so you can watch the hybrid hand-off cleanly; press keys
`1`-`9`/`a` to toggle any overlay, `0` for all.

To run it standalone instead (in another host terminal):

```bash
docker exec -it falcon bash
export DISPLAY=:0
source /catkin_ws/devel/setup.bash
rosrun falcon_adapter bev_click_goal_node.py
```
or
```bash
ssh -Y user@user-agx1
export XAUTHORITY=$HOME/.Xauthority
cp ~/.Xauthority /tmp/.docker.xauth && chmod 644 /tmp/.docker.xauth
XAUTHORITY=/tmp/.docker.xauth ./run_falcon.sh office
rosrun falcon_adapter bev_click_goal_node.py
```

A 2D map window opens. **Left-click** anywhere to publish a goal — A* replans and
the drone flies the new path. The red arrow marks the drone's live pose.

### NavDP click viewer (optional, standalone sanity check)

> Skip unless you're checking the NavDP path in isolation. This viewer doesn't
> fly the drone and doesn't replace the BEV goal flow. It only confirms that
> `click pixel → body-frame (gx, gy) → NavDP → trajectory` is wired up correctly.

Requires the NavDP HTTP server running on `127.0.0.1:8888` (override with
`_port:=`).

In another new host terminal:

```bash
docker exec -it falcon bash
export DISPLAY=:0
source /catkin_ws/devel/setup.bash
rosrun falcon_adapter navdp_click.py
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
# 1. Real XTEND (bring up the bridge so /xtend/rgb_frame_path + /xtend/depth_frame_path flow):
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
#    defaults already match — /xtend/rgb_frame_path + /xtend/depth_frame_path
#    (frame-path strings it loads from disk), raw-K 504x294 intrinsics, and pose
#    from /xtend/localization (PoseStamped):
rosrun falcon_adapter navdp_click_node.py
```

In the **NavDP click** window: LEFT-CLICK the RGB panel (a readout shows the goal
in metres), then ENTER/SPACE to send it and publish the path; `r` clears, `q`
quits.

> **Intrinsics must match the depth frames NavDP receives** — the SAME frames
> FALCON's mapping uses (now via the frame-path topics `/xtend/rgb_frame_path` +
> `/xtend/depth_frame_path`, tiny `std_msgs/String` "<path> <sec> <nsec>" messages
> the node loads from disk), so the `navdp_*` intrinsic args default to the shared
> `cam_*` (raw K on the real drone, `sim_adapter`'s P-target in sim). You normally
> pass nothing; override per camera, or use a `navdp_camera_info_topic` (K; must
> describe the live stream). Wrong intrinsics distort both the pixel→goal mapping
> and the on-image overlay. The loaded RGB and depth must share one resolution —
> the node fails loud otherwise.
>
> **Pose source** — the path is anchored at the drone pose. By default navdp reads
> `/xtend/localization` (`PoseStamped`), present in bag playback, on the real
> drone, and from `sim_adapter`; set `pose_type:=pose` to read a bare `Pose`
> (e.g. the nav stack's `/gt_pose` or Gazebo's `/simple_drone/gt_pose`). The
> pose's `z` is the altitude the overlay projects the trajectory onto.
>
> The node imports `cv2`, and at runtime `requests` + `Pillow` (lazy, only when it
> calls the server). Install them in the container if missing
> (`pip install requests pillow opencv-python`), as with `bev_click_goal`'s
> matplotlib.

## Combination mode (A* global route + NavDP local legs)

`nav_mode:=combination` is a third mode that **fuses** the two planners instead of
choosing one. A* plans the collision-free *global* route to the mission goal;
NavDP supplies the smooth *local* trajectory the drone actually flies, aimed at
the farthest point on the A* route it can currently **see**. The run can start on
A* and switch to the fusion on a signal, or be combined from the first frame.

`combination_planner_node` is the arbiter on `/path/waypoints_combo` (the raw path
`path_corrector` → `trajectory_simplifier` → `waypoint_follower` then process and
fly, exactly as for A*/NavDP). Once enabled it runs a **CRUISE → HOLD → FOLLOW**
state machine:

- **CRUISE** — fly the A* route while watching for a waypoint that is **visible**
  in the current frame (in front, in-frame, and by default not behind a wall) AND
  at least `combination_min_engage_fwd_m` (1.5 m) ahead — a goal worth a leg. None
  visible yet → keep flying A*. (Hysteresis: it takes `combination_engage_confirm_ticks`
  consecutive detections to commit, so depth flicker can't cause stop/go lurching.)
- **HOLD** — a good point appeared → **STOP** the drone (so the first inference is
  from a clean, stationary frame — important on a slow Jetson), settle, then ask
  NavDP for a leg. This is also the **"stop and wait"**: when a re-infer is slow and
  the previous leg has been flown out, the drone holds here for the new route, up to
  `combination_max_wait_s` (then it resumes A*).
- **FOLLOW** — fly the published leg; at its **midpoint** re-infer, with the leg's
  *second half* as a latency buffer. A fast reply → seamless switch; a slow one →
  the drone coasts to the leg end and holds (→ HOLD) until the route comes back.

If no waypoint is visible, or NavDP is unreachable, it falls back to **flying A\***
so the drone always has a route and never dead-stalls; on the final approach it
hands the last stretch to A* so it reaches the true goal.

```bash
# 0. Start the NavDP HTTP server on the GPU box (default 127.0.0.1:8888), as for
#    plain NavDP — combination_planner is just an HTTP client of it.
# 1. Real XTEND — combination is the DEFAULT, so a plain launch fuses from t=0:
roslaunch falcon_adapter real_drone.launch map_name:=office
# 2. Start on A*, switch to the fusion later by publishing the enable signal:
roslaunch falcon_adapter real_drone.launch map_name:=office combination_start_enabled:=false
rostopic pub -1 /combination/enable std_msgs/Bool "data: true"     # back to A*: data: false
# Other modes: nav_mode:=astar (plain A* only) | use_navdp:=true (operator NavDP click)
```

Key knobs (all `combination_*` args, forwarded by `real_drone.launch`):
`combination_start_enabled` (true — combined from the first frame; set false to
start on A* and wait for the enable signal), `combination_enable_topic`
(`/combination/enable`), `combination_min_engage_fwd_m` (1.5, the "reasonable
distance" to engage), `combination_engage_settle_s` (1.0, brake-settle for a clean
frame), `combination_leg_fraction` (0.5, the midpoint hand-off),
`combination_require_unoccluded` (true — visible means clear line-of-sight, not
merely in-FOV), `combination_final_handoff_m` (1.5). **Jetson timing:**
`combination_navdp_timeout_s` (10.0, one inference) and `combination_max_wait_s`
(30.0, how long to hold for a slow route before resuming A*) — raise both if your
Jetson inference is slower. Camera intrinsics, RGB/depth transport, pose source and
the NavDP server reuse the same `navdp_*` / `cam_*` args as NavDP click-to-go above.
The geometry and selection are ROS-free and unit-tested in `core.planning.vlas.navdp`
(`world_to_body_2d`, `select_farthest_visible_waypoint`, `arclength_fraction_2d`).

## Hybrid mode (A* on the easy legs, NavDP only for hard maneuvers) — the default

`nav_mode:=hybrid` is the **default**. It flies plain A* on straight, open stretches
and hands control to NavDP **only for a difficult maneuver** — a hard turn, an
S-bend, or threading a doorway / narrow gap — then takes it straight back once the
hard part is behind. This is the middle ground between `combination` (always fuses
NavDP) and `fallback` (only rescues when A* finds *no* route): `hybrid` engages on a
*geometric* difficulty A* solves cleanly but a stop-and-turn follower flies poorly.

`hybrid_planner_node` is the arbiter on `/path/waypoints_hybrid` (the raw path the
`path_corrector → trajectory_simplifier → waypoint_follower` chain flies). Two-mode
hysteretic state machine:

- **PRIMARY** — echo A* straight through. Each tick it assesses the route just
  *ahead* of the drone (`hybrid_difficulty_lookahead_m`, 3 m) for a **hard turn**
  (accumulated heading change ≥ `hybrid_turn_thresh_deg`, **75°**) or a **narrow
  passage** (free width < `hybrid_passage_width_m`, **0.75 m** — measured
  *perpendicular* to the route against the BEV, so it is tight on **both** sides like a
  real doorway, not a route that merely clips one corner). The assessment runs on the
  **smoothed** route, not the jagged raw A*: near-straight vertices are first merged
  (`hybrid_merge_collinear_deg`, 15°) so grid-staircase / line-of-sight jog noise on a
  straight corridor cannot sum into a false hard turn. Difficult on
  `hybrid_difficulty_confirm` (3) consecutive ticks → STOP and engage NavDP. (It also
  rescues a boxed-in A* like `fallback`, via `hybrid_engage_on_astar_fail`.)
- **ENGAGED** — drive NavDP toward the farthest A* waypoint **visible** in the current
  frame (the "furthest point on A* I can see"), fly each leg to its midpoint
  (`hybrid_leg_fraction`, 0.5), then re-infer. Return to A* once the route ahead is
  easy again **and** A* has a route, on `hybrid_recover_confirm` (5, sticky ≥ the
  engage confirm — anti-zig-zag) consecutive ticks.

```bash
# hybrid is the default, so a plain launch already runs it (NavDP server up):
roslaunch falcon_adapter real_drone.launch map_name:=office
# Switch modes: nav_mode:=astar (plain A*) | nav_mode:=combination | nav_mode:=fallback | use_navdp:=true
```

Key knobs (all `hybrid_*` args, forwarded by `real_drone.launch`):
`hybrid_turn_thresh_deg` (raise to hand off only sharper turns),
`hybrid_passage_width_m` (raise to treat wider gaps as doorways),
`hybrid_difficulty_lookahead_m`, `hybrid_difficulty_confirm` /
`hybrid_recover_confirm` (the engage/return hysteresis),
`hybrid_leg_fraction`, `hybrid_final_handoff_m`, and the Jetson timing
`hybrid_navdp_timeout_s` / `hybrid_max_wait_s`. The difficulty detection is
ROS-free and unit-tested in `core.planning.replanning.route_difficulty`
(`assess_route_difficulty`, `windowed_turn_deg`, `passage_free_width_2d`); the NavDP
leg reuses the same `navdp_*` / `cam_*` args and the `combination` leg engine.

## Object approach (hunt a named object while flying the route, then close on it)

Runs the whole nav stack **and** an open-vocabulary object hunt at once. The planner
flies to `goal_x, goal_y` while the TensorRT YOLO-World detector scans every frame in
the background. Once the target is confirmed for `n_confirm` consecutive frames the
`object_approach` node takes `/cmd_vel` (the follower goes passive via the
`visual_servoing` demo-mode hand-off, so there is exactly one publisher) and visually
servos onto the object until it is centred and very close. There is **no terminal
stop** — it keeps tracking, so a moving object is followed.

It runs as **two processes**: the nav stack + servo inside the FALCON ROS1 container,
and the TensorRT detector as a ROS2 sidecar **on the host** (see
[Why the detector runs on the host](#why-the-detector-runs-on-the-host)). They are
joined only by two `std_msgs/String` topics across the bridge — no image is bridged.

### One command (recommended)

`run_object_approach_mission.sh` starts all three pieces (detector sidecar → bridge →
container mission) in order, wires the **GPU** detector engine, and tears the two host
helpers down on exit. The container stays in the foreground so you keep the HUD and
can `Ctrl-C` the whole mission:

```bash
cd sparx_agency/tasks/planning/falcon
./run_object_approach_mission.sh office gun 0.0 -3.0
#                                 ^env   ^target ^goal_x ^goal_y
# NAV_MODE=astar for pure A* (default combination = A* route + NavDP legs).
# MODEL=l for a bigger detector (detect runs ~2 Hz, so a larger model is affordable).
# Any real_drone_object_approach.launch arg can be appended, e.g. closure_mode:=waypoint.
```

It fails loudly if the GPU engines or `.pt` text weights are missing, pointing you at
`build_all.sh <model>`. The three manual steps below are the same thing by hand.

### The three processes, by hand

**1. The detector**, on the Orin host, in the environment that has `tensorrt` +
`pycuda` (the one that built the engines and ran `object_approach_offline`). Point it
at the **GPU** backbone engine (the DLA is measured slower — see the yolo_world_trt
README):

```bash
source /opt/ros/humble/setup.bash
cd /path/to/repo    # the dir CONTAINING sparx_agency/
PYTHONPATH=$PWD python3 sparx_agency/tasks/mapping/ros2/yolo_detector_ros2_node.py \
    --ros-args -p target_object:=gun \
      -p backbone_engine:=/path/yolo_world_s.backbone.fp16.gpu.engine \
      -p head_engine:=/path/yolo_world_s.head.fp16.gpu.engine \
      -p text_weights:=/path/yolov8s-worldv2.pt
```

**2. The bridge** (it carries `/object_approach/detections` and `/object_approach/goal`):

```bash
cd bridge && ./run_bridge.sh
```

**3. The nav stack + mission**, in the container. This starts map + BEV + A*/NavDP +
follower + servo + the live target-lock HUD + the clickable BEV window:

```bash
./run_falcon.sh office
# inside the container:
source /catkin_ws/devel/setup.bash
export DISPLAY=:0
roslaunch falcon_adapter real_drone_object_approach.launch \
map_name:=office \
target_object:=gun \
goal_x:=0.0 \
goal_y:=-3.0 \
viewer:=false \
bev_viewer:=true \
publish_overlay:=false
```

To add the mission to a nav stack that is **already running**, launch just the
mission half in a second shell (`docker exec -it falcon bash` first):

```bash
roslaunch falcon_adapter object_approach.launch target_object:=monitor
```

Both launches consume the sidecar's detections over the bridge; **no** detector runs
in the container (the FALCON image has no CUDA/TensorRT/pycuda). The detector always
runs as the host-side ROS2 sidecar.

### Why the detector runs on the host

The FALCON Jetson image is built `FROM ros:noetic-perception`: a stock Ubuntu 20.04
ROS image with **no CUDA, no TensorRT, no pycuda**. `--runtime nvidia` bind-mounts
JetPack's *shared libraries* into the container but not the Python bindings, so an
in-container detector dies immediately on `import pycuda.driver`. And `pip install
pycuda` there cannot fix it — pycuda compiles against `nvcc` and the CUDA headers,
which the mounts don't provide.

The Orin host already has a working `tensorrt` + `pycuda` Python environment. So the
detector runs there, and only two String topics cross the bridge:

```
ROS2 (host, GPU)                          bridge          ROS1 (falcon container)
/xtend/rgb_frame_path ─► yolo_detector ─► /object_approach/detections ─► object_approach
                                     ◄─── /object_approach/goal ◄─── operator / mission
```

This costs nothing in bandwidth. On the real drone RGB arrives as a *frame path*, so
the sidecar reads `/xtend/rgb_frame_path` natively on the ROS2 side — **upstream of
the bridge** — and loads the JPEG straight off the host's disk. The detections JSON is
a few hundred bytes. The two nodes were always decoupled by that topic
(`core/common/detection_message.py` is the one definition of the wire format), which
is exactly what makes this split free.

> The sidecar needs the repo importable: run it from the directory *containing*
> `sparx_agency/`, with that directory on `PYTHONPATH`. Its parameters mirror the
> ROS1 node's rosparams — see the footer of `yolo_detector_ros2_node.py`.
>
> **QoS matters.** The sidecar subscribes to the frame-path topic as `BEST_EFFORT` to
> match the drone's publisher. A reliability mismatch here means *no data flows at
> all* — the same trap `bridge/bridge.yaml` documents for depth.

Live control, at any time:

```bash
# Switch the hunted object live -- re-prompts the detector AND re-keys the
# closure's confirmation gate (one topic, both sides). From the HOST use the
# helper (it sets the RELIABLE + TRANSIENT_LOCAL QoS the subscribers require;
# a plain `ros2 topic pub` is VOLATILE and is dropped silently):
./retarget_object.sh knife
# ...or, from INSIDE the container (ROS1, no QoS flags needed):
rostopic pub -1 /object_approach/goal   std_msgs/String "data: 'knife'"

rostopic pub -1 /object_approach/enable std_msgs/Bool   "data: false"   # disarm -> planner keeps the route
rostopic echo /object_approach/status                                   # state, streak, range, at_target
```

> **Why a helper for the switch.** `/object_approach/goal` is subscribed as
> `TRANSIENT_LOCAL` by both the host detector and the bridge. A default
> `ros2 topic pub` is `VOLATILE` → durability-incompatible → the message reaches
> neither and the target never changes, with no error. `retarget_object.sh`
> publishes with the matching QoS so the switch actually lands.

### Prerequisite: build the TensorRT engines on the target

The detector is **TensorRT only** (no ultralytics `.pt` inference path). Engines are
not portable — build them on the Jetson that will fly:

```bash
export WEIGHTS_DIR=/path/to/yolo_world_weights     # holds yolov8s-worldv2.pt
sparx_agency/tasks/mapping/yolo_world_trt/build_all.sh s
```

Then point the **sidecar** at them (`-p backbone_engine:=… -p head_engine:=…`, step 1
above). They are host paths, not container paths — the detector runs only as the
host-side sidecar, so no engine paths appear in the container launch at all.

`text_weights` (the `.pt` checkpoint) is only touched to embed the *prompt* — the
CLIP text branch — so torch runs once per retarget, never per frame. Inference itself
is torch-free. See [`yolo_world_trt/README.md`](../../mapping/yolo_world_trt/README.md).

### What happens when things go wrong (by design)

- **Target lost mid-approach** → the node yaws toward the side the object left and
  re-searches. If it cannot re-acquire within `recover_timeout_s` (6 s) it gives up,
  hands `/cmd_vel` back to the follower, and **re-publishes the last/initial goal** on
  `/waypoint_nav/goal` so the planner resumes the route instead of stalling.
- **Route reaches the goal, object never seen** → the node sweeps the room in place:
  a slow rotate broken by **stops** (`pause → rotate → pause …`), starting with a pause
  so the detector gets a clean, motion-blur-free look down each new bearing before the
  drone turns again. The sweep runs until the object is found or the goal changes.
- **Object found, then it moves** → `HOVER_LOCK` falls back to `APPROACH`. There is no
  success-and-stop state; "centred and close" is a condition, not a terminus.

### Closure version + minimum force

The platform needs a **minimum force per axis** to move at all, so every published
command (servo, re-search, scan, brake) is force-shaped as the final stage before the
wire. `force_mode:=fixed` (the default) is bang-bang: below the deadband the axis is
exactly `0`, above it the axis is driven at exactly the fixed level. That means the
drone approaches at exactly `min_vx` and turns at exactly `min_wz_deg` regardless of
how large the tracking error is.

- `force_mode:=snap` keeps the same minimum-force floor and max clamp but lets the
  command be **proportional** in between — use it if your platform can modulate above
  the floor. `force_mode:=none` disables the floor entirely (analog, max-clamped only).
- Raise `fixed_vx` / `fixed_vy` / `fixed_wz_deg` above the minimums to fly a faster
  bang-bang without leaving `fixed`.

`lock_mode` picks how the box is kept on the target: `detector_tracker` (default) —
the detector seeds the robust Median-Flow tracker, propagated every frame between
detections; or `detector` — the detector's box alone (held for `max_det_age_s`), no
tracking, for when the detector already keeps up with the RGB stream.

`closure_mode` picks how the servo drives the axes, independently of the route
follower's own `controller`:

- `multi_axis` (default) — holonomic: `vx` + `vy` crab + yaw together. Use when the
  platform accepts lateral velocity.
- `waypoint` — yaw **XOR** forward, no crab. Use to match the one-axis `waypoint`
  route follower.

Key knobs: `target_object`, `conf_thresh` (0.40 — the detector's min class confidence,
set on the sidecar (`-p conf_thresh:=`); the offline tools' `--conf`; raise it to
suppress false locks, lower it to acquire sooner), `n_confirm` (3), `min_score` (0.30),
`target_range_m` (0.8,
the hover-lock standoff), `arrive_radius_m` (0.6, when "arrived" triggers the scan),
`scan_yaw_rate` / `scan_rotate_s` / `scan_pause_s`, `recover_timeout_s`, `detect_hz`
(2.0 — the tracker runs at full camera rate between detections). Full rosparam lists
are in the footers of `object_approach_node.py` and the sidecar
`yolo_detector_ros2_node.py`.

### The live target-lock HUD

On by default (`viewer:=true`). `object_approach_node` renders the overlay from the
**actual mission state** — detections, the tracked box, the FSM mode, and the exact
*shaped* command being published — and publishes it as a `sensor_msgs/Image` on
`/object_approach/overlay`; `target_lock_viewer_node` just displays it. It is the same
renderer as the offline `run_live_target_lock` tool, so what you validated offline is
what you see in flight.

Keeping it a separate node means the GUI dependency and the display loop stay off the
control node: run headless with `viewer:=false` (the overlay Image is still published,
so you can view it remotely), or turn the rendering off entirely with
`publish_overlay:=false`. It needs a display — run on the Jetson's own screen, or
`ssh -Y` / VNC in, exactly as for `bev_click_goal` (see the BEV section above). Press
`q` in the window to close it; the mission keeps running.

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

### ROS2 bridge — only if your drone publishes on ROS2

Skip this whole section if your drone already publishes on ROS1.

**Build (one-time):**

```bash
cd ros_bridge_docker
docker build -t ros1_bridge:noetic-foxy .
```

Or just run `./run_bridge.sh` — it auto-builds if the image is missing.

**Start roscore** (inside the bridge container):

```bash
docker run -d --rm --net=host --name=roscore \
  --entrypoint bash ros1_bridge:noetic-foxy -c \
  "source /opt/ros/noetic/setup.bash && roscore"
```

**Start the bridge:**

```bash
cd ros_bridge_docker
./run_bridge.sh
```

Verify it's up — these should appear in `rostopic list`:

```
/flow_depth/pose_est
/xtend/depth_frame_path
```

> Topics only appear on the ROS1 side once a ROS1 subscriber asks for them
> (`dynamic_bridge` is lazy). If `rostopic list` looks empty before FALCON is up,
> that's normal — start FALCON and they'll show up.

## Notes

- The container needs the repo importable: `run_falcon.sh` mounts
  `…/sparx_agency` at `/opt/sparx_agency` and sets `PYTHONPATH=/opt` (ROS's
  setup.bash then prepends its own paths). Override the host location with
  `SPARX_PARENT=/path/that/contains/sparx_agency ./run_falcon.sh office`.
- Only the launch files we use are kept (`nav_stack.launch`, `real_drone.launch`,
  `object_approach.launch`, `real_drone_object_approach.launch`); the original
  `gazebo_exploration.launch`, `playback_exploration.launch`, and
  `visual_servoing.launch` were dropped.
- Adapter node scripts must be **executable** (`chmod +x`) — `roslaunch` refuses to
  start a `type=` script without the bit set, and `run_falcon.sh` bind-mounts them
  from the host, so the host's permissions are what the container sees.
- The image has **no CUDA/TensorRT/pycuda** (it is `FROM ros:noetic-*`, not an L4T
  base). `--runtime nvidia` mounts JetPack's shared libraries but not the Python
  bindings. Anything needing GPU inference therefore runs on the host and talks to
  the container over the bridge — as the object-approach detector does.
- The import chain for the nodes (`core.mapping.bev`,
  `core.planning.planners.astar`, `core.planning.trackers.waypoint_follower`,
  `core.localization`) is numpy-only (the follower is pure-stdlib) — no
  torch/scipy/OMPL/ROS pulled in at import time — so they load in Noetic.