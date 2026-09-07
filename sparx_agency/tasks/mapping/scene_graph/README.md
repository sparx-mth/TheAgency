# Scene graph — LLM-guided object search over a live FALCON exploration

FALCON autonomously explores the hospital and its voxel map is projected to a
2D BEV grid. On the host, that grid is segmented online into **rooms** by
watershedding its clearance field and forcing a boundary at every known
door (`segmentation:=doors` reverts to the flown skeleton cut); YOLO-World
(open vocabulary, GPU) reveals **objects**, which are back-projected through
the depth camera into world-frame landmarks and assigned to rooms. A small
local **LLM** (ollama, CPU) then does the two things a geometric stack cannot:
it names each room from its objects ("this is a ward"), and it ranks the rooms
by the probability that the **search target** ("wheelchair") is in each —
folding in what has already been searched and how much frontier is left. A
**target watcher** latches `/target_seen` the moment a confirmed landmark
matches the target. A live top-down **viz** node draws the whole thing.

Everything below composes wave-1 core modules — read them before changing
behavior here:

- `sparx_agency/core/mapping/topology/` — room segmentation
  (`segment_rooms_watershed`, and the flown `compute_rooms`),
  persistent `RoomRegistry`, door discovery/linking, frontier counting,
  `LLMClient` / `RoomTypeClassifier` / `SearchOracle` / `TargetMatcher`
- `sparx_agency/core/mapping/objects/` — `geometry` (bbox rescale, robust
  depth, back-projection) and `landmarks` (`ObjectLandmarkMap` dedupe)
- `sparx_agency/core/mapping/bev/projector.py` — the BEV grid contract
  (`UNKNOWN=-1, FREE=0, OCCUPIED=100`)
- `sparx_agency/robots/SJTU/maps/hospital_doors.yaml` — the surveyed hospital
  door positions (hospital-only; every other world runs door-less)

## The four runtimes

One 8 GB GPU, and it is **exclusive** (see GPU discipline below). Four
runtimes cooperate, and every arrow that crosses a box is a topic or an HTTP
call:

```
+---------------------- host machine ---------------------------------------+
|                                                                           |
|  +-- docker: sjtu_drone_<world> (ROS2 Humble, CPU/llvmpipe) ------------+ |
|  |  Gazebo Classic: hospital world + SJTU drone                         | |
|  |    /simple_drone/front/image_raw          RGB 600x600 @60Hz          | |
|  |    /simple_drone/front_depth/depth/image_raw  32FC1 m @15Hz          | |
|  |    /simple_drone/odom, /clock, camera_infos, bumper                  | |
|  +---------------+---------------------------------+--------------------+ |
|                  | DDS (CycloneDDS, no SHM,        |                      |
|                  |  ROS_DOMAIN_ID=20)              |                      |
|  +-- docker: sjtu-ros1-bridge -------+   +-- docker: falcon-sjtu ------+  |
|  | ros1_bridge (Foxy<->Noetic)       |<->| FALCON planner (ROS1) +     |  |
|  | bridges depth/odom/clock/cmd/     |   | adapter nodes + BEV         |  |
|  | bumper DOWN, cmd_vel/takeoff/     |   | publisher (enable_bev)      |  |
|  | /falcon/bev_2d UP (latched)       |   | (falcon-rviz beside it)     |  |
|  +---------------+-------------------+   +-----------------------------+  |
|                  | /falcon/bev_2d (OccupancyGrid, transient_local)        |
|                  v                                                        |
|  +-- host: ROS2 jazzy + .venv (CPU, CUDA_VISIBLE_DEVICES="") -----------+ |
|  |                                                                      | |
|  |  detector_client_node --RGB jpeg--> [HTTP :8092] --detections-->     | |
|  |     -> /perception/detections                                        | |
|  |  object_mapper_node: detections + depth + odom -> world landmarks    | |
|  |     -> /perception/objects (latched)                                 | |
|  |  semantic_mapper_node: BEV -> rooms/doors/frontiers                  | |
|  |     -> /scene_graph + /scene_graph/room_labels_grid (latched)        | |
|  |  room_classifier_node: objects-per-room --LLM--> labels              | |
|  |     -> /semantic_mapper/room_labels (latched)                        | |
|  |  llm_oracle_node: rooms+labels+search state --LLM--> P(target|room)  | |
|  |     -> /llm_oracle/probabilities (latched)                           | |
|  |  target_watcher_node: objects x target --LLM match-->                | |
|  |     -> /target_seen (Bool) + /target_seen/info (latched)             | |
|  |  scene_graph_viz_node: draws everything -> <out>/viz/*.png           | |
|  +------------+------------------------------------+--------------------+ |
|               | HTTP POST /detect (jpeg)            | HTTP /api/chat       |
|               v                                     v                      |
|  +-- conda navdp (GPU cuda:0) ---------+  +-- docker: ollama (CPU) ------+ |
|  | detection_server.py                 |  | ollama-scene-graph           | |
|  | YOLO-World yolov8s-worldv2          |  | qwen2.5:3b-instruct          | |
|  | :8092 /health /detect /set_classes  |  | 127.0.0.1:11434              | |
|  +-------------------------------------+  +------------------------------+ |
+---------------------------------------------------------------------------+
```

The world frame is the `/simple_drone/odom` frame; all positions are ENU
metres. The front cameras sit 20 cm ahead of the body origin
(`robots/SJTU/adapters/topics.FRONT_CAMERA_OFFSET_FLU`).

## Runbook

**One prerequisite that is not optional on the FALCON path**: the host needs
CycloneDDS. The ROS1 bridge speaks only CycloneDDS (Foxy's Fast DDS 2.1 is not
wire-compatible with Humble's 2.6), so on a Fast-DDS-only box the bridge comes
up, reads as healthy, and carries **zero** depth, odom and BEV for the whole
run. `run_scene_graph.sh` refuses to start rather than fly that:

```bash
sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp   # once per machine
```

Then:

```bash
export SJTU_PROJECT_DIR=$HOME/GIT/sjtu_project   # branch: see below
export DISPLAY=:1                                # Gazebo cameras die silently without X
bash sparx_agency/tasks/mapping/scene_graph/scripts/run_scene_graph.sh
```

That one command runs, in order, each step gated on the previous: preflight →
GPU-free gate → ollama up + model present + `llm_check` smoke → Gazebo world →
YOLO-World detection server (GPU) → FALCON + bridge + BEV → the eight host
nodes → a status block with every pid, log and health URL. Everything is
nohup'd under the out dir; the script exits 0 and leaves the mission running.
Every step that starts something is idempotent: a healthy detection server, a
running `sjtu_drone_*` world and an ollama container in any state (running,
paused, exited, absent) are all reused rather than duplicated or fought over.

Flags:

| flag | meaning |
|---|---|
| `--world <name>` | Gazebo world (default `hospital`). Any other world warns: `hospital_doors.yaml` **and** the viz backdrop `hospital.yaml` are hospital-only, so the run continues with **no doors** and the trail drawn on the wrong floor plan |
| `--target <str>` | the search target (default `wheelchair`). Warns if it is not in the detector's live vocabulary |
| `--no-sim` | the world is already up; the script asserts a `sjtu_drone_*` container exists and refuses if not |
| `--no-falcon` | skip FALCON/bridge/rviz — no BEV, so no rooms; detector/objects still run. Also the escape hatch on a host without CycloneDDS |
| `--no-llm` | skip ollama: classifier logs failures and keeps stale labels, oracle publishes `source=uniform_fallback` |
| `--out <dir>` | run dir (default `/tmp/scene_graph/<UTC stamp>`) |
| `--attach` | tail the viz log after bring-up instead of exiting |

Env overrides: `DETECT_PORT` `DETECT_MODEL` `LLM_MODEL` `LLM_BACKEND`
`LLM_BASE_URL` `RVIZ` `VIZ_WINDOW` `KILL_STALE` `SKIP_GPU_CHECK`
`ALLOW_FASTDDS_BRIDGE` `FALCON_LAUNCH_ARGS`.

World names and FALCON map-config names are **two namespaces** — the script
translates between them, and gets it wrong loudly rather than quietly:

| `--world` (Gazebo `<name>.world`) | FALCON `config/<name>.yaml` |
|---|---|
| `hospital` | `hospital` |
| `no_roof_small_warehouse` | `warehouse` |
| `small_house` | `small_house` |

Teardown:

```bash
bash sparx_agency/tasks/mapping/scene_graph/scripts/stop_scene_graph.sh --out <dir>        # host nodes + detection server
bash sparx_agency/tasks/mapping/scene_graph/scripts/stop_scene_graph.sh --out <dir> --all  # + all containers (ollama stopped, kept)
```

Without `--out` it picks the newest `/tmp/scene_graph/*` run. The default stop
leaves the world, FALCON and ollama up so a rerun reuses them — including the
host-side `bringup_world.sh` wrapper, which is only killed under `--all`,
alongside the container removal that actually ends the world (killing the
wrapper on its own just detaches from a container that keeps running).

### What you should see, and when

- **Bring-up** (~1–3 min): ollama answers, `llm_check` passes all three checks,
  odom flows, detection `/health` returns the vocabulary, the bridge logs
  "bridge up: N topic bridges".
- **After FALCON takes off** (~30–60 s after FALCON is up): the first latched
  `/falcon/bev_2d` grid arrives; RViz shows the voxel map growing.
- **Within roughly a minute of flight**: the first rooms appear on
  `/scene_graph` (segmentation needs enough contiguous free space to cut), the
  viz starts writing frames under `<out>/viz/`.
- **As objects accumulate**: `/perception/objects` grows (a landmark needs 2+
  observations to confirm), room labels appear on
  `/semantic_mapper/room_labels`, and `/llm_oracle/probabilities` re-ranks the
  rooms. `/target_seen` flips to `True` (latched, with `/target_seen/info`)
  when a confirmed landmark matches the target.

## Room segmentation: two modes

`semantic_mapper_node` turns the free mask of each `/falcon/bev_2d` tick into
rooms through one of two core segmenters, chosen by the `segmentation`
rosparam. Both return the identical `(room_lbl, skeleton, stats)` triple, so
`RoomRegistry`, `room_stats`, the `/scene_graph` payload and the markers are
mode-agnostic — nothing downstream knows which one ran. An unknown value
raises at startup naming the valid ones; the active mode is logged in the
`semantic_mapper up:` line.

- **`segmentation:=watershed` (default)** —
  `core/mapping/topology/room_watershed.py::segment_rooms_watershed`. Rooms
  are the basins of the clearance field: distance-transform the healed free
  mask, seed a marker at every local maximum at least `min_room_separation_m`
  (2.0 m) apart and `min_clearance_m` (0.6 m) from a wall, and watershed the
  negated field. Doors are layered on top rather than relied on — a disk of
  `door_cut_m` is carved out of the watershed *mask* at every discovered door,
  so a listed door always separates whatever the geometry says, and the carved
  cells are handed back to their nearest room afterwards so no free cell is
  orphaned. A final merge stage (`merge_dynamics_m`, below) repairs the one
  thing a watershed gets wrong.
- **`segmentation:=doors`** — `room_segmentation.py::compute_rooms`, the
  skeleton-cut pipeline the sjtu_project stack flew, unchanged. Medial axis of
  the free mask, punched with a `door_cut_m` disk at every discovered door,
  connected-component labelled, then painted to free cells by nearest skeleton
  pixel. This value reproduces the flown behaviour exactly.

**Why watershed is the default — measured, not preferred.** On a real captured
BEV (413x200 @ 0.15 m, 7852 occupied cells) with coverage simulated as a
growing disc around the explored centroid, largest room as a share of the
segmented area:

| free cells | `doors` (1.6 m cut) | `watershed` (2.0 m separation) |
|---:|---|---|
| 19403 | 12 rooms, **29%** | 5 rooms, 35% |
| 35137 | 12 rooms, **65%** | 14 rooms, 28% |
| 48979 | 12 rooms, **79%** | 14 rooms, 26% |
| 57464 | 15 rooms, **76%** | 19 rooms, 23% |

The door-cut decomposition **collapses into one dominant room as coverage
grows**; the watershed one stays separated and improves. That is not a tuning
error and no `door_cut_m` fixes it, for two reasons. The live BEV marks walls
only where the drone actually observed them, so free space leaks between rooms
at openings absent from the 35-entry door list — and only 11 of those 35 carry
a known width, while portals in this building run up to 24.75 m, which a 1.6 m
disk cannot sever. More fundamentally, the medial axis of the explored region
is **one connected component**, so cutting it at 35 points cannot separate it.
Deriving rooms from clearance geometry instead means a missing wall only lowers
the ridge between two basins rather than merging them, which is why the
decomposition degrades gracefully with coverage instead of collapsing.

The regression is pinned by
`core/mapping/topology/tests/test_room_watershed.py`, which replays that same
captured BEV from `tests/fixtures/live_bev_hospital.npz` and asserts the
largest room stays under 40% of the segmented area.

### The merge stage — `merge_dynamics_m` (watershed only)

A clearance watershed cannot *under*-segment: every local maximum of the
distance field seeds its own basin. It can and does **over-segment** — a room
with two wide spots either side of a bed or a cabinet, or a room bent into an
L, grows two peaks and is reported as two rooms sitting against each other.
That is what "R10 and R11 are the same room" means when an operator says it.

The repair lives in `core/mapping/topology/room_merge.py` and runs after the
carved door cells are reclaimed and **before** the `min_room_cells` floor, so
a lobe merged back into its room keeps its cells instead of being dropped as a
runt. For every pair of adjacent basins it takes the *saddle* `s` (the widest
clearance anywhere along their shared border) and each basin's *peak* (the
widest clearance anywhere inside it), and merges the pair with the smallest

    depth = min(peak_A - s, peak_B - s)

while that depth is below `merge_dynamics_m`, repeating until nothing is left
under the bar. This is the basin's **dynamics** — how much clearance is lost
walking down from the shallower of the two peaks to the pass between them.

**Why not simply merge wide borders.** Measured on the same captured BEV, the
saddle alone barely separates the two cases: genuinely spurious splits sit at
1.8–3.8 m, real doorways at 0.0–0.45 m, and the 67 adjacent pairs run p25 0.45,
p50 0.75, p75 1.35, p90 1.50 — so any absolute threshold has to sit near 1.2 m,
and a corridor touches many rooms at 1.2–1.5 m. Merging every wide-saddle pair
therefore **cascades** the whole floor into one region through the corridors:
15 rooms with the largest covering 82%, i.e. exactly the collapse the watershed
replaced. Dynamics do not cascade, because a corridor is narrow and its own
peak sits barely above the saddles it shares.

Measured on the captured BEV with the building's 35 doors carved (ground truth
20 rooms + 7 corridors = 27 regions):

| `merge_dynamics_m` | 0.0 (off) | 0.30 | **0.50** | 0.75 | 1.00 | 2.00 |
|---|---|---|---|---|---|---|
| rooms | 43 | 36 | **29** | 28 | 27 | 26 |
| largest room | 10.7% | 12.2% | **12.2%** | 12.2% | 12.2% | 12.4% |

A third of the rooms disappear and the largest one does not grow — the merge is
a plateau from 0.50 m up, not a knife edge, which is why 0.50 is the default.
`0.0` disables the stage and reproduces the raw watershed cell for cell (an
explicit off switch, not a threshold: a saddle is the maximum over *both* sides
of a border, so a depth can come out slightly negative and a literal `< 0.0`
would still merge).

**A listed door is never merged across**, whatever the dynamics say — "a room
is a closed area bounded by doors". The carve is gone by the time the merge
runs, so the barrier is recovered from the carve mask: any shared border whose
cells lie inside a door disk marks that graph edge permanently unmergeable, and
the flag is inherited when either basin is later absorbed.

**The cost of the stage, honestly.** A *small* room joined to a bigger one by a
wide-enough opening is absorbed: absorption happens when a room's own peak is
less than `merge_dynamics_m` above the saddle at its opening, so a 0.75 m-peak
closet off a 0.30 m-saddle corridor merges into the chamber at the default. In
this building the door list covers those cases; without a door list geometry
alone is much sharper (35 → 22 → 9 rooms at 0.30 / 0.50 m), which is the
measurement behind carving doors rather than tuning this knob.

Efficiency matters — the segmentation itself is the tick's whole budget. The
region adjacency graph is built in **one** pass over the label image,
contracted with union-find on the graph alone (merging A and B takes
`peak = max` and, per shared neighbour, `saddle = max`), and the image is
written once at the end through a vectorised label remap. Measured on the
413x200 fixture with 43 basins, the stage costs **1.17 ms median / 1.74 ms
p90** against **46 ms** for the watershed segmentation it sits inside (the
`doors` segmenter is 32 ms on the same grid) and a 500 ms tick period.

### Room-to-room edges — adjacency, not proximity

The gold `room -> door -> room` dog-legs are the topology an operator reads off
the picture, and they used to be **proximity**: `room_stats.link_doors` collects
whichever rooms have cells in the annulus around a door (between the 1.60 m cut
radius and the 0.90 m `door_match_radius_m`, so an 11–13 cell ring at 0.15 m),
and every pair of them became an edge. Near a corner that ring reaches a room on
the far side of a wall — "how is R11 connected to R16? There is a wall between
them."

The rule now is the operator's own: **two rooms are connected only if their
regions directly touch**, so a path from one to the other crosses no wall and no
third room. `core/mapping/topology/room_adjacency.py::room_adjacency` answers
that from the room label image — wall cells are not free and carry label 0, so
rooms either side of a wall never touch, while rooms either side of a doorway do
(both segmenters hand the carved doorway cells back to their nearest room). The
mapper computes it once per tick and vets the door links through
`room_stats.door_room_pairs`; a door whose candidate rooms do not touch
contributes **no** pair rather than an invented one. It is the same border scan
the merge stage above contracts (`iter_label_borders`), so "these basins touch"
and "these rooms are connected" cannot drift apart.

**Why the payload carries pairs and not just a room list.** `doors[].rooms` was
never a clique: measured on the captured BEV, 23 of the 35 doors see three or
more rooms in their annulus, and at 4 of them rooms A–B and B–C touch while A
and C do not. Filtering the room *list* against adjacency removes nothing there
— every room in it is adjacent to *something* — and any consumer pairing that
list up re-invents the A–C edge. So `door_entry` publishes
`doors[].room_pairs`, the vetted `(pid, pid)` edges, and `doors[].rooms` is
their union. Both drawing paths (`ros2/scene_markers.py::room_edge_markers` and
`viz_graph_overlay.py::draw_room_edges`) read the pairs and enumerate nothing of
their own, so the RViz view and the OpenCV dashboard show the same graph.

Measured on the captured BEV: the annulus proposes **61** distinct room pairs
across 29 rooms and **4** of them join rooms with a wall between (3 through the
node's own world→cell rounding, which shifts a door by a cell). The whole
adjacency scan costs **0.22 ms** on that 413x200 grid, against a ~60 ms tick.

## Topics and QoS

Rule of thumb: **state is latched** (RELIABLE + TRANSIENT_LOCAL, depth 1, on
*both* pub and sub side), **sensors are best-effort volatile** with a small
depth (a best-effort sub happily receives from the sim's reliable publishers).

| topic | type | QoS | who |
|---|---|---|---|
| `/simple_drone/front/image_raw` (+`camera_info`) | Image, RGB 600x600 @60 Hz | best-effort volatile | sim → detector_client |
| `/simple_drone/front_depth/depth/image_raw` (+`camera_info`) | Image 32FC1 m @15 Hz (valid 0.1–10 m) | best-effort volatile | sim → object_mapper |
| `/simple_drone/odom` | nav_msgs/Odometry | best-effort volatile | sim → mappers |
| `/falcon/bev_2d` | nav_msgs/OccupancyGrid (`-1` unknown / `0` free / `100` occ) | **RELIABLE + TRANSIENT_LOCAL** (bridged ROS1 latch — a volatile sub gets nothing until the next map change) | bridge → semantic_mapper |
| `/perception/detections` | String JSON | reliable volatile, depth 5 | detector_client → object_mapper |
| `/perception/objects` | String JSON | latched | object_mapper → classifier/oracle/watcher/viz |
| `/scene_graph` | String JSON | latched | semantic_mapper → everyone |
| `/scene_graph/room_labels_grid` | nav_msgs/OccupancyGrid (`0` no room / `1..100` room grid value) | latched | semantic_mapper → viz |
| `/semantic_mapper/room_labels` | String JSON | latched | room_classifier → oracle/viz |
| `/llm_oracle/probabilities` | String JSON (probs sum to 1) | latched | llm_oracle → viz |
| `/target_seen` + `/target_seen/info` | Bool / String JSON (False published at startup) | latched | target_watcher → mission consumers |
| `/scene_graph/markers` | visualization_msgs/MarkerArray | latched | semantic_mapper → RViz |
| `/scene_graph/planned_path` + `/scene_graph/goal` | nav_msgs/Path + PoseStamped | latched | room_search → RViz/operator |
| `/scene_graph/follow_path` | nav_msgs/Path | latched | room_search (**armed only**) → the follower |
| `/room_search/info` | String JSON | latched | room_search → operator |
| `/room_search/active` + `/room_search/falcon_active` | Bool (inverses; both published at startup) | latched | room_search → whatever arbitrates `cmd_vel` |

JSON payload schemas are pinned in the module docstrings of the publishing
nodes under `ros2/`; the detection HTTP wire types live in `serve/contract.py`
(`DEFAULT_PORT=8092`, `DEFAULT_HOSPITAL_VOCABULARY`, 27 terms).

**Room tints are a two-part contract.** `OccupancyGrid` data is `int8` and room
pids grow without bound, so `/scene_graph/room_labels_grid` carries a small
recycled *grid value* per room (`0` = no room) with the BEV's own `info` and
`frame_id` (but this tick's stamp, not the BEV's older build time), and the
`grid_pid_map` key of `/scene_graph` — `{"<grid value>": pid}` —
resolves those values back to pids. Both are built from one `{pid: grid value}`
mapping per tick in `ros2/payloads.py` (`assign_room_grid_values`,
`room_value_grid`, `grid_pid_map`), which is what keeps them coherent: a grid
value with no map entry would be tinted by the raw value, i.e. some other
room's colour. Values are stable while a room keeps its pid, and recycled only
once the room is gone.

## The RViz view

`semantic_mapper_node` draws the whole scene graph as one latched
`visualization_msgs/MarkerArray` on `/scene_graph/markers` every
`marker_period_s` (1.0 s; `publish_markers:=false` turns it off). Geometry
lives in `ros2/scene_markers.py` and is unit-tested without ROS.

```bash
rviz2 -d sparx_agency/tasks/mapping/scene_graph/config/scene_graph.rviz
```

That config carries three displays — the BEV, the room-label grid and the
markers — all three declared RELIABLE + TRANSIENT_LOCAL, which is what lets an
RViz started at any point in the run see the map that is already on the wire.
Fixed frame is `world`, the BEV publisher's own frame. It is **not** started by
`run_scene_graph.sh`: the FALCON RViz that script brings up is a different
config in a different container.

What the array shows, from the floor up: translucent room fills and the Voronoi
"open space" spine in each room's colour, the centroid sphere, the **gold
centroid → door → centroid dog-legs** of the room topology, then three stacked
text lines per room — `R7 t=42s F=3`, the LLM's `office (0.82)`, and the
oracle's `P=0.41`. Door pillars are orange once reached and a grey ghost while
still pending. The array is rebuilt from scratch every period behind a leading
`DELETEALL`, so nothing survives a room disappearing or a pid restart.

Colours are **not** chosen in the viz: rooms come from
`core.mapping.topology.room_color(pid)` and objects from
`core.mapping.objects.landmarks.class_color(name)` — the same two functions the
`/scene_graph` JSON and the PNG/MP4 dashboard use, so every view of a run
agrees on which room is which.

## Closing the loop: `room_search_node`

Everything above observes. `ros2/room_search_node.py` acts: it joins the
oracle's ranking to the scene graph's centroids, hands them to the ROS-free
`core.planning.exploration.room_search_policy.RoomSearchPolicy` (sample a room
weighted by its probability → pursue it → dwell in it → re-sample), plans to
the chosen centroid with the shared weighted A\* over a fresh `OccupancyGrid2D`
built from the live BEV, and publishes the goal, the route and an operator
payload.

```bash
.venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.room_search_node \
    --ros-args -p use_sim_time:=true -p fly:=false
```

**`fly` is False by default and that is the useful mode.** Unarmed, the node
publishes the chosen room and the route to it for RViz and the dashboard while
FALCON keeps flying its own exploration; it publishes no Twist anywhere, and
`/scene_graph/follow_path` — the only topic a follower reads — stays silent, so
an unarmed node cannot move the aircraft however it was launched.

There is deliberately **no follower here**. Flight is delegated to
`tasks/planning/sjtu_internvla_n1/ros2/trajectory_follower_node.py`, which is
already policy-agnostic and carries the altitude capture, the odom timeout, the
capsize guard and the airframe clamp; `config/room_search_follower.yaml` points
it at `/scene_graph/follow_path` and at `/simple_drone/cmd_vel_raw`.

**Arbitration is not built, and `fly:=true` is inert until it is.** FALCON's
`bspline_follower` publishes a Twist every tick at 50 Hz in every state, and so
does that follower at 20 Hz — two continuous writers on one `cmd_vel` is
last-writer-wins, not a handover. What this node contributes is the *fact*: a
latched Bool on `/room_search/active`, true only while a planned route is being
pursued with flight armed, and its inverse on `/room_search/falcon_active`. To
actually fly it, three things are still missing and all three are outside this
task:

1. a gate on the ROS2 side (FALCON's own `cmd_vel_gate_node` is a ROS1 node
   inside the FALCON container) — today **nothing subscribes**
   `/simple_drone/cmd_vel_raw` on ROS2, so the armed follower's Twist goes
   nowhere and the aircraft simply does not move;
2. `/simple_drone/cmd_vel` bridged ROS1→ROS2 must go through that gate rather
   than straight to the sim (`falcon_sjtu/config/bridge.yaml`);
3. `/room_search/active` and `/room_search/falcon_active` bridged to whichever
   gates they drive.

Room ids are the pin holding all of this together, and they **restart** whenever
the BEV's geometry changes — the mapper resets its registry, dwell times and
door discovery, so room 3 afterwards is a different room from room 3 before.
Both consumers of a pid handle it: the mapper drops its cached room types,
probabilities, skeleton and door links in the same reset, and the search node
drops the latched ranking and the policy's visit cooldowns the moment it sees
the BEV reshape.

## Environment variables

| var | default | meaning |
|---|---|---|
| `SJTU_PROJECT_DIR` | *(required for sim)* | external sim checkout (see branch note below) |
| `DISPLAY` | *(required for sim)* | X display; `:1` on this machine |
| `ROS_DOMAIN_ID` | `20` | every participant, containers included — a mismatch is silent zero data |
| `RMW_IMPLEMENTATION` | auto (cyclonedds if installed, else fastrtps) | CycloneDDS is the only RMW the ROS1 bridge can reach; both profiles under `robots/SJTU/setup/` disable shared memory (SHM does not cross the container boundary) |
| `LLM_BACKEND` / `LLM_BASE_URL` / `LLM_MODEL` | `ollama` / `http://127.0.0.1:11434` / `qwen2.5:3b-instruct` | the `LLMClient.from_env()` contract (also `LLM_API_KEY`, `LLM_TEMPERATURE`, `LLM_TIMEOUT_S`) |
| `DETECT_PORT` / `DETECT_MODEL` | `8092` / `<repo>/yolov8s-worldv2.pt` | detection server (checkpoint is gitignored, per-device) |
| `SKIP_GPU_CHECK` | `0` | `1` bypasses the empty-card gate — **expert-only**, see below |
| `KILL_STALE` | `1` | auto-kill scene-graph nodes left over from a previous run |
| `RVIZ` / `FALCON_LAUNCH_ARGS` / `SAFE_DISTANCE` | — | forwarded to `run_falcon_sjtu.sh` (the script always injects `enable_bev:=true`) |

## GPU discipline

One 8 GB card, and it is **exclusive** — a second CUDA or GL context in
whatever is left has hard-locked this host (not the process, the machine).
Ownership in this mission:

- **YOLO-World owns the card** (`detection_server` in conda `navdp`,
  `CUDA_VISIBLE_DEVICES=0`). It is the only per-frame model here.
- **Gazebo renders on the CPU** — `bringup_world.sh` runs no `--gpus` and
  forces llvmpipe.
- **Every host node exports `CUDA_VISIBLE_DEVICES=""`.**
- **ollama is CPU-only by design** (no `--gpus` in its `docker run`): a
  3B-parameter model answers in seconds on this box's CPU, the calls are
  per-room not per-frame, and putting it on the card would fight YOLO for
  memory it does not have.

The run script gates on
`tasks/planning/sjtu_internvla_n1/scripts/check_gpu_free.py --require-empty`
before starting (skipped when a healthy detection server already holds the
card). `SKIP_GPU_CHECK=1` bypasses it and is for the operator who has
personally read `nvidia-smi` and knows exactly what is resident.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| every topic exists, zero data anywhere | `ROS_DOMAIN_ID` mismatch between host and a container — the classic silent failure. All participants must be on 20; check `docker inspect <c> | grep ROS_DOMAIN_ID` |
| drone flies, odom flows, but no images | `DISPLAY` was unset when the world started: Gazebo Classic silently disables **all** camera sensors without X ("Unable to create CameraSensor" in the gazebo log). `export DISPLAY=:1` and restart the world |
| `/perception/detections` silent, detector_client logs `connection refused` | the detection server (step e) is not up — `curl http://127.0.0.1:8092/health`, read `<out>/detection_server.log` (wrong interpreter? CUDA busy? missing checkpoint?) |
| no `/falcon/bev_2d` ever | `enable_bev:=true` was not passed to FALCON (it is default-off in `exploration.launch`; the run script always passes it) — or the bridge bridged nothing: `docker logs sjtu-ros1-bridge`, look for "create bidirectional bridge for topic /falcon/bev_2d" |
| bridge container Up, "0 topic bridges" | the domain is shared with other ROS2 participants and Foxy's `parameter_bridge` segfault-loops on discovery. Bring the world up on a domain of its own (`--domain 20`) |
| room labels stop updating; oracle rows say `source: uniform_fallback` | the LLM is down or timing out. The classifier **raises and keeps stale labels** (by design — never a silent wrong label); the oracle degrades to uniform and *says so* in the payload. `curl http://127.0.0.1:11434/api/tags`, `docker logs ollama-scene-graph`, rerun `llm_check` |
| viz draws rooms and labels but no room tints; footer says `room grid not yet published` | `/scene_graph/room_labels_grid` is not arriving — `ros2 topic echo --once /scene_graph/room_labels_grid --qos-durability transient_local`. Both sides must be latched; the semantic_mapper heartbeat prints `rooms=N(tinted M)` and warns when `M < N` (more than 100 simultaneous rooms) |
| rooms exist but no doors / `discovered` never true on a non-hospital world | expected: `hospital_doors.yaml` is a hospital survey; other worlds run door-less (the run script warns) |
| nodes died at startup, log shows an import error | wrong interpreter. Host nodes run on `.venv` (system-site rclpy, **no torch**); the detection server runs on conda `navdp` (torch, **no rclpy**). Neither can run the other's job |

## The sim checkout branch

`$SJTU_PROJECT_DIR` must be on branch **`xtend_integration_nadav`** — that is
the branch the CURRENT aircraft (plugin, URDF, camera rig) runs from. The
scene-graph code itself was excavated from the older `detect_navigate_nadav`
branch and ported onto the new core APIs, but the sim assets are not run from
there: mixing the two flies an old drone under new perception.

## Layout

```
scene_graph/
  ros2/        the eight host nodes (plain rclpy scripts, python -m runnable)
  serve/       detection_server.py + contract.py (HTTP wire) + selftest
  scripts/     run_scene_graph.sh / stop_scene_graph.sh / llm_check.py
  tests/       ROS-free unit tests for everything above (see below)
```

## Tests

Run from the repo root, on the plain `.venv` — **no sourced ROS, no torch**:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
    sparx_agency/tasks/mapping/scene_graph/tests/
```

The four files cover the parts that are pure data or pure pixels, which is
everything except the rclpy plumbing itself:

| file | covers |
|---|---|
| `test_detection_contract.py` | `serve/contract.py`: JPEG round-trip, detection JSON round-trip, the hospital vocabulary |
| `test_detector_client_payload.py` | `ros2/detection_payload.py`: the `/perception/detections` payload carries the **source** image stamp (never the server's), malformed replies raise instead of publishing junk, the debug overlay leaves its input frame alone |
| `test_payloads.py` | `ros2/payloads.py`: `/scene_graph` + `/perception/objects` key nesting, numpy scalars coerced so `json.dumps` cannot raise mid-flight, and the `room_labels_grid` value/pid pair never disagreeing |
| `test_viz_render.py` | `viz_render.py` + `viz_canvas.py` + `viz_graph_overlay.py`: renders with a completely empty state (the startup case), room tints land as the palette in BGR, probability bars scale with probability, the target banner appears only once the target is seen, and the room graph — a gold dog-leg edge **only** for a discovered door carrying a vetted `room_pairs` entry between two known rooms (rooms merely listed at a door draw nothing), room nodes in the room color, door labels dropped when the view is too zoomed out to read them, the legend, and a payload with no `doors` key at all |

The three modules under test import **without** a ROS environment on purpose —
if one of them ever pulls in `rclpy`, the import belongs in the node file, not
in the module.

## First flight

What the first end-to-end run actually produced, so the next person knows what
"working" looks like rather than guessing:

- **FALCON explored the hospital under its own frontier planner** — no goal
  clicked, no external route.
- **Detection server warm at ~7 ms per 600x600 frame** on `cuda:0`, serving the
  hospital vocabulary over HTTP the whole run.
- **76 confirmed object landmarks** (each 2+ observations): chair, tv, person,
  cabinet, shelf, medical trolley, instrument cart, desk, sofa, refrigerator, …
- **25 of the 35 doors discovered**, and **4 rooms** on the explored wing with
  **75 of the 76 objects** assigned to one.
- **Room typing answered from the objects** — e.g. `storage_closet` at `0.95`
  confidence off a room of cabinets and shelves.
- **The oracle returned `source="llm"`** (not `uniform_fallback`), with
  probabilities summing to 1.
- **The viz wrote `latest.png`, numbered frames and an MP4** under `<out>/viz/`.

**The room count was the weak part**, and that flight ran `segmentation:=doors`
— the only mode that existed then, which can only cut where the drone has
actually flown *through* a door, so a partially observed wing read as one room
and the whole hospital collapsed into one as coverage grew. The default is now
`watershed` (see "Room segmentation: two modes"), which does not depend on the
door list for separation. A low room count mid-flight is still partly coverage
— check `discovered` on the doors before suspecting the door list — but a
*falling* share of distinct rooms as coverage grows is the door-cut failure
this replaced.

**LLM quality is the other caveat.** `qwen2.5:3b-instruct` accepted "medical
trolley" as a match for "wheelchair". The match ladder is working — it correctly
rejects person, shelf, cabinet and chair — but a 3B model's judgement is loose,
and `LLM_MODEL` is the knob if a run needs a stricter verdict.
