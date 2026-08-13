# FALCON source patches (falcon_sjtu)

The FALCON planner is upstream C++, cloned and compiled into the
`falcon-ros-custom:v1` image. Anything we change *inside FALCON itself* must be
reproducible from a clean image build — otherwise a rebuild or a fresh container
silently drops it. Each such change is a patch here, applied by the image build.

## Where the build actually applies them

The image is built by **`sjtu_project/falcon_docker/Dockerfile`** (an external
repo), which `git clone`s FALCON (`ros1-noetic`) and then runs a series of
self-verifying `fix_falcon_*.sh` scripts before `catkin_make`. All four patches
here are wired in as one step:

- `sjtu_project/falcon_docker/sparx_patches/*.patch` — a copy of this directory.
- `sjtu_project/falcon_docker/fix_falcon_sparx_patches.sh` — `git apply`s each
  one in turn and verifies its sentinel, failing the **image build** rather than
  shipping a silently unpatched FALCON.
- `Dockerfile` step "6a-ter" — `COPY`s both in and runs the script.

The copies in this directory are the **authoritative source**; the build reads
the mirror in `falcon_docker/sparx_patches/`. Keep them in sync.

**Two older fix scripts are subsumed and must not run alongside these.** The
patches were cut with `git diff` inside a container where the sed-based fixes had
already been applied to the working tree, so they carry those exact edits as
their own additions:

| superseded script | now lives inside |
|---|---|
| `fix_falcon_cost_check.sh` (6 cost floors) | `falcon_hgrid_clamp.patch` |
| `fix_falcon_depth_overflow.sh` (2 resizes) | `falcon_visgrid_cadence.patch` |
| `fix_falcon_sop.sh` (1 s → 10 s) | `falcon_deadend_guard.patch` |

Steps 6a-ter and 6a-quater were therefore removed from the Dockerfile. Running
one of them *and* its patch leaves the tree unpatchable — the script inserts the
lines, and `git apply` then fails on context it expected to add itself. The sop
script is the exception that stays wired at step 8b: it checks for its own
result first and self-skips, so the patch simply makes it a no-op.

Verified against upstream `ros1-noetic` as of 2026-08-13: all four apply clean,
no `--3way` fallback needed.

## falcon_deadend_guard.patch — escape unreachable dead ends

Touches three upstream files (and none the other `fix_falcon_*.sh` scripts do):

- `exploration_manager/src/exploration_manager.cpp`
- `exploration_preprocessing/include/exploration_preprocessing/frontier_finder.h`
- `exploration_preprocessing/src/frontier_finder.cpp`

**Why.** The depth camera is blind inside ~0.95 m (its near clip), so an obstacle
the aircraft is about to hit never enters the map. FALCON's A* then routes a
"clear" path straight through it and re-selects the same blocked viewpoint every
cycle — FALCON only marks a frontier *dormant* when it has no viewpoint at all,
never when the viewpoint is merely unreachable. The result is a permanent stall
(coverage frozen, `next_pos ... same as current`, or the aircraft wandering a
1–2 m pocket) that previously only a crash broke, by wiping the map.

**What it adds.**
- `FrontierFinder::addBlockedRegion(pos)` + a `blocked_regions_` list;
  `computeFrontiersToVisit()` retires to dormant any cluster whose average is
  within `/frontier_finder/blocked_region_radius` (default 2.5 m; set in our
  `adapter/launch/exploration.launch`) of a blocked point.
- A no-progress guard in `planExploreMotionHGrid()`: if the aircraft's whole
  excursion stays under 2 m for 25 s while still not at the chosen viewpoint,
  that viewpoint is handed to `addBlockedRegion()` and the coverage tour moves on.
- **(2026-08-12)** A degenerate-tour guard in `solveTSP()`: tours of ≤3 cities
  are enumerated directly and never enter LKH, whose in-process solver corrupts
  the heap on tiny tours — measured 15 respawns in one 45-minute mission, each
  arriving in the respawn→tiny-map→tiny-tour→crash cascade this breaks
  (sentinel: `never enter LKH`).
- **(2026-08-12)** Blocked regions persist to the param server
  (`/frontier_finder/blocked_regions_runtime`) and reload on construction, so a
  respawned frontier finder keeps every physics-vetoed viewpoint shadowed
  (sentinel: `blocked_regions_runtime`).

## falcon_hgrid_clamp.patch — never index outside the grid box

Touches one upstream file: `exploration_preprocessing/src/hierarchical_grid.cpp`
(sentinel: `Clamp into the grid box FIRST`).

**Why.** `UniformGrid::positionToGridCellId()` floors a position into a cell
index with no bound check, and `positionToGridCellCenterId()` then indexes
`uniform_grid_` with the result. The aircraft is *below the box floor on every
takeoff* (`box_min_z` 0.6 while it rests at z=0), so the index is negative
before the mission even starts, and any overshoot of a face does the same thing
later. Upstream knew — the CHECKs at the bottom of that function are commented
out — and shipped the crash: measured as 14 planner deaths in 8 minutes once the
flight box floor was raised.

**What it does.** Clamps the position into the box before flooring, clamps the
per-axis cell index into range, and re-clamps on the caller's side if the id is
still out of range, returning −1 rather than indexing. Every caller is asking
"which tour cell is the aircraft in"; for an aircraft outside the box the
nearest cell *is* the answer. It also carries the six cost floors that
`fix_falcon_cost_check.sh` used to insert.

## falcon_replan_from_pose.patch — plan from the aircraft, not from its ghost

Touches one upstream file: `exploration_manager/src/exploration_fsm.cpp`
(sentinel: `replanning from the real pose`).

**Why.** Once FALCON leaves `static_state_` it stops planning from odometry and
starts every replan from the PREVIOUS trajectory, evaluated `replan_duration_`
ahead and **clamped to that trajectory's end**:

```cpp
double t_r = (time_now - info->start_time_).toSec() + fp_->replan_duration_;
if (t_r > info->duration_) t_r = info->duration_;
fd_->start_pos_ = info->position_traj_.evaluateDeBoorT(t_r);
```

That is fine while the aircraft tracks the curve. Ours frequently does not —
the depth brake, the map gate and the retreat all stop it on purpose — and
`static_state_` is only restored on a planning failure or on FINISH, never on
"the aircraft did not get there". So the start point runs to the end of a curve
that was never flown, and every later plan compounds from that ghost. Measured
on the warehouse: `position_error` 5.4 m, `along_track_lag` 4.5 m,
`reference_time_s` pinned at 8e-05 s across **118 trajectories** — the follower
crawling at `max_catchup_speed` toward a start point that moved again before it
could ever be acquired. On screen it reads exactly as "the planned path starts
several metres away from the drone and it will not follow it".

**What it does.** After computing the ghost start, compare it with `odom_pos_`;
past `/fsm/replan_from_pose_drift` (default 1.5 m) fall back to planning from
the real pose, velocity and yaw — precisely what the `static_state_` branch
above it already does. It is a fallback, not a replacement: while tracking is
healthy the drift never reaches the threshold and upstream behaviour is
untouched.

**Result** (warehouse, `obstacles_inflation:=0.35`, 2026-08-13): coverage
**153.9 m³ of the 161 m³ flight box (95.6%)**, mission reaching FINISH on its
own, 15 retreats and **zero bumper contacts**. The same stack before this patch
plateaued at 99 m³ and milled in one spot.

## Rebuilding vs. the running image

An image that was `docker commit`ed live carries these already and needs no
rebuild. The patch mechanism above is what makes them survive an image
**rebuild** (`docker build sjtu_project/falcon_docker`) or a fresh FALCON clone
— and it is the only thing that does, which matters on a machine that never had
the committed image. `falcon-ros-custom:v1` was rebuilt from scratch this way on
PCN87653 on 2026-08-13.

## Regenerating the patch

If you change FALCON's C++ again in the running container:

```bash
docker exec falcon-sjtu bash -lc \
  'cd /catkin_ws/src/FALCON && git --no-pager diff -- <the files of ONE patch>' \
  > sparx_agency/tasks/planning/falcon_sjtu/patches/falcon_deadend_guard.patch
cp sparx_agency/tasks/planning/falcon_sjtu/patches/falcon_deadend_guard.patch \
   ~/GIT/sjtu_project/falcon_docker/sparx_patches/
```

The fix script self-verifies with a **single-line** sentinel (`confined to <2 m`)
because the guard's full log line is split across two C string literals — grep a
whole-phrase sentinel and it will spuriously miss.

## falcon_simtime.patch — let FALCON run under /use_sim_time

Touches one upstream file no other fix script does:
`exploration_manager/src/exploration_node.cpp`.

**Why.** exploration_node aborts at startup (glog `CHECK(!use_sim_time)`) when
the global `/use_sim_time` param is true. Here every data stamp in the graph is
already Gazebo sim time (bridged depth, odometry, and the camera pose derived
from it), and the hospital world runs below real time (~0.88x), so wall-clock
`ros::Time::now()` stamps B-splines on a clock ~12% faster than the physics —
the follower burns its catch-up margin closing pure clock skew. Sim time is the
only consistent configuration; the upstream check merely predates running
FALCON against a sub-realtime simulator.

**What it does.** Replaces the CHECK with a `LOG(INFO)` recording the clock
choice (sentinel: `sim-time permitted`). Wired in `falcon_docker` as step
6a-sexies (`fix_falcon_simtime.sh`), and compiled into the running
`falcon-ros-custom:v1` via `docker commit` on 2026-08-11.

## falcon_visgrid_cadence.patch — stop publish cost scaling with map growth

Touches one file no other fix script does:
`voxel_mapping/src/map_server.cpp`.

**Why.** `publishOccupancyGrid()` sweeps the entire voxel box every 0.5 s and
serialises occupied, FREE and UNKNOWN clouds whenever anyone subscribes (the
follower's brake gate does, and so does RViz). The free/unknown clouds grow
with the explored volume — hundreds of thousands to millions of points — so
publish and subscriber-parse cost grow linearly with mapping progress: the
measured "the further the mission maps, the slower everything runs" failure.

**What it does.** Occupied keeps its 2 Hz cadence (small, and the brake gate
feeds on it); free/unknown publish every 10th cycle (sentinel:
`publish_bulk`). Wired in `falcon_docker` as step 6a-septies
(`fix_falcon_visgrid_cadence.sh`), compiled into the running image via
`docker commit` on 2026-08-11. The follower additionally throttles its own
free-cloud processing to one per 3 s, so unpatched images degrade gracefully.

## Not FALCON code (already reproducible, no patch needed)

The flight tuning lives in our adapter, mounted at runtime from this repo, so it
is already reproducible: `adapter/launch/exploration.launch` (inflation 0.25,
frontier clearances, `blocked_region_radius`), `adapter/launch/bspline_follower.launch`
(speed cap 0.6, recovery params), and `adapter/scripts/bspline_follower_node.py`
(re-survey, hold-on-silence, 22° attitude reflex).

