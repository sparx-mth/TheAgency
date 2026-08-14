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

## falcon_slow_traj_rescale.patch — re-time onto our speed, and never abort

Touches two upstream files (sentinels: `re-time a sluggish trajectory` in the
first, `never abort the process on a malformed spline` in the second):

- `exploration_manager/src/exploration_fsm.cpp`
- `trajectory/src/bspline/non_uniform_bspline.cpp`

**Why.** After every plan the FSM measures the trajectory's average speed, and if
it is under 0.5 m/s it rewrites the knot vector so the same curve would be flown
at **2.0 m/s** (and yaw at 1.57 rad/s):

```cpp
double yaw_ratio = 1.57 / avg_yaw_vel;
double pos_ratio = 2.0 / avg_pos_vel;
double ratio = 1.0 / std::min(yaw_ratio, pos_ratio);
```

Upstream flies fast, so that is a correction. This aircraft is configured to
cruise at 0.15 m/s, so it is a compression — measured **324 rescales in one
16-minute hospital mission**, ratios clustered at **0.08–0.25**, i.e. every
seventh plan handed the follower a reference four to thirteen times faster than
the limit the B-spline optimiser had just been given. The follower is not
permitted to chase it, so the reference walks away from the aircraft; that is the
same gap `falcon_replan_from_pose.patch` above keeps having to repair, arriving
from the other end.

The same four lines also hold the crash this package spent a long time
misattributing. A trajectory that does not turn has `avg_yaw_vel == 0`, so
`1.57 / 0` is an infinity; a zero-duration trajectory makes both averages NaN.
`std::min` propagates neither the way this code assumes, and the non-finite
ratio that comes out re-times the knots into a shape that later aborts the
process:

```
[FSM] Slow trajectory detected, duration: 25.16, length: 3.89
[FSM] Avg position velocity: 0.15, avg yaw velocity: 0.00, lengthen ratio: 0.08
exploration_node: Eigen/src/Core/Block.h:120: Assertion `(i>=0) && (... i<xpr.rows())' failed.
  #7 ExplorationFSM::visualize()
  #6 PlanningVisualization::drawBspline(...)
  #5 NonUniformBspline::evaluateDeBoorT(double const&)
  #4 NonUniformBspline::evaluateDeBoor(double const&)
```

That matters far more here than upstream, because on this stack the **voxel map
lives in the same node**: one assertion inside a *visualisation* call ends the
mission and erases everything mapped so far. Measured directly — a hospital run
holding 320 m³ was back to 22 m³ one second later.

**What it does.**

- Takes the rescale's targets from parameters (`/fsm/slow_traj_target_vel`,
  `_target_yaw`, `_trigger_vel`, `_ratio_min`, `_ratio_max`), declared in
  `adapter/launch/exploration.launch`. With the target set to the configured
  cruise, a trajectory already flying at the speed it was asked for yields ratio
  1.0 and is left alone.
- Clamps the ratio, and we ship `ratio_min: 1.0` — the rescale may **stretch**
  time only. Stretching is a real safety valve (a too-tight time allocation gets
  slowed to something flyable); compressing is precisely what breaks the velocity
  feasibility the optimiser had already guaranteed. Nothing can now speed a
  reference up.
- Requires finite, positive averages before dividing, and skips the rescale
  (with a log line) when neither axis is moving.
- Guards `NonUniformBspline::evaluateDeBoor` so that no malformed spline can
  abort the process again, whatever produces it. The span search there walks the
  knot vector with no upper bound and then indexes `control_points_` with the
  result; it is bounded now by both the knot count and the control-point count,
  a spline with fewer than `p_+1` control points returns its last point, and a
  non-finite parameter falls back to the curve's start. A clamped evaluation of a
  degenerate curve is wrong in a way the mission can see and recover from.
  `abort()` is not.

## falcon_blocked_region_ttl.patch — a shadow is about the map, and the map changes

Touches the two files `falcon_deadend_guard.patch` already owns, and must be
applied after it (sentinels: `a shadow is a statement`, `has expired`):

- `exploration_preprocessing/include/exploration_preprocessing/frontier_finder.h`
- `exploration_preprocessing/src/frontier_finder.cpp`

**Why.** The dead-end guard hands the frontier finder a viewpoint the aircraft
could not physically reach, and every frontier cluster within
`blocked_region_radius` of it is retired to dormant. That shadow was
**permanent** — it lasted the life of the node, and was persisted to the param
server so it outlived the node too.

A shadow is a statement about the map as it stood when it was cast, and the map
changes. Cast in the first minute of a mission, when almost nothing has been
observed, it is usually wrong. Measured on the hospital, 2026-08-14: the guard
blocked `(5.11, -1.21)` at **t = 53 s**, eight seconds into the flight; at the
3.5 m radius then in force that covered the corridor joining the north wing to
the south. The aircraft mapped the north, exhausted its frontiers and declared
**FINISH at 253.65 m³** with the entire southern half of the building never
visited — a *successful* mission by FALCON's own reckoning.

The severity depends entirely on what got shadowed, and nothing in the guard
knew the difference: over a dead-end pocket the shadow is exactly right, over a
transit corridor it is fatal. Permanence is what made the second case
unrecoverable.

**What it does.** Puts time into the trade rather than picking a side of it.

- Every region carries the time it was cast and expires after
  `/frontier_finder/blocked_region_ttl_s` (90 s; `<= 0` restores the old
  permanent behaviour). `expireBlockedRegions()` runs at the top of
  `computeFrontiersToVisit()`, so an expired region is reconsidered on that
  tour rather than the next one.
- **Each re-report doubles the next shadow**, capped at
  `2^blocked_region_ttl_max_doubling` (3, so 8×). An early mistake in a corridor
  is forgiven after 90 s; a pocket that genuinely cannot be observed — the
  hospital's stairwell-filled corner rooms, its solid-slab elevators, its
  floor-to-ceiling cubicle curtains — is re-reported and retires for 90, 180,
  360, then 720 s, which is the rest of the mission. Termination survives;
  completeness stops depending on the first minute of the flight.
- **Strikes are counted on a coarser key than shadows, and on a history that
  never expires** (`blocked_region_escalate_radius`, 4.0 m). This is not a
  refinement; without it the escalation above does not work at all. An aircraft
  defeated twice by the same obstacle does not fail at the same point twice —
  measured 2.6 m apart in the hospital's north-west corner, outside the 2.0 m
  coalesce radius — so both failures registered as new regions at strike 1, the
  TTL never doubled, and the aircraft went back and ground on one clutter pile
  for **297 s of a 485 s run, 78 bumper reports**. The history is consulted only
  to seed a new shadow's strike count.
- A repeat report inside the radius now **refreshes** the shadow instead of
  being silently dropped, because the aircraft is telling us it is still stuck
  there, and that is precisely the evidence that should extend it.
- The persisted copy is rewritten on expiry, so a respawn cannot restore
  shadows that have just been retired. Restored shadows are stamped with the
  restore time and given one strike: the successor cannot know how much of
  their life they had already spent, and one more TTL is the conservative
  reading.

With expiry doing the work, `blocked_region_radius` goes back **under**
`candidate_rmax` (2.0 against 2.5) in `adapter/launch/exploration.launch`, so
retiring one viewpoint no longer retires the whole frontier that produced it.

## Rebuilding vs. the running image

An image that was `docker commit`ed live carries these already and needs no
rebuild. The patch mechanism above is what makes them survive an image
**rebuild** (`docker build sjtu_project/falcon_docker`) or a fresh FALCON clone
— and it is the only thing that does, which matters on a machine that never had
the committed image. `falcon-ros-custom:v1` was rebuilt from scratch this way on
PCN87653 on 2026-08-13.

## NEVER rebuild `voxel_mapping`. Its binary is not reproducible from its source

`falcon-ros-custom:v1` ships a `libvoxel_mapping.so` that **cannot be rebuilt
from the source in the same image**. Running `catkin_make --pkg voxel_mapping`
produces a library that compiles, links, loads and runs, and silently maps the
world worse. Nothing warns you.

Measured on 2026-08-14. The warehouse finishes in about two minutes at
201–202 m³ and had done so on ten consecutive campaign legs. One
`catkin_make --pkg voxel_mapping` later, the same commit, the same world and the
same parameters gave 98–139 m³ on **twenty-two consecutive legs**, in both
worlds, with the aircraft exploring away from the building and out through its
own flight box. Re-tagging the pre-rebuild image and flying it unchanged
restored 202.01 m³ in 123 s on the first attempt.

The cause is that some fix in `voxel_mapping` exists only in the shipped binary.
The image has been `docker commit`ed live several times in its history (the
header of `fix_falcon_sparx_patches.sh` records two such fixes being folded into
patches later), and at least one edit was never captured as a patch or as source.
Rebuilding the package discards it. The source tree looks complete — the depth
overflow resizes and the `publish_bulk` cadence are both present — so inspection
does not reveal what is missing.

**Consequences for anyone changing FALCON here.**

- Build only the packages you actually edited, and never name `voxel_mapping`.
  `catkin_make --pkg trajectory exploration_preprocessing exploration_manager`
  is safe: those report `Built target voxel_mapping` because it is *up to date*,
  which is not the same as rebuilding it.
- Before concluding that a code change caused a regression, check what the image
  changed. Two hours went into eliminating the repository, the world, the spawn,
  the real-time factor, the machine, the GPU, the Docker daemon and the Gazebo
  model count, all correctly and all irrelevant, because an intermediate image
  had been misidentified: the layer tested as "the last good image" carried a
  timestamp four minutes *after* the last good run finished, and was in fact the
  first bad one. **Check `docker images -a` timestamps against the run's own
  artifacts before trusting a bisect.**
- Docker keeps every intermediate layer, and that is what made the recovery
  possible. Do not prune while an investigation is open.

The real repair is to find what the binary has and the source does not, and cut
it as a patch so the image becomes reproducible. Until then a from-scratch
`docker build` of `falcon_docker/` is **unverified**: it would rebuild
`voxel_mapping` from source like any other package, and on this evidence would
produce the degraded mapper.

## Iterating on a patch against an image that already has it

The fix script is idempotent **by sentinel**, which is exactly right for a fresh
clone and exactly wrong for iteration: once a patch is in the image its sentinel
is present, so editing that patch and re-running the script logs
"already applied, skipping" and changes nothing. A rebuild that reports success
while silently shipping the old code is the failure mode this whole mechanism
exists to prevent, so it is worth stating plainly.

To iterate, copy the edited sources straight into a scratch container built from
the current image, rebuild the affected packages, and `docker commit`. Keep the
patch file in step as the authoritative artifact for the Dockerfile path, where
FALCON is cloned fresh and every patch applies to a pristine tree.

Two things to check when rebuilding by hand:

- **Build every package the change touches.** `frontier_finder.cpp` lives in
  `exploration_preprocessing`, not `exploration_manager`; a `catkin_make --pkg`
  list that omits it builds cleanly and ships nothing.
- **Verify the sentinel in the artifact it actually lands in.** These strings
  compile into `libexploration_preprocessing.so`, not into the
  `exploration_node` binary. Grepping the wrong file fails the verification
  step and, with `set -e`, aborts before the commit.

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

