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

## falcon_blocked_region_widen.patch — a shadow the frontier can actually feel

Touches `frontier_finder.{h,cpp}`, on top of `falcon_blocked_region_ttl.patch`.
Three changes that only make sense together.

**Why.** The dead-end guard shadows the VIEWPOINT the aircraft could not reach,
but `computeFrontiersToVisit()` retires a cluster only when a shadow falls within
`blocked_region_radius` of the cluster's AVERAGE — and `sampleViewpoints()`
places viewpoints `candidate_rmin..rmax` (1.0-2.5 m) from that average *by
construction*. At the shipped 2.0, under `candidate_rmax`, the test can never
fire for the frontier that produced the blocked viewpoint. The frontier
survives, samples another viewpoint centimetres away, and the tour offers it
forever. The TTL ladder cannot help: a shadow matching nothing is refreshed at
no cost. Measured twice on 2026-08-15 in different halves of the hospital — one
run aimed at three viewpoints of a single frontier 91/50/28 times, another at one
point 95 times while the guard reported the aircraft confined short of it four
separate times.

**And the test was only ever applied to freshly detected clusters.** The
categorisation loop runs over `tmp_frontiers_`, which `searchFrontiers()` fills
only from clusters that `haveOverlap` the update box AND `isFrontierChanged`. A
frontier the aircraft is stuck staring at has stopped changing, so it sits in
`frontiers_` untouched: a shadow cast on an already-known frontier could not
retire it at ANY radius, on any strike. Measured: `(-11.60, 9.89)` escalated to
strike 6 with a 3.5 m shadow held for 2880 s while the tour chose its frontier
sixty times in the last 300 s.

**What it does.**

- `blockedRadiusFor(strikes)` puts the radius on the same doubling ladder the
  TTL already uses, capped by `/frontier_finder/blocked_region_radius_max`
  (3.5). Strike 1 stays at 2.0 m and lifts in 90 s, which is what protects a
  transit corridor from one early mistake — the 253.65 m³ failure this file
  records. From strike 2 the shadow exceeds `candidate_rmax` and the frontier
  retires. One escalation law, two quantities, and the wider width is *earned*.
- `sweepBlockedFrontiers()` sweeps the existing `frontiers_` list at the top of
  `computeFrontiersToVisit()`, before `first_new_ftr_` is assigned (it erases
  from the list that iterator points into). It does NOT unflag the retired
  cells: `searchFrontiers()` recycles a dormant cluster through `resetFlag` once
  its region is observed and changes, so this retires rather than sterilises.
- `grantFinishAmnesty()` refuses to let the mission call the world finished
  while shadows are still standing. A FINISH means "the frontier set is empty",
  and every shadowing mechanism suppresses frontiers, so that verdict with
  shadows outstanding means "nothing I am currently willing to consider" — on
  evidence collected from an earlier pose against a much poorer map. It drops
  every shadow, re-offers every retired cluster, and relaxes viewpoint placement
  for that one pass. Bounded by `/frontier_finder/finish_amnesty_max` (2).

**Two traps, both paid for.** `sampleViewpoints()` opens with
`CHECK_EQ(frontier.viewpoints_.size(), 0)`, and the clusters the sweep retires
come off `frontiers_` still carrying their viewpoints — handing one back
unmodified aborts the node (`exit -6`, `Frontier already has viewpoints
(4 vs. 0)`) and takes the voxel map with it. And the strike HISTORY is
deliberately not cleared by the amnesty, so a pocket that has already defeated
the aircraft starts its second life at the strike count it earned and retires
faster, which is what keeps termination.

## falcon_deadend_looping.patch — a guard that can see a moving failure

Touches `exploration_manager.cpp`, on top of `falcon_deadend_guard.patch`.

**Why.** The guard fires when the aircraft's excursion stays under 2 m for 25 s
while short of its viewpoint. That threshold is smaller than this stack's own
recovery ladder: a contact retreat backs out 1.3 m, an unstick runs 0.20 m/s for
6 s, so one strike-retreat-reapproach cycle spans about 2.6 m and reads as
healthy. Measured: an aircraft logging *"commanded 1.9 m of travel and moved
0.06 m"* while the tour aimed at one viewpoint forty times, with the guard
silent throughout. In the run after this patch the old span test fired ZERO
times all mission while the new one fired seven — it is the failure mode, not a
corner case.

**What it does.** Adds the signature that separates the two cases: a transit
spends its path on displacement and on closing the gap to the target; a loop
spends it on neither. Fires when the aircraft has flown over 5 m, more than
twice its net displacement, and ended the window FURTHER from the viewpoint than
it started.

**Both thresholds were wrong once, and both corrections are the point.**

- `closed < 0.5 m` retired an aircraft that was genuinely arriving, only slowly
  — two of seven fires in one run were "closed 0.49 m" and "closed 0.33 m". This
  aircraft plans at 0.15 m/s and legitimately crawls at 0.34-0.61x that through a
  doorway, and rounding a corner converts path into almost no straight-line
  closure while being exactly the right thing to do. Only a NEGATIVE closure
  admits no innocent reading.
- A 25 s window shared with the pinned test was too impatient. "Has not moved at
  all" is conclusive; "moved but got no nearer" is also what a hard doorway looks
  like while the aircraft is still winning — the run that mapped the most of the
  building spent about 300 s looping at (-10.5, -2.5), escaped unaided and went
  on to 732.9 m³. The looping test gets 60 s; the pinned test keeps 25 s.

## falcon_ccl_slice_height.patch — build the tour's world where the drone flies

Touches `hierarchical_grid.cpp`.

**Why.** Under `map_dimension` 2 the coverage tour reduces every hgrid cell by
connected-component labelling on ONE horizontal slice, and its height was the
literal `double height = 1.0;` in `getCCLCenters2D`. On a hospital flight box of
z 0.9-1.6, with the aircraft flying 1.15-1.49 m, that slice sits 0.1 m off the
box floor — inside the layer the building keeps its beds, gurneys, wheelchairs,
sinks and carts in (0.5-1.2 m). Every one of them reads to the tour as a wall at
an altitude the drone is never at, so the tour believes regions are disconnected
that the aircraft crosses without noticing.

**What it does.** Makes it `/map_config/ccl_slice_height`, defaulting to 1.0 so
an unset config is upstream behaviour; `exploration.launch` sets 1.25, the middle
of the usable band and the altitude the follower already calls mid-band. Read
once per call via `ros::param::param` into a function-local static.

**It must stay inside the flight box**, for the same reason `box_min_z` may not
exceed 1.0: a slice outside the box selects no voxels at all and the tour's model
of the world goes blank rather than merely wrong.

## falcon_sjtu_session.patch — the cumulative diff, and how it relates to the rest

Every other file here is ONE change with its own reasoning. This one is the
whole of `falcon_planner` as the running `falcon-ros-custom:v1` differs from the
pristine upstream clone, regenerated from the images themselves
(`v1-pre-patchset` against `v1`).

It exists because the per-change files cannot be applied blind in sequence any
more: several of them touch the same functions, so their hunk offsets only line
up against the tree as it stood when each was written. To rebuild the image from
nothing, apply this one. To understand or revise a single decision, read the
file named for it.

Keep both in step. If you change FALCON's C++ again, regenerate this from the
container as well as writing the per-change patch, or the two will disagree and
the cumulative one is the one a rebuild will believe.

## Rebuilding vs. the running image

An image that was `docker commit`ed live carries these already and needs no
rebuild. The patch mechanism above is what makes them survive an image
**rebuild** (`docker build sjtu_project/falcon_docker`) or a fresh FALCON clone
— and it is the only thing that does, which matters on a machine that never had
the committed image. `falcon-ros-custom:v1` was rebuilt from scratch this way on
PCN87653 on 2026-08-13.

## Do not add a bounds guard to `MapServer::getOccupancy`. It was tried and it broke both worlds

`checkTrajCollision()` walks up to six metres along a trajectory asking
`getOccupancy()` about every sample, and `getOccupancy()` indexes the grid
behind a glog `CHECK` with no bounds test, so a sample that leaves the box kills
the process and takes the voxel map with it. That crash is **real** and was
measured twice in five hospital runs:

```
ExplorationFSM::safetyCallback() -> FastPlannerManager::checkTrajCollision()
  -> MapServer::getOccupancy(Position) -> MapBase::getVoxel(int) -> CHECK -> abort
```

The obvious repair — `if (!isInBox(pos)) return OccupancyType::UNKNOWN;`, using
the `isInBox()` defined on the line above — is **wrong, and catastrophically so**.
Measured 2026-08-14: with it, twenty consecutive legs failed, the warehouse
falling from 201–202 m³ (ten straight finishes) to 98–139 m³ and the hospital
never reaching half the building, in both worlds, with the aircraft exploring
away from the building and out through its own flight box.

`getOccupancy()` is not only the collision check's accessor. Frontier detection
and coverage read it too, and `isInBox()` delegates to the **TSDF's** box, which
is not the occupancy grid's extent. Returning UNKNOWN outside it silently
redefines what the planner considers unexplored, and exploration collapses. The
crash is worth fixing, but the fix belongs in the *caller* clamping its samples,
not in an accessor that changes meaning for everyone reading it.

**Two process lessons from the same episode, both expensive.**

- *Check image timestamps against the runs' artifacts before trusting a bisect.*
  The image was "eliminated" early by re-tagging an intermediate layer and flying
  it, which is the right method — but the layer chosen was stamped four minutes
  **after** the last good run finished, so it was the first bad image, not the
  last good one. That single misreading sent hours into eliminating the
  repository, the world, the spawn, the real-time factor, the machine, the GPU,
  the Docker daemon and the Gazebo model count, all correctly and all
  irrelevant, and produced a confident write-up here claiming
  `libvoxel_mapping.so` could not be rebuilt from its own source. It can:
  `catkin_make --pkg voxel_mapping` on unchanged source yields a **byte-identical**
  library, same md5. That claim was published and is retracted.
- *Do not delete intermediate images while an investigation is open.* Both
  degraded layers were removed as tidy-up before the diff that would have
  explained precisely why the revert of the guard did not restore behaviour, so
  that question is now unanswerable. Recovery came from re-tagging the last
  pre-guard image, `25923b082552`, which is what `falcon-ros-custom:v1` is today.

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

