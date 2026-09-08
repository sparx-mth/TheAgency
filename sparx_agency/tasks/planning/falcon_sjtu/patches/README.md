# FALCON source patches (falcon_sjtu)

The FALCON planner is upstream C++, cloned and compiled into the
`falcon-ros-custom:v1` image. Anything we change *inside FALCON itself* must be
reproducible from a clean image build — otherwise a rebuild or a fresh container
silently drops it. Each such change is a patch here, applied by the image build.

## Where the build actually applies them

The image is built by **`sjtu_project/falcon_docker/Dockerfile`** (an external
repo), which `git clone`s FALCON (`ros1-noetic`) and then runs a series of
self-verifying `fix_falcon_*.sh` scripts before `catkin_make`.

**IT DOES NOT APPLY THE PATCHES IN THIS DIRECTORY.** Verified against the
Dockerfile: the only FALCON edits a clean build performs are `system_info`,
`cost_check`, `depth_overflow`, `deadend_guard`, `simtime`, `visgrid_cadence`,
`hgrid_clamp`, `inflate_astar_by_airframe`, `replan_from_measured_state` and
`sop`. Nothing references `sparx_patches/`, no `fix_falcon_sparx_patches.sh`
exists, and neither does the `sparx_patches/` directory itself. An earlier
revision of this file described that wiring as if it were in place; it never was.

So the running `falcon-ros-custom:v1` is a **`docker commit` lineage**, not a
reproducible build, and everything here beyond that list lives only in its
layers. The patches in this directory are the record that makes those layers
reproducible **by hand** — the authoritative text of each change, and
`falcon_sjtu_session.patch` as the one-shot cumulative diff — but a `docker
build` will not read them.

**Two consequences worth stating plainly.** A rebuild from the external
Dockerfile silently produces a FALCON without any of this session's work, which
looks like a regression with no diff to explain it (this is the "stale image on a
second machine" trap). And the lineage is one machine deep: if these image tags
are lost, the only route back is applying `falcon_sjtu_session.patch` to a fresh
clone and rebuilding.

Wiring the patches into the Dockerfile — a `sparx_patches/` copy plus a
`git apply`-and-verify step that fails the build rather than shipping an
unpatched planner — is the fix, and it is not done.

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

**That 2.6 m excursion figure has since bitten a second time, in our own code.**
A give-up rule for the follower's yaw probe was built to stop re-probing after N
consecutive "not wedged" verdicts, and it reset its counter on 1.5 m of travel
since the last clear — below the ladder's own span, so it reset every cycle and
never once fired, including on a leg that ended `ABORT_NO_GROWTH` at 802.21 m³.
It has been deleted; the measurement behind it (four hospital legs, ~25 probes,
ZERO findings that the aircraft was actually held) and the anchor requirement are
written up in `adapter/launch/bspline_follower.launch`. **Any threshold compared
against aircraft displacement in this stack must sit above ~2.6 m**, or it is
measuring the recovery ladder rather than the failure.

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

## falcon_finish_amnesty_gate.patch — gate the amnesty where the list actually empties

Touches `frontier_finder.cpp`.

**Why.** `grantFinishAmnesty()` refuses to let the mission call a world finished
while shadows are standing and clusters sit retired behind them. It was gated on
`frontiers_.empty() && tmp_frontiers_.empty()`, tested **before** the loop that
categorises `tmp_frontiers_` — so it could only fire on a cycle where nothing was
detected at all. The failure it exists to catch is the opposite one: clusters
*are* re-detected every cycle and every one of them is categorised *into*
dormant, so `frontiers_` only becomes empty once the loop has already run, and
the gate is behind us by then. Measured, all three with the gate in front:

| leg | verdict | dormant at FINISH | shadows | amnesties granted |
|---|---|---|---|---|
| `loop/wviz` warehouse | FINISHED 349.8 m³ | 493 | 2 | **0** |
| `accept9/r02_hospital` | PARTIAL_FINISH 284.1 m³ | 10 | 1 | **0** |
| `accept9/r03_hospital` | PARTIAL_FINISH 300.4 m³ | 36 | 1 | 1 of 2 |

**What it does.** Moves the gate behind the categorisation loop. The loop body
becomes a `categorise` lambda that clears `tmp_frontiers_` when it is done — each
cluster is by then either promoted into `frontiers_` or retired into
`dormant_frontiers_`, and none may be categorised twice — driven by a two-pass
loop that grants the amnesty between passes. Two passes rather than a `while`:
`grantFinishAmnesty()` is already bounded by `finish_amnesty_max_`, and a second
pass that produces nothing must still be allowed to end the mission.

**The old pre-loop gate is deleted, not kept alongside.** The new one strictly
subsumes it — an empty `tmp_frontiers_` makes pass 0 a no-op and the gate fires
on the same cycle — and leaving both in could spend two of the two permitted
amnesties within a single cycle.

**It stays inside the `.cpp` on purpose.** The obvious shape is to extract the
loop body into a private method, but that means a new member in a header shared
by four packages, and rebuilding only the owning package against it is an ODR
violation that corrupts the heap at startup and ships an empty voxel map. A
lambda gets the same structure with no header change and no cross-package
rebuild.

## falcon_raycast_out_of_map.patch — a depth ray that leaves the map must not kill the mapper

Touches `voxel_mapping/src/tsdf.cpp`, which **no other patch in this directory
touches** and which was still the pristine upstream file (dated 2026-05-12)
until this landed. Ported verbatim from
`tasks/planning/falcon_pegasus/patches/fix_falcon_raycast_out_of_map.sh`, where
it has been in force since 2026-08 — the two deployments share the defect and
this one had simply never received the fix.

**Why.** `TSDF::inputPointCloud` walks every voxel between the sensor and each
returned depth point with a DDA raycaster and turns each one into a flat array
address with no bounds check. The ray's *endpoints* are clamped into the map,
but the clamp is in **metres** while the stepping is in **voxels**, so the DDA
can still land one index past a face on the way there. Neither consumer checks
the address: `updateTSDFVoxel` writes the data array directly, and
`updateOccupancyVoxel` reads through a glog `CHECK` that calls `abort()`.

On Pegasus it surfaced as the CHECK; here it surfaces as a raw segfault in the
writer, and it is worse on this stack for the reason everything is worse on this
stack — **`exploration_node` owns the voxel map**, so the mapper dying does not
interrupt the mission, it *resets* it. Measured on the hospital, 2026-08-16:

```
Stack trace (most recent call last) in thread 22475:
#0  Object "/catkin_ws/devel/lib/libvoxel_mapping.so", at 0x716321b67c22, in
    voxel_mapping::TSDF::updateTSDFVoxel(int const&, double const&, double const&)
Segmentation fault (Address not mapped to object [0x715c8adee9b8])
[exploration_node-3] process has died [pid 112, exit code -11]
```

at t = 1563 s of a healthy leg — occupied voxels **468,485 → 9,159** in one
second, coverage 543 m³ at t = 951 s ending as `TIMEOUT` at **194.34 m³**. The
run had been gaining 32.6 m³/min with a plan-origin gap of 0.12 m and nine
dormant frontiers; nothing was wrong with it except that the mapper died.

**What it does.** Guards the loop rather than the two callees: test the voxel
index where it is produced, using the `isInMap(VoxelIndex)` predicate that
already exists, and skip it. One condition per raycast step fixes both
consumers, and a voxel outside the map is somewhere the map cannot record
anything about, so nothing is lost by skipping it. The same file already applies
exactly this policy one function away in `OccupancyGrid::setOccupancy`, so this
is the existing policy applied on the path that was missed, not a new one.

**Verification.** There is no string sentinel to grep — the change compiles to a
comparison, not a message. Verify instead that `libvoxel_mapping.so` differs
from the pre-patch build (measured `6692f4ab…` → `4a67ba3a…`) *and* that the
container's `tsdf.cpp` carries `isInMap(voxel_idx)`. Both are checked at build
time.

> **Build note, and it is this machine rather than the patch.** Linking
> `voxel_mapping` failed once with `collect2: fatal error: ld terminated with
> signal 11` and succeeded on a retry at lower parallelism. That is the known
> intermittent toolchain flakiness recorded in `falcon_pegasus/README.md`, not a
> defect in the change. Retry the link before diagnosing anything.

## falcon_vp_audit.patch — say WHY a cluster retired

Touches `frontier_finder.cpp` only. Pure instrumentation: no behaviour changes,
nothing is accepted or rejected differently (sentinel: `[vp_audit] retire`).

**Why.** A frontier cluster retires to dormant when `sampleViewpoints()` returns
nothing, and *three* entirely different failures wear that one verdict:

1. no candidate POSITION survives the box / occupied / unknown / clearance
   tests, so visibility is never even evaluated;
2. positions exist but every ray to the frontier is judged blocked;
3. positions exist and the frontier is visible from them, but not *enough* of it.

They need opposite fixes, and from outside a premature FINISH looks identical
either way. This package spent a long time acting on (1) — the finish amnesty
exists to relax the margin to unobserved space — on the strength of a diagnosis
that could not be checked against anything. Same shape as the follower's
binding-limiter problem, and the same answer: attribute, do not guess.

**What it adds.** A file-scope histogram, reset per `sampleViewpoints()` call,
counting every candidate rejection by cause, every ray by outcome, and the best
visibility any legal position achieved; and one `ROS_WARN` per retirement
carrying all of it plus the cluster's position and size.

It stays at file scope on purpose. `frontier_finder.h` is included by four
packages, and adding a member to it while rebuilding only the owning package is
the ODR violation that corrupts the heap at startup and ships an empty voxel map
— the same trap `falcon_finish_amnesty_gate.patch` avoids with its lambda.

`rig/vp_audit.py` reads the lines back and aggregates them.

**Result** (hospital, 2026-08-16, first run carrying it). The measurement is
unambiguous and it refuted both standing theories, including the one this
directory had written down:

| | observed |
|---|---|
| retirements with NO legal standing position | **0** |
| candidate positions surviving every test, per cluster | **11–72** |
| retirements where nothing at all was visible | **0** |
| best visibility achieved, per cluster | **6–15 cells** |
| the bar (`min_visib_num`) | **15** |
| cluster sizes (downsampled) | **18–29 cells** |

Every cluster had somewhere to stand, saw the frontier from there, and was
retired on the visibility bar alone. See the next patch.

## falcon_visib_unknown_tolerance.patch — a frontier is not invisible for being a frontier

Touches `frontier_finder.cpp`, on top of `falcon_vp_audit.patch` (sentinel:
`[visib_unknown]`). **This is the fix for the premature FINISH.**

**Why.** `countVisibleCells()` treats UNKNOWN as an occluder, exactly like a
wall. But `searchFrontiers()` only promotes a cell that is knownfree AND has an
unknown neighbour — a frontier cell *is* the boundary of unobserved space — so
a ray arriving at one crosses unobserved voxels by construction. The test
therefore reports "you cannot see this frontier" **because** it is a frontier.

Measured across three warehouse legs with the tolerance at 0: only **6.8% of
rays arrived clear**, against 44.9% rejected on UNKNOWN and 44.1% on real
structure. **43 clusters — 16% of every retirement — had somewhere legal to
stand and ZERO measured visibility from anywhere.**

Those are unrecoverable by every mechanism this package already has. The finish
amnesty drops the visibility bar to zero, and no bar of any height saves a
cluster that nothing can see. They stay dormant, the frontier set empties, and
FALCON declares a half-mapped world finished. That is the mechanism behind the
277–432 m³ `PARTIAL_FINISH` legs, and it is why
`falcon_blocked_region_widen.patch` measured its amnesty re-offering 52 retired
clusters and every one returning straight to dormant.

**What it does.** A ray may cross a bounded prefix of unobserved voxels and no
more. The raycast runs **from the frontier cell toward the viewpoint**, so the
tolerated voxels are exactly the boundary layer that makes the cell a frontier;
unknown encountered further along is a genuinely unexplored volume being looked
through, and still blocks. **OCCUPIED blocks at any distance**, which is what
keeps a sealed cavity unviewable — a ray into the warehouse's hollow
`ClutteringC` crates crosses the shell first.

`/frontier_finder/visib_unknown_tolerance` (2 voxels = 0.20 m at the 0.10 m map
resolution; **0 restores upstream exactly**), declared in
`adapter/launch/exploration.launch`.

**Result**, A/B on the warehouse — same binary, only the rosparam differing,
three legs per arm (`rig/tolerance_ab.sh`):

| | tolerance 0 | tolerance 2 |
|---|---|---|
| cluster retirements | 267 | **147** |
| retired for "nothing visible" | 15.4% (41) | **0%** |
| legal stance, ZERO visibility | **43** (16.1%) | **0** (0.0%) |
| rays arriving clear | 6.8% | **9.2%** |
| legs finished | 2 of 3 | 2 of 3 |
| contacts (total across arm) | 263 | 263 |

The class this exists to eliminate goes to zero, retirements fall 45%, and the
warehouse finish rate and contact count are unchanged — it costs nothing in the
world that had the most to lose. On the hospital the same configuration reached
**812.02 m³**, still gaining when the mission time cap cut it; read that against
the package README's standing warning to cap the hospital at 3900 s or more
before calling a TIMEOUT a failure to map.

**A rejected alternative is recorded rather than deleted.** Making the
*visibility bar* relative to the cluster size was tried first, on the reasoning
that an absolute count of 15 asks 89% of an 18-cell cluster and 16% of a
100-cell one. It is a real observation and it is not the fix: flown at 0.25 it
turned a warehouse FINISH into a pinned abort, because relaxing the bar admits
frontiers bounding the `ClutteringC` crates' permanently unobservable interior
cavities and the aircraft grinds across the crate tops chasing them (measured
pinned at z = 1.78 m on a 1.79 m crate). The general lesson is worth keeping:
**the visibility bar was also, accidentally, the filter that rejected frontiers
which can never be resolved at all** — lower it and you inherit that second job.
The patch was deleted; this paragraph is what survives of it.

## falcon_open_visib_bar.patch — relax the visibility bar only where there is room

Touches `frontier_finder.cpp`, on top of `falcon_visib_unknown_tolerance.patch`
(sentinel: `[open_bar]`).

**Why.** `min_visib_num` (15) is an absolute count of downsampled frontier
cells, but the most any viewpoint CAN see is the cluster's own size. Against a
100-cell cluster it asks a few percent; against an 18-cell one it asks 16 of 18.
The bar is hardest exactly where the cluster is smallest, and small leftover
clusters are what remains late in a mission. Measured on two hospital legs:
**96.9% and 100% of every cluster retirement was "saw something, under the
bar"**, one leg ending with 46 clusters still dormant.

**A global relaxation was tried first and regressed the warehouse**, turning a
finish into a pinned abort at 364.8 m³. The mechanism is specific: relaxing the
bar admits frontiers bounding the hollow `ClutteringC` crates' sealed interior
cavities, which can never be observed and so never stop being frontiers, and the
aircraft grinds across the crate tops chasing them (measured pinned at z = 1.78 m
on a 1.79 m crate).

**What it does.** The discriminator was already in `sampleViewpoints`: those
cavity candidates are all `isNearOccupied`, wedged against the shell they are
trying to see past, while a frontier in an open hospital room is not. So the
relative bar applies **only to candidates with real clearance**; anything near
structure keeps upstream's full absolute bar.

```
bar = near_occupied ? min_visib_num
                    : max(open_visib_floor,
                          min(min_visib_num, open_visib_fraction * cluster_cells))
```

`/frontier_finder/open_visib_fraction` (**0.5**) and `open_visib_floor` (4),
declared in `adapter/launch/exploration.launch`. **0 restores upstream
everywhere.** The bar can only ever be relaxed, never tightened.

**Result**, three warehouse legs and two hospital legs per arm, same image:

| | finishes | coverage | contacts | elapsed |
|---|---|---|---|---|
| warehouse, 0.5 | **2/3** | 335.0 m³ | **38** | **425 s** |
| warehouse, off | 1/3 | 387.0 m³ | 92 | 510 s |
| hospital, 0.5 | **1/2** | **816.6 m³** | 145 | **2277 s** |
| hospital, off | 0/2 | 763.3 m³ | 46 | 2358 s |

More finishes in **both** worlds, more hospital coverage, faster in both, and
warehouse contacts more than halved. The hospital leg that finished did so at
**809.11 m³ with 9 contacts** — the first hospital FINISH recorded in this
campaign, on FALCON's own verdict.

**The number that looked worse, and what it turned out to be.** Hospital contact
*reports* averaged 145 against 46, which read as a safety cost. Counted the way
this package says to count — **objects touched, not reports** — it reverses:

| leg | reports | objects |
|---|---|---|
| 0.5, leg 1 | 280 | **5** |
| 0.5, leg 2 | 9 | **1** |
| off, leg 1 | 73 | **6** |
| off, leg 2 | 19 | **2** |

Mean 3.0 objects against 4.0: the treatment touched FEWER. The 280 reports are
186 of them on a single `IVStand_2` — one object grazed repeatedly, exactly the
"one five-second graze reads as eight contacts" pattern recorded elsewhere in
this file. Set the fraction to 0 to fall back with no rebuild.

> This was nearly a wrong verdict, and the reason is worth keeping: the A/B
> harnesses were reading `contacts` out of `verdict.json`, which is the raw
> report count, while the object count existed only in `both_worlds.sh`'s console
> line. `campaign_run.sh` now writes `contact_objects` into the verdict, and both
> harnesses print OBJECTS as the headline with reports as context.

**A fraction of 0.25 was also measured and rejected**: it finished 3/3 on the
warehouse but at a mean of 258 contacts against the control's 92, buying finishes
by trading safety. 0.5 admits far fewer marginal clusters and was better on every
axis.

> **Reproducibility note.** Verifying this patch found that the refuted 3-D
> viewpoint sampling code from `falcon_visib_unknown_tolerance`'s session was
> sitting in the image with **no patch file recording it** — inert behind a
> parameter, but a planner-crashing path one rosparam away. It has been removed
> from the source outright rather than left dormant. The four patches in this
> directory now apply in sequence to a pristine tree and reproduce the running
> image's `frontier_finder.cpp` byte for byte (md5 `de6f5a21…`), which is checked
> rather than assumed.

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


## falcon_room_confine.patch — a leased fence FALCON actually respects

Touches `pathfinding/astar.{h,cpp}`,
`exploration_preprocessing/{include/…/frontier_finder.h,src/frontier_finder.cpp}`
and `exploration_manager/src/exploration_fsm.cpp`. Built into
**`falcon-ros-custom:v2-confine`**; `v1` is untouched and still the default.

**Why.** An object search wants FALCON to map ONE room and stop there, so the
host can pick the next room by its own ranking. Nothing in the shipped image can
express that. Verified against the image rather than the launch files, which
document intent instead of the binary:

* `/map_config/keep_out_boxes` and `/astar/no_go_runtime` are dereferenced ONLY
  inside the `isValidVoxel` lambda of the five-argument
  `Astar::search(start, end, MODE, bbox_min, bbox_max)` — and **no caller
  anywhere in the workspace uses that overload.** Every live search goes through
  `search(p1,p2)` / `searchBBox` / `searchUnknown*`, which validate via
  `Astar::isBlockedInflated` or a raw `getVoxel == OCCUPIED`. Both lists are
  dead code at runtime.
* Even reached, `no_go_` sits behind `if (inflate_enabled_)`, and
  `search()`/`searchBBox()` deliberately retry with that false on `NO_PATH` — so
  it is suppressed at exactly the moment it is the only thing between the
  aircraft and the door. A preference, never a fence.
* `/frontier_finder/blocked_regions_runtime` is read only in the FrontierFinder
  constructor, rewritten wholesale from memory on every replan, emptied by
  `grantFinishAmnesty`, and deleted every second by `mission_watchdog_node`.
* **Injecting voxels cannot work either**, which is the intuitive approach and
  the one worth writing down. `ESDF::updateLocalESDF` writes only a SMALLER
  distance (`if (dist < value) value = dist`) and no runtime path ever raises
  one, so any voxel ever marked occupied leaves the ESDF dented at 0.0 for the
  rest of the mission — and the bspline optimiser, the `searchUnknown*` family
  and `isFreeInflated` all keep reading that dent long after the block is
  lifted. Blocking a doorway once would degrade it permanently.

**What it does.** A keep-**in** list (`/map_config/keep_in_runtime`) plus door
seals (`/map_config/keep_out_runtime`), armed by a deadline
(`/map_config/confine_deadline`) re-read once a second.

*Why a keep-in and not just door blocks.* Sealing the doors alone does not
confine: `FrontierFinder` retires only clusters whose `average_` is inside a
box, so every frontier in the rest of the building survives, the coverage tour
keeps choosing them, A\* returns `NO_PATH` at the seal, `PLAN_TRAJ` loops on
`FAIL`, and the publish-fail blacklist starts casting shadows over the very
doorway the aircraft must fly back out through. The inclusion list retires the
out-of-room clusters, which stops FALCON *wanting* to leave. The seals are still
needed because a room's axis-aligned bbox leaks through its own doorways.

*Where the test goes.* `confineBlocks` is the first statement of
`Astar::isBlockedInflated` and is **not** gated on `inflate_enabled_` — the one
property `no_go_` lacks, and the whole difference between a fence and a bias.
`FrontierFinder::confineRetires` is ORed into the four existing `insideKeepOut`
tests and has **no** start exemption, because the finder must never offer a
cluster outside the room however close the aircraft is to it.

*Fail-open, always.* The deadline is read FIRST and the geometry is not read at
all when the lease is dead, so a dead host, a dropped bridge or a killed shim
all end in NO confinement rather than an aircraft fenced in for ever. Measured
in flight: the aircraft held station for the whole lease and resumed flying on
the exact second the deadline passed. A\* keeps a `confine_start_exempt` radius
around the aircraft so a drone that drifts into a sealed doorway can still plan
its way out.

**Two things that would each have killed the first flight.** `CHECK_EQ(frontiers_
.size(), 0)` in `exploration_fsm.cpp` SIGABRTs the exploration node — and the
whole voxel map with it — whenever the hgrid collapses while frontiers stand,
which is the NORMAL state inside a confined room; it is demoted to a warning.
And `FINISH` is terminal upstream, which is right for a one-shot survey and
fatal for a room-by-room search, where the FIRST room exhausted ends the mission
for the building; `frontierCallback` now returns to `PLAN_TRAJ` when frontiers
are reachable again, behind `/fsm/resume_from_finish` (default false, so a plain
survey is bit-identical).

**Companion changes outside this patch**, all required together: the follower's
`_finished` latch is cleared by a new trajectory (and the watchdog abort split
onto its own `_aborted` flag, which stays terminal); the watchdog's own
`_finished` stands back up when FALCON plans again; and
`adapter/scripts/room_confine_node.py` turns the bridged
`/scene_graph/confine` topic into the three rosparams, because rosparams do not
bridge.
