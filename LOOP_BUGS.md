# LOOP_BUGS — the live problem queue

Companion to `LOOP_MISSION.md` (immutable) and `LOOP_FIXES.md` (what was done about these).
**Edit this file freely.** Ranked: top of the list is what the loop works on next.

Status vocabulary: `OPEN` · `DIAGNOSED` (mechanism known, no fix yet) · `FIXING` (change in flight,
verdict pending in LOOP_FIXES.md) · `CLOSED` (fixed and verified over enough flights) ·
`REFUTED` (the symptom was real, this explanation was not).

## Index

*Newest findings are at the bottom of the file; this table is the map.*

| id | problem | state |
|---|---|---|
| **B1** | The nose is aimed by a heuristic, not by the planner | OPEN |
| **B2** | The airframe is 3.7x slower sideways than forward, and nothing knows it | DIAGNOSED |
| **B3** | The velocity-damping gain is justified by an inner loop that does not exist | DIAGNOSED |
| **B4** | Nothing in the loop models the plant's asymmetric response | OPEN |
| **B5** | A wall-pinned aircraft: detection WORKS, the gap is the recovery | REFUTED |
| **B6** | FALCON's terminal plan-fail lock | DIAGNOSED |
| **B7** | Altitude runaway | OPEN |
| **B8** | Cold-start deadlock in bring-up | CLOSED |
| **B9** | A failed cycle can inherit the previous flight's telemetry | OPEN |
| **B10** | Mapping rate is not instrumented | DELIVERED |
| **B11** | The velocity the servo closes on is 0.20 s late | CONFIRMED |
| **B12** | The 1.667× mid-curve compression is a deliberate trade, and it is a GOOD one | CLOSED |
| **B13** | Yaw and position run on different clocks after a rescale | DIAGNOSED |
| **B14** | Yaw rate is unconstrained everywhere in FALCON | DIAGNOSED |
| **B15** | `traj_server` freezes the endpoint but keeps reporting READY | DIAGNOSED |
| **B16** | The viewpoint gain model uses the wrong camera | DIAGNOSED |
| **B17** | Time allocation is uninitialised for every segment after the first | DIAGNOSED |
| **B18** | Two of A*'s three failure exits are silent | DIAGNOSED |
| **B19** | `frac_of_box` is inflated | CONFIRMED |
| **B20** | Today's flights map 3–4× more than yesterday's, and nothing in this loop explains it | OPEN |
| **B21** | `config.py` is imported in three environments and one of them has different paths | CLOSED |
| **B22** | Sphera intermittently publishes a SECOND, bogus pawn on `/R1/sphera/state` | CONFIRMED |
| **B22b** | Where the duplicate pawn actually lives (for the operator) | OPEN |
| **B23** | The dual publisher makes the rangefinder alternate, and that is very likely Finding K | CONFIRMED |
| **B24** | Localization can latch onto the impostor pawn and fly blind at full stick | CONFIRMED |
| **B25** | The position correction is 42–44 % of the command, violating the tracker's own design rule | CONFIRMED |
| **B26** | The collisions are against geometry the map never had, and the fix is not in the controller | DIAGNOSED |
| **B27** | A* compute budgets: what the diagnostics actually show | REFUTED |
| **B28** | The operator's headline metric is too noisy to A/B at any reasonable n | MEASURED |
| **B29** | Depth preprocessing deviates from the export contract, but the pipeline is empirically calibrated | DOWNGRADED |
| **B30** | The map is ~37 % occupied voxels, higher than ray geometry predicts | OPEN |
| **B31** | The course-slew limiter is saturated 71 % of the time; the demand it chases is noise | DIAGNOSED |
| **B32** | The aircraft is not *late*, it is *sideways* — cross-track dominates, lag is symmetric | DIAGNOSED |
| **B33** | The aircraft is HELD against geometry: 37 % of flights, 3.6 % of all flight time | CONFIRMED |
| **B34** | Half the flight FALCON's own reference is parked — a plan/replan cadence mismatch | MEASURED |
| **B35** | Three FSM replan knobs written to `/fsm/`, read from `/exploration_manager/fsm/` | CONFIRMED |
| **B36** | `obstacles_inflation` is read by nothing; a clearance metric is named after it | CONFIRMED |
| **B37** | 41 % of plans are rejected by the pre-publish collision check; A* is fine | MEASURED |
| **B38** | Parked plan, collision rejections and wedging are ONE chain, rooted in ~12 cells | CONFIRMED |
| **B39** | The campaign is power-starved; a within-flight paired design would fix it | PROPOSAL |
| **B40** | The optimiser converges in <1 ms of its 10 ms budget; its answer still collides | CONFIRMED |
| **B41** | ESDF and occupancy AGREE at every rejection; it is an optimisation problem | CONFIRMED |
| **B42** | The optimiser converges with control points inside obstacles; it stops too early | CONFIRMED |
| **B43** | The optimiser moves 22 cm median and still ends inside; not a tolerance problem | CONFIRMED |
| **B44** | **ROOT CAUSE**: the ESDF is unsigned, so points inside obstacles feel zero gradient | CONFIRMED |
| **B45** | Seeded blocked regions arrive at strike 1, which is deliberately non-retiring | CONFIRMED |
| **B46** | Our `ttl_max_doubling=1` caps the blocked radius at 3.0 m vs a 5.5 m sampling radius | CONFIRMED |

---

## B1 — The nose is aimed by a heuristic, not by the planner  ·  OPEN  ·  operator's #1 ask

**Symptom.** `YAW_MODE="course"` points the nose along travel. FALCON does not work that way: its
viewpoints carry their own yaw and the paper's inter-viewpoint cost is `max{t_pos, t_yaw}`
(arXiv:2407.00577 §V-B) — yaw exists to aim the 80×60°, 5 m sensor frustum at the frontier it
wants to map. We discard that plan and re-derive heading from velocity.

**Evidence.** `config.py:302` and its own comment: measured in course mode, heading error vs the
planner's yaw is **p50 18–28°, p90 46–50°**. So the camera already spends much of the flight
pointed well away from where FALCON wanted it — we get the cost of a heuristic without its benefit.
`traj_server` publishes `yaw` *and* `yaw_dot` on `/planning/pos_cmd`; the follower ignores `yaw_dot`
entirely (`config.py:308`, `YAW_DOT_FF = 0.0`).

**Why the obvious fix is not obvious.** `yaw_mode="reference"` was tried once (v3/v4.0,
`runs/AUTOLOOP_JOURNAL.md`) and **deadlocked**: coverage 0.33× baseline on both flights, stopped at
n=2. The mechanism of that deadlock was never established. Do not simply re-flip the flag —
find out why it deadlocked first. Prime suspect is B2.

**Consequence if solved — measured, and NOT what I first assumed.** The mapping counters
(`20260902_150951Z`) decompose coverage exactly:

    8.0 % of box voxels  =  11.1 % of the footprint  x  72 % vertical fill

Within the columns it has visited the aircraft maps **3.46 m of the 4.8 m band**. It is *not* flying
a thin horizontal slice, so independent yaw cannot buy much vertical extent — there is little left to
buy where the aircraft has been. **The binding constraint is horizontal: it has reached 11 % of the
floor.**

So the yaw claim has to be stated differently from the obvious one: yaw would buy **more coverage per
metre travelled** — seeing sideways while transiting rather than only where the nose points — and the
primary metric for that experiment is **new voxels per metre flown**, not vertical fill and not
coverage alone. *Recorded because the first reading of these numbers was the wrong one, and it would
have aimed the experiment at a quantity that is already fine.* Cost: translation control gets harder, because the body-frame drive vector now depends on a
nose direction that is moving independently.

---

## B2 — The airframe is 3.7x slower sideways than forward, and nothing knows it  ·  **DIAGNOSED 2026-09-02**  ·  this is the v4.0 deadlock

**The asymmetry.** Two horizontal axes, two different ceilings:

| body axis | cap | speed at that cap (measured curve) |
|---|---|---|
| x, forward | `max_forward_axis = 900` counts | **1.566 m/s** |
| y, lateral | `LATERAL_AXIS_CAP = 600` counts | **0.428 m/s** |

`rooster_twist_control_adapter.py:343` and `config.py:523`; speeds read off the frozen
`ROOSTER_HORIZONTAL_POINTS`. The lateral cap is not arbitrary — 900 was flown and the operator
flagged the roll as too aggressive, so 600 bounds the bank. But the consequence was never followed
through.

**Why it deadlocked v4.0.** In `yaw_mode="course"` the nose points along travel, so essentially the
whole demand lands on the fast forward axis: 1.57 m/s available against a planner `max_vel` of
0.8 — comfortable headroom. Flip to `yaw_mode="reference"` and the nose points where FALCON wants
(at the frontier), which is at an arbitrary angle to travel. The travel component that lands on the
body-y axis is now capped at **0.428 m/s — well below the 0.8 m/s the planner assumes**. Demand at
90° to the nose is served at ~54 %; the aircraft falls progressively behind, `replan_from_pose`
drift keeps firing, and exploration crawls. **Coverage 0.33x is exactly the shape of an aircraft
that can only move at half speed in the directions the plan asks for.**

**And the clip distorts the direction, not just the speed.** Each axis gets its own
`AxisVelocityServo` with its own `output_limit` (`rooster_twist_control_adapter.py:487-493`), so a
saturating lateral axis is clipped **per axis**. That turns an over-demand diagonal into a
*different heading* — the aircraft goes more forward than it was asked to. The tracker's own
`_clamp_velocity` explicitly refuses to do this ("per axis clipping turns an over-speed diagonal
into a different heading, which is a steering error dressed up as a speed limit") and then the
adapter does it anyway, one layer down.

**Consequence for B1.** Re-enabling reference yaw *without* addressing this simply re-runs v4.0.
The fix is not "flip the flag". Candidate routes, in order of preference:
1. **Direction-preserving saturation** at the adapter: scale the (x, y) pair together so the
   commanded heading survives the clip. Cheap, safe, and correct regardless of yaw mode.
2. **Make the demand anisotropy-aware** — the tracker knows the nose angle, so it can cap the
   world-frame speed by what the body axes can actually deliver in that direction, instead of
   asking for something unachievable and being silently clipped.
3. **Raise lateral authority with a gentler slew.** The roll complaint came from fast lateral
   *changes* (the turn-crab feed-forward), not from steady lateral. Higher cap + slower attack may
   buy the authority without the bank. Needs its own A/B against the roll-rate guard.
4. **Bound the yaw plan** — keep the nose within some angle of travel: partial yaw freedom, a
   compromise on B1's benefit.

Route 1 is a strict improvement and independent of the yaw decision; it should land first.

**Verified as correct, so not a bug:** the world→body rotation itself
(`falcon_exploration_follower_node.py:647-649`) uses `self._yaw`, the **measured** yaw from the
odometry quaternion. That is the right angle. The rotation was the original suspect here and it is
exonerated.

## B3 — The velocity-damping gain is justified by an inner loop that does not exist  ·  DIAGNOSED

**Evidence.** `core/planning/trackers/reference_tracker_3d/params.py` sets
`velocity_damping_xy = 0.25` and documents the reason: *"Lower than the source controller's 0.4
because of what sits underneath: PX4's own velocity controller is already a damped loop, so this one
is cascaded on top of it."*

**Why that is wrong here.** Memory `project_rooster_native_offboard_velocity` records the measured
finding that **Sphera's physics runs off the vendor ManualControl pipeline, not PX4's outputs** —
PX4 offboard does not actuate this aircraft. There is no inner damped velocity loop underneath. What
is underneath is an expo stick curve into a plant with τ≈1.15 s to accelerate and τ≈0.27 s to brake.
So the single term the module calls *"the term that does the tracking"* is detuned by ~40 % against
a loop that is not there.

**Expected effect.** Directly attacks the operator's "we are late" complaint. Must be flown as a
pre-registered A/B on a low-variance metric (along-track lag), not on coverage.

---

## B4 — Nothing in the loop models the plant's asymmetric response  ·  OPEN

The measured plant accelerates with τ≈1.15 s and brakes with τ≈0.27 s (memory
`project_rooster_manualcontrol_axis_calibration`). The tracker's only anticipation is a fixed
`accel_lead_s = 0.25` applied symmetrically to `a_ref`. A single symmetric lead cannot compensate a
4× asymmetry: it is simultaneously too little when speeding up and too much when slowing down.

**Caution.** Raising the lead uniformly was tried (v5.0, `ACCEL_LEAD_S` 0.25→0.60) and **failed** —
along-track p90 got worse. Finding F explains why: most along-track error is accumulated while the
aircraft is *stationary*, and no transient anticipation fixes that. So the correct form of this fix
is an **asymmetric / inverse-plant** feed-forward, not more of the same symmetric lead, and it must
be measured on moving-only samples.

---

## B5 — A wall-pinned aircraft: detection WORKS, the gap is the recovery  ·  **partly REFUTED 2026-09-02**

**What is real, measured on two flights.** Joining the follower's own commanded body speed to the
aircraft's actual displacement: episodes of **commanded > 0.15 m/s while achieving ≤ 0.06 m/s, with
no reflex engaged and lasting ≥ 1.5 s** (beyond any plausible acceleration lag at tau≈1.15 s) come to
**39 s = 8 % of a healthy flight** and **109 s = 23 % of a planner-locked one**. So the operator's
concern is real and now quantified.

**What I got wrong, and how it was caught before it flew.** I hypothesised that the detector could
not *arm*: the escape needs 3.0 **continuous** seconds of "asking but not moving", the timer
hard-resets on a single sample, and the velocity it reads is `/odom_world`'s raw 25 Hz finite
difference — of which **5.8 %** of samples read above the 0.06 m/s threshold while the aircraft had
moved under 6 cm in the whole previous second. A spurious reset every ~0.7 s against a 3 s timer
looked decisive.

**It is not.** Replaying both detectors over the recorded flights — the existing velocity test and a
displacement-over-a-window test immune to single-sample noise — gives **8 vs 7** arming events on the
healthy flight and **42 vs 40** on the locked one. **The detector is not the problem.** The
displacement test was written and is wired (`~stall_window_s`, default `0.0` = existing behaviour)
but is **NOT to be flown** on this reasoning; it would cost a flight to measure a difference that is
already known to be about two events per run.

**Where the time actually goes**, and the open question: on the locked flight the reflex is engaged
(escaping or pinned-hold) on **13.6 %** of ticks against **23 %** of the flight stuck. The gap is
mostly the **4 s escape cooldown** between attempts — and MISSION.md P30/P31 tested that cooldown in
*both* directions and closed it. So the remaining lever is not detection and not the cooldown; it is
what the escape *does* (back off 0.30 m/s + yaw 35 °/s) and whether it is the right manoeuvre against
geometry the map never had. **Open, with no candidate fix yet.**

## B6 — FALCON's terminal plan-fail lock  ·  DIAGNOSED (Finding I)  ·  largest single time sink

`runs/AUTOLOOP_JOURNAL.md` Finding I, measured over 41 flights: **91.9 %** of no-moving-reference
time is A*-plan-fail flooding, **86 %** of that is one terminal lock. Worst case **254 s** on a
single cell, **17 925** plan fails, the identical `Next pos` line **17 551** times, 89.6 %
stationary. Corpus-wide **98 episodes = 3587 s = 21.7 % of all flight time**.
`sweepBlockedFrontiers` retired **zero** clusters in the locked runs.

The config-only route is closed: widening the blacklist shadow (v8.0, `blocked_region_radius`
1.5→2.75) worked mechanically and still lost — 4 of 5 flights emptied the frontier set at least
once against 1 of 4 controls. It trades unreachable targets for impermissible ones. **The remaining
route is a C++ patch** — a back-off on the ~70 Hz retry, or a blacklist that actually retires a
viewpoint the tour keeps re-offering. We have the source and permission.

---

## B7 — Altitude runaway  ·  **THREAT MODEL CORRECTED 2026-09-02**  ·  quality problem, not a crash risk

**What was inherited.** `runs/AUTOLOOP_JOURNAL.md` ranks this the top unsolved item and reserves it
for the operator, on the grounds that *"past that ceiling `exploration_node` indexes out of bounds
and segfaults"* and *"the failure mode is a crashed planner mid-mission, and it is a coin-flip
away."* The margin was stated as 0.2 m: worst excursion 3.60 m against a 3.8 m `flight_band`.

**That premise is no longer true, and the check is cheap.** `maps/sphera_jail.yaml` names
`UniformGrid::positionToGridCellCenterId` as the crash site. Reading the **built** source:
`positionToGridCellCenterId` (`hierarchical_grid.cpp:242`) calls `positionToGridCellId` at `:246`
and `:261` — and `positionToGridCellId` is exactly the function **our own
`falcon_hgrid_clamp.patch` clamps**, with the comment *"the aircraft sits below the box floor on
every takeoff (and can overshoot any face)"*. Three clamp blocks are present in the compiled source
and the library is in the shipped image.

**Corroborated in flight, not just in code:** `20260902_154018Z` reached **z = 3.82 m — past the
3.8 m band — and `planner_death: died False`.** A control flight reached 3.60 m, also with no death.

So the crash path is closed. **This is a coverage-quality problem, not a mission-ending one.**

**It is still real and still worth fixing.** Measured over eight in-sample flights today, max world z:

| arm | flights | max z per flight |
|---|---|---|
| candidate | 4 | 1.55, 2.87, 3.40, **3.82** |
| control | 4 | 2.09, **3.60**, 1.89, 1.86 |

**Both arms** produce them, so this is not caused by anything in the current experiment. An aircraft
at 3.4–3.8 m is mapping ceiling rather than the room, and the loop's own numbers show the cost: those
flights sit at the bottom of the coverage distribution.

**Mechanism, still open.** The hold is **purely terrain-relative and has no absolute ceiling** —
`max_ranger_m = 1.0` caps the *target*, not the achieved world z. Flying over raised geometry drops
the rangefinder, the hold reads "too low" and climbs, and nothing bounds where that ends in world
coordinates. Now that the pose stream is trustworthy (F11), an absolute z ceiling is implementable
for the first time — but it is a change to altitude hold and needs its own pre-registration.

## B8 — Cold-start deadlock in bring-up  ·  CLOSED 2026-09-02 (see LOOP_FIXES.md F1)

Every battery/armable probe is a `docker exec` into the `it` container, but `ensure_sphera()` ran
*before* `start_containers()`. With `it` stopped — which is the state after any long idle — the
battery read returned None forever, `_fresh_drone_ready()` could never be true, and bring-up burned
all four Sphera re-entry attempts (each killing a healthy `R1`) before failing the cycle. Hidden for
the whole prior campaign because the stack was always warm. Cost: the first cycle of this loop.

---

## B9 — A failed cycle can inherit the previous flight's telemetry  ·  OPEN (Finding J)

The campaign copies recorder output from a fixed container path. A cycle that fails before the
recorder produces new data copies the **stale** file while still writing `ended: completed`. Caught
by two folders 91 s apart with identical `truth.jsonl` MD5s. Until fixed: **dedupe by MD5 before
computing any statistic**. Proper fix is to stamp the recorder output with the run id, or refuse a
file whose mtime predates cycle start.

---

## B10 — Mapping rate is not instrumented  ·  **DELIVERED 2026-09-03**  ·  operator deliverable

Every item the operator asked for now exists and is recorded raw for the visualization, which is
still deliberately not built (the brief says to save, not to draw).

| asked for | delivered | where |
|---|---|---|
| 2D mapped area and time per m² | **0.70 s per m²** | `/voxel_mapping/map_stats` (F: `sparx_map_stats.sh`) |
| time per 1×1×1 m cell | **0.137 s per cell** | same |
| new voxels per second over time | **614 voxels/s, decaying 4.2× across a flight** | same, 2 Hz |
| free / occupied / unknown counts | published every 0.5 s | same |
| frontier count over time | `/planning_vis/frontier_pcl` | `probe_flight_trace.py` |
| **time budget for every flight second** | **47 % flying the plan, 46 % plan parked, 4.2 % tilt cut, 2.5 % escaping** | B34 (cont.) |
| all of it saved raw | `coverage.jsonl`, `flight_trace.jsonl`, `runs/_analysis/contact_holds.jsonl` | per run |

The headline numbers for the operator: **11.1 % of the box footprint, 11.4 % of its 1 m cells and
8.0 % of its voxels** get mapped in a flight, which is a **72 % vertical fill within the columns the
aircraft actually visits** — the constraint is horizontal reach, not vertical coverage. And the
budget line above is the one that matters most: **under half of a flight is spent flying the plan.**

---

# Findings from the deep FALCON C++ read (2026-09-02)

Source: the patched FALCON source extracted from `falcon-ros:noetic`. Every item below carries a
file:line and, where marked CONFIRMED, was verified against live rosparams or this session's own
flight logs rather than taken from the read.

## B11 — The velocity the servo closes on is 0.20 s late  ·  **CONFIRMED**  ·  FIXING (F3)

`SpheraPawnState.velocity` is all-zero in this vendor build, so `rooster_ground_truth_localization.py`
falls back to a differentiated position on **every** flight — verified in 11 of 11 recent runs, and
the fallback's own log line says *"expect the ~0.25 s of filter lag the servo gains were cut for"*.
Measured by cross-correlating the published estimate against the true derivative over a whole
flight: the peak is at **0.20 s**. `AxisVelocityServo` (`ki=220`) integrates against that.

The filter was not gratuitous: Sphera stamps state at ~129 Hz with **2.3× dt jitter**, which puts
p90 **1.0 m/s** of noise on a 0.5 m/s signal when consecutive samples are differenced. The fix is to
difference over a fixed window instead of lowering the filter. See `LOOP_FIXES.md` F3.

## B12 — The 1.667× mid-curve compression is a deliberate trade, and it is a GOOD one  ·  **CLOSED 2026-09-03, my framing was wrong**

`exploration_fsm.cpp:658-663`. `/fsm/slow_traj_target_yaw` and `/fsm/slow_traj_ratio_max` are
**never set**, so the C++ defaults (1.57, 4.0) apply, and the computed ratio is therefore always far
below `ratio_min = 0.60`. Verified live: `rosparam get` shows `target_yaw` and `ratio_max` UNSET, and
the baseline flight logged **11** rescales, *every one* of them
`Rescale ratio 0.196|0.258|0.299 clamped to 0.600`.

`lengthenTime(0.60)` leaves the boundary spans alone and compresses only the middle knots, so peak
mid-curve reference speed rises by up to **1.667×** — on trajectories that were slow *deliberately*,
because the yaw turn needed the time. `reallocateTime`/`checkFeasibility` are dead code, so nothing
re-checks the result. The 11 fires were long plans (durations 8.1 / 5.8 / 11.3 s), so they account
for roughly **12 % of flight time**.

**FLOWN AND REVERTED (F17).** Disabling it cost **25 % of coverage** and bought **no tracking
improvement** (along-track 1.02). Six metrics moved together: plans longer in time and shorter in
distance, 12 % less distance flown, 47 % more time stationary. The compression exists to stop the
aircraft dawdling on slow curves, and on this platform — where time spent parked is what limits
coverage — that trade is favourable. **I described a working design decision as a bug.** The
description above of *what it does* is accurate and is kept; the conclusion that it should be removed
is withdrawn.

## B13 — Yaw and position run on different clocks after a rescale  ·  DIAGNOSED

`Bspline.msg` carries `float64[] knots` for position but only a scalar `float64 yaw_dt` for yaw.
`exploration_fsm.cpp:723` publishes `yaw_traj_.getKnotSpan()`, and `getKnotSpan()` returns the member
`knot_span_`, which `lengthenTime` (`non_uniform_bspline.cpp:170-185`) **never updates** — it mutates
only `u_`. `traj_server.cpp:307` then rebuilds yaw as a *uniform* spline while position gets its
exact re-timed knots. **After any rescale the yaw reference runs on the pre-rescale clock.**

A scalar could not represent the re-timed yaw knots even if `knot_span_` were fixed, because
`lengthenTime` makes the vector non-uniform. Killing the rescale (B12) removes this entirely; putting
`yaw_knots` on the wire is the correct fix only if the rescale is ever re-enabled.

**This is a second, independent mechanism for the v4.0 reference-yaw deadlock** (see B1/B2): in
course mode a desynced yaw plan is simply ignored, so the bug was invisible.

## B14 — Yaw rate is unconstrained everywhere in FALCON  ·  DIAGNOSED

`bspline_optimizer.cpp:14-17` — the yaw phase is `SMOOTHNESS | START | END | WAYPOINTS`. **No
FEASIBILITY, no MINTIME**, and `:186`'s `if (dim_ != 1)` means the 1-D yaw problem gets no NLopt
bounds at all. `max_yaw_velocity` is read once into `PathCostEvaluator::yd_`, a *tour-cost heuristic*.
Worse, `exploration_manager.cpp:1489`'s `flag_fast_yaw` deliberately distributes an over-large turn
linearly across the duration — i.e. it plans a yaw rate **above** `yd_` on purpose.

So the yaw reference we would follow in `yaw_mode:=reference` can demand rates the platform cannot
deliver (our follower caps at 45 °/s). **Third mechanism for the v4.0 deadlock.** The enabler for
re-enabling yaw is a hard `dt` floor from `time_lb` (`bspline_optimizer.cpp:194-195`).

## B15 — `traj_server` freezes the endpoint but keeps reporting READY  ·  DIAGNOSED

`traj_server.cpp:351-358` — past `traj_duration_` it freezes position and yaw, zeroes velocity and
`yaw_dot`, and keeps publishing at 100 Hz with a **fresh** `header.stamp` and `trajectory_flag`
latched at `TRAJECTORY_STATUS_READY`. Our follower's two liveness gates and
`ReferenceTracker3D.reference_timeout_s = 1.0` are therefore **permanently open and can never fire**.
`replanCallback` also truncates `traj_duration_` on every replan, so each cycle ends in a frozen hold.

This is the mechanism behind Finding C/F's "the aircraft stops because nothing commands it": the
reference is not stale, it is *stationary*, and every layer designed to notice reads it as healthy.

## B16 — The viewpoint gain model uses the wrong camera  ·  DIAGNOSED

`perception_utils.cpp:6-11,15` reads `/uav_model/sensing_parameters/fov/{horizontal,vertical}` =
**90° × 73.7°** and `max_dist` = **5.0 m**, neither ever overridden — while the real camera is
**135° × 90°** and the TSDF fuses to `raycast_max` **8.0 m**. Viewpoint yaw is chosen by
`countVisibleCells` under that wrong frustum, so every viewpoint's information gain is
underestimated **asymmetrically in yaw** — which is precisely the quantity the mapping-speed goal
depends on. Also note `perception_utils.yaml`'s `top_angle`/`left_angle`/`right_angle` keys are dead,
and its `fov_type_str` does not match the code's `/perception_utils/fov_type`.

## B17 — Time allocation is uninitialised for every segment after the first  ·  DIAGNOSED

`planner_manager.cpp:89` declares `Eigen::Vector3d time_xyz;` **uninitialised**. It is assigned only
under `if (i == 0)` and accumulated only under `if (i == pt_num-2)`, yet `times(i) =
time_xyz.maxCoeff()` reads it for every `i`. Built `-O3 -w` with no
`EIGEN_INITIALIZE_MATRICES_BY_ZERO`, so middle segments inherit segment 0's times from a stale stack
slot. A correctness fix with a **large behavioural blast radius** — today's accidental behaviour has
been tuned around, so it must be A/B'd deliberately and late.

## B18 — Two of A*'s three failure exits are silent  ·  DIAGNOSED  ·  blocks B6

`astar.cpp` — the timeout warning at `:314` and the open-set-empty diagnostics at `:400-402` are both
commented out; only node-pool exhaustion logs. **A 1 ms timeout and a genuinely enclosed aircraft are
indistinguishable in the log, and they call for opposite fixes.** The default profile is 0.5 m /
**1 ms**. Restoring three `ROS_WARN_THROTTLE` lines is zero behavioural risk and decides the shape of
any fix for B6.

## B19 — `frac_of_box` is inflated  ·  CONFIRMED-ish  ·  reporting only

`config.py:235 EXPLORABLE_VOLUME_M3` does not match the `sphera_jail.yaml` box
(x[-14.6, 91.0], y[-42.4, 20.4], z[-1, 3.8] ≈ 31 800 m³). Every `frac_of_box` printed to date is
suspect. Verify and correct — it is a reported number, not a control input.

## B20 — Today's flights map 3–4× more than yesterday's, and nothing in this loop explains it  ·  OPEN

Measured over the last 11 runs of 2026-09-01 against the first 3 of 2026-09-02, all carrying
`controller_rev: v2.1d` and all before any flight-behaviour change of this loop had landed:

| | 2026-09-01 (n=11) | 2026-09-02 (n=3) |
|---|---|---|
| coverage gained | 743 – 1046 m³ | 1591 – **3894** m³ |
| distance | 127 – 199 m | 204 – 233 m |
| stationary fraction | 0.13 – 0.56 | 0.12 – 0.20 |

A 3–4× step is far outside the platform's own 27 % coverage CV, so it is not noise. Candidate
explanations, none tested:
1. **A contaminated FALCON voxel map.** `CLAUDE.md` records that the map is long-lived with no
   decay and that any `R1` recreation requires a full `falcon` container recreation; yesterday's
   soak ran for hours across many Sphera restarts, today's runs began from a cold, freshly recreated
   stack.
2. The previous session's interleaved driver alternated `SPARX_BLOCKED_RADIUS` 2.75/1.5, and v8.0's
   documented failure was map sterilisation. If shadows persist across an `exploration_node`
   respawn — `frontier_finder.cpp:302-309` writes them to `blocked_regions_runtime` and reloads
   them — a poisoned run could contaminate later ones on the same roscore.
3. Something environmental that was not recorded.

**Why it matters even though the direction is favourable:** it means run-to-run comparisons that
span a stack restart are unsafe, and it may mean a routine "recreate the falcon container" is worth
far more than any control change in this file. **Never compare arms across a cold-start boundary**
— which the interleaved driver already avoids, and which is now a reason for it rather than a
convention.


## B21 — `config.py` is imported in three environments and one of them has different paths  ·  CLOSED 2026-09-02 (F7)

The recorder runs inside `it`, the twist adapter inside `robotican_dev`, the campaign on the host —
and all three import `tools/falcon_campaign/config.py`. A host-absolute path evaluated at import
time therefore takes down whichever consumer does not share it, and does so *silently* from the
campaign's point of view: the cycle completes and reports `ended: completed`.

Closed by resolving paths relative to `__file__` and by a pre-flight import check inside the
container. **Kept in this file rather than deleted** because the constraint is permanent and the
next edit to `config.py` has to respect it.

## B22 — Sphera intermittently publishes a SECOND, bogus pawn on `/R1/sphera/state`  ·  **CONFIRMED 2026-09-02**  ·  flight safe, recordings were not

**What it is.** `/R1/sphera/state` sometimes carries two publishers. The second reports a different,
**stationary** pawn at a fixed `(-0.59, -4.53, 0.12)` with zero attitude, interleaved with the real
aircraft's messages. Measured on `20260902_133813Z`: **83 % of the 8699 recorded samples were the
bogus pawn**, leaving 17 % real. `ros2 topic info --verbose` on the live stack shows two
`_CREATED_BY_BARE_DDS_APP_` publishers alongside `rooster_backend`.

**It is intermittent, and that is the dangerous part.** `20260902_130620Z` logged **0** rejections;
the very next flight logged **794 warning lines at one per 200**, i.e. roughly **159 000** rejected
poses. Two consecutive cycles, same stack, opposite behaviour.

**Flight control is NOT affected.** `rooster_ground_truth_localization._keep()` already rejects them
(the defect is named in its own docstring), so `/R1/localization`, `/R1/attitude_rpy` and
`/R1/velocity_truth` are clean — the recorded localization stream jumps at most 0.1 m between samples
while raw truth jumps up to **90 m**.

**Recordings and analysis WERE affected, and are now fixed.**
- `recorder.py` subscribes the raw topic by design ("truth is recorded raw"), so `truth.*` is up to
  83 % garbage. It now stamps each row with `truth.suspect` — computed from the exact identity
  `truth.x == -localization.x` that holds for the real aircraft — **flagged, not filtered**, so the
  recording keeps everything and the analyzer can choose.
- `analyze._normalize` preferred `truth.roll`/`truth.pitch` over the gated `attitude.*`. Since the
  bogus pawn reports **zero attitude**, that pulled every tilt metric toward zero: measured on the
  affected flight, `truth.roll` p50 **0.00°** against `attitude.roll` p50 **1.84°**. The order is now
  reversed. Position was never affected — `localization` was already preferred.

**Checked and clear: the F3 derivation is unharmed.** The velocity-noise measurement behind F3 was
taken from a 2958-sample capture that was **100 % real R1** (verified by re-partitioning it), so the
"2.3× dt jitter, p90 1.0 m/s noise" figures stand.

**Open consequence, not yet acted on.** While the defect is active GTL passes only ~17 % of messages,
so the velocity estimator's effective sample rate falls from ~126 Hz to ~22 Hz. The 60 ms window
still spans a valid interval, but F3's noise/lag numbers were derived at the full rate and do not
describe a contaminated flight. **Any flight whose `rooster_gtl.log` shows rejections should be
treated as a different sampling regime** — worth recording per run.

## B22b — Where the duplicate pawn actually lives (for the operator)  ·  OPEN, external

Traced as far as is useful without guessing at GUI clicks:

- It is **not** in our stack. Only one `R1` backend runs
  (`docker run --name R1 ... physical_rooster.launch.py instance:=1`), one localization node and one
  command unit — verified by process listing inside `it`.
- The extra publishers register as `_CREATED_BY_BARE_DDS_APP_`, i.e. Sphera's own C++ side, alongside
  the legitimate `rooster_backend`.
- **`sphera-restart.sh` does not clear it.** It recreates the `drone_simulator` container, and the
  duplicate has survived roughly **ten** such restarts. So the state lives in the Sphera application /
  scenario rather than the container — most likely a second entity assigned to the `R1` namespace.
  Sphera publishes a `<name>/sphera/state` per entity (`/Rock_1/...`, `/Rug_1/...`,
  `/Wooden_Crate_1/...` are all live), which is consistent with a scene object having been given the
  drone's name.
- The GUI automation clicks the `Rooster_1` assignment row on every re-entry, and `bringup` retries
  that sequence up to four times. **A plausible origin** is a retry landing on an already-running
  scenario and adding a second assignment — which would fit it appearing mid-session rather than at
  start-up. Not proven, and not worth proving by clicking at a live simulator.

**The publisher COUNT is not the same as harm — measured 2026-09-02 18:14.** With
`/R1/state` and `/R1/sphera/state` both still reporting `Publisher count: 2`, a flight ran with
**zero** pose rejections and **zero** ranger drops. The second publisher is persistent but only
*intermittently divergent*: sometimes it reports a different pawn, sometimes the same data.

That explains why some flights with the duplicate present were completely normal, and it validates
the choice of indicator: `dual_publisher_active` is derived from the guards' **rejection counts**,
not from the publisher count, so it measures harm rather than exposure. Use the reject counts to
judge a flight; use the publisher count only as a hint that the defect is available to occur.

**Actionable for the operator:** fully quit and relaunch the Sphera application (not just the
container) and check `ros2 topic info /R1/state --verbose` reports `Publisher count: 1` before the
next long unattended run. The loop now survives the defect (F10, F11) but every flight taken with it
active is a degraded sample.

## B23 — The dual publisher makes the rangefinder alternate, and that is very likely Finding K  ·  **CONFIRMED 2026-09-02**  ·  TOP PRIORITY

B22's second publisher does not only corrupt recordings. It publishes on **`/R1/state`** as well, so
the **downward rangefinder alternates sample by sample between the real 0.131 m and the impostor's
3.415 m** — confirmed live with `ros2 topic echo`, and `ros2 topic info` reports
`Publisher count: 2` on both `/R1/state` and `/R1/sphera/state`.

`rooster_unit`'s altitude hold is terrain-relative and reads exactly that value. The command-unit log
shows it thrashing several times a second:

```
ranger=0.131m target=0.900m error=+0.769m wanted_z=1080     <- full climb
ranger=3.415m target=0.900m error=-2.515m wanted_z=320  [AT CEILING]   <- full descent
```

**Its plausibility guard cannot stop this.** The guard rejects a step implying more than 3 m/s — and
then **re-seeds `_hold_prev_ranger` on the rejected value** (`rooster_unit.py`, "re-seeding on it").
Against a strictly alternating pair that makes the *next* sample measure from the impostor, so the
two values pass alternately whenever the interval stretches past ~1.1 s. Contrast
`rooster_ground_truth_localization._keep()`, which solves the same problem correctly by requiring
`relatch_after_rejects` **consecutive** rejects before adopting a new level — and which is why the
localization stream is clean while the ranger is not.

**Measured impact on the loop:** across eight consecutive cycles, the three that failed
`hover never settled` all had the defect active, and the first cycle of the day — with **zero**
rejections logged — is also the best flight recorded (3894 m³). It is not an absolute blocker: one
cycle took off normally with it active, so it is probabilistic, depending on which value the hold
loop happens to sample.

### Why this is probably Finding K, the campaign's top unsolved item

`runs/AUTOLOOP_JOURNAL.md` describes two sub-failures behind one altitude-runaway symptom, and this
mechanism produces **both**:

- `113204Z`: *"`wanted_z` 320 and `z` 320 — 100 % of above-2.5 m ticks in the descend zone. The loop
  commanded full descent and the aircraft did not come down."* If the ranger is reading the
  impostor's 3.4 m while the aircraft is actually at ~1.2 m, full descent is commanded and the
  aircraft correctly does not descend. **A measurement failure wearing the costume of a plant failure.**
- `124555Z`: *"`wanted_z` 320 but `z` sent 891; a ~570-count gap the 15-counts-per-tick slew should
  close in ~4 s did not close over 28 s."* The slew cannot converge on a target that flips between
  320 and 1080 every other sample.

It also explains why the ranger-rate filter adopted in that session *"removes one trigger but does
not solve the runaway"* — it addresses the symptom, and the alternation defeats it by construction.

### TESTED 2026-09-02 — and it explains ONE of Finding K's two sub-failures, not both

The test was pre-specified above (bimodal ranger + GTL rejections in the two dissected runs) and then
run. The two runs split cleanly, which is itself consistent with the journal's own conclusion that
there are *two distinct sub-failures behind one symptom*:

| run | GTL rejections | most common ranger value | verdict |
|---|---|---|---|
| `20260901_124555Z` | **418** | **0.15 m held by 50 % of 8917 samples** | **CONFIRMED — dual publisher** |
| `20260901_113204Z` | **0** | top value holds 5 %, smooth | **NOT this mechanism** |

**`124555Z` is solved.** Half its rangefinder samples reported **0.15 m** while the aircraft was at
~1.05 m. A terrain-relative hold seeing 0.15 m against a 0.90 m target reads "far too low" and
commands climb — so the aircraft climbs, and the journal's puzzle of *"`wanted_z` 320 but `z` sent
891, a gap the slew should close in 4 s and did not close in 28"* is simply a target flipping between
climb and descend faster than the 15-counts-per-tick slew can follow. **The altitude runaway in that
flight was a measurement failure, not a plant or authority failure.**

**`113204Z` is not.** No duplicate publisher, no bimodality, ranger reaching 3.98 m smoothly — a
genuine excursion that still has no mechanism. It stays open, and it is the one that matters for
safety, because it is the case where the aircraft really was high.

For today's flights the impostor's ranger value differs per occurrence (3.415 m in `134727Z`, 0.15 m
in `124555Z`), so **the signature to test for is bimodality, not any particular number.**

### What has been done, and what has not

- `ensure_sphera()` now restarts Sphera once when it sees a duplicate, and `health_report` records
  `state_publishers` per cycle. **Advisory, not a veto** — refusing to fly would ground the campaign
  over something flights can survive.
- **NOT done:** making the ranger guard latch like `_keep()` does. That is the real fix and it is a
  change to altitude hold, which the previous session called the hardest-won behaviour in this stack
  and deliberately left alone. It needs its own pre-registered A/B, and it should be attempted only
  after the bimodality test above confirms the Finding K link.


## B24 — Localization can latch onto the impostor pawn and fly blind at full stick  ·  **CONFIRMED**  ·  guarded (F10), root cause OPEN

`_keep()` latches onto the first pose it sees. With B22's duplicate publisher active, roughly half the
time that is the impostor — and then every real pose is rejected as a jump, silently, for the whole
flight (`re-latching pose` warnings: **0**).

Measured on `20260902_135459Z`: pose frozen at the impostor's `(0.589, 4.526, 0.118)` for 453 s while
the ranger read 1.29 m; follower at **900 counts** (the 1.566 m/s platform ceiling) throughout;
recorded distance **1.6 cm**; coverage **28 m³**; 52 % of ticks inside the stuck reflex, which
correctly detected the condition and could do nothing about it.

**Guarded, not fixed.** F10 aborts such a cycle at hover. The underlying defect — a latch with no way
to tell two pawns apart — is untouched. The principled fix is for `_keep()` to prefer the pawn whose
pose is consistent with the rangefinder once airborne, i.e. to re-latch *away* from a perfectly static
pose while the aircraft is known to be flying. That is a change to the localization node every
consumer depends on, so it needs its own pre-registration.

**Yield risk to watch:** if the duplicate persists, the coin is re-flipped every cycle (bring-up
restarts the localization node each time), so up to half of cycles could abort at hover. If that
materialises, fixing `_keep()` becomes urgent rather than optional.

## B25 — The position correction is 42–44 % of the command, violating the tracker's own design rule  ·  **CONFIRMED**  ·  OPEN

`ReferenceTracker3D`'s module docstring states the invariant plainly: the position-feedback term
*"is capped well below the flight speed on purpose: a correction that can out-run the trajectory is a
correction that flies the aircraft, and then the planner's dynamic-feasibility guarantees stop
describing what the airframe does."*

Measured over ticks where the reference was moving, on three flights spanning **both** experiment
arms:

| flight | feed-forward | damping | correction | correction share | saturated |
|---|---|---|---|---|---|
| `133813Z` (cand.) | 0.733 m/s | 0.126 | **0.628** | **42 %** | 13 % |
| `144746Z` (cand.) | 0.700 | 0.146 | **0.639** | **43 %** | 15 % |
| `142446Z` (ctrl) | 0.717 | 0.156 | **0.696** | **44 %** | 17 % |

**The correction is the same size as the feed-forward**, and it rails at its 1.0 m/s ceiling on
13–17 % of moving ticks. The invariant is not merely bent — the correction *is* flying the aircraft
for a large fraction of the flight, which is exactly the state the docstring says invalidates the
planner's feasibility guarantees.

**Very low variance** (42/43/44 %) across arms and flights, which makes it a good primary metric for
anything aimed at it — unlike coverage.

**What it does NOT yet tell us.** The magnitude is just `kp x error`, so a large correction is a
*symptom* of a large position error, not independently a cause. Two readings remain open and they
have opposite fixes:

1. **The plan asks for more than the plant sustains.** Feed-forward averages 0.70 m/s while the
   through-origin gain measures 0.53–0.70 — so the aircraft lags, the error grows, the correction
   grows to compensate. The fix would be *upstream*: lower `max_vel` so the plan is flyable.
   Note P34 raised `max_vel` to 1.0 and it failed; **nobody has tried lowering it.**
2. **The lag is the feedback, not the plan.** The servo closes on an estimate measured 0.20 s late,
   which is precisely what F3 is testing. If F3 works, the error and the correction should both fall
   without touching the plan.

**Do not act on this until F3 reports** — the two explanations are confounded, and F3 is already
flying the cheaper one. If F3 lands and the correction share stays at ~43 %, reading 1 becomes the
candidate and `max_vel` 0.8 → ~0.6 is the one-line test.

## B26 — The collisions are against geometry the map never had, and the fix is not in the controller  ·  **DIAGNOSED**  ·  next major direction

Measured today over 15 flights (`clearance.jsonl`, distance to the nearest **mapped** obstacle):

| | p50 clearance | inside the 0.40 m inflation |
|---|---|---|
| aircraft | 0.57 m | **40 %** |
| its own reference | 0.69 m | **30 %** |

**Three quarters of the aircraft's proximity to obstacles is inherited from the plan**, not produced
by tracking error. Only the remaining ~10 points are ours to fix by flying better.

**And pushing the plan away has already been tried and is only half a success.** MISSION.md P23 raised
the optimiser's `safe_distance` 0.40 → 0.55 and it worked *as a clearance change*: reference inside
0.40 m fell 73 % → 40 %, aircraft 66 % → 50 %, tracking error p50 0.52 → 0.32. **PINNED events did
not fall** (13, against 7–12 before). Today's numbers sit at that post-fix level, so the lever is
already pulled and pulling it further re-runs a refuted experiment.

**What that leaves.** If proximity to *mapped* geometry does not predict the contacts, the aircraft
is being stopped by geometry the map never had — which is exactly what the operator flagged
independently: DA3 has a near floor (`cam_min_depth` 0.45 m, measured behaviour nearer 0.62 m), so an
obstacle inside that radius is **invisible to the mapper by construction**. No amount of planner
margin or tracking accuracy can avoid what is not in the map.

**So the collision half of the mission is a PERCEPTION and MEMORY problem, not a control one.** That
is a redirect, and it is worth stating plainly because the obvious controller-side work would have
produced clearance numbers that improved while collisions did not — which is precisely what P23
already observed.

**The candidate mechanism, and it is already half-built.** `core/planning/environment/blockage_memory.py`
implements exactly this — remember where the aircraft was blocked, treat it as an obstacle — with its
own tests. **Nothing imports it** outside those tests. The stuck detector (B5) already fires
correctly and identifies these events; today it only triggers an escape and then forgets. Feeding
those events into a persistent obstacle memory is the missing half.

**Open design questions, none of them cheap:**
- FALCON owns the voxel map in C++. Injecting a remembered obstacle means either a synthetic occupied
  point in the depth cloud the mapper consumes, or a C++ addition beside `addBlockedRegion` (which
  today blocks *viewpoints*, not costmap cells).
- A false blockage is expensive — it sterilises space permanently, which is the documented failure
  mode of the v8.0 shadow experiment. It needs a TTL and a confirmation rule.
- The right primary metric is PINNED events per flight, which is low-variance and already recorded.

## B27 — A* compute budgets: what the diagnostics actually show  ·  **coarse half REFUTED, default half PENDING one measurement**

The `sparx-astar` diagnostics turned B6 from a guess into arithmetic. They also refuted my first
reading of them, which is recorded here because the wrong version briefly reached a launch file.

**The timeouts split by PROFILE, and the halves behave nothing alike** (two flights, 2026-09-03):

| overload | budget | timeouts | mean iters |
|---|---|---|---|
| `searchUnknown` | 0.0001 (**coarse**) | 498 | **22** |
| `searchUnknownOnlyBBox` | 0.0010 (**default**) | 457 | **238** |
| `searchBBox` | 0.0010 (default) | 172 | 269 |
| `search` | 0.0010 (default) | 90 | 189 |
| `searchUnknownBBox` | 0.0010 (default) | 30 | 279 |

**My error, and it is instructive.** I reported "iters 13–15, so the budget is the binding
constraint" and wrote it into `nav_stack.launch`. That figure was the **minimum of the coarse bucket
alone**, generalised to all six overloads — the default-profile searches reach 190–280 iterations.
And the inference was a non sequitur: few iterations proves the clock expired, *not* that more time
would have terminated the search. The comment is corrected in place.

**The `coarse` half is refuted for free, no flight needed.** `searchUnknownOnlyBBox` **already runs at
0.0010** — exactly the 10× value a coarse raise would apply — and still times out **457 times at 238
iterations on this same map**. A 10× budget does not resolve these searches. Raising `coarse` is also
the expensive half: it is the one profile feeding the O(n²) HGrid cost matrix, measured at +37 ms of
cycle time per +0.1 ms of budget.

**Two further corrections to the motivation**, both of which weaken the case:
- The `MEDIUM` retry already grants **100 ms** and is *never* within an order of magnitude of being
  used — `budget 0.1000` appears **zero** times in the corpus, and every medium failure exits
  `OPEN_SET_EMPTY`. So the user-visible `No path to next viewpoint` failures are **geometry-bound,
  not time-bound**, and no `default` value rescues them.
- Finding I's terminal lock runs at **70 Hz = a 14.2 ms cycle**, which cannot contain the 52 ms
  coarse cost matrix. **The 21.7 %-of-flight-time sink does not execute the code a budget change
  modifies.** The motivation I reached for was the wrong one.

**Not a re-run of P34.** P34/P33/P14 closed *airframe* `max_vel`. This is planner compute time — no
shared parameter, code path or metric. Recorded explicitly so `MISSION.md`'s "what NOT to do" line is
not misread as barring it.

**What survives.** A narrow case for `default` 0.001 → 0.004 only, acting through the **connectivity
graph**: the 188 default-profile timeouts in `updateGridsFromVoxelMap`/`updateGridsFrontierInfo` each
return a poison cost of `1000.0 + ‖p1−p2‖`, marking connected cell pairs unreachable. Measured
default-profile exposure is 8.4 ms/cycle, so 4× costs at most +25 ms against a 164 ms median cycle.

### REFUTED 2026-09-03 — the free measurement came back, and it kills both halves

Success-path logging landed and the answer is unambiguous. **Iterations a search needs when it
SUCCEEDS**, one flight, 298 successes:

| overload | budget | n | p50 | p90 | max | vs. its own mean TIMEOUT |
|---|---|---|---|---|---|---|
| `search` | default | 66 | **1** | 14 | **114** | times out at **189** |
| `searchUnknown` | coarse | 106 | **9** | 20 | 183 | times out at 22 |
| `searchBBox` | default | 25 | 46 | 126 | **137** | times out at **269** |
| `searchUnknownBBox` | default | 34 | 91 | 145 | 301 | times out at 279 |
| `searchUnknownOnlyBBox` | default | 67 | 95 | 167 | **229** | times out at **238** |

**A search that is going to succeed does so quickly — and every failure has already run longer than
almost every success.** For `search`, the overload that produces the user-visible
`No path to next viewpoint`, the longest success in the corpus is **114** iterations while failures
time out at a mean of **189**. For `searchBBox`, 137 against 269. The failing searches are not
"nearly there"; they are already well past the range in which success happens.

**The censoring objection, checked rather than waved away.** A success distribution truncated at the
budget would look like this artificially. It is not truncated: `search` successes stop at 114 while
the budget permits ~189, and `searchUnknownBBox` records successes at 301 — *above* its own mean
timeout. The distribution is genuinely bounded, not clipped.

**Verdict: do not fly any A* budget change.** More time does not convert these failures, because the
budget is not what they are short of. The failures are geometry-bound — the goal is genuinely not
reachable in the map as it stands — which independently re-derives B6's own conclusion that the
remaining route is a C++ blacklist/back-off patch, or better map connectivity, and not a knob.

**Cost of reaching this: one grep on a flight that was happening anyway.** The alternative was
15 flights per arm — 30 flights, about six hours — to measure an effect the mechanism could not
produce.

**The one measurement that decides it, and it costs nothing extra.** The instrumentation logged only
failures, so *how many iterations a search that SUCCEEDS needs* has never been measured — and a
timeout at 238 iterations means nothing until you know whether success takes 200 or 2000. The patch
now logs `REACH_END` as well. **Decision rule, pre-committed:** if successful default-profile
searches complete within ~950 iterations, `default:=0.004` is justified; if they need more, do not
fly it, and the honest fix is the C++ blacklist/back-off B6 already names.

## B28 — The operator's headline metric is too noisy to A/B at any reasonable n  ·  **MEASURED**

Variance across 17 comparable flights, and the flights-per-arm each implies for a 10 % relative shift
at 80 % power:

| metric | CV | flights per arm |
|---|---|---|
| `trace.plans.mean_speed_mps.p90` (a property of the PLAN) | **3.7 %** | **2** |
| `trace.tracking.moving.along_track_m.p90` | 10.5 % | 17 |
| `trace.tracking.frac_reference_moving` | 14.2 % | 32 |
| `trace.route.moving.frac_outside_safe_distance` | 21.7 % | **74** |
| `coverage.gained_m3` | 27 % (inherited) | ~115 |

**"Share of the flight spent outside the planner's safety margin" is the number that best expresses
the operator's collision complaint — and at CV 21.7 % it needs 74 flights per arm, about 14 hours per
comparison.** It cannot be the primary of any experiment run at a sane cadence. It is a *reporting*
metric, tracked across the campaign, not an experimental endpoint.

**The practical consequence, and it should shape every future design here:** prefer metrics computed
from the PLAN over metrics computed from the FLIGHT wherever the mechanism allows. A plan property
carries none of the aircraft's run-to-run variance — 3.7 % against 21.7 % is a **34x** difference in
required sample size. Where a change acts on the plan, gate it on the plan first and only spend
flights on the outcome once the mechanism is confirmed.

## B29 — Depth preprocessing deviates from the export contract, but the pipeline is empirically calibrated  ·  **DOWNGRADED — DO NOT CHANGE IT**

> **Read the 2026-09-03 in-flight result at the bottom first. It reverses the conclusion that the
> rest of this entry builds toward.** The entry is kept in full because the reasoning was wrong in an
> instructive way: every static check pointed one direction and the flying system pointed the other.

**The contract.** The ONNX the flight stack runs was produced by
`~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/export.py`, whose own verification helper
states the input format (`:215-219`):

```python
T.Compose([T.ToTensor(),
           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```

`ToTensor` scales to [0, 1]; `Normalize` then maps to roughly **[−2.1, +2.6]**. The wrapper's
`forward(image)` (`:37`) consumes that tensor directly — the normalisation happens **outside** the
exported graph.

**What the live pipeline feeds it.** `core/mapping/depth/depth_anything_v3.py:98`, the code every
flight runs (`DA3TensorRTModel`, loaded by `tasks/mapping/ros2/depth_processor_node.py:17`):

```python
img = img.astype(np.float32) * (1.0 / 255.0)
```

**[0, 1], and nothing else.** The same omission exists in the sibling `depth_engine_trt.py:248`.

**Three independent checks that the normalisation is not inside the graph:**
1. The export wrapper takes the already-normalised tensor; the transform is applied by the caller.
2. A full-file scan of the 1.3 GB ONNX finds **no** ImageNet mean/std constants (one incidental
   float hit for 0.406 deep in the weights, none for the std triple, none for a 0.5/0.5/0.5 variant).
3. DA3's own code *de*-normalises with exactly these constants for visualisation
   (`Depth-Anything-3/src/depth_anything_3/api.py:405`, `utils/export/utils.py:19-26`) — which only
   makes sense if the model's input space is normalised.

**Why this could matter more than anything else in this file.** Depth decides where obstacles land
in the map. B26 established that **three quarters** of the aircraft's proximity to obstacles is
inherited from the plan, and the plan is built on this depth; the residual collisions are against
geometry the map never had. Every planner and controller finding in this campaign sits downstream of
it.

### TESTED 2026-09-03 — the alternative is excluded and the scale factor is measured

**Stage 1, run on the real engine and real frames.** Same engine, same frame, two preprocessings:

| preprocessing | depth p50 | p90 | max |
|---|---|---|---|
| live `[0,1]` only | **0.128** | 0.255 | 0.387 |
| + ImageNet normalisation | **0.997** | 1.726 | 2.178 |

**Correlation 0.9845, mean absolute difference 611 % of the live magnitude.** So:

- **The constant-folding alternative is REFUTED.** If the normalisation were folded into the graph
  the two would be identical; they differ by ~7.8x.
- **The effect is dominated by a global SCALE factor, not by structure** — 0.985 correlation means
  the model degrades gracefully and relative depth is largely preserved. My first report framed this
  as "every depth frame is wrong", which over-claimed: the *shape* is nearly right, the *scale* is not.

**And the scale is NOT absorbed downstream, which I checked because it would have made the bug
harmless.** The node applies `metric_depth = focal_px * net_output / 300.0`
(`depth_processor_node.py:335`). That divisor is **upstream DA3's own convention**, not a fudge
tuned against our preprocessing — the same expression appears verbatim in the export script's demo
(`onnx/export.py`, `metric_depth = focal * depth / 300.0`), and that demo feeds **ImageNet-normalised**
input. Upstream: normalised input + /300 = correct metres. Ours: un-normalised input + the same /300.

**Consistent with what the pipeline actually emits.** Live depth files
(`/tmp/rooster_depth/*.npy`, what the mapper consumes) read **p50 0.061 m, max 0.19 m** with the
aircraft parked. Predicted from the measured net output: `145.9 x 0.128 / 300 = 0.062 m`. The
arithmetic closes.

**Why this may be the root cause of the collision problem (B26).** A depth scale ~8x short places
every obstacle ~8x too close. Measured aircraft clearance to the nearest *mapped* obstacle is
**p50 0.57 m** — implausibly tight for a prison interior, and exactly what a foreshortened map
produces. It would also explain why the aircraft sits inside its own 0.40 m inflation radius 40 % of
the time while flying normally, and why pushing the plan away from obstacles (P23) improved every
clearance number without reducing contacts.

**STILL NOT ESTABLISHED, and it gates the fix:** the frames tested were captured with the aircraft
parked and close to geometry, so the absolute magnitudes above are not proof of in-flight error. The
remaining test is to capture in-flight frames and compare both preprocessings against a distance
known from Sphera's ground-truth pose and the map's wall positions. **Do not change the
preprocessing before that** — the correct fix may also require re-deriving the /300 divisor for this
camera, and shipping half of it would be worse than shipping neither.

**Superseded note.** **THE ALTERNATIVE I HAVE NOT EXCLUDED, and it inverts the conclusion.** Constant folding during
export could have absorbed the mean/std into the first convolution's weights and bias. That would
make the current preprocessing **correct**, would be invisible to a constant search, and adding the
normalisation would then *introduce* the error. I judge it unlikely — the export script normalises
outside the wrapper, so there is nothing in the traced graph to fold — but it is the one hypothesis
consistent with all the evidence above, and it is the reason nothing has been changed.

**THE DECISIVE TEST (needs the GPU, so run it at a break, not against a live flight).** Run the
engine on a recorded frame twice, with and without ImageNet normalisation, and compare both against
Sphera's known geometry — ground-truth range to a wall is available from the flight recordings.
Whichever preprocessing produces metrically correct depth is the right one. Secondary tell: DA3 is a
**metric** model, so absolute error against known distances is meaningful, not just relative shape.

**Do not "fix" this before that test.** A one-line change to a perception front end that every flight
depends on, made on inference rather than measurement, is exactly the shape of the errors this
campaign has spent the day cataloguing.


### IN-FLIGHT RESULT 2026-09-03 — the conclusion is REVERSED; the pipeline is fine

500 depth frames captured from the mapper's own input directory **while airborne**:

| | min | mean | max |
|---|---|---|---|
| frame median depth | 0.351 | **1.423** | 2.310 m |
| frame p90 depth | 1.525 | **3.126** | 4.778 m |
| frame max depth | 2.213 | **8.487** | 14.688 m |

**These are entirely plausible for a prison interior**, and they independently reproduce MISSION.md
P22's measurement of this same directory a fortnight earlier (p50 2.10, p90 3.90, p99 8.57, max
11.41). The end-to-end metric output is calibrated and correct.

**So the missing normalisation is NOT producing a gross scale error in flight, and adding it would
BREAK the pipeline** — in-flight depths would jump ~7.8x, putting the median at ~11 m and the max
past 100 m inside a room.

**Where my reasoning went wrong.** Every static check was individually right — the export contract
does specify ImageNet normalisation, it genuinely is not folded into the graph, and `/300` genuinely
is upstream's convention. The error was extrapolating from **parked frames**: with the aircraft on
the ground looking at nearby geometry, net output is small (p50 0.128) and metric depth is
correspondingly small (0.061 m), which I read as "the scale is 8x short". In flight the scene is
open, net output is ~8x larger, and the same arithmetic yields metrically correct depth. **The
absolute magnitude of a depth model's output is a property of the scene, not of the preprocessing —
I compared a parked measurement against an in-flight expectation.**

The most likely reconciliation is that this camera's focal (`focal_px` 145.9, a genuinely wide
~135 deg lens) differs enough from upstream's reference (858) that the deviation is absorbed. Whether
that is luck or an earlier deliberate calibration is not established and does not need to be: the
output is right.

**STATUS: closed as "do not act".** The deviation from the export contract is real and worth knowing,
and it may still cost some *relative* accuracy by running the model outside its trained input
distribution — but the two preprocessings correlate at **0.985**, so any such cost is small, and no
change should be made without re-deriving the divisor and re-validating in flight. **The clearance
p50 of 0.57 m that I attributed to a foreshortened map (B26) therefore has some other cause and that
line of reasoning is withdrawn.**

**Cost of the error: zero flights.** It was caught by capturing 500 frames from a flight that was
happening anyway, before any code changed.

## B30 — The map is ~37 % occupied voxels, higher than ray geometry predicts  ·  **OPEN, and deliberately not acted on**

The new mapping counters expose the map's composition for the first time. Across three consecutive
flights: **99 % of mapped columns contain an occupied voxel**, and **37–48 % of all *known* voxels
are occupied**.

A naive estimate says that should be much lower. A ray crossing 4 m at 0.2 m resolution marks ~20
free voxels and terminates on 1 occupied — about 5 %. Even at the measured DA3 median depth of 1.4 m
it is ~12 %. Observed is 3–5x that.

**The obvious explanation is REFUTED.** `CLAUDE.md` warns that the voxel map is long-lived with no
decay, so spurious returns should accumulate without bound. Sampled through one flight, the occupied
share rises 22.7 % → 38 % over the first ~95 s and then **sits flat at 37 % for the remaining 380 s**.
It plateaus; it does not accumulate. Whatever sets the ratio is structural, not a noise ratchet.

**Plausible innocent explanation, untested:** a short-range wide-FOV sensor in a cluttered interior
clears little volume per frame while seeing a great deal of surface, and the box spans z[−1, 3.8] so
both floor and ceiling are large in-box surfaces while the aircraft only clears a thin band at its
own altitude.

**Why it might still matter:** an over-dense occupied field would place obstacles closer than they
are, which is consistent with the implausibly tight measured clearance (aircraft p50 **0.57 m**,
inside its own 0.40 m inflation radius **40 %** of the time) and would make it a *different* root
cause for the collisions than B26's "geometry the map never had" — spurious obstacles rather than
missing ones.

**NOT ACTED ON, deliberately.** I have been wrong four times in two days by over-reading a single
number, twice by calling a working design decision a defect. The distinguishing measurement is
whether occupied voxels sit where real surfaces are: compare the occupied set against the known
prison wall positions, or against the free-space envelope a correct map would produce. Until that is
done this is an observation, not a finding.

## B31 — The course-slew limiter is saturated most of the time, and that is the real heading error  ·  **MEASURED 2026-09-03**

Diagnosing why F18's yaw feedforward destabilised the aircraft turned up something more useful than
the answer to that question.

The follower derives a desired course from the tracker's commanded velocity direction and rate-limits
the *commanded* course to 45 deg/s (`course_slew_deg_s`). Measuring the rate that limiter actually
applies, per control tick, over two full flights:

| arm | ticks | at the 0.785 rad/s ceiling | sign flips |
|---|---|---|---|
| control (ff = 0) | 9527 | **6804 = 71.4 %** | 7.3 % of ticks |
| candidate (ff = 1.0) | 9470 | 5024 = 53.1 % | 5.6 % of ticks |

**The commanded course is slewing at its maximum rate on 71 % of ticks and reversing direction on
7 %.** The nose is permanently chasing a course it never reaches. That is a structural explanation
for the 25 deg standing heading error, and a better one than "yaw is a P-loop": even a perfect yaw
controller could not hold a course whose *command* is rate-saturated two thirds of the time.

**It also explains F18's failure exactly.** Feeding that rate forward injects a signal that is at its
±45 deg/s ceiling most of the time and reverses every ~15 ticks — a bang-bang yaw command, not the
smooth feed-forward the design assumed. Turning went to 240 deg/m against a 41–85 range.

**Why the desired course jumps that fast, and where this connects.** The desired course is the
direction of the tracker's commanded velocity — and B25 measured the position-correction term at
**42–44 % of that command**, railed at its limit on 13–20 % of ticks. When the correction dominates,
the commanded direction is set by the *error* direction, which swings as the aircraft moves relative
to the plan. So the heading problem is downstream of the correction dominating the command, which is
downstream of the aircraft being far from the plan in the first place.

**Consequences for the fix, and none of them is "lower the gain":**
1. A lower feed-forward gain still injects a scaled square wave. It would reduce the amplitude, not
   the character.
2. **Filtering the fed-forward rate** is the minimum sensible change — feed a low-passed course rate,
   not the raw slew output.
3. The deeper fix is upstream: reduce how much the position correction dominates the commanded
   direction, which is B25's open question and now has a second reason to be answered.
4. **Do not raise `course_slew_deg_s`** to relieve the saturation. MISSION.md P17 records lowering it
   as the campaign's single biggest win; raising it walks straight back into the yaw limit cycle that
   F18 has just demonstrated is still one step away.

### B31 (cont.) — where the jitter comes from: two hypotheses, one refuted, one confirmed

Decomposing the direction rate of each term of the control law, over the same control flight
(9527 ticks, rates in rad/s, ceiling 0.785 = 45 deg/s):

| direction rate of | n | p50 | p90 | p99 | over ceiling |
|---|---|---|---|---|---|
| **plan feed-forward** | 5678 | **0.010** | 0.240 | 0.764 | **1.0 %** |
| position correction | 8660 | 0.365 | 1.632 | 7.375 | 26.5 % |
| commanded (ff+corr+damp) | 8975 | 0.382 | 2.629 | **36.8** | **31.6 %** |
| smoothed (post-EMA) | 8972 | 0.296 | 1.924 | 17.97 | 24.7 % |

**FALCON's B-spline is smooth** — its direction crosses the ceiling on 1 % of ticks. Everything
downstream is not. The correction also *dominates the magnitude*: median share of the command
**61 %**, and **100 % at p90** — on at least a tenth of ticks the feed-forward contributes nothing at
all and the command is pure correction.

**Hypothesis 1 — per-axis clamping distorts the direction (B2's mechanism). REFUTED.** Splitting by
how many horizontal axes sit on the ±1.0 rail, railed ticks are *calmer*, not worse: 4.5 % over the
ceiling railed vs **28.2 % free**. Railing pins the vector into a corner and stops it moving. Do not
port F8's `preserve_demand_direction` here expecting a heading win.

**Hypothesis 2 — the direction of a small vector is ill-conditioned. CONFIRMED, monotonically.**

| \|xy\| band (m/s) | share of ticks | commanded dir. rate p50 | over ceiling |
|---|---|---|---|
| 0.00–0.10 | 18.6 % | 0.952 | **54.8 %** |
| 0.10–0.25 | 29.3 % | 0.656 | 44.7 % |
| 0.25–0.50 | 19.4 % | 0.308 | 27.5 % |
| 0.50–1.00 | 15.7 % | 0.285 | 23.6 % |
| 1.00+ | 17.0 % | 0.174 | **8.3 %** |

This is the textbook *course-over-ground is undefined at low speed* problem. **48 % of ticks sit
below 0.25 m/s**, where roughly half of all direction changes exceed what the slew limiter can pass.
The limiter then saturates absorbing them — which is exactly the 71 % measured above.

**The follower already has the right gate, set an order of magnitude too low.**
`falcon_exploration_follower_node.py` steers only when `world_speed > course_min_speed`, and
`course_min_speed` defaults to **0.05 m/s**. The jitter is severe up to 0.25 and material to 0.50.
Raising that gate is F19.

## B32 — the aircraft is not *late*, it is *sideways*  ·  **MEASURED 2026-09-03**

The brief lists two tracking complaints: "many times we are not on the route and get very far from
it" and "it is also late in tracking many times". The trace now separates them. Eight consecutive
flights, 71 786 control ticks, error resolved into along-track (late/early) and cross-track
(off-route) components:

| component | p50 | p90 | max |
|---|---|---|---|
| along-track lag (+ = behind) | **0.00 m** | 1.12 | 4.45 |
| \|cross-track\| | **0.42 m** | 1.34 | 5.54 |

- **There is no systematic lateness.** Median lag is 0.00 m and the aircraft is behind the reference
  on **49 %** of ticks — a coin flip. Lag is symmetric noise of about ±1.1 m at p90, not a lag.
- **Cross-track is the larger error on 62 % of ticks**, with a p90 of 1.34 m and excursions to 5.5 m.
  "Very far off the route" is the accurate description; "late" is the same noise seen end-on.

**Consequence for what to fix.** Lead/feed-forward terms (`accel_lead_s`, and F18's course-rate
feed-forward) address *lag*, and there is no lag to address — which is a second, independent reason
F18 had nothing to win. Effort belongs on the lateral/steering axis.

**And it feeds back into B31.** A cross-track correction points *perpendicular* to travel, so its
sign reverses every time the aircraft crosses the path — a second jitter source on top of the
small-vector ill-conditioning, and one that operates at *any* speed. This predicts F19 will reduce
the ceiling saturation without eliminating it, matching the offline replay (63 % → 33 %, not → 0 %).

### B31 (correction) — the "48 % of ticks" figure was one flight, not the fleet

B31 above quotes **48 % of ticks below 0.25 m/s**. That is flight `095510Z` alone. Measured across
twelve consecutive flights the exposure is **10 % – 78 %, mean 30 %, CV 67 %** — it varies eightfold
between flights. The mechanism and its direction are unchanged, but the *leverage* is not a constant:
how much F19 can win depends on how slow that particular flight was, which is itself highly variable.

The slew-saturated fraction is far steadier — **mean 61 %, sd 8 pp, CV 13 %** over the same twelve
flights — so it is the better endpoint even though it is a mechanism metric rather than an outcome.

**Consequence, and it constrains every experiment from here.** Per-flight CVs, twelve flights:

| metric | CV | n/arm to resolve a 15 % shift |
|---|---|---|
| `tracking.hdg_err_deg.median` | 10.9 % | 8 |
| slew-saturated fraction | 13.3 % | 12 |
| `trace.motion.distance_m` | 19.6 % | 27 |
| `clearance.abs_cross_track_m.median` | **22.6 %** | **36** |
| `coverage.gained_m3` | 26.2 % | 48 |
| `motion.turning_deg_per_m` | 26.3 % | 48 |
| `clearance.aircraft_frac_inside_inflation` | **43.9 %** | **134** |

**The three things the operator actually asked about — route adherence, collisions and coverage — are
the three least resolvable metrics we have.** At an affordable n = 6 per arm the smallest detectable
effects are ~37 % (cross-track) and ~71 % (inflation contacts). They are reporting metrics and
guard rails; they cannot be primaries. Primaries must be mechanism metrics, and the argument that a
mechanism win becomes an outcome win has to be made on physics, not on these intervals.

## B33 — The aircraft gets HELD against geometry, and 3.6 % of all flight time is spent there  ·  **CONFIRMED 2026-09-03**

The brief says "you have no way of knowing if it's hitting a wall, so it could get stuck in the wall
and then you apply force and it doesn't move." That is happening, it is measurable with signals
already recorded, and it is the single largest loss the campaign has found.

**The signature.** Tilt (`max(|roll|,|pitch|)`) above 40 deg while ground speed is below 0.05 m/s.
Scanning **969 flights** of recorded telemetry:

| | |
|---|---|
| flights with at least one episode ≥ 2 s | **355 / 969 = 37 %** |
| fleet-wide time in this state | **15 681 s of 437 574 s = 3.58 %** |
| mean per affected flight | 44 s |
| worst flights | 53 – **87.5 %** of the entire flight (one lost 381 s of 435 s) |

**It is real, and it is not any of the usual artefacts:**
- **Not the duplicate pawn (B22):** 0.0 % of these samples carry the `suspect` flag.
- **Not on the ground:** altitude during episodes is p10 0.54, p50 0.82, p90 **2.51 m**.
- **Not a corrupt attitude signal:** a down-facing ranger on an aircraft tilted by θ reads `z/cos θ`.
  Measured `ranger/z` during episodes is **1.70**, i.e. θ ≈ 54 deg — self-consistent with the
  reported tilt. The aircraft really is leaning at ~50 deg, airborne, and not moving.

A multirotor cannot hover at 50 deg. Something is holding it: a wall, a ledge, an obstacle.

**What the stack does about it: cuts the motors' horizontal drive and waits.** The tilt reflex
(`~tilt_limit_deg`, added 2026-08-17 after a live capsize) fires, publishes zero velocity, resets the
tracker, and holds until tilt falls under `tilt_resume_deg`. On flight `102543Z` that was **one
contiguous 146 s episode, 31 % of the flight**, and `escaping` fired on **0** ticks — the recovery
machinery keys on position-pinning and never saw this.

**Hypothesis, explicitly NOT yet causally tested:** the reflex's comment says it holds "so it settles
back level rather than tipping further under continued translation commands". That is right for the
free capsize it was written for, and cannot work when contact is what holds the attitude — with drive
cut, nothing pushes the aircraft off the obstacle, so the state is self-sustaining until something
external changes. **I cannot test this from the data**: all recorded telemetry begins 2026-08-18, the
day *after* the reflex landed, so there is no before/after to compare. The argument is from reading
the code, and it stays a hypothesis until a fix is flown.

**The two tilt cases are distinguishable with what is already recorded**, which is what makes a fix
tractable: a free capsize is *falling* (large negative `vz`), while a contact-held tilt has `vz ≈ 0`
and a `ranger/z` ratio consistent with the tilt — still supported at height. See F20.

### B33 (cont.) — the traps are a dozen fixed places, and the aircraft returns to them flight after flight

`tools/falcon_campaign/contact_hold_survey.py` replays every recorded flight through the detector and
writes one record per episode, **with its map location**, to `runs/_analysis/contact_holds.jsonl`
(kept for the visualization the brief asked for). Over **985 runs: 652 episodes, 357 flights (36 %),
16 233 s held.**

**They are not spread over the map. They are a dozen spots.** Binned to 2 m cells, only 77 cells are
ever involved, and:

- **half of all episodes occur in 8 cells**;
- the **top 12 cells hold 62 % of episodes and 68 % of the lost time** (10 965 s);
- the worst single cell trapped the aircraft **68 times across 46 different flights**.

**And it is the place, not just the traffic.** Normalising by time actually spent in each cell
(occupancy sampled over 120 runs, scaled), against a fleet baseline of **3.7 %** of time held:

| cell | held s | occupied s | held per 100 s there |
|---|---|---|---|
| 46,−18 | 1898 | 8139 | **23.3 %** |
| 54,−24 | 1683 | 11997 | 14.0 % |
| 46,−22 | 1333 | 4153 | **32.1 %** |
| 58,−22 | 675 | 3986 | 16.9 % |

Six to nine times the baseline in the well-sampled cells. (Two thinly-sampled cells score above
100 %, which only means their occupancy denominator is too small to trust — the table above lists the
ones with thousands of seconds behind them.) The coordinate range, x 12…80 and y −41.5…4.2, sits
inside the known prison extents, so these are one consistent frame.

**Which fix this rules out.** 44 % of affected flights have more than one episode and 31 % re-enter
the *same* 2 m cell within the flight — but repeat visits are only **21 % of the lost time**. The
other **79 % is that flight's first encounter**, which a within-flight memory cannot prevent by
construction. So `blockage_memory` used per-flight (B26's plan) has a hard ceiling of about a fifth
of the loss. What the data actually supports is **persistent, cross-flight** knowledge: the same
places trap the aircraft on flight after flight, and only knowledge that outlives a flight can act on
the first encounter. See F21.

## B34 — Half the flight the PLAN is standing still, and it is a cadence mismatch  ·  **MEASURED 2026-09-03**

The largest loss the campaign has measured, and it is upstream of every control fix attempted so far.

**Across 8 consecutive flights (76 142 control ticks), FALCON's own reference velocity is below
0.05 m/s on 50 % of ticks.** Not the aircraft — the *reference*. Half the flight, nothing is asking
the aircraft to go anywhere. The command is only that still 9 % of the time, because the tracker's
position correction keeps pushing to close residual error against a target that has stopped.

**Reference speed against time since that trajectory was published:**

| age (s) | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| p50 speed (m/s) | 0.341 | 0.384 | 0.093 | **0.000** | 0.000 | 0.000 |
| frac below 0.05 | 21 % | 20 % | 47 % | 59 % | **93 %** | **100 %** |

**Every trajectory carries about three seconds of motion and then goes dead**, regardless of the
~11 s its knot span nominally covers — FALCON plans to a viewpoint and comes to rest there. Meanwhile
plans arrive every **3.28 s at p50 and 4.88 s at p90**. So the system is racing: when the next plan
lands inside ~3 s the aircraft keeps moving, and when it does not, it hovers. That race is the 50 %.

**Three explanations tested and rejected, in order:**
1. **Planning is too slow.** No. Solve time is p50 **11.8 ms**, p90 39.8 ms, and only **2 %** of
   689 118 calls overran their budget. (The frequent `[FSM] Total time too long!` is a bare
   `ROS_ERROR_COND` log line — it rejects nothing, and I initially misread it as a rejection.)
2. **The plan is deliberately rotating in place at viewpoints** — FALCON does cost yaw explicitly.
   No: during parked ticks the reference `yaw_dot` is p50 **0.000**, p90 0.005 rad/s. Only 7 % of
   parked ticks are yawing. The plan is not turning, it is doing nothing.
3. **Trajectories expire before the next one arrives.** No, not by knot span: they are 11.2 s long
   and replaced every 3.3 s, so a fresh plan always exists. The expiry is in the *motion content*,
   not the timestamps.

**And it is independent of B33.** Long parked-reference episodes (> 10 s, 37 of them, 886 s) overlap
contact holds by **0 seconds** — sensibly, because while the airframe is wedged the plan keeps
advancing. These are two separate losses that must be fixed separately.

**Where the time sits:** 932 parked episodes over 8 flights (116 per flight), median 0.40 s, but the
long tail owns the total — episodes over 10 s hold **46 %** of all parked time and nine episodes over
30 s hold 26 %.

**What this means for everything else in this file.** Course jitter, heading error and the tracking
gains are all downstream of a reference that is stationary half the time. A tracker cannot follow a
plan that is not moving, and coverage cannot exceed what the plan asks for.

## B35 — Three FSM knobs are written to a namespace FALCON does not read  ·  **CONFIRMED 2026-09-03**

Found while looking for the replan cadence behind B34.

`nav_stack.launch` sets `/fsm/replan_thresh1`, `/fsm/replan_thresh2` and `/fsm/replan_thresh3`.
`exploration_fsm.cpp` reads `/exploration_manager/fsm/replan_thresh{1,2,3}`. Read back off the live
parameter server, mid-flight:

| knob | what we set (`/fsm/`) | what FALCON reads (`/exploration_manager/fsm/`) |
|---|---|---|
| `replan_thresh1` | 2.0 | **0.05** |
| `replan_thresh2` | 2.0 | **0.2** |
| `replan_thresh3` | 2.0 | **3.0** |

**None of the three has ever taken effect.** FALCON has been running its stock
`exploration_manager.yaml` values throughout the campaign, and `replan_thresh3 = 3.0` is exactly the
periodic replan interval that B34 measured at 3.28 s — the cadence has never been under our control.

**Three neighbouring knobs are dead for a different reason.** `/fsm/replan_time`,
`/fsm/thresh_replan` and `/fsm/thresh_no_replan` do not appear anywhere in the FALCON source at all
(they look like leftovers from a FAST-Planner-era config). They are inert whatever namespace they are
written to.

**What is NOT broken, and why the distinction matters.** `/fsm/slow_traj_ratio_min` and
`/fsm/slow_traj_target_vel` exist *only* under `/fsm/` — they are parameters our own patch added, and
our patch reads them there. So the F17 rescale experiment did reach the planner and its verdict
stands. The rule is: **our patch params live at `/fsm/`, stock FALCON FSM params live at
`/exploration_manager/fsm/`**, and the launch file mixed them.

**Ordering is not the problem and does not need changing.** `nav_stack.launch` loads FALCON's yamls
at lines 1341–1353 and sets these params at 1636, so a correctly-namespaced override lands after the
yaml and wins.

Fixed in F22, deliberately in two steps: the namespace is corrected while the *values* are set to
FALCON's current effective ones (0.05 / 0.2 / 3.0), so the repair itself changes no behaviour and the
experiment that follows moves exactly one threshold.

### B34 (cont.) — the flight time budget: less than half the flight is productive

The brief asks "how much time does it spend on other things?". Classifying every control tick over
ten consecutive flights (95 014 ticks), each tick counted once in priority order:

| what the aircraft is doing | share |
|---|---|
| **flying the plan** | **47.0 %** |
| **plan parked — the reference is standing still** | **46.0 %** |
| tilt cut, drive zeroed (mostly B33 wall contact) | 4.2 % |
| escaping | 2.5 % |
| plan moving but the command is ~0 | 0.3 % |

Per flight, "flying the plan" ranges from **19.8 % to 67.0 %**. The two worst flights spent 75–78 %
of their window with a parked plan; the one with a long wall contact spent 31 % under a tilt cut.

**So the productive fraction of a flight is under a half, and the single largest non-productive
category is not the controller, the tracker, the map or the airframe — it is that FALCON's own
reference is standing still.** Every per-flight metric in this campaign — coverage, distance,
stationary fraction, even the course-steering exposure that F19 was measured against — is scaled by
this number, which is why it varies so much between flights and why control experiments run on top of
it are noisy.

### B34 (cont.) — what a parked reference actually costs the aircraft

The plan standing still does not freeze the airframe outright: the tracker's position correction
keeps pushing until the aircraft converges on the parked target. Joining the aircraft's own odometry
speed to the reference state, over 76 061 matched ticks:

| reference state | aircraft speed p50 | p90 | share below 0.05 m/s |
|---|---|---|---|
| **parked** | **0.05 m/s** | 0.72 | **52 %** |
| moving | **0.36 m/s** | 1.11 | 18 % |

So the parked half of the flight is flown at about **one seventh** the speed of the moving half, and
the aircraft is genuinely stationary for half of it. The cost is real but not total, which is the
honest version of the claim: the loss is a large speed penalty across ~46 % of the flight, not a dead
aircraft for 46 % of the flight.

### B34 (cont.) — and it is an active command, not a dropped topic

The last alternative explanation, closed. Reference message age during parked ticks is **p50 0.005 s,
p99 0.010 s, 0.0 % older than 0.5 s** — identical to the moving ticks, and consistent with the 100 Hz
`pos_cmd` stream. So `traj_server` is not silent and the follower is not holding a stale message:
**FALCON is actively publishing "stay here", a hundred times a second, for 46 % of the flight.**

### B34 (cont.) — the number that sizes the fix: 1.95 s of motion in a 3.20 s trajectory life

Per trajectory, over 992 of them:

| | p25 | p50 | p75 | p90 |
|---|---|---|---|---|
| seconds of **motion** in it | 1.60 | **1.95** | 2.65 | 3.55 |
| seconds it stayed **current** | 2.45 | **3.20** | 4.05 | 4.90 |
| **dead** seconds (current but still) | 0.45 | 0.50 | **2.40** | 3.95 |

**1939 s of motion inside 3570 s of trajectory life — 54 % productive.** The 3.20 s median life is
`replan_thresh3 = 3.0` showing through, and the median trajectory runs out of motion at 1.95 s. The
dead time is heavily skewed: half of trajectories waste under half a second, the top quartile wastes
2.4 s or more.

This sizes F22 directly. **2.0 s would cut the dead tail while almost never truncating motion; 1.5 s
truncates the median trajectory by ~0.45 s of motion but removes more of the tail.** F22 flies 1.5
because it is the more decisive test of the mechanism and because a truncation is cheap — FALCON
replans from the current *state*, velocity included, so the motion continues rather than restarting
from rest. If 1.5 trips the coverage or distance guard, **2.0 is the pre-declared fallback**, not a
new hypothesis.

### B33 (cont.) — why the threshold is 40 degrees, checked against the data

Joint distribution of tilt and motion over twenty flights:

| tilt | moving | still | % still |
|---|---|---|---|
| 0–20° | 132 950 | 34 670 | 21 % |
| 20–30° | 1 341 | 3 255 | 71 % |
| 30–40° | 302 | 645 | 68 % |
| **40–50°** | 91 | 1 943 | **96 %** |
| 50–60° | 19 | 606 | 97 % |
| 60–90° | 8 | 632 | **99 %** |

Above 40° the airframe is stationary on **96 %** of samples, so the threshold separates contact from
aggressive flight almost perfectly; the speed condition discards only 118 of 3 299 high-tilt samples.

**The 20–40° band is deliberately left out even though it is 68–71 % still**, and that is a judgement
worth writing down rather than a gap. It is very likely the *persistent 25–30° roll bias* recorded
against the Rooster altitude work, which is an unexplained standing attitude offset during ordinary
station-keeping, not wall contact. Counting it would inflate every contact statistic with hovering.
If that bias is ever explained or removed, this threshold should be revisited — until then 40° sits
safely above it, and the cost is that genuinely gentle contacts are under-counted.

## B36 — `obstacles_inflation` is a dead knob, and a collision metric is named after it  ·  **CONFIRMED 2026-09-03**

Found by auditing every global `<param>` `nav_stack.launch` sets against what the FALCON source
actually reads — the systematic version of the B35 hunt.

**The string `inflation` does not appear anywhere in this FALCON.** Not in the C++, not in the
headers, not in any yaml, and `strings` on the built `exploration_node` finds it in no binary. The
parameter is on the live server at 0.4 and nothing reads it. FALCON enforces clearance a different
way entirely: `bspline_opt/safe_distance` (which *is* read, in `bspline_optimizer.cpp:33`) against the
voxel mapper's **ESDF**. There is no inflation step to configure.

**My rebuilds are not the cause, which I checked before writing this.** The pre-rebuild image
`falcon-ros:pre-sparx-instrumentation` has no inflation either, so the knob has been dead for the
whole Rooster/Sphera campaign rather than dropped by a Dockerfile change. (The patch may well be live
in `falcon_sjtu`, which carries far more C++ patches — this is the porting gap, not a regression.)

**Two consequences, and the second one matters more.**

1. **The v3.0 experiment's stated mechanism was impossible.** Its verdict reads "mechanism worked
   (A* 'no path' fell ~10×)" for `obstacles_inflation 0.40 → 0.30`. In this image that parameter
   cannot have caused anything. The *conclusion* was to revert, so nothing harmful was adopted, but
   the supporting reasoning should not be cited again.
2. **`clearance.aircraft_frac_inside_inflation` is measured against a radius nothing enforces.**
   `analyze.py` takes its threshold from `config.OBSTACLES_INFLATION` (0.40 m), and B26 built the
   collision picture on it — "the aircraft is inside the 0.40 m inflation 40 % of the time, its own
   reference 30 %". The *distances* are real and the metric is still a valid clearance measure, but
   the name asserts a planner behaviour that does not exist, and the inference "three-quarters of
   contacts are inherited from the plan" was reasoning about a margin the planner was never applying.
   **The honest restatement: the aircraft spends 40 % of its time within 0.40 m of mapped obstacles,
   and FALCON's actual clearance lever is `safe_distance`.**

Not deleting the parameter: it is harmless on the server, and a future port from `falcon_sjtu` may
make it real. It is annotated as dead where it is defined so nobody spends a campaign on it again.

**Scope of the audit, so the clean result is on record.** All 59 global `<param>` names set by
`nav_stack.launch` were checked against the FALCON source and the built binaries. Findings:

- `/voxel_mapping/obstacles_inflation` — **dead** (above).
- `/fsm/replan_time`, `/fsm/thresh_replan`, `/fsm/thresh_no_replan` — dead, and already known (B35);
  they look like FAST-Planner-era leftovers.
- `/certainty/log_path`, `/thinking/log_path` — absent from FALCON *correctly*: they are our own
  adapter nodes' logging paths and are consumed there.
- **Everything else resolves to a real read**, including the levers the campaign leans on:
  `bspline_opt/safe_distance`, `frontier_finder/cluster_min`,
  `frontier_finder/blocked_region_radius`, `exploration/tour_commit_max_s` and the `astar/profile/*`
  entries.

One methodological note, because it nearly produced a false negative twice. The first pass searched
`/catkin_ws/src` including `falcon_adapter`, so **every parameter matched our own launch file** and
the audit reported nothing wrong. And `strings` on `exploration_node` is not evidence either:
`safe_distance` and `cluster_min` are read inside *libraries*, so they score zero in the executable
while being perfectly live — the same trap as the A* marker count in F15. **Search the planner source
excluding our own files, and trust the source over the binary.**

### B34 (CORRECTION) — the "planning is not slow" rejection used the wrong population

B34 above rejects "planning is too slow" with *"solve time is p50 11.8 ms and only 2 % of 689 118
calls overran their budget"*. **That statistic is wrong for the flights B34 is about.** It was taken
from a glob spanning the whole `runs/2026090*` tree, which is dominated by older, faster-planning
epochs. Measured on the eight flights the rest of B34 analyses, reading only each run's
`falcon_roslaunch.log`:

| | p50 | p90 | p99 | overran its budget |
|---|---|---|---|---|
| planner solve time, 1768 calls | **159 ms** | 199 ms | 243 ms | **86 %** |

So planning **is** slow relative to its own budget — it overruns on roughly nine calls in ten, by
about 60 ms against a 100 ms `replan_duration_`, which is exactly what the frequent
`[FSM] Total time too long!` line has been reporting all along.

**The conclusion does not change, and here is why, stated properly this time.** The FSM sets the new
trajectory's start at `now + replan_duration_`, so a 60 ms overrun starts it ~60 ms in the past — a
small discontinuity, not seconds of stillness. Even at 200 ms per solve across ~235 solves, planning
accounts for ~47 s of a 430 s flight, and it does not stop the aircraft: the *previous* trajectory
keeps playing while the next is computed. The parked reference is 46 % of the flight, so planning
latency cannot be its cause. The cadence mismatch — **1.95 s of motion inside a 3.20 s trajectory
life** — remains the explanation.

**What the recount did change is the picture of the planning cycle.** The planner is called every
~1.85 s but only about **55 %** of calls end in a published trajectory (plan failures plus
collision rejections before publishing), which is why plans arrive every ~3.3 s despite the more
frequent attempts. That makes the first F22 candidate's drop in `plan_fail` (17 against 37 and 47)
more interesting than it first looked: **F22 did not increase the number of planning attempts at all
— 232 against 226 and 235 — it increased the share that succeeded.** A plausible reading is that
replanning at 1.5 s happens while the aircraft is still moving mid-trajectory, whereas at 3.0 s it
has already stopped at the endpoint, and planning out of a stopped pose near geometry fails more
often. That is a hypothesis from n=1 and is not yet tested.

### B34 (cont.) — the stall is self-reinforcing: plan failures happen when the aircraft is already stopped

Joining every `[FSM]` planning call to the aircraft's measured speed at that instant, over eight
flights:

| | n | speed p50 | p90 | stopped (< 0.05 m/s) |
|---|---|---|---|---|
| all planning calls | 1105 | 0.30 m/s | 1.10 | 25 % |
| **calls that FAILED** | 120 | **0.03 m/s** | 0.23 | **77 %** |

**Planning fails three times as often from a standstill as at the base rate.** That closes a loop:

1. the trajectory runs out of motion at ~1.95 s and the reference parks;
2. the aircraft stops;
3. the next replan is attempted from a stopped pose, where 77 % of failures occur;
4. no trajectory is published, so the aircraft stays stopped — back to (2).

**The direction of causality *inside* the loop cannot be separated from this data, and I am not going
to claim it.** A failed plan leaves the aircraft stopped, so the next call is also measured at low
speed; "stopped causes failure" and "failure causes stopped" produce the same correlation. What is
established is that the two states coincide strongly and reinforce each other.

**It does explain the F22 candidate cleanly, and in a way that is falsifiable.** F22 did not increase
planning attempts (232 vs 226/235) but cut failures (17 vs 37/47) and parked time (0.28 vs 0.46).
Replanning at 1.5 s fires while the aircraft is still moving — before step (2) — so it enters the
loop where planning succeeds. Whichever way the causality runs inside the loop, breaking it earlier
helps. **The prediction that follows: across the full F22 sample, the candidate's planning calls
should show a higher median speed than the control's.** That is checkable at n, and it is a
different measurement from the ones the experiment is already powered for.

## B37 — The planner's real bottleneck is collision REJECTION, not path search  ·  **MEASURED 2026-09-03**

Counting FSM outcomes in one control flight (`112908Z`), which is representative of the eight:

| FSM event | count |
|---|---|
| plans produced (`PLAN_TRAJ → PUB_TRAJ`) | 190 |
| published and executed (`PUB_TRAJ → EXEC_TRAJ`) | **122** |
| **rejected: collision before publishing** | **78** |
| of those, the initial path also collided | 69 (**88 %**) |
| `Plan fail` | 47 |

**41 % of everything the planner produces is thrown away by a collision check**, which is a larger
loss than outright planning failure, and the fallback to the un-optimised initial path rescues it
only 9 times in 78. **A\* is not the problem**: the `sparx_astar_legible` instrumentation added
earlier this campaign logs nothing at all in these flights, so path search is finding paths.

**What the check actually does** (`FastPlannerManager::checkTrajCollision`, planner_manager.cpp:384):
it walks the trajectory forward from `t_now + 0.02 s` until it has covered **6 metres of path**, and
rejects if any sampled point is `OCCUPIED`. Two things follow that I got wrong before reading it:

- It does **not** apply `safe_distance` or any margin — it tests raw occupancy. So this is not a
  clearance-tuning question.
- It starts *after* the current pose, so "the aircraft is standing inside its own margin, therefore
  every plan is in collision" is **not** the mechanism. I had assumed that; the code says otherwise.

So the optimiser is producing curves that pass through voxels the map calls occupied, within the
first six metres, on four attempts in ten.

**Hypothesis, not yet tested, and it links to two open items.** The map is ~37 % occupied voxels,
already flagged as higher than ray geometry predicts (B30), and the depth comes from DA3 with the
noise the brief warns about. Spurious occupied voxels would reject perfectly good trajectories at
exactly this check. That would make map quality — not planning, not control — the thing gating
throughput. **Testing it means separating "rejected at a voxel that is really an obstacle" from
"rejected at a voxel that is depth noise", which needs the rejection coordinates**, and
`checkTrajCollision` already logs them (`Collision detected at (x, y, z)`). That is the next
measurement, and it is cheap.

### B37 (cont.) — the rejections are real geometry, and they sit where the aircraft gets wedged

678 rejection coordinates recovered from eight flights' logs:

- **They cluster.** 116 distinct 2 m cells, and **half of all rejections fall in 18 of them**. Depth
  noise would scatter; this is structure.
- **They overlap B33's wall traps.** Rejection hot cells include (48, −18) and (46, −18), against
  hazard cells at (47, −17) and (47, −21) — the same region that has wedged the aircraft on dozens of
  separate flights. The places that block its trajectories are the places that trap it.
- **Height is mostly plausible**: z p50 1.32 m, 78 % inside the 0.6–1.8 m flight band. The remaining
  22 % — including a minimum of **−0.38 m, below the floor** — is the part that looks like map
  artefact rather than obstacle, but it is the minority.

**So the leading hypothesis of the previous entry is largely wrong: this is not mostly depth noise.**
The optimiser is being asked to fly through genuinely tight, genuinely occupied geometry, and the
check is doing its job. That is a better outcome for F21 than for a denoising effort — **the same
dozen cells account for the wedging, the trajectory rejections and a large share of the lost time**,
so keeping the aircraft out of them addresses all three at once rather than one.

It also means the 41 % rejection rate should not be "fixed" by loosening the check. The rejections
are true. The aircraft should not be there.

### B34 (cont.) — the planning-speed prediction does NOT confirm at n=2

The prediction recorded above was: *"across the full F22 sample, the candidate's planning calls should
show a higher median speed than the control's."* Tested at two candidates against seven controls:

| arm | planning calls | speed p50 | p90 | stopped (< 0.05 m/s) |
|---|---|---|---|---|
| candidate (replan 1.5 s) | 394 | 0.337 m/s | 1.103 | **29 %** |
| control (replan 3.0 s) | 1292 | 0.292 m/s | 1.100 | **24 %** |

**The two indicators disagree.** Median speed is 15 % higher as predicted, but the *stopped* fraction
is 5 points **worse**, which is the opposite of what the mechanism story implies. At two candidate
flights this is noise-dominated and settles nothing either way — but it is recorded here rather than
quietly dropped, because a prediction that only gets reported when it lands is not a prediction.

Re-test at the full sample. If the stopped fraction stays flat or worsens while the primary keeps
improving, then "replanning while still moving" is **not** how F22 works, and the real mechanism is
something else — most likely simply that a fresh plan arrives before the old one runs out of motion,
which needs no claim about the aircraft's speed at the moment of planning at all.

## B38 — The whole loss is one chain, and it starts at a dozen map cells  ·  **CONFIRMED 2026-09-03**

B34 said the plan is parked half the flight. F22 tried to fix that by replanning sooner and failed.
This is why: **the parked plan is not a cadence problem at all — it is the *consequence* of failed
replanning**, and the failures have a location.

**The measurement that settles it.** Across 14 flights, 268 parked-reference episodes longer than
2 s: **242 of them (90 %) contain a `Plan fail` or a pre-publish collision rejection.** If the plan
simply ran out of motion on a schedule, most episodes would contain no failure at all.

**And the planned profile shows what "running out" means.** Evaluating 571 published B-splines along
their own span, the planned speed is 0.486 m/s at the start, peaks at **0.733 m/s mid-curve**, and
decelerates to **exactly 0.000 at the end** — FALCON plans every trajectory to come to rest at its
viewpoint. Median plan length is 5.1 m. So while replanning succeeds the reference is always in the
healthy middle of a fresh curve; the moment a replan is *refused*, the current curve runs to its
planned stop and the reference sits at the endpoint until something succeeds.

**The chain, with each link measured separately:**

1. The aircraft approaches one of a dozen trouble cells (B33: half of all wedging in **8 cells**).
2. **41 % of produced trajectories are rejected** by the pre-publish collision check (B37), and the
   rejection coordinates cluster — **half in 18 cells**, overlapping the wedging cells.
3. A refused replan leaves the old curve to decelerate to its planned stop → **the reference parks**
   (B34: 46 % of flight time, 90 % of those episodes carrying a failure).
4. The aircraft crawls at **0.05 m/s** instead of 0.36 while that lasts (B34).
5. Sometimes it also gets physically **held** there (B33: 36 % of flights, 3.6 % of all flight time).

**Every one of those is the same small set of places.** That is why F22 could not work — replanning
twice as often against a 41 % refusal rate produces refusals twice as often, which is exactly what it
measured (plan failures fell, nothing else moved).

**It also promotes F21 from "a fix for wedging" to "the lever for the largest loss in the campaign".**
Keeping the aircraft out of those cells should reduce collision rejections, which should reduce
parked time, which should raise coverage — three effects with one cause. That is a strong claim and
it is *falsifiable*: if seeding the blocked regions does not reduce `exploration.plan_fail` and the
collision-rejection count, this chain is wrong.

## B39 — The campaign is power-starved by design, and a within-flight paired test would fix it  ·  **PROPOSAL 2026-09-03**

Every outcome the operator actually cares about is unaffordable to measure between flights:

| metric | CV | flights per arm for a 15 % effect |
|---|---|---|
| `tracking.hdg_err_deg.median` | 10.9 % | 8 |
| voxels per metre flown | 24.9 % | 43 |
| `clearance.abs_cross_track_m.median` | 22.6 % | 36 |
| `coverage.gained_m3` | 26.2 % | 48 |
| `clearance.aircraft_frac_inside_inflation` | 43.9 % | 134 |

At ~11 minutes a flight, 43 per arm is **16 hours** for one question. F24 is living this out right
now: its primary has walked 1.42 → 1.29 → 1.24 as n grew, and the honest verdict at n=12 will
probably be *inconclusive* — not because the change does nothing, but because the design cannot see a
20 % effect.

**Nearly all of that variance is between flights, not within them.** Battery state, where the tour
starts, which cells the aircraft happens to meet, whether it gets wedged — these differ hugely
between flights and barely within one. A **paired within-flight design** would remove them: alternate
the arm every ~60 s inside a single flight and compare segment against segment. Each flight then
yields its own matched pair, and the between-flight variance cancels instead of being averaged over.

**What it needs.** The follower reads `yaw_mode` once at startup, so the node would have to re-read
it on a timer — a small change, and the campaign would drive the alternation from outside. Metrics
already computed per tick (voxels gained, distance, heading error, cross-track) can be attributed to
segments with no new instrumentation.

**The confound that makes it non-trivial, stated so it is not discovered later.** A flight is not
stationary in time: early segments explore virgin space and late ones revisit, so *voxels per metre*
falls through a flight regardless of the arm (the measured 4.2× discovery decay, B10). Alternating
must therefore be frequent enough that adjacent segments face comparable maps, and the analysis must
compare *neighbouring* segments rather than pooling all A against all B. Get that wrong and the
design manufactures an effect from the decay curve alone.

**Not all questions suit it.** Anything whose effect accumulates over a whole flight — the hazard
seeding of F21, where the point is what the planner avoids over time — cannot be split into 60 s
slices. This is for control and perception changes that act tick-by-tick, which is most of what this
campaign tests.

### B39 (cont.) — which knobs can actually be alternated in flight

The design's reach is set by *where the parameter is read*, and that splits the campaign's knobs in
two:

- **Alternatable cheaply:** anything the Python follower reads, because it can re-read a rosparam on
  a timer with a few lines and no rebuild — `yaw_mode`, the blend thresholds, the course gate,
  tracker gains. This covers most control and perception questions, F24's included.
- **Not alternatable without a rebuild:** anything FALCON's C++ reads in a constructor via
  `nh.param(...)` — `bspline_opt/max_iteration_time` (F25), `safe_distance`, the FSM thresholds,
  `frontier_finder/*`. These are read once at node start, so alternating them means patching the C++
  to re-read, rebuilding the image, and restarting the stack.

So **F25 cannot use the paired design** without a patch, and F21 could not use it anyway (its effect
is cumulative). The first candidate for it is a *re-run of F24's question* with the arm alternating
inside each flight — which would also serve as the design's own validation, since F24's between-flight
answer will already be on record to compare against.

## B40 — The optimiser is not running out of time. It converges, and its answer still collides.  ·  **CONFIRMED 2026-09-03**

F27's instrumentation answered F25's question on the first flight, with no statistics at all.
**70 solves, every single one `result=4` — `NLOPT_XTOL_REACHED`.** Not one `NLOPT_MAXTIME_REACHED`.

| | |
|---|---|
| budget given | 0.010 s |
| elapsed, max over 70 solves | **0.001 s** |
| elapsed, p50 | **0.000 s** (sub-millisecond) |
| termination | 70/70 converged on `xtol_rel = 1e-4` |

**The optimiser uses at most a tenth of its budget and converges every time.** F25's premise — that
ten milliseconds was too little for an L-BFGS over 11–21 control points — is simply false, and
raising the budget to 40 ms could not have changed anything. That would have cost ~7 hours of flying
to discover behaviourally; it cost one flight to observe directly.

**Which makes the real finding sharper, not weaker.** The optimiser *converges*, and its converged
answer still passes through voxels the map calls `OCCUPIED` on **41 %** of attempts (B37). So the
cause lies in what it is converging *to*, and there are two candidates worth separating:

1. **The clearance penalty loses the trade.** It is soft — `cost += pow(dist - safe_distance, 2)` at
   weight 150 — against smoothness, endpoint, feasibility and time. A converged local minimum can
   still sit inside an obstacle if the other terms pay for it.
2. **The two halves disagree about the world.** The optimiser minimises against the **ESDF**;
   `checkTrajCollision` tests raw **occupancy**. If the distance field is stale, smoothed, or
   truncated relative to the occupancy grid, the optimiser can believe a curve is clear that the
   checker knows is not — and no amount of optimisation effort would fix that.

Candidate 2 is the more likely and the more interesting, because it would make the rejection rate a
**map-consistency** problem rather than a planning one. Distinguishing them is cheap: log the ESDF
distance at the exact point `checkTrajCollision` rejects. If the ESDF reports comfortable clearance
where occupancy says OCCUPIED, that settles it.

## B41 — The two halves agree about the world. The optimiser converges into obstacles it can see.  ·  **CONFIRMED 2026-09-03**

The pre-specified test from F27's second patch, answered on the first flight that carried it.
25 pre-publish rejections, each with the ESDF's own opinion at the exact rejecting point:

| ESDF distance at the rejecting point | count |
|---|---|
| < 0.05 m | **20** |
| 0.05 – 0.2 m | 5 |
| 0.2 – 0.55 m | 0 |
| ≥ 0.55 m (would mean "ESDF thinks it is clear") | **0** |

min 0.000, p50 **0.007**, max 0.096 — against `safe_distance = 0.55`.

**Candidate 2 is refuted outright: this is not a map-consistency problem.** The ESDF and the
occupancy grid agree completely — every rejected point is essentially *inside* an obstacle by both
accounts. Nothing here is stale, smoothed or truncated, and the fix does not lie in the mapping
pipeline.

**So the 41 % rejection rate is a trajectory-optimisation problem**, and combined with B40 (the
solver converges every time, in under a millisecond of its ten) the statement is now precise: **the
optimiser converges to curves that pass through obstacles its own distance field can see.**

**One sub-question remains, and it decides the fix.** The optimiser penalises **control points**
(`getDistanceAndGradient(q[i], …)`); the checker samples the **curve**. A cubic B-spline lies inside
its control hull, so an obstacle protruding into that hull can be nearer the curve than any control
point — every control point can satisfy 0.55 m while the curve still clips a corner.

- **If the control points are clear and only the curve is not**, the fix is to penalise sampled curve
  points, or to widen `safe_distance` enough to cover the hull gap. Raising the cost weight would do
  nothing, because the term being weighted is already satisfied.
- **If the control points are themselves inside**, the soft penalty genuinely lost the trade and the
  weight (currently 150) is the lever.

Arguing between them is unnecessary: logging the minimum ESDF distance over the rejected
trajectory's control points, beside the curve value already captured, separates them on one flight —
the same move that settled B40 and this entry.

## B42 — The optimiser converges with control points *inside* obstacles  ·  **CONFIRMED 2026-09-03**

The pre-specified test, answered on the first flight carrying it. 14 rejections, each reporting the
minimum ESDF clearance over the rejected trajectory's own control points:

| minimum control-point clearance | count |
|---|---|
| **< 0.05 m — a control point is INSIDE an obstacle** | **11** |
| 0.05 – 0.55 m — outside, but violating the 0.55 m margin | 3 |
| ≥ 0.55 m — satisfies the margin, so the curve alone clips | **0** |

The curve was nearer the obstacle than any control point in only 2 of 14 cases.

**The hull-gap explanation is refuted as the dominant mechanism** — and it was the one I had called
more likely, before the data. Not a single rejection had control points satisfying `safe_distance`.
The optimiser is not being defeated by B-spline geometry; it is **converging with its own penalised
variables sitting inside obstacles**.

**And the penalty is not small when that happens.** At `dist = 0` the clearance term contributes
`(0 − 0.55)² × 150 = 45.4` per violating control point, against total converged costs of 33–668. The
optimiser accepts a large, clearly-signalled penalty rather than moving the point out.

**Which makes premature convergence the leading explanation, not weight.** B40 measured every solve
terminating on `xtol_rel = 1e-4` in **under a millisecond** — the solver barely moves from its
initial guess before declaring itself done. If the initial trajectory (seeded from A\* or the previous
curve) already clips an obstacle, an L-BFGS that converges in 1 ms has effectively accepted it. That
ties B40 and B42 into one statement: **the optimiser stops too early to fix a bad initial guess, and
the clearance cost never gets the iterations it would need to matter.**

Three candidate levers now, in the order the evidence supports them:
1. **`xtol_rel`** (1e-4) — too loose, so it converges before doing useful work. Testable directly:
   log iteration counts, which `maxeval` is not limiting either (`result` was never 5).
2. **The initial guess** — if it already violates clearance, fixing it upstream is worth more than
   any amount of downstream optimisation.
3. **`pos/distance` weight** (150) — the weakest candidate, because the term is already contributing
   45 per violating point and losing anyway.

## B43 — The optimiser does real work and still ends inside obstacles  ·  **CONFIRMED 2026-09-03**

B42 proposed premature convergence: every solve terminates on `xtol_rel` in under a millisecond, so
perhaps it accepts its initial guess. **Measured over 34 solves, that is false.** Displacement between
the variables handed to NLopt and the ones it returns:

| travel (max over control points, per solve) | |
|---|---|
| p50 | **0.216 m** |
| p90 | 0.514 m |
| max | 1.266 m |
| solves moving < 1 cm | **1 of 34** |

**The optimiser moves control points by about 22 cm at the median and up to 1.27 m.** It is doing
substantial work, not rubber-stamping its input. So the lever is **not** `xtol_rel`, and tightening
tolerances would only burn CPU — the opposite of what I proposed one entry ago, refuted by the
measurement I set up to test it.

**Which leaves the cost landscape, and sharpens it to a specific, checkable mechanism.** The optimiser
moves points freely and still leaves some *inside* obstacles (B42: 11 of 14). The natural explanation
is that **a point already inside an obstacle has no gradient telling it which way is out**: an ESDF
typically reports distance 0 throughout an occupied region, so `getDistanceAndGradient` returns a zero
or meaningless gradient there and the clearance term — however heavily weighted — exerts no force.
The 45-unit penalty B42 measured is real but *flat*: it costs the solution nothing to move, and
nothing to stay.

If that is right, then all three levers from B42 are wrong, and the real one is **the initial guess**:
a control point that never starts inside an obstacle can be kept out by a gradient that only works
outside. That also explains why the weight is irrelevant — the term is a constant in the region where
it matters.

**The check is cheap and is the next step:** log the ESDF *gradient magnitude* at the violating
control points. A gradient near zero inside obstacles confirms it outright, and would make this a
FALCON design property rather than a tuning error — the fix then being to reject or repair initial
trajectories that already violate clearance, not to tune the optimiser at all.

## B44 — The ESDF is UNSIGNED, so a control point inside an obstacle feels no force to leave  ·  **CONFIRMED from source 2026-09-03**

The chain's root cause, and it did not need another flight or rebuild — it is visible in
`voxel_mapping/src/esdf.cpp`. The distance transform is seeded:

```
(occupancy == OCCUPIED) ? 0 : numeric_limits<double>::max()
```

**Occupied voxels are seeded with zero and there is no interior field at all.** The ESDF is unsigned:
distance is positive outside obstacles, and exactly 0 *everywhere* inside one.

`getDistanceAndGradient` computes its gradient as a trilinear finite difference over the eight
neighbouring voxels. Inside an obstacle more than one voxel thick, all eight read 0, so the gradient
is **exactly zero**. And the optimiser's clearance gradient is

```
gradient_q[i] += 2.0 * (dist - safe_distance_) * dist_grad;
```

With `dist_grad = 0`, that contribution is **zero**.

**So a control point inside an obstacle receives a flat penalty of `(0 − 0.55)² × 150 = 45.4` and no
direction in which to reduce it.** The cost is real, large, and *invisible to a gradient-based
solver*: it costs the same wherever the point moves, so L-BFGS has no reason to move it out. Every
observation now falls out of this one fact:

- **B40** — the solver converges quickly and legitimately: there is nothing left to descend.
- **B43** — it still moves other control points 22 cm at the median: the *outside* points have
  gradients and get optimised; the trapped ones cannot.
- **B42** — 11 of 14 rejected trajectories keep a control point inside an obstacle.
- **B41** — the ESDF agrees with occupancy at every rejection, because both say "inside".
- **B37/B38** — 41 % of trajectories are rejected, and the loss concentrates in a dozen cells: the
  places where a plan is most likely to start with a point inside geometry.

**This is a FALCON design property, not a tuning error, and it invalidates all three levers B42
proposed.** Raising `pos/distance` scales a constant. Tightening `xtol_rel` gives more iterations to
a zero gradient. Widening `safe_distance` moves the flat region's edge but not its flatness.

**The three fixes that would actually work**, in increasing order of invasiveness:
1. **Never let a control point start inside an obstacle** — repair or reject the initial trajectory
   before optimising. Cheapest, and it matches B42's "initial guess" candidate.
2. **Make the ESDF signed** — compute an interior distance so the gradient points outward. The
   standard remedy, and a well-understood addition to the same distance transform.
3. **Add an explicit escape term** for points with zero clearance, using e.g. the nearest free voxel
   direction rather than the ESDF gradient.

## B45 — Seeded blocked regions arrive at strike 1, which the code deliberately makes non-retiring  ·  **CONFIRMED from source 2026-09-04**

F21 seeded twelve hazard cells, FALCON logged **"Restored 12 blocked region(s)"**, `ttl = 0`, and the
aircraft's behaviour did not change. The reason is in `frontier_finder.cpp` and is entirely
deliberate — for a different case than seeding.

The restore path pushes `blocked_regions_strikes_.push_back(1)`, and the retirement test is:

```
if ((blocked_regions_[i] - tmp_ftr.average_).norm() < blockedRadiusFor(strikes_[i]))
```

with the code's own comment explaining that at **strike 1 the radius is deliberately under
`candidate_rmax`**, so *"this test cannot fire for the frontier that produced the blocked viewpoint.
That is correct exactly once: it stops a single early mistake retiring a transit route. From the
second strike the width exceeds candidate_rmax and the frontier retires."*

**So every seeded region is loaded at exactly the strength that is designed not to act.** The
mechanism was never given a chance, which is why F21's numbers were uninformative rather than
negative.

**And the design intent does not transfer to seeding.** Strike 1 is weak because a *runtime* veto may
be a one-off — the aircraft failed once at a viewpoint that is actually fine. A *seeded* region is the
opposite: it is the distillation of **652 wedging episodes over 985 flights**, and the worst cell in
it trapped the aircraft on **46 separate flights**. That is the definition of a repeat offender, and
strike 2 is what the code reserves for exactly that.

**Fix: seed at strike 2**, so evidence gathered across a campaign enters at the strength the code
gives to something that has already failed twice — which it demonstrably has. One line in the restore
block, and it makes F21 a real test instead of a no-op.

## B46 — Our own `ttl_max_doubling = 1` caps the blocked radius below the viewpoint sampling radius  ·  **CONFIRMED from source 2026-09-04**

F21's leak has a precise, arithmetic cause, and it is a parameter we set.

`FrontierFinder::blockedRadiusFor` computes

```
doublings = min(strikes - 1, blocked_region_ttl_max_doubling_)
radius    = min(blocked_region_radius_ * 2^doublings, blocked_region_radius_max_)
```

with the live values `blocked_region_radius = 1.5`, `radius_max = 6.0` and
**`ttl_max_doubling = 1`**. So:

| strikes | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|
| blocked radius | 1.5 m | **3.0 m** | 3.0 m | 3.0 m |

**The radius can never exceed 3.0 m.** Meanwhile `candidate_rmax = 5.5` — viewpoints are sampled up
to 5.5 m from a frontier's average. A frontier whose average lies 3.0–5.5 m from a seeded cell is
therefore **never retired**, yet can still offer viewpoints that land inside it. That is exactly the
leak F21 measured: candidate flights selecting 4 % and 28 % inside the cells, against controls at 8 %
and 12 %.

**And the code's own comment states the invariant this breaks:** *"From the second strike the width
exceeds candidate_rmax and the frontier retires."* At the FALCON default `ttl_max_doubling = 3` that
holds — 1.5 × 8 = 12, capped to 6.0, which does exceed 5.5. **We set it to 1**, which silently
reduces the guarantee to 3.0 and makes the sentence false. The comment is right about the design; our
parameter defeats it.

**This is the same shape as B45 and B36**: a mechanism that loads, reports success, and cannot act —
here because a tuning choice quietly violates an invariant the surrounding code documents in prose
but never asserts. **Two candidate fixes, both one parameter:** restore `ttl_max_doubling` to 3 (the
upstream default, restoring the documented invariant), or raise `blocked_region_radius` so that even
one doubling clears 5.5 m. The first is preferable — it fixes the invariant rather than out-scaling it.

### B46 (correction) — the fix was derived a campaign ago, written into the comment, and never applied to the value

My proposed remedy (raise `ttl_max_doubling` to 3) is **not** the right one, and the launch file says
so itself. Directly above the parameter sits a 2026-09-01 note recording the same geometry, measured
over 41 flights: *"A first strike that only shadows 1.5 m cannot retire a frontier whose viewpoints
are sampled out to candidate_rmax 5.5 m, so the tour re-offers the same target forever. **2.75 makes
strike 1 cover 2.75 m and strike >=2 escalate to min(5.5, radius_max) = 5.5 m exactly, which is the
geometric requirement stated above.**"*

**The argument immediately below that comment still reads `default="1.5"`.** The analysis was done,
the correct value was derived and written into the prose, and the number itself was never changed —
so `ttl_max_doubling = 1` is *correct* (it is what makes strike ≥2 escalate exactly once), and the
base radius is what is wrong.

| | strike 1 | strike ≥2 | covers `candidate_rmax` 5.5 m? |
|---|---|---|---|
| shipped `radius = 1.5` | 1.5 m | 3.0 m | **no** |
| intended `radius = 2.75` | 2.75 m | **5.5 m** | **exactly** |

That prior session also measured the cost of getting this wrong: **91.9 % of time-with-no-moving-
reference was A\*-plan-fail flooding, 86 % of it a single terminal lock on one unreachable viewpoint —
worst case 254 s and 17 925 plan fails on the same target**, with `sweepBlockedFrontiers` retiring
**zero** clusters while 10–11 shadows were re-struck. That is the same failure F21 has been unable to
suppress, from the same cause, diagnosed independently a campaign apart.

**Fix: set `frontier_blocked_radius = 2.75`, the value the file already specifies.** No rebuild — it
is a launch argument. This also explains why F21 leaked at strike 2 and why the seeding looked
mechanically sound yet changed nothing.
