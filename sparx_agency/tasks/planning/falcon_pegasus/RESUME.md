# Where this was left, and what to do next

Working note for the "ten clean full-office explorations in a row" task. Delete
it when the streak is done and the README carries the conclusions.

## Newest: measured feedforwards, validated deterministically

Two terms added to the tracker (defaults off in core; the Pegasus mission and
stub set the measured values via `_default_tracking`):

* **Drag feedforward** -- `0.176*v + 0.121` m/s^2 along the planned velocity,
  the airframe's measured curve. Removes the standing bias the damping term
  could only shrink and the gated integrator could not touch.
* **Attitude lead** -- the feedforward acceleration and jerk are sampled
  0.18 s ahead of the reference (PX4's attitude-response constant), so the
  aircraft is commanded the attitude the plan wants when it will actually
  have it. Feedback stays at the reference.

Validated A/B on the stub's AttitudeAircraft over a fixed corner route --
deterministic, unlike a live stub run, whose FALCON variance (0.18-2.61 m on
identical code) swamps the effect:

| configuration | mean err | max err |
|---|---|---|
| plain | 0.189 m | 0.343 m |
| drag only | 0.041 m | 0.143 m |
| lead only | 0.198 m | 0.272 m |
| **drag + lead (ships)** | **0.020 m** | **0.055 m** |

The lead ALONE slightly worsens the mean -- on a lagging aircraft the led
sample does not match where it is -- and pays off only once drag is cancelled.
`test_feedforward_flight.py` pins the division of labour as measured.

Also corrected: the bspline docstring claimed FALCON reparameterises its knots
for velocity feasibility; `reallocateTime` is never called in this build, the
knots are uniform in practice, and feasibility is only a soft optimiser cost.

Not yet flown on Isaac -- the container was replaced by SJTU work when this
landed. The first soak after it returns is the flight validation.

## Read this first: the streak could never have completed

`MIN_COVERAGE_M3` was **2200**, and the most any flight can ever reach in that
box is about **1465 m³**. The bar was 150% of the achievable maximum, so every
attempt was scored a failure on coverage no matter how well it flew. That, and
not the crashes, is why the counter sat at 0.

The mistake was treating the exploration box's *volume* as a coverage target.
FALCON's `Coverage` counts voxels that are no longer `UNKNOWN`, and a voxel only
leaves `UNKNOWN` when a camera ray reaches it. Wall interiors never do; nor does
outdoor space. Flood-filling the surveyed map from the spawn and adding the
occupied shell that reachable free space touches:

| | volume | observable |
|---|---|---|
| before (28.1 × 71.9 × 1.2) | 2424 m³ | 1465 m³ (60%) |
| now (28.1 × 65.9 × 1.2) | 2222 m³ | 1465 m³ (66%) |

The bar is now **1333 m³ = 91% of observable** — the same standard the old
number was reaching for, against the right denominator. The best run ever
recorded, the stub's 1396 m³, **already exceeds it**.

Re-derive this whenever the box changes. Do not lower it to whatever was
achieved.

**Only the south edge moved.** Flush west/east/north edges were tried and
reverted the same afternoon — see the README — and the reasoning that motivated
them was partly wrong, which is worth recording so nobody repeats it.

Round 9's 527-second wedge was attributed to the inset west edge being an
open-space frontier cut. It was not: the aircraft sat at (−21.2, 1.6), **1.8 m
inside** `box_min_x`, in a dead-end alcove bounded by walls at y = 3.0 and
y = 0.0. Nothing about the box edge put it there. The frontier cut is real in
principle, but it is not what that round died of, and moving the edge on that
theory cost a worse failure: round 10 wedged in west-wall shelving at 84 s with
313 m³. The cut is the cheaper failure.

Do not reach for obstacle inflation here: it is already applied and active
(`astar_inflate` 0.35 m in the launch). It is **XY only** by design, so it does
nothing about a slot between two horizontal slabs, which is what that was.

## State

**0 / 10 up to the sim-time fix; a fresh soak is running on it now.** Every
round below stopped at attempt 1, and no two for the same reason. Several were
not what the outcome field claimed — read `postmortem.py` output, not the
verdict string.

Everything from round 11 on was flown BEFORE `use_sim_time`, which cut stub
tracking error 6-7x. Treat those rounds as history: the dominant cause of their
failures is fixed.

| round | reported | actually | fix |
|---|---|---|---|
| 1 | `stalled` | mapper raycast `CHECK` abort | `fix_falcon_raycast_out_of_map.sh` |
| 2 | `stalled` | flew into cruise-height clutter at 1.6 m/s vs a 0.6 m/s plan | speed governor (`max_overspeed`) |
| 4 | `stalled` | **Isaac VRAM exhaustion at start-up**, plus the harness scoring a *stale* result | restart Kit per attempt; delete container output first; `isaac_gpu_oom` outcome |
| 5 | `crashed` | outer loop limit-cycling into the 35° tilt cap at ~0.5 Hz | vector position clamp; `reset()` drops the trajectory |
| 6 | `crashed` | hit a wall while 2.18 m off plan; the simulator's clock runs at 0.66× and FALCON's schedule does not | `link/sim_clock.py` re-bases the schedule |
| 7 | `crashed` | hit the **same pillar twice**, 90 s apart, and had no way to get off it | not yet fixed — see below |
| 8 | — | abandoned at attempt 1: the catch-up fix below landed mid-round | — |
| 9 | `crashed` | wedged 527 s of 878 in one cell at the inset west box edge; **control itself was healthy** — 0.36 m cross-track and 0.14 m *ahead* of plan over the 464 s before its first contact | — |
| 10 | `stalled` | flush box edges let it into west-wall shelving; wedged in a 20 cm slot at (−23.4, 0.2, 1.45), 84 s, 313 m³ | edges reverted |
| 11 | `crashed` | 715 m³; contacts on a replan-heavy route | — |
| 13 | `crashed` | 586 m³; repeated contacts, unwedge never reached | contact reflex |
| 15 | `crashed` | 593 m³ | — |
| 16 | — | VOID: two soaks raced each other, my error, no evidence | one soak at a time |
| 17 | `diverged` | 601 m³ — **first flight to survive its contacts**; reflex fired 3x | breadcrumb pruning |
| 18 | `diverged` | 414 m³; ended on the floor at full throttle | — |
| — | — | **`use_sim_time` landed here** | see below |

**The catch-up was flying the aircraft into things.** See the README section
"Chasing a deadline that does not exist". `max_catchup_speed` 0.5 → 0.15,
`max_overspeed` 0.5 → 0.25; measured speed fell from 1.00 to 0.67 m/s against a
0.57 m/s plan and time above the governor's ceiling from 30–44% to 6.4%, while
lag *improved*. This is the most likely reason the last two rounds hit things.

Round 7 was much healthier than 6 — 169 m, **1159 m³** covered, no node deaths,
no LKH crashes, and `rtf` measured live at 0.69–0.82. It still ended against a
floor-to-ceiling pillar at x ≈ −14.5, y ≈ 13.0–14.1. It brushed that pillar at
t=87.8 s doing 1.97 m/s, flew on for another ninety seconds, came back to the
identical spot at t=174, sat pinned against it for about thirteen seconds with
the reference 9 m away and 10–21° of tilt commanded into it, and flipped.

That is two separate problems and only one of them is about tracking:

1. **It flies into the thing.** Still the tracking error — 2.37 m mean, and the
   approach was at 1.97 m/s.
2. **Once in contact it has no way out.** Nothing in the chain notices that a
   commanded acceleration is producing no motion, so the aircraft grinds
   against the obstacle until the attitude diverges. `STALL_WINDOW_S` is 25 s,
   longer than it took to flip. A contact reflex — stop pushing into what is
   not moving, back off, let FALCON replan — is the obvious missing piece, and
   it is the thing to build next after `use_sim_time`.

**The stub cannot show either of these, and that is worth remembering: it has
no collisions at all.** Its aircraft flies through walls, so a command that
would pin the real one against a pillar costs nothing there. Every stub number
in this file is "how well does the controller track", never "what happens when
it does not".

Logs kept, videos stripped: `~/falcon_pegasus_recordings/soak_evidence/`.

### What went in before round 7

1. **The greedy TSP fallback aborted the node** on a fatal glog `CHECK_NEAR`,
   because it wrote `trunc(100·Σcᵢ)` where upstream's convention is
   `Σ trunc(100·cᵢ)`. The one path meant to survive an LKH crash was killing
   the planner instead. Validated: a stub flight that died at trajectory #180
   after 94 m now runs the full budget — 638 trajectories, 242 m, **1396 m³**,
   node deaths 0. Biggest single win so far.
2. **`soak.sh` could not have scored a clean streak**, because it counted a
   *recovered* LKH crash's stack trace as a planner crash. Now counts
   `process has died`.
3. **Both patch scripts are re-runnable.** They had to be: the image already
   carried the patch, so correcting it otherwise meant an hour-long rebuild
   from the base image. Guard the header too — re-declaring the helpers is a
   compile error, not a no-op.
4. **`SimClock`** re-bases each trajectory onto the aircraft's clock. Worth
   about a fifth of the lag; the rest needs `use_sim_time`.
5. **The stub can now see this class of bug at all** — it had no drag and no
   way to run slow, so it tracked to 0.26 m while Isaac managed 2.18 m on the
   same code. `--real-time-factor`, `--trace`, and a measured drag model.

### What an adversarial audit of the session's own diff found

Worth doing again — it caught two things no amount of flying would have, and a
flight costs half an hour.

* **`ThrustModel.normalized()` mapped a non-finite request to MAX throttle.**
  The same `max(lo, min(hi, nan))` trap that had just been documented at length
  in `observe()`, sitting unguarded forty lines above on the *command* path. One
  NaN velocity component → throttle 0.9, a NaN attitude quaternion, and the
  status line printing a reassuring `tilt= 0.0deg` because `acos(clamp(nan))`
  is 0. Now returns the hover throttle.
* **`SimClock` over-read the real-time factor by ~20%.** It averaged per-tick
  ratios, but Isaac's ticks are twenty cheap physics steps to one expensive
  render, so the cheap ticks dominated an unweighted mean. It reported 0.75 on a
  flight whose true rate was 0.61, with excursions above 1.0 on a simulator that
  never once reached real time. Now accumulates simulated and wall time over a
  2 s window and divides once.
* `postmortem.py` compared an absolute trace clock against a pose-relative
  contact time (~50 s apart) and silently printed nothing; its contact detector
  reported **take-off** as the first contact and missed strikes that arrest the
  aircraft rather than turning it. Both fixed — the arrest detector immediately
  found a 2.65 → 0.28 m/s stop the old one had mischaracterised.
* Three tests that could not fail (a stalled-clock test whose path was skipped
  entirely, and a braking test passing on the damping term with the governor
  deleted). The governor's taper and active-braking branches had **no coverage
  at all**; they are now tested directly, because they are unreachable through a
  straight-line flight fixture and a closed-loop test cannot exercise them.

### Known, deliberately not fixed mid-soak

`AirframeController.deliverable_limits()` cuts the thrust ceiling to what the
throttle can actually buy, but only the **flatness** stage sees that cut — the
tracker still clamps against the base limits and sets `_saturated` from them.
So the one flag that freezes the integrator, and the one `AirframeCommand`
re-exposes, both read False while the command is being trimmed. With the 0.62
seed the two ceilings are 15.69 and 14.24 m/s², so this is live from tick one:
a `(6.0, 0, 3.5)` request comes back as `(5.06, 0, 3.5)` with `saturated=False`,
and the horizontal integrators keep charging against a correction that never
reaches the airframe.

Not fixed here only because it is flight-path code and a soak was in the air.
The fix is to make both stages clamp against the *same* limits — pass the
deliverable envelope into `tracker.update()` for the tick rather than letting
the two stages disagree. Low severity today (the nastier branch, where a climb
zeroes all horizontal correction, needs a learned hover above ~0.75 and this
airframe sits at 0.58–0.60), but it is wrong as written.

### Where it actually stands after 13 rounds

**0/10, and the per-flight failure rate is still close to 100%.** A streak of ten
needs it under about 1%. That is two orders of magnitude, and it will not be
closed by another round of the same loop: rounds 9, 10, 11 and 13 failed four
different ways (planner cycling in an alcove, a physical wedge in shelving, an
altitude loss on a replan-heavy route, and repeated contacts).

What *has* been retired is the class of failures that repeated — the LKH abort,
the fictitious-lag overspeed that put the aircraft into two walls, the
NaN-to-full-throttle path, and a harness that scored stale results, mislabelled
recovered crashes and enforced an impossible bar.

**The remaining gap is one number: cross-track.** Isaac sits at 0.79 m mean,
1.52 p90, 4.20 max; the stub on the same code manages 0.11-0.37 m. In three-metre
corridors that difference is the whole story, and every recent flight ends in
contact rather than in a wedge. The unwedge recovery added in round 13 never
even fired — the aircraft hit something first.

**The dominant known cause is still the clock**, and `SimClock` only recovers
about a fifth of it. FALCON plans from its own previous curve at a wall-clock
horizon the aircraft cannot reach at 0.63x real time, so every replan starts a
little ahead and the deviation regenerates four times a second. That is what
`use_sim_time` fixes at the source, and it is why it is the next item below
rather than more gain tuning. Do not spend more flights on the outer loop until
it is done; the loop is not what is wrong.

### DONE: FALCON now runs on the simulator's clock

This was the top item here all day, and it is the largest single improvement of
the session. Measured on the stub at Isaac's real 0.62x rate, changing nothing
else:

| | mean tracking error | max |
|---|---|---|
| real time (no mismatch to fix) | 0.26 m | - |
| 0.62x, FALCON on the wall clock | **2.08-2.61 m** | 7.0-8.0 m |
| 0.62x, FALCON on `/clock` | **0.36 m** | 1.98 m |

A 6-7x reduction, and it essentially recovers real-time tracking on a simulator
running at two thirds speed. Coverage 981 m3 in 180 s, no node deaths.

Four pieces, all of which have to be present:

1. `patches/allow_sim_time.sh` -- upstream opens with
   `CHECK(!use_sim_time)`, a glog **fatal**, so an unpatched image does not
   degrade, it aborts the node holding the mapper, the frontier finder and the
   FSM. The check guards THEIR wall-clock simulator, not the planner.
2. `pegasus_bridge_node.SimClockPublisher` -- Isaac has no ROS, so the bridge is
   the only process that both sees the aircraft's timestamps and lives inside
   ROS. Publishes `/clock` at 100 Hz, monotonic, held between updates.
3. The aircraft stamps odometry and depth frames with `loop.sim_time` instead of
   `time.time()`. Those stamps ARE the clock.
4. `<param name="/use_sim_time" value="true"/>` in `falcon_pegasus.launch`,
   behind an arg so the old behaviour is one flag away for comparison.

**The bootstrap is the part that bites.** With `/use_sim_time` set, every node
reads time 0 until the first `/clock` arrives, and that cannot arrive until the
aircraft connects -- which the bridge is itself waiting for. `_await_hello` and
the bridge's 10 s report cadence were moved to `time.monotonic()` for exactly
this reason. Anything in that file calling `rospy.Time.now()` before the
aircraft is up will read zero and behave strangely.

`link/sim_clock.py` is no longer used to re-base trajectories -- with the clocks
agreeing there is nothing to convert -- but it is still fed each tick, because
`real_time_factor` is worth having on the status line and in the trace.

### GLASS -- the operator's insight, and probably the biggest remaining one

The depth the aircraft sends comes from Isaac's `distance_to_image_plane`, a
RENDERED annotator: it sees straight through glass to whatever is behind it.
PhysX does not -- the collider is solid. So FALCON is told a glass door is free
space, routes through it, and the aircraft is stopped by a barrier its map says
is not there. The camera can never see the glass, so the map never learns, and
every replan sends it back. That is the "returns to the same wall forever"
behaviour that dominated this campaign, and it explains why post-mortems kept
reporting contacts at points FALCON's map called clear while the SURVEY showed
occupancy: the survey was built by raycasting COLLIDERS, so it contains glass.

Fixed by fusing rather than editing the scene: each frame now sends
`min(rendered, raycast of the surveyed colliders)`. The survey is the right
corrective because it describes what the aircraft can HIT; the minimum keeps
anything the renderer sees that the survey missed. Quarter ray grid, behind
`--no-collider-fusion` for comparison. See `sensing.fuse_with_colliders`.

Pre-loading occupied voxels was the other option and was rejected: FALCON builds
its own map from depth and has no ingestion path for a prior (`map_file` feeds
only upstream's mesh renderer, which is not running here). Correcting the depth
reaches the mapper through the channel it already trusts.

First flight with it crashed at (-16.7, 7.4) -- NOT the x = -1.15 wall that had
ended nearly every previous flight. Encouraging, but one sample.

### A REAL BUG IN FALCON, and the fix

Found from the user's own observation of the recordings -- "the drone cannot
keep up with the point FALCON assumes it is at". It is not a tracking failure.

`exploration_fsm.cpp`, the non-static replan branch, picks the start state for
every new trajectory off the PREVIOUS TRAJECTORY:

    double t_r = (time_now - info->start_time_).toSec() + fp_->replan_duration_;
    fd_->start_pos_ = info->position_traj_.evaluateDeBoorT(t_r);
    fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
    fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);

`fd_->odom_pos_` -- the measured position, which the FSM maintains and uses in
the hover branch immediately above -- appears NOWHERE in this branch.

Upstream this is an identity rather than a bug: `poscmd_2_odom` feeds the
position command back as the vehicle state, so the trajectory point IS the
drone. With a real airframe they diverge and the failure is structural:

* the new curve starts at a point the aircraft is not at;
* the aircraft must chase it, so it begins behind;
* the error is INVISIBLE to the planner, because nothing in the replan reads
  odometry, so nothing can correct it;
* the next replan inherits the error from the new plan's own prediction, so it
  compounds instead of decaying. Measured at up to 9.5 m, sustained.

`patches/replan_from_measured_state.sh` plans from the measured state once the
predicted start is further than `~replan_start_tolerance` (1.0 m, set in the
launch) from the aircraft. CONDITIONAL on purpose: always using odometry
discards why the prediction exists -- a curve starting at the aircraft's current
position ignores the planning time, so consecutive curves stop joining smoothly.
Predict while the prediction is true; measure once it demonstrably is not. It
fires ~twice per 180 s, which is the right rate for an exception path. Defaults
to 1e9 (inert), so an unpatched image behaves as upstream.

**Effect: stub mean tracking error 0.36 -> 0.18 m** at Isaac's real 0.62x rate.
Cumulative with the clock fix: 2.08-2.61 -> 0.18 m, about 13x.

### LATEST STATE (end of session)

Best clock-fixed flights: **876 and 877 m3** against the 1333 bar, both ending
crashed at the wall around x = -1.0 to -1.2, y = 15-20. Still **0/10**.

Landed since the clock fix, all evidence-backed:

* **Near depth returns clamp instead of zeroing** (`link/depth_codec.py`). THE
  root cause of the endless return trips: pinned against a wall the whole frame
  reads under NEAR_M (measured min 0.08 / median 0.09 / max 0.13 m), every pixel
  went out as 0 = "no measurement", and FALCON never mapped the wall it was
  touching. It then re-planned the same unreachable viewpoint forever.
* **`_unwedge` fallback** when the breadcrumb trail has aged out -- it used to
  print "no flown path to retreat along" and hold, which is a no-op exactly when
  it is needed. Now backs out along -body-x.
* **`_pinned` detector** -- sustained commanded tilt with no motion. `_touched`
  structurally cannot see a grind: both its signatures need speed, and the
  107 s pin never reached 0.6 m/s. Has not yet fired in anger.

**Null result, recorded so it is not retried:** raising `CONTACT_REFLEXES` 6->25
changed nothing (876 -> 877 m3; only 8 were used). The cap was not the limit.

**What is still killing flights:** a contact hard enough to flip the aircraft
before the reflex can act. The reflex recovers glancing strikes well -- six in a
row on one flight -- but it is inherently reactive. The remaining idea is to
stop arriving at the wall that fast, or to refuse FALCON's route back into a
neighbourhood already struck (`blockage_memory`).

### START HERE next session: the wall at x = -1.15

Control is no longer the problem. After `use_sim_time`, early-flight cross-track
is **0.12 m**, along-track lag is *negative* (slightly ahead), and time over the
speed ceiling fell from 54% to 13%. What ends flights now is one place:

**Every recent flight dies pinned against the north-south wall at x ~ -1.15,
y ~ 13-16** -- the same wall round 6 hit. FALCON routes there, the aircraft gets
stuck, the contact reflex and the unwedge retreat both fire, and it goes
straight back. Three round trips in one flight.

Two things tried and measured, so do not repeat them:

* **A longer retreat (2.5 -> 4.0 m).** Worse: 385 m3 against 567 on one flight
  each. The extra seconds flying backwards are seconds not exploring, and it
  did not stop the return trip. Reverted.
* **Moving the box edge off the wall.** Worse still -- see above; it let the
  aircraft into wall shelving.

**The idea not yet tried, and the one worth trying:** after an unwedge, refuse
FALCON's route back to the same neighbourhood for some seconds -- hold, or
follow only trajectories whose first few metres lead away from the spot just
escaped. The aircraft currently has no memory of where it just got stuck, which
is why the retreat is undone immediately. `core/planning/environment/`'s
`blockage_memory` is the existing vocabulary for this idea on the XTEND side and
is probably the right thing to reuse.

### The next real piece of work
