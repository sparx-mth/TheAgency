# falcon_pegasus — the whole FALCON stack, flown on a simulator with physics

Runs [FALCON](https://github.com/HKUST-Aerial-Robotics/FALCON) end to end —
**its own** voxel mapping, frontier finding, coverage tour, kinodynamic search,
B-spline trajectory and yaw plan — against the Isaac Sim + Pegasus + PX4 SITL
aircraft in `robots/PEGASUS/`. FALCON decides everything about where to go. The
only things this package adds are the two halves its own simulator never had to
provide: a real sensor, and a real airframe.

That second one is the point. Upstream, FALCON's `poscmd_2_odom` feeds the
position command straight back as the aircraft's state, so the aircraft is
*defined* to be exactly where the planner asked — no lag, no drift, and tracking
error identically zero. Put the same planner on an airframe with mass and an
autopilot and the two come apart, and nothing in FALCON notices, because FALCON
is simply told where the aircraft is and replans from there — and in flight not
even that, because it starts each new curve from its *own previous one* rather
than from the measured position, so tracking error is invisible to it and never
corrected by replanning. Closing that gap is what `core/control/` exists for.

```
FALCON container (ROS1 Noetic)                Isaac Sim container (no ROS)
──────────────────────────────                ────────────────────────────────
exploration_node   ── unmodified              Isaac Sim + Pegasus Iris + PhysX
traj_server        ── unmodified              PX4 SITL (lockstep)
                                              AirframeController
pegasus_bridge_node        ◄── TCP 5599 ──    depth frame + camera pose
  /uav_simulator/depth_image                  ground-truth odometry
  /uav_simulator/sensor_pose
  /uav_simulator/odometry
  /planning/bspline        ─── TCP 5600 ──►   the trajectory itself
  /planning/pos_cmd        ─── TCP 5600 ──►   ... and 100 Hz samples of it
map_recorder_node → map.mp4                   FlightRecorder → flight.mp4
```

`sim_flight_recording/` is the sibling package that flies **planned A-to-B
routes** to collect training data. This one hands the whole navigation problem to
FALCON instead. They share the platform, the bring-up sequence and the PX4
parameter set.

## Quick start

```bash
# once
cd sparx_agency/tasks/planning/falcon_pegasus && docker build -t falcon-pegasus:noetic .

# the fast loop -- no Isaac Sim, no GPU, about four minutes
./stub/check.sh 3_open_plan

# one real flight: FALCON side first, aircraft second
./run_falcon_pegasus.sh 3_open_plan          # terminal 1
./run_isaac_side.sh 3_open_plan --video      # terminal 2

# all six runs, unattended
./run_campaign.sh

# is it RELIABLE? fly one run ten times and count the clean streak
./soak.sh 6_whole_office 10
```

`run_campaign.sh` flies the six configurations once each — that is the
deliverable. `soak.sh` asks the other question, because almost every failure on
this stack has been intermittent: LKH segfaulted on some coverage-tour instances
and not others, the hierarchical grid crashed only once the aircraft left the
exploration box, the aircraft wedges in a doorway on one seed and sails past it
on the next. A single green run proves very little, and a streak is the only way
to tell a fix from a lucky seed. It counts a flight clean only if the outcome is
good, `exploration_node` did not crash (checked in FALCON's log, because the
aircraft **cannot** see it) and coverage cleared the bar — then stops at the
first dirty flight, so the logs are there to read instead of buried under
another hour of GPU.

Order matters: the FALCON side **binds** the two localhost sockets and the
aircraft **connects** to them, because the ROS stack is up in seconds and Kit
takes minutes to load a stage. Both containers use `--network host`, which is
what makes `127.0.0.1` the same loopback device in each.

## The six runs

One `runs/*.yaml` describes a whole run: FALCON's `map_config` **and** the
aircraft's spawn pose, in one file, so the two cannot disagree — and they must
not, because `traj_server` parks its pre-trajectory command at
`(init_x, init_y, init_z)` and an init pose that is not where the aircraft
actually is makes the very first reference a step-jump.

**All six share one exploration box — the whole building** — and differ in where
the aircraft starts and how long it flies. Six starting points across a
28 × 72 m compound produce six different orders of discovery.

| run | starts | budget | what it exercises |
|---|---|---|---|
| `1_north_hall` | the open hall at the north end | 8 min | long sight lines; fast wide sweeps, then south |
| `2_room_warren` | inside the dense block of small rooms | 10 min | doorways; the yaw plan matters as much as the route |
| `3_open_plan` | the central floor, desks and partitions | 10 min | floor plan vs. what can actually be flown |
| `4_south_wing` | deep in the southern office wing | 10 min | a wing fed by one ~3 m corridor |
| `5_east_spine` | the east side, facing along the length | 12 min | a corridor march; fast, nearly straight references |
| `6_whole_office` | the centre, with the longest budget | 20 min | the whole compound appearing |

They used to carve the building into six sub-regions instead, and that was
wrong — see "an exploration box edge in open space is a trap" below.

## What comes out

Per run, under `~/falcon_pegasus_recordings/<run>/`:

- **`<run>_map.mp4`** — the map being built. Unknown space is the background, so
  the building literally appears out of the dark as the aircraft flies. Drawn
  are the occupancy slab at cruise height, the flight trail, the sight wedge, and
  a red line from the aircraft to the point FALCON is currently commanding —
  which is the tracking error, and which would be zero length in a geometry-only
  simulator.
- **`<run>_flight.mp4`** — the aircraft itself, from a chase camera.
- **`recording/`** — RGB, depth and full 6-DoF pose in the repo's flight-recording
  schema, so a FALCON exploration is also a training-data source.
- **`result.json`** — outcome, distance flown, trajectory count, and the mean and
  worst plan-to-aircraft gap.
- **`falcon.log`, `isaac.log`, `px4.log`.**

Three outcomes are successes and mean different things:

- **`explored`** — FALCON's frontier set emptied. It decided the space was
  covered.
- **`flight_timeout`** — it was still working when the budget ran out. On a
  2400 m³ compound this is the normal ending: a whole building is more than one
  flight's worth of exploring.
- **`planner_stopped`** — FALCON stopped producing new trajectories. Everything
  flown up to that point is real exploration; see the next section for why this
  happens and why it has to be detected rather than waited out.

The failures are `stalled` (the aircraft stopped moving while FALCON kept
replanning — it is wedged against something), `diverged` (more than 3 m behind
the plan for 30 s), `offboard_lost`, `crashed`, `arm_failed` and `no_commands`.
`result.json` says which, with the distance flown, the trajectory count and the
mean and worst plan-to-aircraft gap.

## Watching it live in FALCON's own RViz

```bash
./run_falcon_pegasus.sh 3_open_plan --rviz     # terminal 1
./run_isaac_side.sh 3_open_plan --video        # terminal 2
```

That opens `exploration_manager/config/rviz.rviz` unmodified, on the host's X
display, showing what FALCON is actually thinking: the occupancy voxels and the
unknown space, the ESDF and its slices, the frontier clusters (live, dormant and
too-small), the sampled viewpoints, the hierarchical grid, the executed
trajectory, and the sensor FOV cone at the commanded pose. Several of its
displays point at rigs that are not running here (VINS, AirSim, a motion-capture
pose) and simply stay empty.

`roslaunch exploration_manager rviz.launch` on its own is the right file and
will not work by itself. Three things have to be arranged around it, and
`--rviz` is what arranges them:

- **An X display inside the container.** The stack is otherwise fully headless.
  `--rviz` mounts `/tmp/.X11-unix`, passes `DISPLAY`, and runs `xhost
  +local:docker`. It also points GL at the NVIDIA card; without that the
  container's Mesa tries the laptop's Intel iGPU, cannot find the i915 driver,
  and falls back to software rendering at about 11 fps.
- **A TF tree.** FALCON uses none — it takes the camera pose off a topic and
  never asks a listener for anything — but RViz's Fixed Frame is `world` and
  must exist in TF or every display reports "Fixed Frame [world] does not exist"
  against a stack that is working perfectly. `--rviz` turns on the bridge's
  `~publish_tf`, which broadcasts `world -> base_link` from the odometry and
  `world -> camera` from each depth frame, so a view can also be attached to the
  aircraft and follow it.
- **A drone.** FALCON's RViz config already has the displays for
  `/odom_visualization/{robot,pose,path}` — enabled, in its `Simulation` group —
  and nothing was publishing them, which is exactly why the map appeared and the
  aircraft did not. `--rviz` starts FALCON's own `odom_visualization` fed from
  our odometry, so the drone mesh shows up at its real size and its footprint
  against the voxels is answerable by looking. To make the view *follow* it, set
  **Target Frame: `base_link`** on the Views panel's ThirdPersonFollower (the
  saved views use `<Fixed Frame>`, which does not move); the bridge's TF makes
  that frame exist.
- **A visualisation box worth drawing.** `vbox` is normally a 20 cm slab at
  cruise height, because that is all the map recorder needs and every extra
  layer is another full pass over the voxel grid at 2 Hz inside the same thread
  as the mapper. Drawn as-is the voxel view is a single contour line. `--rviz`
  widens it to the flight band; `viz_min_z` / `viz_max_z` set the range.

It is off by default because it is not free: the occupancy clouds are only
computed when something subscribes, and the wider box costs `exploration_node`
real time on the thread that also services the depth callbacks.

### The post-mortem, before you read a single log

```bash
.venv/bin/python sparx_agency/tasks/planning/falcon_pegasus/postmortem.py \
    ~/data/sim/falcon_pegasus/soak/1_20260808_133925
```

**Run this first.** The `outcome` field is a symptom, not a diagnosis: of the
seven soak rounds analysed so far, five were something other than what they
reported, and each cost an hour of log-reading to establish. The tool answers
the four questions that separate those cases, from the recording alone:

* **Did it touch something?** A horizontal velocity that reverses through more
  than 120° inside 250 ms is not a control response — nothing on this aircraft
  can turn 1.4 m/s around that fast — it is a contact. This matters more than
  anything else in the output, because a contact makes every number after it
  (tilt, tracking error, "stalled") a *consequence*.
* **What is at that point**, checked against the surveyed voxel map rather than
  FALCON's, so the answer does not depend on the mapper being right.
* **Was it upset or was it parked?** Judged on *net displacement*, not speed: an
  aircraft lying on its side against a wall still shows 0.18 m/s of scraping
  while moving 24 cm in ten seconds.
* **Was it looking where it flew?** Reported for context and almost never a
  fault — see round 6 below for why that number is a trap.

On round 7 it found, unprompted, that the aircraft had hit the *same pillar*
twice ninety seconds apart.

### What to look at when the aircraft hits something

The crashes are the aircraft arriving somewhere its plan did not go, so the
question is always *which* of the two was wrong. RViz separates them:

- **Was the obstacle in the map?** Look at the occupancy voxels around the crash
  point. If the wall is there, FALCON knew about it and the trajectory was
  routed around it — the aircraft failed to follow the route, which is a
  tracking problem (speed, lag, `bspline_opt/safe_distance`).
- **Was the trajectory through it?** The executed trajectory (`Trajectory` group,
  red) is what the aircraft was *asked* to fly. If it passes through mapped
  voxels, the planner is at fault. If it goes around them and the aircraft did
  not, the tracker is.
- **Was the map late?** Watch the voxels appear as the aircraft advances. A wall
  that only appears after the aircraft is next to it means the mapper was
  starved — usually by the planner, which shares its thread (see the A* budgets
  in the launch file).
- **Where does it want to go?** The viewpoints and frontier clusters show the
  target. A viewpoint on the far side of a wall, or one it keeps re-picking, is
  the signature of a planning dead end rather than a control one.

The map video records a top-down version of the same thing with a line drawn
between the plan and the aircraft, so a run that already happened can be
reviewed without re-flying it.

## Where the aircraft cuts into PX4

There are two control paths, selected with `--control`, and the difference
between them is **how many of PX4's loops are still in the chain**.

```
                       │ velocity cut (--control velocity)  │ attitude cut (default)
───────────────────────┼────────────────────────────────────┼────────────────────────
what arrives from      │ /planning/pos_cmd, 100 Hz samples  │ /planning/bspline, the
FALCON                 │                                    │ curve, on each replan
what we run            │ ReferenceTracker3D                 │ AirframeController
what PX4 is sent       │ world velocity + heading           │ attitude + throttle
PX4 still runs         │ velocity → attitude → rate → mixer │ attitude → rate → mixer
```

The attitude cut exists to delete one loop: PX4's velocity controller runs at
tens of Hz off the same position estimate our outer loop already has, so it adds
a stage of lag without adding any information — and that lag is the metre of
tracking error the campaign kept measuring at corners.

Taking it out costs three things, and all three live in `core/control/`:

- the **acceleration** has to be produced here rather than asked for, which is
  `trajectory_tracking`;
- the acceleration has to become a **tilt**, which is `flatness` — and needs the
  trajectory's *jerk*, which is not in the sampled command and is why the spline
  is carried;
- the thrust has to become a **throttle**, which needs the one number nobody can
  assume: how much acceleration a unit of throttle buys. `thrust_model` measures
  it in flight, because it moves with battery voltage.

Both paths honour the condemned-trajectory hold identically, and both are flown
from the same FALCON-side run — the bridge forwards the curve *and* the samples,
which costs a couple of kilobytes a second and means a comparison never needs a
rebuild.

### What the stub run does and does not tell you

```
./stub/check.sh 3_open_plan 150 --control attitude --hover-throttle 0.72
./stub/check.sh 3_open_plan 150 --control velocity
```

| stub, `3_open_plan`, 150 s | mean gap | worst gap | trajectories |
|---|---|---|---|
| attitude cut, three runs | 0.25 – 0.33 m | 0.77 – 2.33 m | 227 – 264 |
| velocity cut, one run | 0.12 m | 0.34 m | 317 |

**Neither column is a verdict, for two separate reasons.**

*The runs are not repeats.* FALCON's exploration is not deterministic — a
slightly different frontier order gives a different coverage tour and different
trajectories — so the three attitude rows above are three different flights, not
three measurements of one. The worst gap in particular swings by a factor of
three between them. Any comparison needs several runs per arm, and the six-run
campaign is the smallest honest sample.

*The two rows fly different stand-in airframes*, because they have to: a
velocity-commanded body reaches any velocity it is given, cannot fail to hold
altitude and has no thrust curve to learn, while the attitude stand-in has to
build a tilt before it can accelerate at all and is not told its own thrust
curve. The second is strictly harder, and it is the more honest model — the
first is the same "the aircraft is defined to be where you asked" flattery that
made upstream FALCON look perfect, moved down one layer.

What the stub *does* establish is that the whole attitude path works end to end:
the spline crosses the link, is rebuilt into the identical polynomial, is
projected onto, becomes a tilt and a throttle, and flies a real 150 s FALCON
exploration — 260-odd trajectories and 135 m, every run. And that the thrust
estimator acquires a wrong seed in flight: seeded at 0.62 against an airframe
hovering at 0.72, it converged to 0.7200 and held there.

**The two cuts can only be compared on Isaac Sim**, over a full campaign, where
both face the same PhysX airframe and the same PX4. Until that has been run, the
attitude cut is *validated* but not *demonstrated better*, and the velocity cut
is still there precisely so the comparison can be made.

### Where reliability actually stands

`soak.sh 6_whole_office 10` is the standing measurement, and the honest answer
today is **the crashes are fixed and the aircraft is not**. Successive rounds:

| round | patches in the image | how it ended |
|---|---|---|
| 1 | grid index + LKH isolation + A* inflation | `stalled`, **1 crash** (the mapper's raycast CHECK), 595 m³ |
| 2 | + raycast guard | `stalled`, **0 crashes**, 382 m³ |

So the failure moved from the planner dying to the aircraft stopping — which is
progress, and is also the harder half. In round 2 the aircraft halted at
(−21.8, −5.2, 1.45) and never moved again. Three measurements settle what
happened, and the first two are worth keeping as method:

- **The controller was asking.** Reconstructing the exact reported state offline
  — `err=1.03 lag=0.00 xte=1.03`, at rest — the chain commands **11.4° of tilt
  and 1.98 m/s² forward**. So the outer loop is not at fault, and that took
  seconds to establish rather than a 25-minute flight.
- **The airframe was held.** In the recording the aircraft *is* tilting (mean
  4.2°, peaks 14.4°) while x, y and z stay frozen to the centimetre for 25 s. A
  free body at 11° accelerates at ~2 m/s²; frozen position under active tilt
  means PhysX is holding it.
- **It was inside an obstacle.** At its exact position the surveyed map is
  occupied at z = 1.45, with clear air at 1.2 and 1.6 — it is embedded in the
  cruise-height clutter this scene is deliberately augmented with.

**A correction worth recording**, because it sent the first pass down a blind
alley: an earlier version of this section claimed the aircraft was in free
space. That came from indexing the voxel array as `(x, y, z)` when it is
`(nz, ny, nx)` — `voxel_camera.py` says so explicitly — so the query read
entirely the wrong cells. Always index those grids `v[k, j, i]`.

### Why it flew into the clutter

It was doing **1.6 m/s into a 0.6 m/s plan**, and across the flight 42% of the
time was above 1.1 m/s, 29% above 1.5, peaking at 2.85.

That is not a tuning detail. FALCON checks its trajectory against the map at the
speed *it* planned, with `bspline_opt/safe_distance` of clearance around it. Fly
the same curve three times faster and the margin goes on stopping distance — and
the airframe is 0.7 m across.

The position loop has no natural ceiling: a metre of error asks for
`kp * clamp` of acceleration, which the damping term balances only once the
aircraft is `kp * clamp / kd` — about 0.9 m/s — **faster than the plan**. So the
tracker now has a speed governor, ceilinged at *planned speed + `max_overspeed`*
so it moves with the plan and can never hold the aircraft behind a faster
trajectory. It tapers in below the ceiling rather than switching on at it,
because an airframe's acceleration decays over a time constant and a limiter
that waits for the limit has already lost.

### The soak rounds, and what each one actually was

Every round so far has stopped at attempt 1, and no two for the same reason.
The table is worth keeping because three of the five were **not** what the
outcome field said:

| round | reported | what it really was |
|---|---|---|
| 1 | `stalled` | the mapper's raycast `CHECK` aborting — patched |
| 2 | `stalled` | flew into cruise-height clutter at 1.6 m/s — speed governor added |
| 4 | `stalled` | **Isaac Sim ran out of VRAM 36 s into start-up**; the aircraft never existed |
| 5 | `crashed` | outer loop limit-cycling into the 35° tilt cap, then over |
| 6 | `crashed` | flew into a wall while 2.18 m off plan — see below |

**Round 4 is the cautionary one.** `isaac-sim` had been up 34 hours across many
Kit sessions and was holding 6134 MiB of an 8 GB card with nothing running; Kit
died on `ERROR_OUT_OF_DEVICE_MEMORY` before the drone spawned. A `docker restart`
returned it to 96 MiB. Worse, the attempt produced no output, so `docker cp`
copied the **previous** attempt's `result.json` and the harness scored a flight
that never happened — the verdict came back byte-identical to round 2, down to
the distance flown. A stale *clean* result would have counted toward the streak.
`soak.sh` now deletes the container's output directory before each attempt,
restarts Kit between attempts and prints free VRAM, and reports `isaac_gpu_oom`
as its own outcome rather than letting it masquerade as the aircraft stalling.

### Two control defects the crash exposed

Round 5 flew properly and then diverged in attitude — tilt oscillating
1.8° → 20 → 5 → 28 → 38 → 1 → 35 → 54 → 89 while barely translating, at about
**0.5 Hz**. That is far too slow for PX4's attitude loop; it is the outer loop
limit-cycling into the 35° tilt ceiling. The map is clear at that point, so it
was not a collision.

**The position-error clamp was rotating the correction.** It clamped each axis
independently, so an error of (5.0, 1.0) m — the magnitude in this crash, whose
worst tracking error was 5.25 m — clamped to (1.0, 1.0) and pointed **33.7°
away from the reference**, with 45° as the worst case. The docstring promised
the exact opposite ("keeps the correction pointed the right way and bounds only
how hard it pulls"), and `limit_acceleration` ten lines away already scales the
horizontal pair together for precisely this reason. Now the horizontal pair is
scaled together and only the vertical axis is clipped alone.

**`reset(hold_position=...)` did not hold.** It cleared the integrators and the
projector but left the loaded trajectory in place, so the next tick found a
usable curve and carried on flying it. The one call a caller has to say "stop
flying the plan" silently did nothing. It now drops the current and queued
curves.

The speed governor from round 2 did work: exploring speed fell from mean 0.97 /
p90 1.76 to mean 0.73 / p90 1.25 against a 0.6 m/s plan, and altitude held at
1.11–1.95 m around a 1.4 m cruise.

### Round 6: the simulator's clock, and a fallback that aborted the node

Round 6 was the best flight so far and still ended in a wall. It flew **129 m**
and mapped 872 m³ — five times the distance of any earlier round — then at
t=139.7 s reversed from +0.94 to −1.83 m/s in 0.2 s. Nothing in a control law
does that; it is a contact. The surveyed map confirms it: a floor-to-ceiling
wall 20 cm thick at x ≈ −1.1, and by t=144.6 the aircraft's own centre reads
occupied. The attitude divergence that the `crashed` verdict fired on was the
*consequence* of bouncing off it, not the cause.

**A statistic that looks damning and is not.** 45.7% of the moving flight —
36 of 89.7 m — was flown with the direction of travel outside the camera's 90°
FOV, median 37.8° off axis and p75 at 99.6°, i.e. flying backwards relative to
where it looked. That is *not* a fault. FALCON's yaw planner aims at frontiers,
not along travel, because it plans against its accumulated map rather than the
current image; the reconstruction matches `traj_server` exactly
(`NonUniformBspline(yaw_pts, 3, yaw_dt)`, same `t_cur`, same clamp), and a stub
flight shows the same 54% while covering 209 m without incident. Do not spend a
day on this number again.

**What was actually wrong is the clock.** Mean tracking error was 2.18 m against
the stub's 0.26 m on identical control code. Isaac Sim runs at about **0.66×
real time** here (665 s of wall clock for ~418 s of simulation), while FALCON
plans in its container on the wall clock and stamps every trajectory with
`ros::Time::now()`. The schedule therefore advances about 1.5 s for every 1 s of
flight the aircraft is given, and the tracker chases a deadline that recedes as
fast as it closes. Reproduced on the stub by slowing nothing but the aircraft:

| stub configuration | mean err | mean lag | mean xte |
|---|---|---|---|
| real time | 0.26 m | −0.08 m | 0.11 m |
| 0.66× real time | 0.92 m | **+0.69 m** | 0.31 m |
| 0.66×, schedule re-based | 0.84 m | +0.53 m | 0.33 m |

The lag flips from *ahead of the plan* to two thirds of a metre behind it.
`link/sim_clock.py` re-bases each trajectory's start time onto the clock the
aircraft actually experiences, which recovers about a fifth of the lag. The rest
is structural: FALCON plans from a predicted future state on its own clock that
a slow aircraft never reaches, and only ROS `use_sim_time` with Isaac publishing
`/clock` fixes that at the source. That is the next real piece of work here.

Two rig gaps were closed on the way, both of which had been hiding this. The
stand-in airframe had **no drag**, so it tracked to centimetres no matter what;
it now carries the drag fitted from a real flight (`0.176·v + 0.121` m/s², i.e.
0.30 m/s² at 1 m/s, measured as the residual between specific force and thrust
axis over 501 samples of steady cruise). And `run_stub.py` gained
`--real-time-factor` and `--trace`, which is what made the clock visible at all.

### The coverage bar was 150% of what can ever be covered

Worth knowing before reading any of the reliability results below, because it
invalidates the pass/fail on all of them. `MIN_COVERAGE_M3` was 2200, derived as
~91% of the exploration box's 2424 m³. But a box volume is not a coverage
target. FALCON's `Coverage` (`MapServer::publishMapCoverage`) counts voxels that
are no longer `UNKNOWN`, and a voxel leaves `UNKNOWN` only when a camera ray
reaches it — so the interior of every wall, and every cubic metre of outdoor
space the box hangs over, is permanently uncountable.

Flood-filling the surveyed voxel map from the spawn through free space, and
adding the occupied shell that free space touches, gives what a camera inside
this building can ever see:

| | box volume | observable | 91% bar |
|---|---|---|---|
| before | 2424 m³ | **1465 m³ (60%)** | — |
| now (south edge fixed) | 2222 m³ | **1465 m³ (66%)** | **1333 m³** |

So the old bar was **150% of the maximum achievable** and no flight could ever
have met it. The best run on record, 1396 m³, was **95% of achievable** — an
essentially complete exploration of the office, scored as a failure. The soak
counter reading 0/10 was measuring the bar, not the aircraft.

Only the **south** edge moved, from y = −33.2 to −27.2. The building stops at
−27.2, so six metres of outdoor nothing had been sitting inside the coverage
denominator: ~200 m³ that can never be observed and never stops being `UNKNOWN`.
Removing it opens no new floor to fly into, so it carries no risk.

**Moving the other three edges onto the walls was tried, and reverted.** The
reasoning looked sound — an inset edge is an open-space frontier cut, exactly
what the run config warns against, and one flight had spent 527 of its 878
seconds wedged in a single 2 × 2 m cell beside `box_min_x`. But flush edges hand
the aircraft a metre of floor it had never been allowed into, and on the west
wall that floor carries shelving with ~20 cm *vertical* gaps at cruise height —
solid slabs at z = 1.4 and z = 1.7 with a slot between them. FALCON planned
straight into it and the aircraft wedged between two shelves at (−23.4, 0.2, 1.45), frozen
to the centimetre for twenty seconds while the reference oscillated 0.9 m either
side of it. That flight ended after 84 s with 313 m³, against 878 s and 857 m³
for the inset box the round before.

Inset edges cost a frontier cut that makes FALCON cycle and loiter; flush edges
cost a physical wedge that ends the flight. The cut is the cheaper failure, so
the inset stays.

**Obstacle inflation does not save you here, and it is worth knowing why.**
`patches/inflate_astar_by_airframe.sh` *is* applied to this image and *is*
active — `astar_inflate` defaults to 0.35 m, `start_clearance` to 0.50 m. But it
is **XY only**, deliberately: the rotor disc is horizontal, and the patch's own
header explains that inflating in z would eat most of the 1.2 m exploration
band. A slot between two horizontal slabs is open in XY, so A* routes into it
happily. Anyone moving these edges out again needs a *vertical* clearance rule,
or a `box_min_z` above the shelf band — not more XY inflation.

### Chasing a deadline that does not exist, into a wall

The chain that ended rounds 6 and 7, and the most useful thing found this
session. It starts with the clock and ends against a pillar:

```
Isaac runs at ~0.7x real time, FALCON plans on the wall clock
  -> the aircraft is structurally behind a schedule it cannot meet
  -> the catch-up term saturates at max_catchup_speed for most of the flight
  -> it flies 1.1-1.3 m/s along a route FALCON cleared for 0.6
  -> any tracking error at all now spends clearance it does not have
  -> contact
```

Measured on the two crashed rounds, airborne samples only: **25% and 44% of the
flight above the governor's own 1.1 m/s ceiling**, p90 1.24 and 1.34, peaks of
2.1 and 2.2 m/s — against a **0.6 m/s** plan (`max_linear_velocity` in
`adapter/launch/falcon_pegasus.launch`, which overrides the 2.0 in FALCON's
`uav_model_simulator.yaml`; check the launch file, not the yaml). Round 7
brushed its pillar at 1.97 m/s, survived, came back ninety seconds later and
died on the same corner.

**The governor was not broken.** Probed directly it does the right thing — at
1.3 m/s on a 0.6 m/s plan it commands −1.98 m/s², at 2.0 m/s it commands −5.06.
The aircraft was fast because the controller was *deliberately* driving it fast:
the catch-up term adds up to `max_catchup_speed` to the target velocity, and it
was pinned there by a lag that is mostly an artefact of the clock rather than a
real schedule deficit. The one term whose job is to recover time was spending
clearance to chase a deadline that cannot be met by construction.

So `max_catchup_speed` went 0.5 → **0.15** and `max_overspeed` 0.5 → **0.25**.
On the stub at Isaac's measured rate:

| | before | after |
|---|---|---|
| actual speed, mean | 1.00 m/s | **0.67** (plan 0.57) |
| above 1.1 m/s | 30–44% | **6.4%** |
| along-track lag, mean | +0.53 m | **+0.41 m** |
| cross-track, mean | 0.33 m | 0.37 m |

The lag got *better*, not worse, because the aircraft stops overshooting and
having to come back. That is the trade the tracker's own `_diagnose` argues for
in as many words: *being late is benign, being sideways is what hits walls.*

### The greedy TSP fallback was aborting the node it existed to save

Found by an adversarial audit and then caught live. `writeGreedyTourImpl` wrote
`COMMENT : Length = trunc(100 · Σcᵢ)`, summing raw doubles once at the end.
Upstream's convention is the opposite — `solveTSP` writes the cost matrix as
per-edge truncated integers, so LKH's reported cost is `Σ trunc(100·cᵢ)` — and
`planExploreMotionHGrid` re-derives exactly that per edge and asserts agreement:

```cpp
CHECK_NEAR(cost, grid_tour2_cost_sum, 1e-4)   // exploration_manager.cpp:246
```

glog's `CHECK_NEAR` is fatal. The two conventions differ by up to a hundredth
per edge — 0.04 on a ten-city tour, 400× the tolerance, and over tolerance on
400 of 400 random ten-city instances. So the fallback added to keep
`exploration_node` alive when LKH dies was killing it instead: **SIGABRT
(exit −6) immediately after "the LKH solver died on 'coverage_path'"**, with
`traj_server` still republishing the last endpoint so the aircraft looked
healthy while the planner was gone. The accumulator now sums hundredths per
edge. A stub flight that previously died at trajectory #180 after 94 m now runs
the full budget: 638 trajectories, 242 m, **1396 m³ covered**, node deaths 0.

Both patch scripts are now **re-runnable**, which they had to become to fix
this: the image already carried the patch, so the only way to correct a bug in
it was a rebuild from the base image — an hour, on a machine whose compiler
segfaults at random. Re-running now corrects an already-patched source in place,
and the header guard matters as much as the source one (re-declaring the helpers
is a hard compile error, not a no-op).

**And `soak.sh` could never have scored a clean streak.** It counted every
`Stack trace` in FALCON's log as a planner crash, but a *recovered* LKH crash
prints one from the forked child while the parent carries on — the isolation
patch working exactly as designed would have marked the flight dirty. Crashes
are now counted as `exploration_node … process has died`, with stack traces and
LKH recoveries recorded alongside as information.

### First real flight on the attitude cut — one run, `3_open_plan`

| | velocity cut (campaign, below) | attitude cut (n = 1) |
|---|---|---|
| exploring | 34 s | **163 s** |
| flown | 51 m | **88 m** |
| trajectories | 57 | **162** |
| mean / worst gap | 1.81 / 10.09 m | **1.01 / 4.75 m** |
| ended by | `crashed` | `stalled` |

Better on every column, and it did not hit anything — but **this is one run
against one run**, and FALCON is not deterministic. Treat it as "the path works
on real physics", not as a result.

Two things in it are worth more than the summary numbers:

**Steady-state tracking is now centimetres.** The status line sat at
`err=0.02m xte=0.00m` for tens of seconds at a time. The mean of 1.01 m is not
a description of how it flies — it is dominated entirely by transients.

**Those transients are all the same event.** Every spike follows FALCON
condemning its live trajectory: the aircraft holds, FALCON replans, and the
resume costs a metre or two (the worst was `err=3.55m xte=2.13m`, one tick after
`new trajectory #131 -- following again`). So the dominant error source on this
cut is no longer tracking a curve — it is *rejoining* one.

### The lag that would not close

A later flight made the second point sharper, and turned up a real bug. The
aircraft settled at a **persistent ~1.3 m of gap that was almost entirely
along-track** — flying at 1.0–1.4 m/s against a 0.6 m/s plan and *still* losing
ground. It was pushing hard and never catching up.

The catch-up term was inert. It measured the schedule deficit as
`elapsed - projected_time`, a difference of two times on the current curve, and
that reads zero in exactly the case it exists for:

- FALCON does not plan the next curve from the aircraft. It plans it from its
  **own previous curve**, at `now + replan_duration`.
- So a lagging aircraft is behind the new curve's **start**.
- A curve has no negative time, so the projection clamps at 0 and the deficit
  disappears. A true 1.30 m read as **0.03 m**, and the catch-up contributed
  0.03 m/s instead of its 0.5 m/s ceiling.
- FALCON replans about four times a second, so no lag ever survived on one curve
  long enough to be noticed.

The fix measures the deficit **in space** — the displacement to where the plan
says the aircraft should be *now*, projected onto the direction of travel — so
it does not care which curve is current or how long it has been. Same
displacement, resolved across the direction of travel, gives cross-track, so the
three numbers are one consistent decomposition and `along² + cross² == gap²`.

Flown after the fix: `err=0.02m lag=0.00m xte=0.00m`, held for tens of seconds.

The flight status line now prints the split, because the halves mean opposite
things — `lag` is benign, `xte` is what hits walls:

```
t= 40.0s pos=(-22.02, -7.89, 1.44) err=0.02m lag= 0.00m xte=0.00m traj#59
```

The flight ended with 288 m³ mapped, and the reported outcome — `stalled` at
(-20.1, -6.7) — is the *symptom*, not the cause. `exploration_node` segfaulted,
in `ExplorationManager::planExploreMotionHGrid`. The aircraft then held station
accurately against a frozen plan until the stall watchdog gave up, which is
exactly what it should do and exactly what it looks like from the outside.

**Read every Isaac Sim outcome against the FALCON log before believing it.** See
"exploration_node keeps segfaulting" below: on this stack the aircraft-side
outcome usually describes how the aircraft *noticed* the planner had died,
not why it died.

## Holding when FALCON condemns its own trajectory

FALCON checks the trajectory it is currently executing against the map every
cycle, and when it finds an obstacle on it, it says so — `[FSM] Collision
detected on the trajectory!` — publishes `1` on `/planning/replan`, and re-plans.
That is the correct response and it is also incomplete, for the same reason
everything else here needed a control layer: upstream, re-planning was free.
The geometry simulator's aircraft *was* its reference, so it stopped the instant
the reference did and the hundred-odd milliseconds of planning cost nothing. An
aircraft with mass spends those milliseconds flying into the thing FALCON just
found.

So the bridge forwards that `1` as a `trajectory_unsafe` event (edge-triggered —
the FSM re-fires it on every planning attempt), and the mission withholds the
reference from the tracker until a new trajectory arrives. Withholding rather
than zeroing: the tracker's station hold latches a point and flies back to it, so
the aircraft brakes to a stop instead of drifting, which is what a velocity-
controlled multirotor does when it is sent zeros. The stall and planner-stopped
watchdogs still run underneath, so a hold that never lifts ends the run rather
than hanging it.

## Telling FALCON how big the drone is

FALCON reasons about the aircraft as a point in two of the three places that
matter, and the three are worth keeping apart because they want different
settings:

| where | knob | set to | why |
|---|---|---|---|
| the trajectory | `bspline_opt/safe_distance` | 0.7 m (upstream) | the corridor it flies down — raising this closes doorways |
| how hard it avoids | `bspline_opt/pos/distance` (`obstacle_weight:=`) | **100** (was 50) | the weight on that clearance, which is the dial that actually centres a route |
| the viewpoint | `frontier_finder/min_candidate_clearance` | **0.45 m** (was 0.21) | where it *stops to look*, which costs nothing in doorways |
| the route | `astar/inflate_radius` | **0.35 m** (was nothing) | how wide A* believes the drone is; see below |

The middle row was upstream's 0.21 m — smaller than this airframe's own 0.35 m
radius — so FALCON was free to choose observation points the drone physically
cannot occupy and then route to them perfectly correctly. That is the "it does
not know how big the drone is" problem in its most concrete and most fixable
form, and unlike the first row it constrains only where the aircraft stops, not
what it can fly through.

The second row is the one worth understanding, because the first row is the
knob everyone reaches for and it is the wrong one. `safe_distance` is not a
constraint — it is the knee of a soft penalty, `(dist - safe)²` while
`dist < safe`, summed against smoothness, start and end terms. Raise the
*distance* and the optimiser fights itself in every corridor narrower than twice
it, which is how 0.9 m shut every doorway in the building. Raise the *weight* and
it wins where there is room to win and loses where the endpoints demand a thread.
That is the soft half of the Gazebo stack's wall handling (a hard dilation plus a
wall-proximity cost that centred routes in corridors), and it is one parameter.

The hard half — the third row — is **done as of `patches/inflate_astar_by_airframe.sh`**, and the
fourth row of that table is now `/astar/inflate_radius`, default **0.35 m**
(`AIRFRAME_RADIUS_M`). Upstream A* tests one voxel per node with no notion of a
radius, so it routes a 0.7 m aircraft through a 0.3 m gap and the B-spline
optimiser then cannot thread the route it was handed — the FSM loops on
`Collision detected on the trajectory before publishing` until the aircraft's
stall watchdog ends the run.

Three things about the patch are worth knowing before touching it:

- **XY only.** The rotor disc is horizontal and the airframe is 0.15 m tall;
  inflating z would eat most of the 1.2 m exploration band.
- **The route planner only** — the two-argument `search()`. The five-argument
  overload answers connectivity and tour-cost questions between grid cells, and
  inflating those changes which cells FALCON believes are reachable at all,
  which is a different and much larger claim than "the drone is 0.7 m wide".
- **The start is exempt within `/astar/start_clearance`** (0.5 m). Inflation
  stops the aircraft *entering* a pocket; it does nothing to get one out, and an
  aircraft already inside the skirt of a wall would have no valid start node and
  never plan again. Without this the dilation meant to prevent wedging becomes
  the wedge.

Both are ROS parameters defaulting to **zero**, so an unpatched or baseline image
behaves exactly as upstream and the radius can be retuned from the launch file
without another rebuild.

### What the stub says about it, and what it does not

Two 150 s stub flights on the patched image, against the pre-inflation run of the
same config:

| | trajectories | mean / worst gap | `No path` | collision loops |
|---|---|---|---|---|
| `3_open_plan`, no inflation | 245–264 | 0.25–0.33 / 0.77–2.33 m | 0 | 13 |
| `3_open_plan`, 0.35 m | 316 | 0.35 / 1.25 m | 0 | 11 |
| `2_room_warren`, 0.35 m | 268 | 0.31 / 1.04 m | **0** | 3 |

**The feared downside did not happen.** The concern was that a full-radius
dilation closes the warren's 0.8–0.9 m doorways — an 0.8 m door inflated by
0.35 m each side leaves one voxel at 0.1 m resolution, and raising
`safe_distance` to 0.9 m had previously shut every door in the building. It
does not: the warren flew 268 trajectories with **zero** `No path to next
viewpoint` failures. Nor does the stencil starve A* — 144 samples per node, and
the trajectory count went *up*.

**The upside is not demonstrated either.** Collision loops on `3_open_plan` went
from 13 to 11, which is noise between two non-deterministic runs. The stub
raycasts a clean ground-truth voxel map, so its walls are far tidier than the
ones a real depth stream builds, and the wedge this patch exists to prevent
happened on Isaac Sim rather than here. **Whether it fixes that is still
unmeasured** — it needs a real flight, and ideally the six-run campaign.

### Rebuilding: `docker build` is unreliable on this machine

`docker build` failed three times with gcc **internal compiler error:
Segmentation fault**, in a *different* translation unit each time
(`hierarchical_grid.cpp`, `exploration_fsm.cpp`, `frontier_finder.cpp`), each
inside standard-library template instantiation. It is not the patch: the same
patch, the same compiler and the same `-j2` compile cleanly when run as

```bash
docker run --name falcon-patchbuild -v "$PWD/patches:/patches:ro" falcon-pegasus:control \
  bash -lc 'source /opt/ros/noetic/setup.bash && bash /patches/inflate_astar_by_airframe.sh \
            && cd /catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release -j2'
docker commit -c 'ENV PYTHONPATH=/opt' -c 'WORKDIR /catkin_ws' \
  falcon-patchbuild falcon-pegasus:noetic
```

**And retry the link.** The failures are intermittent, not deterministic:
`collect2: fatal error: ld terminated with signal 11` hit three attempts in a
row and then succeeded on the fourth, with no change in between, and the ICEs
land in a different translation unit each time. Wrap the `catkin_make` in a
retry loop rather than debugging the patch — a toolchain that crashes at random
stages is a machine-health signal, not a code one. Worth a memtest.

One exception, because it *was* real: inserting `#include <algorithm>` before
`hierarchical_grid.cpp`'s first include — a PCL header — reproducibly killed
gcc with `internal compiler error ... during RTL pass: fwprop1`. PCL and Eigen
are include-order sensitive and the templates behind them are heavy enough to
turn that into a crash rather than a diagnostic. The file already reaches
`<algorithm>` transitively, so the patch now adds nothing and asserts that it
still does.

## What a campaign actually produces today

The six runs, measured on an RTX 5070 Laptop (8 GB), `office` scene, one aircraft
at a time:

| run | exploring | flown | trajectories | mean / worst gap | ended by |
|---|---|---|---|---|---|
| `1_north_hall` | 110 s | 90 m | 212 | 1.25 / 4.60 m | `stalled` |
| `2_room_warren` | 24 s | 44 m | 17 | 5.31 / 8.67 m | `crashed` |
| `3_open_plan` | 34 s | 51 m | 57 | 1.81 / 10.09 m | `crashed` |
| `4_south_wing` | 87 s | 66 m | 131 | 0.95 / 4.97 m | `planner_stopped` |
| `5_east_spine` | 52 s | 68 m | 65 | 5.69 / 12.21 m | `diverged` |
| `6_whole_office` | 74 s | 54 m | 117 | 0.90 / 2.58 m | `stalled` |

Every run is a real exploration: FALCON maps, finds frontiers, solves its
coverage tour, plans B-splines and flies them, and the map video shows the
building appearing. **None of them finishes the building**, and the runs that end
in `crashed` or `diverged` end because the aircraft hit something.

The honest summary of why, in order of how much each costs:

1. **Tracking lag versus clearance.** The aircraft is 0.7 m across the rotor tips
   and flies about a metre behind its reference through corners. FALCON's
   trajectories keep 0.7 m from mapped obstacles, which is enough for a point
   mass and not enough for a metre of lag. The two runs with a mean gap above 5 m
   are the two that hit walls. Slower flight helps and was measured helping
   (see `max_linear_velocity`); it does not eliminate it.
2. **Doorways.** The room warren's doors are 0.8-0.9 m. That is inside the
   airframe's own diameter plus its lag, and it is why `2_room_warren` is the
   worst of the six. Raising the clearance to fit the lag closes the doors
   entirely -- there is no setting that does both.
3. **The planner stops before the building is finished**, either because its LKH
   solver segfaults (below) or because a long route exceeds the A* budget that
   keeps the mapper fed. Both end the run cleanly now rather than hanging.

What would move the needle next, roughly in order: an inner loop that tracks the
*nearest point on the trajectory* rather than the reference at time *t* (which is
what removes corner-cutting as a class, rather than trading it against
convergence); giving FALCON's planner its own thread so its budget stops
competing with the mapper; and a smaller airframe for the room warren.

## `exploration_node` segfaults, and it was the reason no flight finished

This was the dominant failure mode on Isaac Sim — ahead of tracking, ahead of
clearance. **Every real flight ended with `exploration_node` dead**, at a
different site each time:

| flight | mapped | crashed in |
|---|---|---|
| attitude cut, no inflation, 163 s | 288 m³ | `ExplorationManager::planExploreMotionHGrid` |
| attitude cut, 0.35 m inflation, 43 s | — | `UniformGrid::positionToGridCellCenterId` |
| (earlier, twice) | — | `solveTSPLKH` → `LinKernighan`, below |

No stub run has ever crashed, on the same planner and the same building. The
stub raycasts a clean ground-truth voxel map, so it never produces the awkward
map states a real depth stream does — excellent for configuration, useless for
finding these.

`patches/fix_falcon_grid_cell_index.sh` fixes the second one, and it is the
**same bug shape** as `patches/fix_falcon_viewpoint_index.sh`: an unchecked
index into a `std::vector`, with the assertion that would have caught it sitting
in the source, commented out.

```cpp
int x = std::floor((pos.x() - config_.bbox_min_.x()) / config_.cell_size_.x());   // may be < 0
...
return x + y * num_cells_x_ + z * num_cells_x_ * num_cells_y_;                     // unchecked
...
Position bbox_min = uniform_grid_[cell_id].bbox_min_;                              // boom
// CHECK_GE(cell_id, 0) << "Invalid cell id in positionToGridCellCenterId";        // <-- upstream
```

**Why it bites here and not upstream: the exploration box is not the flyable
space.** In these runs it is inset from the building (`box_min_x` −23.0 against
a `map_min_x` of −25.0) and its floor is z = 1.0 while the aircraft cruises at
1.4 m. An aircraft working an outer wall, or dipping under the band during a
manoeuvre — the earlier flight logged a collision check at z = 0.63 — is
legitimately outside the box while flying perfectly. That is routine for a
building and rare in the single rooms FALCON ships configured for.

The fix clamps rather than rejects. "Which cell is this position in" has a
sensible answer just outside the grid — the nearest edge cell — and all four
callers want it: two only compare the id, and the third subscripts a vector the
clamp now keeps in range. Rejecting would mean teaching four call sites what −1
means, for nothing.

### What it bought

Same run, same config, the only difference being the guard:

| | explored | mapped | flown | trajectories | ended by |
|---|---|---|---|---|---|
| before | 43 s | 288 m³ | 31 m | 37 | **crash** |
| after | 133 s | **621 m³** | 107 m | **229** | `stalled` |

Zero crashes, and more than double the building mapped. The run now ends the way
the README's failure list describes: genuinely wedged at (−18.9, −8.8), holding
a 1.2–1.4 m **cross-track** error in a doorway at the south end while FALCON
went on planning to trajectory #365. That is a real control-and-clearance
problem to work on, which is progress — it is the first Isaac Sim flight whose
outcome describes the aircraft rather than the planner's memory safety.

Mean tracking error was 1.19 m and worst 2.74 m, against 1.01 / 4.75 m on the
unguarded flight: worst-case halved, mean slightly up because the aircraft spent
its last 80 s pinned against something rather than flying.

## FALCON's TSP solver crashes, and it is invisible from the aircraft

`exploration_node` segfaults inside its vendored LKH solver on some coverage-tour
instances. Its own backward-cpp trace names the frame:

```
#3  ExplorationManager::solveTSP(...)
#2  solveTSPLKH(char const*)
#1  FindTour
#0  LinKernighan            <-- SIGSEGV
```

Seen at 43 s, 133 s and 163 s into otherwise healthy flights, and twice more
before that. It is third-party C with global state, written to be a one-shot
program — read a file, solve, exit — and FALCON calls it thousands of times in
one process instead. **It was the single biggest reason no flight finished.**

What makes it dangerous is that **nothing downstream notices**. `traj_server` is
a separate process and outlives the planner, and its command callback holds the
*final point of the last trajectory* forever once the trajectory's duration has
elapsed. So commands keep arriving at 100 Hz, the aircraft tracks them to a
centimetre, and every health check reads perfect while the drone hovers for the
rest of its budget. The trajectory id is the only thing that stops moving, which
is why `PLANNER_STALL_S` watches exactly that.

### Isolating it, rather than repairing it

`patches/isolate_lkh_tsp_solver.sh` runs LKH in a **forked child**. Nothing here
can debug its internals and nothing needs to: `solveTSP` already talks to it
entirely through files — it writes `<name>.tsp`, calls the solver, reads
`<name>.txt`. So when the child dies, the parent is untouched and writes a
**greedy nearest-neighbour tour** into the file the reader is about to open, in
the same TSPLIB shape, leaving every line of upstream's parsing and all three id
conventions (`skip_first_`, `skip_last_`, `result_id_offset_`) exactly as they
were.

A greedy tour is worse than LKH's, and that is the right trade: a coverage tour
that visits the same cells in a slightly worse order costs seconds of flight,
while a dead planner costs the flight. It runs only after LKH has already
failed, and says so in the log (`the LKH solver died on ...`).

The child also gets a five-second deadline. `fork()` from a process with ROS's
background threads can in principle leave the child deadlocked on a malloc lock
held at fork time; LKH normally answers in milliseconds, so anything slower is
killed and treated as a failure rather than hanging the FSM.

### Three more, all the same shape

Every remaining crash found here is an **unchecked index into a container**, in
code that only meets the bad input on a building-sized map — and in two of the
three, the very same file bounds-checks the neighbouring function.

| patch | where | what reaches it |
|---|---|---|
| `fix_falcon_viewpoint_index.sh` | `candidates[min_cost_id]`, three sites | no candidate won, so the id is still −1 |
| `fix_falcon_grid_cell_index.sh` | `uniform_grid_[cell_id]` | the aircraft outside the exploration box |
| `fix_falcon_raycast_out_of_map.sh` | `data[addr]` on the TSDF fuse path | a depth ray stepping one voxel off the map |

The third is worth its own note because it is not a segfault. `TSDF::inputPointCloud`
raycasts from the sensor to each returned point; the endpoints are clamped into
the map in **metres**, but the DDA steps in **voxels** and can still land one
index past an edge. `updateTSDFVoxel` then writes the array directly and
`updateOccupancyVoxel` reads through a glog `CHECK` — and a failed `CHECK` is
`abort()`:

```
F0806 21:12:34 map_base_inl.h:173] Check failed:
  addr < map_data_->data.size() (6334695 vs. 6334614)
```

Eighty-one addresses past the end, one index off one face. It ended a
`6_whole_office` flight at 595 m³. The guard goes on the loop rather than on the
two callees — one condition, in voxel space, where `isInMap(VoxelIndex)` already
exists — and `OccupancyGrid::setOccupancy` one function away already does
exactly this check, so it is the file's own policy applied where it was missed.

**Together these took a `6_whole_office` flight from "crashes within a minute"
to zero crashes.** What ends a flight now is the aircraft, not the planner.

## The parts

```
falcon_pegasus/
  runs/<n>_<name>.yaml       one run: FALCON's map_config + the aircraft's spawn
  link/                      the wire protocol, imported by BOTH containers
    protocol.py                framing, message kinds, the 16-byte header
    socket_link.py             the two TCP endpoints
    depth_codec.py             float32 metres -> uint16 mm, and what inf means
  isaac/                     the aircraft (Isaac Sim's Python 3.12)
    run_exploration.py         entrypoint
    setup.py                   bring-up, in the one order that works
    mission.py                 arm -> climb -> survey turn -> hand over -> land
    sensing.py                 depth frame + camera pose, and the check on it
    falcon_client.py           the link, threaded so it can never stall physics
    px4_exploration_params.py  what exploration changes about the PX4 set, and
                               which of its gains each control cut can reach
  adapter/                   the ROS1 catkin package `falcon_pegasus`
    scripts/pegasus_bridge_node.py   Isaac <-> FALCON's simulator topics
    scripts/map_recorder_node.py     the map video, with no display attached
    launch/falcon_pegasus.launch     exploration.launch, simulator amputated
  viz/exploration_frame.py   how a frame of the map video is drawn
  run_campaign.sh            the six runs, once each -- the deliverable
  soak.sh                    ONE run, N times -- is it reliable?
  stub/                      the same mission without Isaac Sim (see below)
    airframe.py                the two stand-in bodies, one per control cut
  patches/                   five fixes to upstream FALCON, each explained in situ
                             (three of them are crashes; see the crash section)
  Dockerfile                 FROM the FALCON image, adds this catkin package
```

Everything pure lives outside this package: the whole control chain is
`core/control/` (with the velocity-cut controller still at
`core/planning/trackers/reference_tracker_3d/`), FALCON's B-spline is rebuilt by
`core/planning/trajectories/bspline/`, the camera extrinsics are
`robots/PEGASUS/adapters/camera_pose.py`, and the platform itself is
`robots/PEGASUS/`.

## The stub, and why to use it first

`stub/check.sh <run>` flies the whole mission against the real FALCON stack with
no Isaac Sim, no GPU and no PX4 warm-up. Depth comes from raycasting the surveyed
ground-truth voxel map — the same building, measured rather than rendered — and
the airframe is a lag: a first-order *velocity* lag on the velocity cut, which is
what PX4's velocity controller looks like from outside, or a *thrust-axis* lag
plus an unlearned thrust curve on the attitude cut, which is what its attitude
and rate loops look like. Everything else is the code that flies the real
aircraft: the same protocol, the same handover order, the same control chain, the
same exit conditions.

A green stub run means FALCON's configuration, the exploration box, the camera
contract, the bridge and the controller are all correct, and the only thing left
to prove on Isaac Sim is the simulator. A red one localises the fault in a
minute instead of an hour. Every problem documented below was found this way.

## Things that are true, and cost a day each to find out

**FALCON aborts on `use_sim_time`.** `exploration_node.cpp` opens with
`CHECK(!use_sim_time)`, a glog fatal. Publishing `/clock` does not degrade this
system, it kills the node that contains the mapper, the frontier finder and the
FSM. So everything is stamped on the **wall clock**, which both containers share
because they share a kernel — and the Isaac loop runs with `realtime=True`,
which is not a preference either: FALCON walks its B-spline at
`ros::Time::now() - start_time` on a 100 Hz timer, so a simulation running at any
rate other than 1× slides the commanded position along the trajectory at a speed
the airframe is not flying at.

**The aircraft must tell FALCON it is at its camera, not at its body.** The
camera is mounted 20 cm forward of the body origin and carves free space outward
from itself, so the body origin sits in the one place the camera can never
observe: 20 cm behind it, in every heading, for the whole flight. FALCON's A*
validates every 10 cm of every candidate step and treats UNKNOWN exactly like
OCCUPIED, so with the body origin as its start it rejects every neighbour on the
first checkpoint and returns NO_PATH before expanding a single node. The symptom
is `planTrajToView: No path to next viewpoint`, forever, for a viewpoint three
metres away in open space, against a map that is visibly fine. Upstream never
hits it because its `T_b_c` has zero translation. See `isaac/sensing.py`'s
`nav_position`.

**One depth frame and one camera pose, one timestamp.** FALCON refuses to fuse a
depth image it cannot pair with a camera pose to within 1 ms. Both messages are
built from a single `FRAME` carrying one capture time, and the pose is published
first, so the tolerance is satisfied by construction whatever the link does.

**The pose must be the camera's optical frame**, not the aircraft's. Feeding the
body pose produces a complete, self-consistent map rotated ninety degrees, and
raises nothing anywhere. `sensing.verify_camera_pose` cross-checks the pose
computed from the airframe's mount constants against Isaac's own
`Camera.get_world_pose(camera_axes="ros")` once at start-up, because that is the
only cheap moment the two can be compared.

**The camera is not the XTEND's.** `robots/PEGASUS/config/camera_falcon_explorer_640x480.yaml`
is a symmetric 90° × 74° pinhole, deliberately not the XTEND-matched calibration
the data-collection campaigns render. FALCON's frontier-visibility model
(`PerceptionUtils`) assumes a symmetric cone about the body boresight; the XTEND
crop has its principal point 67 px above centre, so FALCON would believe it can
see frontiers that are outside the image, fly to viewpoints for them, observe
nothing, and choose them again. The bridge compares the announced camera against
the rosparams FALCON unprojects with and refuses to run on a mismatch — a silent
disagreement there is the worst failure this system can have, because it
produces a confident map of a building the wrong size.

**One turn on the spot before handing over.** The camera sees a 90° wedge, so an
aircraft that has only ever pointed one way hands FALCON one wedge of free space
in a 30 m room; the coverage tour then picks a cell fifteen metres away that A*
cannot route to, and the FSM never leaves `PLAN_TRAJ`. One slow turn gives it a
closed bubble to plan out of — which is also what a real exploration drone does
on take-off.

**Two upstream fixes, in `patches/`.** `fix_falcon_viewpoint_index.sh` guards
three `candidates[min_cost_id]` reads where `min_cost_id` is still `-1` because
no candidate won; `std::vector::operator[](-1)` is undefined behaviour, and it
killed a healthy 85-second exploration with `exit code -11` and no message.
`log_plan_failure.sh` makes the planning-failure log name the two positions it
could not connect, which turns an unactionable line into a diagnosis.

**A* budgets and sensor range are upstream's, tuned for upstream's rooms.**
`astar.yaml` gives the default profile one millisecond and `voxel_mapping.yaml`
trusts depth to 5 m — fine for a 30 × 30 m room, not fine for a 30 × 74 m
building where the next cell is routinely fifteen metres away. Both are raised
in the launch file, with the reasoning next to the value. The A* budgets are
raised carefully: the planner runs inside the same single-threaded node that
services the depth callbacks, and on a failed plan the FSM retries at 100 Hz, so
a budget large enough to starve the mapper turns a planning problem into a
mapping problem.

**FALCON announces that it is finished by killing its own trajectory server.**
`/planning/replan == 2` sets `task_finished_` and `traj_server` exits, so the
command stream simply stops. The bridge watches for it and tells the aircraft;
without that the aircraft hovers until a timeout it did not need to wait for.

## Two more traps, both about the exploration box

**An exploration box edge in OPEN SPACE is a trap.** FALCON finds frontiers along
the cut, samples viewpoint candidates on a ring around each one, and discards
every candidate that falls outside the box — so a frontier *on* the boundary has
half its ring thrown away and often nothing usable left. The aircraft flies to
the edge, the hierarchical grid stops being able to place it (`Current cell id:
-5` in the log), and the FSM cycles `PLAN_TRAJ -> PUB_TRAJ -> PLAN_TRAJ` forever
while the aircraft holds station perfectly. A box edge on a **wall** is fine,
because nothing needs to be observed from beyond it. That is why there is one box
covering the whole building, inset 1.2 m from its outer walls, rather than six
sub-regions.

**The box's z floor is a furniture decision, not a safety margin.** FALCON plans
anywhere inside the box. With the floor at 0.6 m it flew the aircraft at 0.74 m,
which is desk height in this building, and wedged it against furniture twice. The
band is 1.0–2.2 m: the airspace above the clutter and below the lights.

And one dial that looks like the fix and is not: **`bspline_opt/safe_distance`.**
Raising it from upstream's 0.7 m to 0.9 m does stop the aircraft clipping walls,
and also makes every office doorway impassable — a 0.9 m gap cannot contain a
trajectory required to stay 0.9 m from both sides, so the aircraft parks in the
first room it enters. The wall clipping had a different cause (a box edge lying
on the wall) and a different fix (inset the box).

## Known limits

- **Only the `office` scene.** It is the only surveyed one, and `hospital`
  crashes Kit whenever `pegasus.simulator` is enabled (an upstream Kit bug — see
  `robots/PEGASUS/README.md`). The six runs vary the region, not the building.
- **One run at a time.** Not a port limit: Kit's start-up is the heaviest moment
  of a worker's life and this is an 8 GB laptop GPU.
- **`pymavlink` lives in the container's writable layer** and is lost on a
  container recreation. `run_isaac_side.sh` checks for it and reinstalls it
  rather than failing five minutes into a boot.
- **PX4 refuses to arm for the first ~150 simulated seconds** after boot, for no
  documented reason. `isaac/setup.py`'s `_wait_until_armable` absorbs it and says
  what it is waiting on; it is why a run takes minutes to get off the ground.
- **An exactly axis-aligned spawn heading never arms at all.** An Iris spawned at
  exactly `(-9.00, 0.00)` facing exactly 0° sits in OFFBOARD with a valid position
  estimate and refuses to arm for 420 simulated seconds without a word, its EKF
  stuck on `cs_fake_pos` and never reaching `cs_yaw_align`. Nudged 7 cm and 7°
  off the axis it armed in 4 seconds. Every run config's spawn is therefore
  deliberately off-axis and off round numbers. Use `--arm-only` to check a new
  one in a few minutes rather than discovering it a whole flight later.
- **FALCON's planned speed has to fit the airframe, not just PX4.**
  `max_linear_velocity` must stay under `MPC_XY_VEL_MAX`, but the binding limit
  is what the aircraft can actually follow: at 1.0 m/s it ran two to three metres
  behind the reference through every corner; at 0.6 m/s it holds it. This is the
  first dial to reach for when a run diverges.
- The exploration box's `box_max_z` must stay under ~2.8 m: above that the
  surveyed map is outdoor sky over the roof.
