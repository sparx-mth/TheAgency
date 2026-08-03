# sim_flight_recording — autonomous expert flights, recorded for VLA training

Fly a simulated drone from a random point A to a random point B inside a
furnished building, over and over, unattended, and write every flight as a
**flight recording** in the same on-disk schema real rosbag extractions use
(`tasks/planning/vlas/common/finetune/datasets/recording.py`). A simulated
flight is therefore a drop-in `data.recording` source for VLA fine-tuning —
there is no parallel dataset format.

The platform (scene loading, the Iris, its camera and intrinsics, the Isaac
Sim 6.0.1 and PX4 compatibility work all of this rests on) lives in
`robots/PEGASUS/` — **read that README first**.

## The pipeline

```
survey_scene.py            ONCE per scene: one C++ sweep of the whole building
   │                       at 10 cm  →  <scene>_voxels.npz   (3D ground truth)
   │                                    <scene>_alt0150cm.npz (a slab of it, for planning)
   │                       view_voxel_map.py opens the result and lets you walk it
   ▼
run_collection.sh          host-side launcher: syncs the repo, starts N workers
   │
   ▼
collect.py                 ONE worker = one Isaac Sim process + one PX4 instance.
   │                       Boots Kit once, then loops:
   │
   ├── episode_plan.py       sample a reachable goal (core free_space_sampler)
   │                         → plan a wall-avoiding route (core weighted A*)
   ├── episode.py            arm → take off → track the route → land at the goal
   └── flight_session.py     stream RGB + depth + full 6-DoF pose to disk
                             → <out>/<scene>_w0_e000/
```

Everything left of the simulator is pure and unit-tested: given a map and a
seed, the whole route plan for a campaign can be generated and inspected on a
laptop with no GPU. Only `collect.py`, `survey_scene.py` and `fly_direct.py`
need Isaac Sim.

## Quick start

From the **host**, with the `isaac-sim` container running.

**1. Survey the scene** — once, in 3D. Every altitude is then a slice of it:

```bash
docker exec isaac-sim bash -c "cd /tmp/dev/repo && /isaac-sim/python.sh \
  sparx_agency/tasks/planning/sim_flight_recording/survey_scene.py \
  --scene office --altitude 1.5 --preview"
```

Look at the `.png` it writes next to the map before trusting a new scene. A
survey that came out mostly empty is obvious in a picture and invisible in a
cell count. See **The ground-truth 3D map** below for what else comes out and
how to open it.

**2. Collect:**

```bash
sparx_agency/tasks/planning/sim_flight_recording/run_collection.sh \
    --scene office --episodes 20 --workers 4
```

```
--scene       office | simple_room  (surveyed) | warehouse | full_warehouse (not yet)
--episodes    flights per worker
--workers     concurrent Isaac Sim processes, 1..10 (see "Running many at once")
--altitude    cruise altitude, metres. Must match a surveyed map (default 1.5)
--resolution  camera WxH, e.g. 640x480. Default is the platform's own 504x392
--rate-hz     frame capture rate (default 10)
--out-dir     default /tmp/dev/recordings/<scene>
--seed        base RNG seed; worker N gets seed+N
--video       also write a chase-camera MP4 per flight
--stream      WebRTC livestream on :49100. One worker only — the port is a singleton
```

`collect.py` takes more (`--min-distance`, `--max-distance`, `--clearance`,
`--standoff`, `--depth-format`, `--settle-s`, `--max-consecutive-failures`,
`--realtime`); pass them through after `--`.

**3. Copy the recordings out:**

```bash
docker cp isaac-sim:/tmp/dev/recordings/office .
```

## What one episode is

1. **Sample a goal.** `core/planning/mission/free_space_sampler.py` draws a
   point from the largest *connected* block of clear space, at least
   `--min-distance` away, with `--clearance` metres of room around it. Sampling
   from one connected component is what makes an unreachable goal structurally
   impossible rather than something the planner discovers and the caller retries
   around. Ends are further restricted to the map's **landable** cells — those
   clear all the way to the floor, not merely at cruise altitude. A goal over a
   desk is somewhere the aircraft can hover and cannot be put down: it lands on
   the desk, tips, and every later episode is refused with `Preflight Fail:
   Attitude failure (roll)`. In `office`, 731 m² of the 788 m² flyable space is
   landable; the missing 7% is furniture.
2. **Plan a route to it.** `core/planning/planners/astar`'s
   `WeightedAStarPlanner2D` — the same planner the real drones fly. It inflates
   every obstacle by `--standoff`, then prefers the middle of a corridor to its
   edge, and emits corner-rounded waypoints every 2 m.
3. **Fly it, continuously.** The planner's corners are smoothed into a spline
   and chased with a moving carrot (`path_follower.py`), so the aircraft holds
   cruise speed the whole way instead of decelerating onto every waypoint. The
   output is a world-frame **velocity**; PX4 SITL is the inner-loop velocity
   controller. Three phases: climb straight up, turn to face the route, then
   follow it.
4. **Land at the goal**, and start the next episode from there.

The next episode starting where the last one landed is deliberate: it avoids
teleporting the aircraft (which would invalidate the autopilot's estimator), and
after the first flight every start point is itself a previously-drawn random
point. Combined with a random spawn per worker, that is where a campaign's
variety comes from.

## What comes out

```
<out-dir>/
  office_w0_e000/
    depth/000000.png     (H, W) uint16 millimetres        <- --depth-format npy for float32 metres
    rgb/000000.jpg       (H, W, 3) uint8
    intrinsics.json      the camera at the resolution actually rendered
    poses.npy            (N, 21) float32, see below
    meta.json            everything about how this flight was produced
  office_w0_e001/
  ...
  campaign_w0.json       manifest: every episode, its outcome, its geometry
  px4_worker0.log        PX4's own console — where a refused arming says why
  worker0.log            the worker's stdout
```

`poses.npy` columns (`sim_extract.POSE_COLUMNS`):

| cols | what |
|---|---|
| 0–3 | `t, x, y, yaw` — the original schema. Every existing reader uses exactly these |
| 4 | `z`, world up |
| 5–8 | `qx, qy, qz, qw`, body FLU → world ENU |
| 9–11 | `vx, vy, vz`, world-frame linear velocity |
| 12–14 | `wx, wy, wz`, body-frame angular velocity |
| 15–17 | `ax, ay, az`, world-frame linear acceleration |
| 18–20 | `ux, uy, uz`, body-frame linear velocity |

Columns 15–20 are kept because they are free at capture time and expensive
afterwards: Pegasus already holds them on the vehicle state, whereas recovering
acceleration from a finished recording means differentiating a velocity sampled
at the render rate — noisy exactly where the transients are. Nothing in the NavDP
pipeline reads them today; they are there so a later model that wants dynamics
does not need the campaign re-flown.

`t` is the **simulation clock**, not a frame index over a nominal rate, and the
pose in each row was read from the physics state at the instant its images were
rendered. A recording with only the first four columns (a rosbag extraction)
still loads — nothing reads past column 3 unless it asks for `pose_full`.

`meta.json` carries the provenance a training run wants: `scene`, `seed`,
`worker`, `episode`, `start_xy`, `goal_xy`, `planned_waypoints`,
`planned_path_length_m`, `detour_ratio`, and — the important one —
**`outcome`** / `outcome_ok`.

### Read `outcome` before training on a recording

Every flight is written out, including the ones that went wrong, because a
partial recording of a flight that hit something is worth inspecting and a
silently-dropped one is not. `outcome` is one of:

| outcome | meaning |
|---|---|
| `landed` | clean flight, reached the goal, landed within 2 m of it. **The only one you want to train on.** |
| `missed_goal` | landed, but more than 2 m from where it was sent |
| `crashed` | held past 60° of tilt for 3 s — it is lying against something |
| `stalled` | moved less than 0.5 m in 20 s while it should have been flying |
| `offboard_lost` | PX4 left offboard mode (a failsafe) and would not come back |
| `arm_timeout` | PX4 would not arm into offboard for 60 simulated seconds; `meta.json` quotes what it said |
| `flight_timeout` | route not finished within its budget |
| `land_timeout` | still airborne 60 s after the land command |

`campaign_w*.json` summarises the lot, so filtering is a one-liner. Two further
fields are worth filtering on even when the outcome is `landed`:

* **`waypoints_skipped`** — a waypoint PX4 could not reach in 30 s is skipped so
  the mission can carry on. Non-zero means the aircraft cut a corner somewhere,
  which is fine for most purposes and not if you are training on the geometry.
* **`estimator_drift_m`** — how far PX4's position estimate wandered relative to
  ground truth over the flight. Tens of centimetres is healthy. Metres means the
  aircraft was being commanded to the wrong place and the recording's *images*
  do not show what its *plan* says they should.

### Looking at them

```bash
.venv/bin/python sparx_agency/tasks/planning/sim_flight_recording/inspect_recording.py \
    ~/flight_dataset/office
```

Prints a table of every recording and writes two pictures into each: a contact
sheet (evenly spaced RGB frames over their depth maps — a black camera or a
drone facing a wall the whole way is obvious at a glance) and a plan view (the
flown path over the scene map, next to the route that was planned). Runs in the
repo venv; no GPU, no Isaac Sim.

## Running for hours: `campaign_supervisor.py`

`run_collection.sh` starts N workers and waits for them. That is right for a
campaign you are watching and wrong for one that runs overnight, because a
worker is *designed* to give up: three failed episodes in a row means the
aircraft is wedged against something it cannot right itself from, so it exits.
A wedge at episode 20 of 400 costs the other 380.

```bash
python3 sparx_agency/tasks/planning/sim_flight_recording/campaign_supervisor.py \
    --scene office --workers 4 --episodes 3000 --max-bytes 130e9 --hours 16 \
    --host-dir ~/data/sim/office --container-dir /data/office \
    --stop-file /tmp/stop-collection \
    -- --min-distance 12.0
```

It relaunches any worker that exits, so a wedge costs one Kit boot (about three
and a half minutes) rather than a worker's whole remaining share. It stops on
whichever comes first: `--episodes`, `--max-bytes`, `--hours`, `--min-free-gb`,
or `touch`ing `--stop-file` — which brings the campaign down at the next episode
boundary, leaving every finished recording valid.

Three things it does that are not obvious, and each of which cost a campaign
before it did them:

* **Every launch gets its own directory**, `w<worker>_c<launch>/`. `collect.py`
  numbers recordings `<scene>_w<W>_e<NNN>` from zero each launch, so a
  relaunched worker pointed at the directory it used before overwrites its own
  earlier flights one at a time, silently, and the campaign never grows. On
  restart the supervisor continues the launch numbering rather than restarting
  it, so an interrupted campaign can simply be started again.
* **It reaps orphaned PX4s.** A killed worker never reaches
  `px4_launch.terminate_px4`, so its PX4 outlives it holding TCP 4560+N; the
  next worker on that instance then fails to bind and PX4 says nothing useful
  about why. Instances are told apart by working directory, so a sweep never
  touches a live sibling.

  This is now belt *and* braces: `px4_launch.kill_stale_px4` does the same sweep
  from inside the container, and `launch_px4`/`terminate_px4` both call it. PX4
  runs with `-d`, so it **daemonises** — the process `Popen` holds is only the
  launcher, and the `bin/px4` that binds the HIL port detaches and survives
  terminating it. Every run therefore leaked one, and the supervisor was
  covering for it; a caller that runs two flights back to back in one process
  (`world_goal/fly_navdp.py`) had nothing covering for it and lost an arm to
  `no PX4 heartbeat after 300 simulated seconds`. If you see PX4 report sensors
  `STALE!` and never send a heartbeat, look for a second `bin/px4` on that
  instance before looking anywhere else.
* **It writes `supervisor.json` and `supervisor.log`** into the campaign
  directory — what was relaunched and why.

Watch it with `tools/campaign_monitor`, which shows flights, outcome mix, disk
against the cap and an ETA.

### Do not run more workers than the machine's *memory* allows

The limit is not VRAM. Measured on a 24-core / 62 GB / RTX 4090 machine, one
worker costs **4.3 GB of VRAM and 7–8 GB of RSS**, so VRAM allows six and
memory allows four. Five was tried and the machine swapped: the run queue sat
at 25 on 24 cores, context switches hit 4.5 M/s, and the symptom in PX4 was
`BARO #0 failed: STALE!`, `MAG #0 failed: STALE!` and `Preflight Fail: Battery
unhealthy` — a starved worker misses its `HIL_SENSOR` deadline, and PX4 then
refuses to arm for the rest of the campaign. Two workers of five produced
nothing but `arm_timeout` and stopped within five minutes.

**A campaign that will not arm is a memory-pressure symptom first.** Check
`vmstat` for swap-in before believing anything about PX4 or the aircraft.

## Running many at once

Each worker is a whole Isaac Sim process with its own PX4 instance. The identity
that keeps them apart is all derived from the worker index — see `px4_launch.py`:

| resource | worker `N` |
|---|---|
| offboard UDP (PX4 → us) | `14540 + N` |
| simulator HIL TCP | `4560 + N` |
| lock / socket files | `/tmp/px4_lock-N`, `/tmp/px4-sock-N` |
| PX4 working directory | `build/px4_sitl_default/instance_N` |
| RNG seed | `seed + N` |

**The working directory is the one that bites.** PX4 keeps `parameters.bson`,
`dataman` and `log/` relative to its cwd, and saves parameters with an
in-place `O_TRUNC` under a *process-local* lock. Two instances sharing one
directory silently corrupt each other's configuration on every flight. The old
launcher did exactly that, which is why it could only ever run one.

**PX4 caps this at 10.** `px4-rc.mavlink` sends every instance from 10 up to the
same offboard port, so they cannot be told apart. `run_collection.sh` refuses
more.

Practical limits below that are GPU memory and VRAM: each Isaac Sim process
wants several GB. `run_collection.sh` staggers worker starts by 45 s because
Kit's start-up is the heaviest moment of a worker's life and overlapping two of
them contends for the GPU hard enough to crash the RTX shader compiler.

## The bugs this is built around

Full detail, including how each was confirmed, is in `robots/PEGASUS/README.md`.

**1. Isaac Sim 6.0.1 stops dispatching physics callbacks** a couple of steps
after `world.reset()`, silently and with no exception. Pegasus drives
*everything* off those callbacks — state, sensors, rotor forces, and the
`HIL_SENSOR` stream PX4's lockstep clock runs on. `ManualPhysicsDriver` calls
those four methods by hand every step instead; `sim_loop.py` is the loop that
calls it. Without it PX4 never boots far enough to emit a heartbeat; with it,
the heartbeat arrives in about 1.5 s.

**2. Rotor thrust must be applied to the body, not the rotors.** In Isaac Sim
6.x a `RigidPrim` force on an articulation link does not land at the link's
centre of mass — it induces a parasitic pitch torque that flips the aircraft
onto its nose within two seconds. Fixed in `pegasus_isaac6_compat.patch` by
summing the rotor thrusts into the equivalent body wrench.

**3. Scene floors do not stop a rigid body.** The drone falls through
`simple_room`'s floor and comes to rest upside down beneath it, which PX4
refuses to arm on. `flight_session.add_collision_ground()` supplies just the
missing floor, invisible so it neither z-fights nor appears in recorded images.

**4. `world.step()` advanced a different amount of time depending on whether it
rendered.** This one was found late and mattered most. Pegasus's world defaults
to `physics_dt = 1/250` and `rendering_dt = 1/60`, which Isaac Sim turns into 4
physics substeps per rendered frame — so a step advanced 4 ms or 16 ms, and a
caller stepping the vehicle by one fixed `dt` was wrong on every step. PX4's
lockstep clock is integrated from that `dt` (so it ran roughly twice as fast as
the world), the simulated accelerometer is `(v − v_prev)/dt` (so specific force
alternated between 0.4× and 1.6× of truth at 25 Hz, straight into the attitude
estimator), and every recorded timestamp was a frame index over a rate the
simulation was not running at. `flight_session.build_world` now sets
`rendering_dt == physics_dt`, so one `world.step()` is exactly one physics step
whether or not it rendered.

## The ground-truth 3D map

The scene is swept **once**, in 3D, at 10 cm, and everything else is derived
from that. Isaac's own `isaacsim.asset.gen.omap` generator does the sweep in
C++: one PhysX box-overlap per voxel, 27 seconds for 27 million voxels. Doing it
per-voxel from Python is not an option at that scale.

```
robots/PEGASUS/maps/
  office_voxels.npz      307x745x63 voxels @ 0.1 m = 30.7 x 74.5 x 6.3 m
                         1.81 M occupied, 10.8 M free, 1.77 M unknown.
                         183 kB compressed, and committed.
  office_voxels.ply      the same occupied voxels as a point cloud, 27 MB.
                         Regenerated, not committed.
  office_alt0150cm.npz   the 2D map: a 0.6 m-thick slab of the above at 1.5 m,
                         plus the `landable` layer.
```

`VoxelGrid3D` satisfies the existing `VoxelMap3D` protocol, so the 3D planners
can use it directly. **A second altitude is free** — it is a slice, not another
sweep — and the 2D and 3D maps cannot disagree because one is computed from the
other.

### Walking around it

**No container.** The map is a file; the viewer reads it directly:

```bash
.venv/bin/python sparx_agency/tasks/planning/sim_flight_recording/view_voxel_map.py \
    --scene office --max-z 2.2 --recordings ~/sim_flight_recordings
```

Drag to orbit, scroll to zoom, and:

| key | |
|---|---|
| `[` / `]` | lower / raise the ceiling cut, 0.2 m a press |
| `C` | points ↔ cubes (points by default — 1.8 M cubes will not orbit) |
| `P` | print the current cut, to reuse as `--max-z` |
| `R` | reset the view |
| `H` | Open3D's own help |

`--recordings` overlays the flown routes: an amber track per flight, green at
the start, red at the goal. That is the fastest way to see whether the planner
is hugging walls or cutting corners it should not.

**Use the cut.** The ceiling is the largest surface in the building and it sits
between you and everything you wanted to see; without clipping you are looking
at a photograph of a roof. Just under the real ceiling (2.2 m here) turns it
into a floor plan with the furniture standing on it. `--min-z` drops the floor
the same way. `--screenshot out.png` renders one frame and exits, which is how
the viewer is checked over SSH.

**On Wayland, Open3D cannot get a GL context** — it fails with `Failed to
initialize GLEW` and a window that never appears. The script clears
`WAYLAND_DISPLAY` before importing Open3D so GLFW takes the XWayland path
instead; that is why it works here at all, and why it must run under a session
that has an `DISPLAY` to fall back to.

### Looking at it without Open3D

Only the `.npz` is committed — the point cloud is 27 MB and regenerated. Rebuild
it from the committed map, which needs **neither Isaac Sim nor Open3D**:

```bash
.venv/bin/python sparx_agency/tasks/planning/sim_flight_recording/voxel_export.py \
    --scene office --preview --max-z 2.2
```

That writes `office_voxels.ply` next to the map — openable in MeshLab or
CloudCompare as well as Open3D — and, because of `--preview`, also renders a
shaded isometric PNG using just numpy and cv2. That path is what checks the map
on the machine that *produced* it: Isaac's Python has no Open3D, and a 450 MB
wheel to look at one file is a poor trade.

The exporter writes the minimal binary PLY (`float x/y/z` + `uchar
red/green/blue`, 15 bytes a point) by hand rather than taking an Open3D
dependency it cannot satisfy where the file is produced. Points are coloured by
height, and the voxel size in the `.npz` reconstructs the cubes exactly.

### Three things about that generator

Each of them costs an afternoon:

* **Importing the binding before enabling the extension kills the process.**
  `from isaacsim.asset.gen.omap.bindings import _omap` on its own terminates Kit
  with exit code 0, no traceback — indistinguishable from a clean shutdown.
* **`set_transform`'s bounds are relative to the origin**, not absolute world
  coordinates.
* **`get_buffer()` is useless after `generate3d()`** — it returns a flattened
  2D-sized buffer with no occupied cells in it. The 3D result comes out of
  `get_occupied_positions()` / `get_free_positions()`.

And one thing about the result: the generator floods outward from its origin and
marks what it cannot reach as unknown, which sounds like it separates the
building from the field it stands in. **It does not** — the free space above the
roof connects to everything, so the flood escapes over the top. Measured on
`office`, 867 m² of building came back as 4618 m² of building-plus-car-park with
zero unknown voxels. A ceiling test (`restrict_to_indoor`) is what actually
separates them, and with a voxel column in hand it is one array reduction.

## How the aircraft is actually flown

**A smoothed spline, a moving carrot, and velocity setpoints closed on ground
truth.** Two decisions, both arrived at the hard way.

### Continuous, not waypoint-by-waypoint

The planner emits corner points. Flying them literally — aim at the next corner,
decelerate onto it, accept it, aim at the next — gives stop-and-go motion: the
aircraft pauses at every waypoint and the camera swings. `path_follower.py`
composes two components the repo already had, both of which the real drones use:
`HermiteSmoother3D` turns the corners into a G1-continuous spline, and
`PurePursuitTracker3D` chases a carrot along it with a lookahead that shortens
on tight curves.

Measured against a perfect vehicle over a route with two right angles and a
doubling-back: **mean speed 1.14 m/s against a 1.2 m/s cruise, standard
deviation 0.055 m/s, and the slowest moment is 79% of cruise** — that dip is the
deliberate corner easing, not a stop.

One number in there can fly the aircraft into a wall. The spline cuts *outside*
the planner's corner points, and the planner's obstacle standoff is the entire
budget for that; measured, the worst-case bulge is about `1.8 x tangent_scale`
metres. The default 0.2 spends 0.36 m of a 0.6 m standoff. Raising it makes
prettier curves and less clearance.

### Velocity, not position

PX4's offboard *position* path does not work in this setup. Given a
`SET_POSITION_TARGET_LOCAL_NED` one metre away — in a healthy offboard mode
(`custom_mode` main mode 6), no failsafe active, PX4's own estimate tracking
ground truth to within 30 cm, correct `type_mask` — the aircraft closed the gap
at **one centimetre per second** and every flight timed out. Nothing in PX4's
console explained it. Three different frame treatments (live offset, latched
translation, latched translation + rotation) all produced the same stall, which
is what eventually ruled the frame maths out as the cause.

`episode.guidance_velocity` instead computes a clamped proportional velocity
from the **simulator's exact position**, and `px4_offboard.send_velocity_world`
rotates it into PX4's frame and streams it as a velocity setpoint. That is a
better design regardless of the bug:

* PX4's estimator drift stops mattering. Only the aircraft's *true* path
  converges, so a metre of accumulated estimator error changes nothing.
* PX4's local-frame **origin** drops out entirely — only its *heading* is still
  needed, to rotate the velocity vector, and that is measured (see below).
* It matches the rest of this repo, where every follower emits `/cmd_vel`.

Two consequences worth knowing:

* **Velocity control does not hold position.** Commanding zero horizontal
  velocity during the takeoff climb let the aircraft drift three metres
  sideways in five seconds. The climb therefore guides toward the take-off
  point, which holds position properly.
* **PX4's heading reference is not the simulator's grid.** Measured between
  −1° and −11° on different boots, from its magnetometer against a world whose
  +y is simply "north" by convention. `PX4Offboard.latch_frame` measures it
  while the aircraft is stationary and every command is rotated by it.

## Position accuracy: exact sensors, not external vision

Pegasus simulates a GPS receiver, an IMU, a magnetometer and a barometer, noise
and biases included. Outdoors that is the right model; indoors it cost metres of
position hold — comparable to the gap between two office desks — and about half
of `office` flights ended against furniture.

The fix is `robots/PEGASUS/adapters/sensors.py`: **every configurable noise term
is zeroed**, which makes PX4's estimator input ground truth. The barometer needs
a subclass because, alone among the four, it has no config key for its noise —
it unconditionally injects ~1 Pa, which is 8.4 cm of altitude. `px4_params.py`
then tells the estimator to believe the now-exact GPS (`EKF2_GPS_P_NOISE` 0.5 →
0.05) and stops it gating a perfect fix on invented quality metrics
(`EKF2_GPS_CHECK`).

The magnetometer is kept rather than disabled: noiseless, it is what makes yaw
observable while the aircraft sits still on the ground, which GNSS-velocity yaw
is not.

**External vision (`px4_vision_pose.py`) is the road not taken.** It is a larger
change — switching the estimator's aiding source — for a problem that turned out
to be sensor noise plus the timestep bug. It stays in the tree, unwired, with
its investigation and the reasons it stalled documented in the module. Read it
before starting that work again; two of its three remaining suspects are now
known.

## Files

| file | role |
|---|---|
| `run_collection.sh` | **host-side launcher — the entry point.** Syncs, starts N workers |
| `campaign_supervisor.py` | the entry point for an *unattended* campaign: relaunches workers, caps disk |
| `collect.py` | one worker: boot Kit once, fly N episodes, write a manifest |
| `survey_scene.py` | one-off: sweep a scene into a reusable 3D voxel map + 2D slab |
| `voxel_export.py` | turn a saved map into a `.ply` + an isometric PNG, without Open3D |
| `view_voxel_map.py` | the interactive Open3D viewer — orbit the map, overlay flown routes |
| `campaign_setup.py` | the order-sensitive bring-up: map → world → vehicle → PX4 → params |
| `episode_plan.py` | sample a reachable goal, plan a wall-avoiding route to it |
| `episode.py` | arm, take off, track the route, land — one flight, one outcome |
| `sim_loop.py` | the hand-driven physics loop everything hangs off |
| `manual_physics_driver.py` | **the fix that makes any of this work** — see bug 1 above |
| `flight_session.py` | boot Kit, build the world, spawn the aircraft, stream frames out |
| `path_follower.py` | the continuous follower: spline, carrot, velocity + yaw command |
| `px4_launch.py` | start/stop PX4 instances with per-instance ports and directories |
| `px4_offboard.py` | non-blocking MAVLink: parameters, arm, setpoints, land, land-detect |
| `px4_params.py` | the parameter sets an indoor simulated drone needs |
| `chase_camera.py` | external camera that follows the drone, for video and streaming |
| `inspect_recording.py` | review what came out — contact sheets and plan views, no GPU |
| `fly_direct.py` | no autopilot: forces from a Python PD controller. Debugging only |
| `px4_vision_pose.py` | unfinished external-vision fusion, kept for its notes |

## Frames

Two conversions, both isolated in one place each:

* **Simulator → MAVLink setpoints.** The world is ENU; MAVLink local setpoints
  are NED. `px4_offboard.enu_to_ned()`.
* **PX4's local origin is not the world origin.** Its estimator anchors wherever
  the vehicle sat when PX4 booted, so world coordinates sent as local setpoints
  fly the drone off by exactly the spawn offset — measured: commanded
  `(-4.0, 3.5)`, arrived `(-8.2, 7.8)`, from a spawn at `(-4.6, 4.4)`.
  `PX4Offboard` recovers that offset by comparing PX4's reported
  `LOCAL_POSITION_NED` against ground truth; always send setpoints via
  `send_setpoint_world()`.

  **That offset is latched once per flight, and it has to be.** Recomputing it
  on every setpoint looks more accurate and is in fact a positive feedback loop:
  the commanded point becomes `target − truth + estimate(truth)`, so any
  position-*dependent* error in PX4's estimate moves the setpoint as the
  aircraft moves toward it. With a small rotation between the two frames — which
  is what a magnetic-heading reference against a grid-aligned simulator world
  gives you — that displacement is perpendicular to the motion, and the aircraft
  flies a circle around its waypoint instead of arriving. It was caught doing
  exactly that: a stable 1.1 m orbit, held for 100 seconds, through three
  waypoint timeouts, with PX4 reporting a healthy offboard mode throughout.
  `estimator_drift_m` in each recording's metadata is how you would notice the
  latched value going stale.

Everything else — the map, the planner, the waypoints, the recorded poses — is
world ENU with an FLU body, the repo-wide convention.

## Watching a run

Isaac Sim runs fully headless here, so "watching" means either an MP4 (`--video`,
needs nothing extra) or NVIDIA's **Isaac Sim WebRTC Streaming Client** pointed at
`localhost:49100` once a `--stream` run prints `STREAMING_READY`. Only one
process can bind that port, so `--stream` and several workers do not mix.

`--realtime` (a `collect.py` flag) throttles the simulation to wall-clock time.
Off by default: with warm GPU caches the simulation runs faster than real time,
and a collection run has no reason to wait.

## Testing

Everything that does not need Isaac Sim is unit-tested and runs in the repo venv:

```bash
.venv/bin/python -m pytest \
  sparx_agency/tasks/planning/sim_flight_recording/tests \
  sparx_agency/robots/PEGASUS/tests \
  sparx_agency/core/planning/mission/tests \
  sparx_agency/core/planning/environment/tests \
  sparx_agency/tasks/planning/vlas/common/finetune/tests/test_sim_extract.py
```

The episode-planner tests build a two-room map with a doorway and assert that no
leg of an emitted route passes through a wall — which is the one property the
whole pipeline rests on.

## Verified

`office`, 2026-07-27, one worker, six episodes back to back from a single Kit
boot, chaining across the building. **Six of six landed.**

```
recording        outcome    frames      s     Hz  flown m  plan m  goal m  yaw deg
office_w0_e000   landed        542   54.1  10.00     45.5    42.7    0.28      143
office_w0_e001   landed        266   26.5  10.00     15.8    11.5    0.63       60
office_w0_e002   landed        292   29.1  10.00     17.1    13.7    0.66      158
office_w0_e003   landed        280   27.9  10.00     16.3    13.0    0.62      121
office_w0_e004   landed        336   33.5  10.00     20.2    16.8    0.76      237
office_w0_e005   landed        520   51.9  10.00     43.5    40.0    0.66       73
```

Every recording is at **exactly 10.00 Hz** — the timestep fix, visible. Yaw
coverage of 60–237° per flight means the camera really does turn.

Measured over the transit phase (first movement to last), against the same
scene flown waypoint-by-waypoint before the continuous follower existed:

| | before | after |
|---|---|---|
| cruise speed | 0.42 m/s | **0.95 m/s** |
| speed std dev | 0.187 | **0.276** |
| stopped (<0.3 m/s) during transit | 30.0 % | **9.7 %** |
| yaw rate, 95th percentile | 16.4 °/s | **14.1 °/s** |
| a typical flight | 19 m in 50 s | **26 m in 37 s** |
| landed, from the goal | 1.37 m | **0.60 m** |

The residual 9.7 % is the acceleration out of the initial hover and the
deceleration into the goal. Through the middle of a flight the speed does not
drop at all: sampled in 5-second windows across a 46 m route, every window from
t=15 s to t=50 s was **0.0 % stopped** at 1.02–1.19 m/s.

The recordings load through `recording.load_recording()` with `(392, 504)` depth
in metres, co-registered RGB, matching intrinsics, and 21-column ground-truth
poses; `future_path_body()` and `goal_body()` — what the ESDF label generator
calls — run against them.

**One thing PX4 does that no parameter fixes:** it refuses to arm for the first
~180 simulated seconds after boot, with no stated reason. `wait_until_armable`
absorbs that before the first episode is planned, because otherwise those
attempts are recorded as failed episodes — and three failures in a row is what
stops a worker, so a campaign could be killed by its own warm-up.

## Known gaps

* **Flights still occasionally end against something** — none in the six-episode
  run above, one in seven and one in six on earlier runs. The route is verified
  clear in advance and the aircraft tracks it to well within the planner's
  standoff, so the remaining suspects are the smoothed spline cutting outside
  the corner points (bounded, but it spends 0.36 m of a 0.6 m standoff) and the
  2D slab hiding something the rotor arms meet outside it. The 3D map is now
  there to check the second properly; nothing does yet.
* **A crash ends the worker.** The aircraft lands on its side, PX4 refuses to
  arm on attitude, and after three refused episodes the worker stops — correctly,
  since nothing here can right it. Recovering would mean teleporting the airframe
  and restarting PX4 under it, which is not implemented. A crash therefore costs
  the rest of that worker's campaign, not just one episode.
* **Only one worker has been run at a time**, on an 8 GB laptop GPU that cannot
  fit two Isaac Sim processes. The per-worker isolation (ports, lock files,
  working directories, seeds) is implemented and unit-tested, and
  `run_collection.sh` drives it, but the multi-worker path has not been executed
  end to end. That is the first thing to try on the 4090.
* `hospital` cannot be flown: enabling the `pegasus.simulator` extension at all
  crashes Kit inside `libomni.anim.behavior.core.plugin.so`, before the scene
  even loads. Upstream bug, no known workaround. See `robots/PEGASUS/README.md`.
* `warehouse` and `full_warehouse` have not been surveyed.
* Routes are planned once, before takeoff, against a static map. Nothing here
  reacts to anything in flight — the route is verified clear in advance and the
  autopilot is trusted to track it. That is appropriate for generating *expert*
  demonstrations and would not be for flying a real aircraft.
* Every episode flies at one fixed altitude. The map is 2D, so a route never
  goes over or under anything.
* The consumer of this data is
  `tasks/planning/vlas/navdp/finetune/world_goal/`, which labels every admitted
  frame against the same surveyed map these flights were planned on. It reads
  recordings straight out of the campaign directories — no export step — and
  filters them itself (cruise altitude ±0.35 m, under 25° of tilt, inside the
  map, not standing in geometry), so a crashed or stalled episode is still worth
  keeping: the label taught at each frame is the route a planner *would* have
  flown, not the one that ended in a wall.
