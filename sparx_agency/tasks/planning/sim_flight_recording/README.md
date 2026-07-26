# sim_flight_recording — fly a simulated drone and record training data

Mission-level harnesses that fly the PEGASUS Iris around an Isaac Sim indoor
scene and write the result as a **flight recording** in the same on-disk schema
real rosbag extractions use
(`tasks/planning/vlas/common/finetune/datasets/recording.py`). A simulated
flight is therefore a drop-in `data.recording` source for NavDP fine-tuning —
there is no parallel dataset format.

The platform itself (scene loading, the Iris, its camera and intrinsics) lives
in `robots/PEGASUS/` — read that README first for setup, and for the Isaac Sim
6.0.1 / PX4 compatibility work these scripts depend on.

## Quick start

From the **host**, with the `isaac-sim` container running:

```bash
sparx_agency/tasks/planning/sim_flight_recording/run_flight.sh --scene office
```

It syncs the repo into the container, clears anything a previous run left
behind, and starts the flight. When the log prints `STREAMING_READY`, open
NVIDIA's *Isaac Sim WebRTC Streaming Client* and connect to `localhost:49100`
to watch. Add `--video` to also write an MP4.

```
--scene     office (surveyed) | simple_room (surveyed) | hospital (crashes, see below)
            | warehouse | full_warehouse (not surveyed yet)
--mode      px4 (default) | direct
--altitude  cruise altitude in metres (default 1.5)
--out-dir   where the recording goes (default /tmp/dev/recordings/<scene>_<mode>)
```

Run **one flight at a time**: the GPU, PX4's UDP ports and the livestream port
are all singletons, and a second run will fight the first for them.

## What is here

| file | role |
|---|---|
| `run_flight.sh` | host-side launcher — the entry point |
| `fly_px4.py` | **the main path.** PX4 SITL flies; this streams offboard setpoints |
| `fly_direct.py` | no autopilot: applies forces from a Python PD controller |
| `probe_scene.py` | raycast survey of a scene's free space → spawn + route |
| `flight_session.py` | boot Kit, load scene + vehicle, warm the camera, record |
| `manual_physics_driver.py` | **the fix that makes any of this work** — see below |
| `px4_launch.py` | start/stop PX4 SITL with a working directory that boots |
| `px4_offboard.py` | non-blocking MAVLink: arm, offboard setpoints, land |
| `px4_vision_pose.py` | feed PX4 a precise pose instead of simulated GPS |
| `waypoint_mission.py` | sequence waypoints into setpoints |
| `chase_camera.py` | external camera that follows the drone, for viewing |
| `record_flight.py` | original stationary smoke test (no flight control at all) |

## The three bugs you need to know about

Everything here is shaped by these. Full detail, including how each was
confirmed, is in `robots/PEGASUS/README.md`.

**1. Isaac Sim 6.0.1 stops dispatching physics callbacks** a couple of steps
after `world.reset()`, silently and with no exception. Pegasus drives
*everything* off those callbacks — state, sensors, rotor forces, and the
`HIL_SENSOR` stream PX4's lockstep clock runs on. `ManualPhysicsDriver` calls
those four methods by hand every step instead. Without it PX4 never even boots
far enough to emit a heartbeat; with it, the heartbeat arrives in 1.4 s.

**2. Rotor thrust must be applied to the body, not the rotors.** Pegasus
applies each rotor's thrust to its own `/rotorN` prim. Those are articulation
links, and in Isaac Sim 6.x a `RigidPrim` force on a link does not land at the
link's centre of mass — it induces a large parasitic pitch torque that flips
the aircraft onto its nose within two seconds. Measured, at 1.5× hover thrust:
per-rotor forces reached −76° of pitch by step 200 and ended upside down,
while the identical total force applied to the body climbed cleanly to 3.3 m.
Fixed in `pegasus_isaac6_compat.patch` by summing the rotor thrusts into the
equivalent body wrench (total thrust plus `Σ rᵢ × Fᵢ`), which is the same rigid
body dynamics.

**3. Scene floors do not stop a rigid body.** The drone falls straight through
`simple_room`'s floor and comes to rest upside down beneath it (`z = −0.72`,
`roll = 180°`), which PX4 refuses to arm on. The *walls* do collide — the
raycast survey hits them — so `flight_session.add_collision_ground()` supplies
just the missing floor, invisible so it neither z-fights with the scene's own
floor nor appears in recorded images.

## Why flights need a surveyed route

These are furnished buildings with no machine-readable floor plan. Spawning at
the origin and flying a fixed pattern does not work: the first `office` run
wedged the drone against an obstacle 1.7 m behind its spawn point.

`probe_scene.py` raycasts a grid at flight altitude and keeps only cells that
are genuinely *indoors* — floor below, ceiling above, **and** enclosed on all
eight horizontal directions. That last test matters: these assets sit on a
kilometre-wide ground plane, and a floor-and-ceiling test alone accepted 2.9
million cells, most of them open field outside the building. It then chains the
most open cells into a route, raycasting every leg over a 1 m-wide corridor
before accepting it.

Results are pasted into `SCENE_SURVEYS` in
`robots/PEGASUS/adapters/scene.py`. Re-run the probe if you change altitude —
clearance at head height and at desk height are different things.

## Position accuracy: GPS today, vision not yet working

Pegasus simulates a GPS receiver, noise and all. Outdoors that is the right
model; indoors it is marginal. On GPS, PX4's position hold wanders 2-5 m around
each setpoint — comparable to the gap between office desks — so a run either
completes the route or clips furniture on the way. **Roughly half of `office`
runs complete; the rest hit something.** `fly_px4.py` detects that (60° of tilt
held for 3 s), aborts with a clear message, and still writes the partial
recording.

`px4_vision_pose.py` implements the proper fix — feed PX4 a ground-truth pose
as a vision estimate and switch EKF2 off GPS, which is what a real indoor drone
does with VIO or motion capture. **It does not work yet**: PX4 accepts the
parameters but then refuses to arm with `Preflight Fail: ekf2 missing data`.
Two real bugs were found and fixed on the way there (REAL32/INT32 parameter
type mismatch; vision sent over the HIL link, which PX4 ignores) and the
symptom persists. It is opt-in via `--vision` and documented in that module.

What *does* help, and is on by default: conservative indoor limits
(`INDOOR_LIMITS` in `px4_offboard.py`) — 1.5 m/s, 20° of tilt, gentle
acceleration, and a 3 s takeoff thrust ramp. PX4's shipped defaults are tuned
for open sky and overshoot every indoor waypoint into a wall; without the
takeoff ramp the airframe has flipped onto its back two seconds after arming.

## Frames

Two conversions, both isolated in one place each:

* **Simulator → MAVLink setpoints.** The world is ENU; MAVLink local setpoints
  are NED. `px4_offboard.enu_to_ned()`.
* **PX4's local origin is not the world origin.** Its estimator anchors
  wherever the vehicle sat when PX4 booted, so world coordinates sent as local
  setpoints fly the drone off by exactly the spawn offset — commanded
  `(-4.0, 3.5)`, arrived `(-8.2, 7.8)`, from a spawn at `(-4.6, 4.4)`.
  `PX4Offboard.frame_offset()` recovers the offset continuously by comparing
  PX4's reported `LOCAL_POSITION_NED` against ground truth; always send
  setpoints via `send_setpoint_world()`.

## Verified

`office`, 2026-07-26, PX4 in the loop, launched via `run_flight.sh` with WebRTC
streaming live on port 49100: PX4 heartbeat 1.4 s into simulated time, armed at
t=16 s, climbed to 1.5 m, and flew the surveyed route — reaching `(4.0, -1.0)`,
`(-9.0, -10.3)`, `(-12.5, -1.4)` and `(5.1, -10.7)` — then returned to the
spawn and landed. 1516 frames over 151 s, 109 m of path, full 360° of yaw
coverage, altitude held within 10 cm.

The recording loads through `recording.load_recording()` with correct
`(392, 504)` depth and RGB, matching intrinsics, and real per-frame
ground-truth poses. `future_path_body()` — what the ESDF label generator
calls — runs against it.

Not every run gets that far; see the position-accuracy section above.

## `hospital` does not load (unresolved)

`hospital` is **not usable in this Isaac Sim 6.0.1 build**. Loading
`hospital.usd` crashes Kit roughly 25 s in, every time, inside
`libomni.anim.behavior.core.plugin.so`:

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  no null terminator at count
```

The Hospital environment ships animated character behaviour graphs, which
`office` and `simple_room` do not — that is the obvious suspect. But disabling
`omni.anim.behavior.core`, `omni.anim.behavior.schema`,
`omni.behavior.scripting.core`, `omni.anim.graph.core` and `omni.anim.people`
before loading the stage does **not** prevent the crash, so the animation
extensions are at most part of it. Reproduced three times, including with
nothing else running on the GPU.

`office` is the working furnished indoor scene; `simple_room` is the small one.
`scene_spawn()`/`scene_route()` raise a clear error naming the probe command
for any scene that has not been surveyed.

## Known gaps

* **Vision-pose fusion does not work** (`--vision`), so position hold is only
  GPS-grade and about half of `office` runs end against furniture. This is the
  single highest-value thing to fix next — `px4_vision_pose.py` documents
  exactly where the investigation stopped.
* Waypoints are a fixed surveyed route per scene, not a planner. Nothing here
  reacts to obstacles in flight; the route is verified clear in advance and the
  autopilot is trusted to track it.
* `warehouse` and `full_warehouse` have not been surveyed.
* No NavDP fine-tuning has been run against this data yet.
