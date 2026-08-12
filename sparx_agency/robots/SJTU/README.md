# SJTU — simulated drone platform (ROS 2 Humble + Gazebo Classic 11)

The "robot" layer for a velocity-controlled quadrotor in Gazebo Classic. The
aircraft is the `sjtu_drone` model driven by `libplugin_drone.so`, a PID cascade
that owns attitude and velocity and exposes a single body twist to the outside
world. That one fact shapes everything here: this platform gets a **velocity**
backend (`core/control/velocity_servo/`) and not an attitude one, because there
is no attitude, rate, thrust or motor input to send while it flies.

It is the cheapest full-stack loop in the tree. No PX4, no Isaac Sim, no GPU
physics — a world comes up in about a minute on a laptop, publishes RGB, depth,
a point cloud, IMU, sonar and 30 Hz odometry, and takes velocity commands. What
it is *not* is a fidelity model: the plugin is a kinematic-ish PID rig, not a
rotor-level simulation, so a controller that works here has been shown to work
against a plausible plant, not against an airframe.

## What lives here and what does not

**The simulator is an external repository and is never vendored.** The worlds,
the URDF, the plugin source and the Docker image build context live in a
separate ~207 MB git repo, referenced through one environment variable:

```bash
export SJTU_PROJECT_DIR=/path/to/sjtu_project
```

Everything in this package derives from that — `setup/env.sh` validates it and
exports the rest, and no absolute path to the simulator is written down anywhere
in TheAgency. A machine without the checkout fails with a sentence rather than a
stack trace, and two checkouts on one machine are the same case as one.

What this package owns is the contract: the topic and frame names, the camera
calibrations, the measured airframe plant, and the bring-up script. All of it is
importable and testable in the plain `.venv` with no ROS 2 and no simulator —
`pytest sparx_agency/robots/SJTU` needs neither.

## The control surface is one twist and five latches

`geometry_msgs/Twist` on `/simple_drone/cmd_vel`, read in the **yaw-aligned body
frame** (FLU, REP-103):

| field | meaning |
|---|---|
| `linear.x` | body-forward speed, m/s |
| `linear.y` | body-**left** speed, m/s |
| `linear.z` | climb rate, m/s, positive up |
| `angular.z` | yaw rate, rad/s, positive counter-clockwise |

`angular.x`/`angular.y` are read only while the aircraft is *not* flying, where
they become roll/pitch angles or horizontal velocity targets depending on
`/simple_drone/dronevel_mode`. A flight command must leave them alone;
`adapters/velocity_command.py` does, and there is a test for it.

The plugin projects the measured velocity through a **yaw-only** quaternion
before comparing, so `cmd_vel` is in the heading frame rather than the fully
rotated body frame. While the aircraft is level — the only attitude it commands
— those are the same thing.

Five latches, all in `adapters/velocity_command.py` as `LATCHES`:

| topic | type | effect |
|---|---|---|
| `/simple_drone/takeoff` | `std_msgs/Empty` | **ignored unless LANDED** |
| `/simple_drone/land` | `std_msgs/Empty` | **ignored unless FLYING** |
| `/simple_drone/reset` | `std_msgs/Empty` | back to spawn, PID integrators cleared |
| `/simple_drone/posctrl` | `std_msgs/Bool` | re-reads `cmd_vel.linear` as a **world position** |
| `/simple_drone/dronevel_mode` | `std_msgs/Bool` | `angular.x/y` as velocities vs. angles, while not flying |

Two traps in that table. Takeoff and land are silently dropped from the wrong
state, so a mission that fires takeoff once and assumes it worked has no way to
notice that it did not — gate on `/simple_drone/state`. And `posctrl` is a mode
switch, not a scaling: a stack that leaves it latched and then publishes
velocities flies to the coordinate whose numbers happen to equal its speeds.

There is also no disarm and no failsafe reachable from outside. "Stop" is a zero
twist that must keep being published — the plugin holds the last command it was
given, so publishing *nothing* does not stop the aircraft.

## Telemetry, and the two topics that lie

| topic | type | rate | notes |
|---|---|---|---|
| `/simple_drone/odom` | `nav_msgs/Odometry` | 30 Hz | **use this.** Ground-truth pose; twist correctly in the child frame |
| `/simple_drone/gt_pose` | `geometry_msgs/Pose` | — | correct, but unstamped and frameless |
| `/simple_drone/gt_vel` | `geometry_msgs/Twist` | — | **mis-rotated, do not use** |
| `/simple_drone/gt_acc` | `geometry_msgs/Twist` | — | **mis-rotated, do not use** |
| `/simple_drone/imu/out` | `sensor_msgs/Imu` | 100 Hz | in `simple_drone/base_link` |
| `/simple_drone/sonar/out` | `sensor_msgs/Range` | 30 Hz | 0.02–10 m, looking down |
| `/simple_drone/bumper_states` | `gazebo_msgs/ContactsState` | — | the collision truth an episode is scored on |
| `/simple_drone/state` | `std_msgs/Int8` | — | LANDED / FLYING / TAKINGOFF / LANDING |
| `/simple_drone/cmd_mode` | `std_msgs/String` | — | which `cmd_vel` interpretation is latched |

**`gt_vel` and `gt_acc` are rotated the wrong way, and it is not obvious from
looking at them.** The plugin computes them as
`pose.Rot().RotateVector(world_velocity)` — the body-to-world rotation applied
to a vector that is already in world. The correct call is `RotateVectorReverse`.
The result is a velocity expressed in no frame at all: exactly right while the
heading is zero, and off by *twice* the yaw angle otherwise. So it survives a
straight-line test flight looking perfectly plausible, and silently inverts
after a 180-degree turn. `/simple_drone/odom`'s twist is computed correctly
(`pose.Rot().Inverse().RotateVector(...)`) and is the feedback source for
everything here.

## Cameras

Both front sensors are separate Gazebo sensors on the same link, rendering the
same scene at different rates. There is no calibration run behind these numbers
and there does not need to be: Gazebo renders an ideal pinhole straight from the
SDF, so the model file *is* the ground truth.

| topic | content |
|---|---|
| `/simple_drone/front/image_raw` + `camera_info` | RGB 600x600, 60 Hz |
| `/simple_drone/front_depth/depth/image_raw` + `camera_info` | depth 600x600, 15 Hz, **32FC1 metres**, valid 0.1–10 m |
| `/simple_drone/front_depth/points` | the same depth already back-projected |
| `/simple_drone/bottom/image_raw` | RGB 640x360, 15 Hz, straight down |

Intrinsics for the 600x600 front pair, from `<horizontal_fov>1.3098</horizontal_fov>`
and `<image>600x600</image>` in `sjtu_drone.urdf.xacro`:

```
fx = fy = (600 / 2) / tan(1.3098 / 2) = 390.642735
cx = cy = 600 / 2 = 300
```

which is a 75.05-degree field of view, and — the image being square — 75.05
degrees *vertically* as well. That is an unusually tall frame for a forward
camera; do not assume it crops like an XTEND one.

Four things to carry away, all of them in `config/camera_front_600x600.yaml` and
`config/camera_front_depth_600x600.yaml`:

- **depth is float metres (32FC1), not 16UC1 millimetres.** Code ported from an
  XTEND bag is off by a factor of 1000.
- **the depth far clip is 10 m.** Beyond it the image carries no measurement, so
  a mapper must range-gate at 10 m or it fuses whatever fills the out-of-range
  pixels as if it were geometry.
- **RGB is 60 Hz and depth is 15 Hz.** An RGBD consumer pairs them by timestamp;
  pairing by arrival order silently associates a colour frame with a depth frame
  up to 60 ms older.
- **the image headers say `camera`, and nothing publishes that frame.** Both
  sensors set `<frame_name>camera</frame_name>`, unnamespaced, and no transform
  for it exists. Reading the camera pose out of TF by that name returns nothing.
  Use `simple_drone/front_cam_link` plus `topics.FRONT_CAMERA_OFFSET_FLU` — the
  camera sits 20 cm ahead of the body origin, which is not negligible: it carves
  free space outward from *itself*, so the body origin is the one place it can
  never observe.

The xacro's own comment block above the depth sensor claims the parameters were
"matched to FALCON simulator exactly" at 640x480 / 90 deg / 5 m. They were not —
the tags in the same file say 600x600 / 75 deg / 10 m. Trust the tags.

## Frames, and their three different conventions

Mixing these is the usual first bug on this robot.

- **URDF links** carry the namespace as a prefix: `simple_drone/base_link`,
  `simple_drone/base_footprint`, `simple_drone/sonar_link`,
  `simple_drone/front_cam_link`, `simple_drone/bottom_cam_link`.
- **The odom frames arrive with a leading slash.** The plugin builds them as
  `get_namespace() + "/odom"`, and `get_namespace()` already starts with one, so
  what is on the wire is `/simple_drone/odom`. Strip it before using it as a TF
  frame.
- **The camera frame is not namespaced at all** — see above.

A static transform ties `world` to `simple_drone/odom` with an identity
transform, so world and odom coincide exactly.

`adapters/topics.py` holds every one of these as a constant, and
`tests/test_topics.py` pins them against the literals read out of the plugin
source, so a rename in the simulator's `drone.yaml` cannot drift silently into a
"no data" symptom here.

## The plant this stack inverts

The plugin is four nested PID loops (`drone.yaml`; values as it prints them at
startup):

| loop | kP | kD | limit |
|---|---|---|---|
| roll/pitch | 10 | 5 | 0.5 rad |
| yaw | 2 | 1 | 1.5 rad/s |
| velocity XY | 5 | 2.3 | 2 m/s |
| velocity Z | 5 | 1 | *disabled* (−1) |
| position XY | 1.1 | 0 | 5 m |
| position Z | 1 | 0.2 | *disabled* (−1) |

Airframe: 1.477 kg, `maxForce` 30 N — thrust-to-weight 2.07, comfortable but not
aerobatic. Tilt limit 0.5 rad, yaw rate 1.5 rad/s, body velocity 2 m/s.

What actually matters to a controller is not those gains but the **closed-loop
response they produce**, measured by stepping `cmd_vel` and fitting `odom`'s
twist to a first-order lag behind a transport delay:

| axis | DC gain | delay | time constant |
|---|---|---|---|
| horizontal | 0.998 | 0.181 s | 0.510 s |
| vertical | 1.024 | 0.033 s | 0.409 s |
| yaw | 0.999 | 0.055 s | 0.477 s |

Three readings from that table:

- **All three DC gains are within 2.5% of unity**, so the cascade really does
  close its own velocity loop and a velocity feedforward alone flies the
  aircraft in steady state. What it does not do is do so promptly.
- **The horizontal delay caps the outer loop.** 0.181 s puts the crossover
  ceiling at roughly `1 / (3 × 0.181) = 1.8 rad/s` before phase margin runs out.
  Above that a position loop closed around this plant rings, and the symptom
  reads as "the aircraft is mysteriously always behind" — which invites raising
  the very gain that is already too high.
- **Vertical and yaw answer three to five times sooner than horizontal**,
  because neither waits for the airframe to rotate first. One lag for all three
  axes is sluggish in two of them and aggressive in the third.

These live in `config/airframe.yaml`, not in a constant inside a control module,
and `adapters/plant_config.py` turns them into a core `VelocityPlant`. Every key
is required — a missing one raises rather than falling back to `AxisPlant`'s
generic-quadrotor defaults, because a plant that is half measured and half
assumed is the one failure mode that cannot be spotted in a flight log.

## Gazebo Classic needs a display, even headless

**Gazebo Classic disables every camera sensor when it cannot open a display —
with the GUI off as well.** The log says `Can't open display` and then `Unable to
create CameraSensor. Rendering is disabled.`, after which the drone still flies
and still publishes odometry, so the failure presents as a camera bug rather
than a display one.

A working bring-up must therefore provide an X display. On this machine
`DISPLAY=:1` (XWayland) with `/tmp/.X11-unix` bind-mounted works;
`setup/bringup_world.sh` refuses to start without one rather than letting you
discover it thirty seconds later in the topic list. "Headless" in that script
means no `gzclient` viewer, not no X.

## Bringing a world up

```bash
export SJTU_PROJECT_DIR=/path/to/sjtu_project
export DISPLAY=:1

sparx_agency/robots/SJTU/setup/env.sh            # check the checkout and the image
sparx_agency/robots/SJTU/setup/bringup_world.sh hospital           # gzserver only
sparx_agency/robots/SJTU/setup/bringup_world.sh --gui playground   # with a viewer
```

Flags: `--gui` / `--headless` (default), `--domain <N>` (ROS_DOMAIN_ID, default
20 — a mismatch drops all traffic silently and looks exactly like a simulator
that never started), `--name <NAME>`, `--skip-build`, `--help`. `--help` lists
the worlds actually present in your checkout.

Only two worlds are in the checkout today: **`hospital`** (from
`aws-robomaker-hospital-world`, plus its `hospital_two_floors` /
`hospital_three_floors` variants) and **`playground`** (sjtu's own).
`small_house`, `bookstore` and `small_warehouse` are *not* — they are separate
aws-robomaker repositories. Cloning one next to `sjtu_drone/` makes it appear
with no change to the script.

**Why not the external repo's `run.sh`.** Three reasons, and the first is the
one that matters:

1. It launches `sjtu_drone_bringup.launch.py`, which also starts rviz2, a joy
   node and an xterm teleop — and **the teleop publishes to
   `/simple_drone/cmd_vel`**. Two publishers on the only control input make
   every control experiment unrepeatable, with no warning anywhere.
   `bringup_world.sh` launches the inner `sjtu_drone_gazebo.launch.py` instead:
   `robot_state_publisher` + `gzserver` + optional `gzclient` + `spawn_drone` +
   the `world`→`odom` static TF, and nothing else.
2. It appends `.world` to its argument, so `./run.sh hospital.world` becomes
   `hospital.world.world` and aborts with "world file not found".
   `bringup_world.sh` takes a bare name, tolerates a trailing `.world`, and
   resolves the file itself.
3. It clones and builds `gazebo_ros_2d_map` on the way past.

The workspace is built inside the container on each run (`--skip-build` reuses
`$SJTU_PROJECT_DIR/install`, which is fast and wrong after a plugin edit). The
mount point matches `run.sh`'s on purpose — colcon bakes absolute paths into
`install/`, so building under a second mount point would invalidate a workspace
built under the first.

## Layout

```
robots/SJTU/
  config/camera_front_600x600.yaml         RGB pinhole, derived from the SDF (provenance in the file)
  config/camera_front_depth_600x600.yaml   depth pinhole: same optics, 15 Hz, 32FC1 metres, 10 m clip
  config/airframe.yaml                     mass, saturations, and the MEASURED velocity plant
  adapters/topics.py                       every topic and frame name, composed from NAMESPACE. No logic
  adapters/velocity_command.py             BodyTwistCommand -> Twist fields + the latches. No ROS import
  adapters/plant_config.py                 airframe.yaml -> core VelocityPlant / BodyVelocityLimits
  adapters/gazebo_ros2_ingest.py           legacy RGB->depth->costmap ROS 2 node (see below)
  setup/env.sh                             resolve and validate $SJTU_PROJECT_DIR
  setup/bringup_world.sh                   bring one world up in docker, cleanly
  tests/                                   runs in the plain .venv; no ROS 2, no simulator
```

## `adapters/gazebo_ros2_ingest.py` is not part of this contract yet

It predates the package and does not meet the repo's conventions: it hardcodes
two absolute `/home/<someone>/...` paths as parameter defaults, its `main()`
constructs the node without the two arguments its `__init__` requires, and it
imports `DA3TensorRTModel` eagerly, which makes the module unimportable on any
machine without TensorRT. It is left in place because the mapping wiring in it
is worth salvaging, but nothing here imports it and it should not be treated as
an example. Rewriting it against `adapters/topics.py` is the obvious next piece
of work.

## What's still open

- Nothing flies this platform yet. The contract, the calibrations and the plant
  are in place; the mission-level node that composes them with a core velocity
  servo is not, and belongs under `tasks/`, not here.
- No `config/vla/*.yaml`, per the VLA layering rule — no policy has been pointed
  at this robot.
- The measured plant is from one campaign at one world and one flight envelope.
  It is a first-order-plus-delay fit to a second-order reality, which is the
  right trade for a lead term that cancels the dominant pole, but it has not
  been re-checked at speeds near the 2 m/s ceiling.
- `hospital` is the only large world available locally. `playground` is small
  enough that it exercises the plumbing rather than the navigation.
