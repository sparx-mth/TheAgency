# falcon_sjtu — FALCON flown in the SJTU Gazebo worlds

The third deployment of FALCON in this repo, and the one whose aircraft is
furthest from the other two. Read `tasks/planning/falcon/README.md` for the
ROS1/Docker mechanics that all three share and
`tasks/planning/falcon_pegasus/README.md` for the simulator-side conventions
this package copies.

## What is different here, in one paragraph

`falcon/` flies a real XTEND and `falcon_pegasus/` flies a PX4 SITL aircraft;
both accept attitude or velocity setpoints from an autopilot we control. The
SJTU Gazebo drone accepts **only** a body-frame `geometry_msgs/Twist` on
`/simple_drone/cmd_vel`. Its model plugin owns a complete internal PID cascade —
velocity → tilt → attitude → body torque — and the branch that would read
`cmd_vel.angular.x/y` as roll and pitch setpoints is unreachable while the
aircraft is in `FLYING_MODEL`. There is no attitude port, no thrust port and no
motor port. So `core/control/flatness` and `core/control/thrust_model` have
nowhere to land, and this package flies the `core/control/velocity_servo`
backend instead. **Exercising the attitude path needs PEGASUS, not this.**

## The environment lives in another repo

The worlds, the drone model and its plugin are in `sjtu_project`, a separate
checkout. Nothing here vendors them. Point `SJTU_PROJECT_DIR` at it and use
`robots/SJTU/setup/bringup_world.sh`; see `robots/SJTU/README.md` for the topic
and frame contract and the measured airframe numbers.

Two traps in that external repo, both of which cost a first-run afternoon:

* **Gazebo Classic disables every camera sensor when it cannot open a display**,
  even with the GUI off. The log says `Can't open display:` and then
  `Unable to create CameraSensor. Rendering is disabled.` — and the sim comes up
  looking perfectly healthy with odometry, IMU and sonar, just no images at all.
  A bring-up must provide an X display (bind-mount `/tmp/.X11-unix` and pass
  `DISPLAY`), or run an Xvfb inside the container. `use_gui:=false` is about
  `gzclient`, not about rendering.
* **`sjtu_drone/run.sh` appends `.world` to its argument**, so the command its
  own `launch_all.sh` documents (`./run.sh hospital.world`) becomes
  `hospital.world.world` and aborts. It also starts `rviz2`, a joy node and an
  xterm teleop that publishes to the same `/simple_drone/cmd_vel` this package
  drives — two publishers on the actuation topic. Use the inner
  `sjtu_drone_gazebo.launch.py`, which is the clean bring-up.

## The camera intrinsics were wrong, and it is a map bug not a flight bug

This is the single most consequential thing found while bringing the stack up,
and it is worth stating plainly because it will be blamed on the controller.

The previous stack told FALCON `fx = fy = 320, cx = 320, cy = 240` at
`640 × 480`. The sensor it was reading is `600 × 600` at
`horizontal_fov = 1.3098 rad`, i.e.

```
fx = fy = (600/2) / tan(1.3098/2) = 390.64        cx = cy = 300.5
```

verified live off `/simple_drone/front_depth/depth/camera_info`. Two independent
errors follow, and both corrupt the **map**:

* `fx` 18% low → a depth ray back-projects as `X = (u − cx)·Z/fx`, so every
  obstacle is placed about **22% further out laterally** than it is. Walls are
  mapped wider apart than they are, and the free space between them is fiction.
* `cy` 60.5 px off, in a frame FALCON also believed was 480 rows tall → about
  **8.8° of spurious downward tilt** in all reconstructed geometry, so the floor
  climbs and the ceiling descends with range.

A planner handed that map routes confidently through walls that are not where it
thinks they are. If the aircraft is getting stuck on geometry, fix this before
touching a gain — no controller recovers a wrong map, and a tracking number
measured against one means nothing.

`adapter/launch/bspline_follower.launch` publishes the corrected values.

## What this package adds

```
adapter/scripts/bspline_follower_node.py   the only node: plan + state -> Twist
adapter/launch/bspline_follower.launch     the follower, and the corrected camera model
```

The node is thin on purpose: subscribe, convert, one `update()`, publish. The
control law is in `core/control/velocity_servo/` and is unit-tested against a
simulated airframe, so it never needs Gazebo to be exercised.

### Why it reads `/planning/bspline` and not `/planning/pos_cmd`

The previous stack's `cmd_to_vel.py` subscribed to `/planning/pos_cmd`, the
100 Hz point stream `traj_server` produces by sampling the B-spline. That stream
arrives 15–35 ms stale, cannot be evaluated at any other instant, and carries no
jerk. Worse, `traj_server` will happily keep republishing the final point of a
trajectory whose planner died minutes ago, so the stream's liveness proves
nothing.

`/planning/bspline` carries the whole curve — control points, an explicit knot
vector, a separate degree-3 yaw spline, `yaw_dt` and a `start_time` — so the
follower evaluates position, velocity, acceleration, jerk, yaw and yaw rate on
its own clock at whatever instant the controller asks for, exactly. The
acceleration is what the inverse-plant lead term needs, and it is the reason the
curve is carried rather than its samples.

`/planning/replan` is treated as a **control input**, not telemetry: `1` means
`safetyCallback` found the executing trajectory in collision, and a follower
that ignores it keeps carrying the aircraft's momentum into the obstacle FALCON
has just found. `2` means the mission is over.

### Tracking diagnostics

The node publishes `~tracking` as a `Float32MultiArray`, positionally:

```
0 position_error_m   1 along_track_lag_m  2 cross_track_error_m  3 yaw_error_rad
4 world_vx           5 world_vy           6 world_vz             7 yaw_rate
8 trajectory_id      9 reference_time_s  10 saturated           11 holding
12 past_end
```

A flat array rather than a custom message so nothing has to be built into the
container to record it. `along` and `cross` always satisfy
`along² + cross² == gap²`, so a gap is fully attributable — and the two halves
are not equally dangerous. Being late is benign; being sideways is what hits
walls.

## Status

Verified: the Gazebo world, drone, all sensors and the actuation path come up
and fly; the airframe's velocity response has been measured (below); the control
law is unit-tested and runs unmodified on Python 3.8 / numpy 1.17 inside
`falcon-ros-custom:v1`; and `rig/track_bspline.py` flies a FALCON-shaped
B-spline on the real aircraft through the real `VelocityServo`.

**Measured on the aircraft** (playground world, attitude confirmed upright
before and throughout the run, real-time factor 1.00, 0.6 m/s plans, circle
route — radius 1.43 m, *derived* from the requested speed, not chosen — mean
over the run after the settling window is discarded):

| settle window | configuration | mean gap | mean cross-track | max yaw error |
|---|---|---|---|---|
| 1 s | inverse-plant lead | 0.188 m | 0.079 m | 7.9° |
| 1 s | P + feedforward | 0.358 m | 0.199 m | 16.1° |
| 5 s | inverse-plant lead | **0.091 m** | **0.054 m** | 7.6° |
| 5 s | P + feedforward | 0.257 m | 0.202 m | 9.5° |

Two things to read off that table.

**The lead term is worth about 2.8x on mean gap and 3.8x on cross-track** once
five seconds of acquisition are discarded — and cross-track is the half that
hits walls. The ordering matches the simulated result in
`core/control/README.md`, and now so does the magnitude to within about 3x.

**The error falls as more of the start is discarded** (0.188 m → 0.091 m for the
lead configuration). That is the signature of a start-from-hover transient being
excluded, which is what a settling window is for. It is the *opposite* of
divergence.

Both configurations are flown by the same rig, in the same world, on the same
route, back to back, so the *comparison* is the solid part. The absolute numbers
are one route in one world at one speed: 0.09 m of mean gap on a 1.43 m circle
is good tracking for this airframe, but it is 6% of the circle's radius, and it
is not a claim about a corridor, a corner or a replan.

### The "progressive divergence" result was a capsized airframe

An earlier campaign reported that the aircraft diverges around the circle —
0.173 m of mean gap with a 1 s settling window against 1.612 m with a 6 s one,
almost all of it cross-track — and hypothesised that the plugin closes its
velocity loop in the **body** frame while this controller reasons in the world
frame, so a continuous yaw rate rotates the plant out from under a per-axis
scalar plant model.

**That measurement was invalid and the hypothesis was never tested by it.** The
aircraft had capsized: measured roll 81.7°, pitch −68.4°. The plugin thrusts
with `AddRelativeForce` along **body z**, so a capsized model points its whole
thrust sideways and cannot climb, translate or even yaw — while continuing to
report `FLYING_MODEL` and healthy 30 Hz odometry. Nothing in the topic list says
anything is wrong. What the rig then measured was not tracking error; it was an
aircraft on its back, and the growth with window length was simply more of the
run spent immobile.

The re-measurement above was flown on a clean world with the attitude checked,
and it shows no divergence at all. There is **no known frame-rotation problem**,
and there is no evidence for one either way: the body-frame hypothesis is
neither confirmed nor refuted, it is untested. Do not carry it forward as a
finding.

`rig/track_bspline.py` now refuses to start a run on a capsized airframe and
aborts mid-run if the aircraft goes past ~35° of roll or pitch, which is where
the plugin's own attitude clamp sits. Any tool that flies this drone needs the
same check.

### Reading a bad run

Three failures here produce numbers that look exactly like a mistuned
controller, and all three are cheap to tell apart once you know the signature.
Check these before touching a gain.

* **Blocked** — the along-track lag grows **linearly**, cross-track stays near
  zero, and the command sits saturated. The aircraft is being asked to go
  somewhere it physically cannot. Flown open-loop in the playground this walked
  the drone eleven metres from spawn into the scenery, where it made +0.007 m of
  travel forward against −1.99 m backward on the same commanded speed. Fixed by
  making every rig route a closed loop that returns home.
* **Capsized** — **no motion on any axis, including yaw**, while `navi_state`
  reads `FLYING_MODEL` and odometry keeps arriving at a healthy 30 Hz. The
  giveaway is yaw: a controller can be mistuned in translation, but nothing that
  is merely mistuned stops the aircraft rotating too. Read the attitude off
  `/simple_drone/odom` and restart the world; a capsized model never recovers.
* **Sim time** — the wall clock outruns the physics. A control loop timed off
  the wall clock under a real-time factor below 1 advances the plan faster than
  the aircraft can be simulated, and the resulting lag is a pure artifact of the
  clock. The **hospital world runs at roughly half real time** (depth measured
  at 7.5 Hz against a 15 Hz nominal), so a plan there advances twice as fast as
  the aircraft. Benchmark in the playground, watch the reported real-time
  factor, and drive the loop off the ROS clock.

### Not yet done

**FALCON has never been flown in this simulator.** Everything above is our
follower carrying a FALCON-*shaped* B-spline that the rig authored; no frontier
has been chosen, no map has been built, and the corrected intrinsics have not
been exercised by the real mapper. A full exploration flown end to end through
this follower needs the ROS1 bridge configured (CycloneDDS with shared memory
disabled on both sides -- the sim image does not currently ship
`rmw_cyclonedds_cpp`, and the bring-up script leaves the default FastRTPS in
place), a roscore, and FALCON's own planner launch ported across with the
corrected intrinsics above.

The rig's other routes -- `straight`, `corner` and `slalom` -- have **not** been
re-flown since the capsize was found, so there are no trustworthy numbers for
them and the ones this file used to quote have been removed rather than
softened. The circle is the primary benchmark anyway: it is the only route that
asks the airframe for nothing infeasible, so what is left over is the
controller.

### Measured plant (`/simple_drone`, step response on the odometry)

| axis | DC gain | transport delay | time constant |
|---|---|---|---|
| horizontal (vx, vy) | 0.998 | 0.181 s | 0.510 s |
| vertical (vz) | 1.024 | 0.033 s | 0.409 s |
| yaw rate | 0.999 | 0.055 s | 0.477 s |

Re-measure these if the plugin's gains in `sjtu_drone_bringup/config/drone.yaml`
change, or if the world is heavy enough to push the real-time factor below one.
