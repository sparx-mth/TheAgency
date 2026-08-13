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

## Running it, with the map on screen

```bash
# 1. the world (external repo; drone spawns at (1,1,0) in hospital)
export SJTU_PROJECT_DIR=~/GIT/sjtu_project
bash sparx_agency/robots/SJTU/setup/bringup_world.sh --skip-build hospital

# 2. FALCON + bridge + RViz (in another terminal)
./run_falcon_sjtu.sh hospital
```

The warehouse is the same two commands with the **world and the map config
named differently** — `no_roof_small_warehouse` is the Gazebo world,
`warehouse` is the FALCON map config:

```bash
bash sparx_agency/robots/SJTU/setup/bringup_world.sh --skip-build no_roof_small_warehouse
./run_falcon_sjtu.sh warehouse
```

`rig/campaign_run.sh` handles that split itself via `WORLD=`; see its header.

**`--skip-build` reuses the installed simulator, including its URDF.** It is the
fast path and the rig uses it, but after any edit to `sjtu_drone.urdf.xacro` it
flies the *previous* aircraft — wrong camera intrinsics included, silently. See
"Cameras" in `robots/SJTU/README.md` for what that does to the map, and check
`camera_info` against `bspline_follower.launch` before trusting a run.

> **Everything measured on the warehouse before 2026-08-13 was flown through a
> stale 640x360 / fx 185.69 depth camera** while the stack was configured for
> 600x600 / fx 390.64. That includes the runs behind `config/warehouse.yaml`'s
> box limits and its notes on contacts and "pocket-land" — those conclusions
> were drawn against a map whose obstacles were pulled 2.1x toward the optical
> axis, so re-derive them before treating them as geometry.

### On a machine that has never run this stack

Three images and one world repo, none of which live in TheAgency. Bringing them
up from nothing on PCN87653 on 2026-08-13 took four fixes, all of them in the
external repos, and all of them the kind that only appear on a *fresh* build —
an image that was `docker commit`ed live carries the workarounds invisibly:

| what | why it does not just work |
|---|---|
| `aws-robomaker-small-warehouse-world` | archived repo: the default branch has **only a README**, the worlds are on `ros1` |
| `sjtu_drone_sparx:humble` | `docker build -t sjtu_drone_sparx:humble sparx_agency/robots/SJTU/setup` (~10 s). Without it bring-up silently falls back to Fast DDS and the bridge sees nothing |
| `ros1_bridge:noetic-foxy` | needs `ros-{noetic,foxy}-gazebo-msgs` at **build** time or `bumper_states` cannot bridge, and `cyclonedds_localhost.xml` baked in — `run_falcon_sjtu.sh` points `CYCLONEDDS_URI` at a path only `run_bridge.sh` mounts |
| `falcon-ros-custom:v1` | `docker build sjtu_project/falcon_docker` — see `patches/README.md`, plus the three build breaks below |

The three that stop `catkin_make` in a fresh `falcon_docker` build, in the order
they fire:

1. `falcon_adapter/CMakeLists.txt` lists `scripts/plot_trajectory_ros1.py`,
   which has never been committed to `sjtu_project`. `catkin_install_python`
   fails the *configure* step on any checkout that does not also carry that
   untracked file.
2. `uav_simulator/so3_disturbance_generator` `add_dependencies` on
   `${PROJECT_NAME}_gencfg`, but no FALCON package calls
   `generate_dynamic_reconfigure_options`, so the target cannot exist.
3. `camera_sensing/mesh_render` needs Open3D's GUI/Filament API, and step 5
   builds Open3D with `-DBUILD_GUI=OFF` deliberately — so it fails at *link*
   time, after the whole workspace has compiled.

2 and 3 are now seeded in `ignore_cuda_pkgs.sh` for every platform (they were
previously only excluded on the `WITH_SIM=0` Jetson path). Neither package is
launched by any of our three deployments.

`run_falcon_sjtu.sh` opens FALCON's own RViz view automatically whenever a
display is present (`RVIZ=0` to suppress, e.g. for soaks): the voxel map,
frontiers, hgrid, the planned and travelled trajectories, and the drone itself.
RViz is started before the bridge, so the view is up before the first depth
frame is mapped — a fresh run shows the map growing from the very first voxel.
The drone model is published by FALCON's `odom_visualization`, which
`exploration.launch` runs fed from `/odom_world` (FALCON's rviz config expects
it, and nothing else in this stack would publish it — without it the map grows
with no aircraft in view).

## Why it crawls near obstacles: ask the attribution log, do not guess

Five mechanisms can slow or stop this aircraft — depth brake, map-gate scaling,
proximity governor, personal-space retreat, creep — they are checked in
sequence, and each was added for a different incident. From outside they are
indistinguishable: the aircraft is simply slow, and the log only ever shows the
loudest one (a retreat) rather than the one that actually binds. So the
follower records exactly ONE binding limiter per tick, with the fraction of the
planned speed that survived it, and prints a histogram every
`limiter_report_s` (15 s):

```
[follower] limiter share over 15s: proximity_cap 71% (x0.49), following 29% (x1.00)
```

Read that as: for 71% of ticks the proximity governor was the binding
constraint and it allowed 49% of the planned speed. `~limiter` publishes the
same name live, one word per tick, for `rostopic echo`.

**That log found two throttles that no amount of watching RViz would have.**

*The proximity governor was fighting the planner.* It capped speed at
`max(0.08, 0.7 * (d_near - 0.30))` — a curve set independently of the clearance
FALCON is asked for. At `safe_distance` 0.55 m it allowed 0.175 m/s against a
0.25 m/s plan: **the aircraft was throttled to 70% while flying exactly where
the planner intended**, and to 32% whenever the soft ESDF penalty let the curve
drift to 0.4 m. The knee is now the airframe radius, where speed genuinely must
be zero (`prox_stop_m` 0.25, `prox_slope` 1.2, `prox_floor` 0.08), so a
correctly flown path is not braked at all.

*Creep never let go.* The concession to a contested spot released on a 20 s
timer alone, so once triggered the aircraft crawled at 0.12 m/s everywhere it
flew next — measured at 48-100% of ticks at 0.22x the plan, one window at
100%. It now also releases spatially, as soon as the aircraft is
`creep_clear_m` (2.0 m) from the spot it conceded at; the timer remains the
backstop for a concession that never gets anywhere.

Measured across the same world and start, fresh map each run:

| configuration | FINISH | coverage | retreats | binding limiter |
|---|---|---|---|---|
| `safe_distance` 0.55 + vertical escape | 120 s | 154.2 m³ | 9 | `proximity_cap` 43-71% (x0.50) |
| + governor knee at the airframe radius | ~150 s | 154.7 m³ | 6 | `creep` 48-100% (x0.22) |
| + spatial creep release | **120 s** | **154.6 m³** | **5** | **`following` 83% (x1.00)** |

Zero bumper contacts in every one of them.

### Clearance is 3D, so the CEILING is a clearance parameter

FALCON's ESDF is a genuine 3D Euclidean distance transform (three separable
passes over z, y, x in `esdf.cpp`), swept over the map box and clamped by
`boundIndex` to the **vbox**, not the flight box. The floor is in it — measured
41,446 occupied voxels at z≈0, half the map — so `safe_distance` really is
enforced downward as well as sideways.

Which makes the flight box ceiling part of the clearance arithmetic, and it is
easy to set it somewhere that quietly forbids every overflight:

| obstacle | top | altitude needed at `safe_distance` 0.85 |
|---|---|---|
| ClutteringA piles | 1.10 m | 1.95 m |
| buckets | 1.41 m | 2.26 m |
| tall C-piles | 1.78 m | 2.63 m |

At the old `box_max_z` 1.8 **none of those fit**. The optimiser cannot satisfy
a constraint the box forbids, so it traded the margin away and skimmed the
tops — which is where the belly strikes came from, and why the route looked
like it had no room underneath it. `box_max_z` is now 2.6.

The old argument for a low ceiling (stay under the 1.96 m shelf tops so
coverage is flown in the aisles rather than over the racking) no longer binds:
the shelf rows are at x 4.28..5.16 and x −6.89..−4.79, both outside
`box_max_x` 3.9 and `box_min_x` −4.4, so the aircraft cannot reach them at any
altitude. Raising the ceiling buys headroom over the clutter and nothing else.

Measured, same world and start: coverage 95.5% of the box at 1.8 m against
**97.1% at 2.6 m**, retreats 2 → **1**, and the aircraft using 1.24–2.48 m of
altitude instead of 1.1–1.7. Compare percentages, not cubic metres — raising
the ceiling changes the box volume from 161 m³ to 269 m³.

### Known, characterised, NOT fixed: the depth brake over-cuts sideways

`f = v_allow / command.vx` is derived from the FORWARD axis alone and then
multiplies the whole 2D vector:

```python
f = v_allow / command.vx
command = dataclasses.replace(command, vx=command.vx * f, vy=command.vy * f, ...)
```

The forward component lands exactly on `v_allow`, which is right. The lateral
component is scaled by that same ratio, which is not: the depth corridor is a
forward tube 0.70 m wide, so an obstacle inside it says nothing about moving
sideways — and sidestepping is precisely how this aircraft passes a shelf end.
The over-cut is 1/cos(beta) off the nose: 1.41x at 45 degrees, 2x at 60.

It is left alone deliberately. The one-factor-on-every-component rule is load
bearing for the stage below it — the map gate sweeps its corridor from the
WORLD vector, so scaling body and world components differently mis-aims it
(see the comment at that call site). Fixing this properly means limiting the
forward axis only and recomputing the world vector from the corrected body
command, then re-verifying the gate's corridor still points where the aircraft
is actually going. Worth doing; not worth doing untested.

The other four limiters enforce four different standoffs against the same wall
— depth veto 0.55 m (was 1.05), map gate 0.35 m, bubble 0.28 m, governor knee
0.25 m — and the binding envelope is the MAX of them, not any single design
value. Keep them within sight of `safe_distance` (0.55 m) or the follower
starts refusing curves the planner was asked to produce.

## Getting unstuck: the escape is three-axis, not two

A wedge in this world is almost never a contact — across every warehouse run on
2026-08-13 the Gazebo bumper reported **zero** contacts while the follower
retreated dozens of times. What fires is the map gate's personal-space bubble,
and the aircraft was trying to resolve it by backing out *in the plane*. When
the way back is walled too, that fails, and before the fixes below it failed
silently: the maneuver fell through to FACE and DWELL, logged "retreat done",
and the aircraft had not moved a centimetre.

Two things resolve it, in order:

1. **A back-out that never moved is recorded as a block** and feeds
   `_note_block_episode` — the same escalation the map gate uses — so a second
   one within 1.5 m concedes to the planner and creeps across at `creep_speed`
   instead of looping forever.
2. **A pinned aircraft climbs.** `voxel_brake_gate` keys occupancy into z
   LAYERS and `_layers_for_z` only tests the layers the airframe can strike at
   its current altitude, so gaining height genuinely changes what blocks it: a
   1.10 m clutter pile stops obstructing an aircraft at 1.6 m. Horizontal
   retreat was one axis of a three-axis escape and the only one being used.
   `escape_climb_s` (4 s), `escape_climb_speed` (0.4 m/s) and
   `escape_climb_max_z` (1.65 m, **under** the 1.8 m box ceiling — climb out of
   the box and no frontier is reachable from up there); `escape_climb_s:=0`
   restores horizontal-only behaviour.

Measured on the warehouse, fresh map each time, same world and start:

| configuration | time to FINISH | coverage | retreats | contacts |
|---|---|---|---|---|
| ghost-start fix only | ~420 s | 153.9 m³ | 15 | 0 |
| + `safe_distance` 0.35 | ~250 s | 149.6 m³ | 9 | 0 |
| + wider frontier clearances, `max_vel` 0.4 | ~470 s | 153.4 m³ | 22 | 0 |
| **+ vertical escape** | **120 s** | 151.0 m³ | **8** | 0 |

The third row is the negative result worth keeping: raising
`frontier_min_occ_clearance` to 0.70 and `candidate_rmin` to 1.5 to stop FALCON
choosing viewpoints in tight gaps made it *worse* — it refuses viewpoints in
the aisles, so early mapping crawled and retreats went up. The flight box in
`config/warehouse.yaml` remains the tool for keeping the aircraft out of a
region; the frontier clearances are not.

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
