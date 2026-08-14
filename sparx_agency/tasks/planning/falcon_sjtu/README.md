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
adapter/scripts/bspline_follower_node.py   plan + state -> Twist
adapter/scripts/sensor_pose_node.py        odom -> camera pose for the mapper
adapter/scripts/mission_watchdog_node.py   is this run still worth its clock?
adapter/launch/bspline_follower.launch     the follower, and the corrected camera model
adapter/launch/exploration.launch          FALCON, the adapters and the watchdog
```

The nodes are thin on purpose: subscribe, convert, one call, publish. Everything
they decide with lives ROS-free in `core/` and is unit-tested without Gazebo:

| node | the logic it wires | tests |
|---|---|---|
| `bspline_follower_node` | `core/control/velocity_servo/` | against a simulated airframe |
| | `core/planning/safety/clearance_envelope.py` | a warehouse aisle and a hospital doorway, one config |
| | `core/planning/safety/voxel_brake_gate.py` | synthesised occupancy |
| | `core/planning/local_planners/corridor_centering.py` | analytic corridors |
| `mission_watchdog_node` | `core/planning/exploration/progress_monitor.py` | every stuck shape, and the healthy mission it is confused with |

**A new node is not free: add it to the per-file mount loop in
`run_falcon_sjtu.sh`.** An unmounted file silently runs whatever is baked into
the image, or nothing at all. `rospack find falcon_adapter` resolves to
`/catkin_ws/src/falcon_adapter`, so a mounted, executable script is found
without an image rebuild; `run_falcon_sjtu.sh` does the `chmod +x` itself.

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

**`rig/both_worlds.sh` is the acceptance test**, and it is the one to run rather
than a single-world soak. It flies the hospital and the warehouse back to back
in one session, with the same binary and the same parameters and no editing in
between, so a change that fixes one world by breaking the other fails there. It
takes no per-world arguments and passes none: the only thing that differs
between its two runs is the map config naming the building's own geometry.
`rig/soak.sh` remains for repeated runs of a single world.

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

## One configuration, two worlds

The two Gazebo worlds looked like they wanted opposite tunings, and for a while
this package shipped opposite tunings: `safe_distance` 0.85 in the warehouse
against 0.45 in the hospital, on the belief that warehouse aisles are ~1.4 m
wide and hospital doorways 0.9 m. Improving one world then regressed the other,
which is the signature of a parameter standing in for something that was never
measured.

**Both beliefs were wrong, and in the same direction.** Measured off the
collision meshes:

| | measured | previously believed |
|---|---|---|
| hospital doorways | 18 at **0.930 m**, 8 at 1.500 m, every lintel at **2.350 m** | 0.9 m, lintels ~2.0 m |
| warehouse shelf aisles | **0.909 / 0.916 / 0.942 / 1.035 / 1.044 / 1.216 m** | "~1.4 m" |
| warehouse tightest clutter slots | **0.81 m** and 0.95 m | "1.4 m gaps" |
| drone collision mesh | **0.52 m across the arms, 0.63 m corner to corner**, 0.11 m tall | "0.25 m radius" |

The warehouse's aisles are the same width as the hospital's doorways. There was
never a conflict between the worlds to trade off — there was a geometry error.
`safe_distance` is now **0.45 in both**, and the reason it sits exactly at the
doorway half-width rather than under it is in `run_falcon_sjtu.sh`.

The warehouse numbers were wrong for a reason worth knowing: the ShelfD/ShelfE
collision DAE carries a 90-degree `<node><matrix>` that Gazebo applies and a
naive vertex-extent parse does not. Skipping it rotates the whole rack and
quotes the mesh's half-length as a height. The units actually run **east-west**
at x 2.772..6.691 with decks solid to **2.643 m** — so the old `box_max_x` 3.9
put 1.128 m of every shelf *inside* the flight box, under a `box_max_z` of 2.6
that was 4 cm below their tops. The aircraft was being invited in among the
racking with no way to climb out: the "illogical route between crowded shelves"
this config exists to forbid, written into the config.

The hospital box had the opposite fault. It was x,y ±20 while the building runs
x ±12.44 and **y −34.94 to 20.94**: it claimed 7.6 m of solid ground beyond each
side wall and cut the south wing off entirely. About a quarter of the hospital
was outside the volume FALCON is allowed to plan in, so its frontier finder
could not see it and its coverage tour could never route there. The baseline run
stalled at (−4.3, −18.0), two metres short of that face.

## Reflexes are measured against the plan, not against a constant

The follower carries protective reflexes the planner does not: a personal-space
bubble, a speed governor keyed on the nearest obstacle, a depth veto. Every one
of them used to be an absolute distance, and an absolute distance is the wrong
shape for this problem, because **the room a correctly flown aircraft has is a
property of the corridor.** The same constants leave a warehouse aisle alone and
make a hospital doorway unflyable; tuning them down re-opens the warehouse
contacts they were raised to stop.

`core/planning/safety/clearance_envelope.py` replaces the constant with a
comparison. The reference point on FALCON's curve is itself a clearance
measurement — its distance to the nearest occupied voxel is the room the planner
believed it had and chose to use — so the follower asks the only question that
transfers between worlds:

> **Is the aircraft closer to something than the plan is?**

If it is not, no proximity reflex may fire, whatever the absolute distance. If it
is, they fire in proportion to the **deficit**: how much of the plan's own margin
has been spent, which is the same number in both buildings. Underneath sits one
absolute, `hard_floor_m` 0.30 — the distance at which the airframe is about to
touch something — and that is never relaxed for any plan.

Two consequences are worth stating because they are what actually changed:

* **A hard stop is not a breach.** In a 0.90 m opening the plan holds 0.45 m, so
  the floor is reached after only 0.15 m of drift — routinely, on a pass the
  aircraft is entitled to make. Reading that as a breach retreated the aircraft
  1.3 m back out of the door, whereupon it replanned the identical curve and did
  it again. Stopping is right; reversing out of the building is not. In a 1.4 m
  aisle the same 0.30 m means 0.40 m of margin has been thrown away, and there
  the breach still fires. Same code, same threshold, opposite verdict, because
  the plan is different.
* **The inferred wedge is gated; the bumper is not.** `_is_wedged` was the
  dominant backwards path and the only one the trust rule did not cover — it is
  evaluated several branches before `on_plan` is computed, so an aircraft
  deliberately crawling through a doorway at a brake-limited speed read as
  pinned and was driven 1.3 m back out of it. "Going nowhere" and "going slowly
  on purpose" are the same measurement; only the geometry says which. A real
  contact stays ungated, because a bumper report is not an inference.

### The speed ceiling is derived from the clearance, not chosen

The same principle, applied to speed, and it caught a defect that had nothing to
do with either world's geometry: **the follower was allowed to fly at a speed
whose stopping distance exceeded the margin the planner had reserved.**

The margin is `safe_distance` (0.45) minus the airframe half width (0.26), i.e.
0.19 m. The stopping distance at this airframe's measured 0.30 s of command
latency and 0.8 m/s² of braking is `d(v) = 0.30·v + v²/1.6`:

| v | d(v) | against a 0.19 m margin |
|---|---|---|
| 0.25 m/s (plan) | 0.11 m | inside |
| **0.35 m/s** | **0.18 m** | **at it — this is the ceiling** |
| 0.40 m/s | 0.22 m | over |
| 0.60 m/s (old cap) | 0.41 m | more than twice it |

The servo's per-tick ceiling is `planned_speed + max_overspeed`, capped by
`max_speed_xy`, so at a 0.25 m/s plan the old `max_speed_xy` 0.6 with
`max_overspeed` 0.35 let it sprint to 0.60 m/s to close a lag — flying at 2.4×
the speed FALCON had checked the curve at, spending the whole clearance budget
on stopping distance before anything had gone wrong. Every contact in the first
acceptance attempt was at **0.48–0.59 m/s**, all of them the aircraft
accelerating into an obstacle the map did not yet have.

And this was not an edge case the aircraft touched occasionally. Run 003's
commanded speed was **p50 0.45, p90 0.60, p99 0.60 m/s**, with the servo
saturated on **62%** of its following ticks: the old cap was not a ceiling the
aircraft approached, it was the speed it flew at. `analyze_run.py` now prints
that percentile and the stopping distance it implies, so the invariant is
checkable from any run rather than being an argument about defaults.

**But bounding it the obvious way broke something else, three times, and that is
the more transferable lesson.** `max_overspeed` is not a cruise setting —
`servo.py:320` passes `planned_speed + max_overspeed` to `limit_velocity` *as*
the horizontal cap — so it is the **correction budget**: everything the position
loop has above the feedforward, which is what closes a cross-track error.
Shrinking it to hold the ceiling down removes the authority to correct.

What decides whether an aircraft can thread a 0.93 m opening is not its speed.
It is **correction per metre travelled** — how much cross-track it can take out
before it arrives — and that is `max_overspeed / max_vel`:

| configuration | plan | ceiling | correction | **ratio** | stopping | outcome |
|---|---|---|---|---|---|---|
| run 003 | 0.25 | 0.60 | 0.35 | **1.40** | 0.41 m | finished the world; contacts at 0.48–0.59 m/s |
| ceiling capped, plan kept | 0.25 | 0.35 | 0.10 | 0.40 | 0.18 m | stalled at a doorway, 670 m³ |
| ceiling capped, plan 0.20 | 0.20 | 0.35 | 0.15 | 0.75 | 0.18 m | stalled at a doorway, 297 m³ |
| ceiling capped, plan 0.20, floor lifted | 0.20 | 0.35 | 0.15 | 0.75 | 0.18 m | stalled at a doorway, 211 m³ |
| **current** | **0.15** | **0.35** | **0.20** | **1.33** | **0.18 m** | **hospital CLEAN: finished, 0 contacts** |

Three consecutive stalls, all at 0.93 m doorways, all with **zero contacts** —
the aircraft never hit anything, it simply could not centre itself before it
arrived. With the ceiling pinned by clearance, the only way to buy back the
ratio is to **lower the plan**, which is what `max_vel` 0.15 does. It restores
run 003's ratio while keeping run 003's stopping distance problem fixed.

Note what the ceiling is *not*: not a comfort setting and not a reaction to a
crash, but an arithmetic consequence of `safe_distance`. And note what the plan
speed is not: not a throughput knob. Together they are one decision with two
constraints — the ceiling bounds what an unmapped obstacle costs, the ratio
bounds what a doorway costs — and moving either alone breaks the other.

### Inside a passage, a wedge is not a wedge

The wedge reflex exists for an obstacle **the map does not have** — something
inside the depth near clip that FALCON planned straight through. Backing out is
right there, because the aircraft is somewhere nobody knew about. In a 0.93 m
doorway the map has both jambs, the aircraft is exactly where it should be and
merely slow, and a 1.3 m back-out only replays the approach: measured in run
003, whose retreat counter reached 35 while it cycled at one doorway with
coverage frozen at 1565.9 m³.

So the reflex is suppressed inside a passage — and getting that test right took
three attempts, each of which is worth recording because each looked correct:

1. **Zero clearance deficit.** With the plan on the centre line of a 0.90 m
   opening, a 0.15 m tracking error is already a deficit. The rule released
   exactly where it was needed.
2. **The plan's clearance is tight.** The clearance at the *reference point* is
   the planner's own statement about the corridor — but it is a statement about
   where the reference is, and the case that matters is precisely the one where
   the aircraft is not there. Measured: the aircraft wedged in a doorway while
   its reference sat in the room beyond holding 0.90 m, so the test read "not a
   passage" at the moment the aircraft was inside one.
3. **Structure on both sides of the aircraft**, which is what it actually
   means. `CorridorCentering.across_width` casts a ray to each side across the
   direction of travel and returns their sum, or `None` when either side is
   open. Under `passage_width_m` (1.20 m, above every opening in either world
   and below any room) the aircraft is in a passage.

It has to be a **ray**, not the clearance field the rest of that class uses.
Clearance is direction-agnostic: probed 0.25 m either side of an aircraft
standing 0.30 m from a *single* wall, both probes return distances to that same
wall (0.05 m and 0.55 m) and their sum reads as a 1.10 m corridor that does not
exist. Only a ray can answer "is there something on the OTHER side".

The suppression is bounded by the mechanisms that can actually resolve a
doorway rather than reverse out of one: a real bumper contact is still ungated,
the map gate's own 4 s hard-block escalation still retreats, and the mission
watchdog still ends a run that has stopped making map.

### Three strikes: when holding still is the action

A bumper contact retreats the aircraft, and nothing stops FALCON re-issuing the
same route. An obstacle standing *in* a doorway — in the measured case a blood-
pressure cart parked in a 0.93 m opening — therefore produces an unbounded
strike-retreat-strike loop: **55 bumper reports on two objects at one hospital
doorway**, the run ending on the confinement watchdog at 216 m³.

Retreating is right the first time and the second. By the third it is evidence
that this approach does not work, and the only actor who can do anything about
that is FALCON. Its dead-end guard retires a viewpoint the aircraft stays within
2 m of for 25 s without reaching — so **holding station is the action**, not a
failure to act. It is exactly the condition the guard watches for, and it turns
a grind into a retired viewpoint and a tour that moves on. Contacts are grouped
by position rather than by object, because the aircraft cannot identify what it
touched and the geometry is what repeats.

**The ordering inside that rule is the whole of its correctness**, and getting
it wrong is instructive. Holding the moment the third contact is detected parks
the aircraft *against the thing it has just hit*: the bumper keeps reporting,
the hold re-arms itself, and it never moves again — measured, 71 bumper reports
and 46 give-ups in one run with the aircraft pinned. The hold is therefore armed
on the third contact but **acts only after the retreat has backed the aircraft
1.3 m clear**, which is also precisely where the guard wants it.

### Flying through the middle of an opening

FALCON's distance cost is soft and evaluated only at control points ~0.5 m
apart, so its curve is *near* the middle of a doorway, not *on* it; the
follower's tracking error lands on top of that. In a 0.90 m door a 0.52 m
airframe has about 0.19 m of budget, so the two errors together are a jamb
strike.

`core/planning/local_planners/corridor_centering.py` probes the clearance field
to either side of the direction of travel and biases toward its peak. Three
properties make that safe to add to a tracked trajectory rather than a fight
with it: it is exactly zero above `centering_engage_m` (so warehouse cruise is
untouched), it is capped at 0.15 m/s, and it **redirects at constant speed**
rather than adding to it — threading a door costs forward progress instead of
buying sideways motion out of the stopping-distance budget the brakes are sized
against.

The estimator is half the difference of the two probes, which is *exact*: a
clearance field is a distance field, so across a corridor it falls 1:1 with
lateral offset and the half-difference IS the error. The obvious alternative — a
parabolic sub-sample peak fit — is biased and biased dangerously, returning
`-d·e/(d − e)`, which asks for 0.30 m of correction to fix a 0.10 m error at a
0.15 m probe. Parabolic fitting assumes a smooth quadratic peak; a corridor's
clearance field has a *corner* at its centre line.

The centring also survives a hard block, and it is the only thing that can end
one: a dead halt in a doorway resolves in exactly one way — the aircraft moves
back toward the middle — and zeroing every horizontal axis removes the one
motion that would clear it.

## Where FALCON thinks it is, versus where the aircraft is

FALCON does not replan from odometry. It derives each replan's start point by
evaluating the *previous* trajectory at the current instant
(`exploration_fsm.cpp`), and `t_r` saturates at the curve's duration — so every
second this stack spends braked, held or retreating opens a gap between the
planned world and the real one, and the next curve is then anchored in the
planned one. `falcon_replan_from_pose.patch` corrects the origin to odometry
once that gap passes `/fsm/replan_from_pose_drift`.

That threshold was 1.5. It ends the runaway it was written for (gaps of 5.4 m),
but it also *licenses* 1.5 m as normal, and 1.5 m is not a constant cost across
worlds: in a 1.4 m aisle the optimiser absorbs it, and in a 0.90 m doorway it is
three times the whole clearance budget, spent before the aircraft has moved. It
was measured **permanently saturated** in the hospital — 82 corrections in one
run, every one of them 1.5–2.1 m.

It is now **0.40**, the follower's own acquisition tolerance: under it the
previous curve's endpoint is the better origin (it joins the trajectory being
flown with no velocity step), over it the curve is a fiction and odometry wins.
`mission_watchdog_node.py` records the gap per plan in `progress.jsonl`, so it
is a number in every run rather than something to be inferred.

## The mission watchdog: never burn wall-clock on a doomed run

An exploration fails in ways every individual node reads as success. The FSM is
in `EXEC_TRAJ`; the follower is tracking; the aircraft is moving; and the map has
not gained a voxel in four minutes because the drone is orbiting a room whose
only exit is a doorway it cannot thread. The two facts that settle it — where the
aircraft has been, and how much map that bought — were not in the same place
anywhere in the graph.

`adapter/scripts/mission_watchdog_node.py` puts them there, judging with
`core/planning/exploration/progress_monitor.py`:

| signal | what it catches | why not alone |
|---|---|---|
| **confinement radius** — smallest disc containing a 120 s window of positions | orbiting a mapped region; doorway loops | fires on a legitimately slow sweep of one crowded room |
| **coverage growth** — m³/min over the same window | a mission that has stopped learning | fires during a long transit through mapped corridors |
| both at once | the actual failure | — |
| net displacement | pinned against a wall | a doorway loop has plenty of it |
| wall clock | the mission is simply over | says nothing about why |

Confinement is a **two-stage escalation**, and that is the point: a
confined-and-barren window is first a `/mission/nudge`, which the follower
answers with a fresh survey turn (rebuilding the local map and re-arming
FALCON's frontier finder on a region it has stopped seeing frontiers in), and
only becomes `/mission/abort` if the nudges do not restore growth. A watchdog
that can only kill wastes every run it fires on.

It earns that on real runs. Measured in an acceptance flight: coverage flat at
229.6 m³ with growth at 0.09 m³/min in the operating wing, **three nudges**, and
the mission came back — 320.2 m³ and 22 m³/min four minutes later, having moved
from (2.2, −17.3) to (6.4, −23.5). An abort-only watchdog would have thrown that
run away.

The node **declares; it never kills.** `rig/campaign_run.sh` greps the
`[watchdog] MISSION ABORT` banner and ends the run, so an operator running the
stack interactively is not torn down under their own session. The harness keeps
its own coarser watchdogs underneath, with longer caps, for the cases the node
cannot cover (it being disabled, failing to start, or its container dying).

### A planner respawn wipes the map, and every progress watchdog has to know

FALCON's `exploration_node` aborts intermittently and roslaunch respawns it with
an **empty map** that then rebuilds from scratch. Coverage does not dip on that
event; it falls off a cliff, measured 728 → 271 m³ in one acceptance run.

**It is not (only) the LKH solver, and this file said it was.** The measured
stack trace, from a run that aborted at t = 481 s, is:

```
ExplorationFSM::visualize()
  → PlanningVisualization::drawBspline(NonUniformBspline …)
    → NonUniformBspline::evaluateDeBoorT(double const&)
      → NonUniformBspline::evaluateDeBoor(double const&)
        → __assert_fail   (/usr/include/eigen3/Eigen/src/Core/Block.h:120)
```

That is an Eigen block assertion inside the **RViz visualisation path**, firing
when the B-spline's span index runs past its own control points. It is not a
planning failure at all — the FSM calls `visualize()` every cycle, and drawing
one malformed curve kills the process, and with it the map, because on this
stack `exploration_node` owns the voxel grid too.

**The cause is FALCON re-timing our trajectories onto a speed we do not fly, and
it is fixed in `falcon_slow_traj_rescale.patch`.** The line above the assert in
the log is the tell, and it was there all along:

```
[FSM] Slow trajectory detected, duration: 25.16, length: 3.89
[FSM] Avg position velocity: 0.15, avg yaw velocity: 0.00, lengthen ratio: 0.08
```

After each plan the FSM rescales any trajectory averaging under 0.5 m/s so that
it would average a hardcoded **2.0 m/s**, yaw against 1.57 rad/s, via
`ratio = 1.0 / min(1.57/avg_yaw_vel, 2.0/avg_pos_vel)`. Two things follow. On an
aircraft configured to cruise at 0.15 m/s that ratio is a **compression of every
plan** — 324 rescales in one 16-minute mission, ratios clustered at 0.08–0.25,
each one handing the follower a reference four to thirteen times faster than the
limit the optimiser had just been given. And a trajectory that does not turn has
`avg_yaw_vel == 0`, so `1.57 / 0` is an infinity that `std::min` does not
propagate the way this code assumes; the non-finite ratio re-times the knot
vector into the shape that trips the assert. Both defects live in the same four
lines.

The patch takes the targets from parameters, ships them set to the configured
cruise (so a trajectory already flying at the requested speed is left alone),
permits the rescale to stretch time but never to compress it, and refuses to
divide by a stationary axis. It also bounds the span search in
`NonUniformBspline::evaluateDeBoor` by both the knot count and the control-point
count, so that no future malformed spline — whatever produces it — can abort the
node again. See `patches/README.md`.

**Two earlier diagnoses in this file were wrong; both are recorded because the
evidence that refuted them is reusable.** The first blamed LKH. The second, after
the stack trace ruled LKH out, blamed an endpoint off-by-one in the visualiser's
sampling loop and declared the crash untunable from outside the image — wrong on
both counts: the sampling loop is bounded, and the image has had a patch
mechanism (`patches/`) the whole time. A third theory, that the curves were
degenerate because FALCON plans to viewpoints the aircraft is already standing
on, was tested by raising `min_candidate_dist` from 0.5 m to 1.2 m and refuted
by a **byte-identical** recurrence. That setting is kept — refusing a viewpoint
the aircraft is standing on is right on its own terms, and 1.2 m is well clear of
the 0.6 m the dead-end guard calls "at target" — but it was never the fix.

The recovery hardening those theories motivated is kept as well, since a crash
from any other cause still wipes the map: the A\* node pool, the blacklist
non-inheritance and the re-survey cap below. What changes is that the crash is no
longer weather to be endured.

Stalls and crashes correlate strongly — the run that mapped the whole hospital
never stalled badly and never crashed, while every stalling run crashed — and
the mechanism is now legible in both directions: the rescale that eventually
aborts the node is also, every time it fires without aborting, a reference the
follower cannot track.

Any watchdog keyed on "has the map grown since its high-water mark" then kills a
perfectly healthy rebuild, because the mark belongs to the previous incarnation.
That is exactly what happened: the harness's coarse discovery watchdog cut a run
at 420 s of "no new voxels" while the mission's own coverage was climbing at
60 m³/min, its mark stuck at 472,587 occupied voxels from before the crash.

Both watchdogs now re-baseline on a collapse rather than treating it as a stall
— `ExplorationProgressMonitor._track_coverage` restamps on a material drop, and
`campaign_run.sh` does the same when the occupied count halves. The rest of the
stack already survives the respawn: `falcon_deadend_guard.patch` persists its
blocked regions through a rosparam and logs `Restored N blocked region(s) from a
previous incarnation`, and the follower's re-survey turn gives the fresh frontier
finder something to fire on.

**Surviving the crash is not the same as affording it.** Because the map is
wiped, a crash does not interrupt the mission, it *resets* it: measured, one
hospital run losing 728 m³ and another 577 m³, each then needing about ten
minutes to re-earn ground it had already mapped. Two crashes in one run put a
complete map out of reach of any sane time cap — one such run ended at 116 m³
having twice been near 600.

So the crash RATE is a first-class parameter of this deployment, and the lever
is the tour: LKH's failure is in `FindTour → Best5OptMove → Flip_SL` and scales
with the number of cities. `hgrid_cell_size_max` is now **14.0** (from 10.0),
which on the hospital's 24 × 55 m box is 2 × 4 = 8 cells rather than 3 × 6 = 18,
less than half the cities. The tour is less optimal; that is a far smaller cost
than starting over. `falcon_deadend_guard.patch` already removes the other
crash source by enumerating tours of three or fewer cities directly instead of
entering LKH at all.

#### The blacklist must not outlive the map it describes

This is the sharpest correlation in the whole campaign, and it explains why
crashed runs did not merely lose time but **finished early on a partial map**:

| run | planner crashes | outcome |
|---|---|---|
| the one that mapped the building | **0** | finished, 760 m³, 0 contacts |
| every other finishing run | ≥1 | finished at 355.9, 260.8, 116.1 m³ |

`falcon_deadend_guard.patch` persists physics-vetoed viewpoints in the rosparam
`/frontier_finder/blocked_regions_runtime` so they survive a respawn, and its
constructor logs `Restored N blocked region(s) from a previous incarnation`.
Within one incarnation that list is exactly right — the aircraft proved it
cannot reach those places, and the map has not changed. **Across a respawn it is
not**, because the respawn also wipes the map: the rebuilt world arrives with
regions already shadowed in places the aircraft has not re-explored, the
frontier finder runs out early, and FALCON declares a third of a building
complete.

`mission_watchdog_node` therefore deletes that param on every tick
(`clear_blocked_regions_on_respawn`, default true). The in-memory blacklist is a
C++ member, not the param, so blocking works exactly as designed for as long as
the node lives; only the *inheritance* is cut. At most one second of blocks can
survive a crash, which is the width of the watchdog loop.

It also writes `progress.jsonl` at 1 Hz — coverage, confinement radius, growth
rate, plan-origin gap, sensor-pose lag — which is the only artifact that says
*why* a run went nowhere rather than merely that it did. `verdict.json` now
carries `coverage_m3`, `retreats` and `plan_origin_corrections`.

## What it bought, measured

Hospital, same world, same spawn, fresh map each run, one configuration.

**Read the run numbers as a sequence, not as the shipped configuration.** Each
column is the stack as it stood when that run flew, and three changes landed
*after* run 003 — the derived speed ceiling, the ray-based passage test, and the
look-ahead on the vertical escape — each of which was found by reading run 003
or the acceptance attempts that followed it. `rig/both_worlds.sh` is what
measures the configuration as shipped.

| | baseline | box + clearance envelope | + retreat fixes |
|---|---|---|---|
| run | 001 | 002 | 003 |
| verdict | `STALLED_POSITION` | `ABORT_NO_MOVEMENT` | **`FINISHED`** (dirty) |
| coverage | 598 m³ | 476 m³ | **1567.2 m³** (explorable 1583.3) |
| time | 1592 s (cap) | 480 s (killed) | 1646 s, 517 m flown |
| retreats | 60 | 25 | 36 |
| bumper reports / distinct objects touched | 4 / – | 27 / 1 | 15 / **3** |
| plan-origin corrections | 82, all 1.5–2.1 m | 108, capped at 0.43 m | 362, capped at 0.46 m |
| binding limiter, open corridor | `proximity_cap` 77% (×0.36) | `following` 100% (×1.00) | `following` 95–100% |
| furthest south reached | y = −18 (box edge) | y = 1.0 | **y = −33.3** |

**Run 003 is the first time FALCON has ever declared the hospital finished**: its
frontier finder ran out, having retired 107 clusters as unreachable.

Read its 1567.2 m³ against the 1583.3 m³ of explorable volume carefully — they
are close, but they are not quite the same quantity, and the ratio flatters. Of
that explorable volume **80.6 m³ is sealed and can never leave UNKNOWN at any
airframe size**, so the most free space any aircraft could ever report here is
1502.7 m³; the remaining ~65 m³ of the run's figure is observed obstacle
*surface*, which `map_coverage` also counts. The honest statement is not a
percentage but this: **the run ended with one active frontier and 107 retired
ones, having flown 517 m and reached every corner of a 24 × 56 m building** —
x −11.0..12.1, y −33.9..18.8 against walls at x ±12.44, y −34.94..20.94.

Count the contacts by OBJECT, not by bumper report. Gazebo's bumper emits a
fresh `states` entry for every contact point in every physics step, so one
five-second graze along a crate face reads as eight contacts and reads as a
disaster. Run 003 touched **three** things — two AWS warehouse clutter crates
that the hospital world reuses, and an X-ray machine — and all three were first
approaches to geometry not yet in the map. That is the residual near-clip
problem in "Still genuinely open" below, not a controller fault: the depth
camera cannot see the last 0.95 m, and nothing in this stack slows the aircraft
for flying into cells it has never observed.

### What "fully mapped" is, as a number

Computed from the collision meshes rather than asserted — surface-sampled at
0.025 m, shells closed by an inscribed-radius test so wall cavities and crate
interiors count as solid, downsampled to FALCON's 0.10 m grid, then labelled
with 3D 6-connectivity after eroding by the airframe's 0.26 m half width:

| | hospital (z ≤ 1.9) | warehouse |
|---|---|---|
| box | 24.0 × 55.0 × 1.3 m | 7.0 × 16.2 × 1.8 m |
| box volume | 1716.0 m³ | 204.1 m³ |
| **explorable** (box minus solid) | **1583.3 m³** | **192.8 m³** |
| reachable by the drone's centre | 1280.0 m³ | 185.4 m³ |
| sealed, unobservable at any airframe size | 80.6 m³ in 21 pockets | none |

> **The hospital column is for `box_max_z` 1.9 and the ceiling has since moved
> to 1.6** (see "Clearance is 3D" for why: 1.9 sat inside the furniture height
> cluster and guaranteed skimming). The box is now 24.0 × 55.0 × **1.0** m =
> 1320 m³, so every hospital figure above needs recomputing before it is
> compared against a run flown under the current config. The scripts that
> produced them are described in the method notes of that analysis: surface-
> sample the collision meshes at 0.025 m, close the shells with an
> inscribed-radius test, downsample to 0.10 m with `any`, then label with 3D
> 6-connectivity after a 0.26 m horizontal erosion. The warehouse column is
> current.

`explorable` is the comparator for `/voxel_mapping/map_coverage`, which counts
voxels inside the box that have left UNKNOWN — so it counts observed obstacle
*surfaces* as well as free space flown through, and is therefore closer to
explorable than to reachable.

The hospital's 80.6 m³ of sealed volume is the honest ceiling on what any
aircraft could ever map there, and it is worth knowing what it is before reading
a run as incomplete: both north stairwells (raked soffits that step west as they
rise, solid in every 0.10 m layer of the band), seven closed cubicle bays
(curtains solid 0.095–2.483 m), and both elevator shafts behind closed door
slabs. **The 26 doorways are all passable** — the reachable set is a single
component spanning x −11.95..11.95 and y −34.45..20.45 with no gap, so a stalled
hospital run is never a disconnected box.

The warehouse box has **no unreachable pockets at all** after being narrowed off
the shelf block, which is the property that matters most about that change: the
six dead-end aisle stubs it removed were the frontier finder's favourite place
to put viewpoints, and nothing west of x = 2.6 depended on them.

Three things in that table are worth separating, because they are three
different fixes and only one of them is a controller change.

**The baseline was not slow, it was throttled.** `proximity_cap` binding 77% of
ticks at 0.36× the planned speed is an aircraft being held back by its own
reflexes while flying exactly where the planner put it. Replacing the absolute
governor with the clearance comparison is what turns that into `following`
100% (×1.00), and it is the single largest change in how the aircraft moves.

**Run 002 covering *less* than the baseline is not a regression.** It is the
box: run 001 had a box that stopped at y = −20 and spent its whole mission in
the north, so its 598 m³ was 598 m³ of a quarter of the building it was allowed
to see. Run 002 opened the south wing and then lost its time to the retreat loop
at one north-west pocket. Coverage between runs with different boxes is not
comparable, which is why the row below it — how far south the aircraft actually
got — is the one that shows what the box fix did.

**The retreat fixes are what let it finish the sweep.** Run 003 differs from 002
only in the three ways a doorway is treated: creep suppresses the inferred
wedge, a map-caused retreat skips its turn-and-look, and a deliberate stop
clears the wedge window. Retreats fall 25 → 9 and contact episodes 27 → 2, and
the aircraft gets from y = 1.0 to y = −33.3, i.e. the whole 56 m of the
building.

Warehouse ground truth, computed from the collision meshes for the box as it now
stands (x −4.4..2.6, y −9.0..7.2, z 0.6..2.4): box 204.1 m³, **explorable
192.8 m³**, reachable-from-spawn 185.4 m³ after eroding by the airframe radius,
and **no unreachable pockets** — the narrowed box is a single connected
component.

### Distance flown is not the test — and it failed a complete map

`campaign_run.sh` guards against FALCON's "exploration finished" firing on a
near-empty map, because that has happened: runs 026/031 and three soak runs
"finished untouched" having flown 0.0–10.8 m. The guard it used was **distance
flown**, with a 40 m bar.

That is a proxy, and it was only ever chosen because the mapper's own coverage
figure was not available at the verdict. **An exploration's job is to observe the
volume, not to visit it.** The depth camera reaches 5 m, so in an open world the
aircraft legitimately finishes having flown a fraction of the floor it mapped:
measured, a warehouse run covering **201.9 m³ of a 204.1 m³ box — 98.9%, a
complete map, zero contacts** — was failed by this guard for flying 30.1 m.

The guard now needs **both** signals to agree before calling a finish trivial: a
short flight AND a nearly empty map (`MIN_PATH_M` 40 and `MIN_COVERAGE_M3` 50).
Either alone is a proxy; coverage is the thing itself.

### A finish that maps a third of the building is a failure, and it was passing

The opposite error, and the more dangerous one. FALCON declares "exploration
finished" the moment its frontier finder comes up empty, and that can happen with
most of a world unvisited. Measured: a hospital run that **FINISHED** having
covered 260.8 m³ and never gone south of y = −2.3 — the north third of a 56 m
building — while the same configuration had reached y = −33.6 and 760 m³ an hour
earlier. The trivial-finish guard passed it, correctly: 260.8 m³ is not an empty
map. It is simply not the world.

`both_worlds.sh` now sets a per-world **`FINISH_MIN_COVERAGE_M3`** and
`campaign_run.sh` returns `PARTIAL_FINISH` below it. The floor belongs to the
caller because only the caller knows what the world affords: warehouse 170
(explorable 192.8, complete runs land at 201.5), hospital 600 (box 924, best
complete run 760, worst partial 260.8).

**The cause was not the follower and not frontier retirement.** Comparing the two
runs: the complete one logged **0** `No path to next viewpoint` lines and retired
94 frontier clusters; the partial one logged **58** and retired only 49. It
retired *fewer* frontiers — A\* simply could not route to them, and FALCON then
gave up on regions it could see but not reach.

Upstream's A\* default profile is `resolution: 0.5, max_search_time: 0.001` —
**one millisecond** for a route across a 24 × 56 m building — and on a 0.5 m
lattice a 0.93 m doorway is two cells, so one occupied cell closes it to the
search. `exploration.launch` now overrides both (0.3 m, 0.05 s) after the yaml
load. The coarse profiles are left alone: they build the tour's cost matrix over
many pairs at once and are meant to degenerate to a straight-line estimate.

**Refining the lattice without moving the node pool makes it worse, not
better**, and doing exactly that cost a run. `allocate_num` is A\*'s pool and
upstream pairs 100000 nodes with the 0.5 m grid:

| lattice | cells over the hospital's 28 × 60 × 4 m map | pool |
|---|---|---|
| 0.5 m | 53,760 | 100,000 — comfortable |
| 0.3 m | **248,889** | 100,000 — **2.5× short** |

Every query then exhausted the pool and returned no-path: **2012** of them in
one run against 58 in a healthy one. FALCON went silent, the follower's
respawn-recovery survey fired seven times, and the aircraft spent **93% of its
ticks turning on the spot** until the watchdog ended it for not moving.
`astar_allocate_num` is now 500000, about 25 MB.

That failure also exposed a missing bound in the follower. The recovery survey
rebuilds the MAP; it does nothing for a planner that has the map and cannot
ROUTE through it, and from inside the follower those two look identical — both
are silence. `max_resurveys` (4) caps it, for the silence recovery and the
watchdog nudge alike. Past the cap the follower holds, which is honest (it has
nothing left to try), lets FALCON's dead-end guard see a stationary aircraft,
and lets the watchdog reach a verdict on a mission that is genuinely stuck
rather than on one that is busy spinning.

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
drift to 0.4 m. Moving the knee to the airframe radius fixed the warehouse and
did nothing for the hospital, where every corridor puts a wall inside 0.6 m: the
governor still bound **71–77% of ticks at 0.36× the planned speed**. That is the
measurement that killed the absolute curve altogether — `prox_stop_m`,
`prox_slope` and `prox_floor` are gone, replaced by the clearance envelope
above, and the first hospital run flown on it reported `following 100% (x1.00)`
in open corridors instead.

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

| warehouse obstacle | top | altitude needed at `safe_distance` 0.45 |
|---|---|---|
| ClutteringA piles | 1.10 m | 1.55 m |
| buckets | 1.41 m | 1.86 m |
| tall C-piles | 1.79 m | 2.24 m |

At the old `box_max_z` 1.8 (and the old `safe_distance` 0.85, which demanded
1.95/2.26/2.63) **none of those fit**. The optimiser cannot satisfy a constraint
the box forbids, so it traded the margin away and skimmed the tops — which is
where the belly strikes came from. `box_max_z` is 2.4: enough for every pile in
the box at the unified 0.45 clearance, and no more, because ceiling the aircraft
never needs is box volume the mission still has to explore before it can finish.

**The same arithmetic runs the other way in the hospital**, where the doorway
lintels are all at z 2.350 and `safe_distance` is isotropic, so above 1.90 m no
route through any door can satisfy the clearance it is asked for. The baseline
run was flying at 1.899 when it stalled, pinned against its own ceiling clamp.

**But the doorway is not the binding constraint there — the furniture is, and
that is the general rule this section was missing.** A flight band whose ceiling
falls *inside* the height range of the world's tall obstacles guarantees
skimming: the optimiser is asked for clearance no altitude in the box can give
(a 1.79 m crate needs 2.24 m at `safe_distance` 0.45), so it spends the margin
and routes over the top with centimetres to spare. The hospital's tall items
cluster tightly at **1.74–1.83 m** — storage racks 1.740 and 1.799, vending
machines 1.830, an AWS crate 1.790 — and a 1.90 ceiling sits right in the
middle of that cluster. Measured: the aircraft grinding along at z 1.78–1.81
over a 1.79 m crate, **25 bumper reports in one corner** with coverage frozen.

`box_max_z` is now **1.60**, cleanly below the cluster, and the rule it buys is
one the aircraft can always satisfy: **overfly anything under about 1.35 m, go
around anything taller.** Neither the planner nor the vertical escape can reach
the tops at all, because `alt_max` is 1.50.

> Read a ceiling as three constraints at once, in this order: what the
> structure above allows (lintels, roof), what the *obstacles* allow (never end
> the band inside their height cluster), and what the volume costs in mission
> time. The warehouse note below derives its ceiling from the first and third;
> the hospital needed the second, and it is the one that bites hardest because
> it fails as contacts rather than as a no-path.

### Lowering a ceiling is half a decision — the floor has to follow it

Dropping the hospital ceiling to 1.60 stopped the skimming and immediately
produced a *different* failure, because it left `box_min_z` at 0.60. The usable
band became 0.70–1.50 m and the aircraft flew most of it at **0.87–1.05 m**,
which is inside the hospital's floor-clutter layer — beds, gurneys,
wheelchairs, toilets, sinks and carts all live between 0.5 and 1.2 m. It could
thread between them and then not manoeuvre, and it ended up boxed into a 2 m
bathroom stub with FALCON offering a viewpoint 1.6 m away that it could not
reach: 23 retreats, coverage frozen at 297.7 m³, watchdog abort with **zero
contacts** — it never hit anything, it simply had nowhere to go.

`box_min_z` is now **0.90**, so the usable band is 1.00–1.50 m. Because the
voxel gate tests only the z layers the airframe can strike, at 1.50 m anything
topping out under 1.2 m stops being an obstacle at all, and most of that clutter
disappears from the aircraft's world. Items over 1.2 m still have to be flown
around, which is correct and which a room has the space for.

**There is a hard ceiling on how high that floor may go, and it is not
obvious.** FALCON's coverage tour reduces each hgrid cell by connected-component
labelling on ONE horizontal slice at a **hardcoded z = 1.0 m** —
`map_dimension: 2` selects `getCCLCenters2D` (`hierarchical_grid.cpp:1571-1575`).
A box whose floor sits above 1.0 m puts that slice outside the box entirely and
the tour's entire model of the world goes blank. So the floor is bounded above
by 1.0 regardless of what the clutter would prefer, and `box_min_z` 0.90 is the
highest value that keeps the slice inside with margin. Anyone raising it further
must first move `map_dimension` to 3 and confirm `getCCLCenters3D` behaves.

There is a second, cheaper version of the same mistake in the FOLLOWER, and it
cost more than the box did. `VoxelBrakeGate` tests only the z layers the
airframe can strike, and its `body_halfheight_m` defaulted to 0.35 m against a
collision mesh that is **0.11 m tall** (z −0.040..0.070). A half-height six times
too large means the aircraft must clear every obstacle by 0.35 m instead of by
its own body: it forced 1.8 m of warehouse altitude to overfly a 1.10 m pile
that 1.5 m clears, and in the hospital it made a 1.79 m clutter pile unflyable
at *any* altitude a 1.9 m ceiling allows — it demanded 2.14 m — which is where
27 of run 002's contacts came from. It is now 0.15 (0.055 measured plus one
voxel), and `DepthProximityBrake`'s corridor half-height, which had the same
0.35 default and vetoed on a 1.14 m desk while cruising at 1.30 m, with it.

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

This used to be the place that warned about four limiters enforcing four
different absolute standoffs against the same wall — depth veto 0.55 m, map gate
0.35 m, bubble 0.28 m, governor knee 0.25 m — with the binding envelope being
the MAX of them rather than any single design value, and an instruction to keep
them all "within sight of `safe_distance`". **That instruction is the thing the
clearance envelope removes.** Three of those four numbers are gone: the governor
knee and the bubble are now the deficit against the plan's own clearance, and
what is left absolute is `clearance_hard_floor_m` (0.30, the airframe) and the
depth veto (0.55, which is a stopping distance and genuinely does not scale with
the corridor). Two numbers instead of four, one of them physics and one of them
kinematics, and neither needs retuning when the building changes.

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
   `escape_climb_max_z`; `escape_climb_s:=0` restores horizontal-only
   behaviour. The climb ceiling is **0, meaning "the flight box ceiling, less
   `escape_climb_ceiling_margin_m`"**, and it must stay inside the box: a
   planner cannot plan from a pose outside its own box, so an escape that
   climbs out strands the mission it was rescuing. It used to be a hardcoded
   1.65 m, chosen when the warehouse box topped out at 1.8, which became
   simultaneously unsafe and wasteful the moment either world's box moved —
   which both of them now have.

   The 0.20 m margin is not decoration, and the reason is the third axis of the
   same problem. See below.

### The top of the flight box is the worst altitude in the building

FALCON checks a candidate trajectory against a map where every obstacle has been
grown by `obstacles_inflation` (0.25 m), so an item of height *h* seals the map
up to *h* + 0.25. The hospital's tall furniture tops out at 1.74–1.83 m — storage
racks, vending machines, the ClutteringC crate — which seals it **from 1.49 m
up**. `box_max_z` is 1.6, and the follower's own altitude guard puts `alt_max` at
1.50.

So the vertical escape, whose whole purpose is to climb until nothing blocks the
aircraft any more, climbed to exactly the one altitude at which *everything*
does. And it is worse than a bad altitude, because of what the collision check
is: `checkTrajCollision()` evaluates a candidate **from the aircraft's own
position**. An aircraft inside an inflated shell makes every candidate collide at
its first sample. FALCON replans, rejects, and never publishes — silently, since
the rejection happens in `PUB_TRAJ` after the plan has already succeeded.

Measured on the run of 2026-08-13 that this fixed: **1781 of 1808** publish
attempts ended `[FSM] Replan: collision also detected on the initial
trajectory`, and the aircraft held `(-8.19, -1.09, 1.50)` **to the centimetre for
180 s** until the mission watchdog cut the run at 360.7 m³. The same signature
appears three times in that one run, every time at z = 1.49–1.50; twice it
escaped by luck.

Two changes, and they are deliberately different in kind:

- **`escape_climb_ceiling_margin_m` (0.20)** keeps the climb out of the top of
  the band, so the recovery stops steering into the trap. Prevention.
- **The unstick manoeuvre** gets the aircraft out when it is in one anyway.

### The other silence, and why a re-survey cannot answer it

From inside the follower, "FALCON has lost its map" and "FALCON has the map and
rejects every trajectory" are the same observation: no message on
`/planning/bspline`. They need opposite responses.

The re-survey answers the first — a respawned node needs to be *shown* the room.
It cannot answer the second, and this is the whole point: **turning on the spot
does not move the start point out of the occupied cell**. So the four re-surveys
are spent on a problem they cannot touch, and the follower then holds station
forever. That is precisely the 180 s freeze above.

The discriminator is movement. After a re-survey a healthy aircraft gets a plan
and flies, so silence this long *combined with* the aircraft not having moved is
the second case. The response (`unstick_after_s` 25 s, `unstick_move_m` 0.35 m
over the 15 s stall window) is to **move**:

- sideways on a bearing that **rotates each attempt** — back, left, right,
  forward off the nose — so a direction that is itself walled is never retried
  identically;
- and vertically back toward `alt_mid`, the middle of the band, which is where
  the fewest inflated shells overlap: floor clutter inflates upward, ceilings and
  tall furniture inflate downward, and the middle is what is left.

`unstick_after_s` sits above `resurvey_after_s` so the cheap answer is always
tried first, and the manoeuvre moves the aircraft far enough (0.20 m/s for 6 s)
to reset the mission watchdog's own no-movement timer — a deliberate escape must
not read as a pinned airframe. `max_unsticks` (8) bounds it, because an aircraft
that has tried four bearings twice and still cannot get a plan is a run worth
ending rather than continuing.

   How much this buys depends entirely on `gate_body_halfheight_m` being
   right, because that is what decides which layers "the airframe can strike".
   At the old 0.35 the climb had to gain 0.35 m of clear air above an obstacle
   before the gate would release; at the measured 0.15 it needs the aircraft's
   actual body.

   **And it must look before it climbs.** A climb that would not clear the
   obstacle either is not an escape, it is a slow grind up its face — measured
   in an acceptance run, where the aircraft rose to 1.80 m (the flight ceiling)
   beside a clutter pile whose top is 1.79 m and scraped along it for 100 s,
   eight bumper contacts, coverage frozen. This is the failure mode a *fixed*
   climb ceiling used to hide by accident: at the old hardcoded 1.65 the
   aircraft stopped below that pile rather than at it, which looked like the
   number being right and was luck. The climb now tests occupancy one gate
   layer above itself and concedes to the planner when climbing would change
   nothing, which is correct at any ceiling.

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

**Both worlds map to completion on one configuration, repeatably.**
`rig/both_worlds.sh`, five rounds in one invocation, no per-world tuning of any
kind and no editing between rounds (2026-08-14, `KEEP_GOING=1` so a failure does
not end the campaign):

| round | hospital | warehouse |
|---|---|---|
| 1 | ABORT_CONFINED, 578.7 m³ | FINISHED CLEAN, 201.7 m³ |
| 2 | **FINISHED, 769.0 m³** | **FINISHED, 201.5 m³** |
| 3 | **FINISHED, 772.7 m³** | **FINISHED, 201.8 m³** |
| 4 | **FINISHED, 803.4 m³** | **FINISHED, 200.8 m³** |
| 5 | **FINISHED, 825.5 m³** | **FINISHED, 201.0 m³** |

**Nine of ten legs finished, and both worlds finished together in four rounds of
five.** FINISH here is FALCON's own verdict — its frontier set emptied — not a
watchdog or a time cap. The warehouse figure is 98–99% of the 204.1 m³ its box
affords, and its spread across five runs is 1.0 m³. The hospital's best round
mapped 825.5 m³, against 760 for the best run this package had ever recorded
before this work.

The single failure is the honest one to look at: round 1 reached 578.7 m³, most
of the building, before the aircraft confined itself. That is the residual, and
it is a hospital problem rather than a configuration problem — the warehouse has
not failed a leg in fifteen consecutive attempts.

Earlier campaigns on this same image, before the follower work below, finished
**one hospital leg in five, twice over**. The escape ladder is what moved that to
four in five: a jam is caught by comparing demanded travel with achieved travel,
a second give-up at one place escalates to a manoeuvre instead of another hold,
the watchdog's nudge does something once the surveys are spent, all three are
rate-limited so the escape cannot become the flight plan, and the bearing is
chosen from the occupancy map rather than by rotating blindly.

Contacts are still present and are counted per Gazebo contact point per physics
step, so the figures read higher than the number of events: the finishing rounds
range from 3 to 76 reports on 1 to 7 objects. They are first approaches to
geometry inside the 0.95 m depth near clip — the known limit described under the
depth brake below — and remain the largest open item on this stack. **Planner
respawns are zero across all ten legs**, against a baseline where a single
hospital run took 15.

What it took is recorded in the sections below and in `patches/README.md`; the
short version is that four defects had to be fixed together, and each was found
by measurement after a theory about it turned out to be wrong:

1. FALCON re-timed every plan onto a hardcoded 2.0 m/s, compressing this
   aircraft's trajectories 4-13x and eventually aborting the node from inside
   its own visualiser, which erased the map (`falcon_slow_traj_rescale.patch`).
2. A jam with no bumper contact, invisible to both the contact reflex and the
   silence reflex, caught now by comparing demanded travel with achieved travel.
3. A viewpoint blacklist with no expiry, which retired the corridor to the south
   wing at t = 53 s of one run and let FALCON declare success having mapped half
   the building (`falcon_blocked_region_ttl.patch`).
4. This follower's own give-up hold, which escalated to 90 s and parked the
   aircraft in the exact spot that had defeated it for a continuous 300 s.

### The regression that followed it was the IMAGE, not the code

**Resolved.** Between those runs and the campaigns after them, the warehouse fell
from 201–202 m³ (ten consecutive finishes) to 98–139 m³ on twenty-two
consecutive legs, in both worlds, on unchanged code. The cause was a single
`catkin_make --pkg voxel_mapping` inside the FALCON image: its shipped
`libvoxel_mapping.so` carries a fix that is **not in the source beside it**, so
rebuilding that one package silently produces a worse mapper. Re-tagging the
pre-rebuild layer and flying it unchanged restored 202.01 m³ in 123 s on the
first attempt. Full account, and the rule that follows from it, in
`patches/README.md` under "NEVER rebuild `voxel_mapping`".

Two things in the investigation below are worth keeping even though the answer
turned out to be elsewhere, because both are now permanent parts of the rig.

The section that follows was written while the cause was still unknown. It is
kept as the record of what was eliminated, and because its central mistake is
instructive: the image was "eliminated" by re-tagging an intermediate layer and
flying it, which is the right method — but the layer chosen carried a timestamp
four minutes *after* the last good run, so it was the first bad image, not the
last good one. The conclusion "the image is exonerated" was drawn from testing
the wrong image, and cost several hours. Check layer timestamps against the
runs' own artifacts before trusting a bisect.

### The investigation, and what it eliminated

Read the table above with this section next to it. The run it describes is real
and its artifacts are on disk, but a campaign is one sample, and the samples
after it do not agree with it.

`rig/both_worlds.sh` now takes `KEEP_GOING=1`, which continues through a failed
leg instead of stopping the campaign, because stopping at the first failure
spends a whole campaign to learn a single bit. Measured with it, five rounds per
campaign:

| campaign | hospital | warehouse | notes |
|---|---|---|---|
| rate_A | 1/5 finished | **5/5** | warehouse 200.99–201.41 m³ |
| rate_B | 1/5 finished | **5/5** | warehouse 201.19–201.67 m³ |
| rate_C | 0/5 | 0/5 | warehouse 103–132 m³ |
| rate_D | 0/3 | 0/3 | after reverting the suspected change |
| rate_HEAD | 0/3 | 0/3 | **the committed configuration itself** |

The warehouse succeeded ten times running, then failed twelve times running, and
the break is sharp rather than a drift. What makes it worth writing down is that
the second half includes the exact commit that produced the first half.

**What has been eliminated, each by measurement rather than by argument:**

- *The repository.* `rate_HEAD` is `git checkout`ed to the commit that passed.
  Its FALCON startup parameters were diffed against a passing run's and are
  identical apart from parameters that did not exist then.
- *The FALCON image.* Docker kept every intermediate layer, so the exact image
  the passing campaigns ran on was re-tagged and flown: 125.48 m³ with 79
  contacts, the same degraded shape.
- *The world and the aircraft's start.* `campaign_run.sh` verifies the world by
  name in every run, and the spawn is logged at (1.0, 1.0, 2.0) in both.
- *Simulation speed.* Real-time factor is 1.00 in both, from `rtf.log`.
- *The machine.* Load average under 1.4, 374 GB free, 54 GB RAM available, GPU
  idle at 33 °C with no throttling, no stray ROS participants on domain 20, and
  per-container CPU and memory in `stats.log` that match a passing run within a
  few percent.
- *The assets.* Nothing under `sjtu_project/` or the map configs has a mtime
  inside the window, and the drone container builds nothing (`SKIP_BUILD=true`).

**What the failure looks like** is consistent and is the same in both worlds: the
aircraft leaves its own flight box. A warehouse box of x[-4.4, 2.6] y[-9.0, 7.2]
with the aircraft measured at (3.7, 7.4). FALCON cannot plan from outside its
box, so coverage stops and a watchdog ends the run. Whatever changed acts on the
aircraft's position, not on the planner's configuration.

Every one of those eliminations was correct and none of them was the answer,
because the thing that changed was the image and the image had been checked
against the wrong layer. The Docker daemon was restarted too, on the reasoning
that it was the last untested hypothesis; it made no difference, and the
warehouse still timed out at 139 m³ afterwards.

**Two pieces of the rig came out of this and are worth keeping.**

`rig/both_worlds.sh` takes `KEEP_GOING=1`, which continues through a failed leg
and reports a per-world tally instead of stopping the campaign. A campaign that
stops at the first failure spends an hour to learn one bit, and on a process
this variable that is not enough to tune against.

`rig/campaign_run.sh` waits for the Gazebo model count to stop changing before it
flies, and records that count in `verdict.json`. The checks it had proved that
gzserver *began* loading the right world and that a depth topic existed, neither
of which says the furniture is in the room, and a mission that starts early maps
free space where a shelf is about to appear. The count needs no per-world
constant — stability is the property that matters — and it earned its place
immediately by killing the half-loaded-world hypothesis outright: 72 models in
every hospital run and 26 in every warehouse run, without exception.

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
  clock. This used to say the hospital runs at roughly half real time (depth
  measured at 7.5 Hz against a 15 Hz nominal); **on this machine it does not** —
  `gz stats` reports a factor of 1.00 across full hospital explorations, and sim
  elapsed tracks wall elapsed to within a few seconds over ten minutes. Do not
  assume either way: the factor is a property of the machine and the world
  together, it is sampled into `rtf.log` every 60 s by `campaign_run.sh`, and
  `analyze_run.py` prints its mean and minimum. The mitigation is the same
  whatever it reads — `use_sim_time` is true and every loop here is driven off
  the ROS clock.

### Not yet done

> The paragraph below is **superseded** and kept only because the numbers above
> it in this section were measured under it. FALCON now flies both worlds end to
> end: the bridge, the roscore and the planner launch are all in place, the
> corrected intrinsics are exercised by the real mapper, and `rig/both_worlds.sh`
> is the acceptance test. What is genuinely still not done is at the end of this
> section.

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

**Still genuinely open**, in rough order of how much they cost:

* **The ESDF ratchets and never recovers.** `ESDFVoxel::value` starts at
  `double` max and `updateLocalESDF` writes only if SMALLER (`esdf.cpp:38-41`).
  Occupancy can flip OCCUPIED back to FREE, but the distance field cannot: one
  frame of depth noise plants a permanent low-ESDF scar that the optimiser's
  distance cost honours for the rest of the mission. Every long run is therefore
  flying a world that looks slowly tighter than it is, and no amount of tuning
  on our side reaches it. Fixing it means a FALCON patch that re-seeds the ESDF
  over the update box rather than min-ing into it.
* **The coverage tour's model of the world is one horizontal slice.**
  `map_dimension: 2` selects `getCCLCenters2D`, which labels connectivity at a
  **hardcoded z = 1.0 m** (`hierarchical_grid.cpp:1571-1575`). That is below the
  altitude the aircraft actually flies and it is where the hospital's desks,
  racks and vending machines are, so the hgrid can believe two regions are
  disconnected that the drone can fly between at 1.5 m. Worth testing
  `map_dimension: 3` against a run that is otherwise clean.
* **`isNearUnknown` does not scale in z.** `min_candidate_clearance` sets the
  x,y half-extent in voxels but the z extent is hardcoded to +-1 voxel
  (`frontier_finder.cpp:1140-1151`), so the parameter means something different
  vertically than horizontally.
* **The depth brake still over-cuts sideways** (see above), and the yaw a
  doorway is entered at is unmanaged: FALCON chooses yaw to aim the camera at
  the next frontier, which is not the same as aligning the 0.52 m airframe with
  a 0.93 m opening. At 45 degrees of misalignment the swept width is 0.63 m and
  the budget falls from 0.19 m to 0.14 m.
* **FALCON plans through UNKNOWN, and some of this world's obstacles are hollow
  shells.** `checkTrajCollision` rejects a trajectory sample only if it is
  literally OCCUPIED (`planner_manager.cpp:384-404`); unknown is fine, and the
  A* underneath it is no stricter. Meanwhile several AWS props are modelled as
  *hollow* collision meshes — the `ClutteringC` crate is a 1.77 × 2.06 m shell
  whose 2.3 m³ interior is a cavity — so their insides are never observed, stay
  UNKNOWN forever, and read to the planner as somewhere it may route. The
  aircraft is then sent at a wall it cannot see through, and the follower's gate
  stops it only once the near face has been mapped.

  This is the single largest remaining source of contacts: one `ClutteringC`
  crate in the hospital's north-west corner accounts for 14 of 16 bumper reports
  in an acceptance run, and it recurred across three separate runs at the same
  place. It is bounded rather than fixed — impact speed is now capped by the
  derived ceiling, the retreat clears it, and FALCON's dead-end guard eventually
  retires the viewpoint — and the mission finishes regardless. Fixing it
  properly means either treating unknown as non-traversable in the follower's
  gate (cheap, and it would stop the aircraft entering any unmapped volume,
  which is a large behaviour change to validate) or closing the shells in the
  map, which is the world's problem rather than the stack's.

* **Nothing slows the aircraft for flying into UNKNOWN space**, and that is the
  other half of the same story. Both episodes in run 003 were first approaches to
  geometry that was not yet in the map: the depth camera's ~0.95 m near clip
  makes the last metre blind, so an unobserved cell can be solid and the
  follower has no way to know. The voxel gate accumulates OCCUPIED only, so
  "unknown" and "free" are the same thing to it. Subscribing
  `/voxel_mapping/occupancy_grid_free` as well would let the corridor sweep
  distinguish them and cap speed on unobserved cells, which is the shape of the
  fix; the cost is a second multi-megabyte cloud on the follower's callback
  thread, so it wants measuring before it is built.

### Measured plant (`/simple_drone`, step response on the odometry)

| axis | DC gain | transport delay | time constant |
|---|---|---|---|
| horizontal (vx, vy) | 0.998 | 0.181 s | 0.510 s |
| vertical (vz) | 1.024 | 0.033 s | 0.409 s |
| yaw rate | 0.999 | 0.055 s | 0.477 s |

Re-measure these if the plugin's gains in `sjtu_drone_bringup/config/drone.yaml`
change, or if the world is heavy enough to push the real-time factor below one.
