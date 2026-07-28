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
is simply told where the aircraft is and replans from there. Closing that gap is
what `core/planning/trackers/reference_tracker_3d/` exists for.

```
FALCON container (ROS1 Noetic)                Isaac Sim container (no ROS)
──────────────────────────────                ────────────────────────────────
exploration_node   ── unmodified              Isaac Sim + Pegasus Iris + PhysX
traj_server        ── unmodified              PX4 SITL (lockstep)
                                              ReferenceTracker3D
pegasus_bridge_node        ◄── TCP 5599 ──    depth frame + camera pose
  /uav_simulator/depth_image                  ground-truth odometry
  /uav_simulator/sensor_pose
  /uav_simulator/odometry
  /planning/pos_cmd        ─── TCP 5600 ──►   tracker → PX4 velocity setpoints
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
```

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
- **A visualisation box worth drawing.** `vbox` is normally a 20 cm slab at
  cruise height, because that is all the map recorder needs and every extra
  layer is another full pass over the voxel grid at 2 Hz inside the same thread
  as the mapper. Drawn as-is the voxel view is a single contour line. `--rviz`
  widens it to the flight band; `viz_min_z` / `viz_max_z` set the range.

It is off by default because it is not free: the occupancy clouds are only
computed when something subscribes, and the wider box costs `exploration_node`
real time on the thread that also services the depth callbacks.

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

## FALCON's TSP solver crashes, and it is invisible from the aircraft

`exploration_node` segfaults inside its vendored LKH solver on some coverage-tour
instances. Its own backward-cpp trace names the frame:

```
#3  ExplorationManager::solveTSP(...)
#2  solveTSPLKH(char const*)
#1  FindTour
#0  LinKernighan            <-- SIGSEGV
```

Seen twice here, both times mid-flight after minutes of healthy exploration. It
is third-party C with global state; it is not fixable from this repo.

What makes it dangerous is that **nothing downstream notices**. `traj_server` is
a separate process and outlives the planner, and its command callback holds the
*final point of the last trajectory* forever once the trajectory's duration has
elapsed. So commands keep arriving at 100 Hz, the aircraft tracks them to a
centimetre, and every health check reads perfect while the drone hovers for the
rest of its budget. The trajectory id is the only thing that stops moving, which
is why `PLANNER_STALL_S` watches exactly that.

A second upstream crash — three unguarded `candidates[min_cost_id]` reads where
no candidate won — *is* fixable and is fixed, in
`patches/fix_falcon_viewpoint_index.sh`.

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
    px4_exploration_params.py  what exploration changes about the PX4 set
  adapter/                   the ROS1 catkin package `falcon_pegasus`
    scripts/pegasus_bridge_node.py   Isaac <-> FALCON's simulator topics
    scripts/map_recorder_node.py     the map video, with no display attached
    launch/falcon_pegasus.launch     exploration.launch, simulator amputated
  viz/exploration_frame.py   how a frame of the map video is drawn
  stub/                      the same mission without Isaac Sim (see below)
  patches/                   two fixes to upstream FALCON, both explained in situ
  Dockerfile                 FROM the FALCON image, adds this catkin package
```

Everything pure lives outside this package: the outer-loop controller is
`core/planning/trackers/reference_tracker_3d/`, the camera extrinsics are
`robots/PEGASUS/adapters/camera_pose.py`, and the platform itself is
`robots/PEGASUS/`.

## The stub, and why to use it first

`stub/check.sh <run>` flies the whole mission against the real FALCON stack with
no Isaac Sim, no GPU and no PX4 warm-up. Depth comes from raycasting the surveyed
ground-truth voxel map — the same building, measured rather than rendered — and
the airframe is a first-order velocity lag, which is what an inner-loop velocity
controller looks like from outside. Everything else is the code that flies the
real aircraft: the same protocol, the same handover order, the same tracker, the
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
