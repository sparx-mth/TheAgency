# PEGASUS — simulated drone platform (Isaac Sim + Pegasus Simulator + PX4 SITL)

The "robot" layer (see `tasks/planning/vlas/README.md`'s three-layer VLA split)
for a simulated multirotor: an Isaac Sim indoor scene, a Pegasus Simulator Iris
quadrotor with a PX4 SITL autopilot, and an RGBD camera matching the real
XTEND's calibration. Its job is producing **flight recordings** in the same
on-disk schema (`tasks/planning/vlas/common/finetune/datasets/recording.py`)
that real rosbag extractions use, so simulated flights are a drop-in
`data.recording` source for `tasks/planning/vlas/navdp/finetune/configs/
navdp_finetune.yaml` — no parallel dataset format.

This platform names no policy, per the VLA layering rule. There is currently no
`config/vla/*.yaml` here because nothing runs a live policy against it yet —
today it only *generates training data*.

## Why Pegasus + PX4, not Isaac Lab's stock Crazyflie

Isaac Lab ships a `CRAZYFLIE_CFG` / `QuadcopterEnv`
(`isaaclab_tasks/direct/quadcopter/`) out of the box, but it's a bare RL hover
task: no camera, flat-ground terrain only, 10 s episodes, 4D thrust/moment
action space. It's built for RL, not for generating photorealistic
long-horizon flight recordings. Pegasus Simulator puts a real PX4 autopilot in
the loop flying an Iris-class multirotor, which better matches "long pilot"
missions and how the real drones actually fly.

## The Isaac Sim 6.0.1 compatibility problem (and fix)

As of 2026-07-26, Pegasus Simulator's latest release/`main` (commit
`e13dc659`) is still built and tested only against **Isaac Sim 5.1.0** — no
tag, branch, or doc mentions 6.x, and each Pegasus release is explicitly
version-locked to one Isaac Sim release with no stated backward compat.

Investigating inside the `isaac-sim:6.0.1` container showed the actual gap was
narrower than the version-lock warning suggests: the vehicle/camera code was
**already migrated** to the modern `isaacsim.core.*` / `isaacsim.sensors.*`
namespaces. The one real holdout was `omni.isaac.dynamic_control`, used in
exactly two files (`vehicle.py`, `multirotor.py`) for force/torque
application, articulation access, and DOF control — an extension that no
longer exists in 6.0.1 at all (not even as a deprecated shim).

**Fix** (`setup/pegasus_isaac6_compat.patch`, pinned to commit `e13dc659`):
replaced the `_dynamic_control` interface with `isaacsim.core.prims.RigidPrim`
/ `Articulation` views (present under `extsDeprecated/isaacsim.core.prims`,
deprecated but functional):

- `apply_force`/`apply_torque` → `RigidPrim.apply_forces_and_torques_at_pos(..., is_global=False)`
- `update_state`'s position/velocity reads → `RigidPrim.get_world_poses()` / `get_linear_velocities()` / `get_angular_velocities()`
- `get_articulation()` + `find_articulation_dof`/`set_dof_velocity` (propeller spin animation) → `Articulation.get_dof_index()` + `set_joint_velocities()`
- `force_and_torques_to_velocities`'s relative rotor positions → computed manually from `RigidPrim.get_world_poses()` (no direct `get_relative_body_poses` equivalent)
- `extension.toml`'s `"omni.isaac.core"` dependency → `"isaacsim.core.api"` + `"isaacsim.sensors.camera"`

Verified headless: the patched extension loads in Isaac Sim 6.0.1, an Iris
spawns, and 10 physics steps run cleanly through the patched force/torque/
articulation path (`SPAWN_OK` / `STEP_OK` in the compatibility smoke test).

**Discovery/registration gotcha:** dynamically registering the extension
folder at runtime (`ext_manager.add_path(...)` + `enable_extension(...)`)
makes Kit try to resolve it against NVIDIA's *online* extension registry
first, which fails (permission error writing the registry index cache, and
`pegasus.simulator` isn't a published registry extension anyway). Fix:
symlink the extension into Isaac Sim's `extsUser/` directory instead, which is
scanned locally at startup with no registry round-trip —
`setup/install.sh` does this for you.

## The PX4 SITL build problem (and fix)

PX4 v1.14.3 (the version Pegasus was developed against) predates official
Ubuntu 24.04 / GCC 13 support, which is what the `isaac-sim:6.0.1` container
ships. Two GCC-13-specific build failures, both fixed in `setup/install.sh`:

1. `platforms/posix/src/px4/common/px4_daemon/pxh.cpp` used `uint8_t` without
   `#include <cstdint>` — older GCC pulled it in transitively via another
   header; GCC 13 doesn't. One-line fix.
2. `matrix::inv`'s 1x1 `SquareMatrix` specialization trips a GCC 13
   `-Werror=array-bounds` false positive (the compiler can't prove the
   out-of-bounds branch is dead code in the 1x1 case). Downgraded from fatal
   to warning-only for `array-bounds` in `cmake/px4_add_common_flags.cmake`.
3. `Tools/setup/requirements.txt` pins `empy>=3.3` with no upper bound, which
   resolves to `empy` 4.x and breaks the PX4 build (a well-known PX4 issue) —
   pin `empy==3.3.4` explicitly.

Also: `make px4_sitl_default none` does not just build — after a successful
build it **launches** the `px4` binary into an interactive shell (`pxh>`).
Run it with no TTY attached (e.g. backgrounded under `docker exec`) and it
spins printing the prompt forever, producing a multi-GB log. Either run it
attached to a real terminal, or build-then-kill the launched process, or pipe
its output through `head`.

## Confirmed indoor scenes

`isaac-sim` has outbound access to NVIDIA's asset CDN. The full bucket listing
for `Assets/Isaac/4.5/Isaac/Environments/` is: `Digital_Twin_Warehouse`,
`Grid`, `Hospital`, `Jetracer`, `Modular_Warehouse`, `Office`, `Outdoor`,
`Simple_Room`, `Simple_Warehouse`, `Terrains`. `adapters/scene.py`'s
`INDOOR_SCENES` covers the indoor-relevant subset: `simple_room`, `hospital`,
`office`, `warehouse`, `full_warehouse`.

**There is no bundled home or library scene** — confirmed both by probing
specific guessed paths (404) and by the authoritative bucket listing above.
Best-effort alternative found but not yet integrated: the
[`spatialverse/InteriorAgent`](https://huggingface.co/datasets/spatialverse/InteriorAgent)
dataset on Hugging Face — publicly downloadable USD residential interior
scenes (`kujiale_xxxx` folders, with materials/lighting/room metadata),
confirmed compatible with Isaac Sim 4.2/4.5. Wiring it in is future work: it
would need its own loader (its scenes aren't single-USD CDN references like
`INDOOR_SCENES`) and a spawn-pose survey per scene.

## Layout

```
robots/PEGASUS/
  config/camera_pegasus_iris_504x392.yaml   sim camera intrinsics (== XTEND's 392x504 depth-engine calibration)
  config/camera_falcon_explorer_640x480.yaml a symmetric 90x74 deg pinhole, for planners that model the sensor
  adapters/camera_pose.py                    world pose of the camera OPTICAL frame (what a mapper needs)
  maps/<scene>_voxels.npz                    ground-truth 3D occupancy at 10 cm -- the source of everything else
  maps/<scene>_voxels.ply                    the same, as a point cloud for Open3D (regenerated, not committed)
  maps/<scene>_alt<NNN>cm.npz                a horizontal slab of it at one altitude, for 2D planning
  adapters/scene.py                          load_indoor_scene() + a hand-measured spawn per scene
  adapters/scene_map.py                      where the maps live and how to read them (no Isaac import)
  adapters/voxel_survey.py                   sweep a loaded scene into the 3D map (needs a live sim)
  adapters/sensors.py                        the PX4 sensor suite with every noise term zeroed
  adapters/vehicle.py                        PegasusIrisVehicle -- spawn Iris (+ optional PX4 backend) + RGBD camera
  setup/pegasus_isaac6_compat.patch          the Isaac Sim 6.x compatibility fixes, pinned to commit e13dc659
  setup/install.sh                           clone+patch Pegasus, clone+build PX4 SITL
```

The recording writer (`sim_extract.py`) and the flight harnesses live outside
this package since they're platform-agnostic and mission-level respectively.
**To actually fly something, see
`tasks/planning/sim_flight_recording/README.md`** — that is the operating
manual; this file covers the platform and the compatibility work underneath
it.

## Simulated sensor noise, and why it is all switched off

Pegasus models a real GPS receiver, IMU, magnetometer and barometer, noise and
biases included. `adapters/sensors.py` zeroes every configurable noise term,
which makes PX4's estimator input ground truth. That is the difference between
a campaign that works and one that does not: on the stock sensors PX4's
position hold wandered metres indoors — more than the gap between two office
desks — and roughly half of `office` flights ended against furniture.

Three things are worth knowing before changing it:

- **The correlation times must stay non-zero.** They appear in divisors
  (`imu.py:111`/`:131`, `magnetometer.py:118`, `gps.py:130`); zeroing one is a
  `ZeroDivisionError`, not a quieter sensor.
- **The barometer needs a subclass.** Alone among the four it has *no config
  key* for its noise — `Barometer.update` unconditionally draws ~1 Pa of
  Gaussian pressure error, which at sea-level density is 8.4 cm of altitude, on
  every update. `NoiselessBarometer` overrides the method. (It is also dropped
  from PX4's height fusion entirely — see `px4_params.py` — because PX4 was
  observed switching to a second, stale `sensor_baro` instance mid-flight.)
- **The GPS runs at 20 Hz, not Pegasus's 250 Hz default.** 250 Hz is one fix per
  physics step, 50x what hardware produces and past the ~90 Hz PX4's observation
  buffer accepts anyway.

Exact sensors do **not** want a matching tightening of PX4's own process noise.
That was tried and reverted: it makes the filter trust the accelerometer
absolutely, so every transient disagreement with GPS lands in the accel-bias
state instead, and the result was `Preflight Fail: High Accelerometer Bias`
from the moment of takeoff.

## Where a flight is safe to go: one 3D sweep, sliced

`adapters/voxel_survey.py` sweeps the **whole building once** into a ground-truth
occupancy voxel grid at 10 cm, using Isaac's own `isaacsim.asset.gen.omap`
generator — one PhysX box-overlap per voxel, in C++. Measured on `office`: 27
million voxels in 27 seconds. A per-voxel Python query at that scale would take
hours, which is why this is the only viable route to a 10 cm 3D map.

Everything else is derived from it, so a scene is surveyed once and any altitude
is free:

- **the 2D planning map** is a horizontal slab of the voxel grid
  (`project_to_occupancy_2d`), which also means the 2D and 3D maps cannot
  disagree;
- **the `landable` layer** is whether the voxel column is clear from the floor
  up to cruise height. Flyable and landable are different questions: a cell can
  be wide open at 1.5 m with a desk at 0.7 m under it, and a goal chosen there
  lands the aircraft on the desk, where it tips, and every subsequent episode is
  refused with `Preflight Fail: Attitude failure (roll)`. One campaign lost four
  of six episodes that way.

`office` at 10 cm is 307x745x63 voxels covering 30.7 x 74.5 x 6.3 m — 1.81 M
occupied, and **183 kB compressed**, so it is committed. Sliced at 1.5 m that is
788 m² of contiguous flyable space, of which 731 m² is landable.

The `.ply` beside it is the same occupied voxels as a point cloud, for opening
in Open3D. The exporter has no Open3D dependency (it writes the minimal binary
PLY by hand), because Open3D is not installed on the machine that produces it.

**The flood fill does not separate indoors from outdoors, though it looks like
it should.** The generator marks what it cannot reach from its origin as
UNKNOWN, but the free space above the roof connects to everything, so it escapes
over the top: `office` came back as 4618 m² of building-plus-car-park with zero
unknown voxels. A ceiling test (`restrict_to_indoor`) is what actually separates
them — one array reduction, once the voxel column exists.

**A map is only valid at the altitude it was surveyed at** — clearance at head
height and at desk height are different buildings — so the altitude is in the
filename, and `load_scene_map` refuses to substitute a different one.

## The two dynamics bugs the port introduced (and fix)

Getting the API port to *load* was not the same as getting it to *fly*. Two
further problems only showed up once something tried to apply real thrust.
Both are fixed in `setup/pegasus_isaac6_compat.patch`.

**1. Rotor thrust must be applied to the body, not to the rotors.**
`Multirotor.update()` applied each rotor's thrust to that rotor's own
`/rotorN` prim. Those prims are articulation *links*, and in Isaac Sim 6.x a
`RigidPrim` force on a link does not land at the link's centre of mass — it
induces a large parasitic pitch torque. Measured directly, applying 1.5× hover
thrust with no controller in the loop:

| how the same total force was applied | result after 200 steps |
|---|---|
| split across the four `/rotorN` links | pitch −76°, tumbling, upside down by step 250 |
| summed onto `/body` | clean climb, ±2° attitude, 3.3 m altitude |

The patch aggregates the rotor thrusts into the equivalent body-frame wrench —
total thrust along body +Z plus the torque those offset thrusts generate
(`τ = Σ rᵢ × Fᵢ`), plus the rotors' rolling moment. Same rigid-body dynamics,
one force application. The rotor offsets are measured from the stage once and
cached (`Multirotor.rotor_body_positions()`), which also removes the
duplicate copy of that computation in `force_and_torques_to_velocities`.

**2. A deadlock in the PX4 backend's heartbeat gating.** PX4 is built with
`ENABLE_LOCKSTEP_SCHEDULER`, so its entire internal clock — including
heartbeat generation — only advances when it receives `HIL_SENSOR` data. But
`PX4MavlinkBackend.update()` had a hard `if not self._received_first_hearbeat:
return`, refusing to send any sensor data until it saw PX4's heartbeat first.
Mutual wait, never resolves. The patch sends sensor data unconditionally and
checks for the heartbeat opportunistically instead.

## The Isaac Sim 6.0.1 physics-callback bug (no fix, but a reliable workaround)

**Isaac Sim 6.0.1 stops dispatching `World.add_physics_callback()`-registered
callbacks after ~2 calls following `world.reset()`.** Confirmed by wrapping a
call-counter around `Multirotor.update()`: it printed `call #1`, `call #2`,
then never again across 750+ further `world.step()` calls, with **zero
exceptions raised anywhere**. Meanwhile the vehicle's live PhysX pose
(`RigidPrim.get_world_poses()`, bypassing Pegasus's cached `self._state`)
showed the rigid body integrating normally — PhysX itself is fine. Only the
Python-level callbacks stopped.

Pegasus registers four of them per vehicle, and all four die together: state
refresh, sensor generation, backend state push, and force application (which
is also what pushes `HIL_SENSOR` to PX4). That single bug is why earlier
recordings were always frozen *and* why PX4 never booted far enough to emit a
heartbeat.

No fix for the dispatch itself was found — it would need to be traced through
whatever changed in Kit/PhysX callback wiring in 6.0.1, or moved to
`isaacsim.core.simulation_manager`, which the Kit logs reference throughout.
**The workaround, which is what everything here uses:**
`tasks/planning/sim_flight_recording/manual_physics_driver.py` calls those four
methods by hand once per step from an ordinary Python loop, which does keep
running. With it, PX4's first heartbeat arrives 1.2 s into simulated time.

## The timestep bug: `world.step()` did not advance a fixed amount of time

Found last and mattering most. Pegasus's PX4 world defaults to `physics_dt =
1/250` and `rendering_dt = 1/60`, which Isaac Sim turns into `substeps =
int(rendering_dt / physics_dt) = 4`. So **`world.step()` advanced 4 ms without a
render and 16 ms with one**, and a caller driving the vehicle by one fixed `dt`
was wrong on every step, in one of two different directions.

Everything downstream is derived from that `dt`:

- PX4's lockstep clock is integrated from it (`_current_utime += dt * 1e6`), so
  its clock ran about twice as fast as the world it was flying in.
- Pegasus's simulated accelerometer is `(v − v_prev) / dt`, so specific force
  alternated between 0.4x and 1.6x of truth at 25 Hz — a square wave straight
  into the attitude estimator. This is the most likely identity of the
  "compass/accelerometer-bias estimator divergence" earlier notes blamed for
  the ~50% failure rate.
- Every recorded timestamp was a frame index over a nominal rate the simulation
  was not running at, so the poses in a recording were mis-stamped.

**Fix:** `flight_session.build_world` calls
`PegasusInterface.set_world_settings(physics_dt=PHYSICS_DT,
rendering_dt=PHYSICS_DT)` before `initialize_world()`. Equal timesteps mean
`substeps == 1`, so one `world.step()` is exactly one physics step whether or
not it rendered. Rendering stays occasional — the caller chooses when — it just
no longer changes how much time passes. Confirmed after the fix: a 45-second
flight recorded 450 frames, i.e. exactly the requested 10 Hz.

## Running it

Everything must run inside a live Isaac Sim process (Isaac Sim's own Python,
not the repo's `.venv` — it needs `omni`/`carb`/`pegasus.*`, which only exist
inside a running Kit app).

```bash
# One-time setup, inside the isaac-sim container:
sparx_agency/robots/PEGASUS/setup/install.sh /path/to/dev/root
```

Then survey a scene once (per altitude), and collect **from the host** with the
launcher, which handles syncing the repo into the container, clearing stale PX4
locks, and starting one worker per aircraft:

```bash
docker exec isaac-sim bash -c "cd /tmp/dev/repo && /isaac-sim/python.sh \
  sparx_agency/tasks/planning/sim_flight_recording/survey_scene.py \
  --scene office --altitude 1.5 --preview"

sparx_agency/tasks/planning/sim_flight_recording/run_collection.sh \
  --scene office --episodes 20 --workers 4
```

See `tasks/planning/sim_flight_recording/README.md` for the flags, what one
episode is, what comes out, and how several workers stay out of each other's
way.

## Two cameras, and which to render with

`camera_pegasus_iris_504x392.yaml` is a copy of the real XTEND's calibration, so
simulated recordings are geometrically interchangeable with real flights. It is
what a data-collection campaign should render.

It is the wrong camera for a planner that *models* the sensor. Its principal
point is 67 px above centre (a crop of a real camera), so its field of view is
asymmetric about the optical axis: ~15.5 deg up, ~29.7 deg down. FALCON's
frontier-visibility model assumes a symmetric cone about the body boresight, so
fed this camera it believes it can see frontiers that are outside the image,
flies to viewpoints for them, observes nothing, and picks them again.
`camera_falcon_explorer_640x480.yaml` is a symmetric 90 x 74 deg pinhole for
exactly that case — see `tasks/planning/falcon_pegasus/`.

`flight_session.spawn_vehicle` takes an explicit `intrinsics=` for this;
`resolution=` still rescales the platform calibration when that is what you want.

## Where the camera actually is, and why it matters more than it sounds

`adapters/camera_pose.py` gives the world pose of the camera's **optical** frame
(x right, y down, z forward), which is what a depth mapper back-projects with.
Handing a mapper the *body* pose instead produces a complete, self-consistent map
of the building rotated ninety degrees, and raises nothing anywhere.

The 20 cm forward mount has a second consequence that is easy to miss. The camera
carves free space outward *from itself*, so the body origin sits in the one place
it can never observe: 20 cm behind the lens, at every heading, for the whole
flight. Any planner that treats unobserved space as impassable — FALCON's A* does
— will refuse to expand a single node from the aircraft's own position. The fix
is to report the aircraft's position **at the sensor**, which is also the
assumption FALCON's own configs encode (their `T_b_c` has zero translation).

## Aerial Gym: deliberately not installed

Aerial Gym Simulator (`ntnu-arl/aerial_gym_simulator`) is still built
exclusively on NVIDIA's deprecated IsaacGym (Isaac Sim/Isaac Lab support has
been "under development" upstream for a long time with no release). IsaacGym
Preview 4's bundled `libPhysXGpu_64.so` has no compiled kernels for Blackwell
(`sm_120`) and fails outright with `CUDA error: no kernel image available for
execution` on RTX 50-series cards — this is a shipped-binary limitation with
no driver/CUDA-toolkit workaround and no maintained fork. It will not run GPU
physics on this machine's RTX 5070 Laptop GPU; skipped rather than installed
for a known dead end.

## Isaac Lab (installed for the future training phase, not used by PEGASUS)

Isaac Lab 3.0 Beta2 (`v3.0.0-beta2`, tag pinned) is installed into the same
`isaac-sim` container/Isaac Sim Python, at `$DEV_ROOT/IsaacLab`, layered on
top of the existing Isaac Sim install via a `_isaac_sim` symlink
(`ln -s /isaac-sim $DEV_ROOT/IsaacLab/_isaac_sim`) rather than a second
container. Install with `./isaaclab.sh -i`, run as root inside the container
(`docker exec -u root`) since the installer shells out to `sudo`, which this
minimal image does not have. It is unrelated to how PEGASUS flies today
(PX4 + Pegasus, not an Isaac Lab env) — it is prep for RL-based training,
which is why `nav_mode`/flight scripts here don't reference it.

Two things to know before using it:

- **`isaaclab_newton` and `isaaclab_rl`'s wheel builds intermittently
  segfault** (`exit code: -11`) during `pip`'s isolated build-dependency
  install step, for no reproducible reason found so far (not memory
  pressure — confirmed >20GB free RAM at the time). A bare retry of
  `./isaaclab.sh -i` succeeded both times this was hit; treat it as a flake,
  not a real blocker, but don't assume a failed install run is authoritative
  without one retry.
- **Environment construction is real but very slow on this GPU.** Verified
  with `py-spy dump` against a running `zero_agent.py --task
  Isaac-Cartpole-v0` process (needs `--cap-add=SYS_PTRACE` on the container):
  the stack is genuinely inside `ManagerBasedRLEnv.step()` →
  `Articulation._apply_actuator_model()` → real Warp kernel launches, not
  hung or deadlocked. But a trivial 1-4 env, 100-step cartpole episode took
  20-30+ minutes wall-clock in both `--device cpu` and the GPU default, far
  beyond what such a small workload should cost. **First-time Warp/NVRTC JIT
  compilation was the leading theory but is now ruled out**: `/isaac-sim/
  .cache/warp` (not one of the persisted bind mounts, so this only holds
  within one container's lifetime) was confirmed populated with real compiled
  kernels — including `sm120.ptx`, the Blackwell target — left over from
  earlier slow runs in the same container, yet a later run in that same
  container with a warm cache was still just as slow. So the cost is not
  (primarily) compilation. Root cause is still unknown; candidates worth
  checking next are per-step Python/Kit overhead unrelated to Warp, or
  something specific to this pre-release Isaac Lab beta's manager-based env
  architecture being inefficient at small batch sizes. Budget real time for
  this (or fix it) before relying on it for actual training throughput. A
  GPU-mode run also segfaulted once inside `warp.so!wp_cuda_graph_end_capture`
  (exit 132, `SIGILL`) at a smaller `--num_envs`; a later run with different
  `--num_envs` did not reproduce it, so its determinism is also still unclear.

## Watching it live: WebRTC streaming

Isaac Sim runs fully headless in this container (no X11/GUI), so "watching"
means connecting a WebRTC client. Enabling the `omni.kit.livestream.app`
extension binds `0.0.0.0:49100` (confirmed by checking listening ports inside
the container) — since the container uses **host networking**
(`docker inspect isaac-sim --format '{{.HostConfig.NetworkMode}}'` → `host`),
that port is directly reachable at `localhost:49100` on the machine the
container runs on, no port publishing needed.

This build has **no bundled browser client** — the `omni.kit.livestream.*`
extensions only ship a native shared library, no HTML/JS. You need NVIDIA's
**Isaac Sim WebRTC Streaming Client** (a small desktop app for
Linux/Windows/macOS, downloadable from the "Latest Release" section of
`docs.isaacsim.omniverse.nvidia.com`'s download page). Point it at
`localhost:49100` once the run prints `STREAMING_READY`.

Passing `--video` to `run_flight.sh` writes an MP4 instead/as well, which
needs no extra software.

## Operational notes

- **Only the `/tmp/dev` mount and the explicit cache mounts survive a
  container recreation — nothing else does.** `DEV_ROOT` is bind-mounted to a
  host directory (`/home/nadavc/isaac_dev_root` → container `/tmp/dev`), and
  Isaac Sim's own caches are bind-mounted too (`/isaac-sim/kit/cache`,
  `/root/.cache/ov`, `/root/.cache/pip`, GLCache, ComputeCache, logs, data,
  Documents — see the `docker run` invocation this README's setup implies).
  **Everything else inside `/isaac-sim` — apt packages (`git`, `cmake`,
  `build-essential`, ...) and anything pip-installed into Isaac Sim's own
  Python (e.g. Isaac Lab's submodules) — lives in the container's writable
  layer and is gone the moment the container is recreated.** Recreating the
  container (e.g. to add `--cap-add=SYS_PTRACE` for `py-spy`) silently loses
  both; re-run the `apt-get install` and `isaaclab.sh -i` steps after any
  recreation. The `extsUser/pegasus.simulator` symlink (`/isaac-sim/extsUser`)
  is also container-local and must be recreated the same way.
- **The bind-mounted cache directories must be owned by the container's
  `isaac-sim` user (uid 1234), not whatever host user created them.** A
  `mkdir` on the host followed by a straight bind mount leaves them
  host-owned; Isaac Sim then silently fails to acquire its shader
  `DerivedDataCache` lock (`Failed to acquire exclusive lock to data store`)
  and the RTX renderer never actually runs — physics and PX4 still work, so
  the flight *looks* successful but every recorded frame is empty/near-empty
  (a ~250-byte MP4, an empty `rgb/` dir). Fix from **inside** the container as
  root (`docker exec -u root isaac-sim chown -R isaac-sim:isaac-sim
  /isaac-sim/kit/cache /root/.cache/ov ...`) — chowning from the host side
  fails with `Operation not permitted` since the host user has no rights over
  files the container's root wrote.
- **Installing Isaac Lab into this same Isaac Sim breaks *every* scene, not
  just one, until fixed.** `isaaclab.sh -i` pip-installs its own modern
  `torch`/`nvidia-nccl-cu12` into Isaac Sim's main site-packages, and tries to
  move Isaac Sim's own older bundled copies
  (`/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/{torch,
  torchvision,nvidia}`) out of the way so the two don't collide — but that
  move is a `rename()`, which fails silently across the overlayfs copy-up
  boundary (`Invalid cross-device link`, logged as a skippable `[WARNING]`
  during install, easy to miss). The stale bundled `torch` is then left in
  place, its `libtorch_cuda.so` has an `nvidia-nccl-cu12` version mismatch
  (`undefined symbol: ncclDevCommCreate`), and loading it during Kit's core
  extension bring-up (right after `isaacsim.core.prims`, before
  `isaacsim.core.api`) silently terminates the whole process with **exit code
  0, no traceback, no crash dump** — indistinguishable from a clean shutdown
  unless you check whether extension loading actually finished. This broke
  *both* `office` and `hospital` identically; it has nothing to do with which
  scene is loaded. **Fix:** delete (don't rename) the stale bundled copies
  after installing Isaac Lab:
  `docker exec -u root isaac-sim rm -rf /isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/{torch,torchvision,torchaudio,nvidia,torch-*.dist-info,torchvision-*.dist-info,torchaudio-*.dist-info,nvidia_*.dist-info}`
  — confirmed this restores both scenes. Re-check for this after *any*
  `isaaclab.sh -i` run (including a reinstall after a container recreation).
- **Starting a second Isaac Sim process while the first is still booting
  crashes it.** The crash lands inside Kit
  (`libomni.anim.behavior.core.plugin.so`, `std::out_of_range: no null
  terminator at count`) with a stack trace that points nowhere near the real
  cause. Wait for `Simulation App Shutting Down` — teardown takes a while after
  the Python process looks done. The same GPU/Kit contention is also the likely
  cause of a one-off `libnvidia-rtcore.so` / `libnvidia-gpucomp.so` segfault
  (exit 139) seen when a flight was launched while a previous run's PX4/Kit
  processes were still alive (orphaned, not killed) — it did not recur once
  stray processes were cleaned up before relaunch, so treat any
  RTX-shader-compiler crash as a process-hygiene symptom first, not a Blackwell
  driver bug, before assuming the GPU itself is at fault.
  This is **not** a hard one-process-at-a-time limit — several concurrent
  workers are supported and are the point of `run_collection.sh` — but it is
  why that script staggers worker starts by 45 s. Kit's start-up is the
  heaviest moment of a worker's life; overlapping two of them is what triggers
  the shader-compiler crash. On this laptop's 8 GB GPU one worker is the
  practical limit anyway; a 24 GB card fits several.
- **PX4 lock files.** `/tmp/px4_lock-<N>` and `/tmp/px4-sock-<N>` survive an
  abruptly-killed PX4 and make the next instance exit immediately (`PX4
  Exiting...`) with no explanation. `px4_launch.clear_stale_locks(instance)` and
  `run_collection.sh` both remove them; do it by hand if you launch PX4 yourself.
- **Every PX4 instance needs its own working directory.** `parameters.bson`,
  `dataman` and `log/` are all relative to PX4's cwd, and `param_save_default`
  writes the parameter file in place with `O_TRUNC` under a *process-local*
  lock. Two instances sharing a directory silently corrupt each other's
  configuration every time either receives a `PARAM_SET`, which is every
  flight. `px4_launch.working_dir()` gives each one
  `build/px4_sitl_default/instance_<N>`.
- **PX4 persists every parameter it is sent**, so an experiment leaks into every
  later run from the same directory. This has already caused one multi-day
  false trail (a `--vision` run left `EKF2_GPS_CTRL=0` behind and every
  subsequent flight failed pre-flight with `ekf2 missing data`). `collect.py`
  deletes the parameter file at campaign start for exactly this reason.
- **Orphaned PX4 processes.** `make px4_sitl_default none`'s auto-launched
  `px4` can outlive its parent Python process (e.g. under a `timeout`
  wrapper). Check for stray `rcS`/`px4-param`/`build/px4_sitl_default/bin/px4`
  before the next run, or PX4's TCP port stays bound.
- **`make px4_sitl_default none` also launches PX4** after building, into an
  interactive `pxh>` shell. With no TTY attached it spins printing the prompt
  forever and produces a multi-GB log. Run it attached, or pipe through `head`.

## What's still open

- **The hospital scene has two, separate problems layered on top of each
  other.** ~~It reproducibly fails to load~~ that part is **retracted** — the
  original "crashes Kit" symptom was never scene-specific, it was the Isaac
  Lab/stale-bundled-torch conflict described in the operational notes above,
  which broke `office` identically. Confirmed: `load_indoor_scene("hospital")`
  alone loads cleanly (`HOSPITAL_LOADED_OK` in ~25s) after deleting the stale
  bundled torch. **But there is a second, real, unrelated crash underneath:**
  any run that also enables the `pegasus.simulator` extension — i.e. every
  actual flight or survey attempt, via `flight_session.boot_isaac()` — crashes
  Kit reproducibly (confirmed twice) ~2-3 seconds in, **before scene loading
  even starts**, with `std::out_of_range: no null terminator at count` inside
  `libomni.anim.behavior.core.plugin.so`. This is the same signature the
  original code comment described, just misattributed to hospital's USD
  content — it is actually triggered by `pegasus.simulator` being enabled at
  all, independent of which scene loads after it (a genuinely surprising
  finding: `office` and `simple_room` flights enable the exact same extension
  combination and never hit it, so whatever timing/state this race depends on
  isn't scene-choice alone). Tried disabling `omni.anim.behavior.core` (and
  the animation extensions that depend on it) as a workaround: it does avoid
  the crash, but takes `pegasus.simulator` down with it (a hard dependency),
  so it's not usable. This is a genuine Kit/`omni.anim.behavior.core` bug,
  not something fixable from this repo — root cause unknown, no known
  workaround. `--scene hospital` will not fly until either this upstream bug is
  fixed/patched or a different way to avoid it is found. A hospital survey
  (`survey_scene.py --scene hospital`) cannot even be computed while this crash
  stands, since surveying also enables `pegasus.simulator`.
- No home or library scene is wired in (see "Confirmed indoor scenes").
- Only `office` has been surveyed. `simple_room` should survey fine;
  `warehouse` / `full_warehouse` are untried.
- No VLA fine-tuning has been run against simulated data yet.
- `fly_direct.py`'s PD gains were tuned empirically for the Iris's ~1.6 kg
  mass on one scene; expect to retune for others.
- **The aircraft creeps a little on the floor between flights** — measured about
  0.8 m over 15 s of sitting still after a landing. Harmless for the data (every
  pose is ground truth either way) and no longer harmful to the estimator now
  that PX4's own sensor auto-calibration is off, but the cause was never
  established. Suspect PhysX resolving a small penetration between the Iris's
  landing gear and the scene floor.
- **Pegasus sends the aircraft's altitude in `HIL_SENSOR`'s temperature field.**
  `send_sensor_msgs` passes `self._sensor_data.altitude` where the MAVLink
  message expects `temperature`, which is why PX4's console reports absurd
  sensor temperatures (`90061.0 degC`). Harmless with PX4's thermal
  compensation off (the `TC_*_ENABLE` defaults), but it would silently corrupt
  any run that turned it on.
