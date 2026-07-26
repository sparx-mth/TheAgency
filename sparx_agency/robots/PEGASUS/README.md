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
  adapters/scene.py                          load_indoor_scene() + SCENE_SURVEYS (surveyed spawn + route per scene)
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
running. With it, PX4's first heartbeat arrives 1.4 s into simulated time.

## Running it

Everything must run inside a live Isaac Sim process (Isaac Sim's own Python,
not the repo's `.venv` — it needs `omni`/`carb`/`pegasus.*`, which only exist
inside a running Kit app).

```bash
# One-time setup, inside the isaac-sim container:
sparx_agency/robots/PEGASUS/setup/install.sh /path/to/dev/root
```

Then fly it **from the host** with the launcher, which handles syncing the
repo into the container, clearing stale PX4 locks, and starting the run:

```bash
sparx_agency/tasks/planning/sim_flight_recording/run_flight.sh --scene office
```

See `tasks/planning/sim_flight_recording/README.md` for the flags, the flight
modes, how the per-scene routes are surveyed, and why PX4 is given a vision
pose instead of GPS.

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

- **Only one Isaac Sim process at a time.** A second one crashes the first
  inside Kit (`libomni.anim.behavior.core.plugin.so`, `std::out_of_range: no
  null terminator at count`) with a stack trace that points nowhere near the
  real cause. Wait for `Simulation App Shutting Down` — teardown takes a while
  after the Python process looks done.
- **PX4 lock files.** `/tmp/px4_lock-0` and `/tmp/px4-sock-0` survive an
  abruptly-killed PX4 and make the next instance exit immediately (`PX4
  Exiting...`) with no explanation. `px4_launch.clear_stale_locks()` and
  `run_flight.sh` both remove them; do it by hand if you launch PX4 yourself.
- **Orphaned PX4 processes.** `make px4_sitl_default none`'s auto-launched
  `px4` can outlive its parent Python process (e.g. under a `timeout`
  wrapper). Check for stray `rcS`/`px4-param`/`build/px4_sitl_default/bin/px4`
  before the next run, or PX4's TCP port stays bound.
- **`make px4_sitl_default none` also launches PX4** after building, into an
  interactive `pxh>` shell. With no TTY attached it spins printing the prompt
  forever and produces a multi-GB log. Run it attached, or pipe through `head`.

## What's still open

- The Pegasus and PX4-Autopilot checkouts and the built `px4` binary live
  under the container's `/tmp` by default. **Move `DEV_ROOT` to a bind-mounted
  host directory before the container is ever recreated, or this work vanishes
  with it** — the container is currently the only copy.
- No home or library scene is wired in (see "Confirmed indoor scenes").
- No NavDP fine-tuning has been run against simulated data yet.
- `fly_direct.py`'s PD gains were tuned empirically for the Iris's ~1.6 kg
  mass on one scene; expect to retune for others.
