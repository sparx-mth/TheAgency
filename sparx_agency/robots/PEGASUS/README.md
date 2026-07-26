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
  adapters/scene.py                          load_indoor_scene() -- reference a stock CDN environment
  adapters/vehicle.py                        PegasusIrisVehicle -- spawn Iris (+ optional PX4 backend) + RGBD camera
  setup/pegasus_isaac6_compat.patch          the compatibility + heartbeat-deadlock fixes, pinned to commit e13dc659
  setup/install.sh                           clone+patch Pegasus, clone+build PX4 SITL
```

The recording writer (`sim_extract.py`) and the driver/flight harnesses
(`record_flight.py`, `fly_direct.py`, `fly_and_watch.py`) live outside this
package since they're platform-agnostic / mission-level respectively — see
below.

## Running it

Everything here must run inside a live Isaac Sim process (Isaac Sim's own
Python, not the repo's `.venv` — it needs `omni`/`carb`/`pegasus.*`, which only
exist inside a running Kit app).

```bash
# One-time setup inside the isaac-sim container:
sparx_agency/robots/PEGASUS/setup/install.sh /path/to/dev/root

# Smoke-test the full chain (scene + vehicle + PX4 + camera + recorder):
/isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/record_flight.py \
    --pegasus-root /path/to/dev/root/PegasusSimulator/extensions/pegasus.simulator \
    --px4-dir /path/to/dev/root/PX4-Autopilot \
    --scene simple_room --out-dir /path/to/recordings/simple_room_smoke
```

`record_flight.py` is a **smoke test, not a piloted flight** — it spawns the
vehicle and steps the sim so PX4 and the camera come up, but sends no
arm/takeoff command. Scripted or policy-driven missions are future work.

**Verified end-to-end on 2026-07-26**: ran the full chain against `simple_room`
— scene load, PX4 SITL connects to the Pegasus vehicle over MAVLink (TCP
4560), the camera streams RGBD, and `sim_extract.py` writes a 30-frame
recording. Loaded it back with `recording.load_recording()` (repo `.venv`,
outside Isaac Sim): correct `(392, 504)` depth/RGB shapes, intrinsics matching
this file, and real per-frame ground-truth poses (`poses.npy`) — flat/constant
in this run since no arm/takeoff command was sent, as expected for a
stationary smoke test. `future_path_body`/`goal_body` (the methods the ESDF
label generator calls) also ran correctly against the output.

One implementation bug found and fixed during this: `adapters/vehicle.py`'s
`_IRIS_USD` was initially missing the `pegasus/simulator/` path segment (the
asset actually lives at `.../pegasus.simulator/pegasus/simulator/assets/...`,
not `.../pegasus.simulator/assets/...`), which silently spawned an empty
placeholder prim instead of the Iris model.

Operational note: `make px4_sitl_default none`'s auto-launched `px4` process
tree can survive its parent Python process being killed (e.g. by a `timeout`
wrapper) — if a run is interrupted, check for and kill orphaned
`rcS`/`px4-param`/`build/px4_sitl_default/bin/px4` processes before the next
run, or PX4's TCP port stays bound.

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
`localhost:49100` once the script prints `STREAMING_READY`.

## Getting the drone to actually fly: three more bugs, one fundamental one

Getting past "spawns and sits there" to "arms, takes off, flies a route" hit
four real bugs, three fixable and one that isn't (for now):

1. **A deadlock in our own script.** Any *blocking* pymavlink call
   (`wait_heartbeat()`, `motors_armed_wait()`) stops `world.step()` from
   running while it waits — which stops Pegasus from feeding PX4 sensor
   ticks — which means PX4 can never finish booting and send the heartbeat
   we're blocked waiting for. Fix: never block; poll MAVLink non-blockingly
   from inside the stepping loop instead (see the (retired) `fly_and_watch.py`
   for the pattern).

2. **Pegasus's own `PX4LaunchTool` launches PX4 from a broken working
   directory.** It runs PX4 with `cwd=tempfile.TemporaryDirectory()` — a
   fresh empty directory every time — but PX4 sources
   `$PWD/etc/init.d/rc.vehicle_setup` at boot, which only exists under the
   real build output, `px4_dir/build/px4_sitl_default`. Every run failed with
   `rc.vehicle_setup: No such file` until PX4 was launched manually with the
   correct `cwd` (`px4_autolaunch=False` on `PegasusIrisVehicle`, see the
   retired `fly_and_watch.py._launch_px4`).

3. **Stale PX4 lock files.** `/tmp/px4_lock-0` / `/tmp/px4-sock-0` survive an
   abruptly-killed PX4 process and make the next PX4 instance exit
   immediately (`PX4 Exiting...`) with no explanation. Remove them before
   every run if a prior run was interrupted.

4. **A real deadlock inside Pegasus's own backend.** PX4 is built with
   `ENABLE_LOCKSTEP_SCHEDULER` — its entire internal clock, including
   heartbeat generation, is driven by receiving `HIL_SENSOR` data from the
   simulator. But Pegasus's `PX4MavlinkBackend.update()` has a hard
   `if not self._received_first_hearbeat: ...; return` — it refuses to send
   *any* sensor data until it sees PX4's heartbeat first. Mutual wait, never
   resolves. Fixed in `pegasus_isaac6_compat.patch`: send sensor data
   unconditionally; check for the heartbeat opportunistically instead of
   gating on it.

5. **The fundamental one, found only by direct instrumentation: Isaac Sim
   6.0.1 stops dispatching `World.add_physics_callback()`-registered
   callbacks after ~2 calls following `world.reset()`.** Confirmed two ways:
   a call-counter wrapped around `Multirotor.update()` (Pegasus's own
   physics-callback method — this is what applies rotor thrust *and* what
   feeds PX4 sensor data) printed `call #1`, `call #2`, then never again,
   across 750+ further `world.step()` calls, with **zero exceptions** raised
   anywhere. And reading the vehicle's *live* PhysX pose directly
   (`RigidPrim.get_world_poses()`, bypassing Pegasus's own cached
   `self._state`) showed the rigid body **free-falling normally** — PhysX
   itself is fine; only the Python-level state cache and everything gated on
   the physics-callback (sensor pushes to PX4, force application) had
   silently stopped updating after the first couple of ticks. This explains
   fix #4 not being sufficient on its own, and why the drone never moved in
   earlier recordings even without any PX4 involvement at all.

   No fix for the callback dispatch itself was found (would need
   understanding what changed in Isaac Sim 6.0.1's Kit/PhysX callback wiring,
   or whether `isaacsim.core.simulation_manager` — referenced throughout the
   Kit logs — is the intended replacement API). **Workaround, and what
   actually works today:** `tasks/planning/sim_flight_recording/fly_direct.py`
   sidesteps the callback system entirely — it manually calls
   `vehicle.update_state(dt)` and `vehicle.apply_force()`/`apply_torque()`
   once per step from its own loop (which reliably keeps running), driven by
   a simple world-frame PD controller (altitude + XY position hold, plus
   attitude leveling). No PX4, no MAVLink. `fly_and_watch.py` (the PX4/MAVLink
   path, fixes 1-4 above) is left in the tree as a documented, known-incomplete
   path — it still relies on the same broken callback dispatch to feed PX4,
   so getting it fully working would need the same manual-driving treatment
   applied there too, feeding PX4 the state/sensor data by hand each step.

## Running it

Everything here must run inside a live Isaac Sim process (Isaac Sim's own
Python, not the repo's `.venv` — it needs `omni`/`carb`/`pegasus.*`, which only
exist inside a running Kit app).

```bash
# One-time setup inside the isaac-sim container:
sparx_agency/robots/PEGASUS/setup/install.sh /path/to/dev/root

# Fly it (direct Python control, no PX4) and watch it live:
/isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/fly_direct.py \
    --pegasus-root /path/to/dev/root/PegasusSimulator/extensions/pegasus.simulator \
    --scene simple_room --out-dir /path/to/recordings/simple_room_flight \
    --altitude 2.0 --cruise-s 15
# then connect the Isaac Sim WebRTC Streaming Client to localhost:49100
# once the log prints STREAMING_READY. Pass --no-stream to skip streaming.
```

`fly_direct.py` climbs to `--altitude`, flies a slow 3 m forward/back sine-wave
cruise for `--cruise-s` seconds, then descends — a real, verified flight
(non-flat, non-frozen `poses.npy`; validated by loading the output through
`recording.load_recording()`), not just a stationary smoke test.

**Verified end-to-end on 2026-07-26**: `simple_room`, both with and without
streaming enabled. 230 frames recorded over a 23 s mission; the loaded
recording's `poses.npy` shows real X motion tracking the commanded sine-wave
cruise (e.g. `x ≈ 0.63 → 1.03 → -2.70 → 0.30` across the captured frames) and
near-zero yaw throughout (attitude leveling holding steady, no tumbling).

`record_flight.py` still exists as the original stationary infra smoke test
(scene + vehicle + camera + recorder, no flight control at all) — useful for
quickly checking the base chain still works without needing a flight
controller in the loop.

One implementation bug found and fixed along the way: `adapters/vehicle.py`'s
`_IRIS_USD` was initially missing the `pegasus/simulator/` path segment (the
asset actually lives at `.../pegasus.simulator/pegasus/simulator/assets/...`,
not `.../pegasus.simulator/assets/...`), which silently spawned an empty
placeholder prim instead of the Iris model.

## What's still open (not done in this pass)

- The PX4/MAVLink path (`fly_and_watch.py`) is not fully working — see bug #5
  above. `fly_direct.py` (no PX4) is the flight path that actually works.
- No NavDP fine-tuning has been run against simulated data yet.
- No home/library scene is wired in (see above).
- The Pegasus/PX4-Autopilot checkouts and the built `px4` binary live under
  the container's `/tmp` by default in `install.sh` — move `DEV_ROOT` to a
  bind-mounted host directory before the container is ever recreated, or this
  work vanishes with it.
- `fly_direct.py`'s PD gains were tuned empirically for the Iris's ~1.6 kg
  mass on this one scene; expect to retune for other scenes/missions.
