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
- **Only one Isaac Sim process at a time.** A second one crashes the first
  inside Kit (`libomni.anim.behavior.core.plugin.so`, `std::out_of_range: no
  null terminator at count`) with a stack trace that points nowhere near the
  real cause. Wait for `Simulation App Shutting Down` — teardown takes a while
  after the Python process looks done. The same GPU/Kit contention is also the
  likely cause of a one-off `libnvidia-rtcore.so` / `libnvidia-gpucomp.so`
  segfault (exit 139) seen when a flight was launched while a previous run's
  PX4/Kit processes were still alive (orphaned, not killed) — it did not
  recur once stray processes were cleaned up before relaunch, so treat any
  RTX-shader-compiler crash as a process-hygiene symptom first, not a Blackwell
  driver bug, before assuming the GPU itself is at fault.
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
  workaround. `run_flight.sh --scene hospital` will not fly until either this
  upstream bug is fixed/patched or a different way to avoid it is found. A
  hospital route survey (`probe_scene.py --scene hospital`) cannot even be
  computed while this crash stands, since the survey step also enables
  `pegasus.simulator`.
- No home or library scene is wired in (see "Confirmed indoor scenes").
- No NavDP fine-tuning has been run against simulated data yet.
- `fly_direct.py`'s PD gains were tuned empirically for the Iris's ~1.6 kg
  mass on one scene; expect to retune for others.
