## UPDATE 2026-07-14 — read this first, then the rest of the doc below (still valid as the detailed reference)

**Status of this doc's own action items:**
- ✅ **DONE just now**: the stale `climb_z:=600.0` tuning this doc flagged (line ~40-42 below) is fixed in both `rooster_position_ctrl` and `rooster_command_unit_R1` — now `climb_z:=700.0`, `climb_duration_sec:=5.0` added, `hover_z:=550.0` unchanged.
- ❌ **STILL NOT DONE — this is the main task for tomorrow**: none of the 5 new `Service` entries (frame publisher, depth processor, ground-truth localization, dome main, offline replay) have been added to `ROBOTICAN_SERVICES` in `mission_control.py` yet. Everything in the "Where to add things" / "Exact commands per service" / "Topics reference" sections below is still the accurate plan for that — just hasn't been executed.
- ✅ `run_ground_truth_localization.sh` now exists (this doc said it didn't, or was local/untracked) — still untracked in git as of today, but content is correct and includes an important detail this doc's original version of the plan didn't have: it **must run via `docker exec it ...`**, not directly on the host, because `rooster_ground_truth_localization.py` imports `sphera_common_interfaces` (Foxy-only, only built inside `it`). If writing the new `Service` entry for it in `mission_control.py`, make sure `machine="container"`/`container_name=ROOSTER_CONTAINER` is used (matching the gateway's own entries), not `machine="pc"`.
- ✅ Known-gotcha #2 below ("Docker single-file bind mounts break on host-side edits... affects `run_falcon.sh`'s per-file script/launch mounts") is **now fixed** for FALCON's scripts/launch (directory mounts, no restart needed for those anymore) — only FALCON's `maps/<env>.yaml` still needs a restart after editing. Doesn't affect `mission_control.py` itself.

**New bugs found and fixed 2026-07-13/14 (not in this doc's original scope, but same codebase, confirmed live via the actual UI — "arm/takeoff/land buttons in the UI - ALL WORKS GOOD"):**
1. **Yaw reversed**: `rooster_command_unit.py`'s `_MOVE_ACTIONS["turn_left"/"turn_right"]` — the `r`-axis sign was never live-validated (only the already-fixed `left`/`right` `y`-axis had been). Flipped to `turn_left: r=-1, turn_right: r=1`. Direction itself wasn't separately re-confirmed word-for-word, but very likely correct.
2. **LAND/TAKEOFF could get permanently stuck**: `RoosterUnit.arm()` in `rooster_unit.py` had an early-return (`if self.arm_pending: ...`) that never invoked `on_failed`, so `takeoff()`'s `busy_action` could get stuck forever if `arm()` was called while another arm was already in flight. Fixed by invoking `on_failed` there too.
3. **New restart-ordering gotcha**: restarting `rooster_command_unit` (inside `it`) too soon after R1 itself restarts causes a silent DDS discovery race — its `force_arm` client and `/R1/state` subscription never properly match with R1's fresh participants, and ARM/TAKEOFF silently do nothing forever (no error at all) until `rooster_command_unit` is killed and restarted again, this time with R1 fully settled. Now documented as gotcha #5 in the "Known operational gotchas" section below.
4. A transient PX4 "heading estimate not stable" preflight failure + failsafe activate/deactivate flapping was hit once (not a code bug, resolved on its own after ~30-60s).

**IMPORTANT — user is moving to a different physical PC for the next session.** This doc already anticipated that (see "machine-specific, recheck" callouts throughout below) — those callouts are exactly what to re-verify: the CycloneDDS interface/IP, the TensorRT version match, whether the recorded dome-capture sessions exist on the new machine, and whether `falcon-ros`/`ros1_bridge` Docker images need rebuilding there. The code fixes (yaw sign, arm callback gap, mission_control.py tuning) travel with the repo; the network/container/interface facts do not.

# Next session: add the ROBOTICAN sensing/dome pipeline to mission_control.py

Self-contained reference for adding `rooster_frame_dir_publisher.py`,
`rooster_depth_processor.py`, `rooster_ground_truth_localization.py`,
`rooster_dome_main.py`, and `rooster_offline_frame_dir_publisher.py` as
Streamlit-launchable services in `sparx_agency/tools/mission_control.py`,
alongside the existing `ROBOTICAN_SERVICES`. Written to be usable from a
different PC with no conversation history — every path/command below was
verified working on 2026-07-13 on the original dev machine (`pcn87652`);
recheck machine-specific bits (marked below) on the new machine.

## Where to add things in mission_control.py

- `Service` dataclass (~line 78): `name, key, group, description, cmd, env,
  docker_container, stop_extra, machine, container_name, is_interactive,
  proc_container, proc_pattern`.
- `_ENVS` dict (~line 65) already has the two envs needed — reuse them:
  - `"container"` → Foxy env inside the `it` container.
  - `"rooster_pc"` → Jazzy env on the host/PC.
- `ROOSTER_CONTAINER = "it"` (~line 43) already defined — reuse for
  `container_name=`/`proc_container=`.
- `ROBOTICAN_SERVICES` list (~line 348-447) — this is where new `Service(...)`
  entries go. Existing entries use `group="rooster_core"` /
  `"rooster_monitor"` — new ones should probably use something like
  `"rooster_sensing"` (frame publisher, depth, localization) and
  `"rooster_dome"` (dome main, offline replay) so the Streamlit tab can
  group them into their own card sections, mirroring how XTEND's tab does
  `group="dome"` / `"room_mapper"` / `"localization"` (see
  `xtend_dome` Service ~line 163 as the closest existing template — same
  concept, same output format, XTEND-specific paths).
- `ALL_SERVICES = XTEND_SERVICES + NANOOWL_SERVICES + ROBOTICAN_SERVICES`
  (~line 449) — no change needed, new entries just need to be in
  `ROBOTICAN_SERVICES`.
- Streamlit ROBOTICAN tab (~line 749 `tab_rooster`, body ~line 860-868) —
  currently only renders `group="rooster_core"`/`"rooster_monitor"` via
  `_service_cards(...)`. Adding new groups needs a corresponding
  `_service_cards([s for s in ROBOTICAN_SERVICES if s.group == "..."], states)`
  call added there, mirroring the XTEND tab's per-group sections
  (~line 777-797: `core`, `dome`, `localization`, `room_mapper`, `planner`, `pc`).
- **Existing `rooster_position_ctrl`/`rooster_command_unit_R1` entries use
  stale tuning** (`climb_z:=600.0`, no `climb_duration_sec`) — update both to
  the confirmed-working values below.
- `force_arm_rooster()` (~line 646) — unrelated helper, no change needed,
  just confirms the same `CYCLONEDDS_URI` path is already correct there.

## Confirmed-working values (verified live, 2026-07-13)

**Gateway climb tuning** — code defaults are too weak/short to leave the
ground in Sphera:
```
climb_z:=700.0  hover_z:=550.0  climb_duration_sec:=5.0
```
(`hover_z` code default of 550 was already fine — only `climb_z`/duration
needed bumping.)

**Left/right axis fix**: already committed
(`rooster_command_unit.py`'s `_MOVE_ACTIONS`, `left`/`right` sign was
inverted vs. standard MAVLink roll convention — fixed, don't re-flip it).

**DA3 engine**: use
`~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_546x364.engine`
— **not** `run_depth_processor.sh`'s stale default filename
(`*_fp16_trt10.engine`, doesn't exist). 546x364 is a multiple of 14 (ViT
patch size) at the exact same 3:2 aspect as ROBOTICAN's native 540x360, so
no crop/distortion.

**TensorRT version must match exactly**: `pip install tensorrt` resolves to
the newest version (11.x as of this writing) — **wrong**, incompatible
engine format. Must be the version matching
`~/depth_anything_ws/src/ros2-depth-anything-v3-trt/da3_venv`'s tensorrt
(check with `da3_venv/bin/python -c "import tensorrt; print(tensorrt.__version__)"`
— was `10.16.1.11` on the original machine, **recheck on the new machine,
don't assume**). Install the exact matching version into
`~/GIT/TheAgency/venv`: `venv/bin/pip install "tensorrt==<exact version>"`.

**Camera calibration**: `sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml`
— an *approximation* (assumed 130x90 FOV, not verified against Sphera's
real camera definition), native 540x360 resolution. `camera_info_mode:=base`.

**Cage-mask inpainting is currently disabled** (all-zero mask at
`config/cage_static_mask.npy`, correct 540x360 shape but no actual masking)
— the fixed-brightness-threshold approach proved too fragile even with
real diverse calibration frames (max persistence landed just under the 80%
cutoff). Don't re-enable without a better algorithm; not blocking anything.

## Exact commands per service

All host-side (`machine="pc"`, `env="rooster_pc"`) commands assume this
preamble (already what `_ROOSTER_PC_ENV` provides):
```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml
```
All container-side (`machine="container"`, `env="container"`,
`container_name="it"`) commands assume `_CONTAINER_ENV` (already correct in
mission_control.py, same `CYCLONEDDS_URI` file via a different in-container
path — verified identical content, not a stale duplicate).

**Note on `venv` vs `.venv`**: host-side ROBOTICAN scripts run with
`/home/$USER/GIT/TheAgency/venv/bin/python` (the ROS-enabled venv with
rclpy/tensorrt/pycuda/gi installed) — **not** `.venv` (the core-algorithms-only
Poetry env CLAUDE.md's general test instructions refer to). Getting this
wrong is a common source of `ModuleNotFoundError: rclpy`.

1. **Frame publisher** (host) — `sparx_agency/robots/ROBOTICAN/rooster_frame_dir_publisher.py`
   (wrapper: `run_rooster_frame_dir_publisher.sh`):
   ```
   --rooster-id R1 --out-dir /tmp/rooster_frames --port 5001
   ```
   `--port` must match the gateway's `video_port` param (default 5001 both
   sides). Publishes native 540x360 frames, no crop/resize — the DA3 engine
   is sized to match directly.

2. **DA3 depth processor** (host) — `sparx_agency/robots/ROBOTICAN/rooster_depth_processor.py`
   (wrapper: `run_depth_processor.sh` — **this wrapper was being actively
   edited for FALCON integration in a parallel session as of 2026-07-13,
   check its current content before assuming the args below are still
   exactly what's in the file**):
   ```
   -p engine_path:="$HOME/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_fp16_546x364.engine"
   -p config_yaml:="$HOME/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/config/camera_rooster_calib_540_360.yaml"
   -p frame_path_topic:=/R1/rgb_frame_path
   -p camera_info_topic:=/R1/camera_info
   -p camera_info_mode:=base
   -p depth_topic:=/R1/depth_m
   -p depth_path_topic:=/R1/depth_frame_path
   -p depth_dir:=/tmp/rooster_depth
   ```
   Real-time performance confirmed: ~7-9ms/frame on an RTX 4090.

3. **Ground-truth localization** (container `it` — needs
   `sphera_common_interfaces`, only importable there, simulator-only, never
   applicable to a real drone) —
   `sparx_agency/robots/ROBOTICAN/rooster_ground_truth_localization.py`.
   No wrapper script from me exists in git as of this writing, but the user
   created `run_ground_truth_localization.sh` locally (untracked as of
   2026-07-13 — check if it's been committed since):
   ```
   --ros-args -p rooster_id:=R1
   ```
   Publishes `/R1/localization` (PoseStamped) + `/R1/localization_source`
   (String, `"sphera_ground_truth"`) by republishing `/R1/sphera/state`
   (`sphera_common_interfaces/SpheraPawnState`) — yaw-only planar
   orientation, quaternion encoded as `z=sin(yaw/2), w=cos(yaw/2)`.

4. **Dome main / full mission** (host) — `sparx_agency/robots/ROBOTICAN/rooster_dome_main.py`
   (wrapper: `run_rooster_dome_main.sh`):
   ```
   --rooster-id R1 --pose-topic /R1/localization
   --out-dir ~/rooster_dome_capture --capture-interval-sec 1.0 --yaw-bucket-deg 30.0
   ```
   Verified end-to-end twice: video_on → arm → takeoff → 360° rotate (4×90°
   chunks) → capture → land → disarm, producing complete `.jpg`/`.json`/`.npy`
   triples (49 each run). `is_interactive` not needed — runs to completion
   and exits (unlike `ui.py`, which needs a real terminal).

5. **Offline replay (no Sphera needed)** (host) —
   `sparx_agency/robots/ROBOTICAN/rooster_offline_frame_dir_publisher.py`
   (wrapper: `run_rooster_offline_replay.sh`):
   ```
   --session-dir ~/rooster_dome_capture/latest --rooster-id R1 --rate 2.0 [--loop]
   ```
   Republishes a previously-recorded session onto the exact same topics as
   the live pipeline. Two full recorded sessions exist from 2026-07-13:
   `~/rooster_dome_capture/latest` and `~/rooster_dome_capture_2/latest`
   (49 triples each, all with valid non-zero ground-truth pose) — **these
   are local to the original machine's home directory, won't exist on a
   different PC unless copied over separately.**

## Topics reference

```
/R1/cmd_nav              std_msgs/String   (JSON in: {"action": "...", "value": ...})
/R1/rooster_status        std_msgs/String   (JSON out: armed/airborne/battery_pct/video_on)
/R1/rgb_frame_path        std_msgs/String   ("{path} {sec} {nanosec}")
/R1/depth_frame_path      std_msgs/String   (same format)
/R1/localization          geometry_msgs/PoseStamped
/R1/localization_source   std_msgs/String
/R1/sphera/state          sphera_common_interfaces/SpheraPawnState  (Foxy-only, ground truth)
/R1/fcu/battery           fcu_driver_interfaces/Battery             (Foxy-only)
/R1/state                 rooster_manager_interfaces/RoosterState   (Foxy-only)
```
"Foxy-only" = only importable/echoable inside the `it`/`R1`/`drone_simulator`
containers (Foxy), not from the Jazzy host directly — confirmed repeatedly
this session (`ModuleNotFoundError`/`invalid` message type errors on host).

## Docker containers

- `it` (image `sphera-backend:rooster-with-sparx`) — has the Rooster ROS2
  interfaces + `sparx_agency` bind-mounted read-write at
  `/home/rooster/sparx_agency` (edits on the host are live inside
  immediately, no rebuild). This is where the gateway and ground-truth
  localization run.
- `R1` (image `sphera-backend:rooster`) — the actual FCU/drone backend,
  **spawned internally by the Sphera app itself** (compiled, closed-source;
  `drone_simulator` has Docker socket access and spawns it). No compose
  file/script/env var controls its launch from outside.
- `drone_simulator` (image `sphera:drone_simulator`) — the Sphera app
  itself, managed via `~/.sphera/sphera-restart.sh` (wraps
  `docker compose up -d run_sphera` in `~/.sphera/`, using
  `~/.sphera/docker-compose.yaml`).

All three use `NetworkMode: host` — no container/host network boundary.

## The CycloneDDS network fix (read `DOME_CAPTURE_README.md` for the full story)

This machine had two active NICs; `R1` (Sphera-spawned, uncontrollable) always
picked the wrong one, and the "wrong" one turned out to be the *only* one
that actually works given we can't configure `R1` at all. Everyone else is
pointed at whichever interface `R1` lands on via the single shared file
`~/rqs_iai_ws/src/cyclonedds.xml` (bind-mounted as `/etc/cyclonedds.xml` in
`it`/`drone_simulator`). **This is almost certainly machine-specific** — a
different PC will likely have different interface names/IPs. On a new
machine: check `docker logs R1 | grep "selected arbitrarily"` to find out
which interface `R1` actually lands on there, and point
`~/rqs_iai_ws/src/cyclonedds.xml`'s `NetworkInterfaceAddress` at that same
one. Don't assume `172.16.17.10` carries over.

## Known operational gotchas (from a separate, already-working FALCON↔ROBOTICAN
bridging session on 2026-07-13 — check these before deep-diagnosing a "new" bug)

1. **Restart ordering**: if FALCON's `roslaunch` restarts, `ros1_bridge`
   must be manually restarted too (new `rosmaster` each time, bridge
   doesn't reconnect on its own).
2. **Docker single-file bind mounts break on host-side edits** (inode
   swap on save). **Fixed 2026-07-14 for `run_falcon.sh`'s `adapter/scripts/`
   and `adapter/launch/` mounts** — those are now whole-directory mounts, so
   host edits are live immediately, no restart needed. `maps/<env>.yaml` is
   still a single-file mount on purpose (FALCON's image ships upstream
   example maps not in this repo's `maps/` dir; a directory mount would
   shadow them) — editing the current env's map YAML still needs the
   container recreated.
3. **Duplicate background processes are the default failure mode** in this
   long-lived multi-terminal workflow — always `pgrep -af <script_name>`
   before assuming a timing/race bug is something new.
4. **Sphera world-frame map bounds must match the drone's real spawn
   position**, not the origin — check `/R1/localization`/`/R1/sphera/state`
   for the actual position before authoring a new FALCON map YAML.
5. **Restarting `rooster_command_unit` too soon after R1's own restart
   silently breaks it** (found 2026-07-14) — its `force_arm` service client
   and `/R1/state` telemetry subscription can fail to properly match with
   R1's freshly-restarted DDS participants, so ARM/TAKEOFF silently do
   nothing forever (no error) until it's restarted again with R1 fully
   settled. Symptom looks identical to a code bug (repeating "Arm already
   in progress" log spam) but a fresh one-off `ros2 service call` to the
   same `force_arm` service succeeds instantly, proving the service itself
   is fine. Rule: after relaunching the Sphera sim/R1, wait for R1 to fully
   settle (boot-complete, no more "Communication lost" in `docker logs R1`)
   before restarting `rooster_command_unit` — don't restart both together.

## Verification checklist for the new session

1. `docker ps` — confirm `it`/`R1`/`drone_simulator` are up (or start them).
2. `docker logs R1 2>&1 | grep "selected arbitrarily"` — confirm which NIC
   R1 lands on *on this machine*; update `cyclonedds.xml` if different from
   `172.16.17.10`.
3. Start the gateway (container `it`) with the tuned climb params, confirm
   `ros2 topic echo /R1/rooster_status --once --full-length` shows real
   (non-zero) `battery_pct`.
4. Add the four/five new `Service` entries to `ROBOTICAN_SERVICES` in
   `mission_control.py`, using the commands above.
5. `streamlit run sparx_agency/tools/mission_control.py`, launch each new
   card from the ROBOTICAN tab, confirm it starts (green/running state) and
   its log file (`/tmp/<key>.log`) shows the expected "ready" message.
