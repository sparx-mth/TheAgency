# Lessons

Errors, gotchas, and failure modes that cost real debugging time — kept here so they're
never re-derived from scratch. Check here before debugging anything that feels familiar.

Format per entry:
- **Symptom** — what you actually observed
- **Root cause** — what was really going on (often different from the first suspect)
- **Fix / workaround** — what resolved it, or the current best mitigation if unresolved
- **Don't** — anything that looked like a fix but wasn't, or made it worse

---

## 2026-07-29 — new detector container: numpy/CLIP/TensorRT-version traps, in sequence

**Symptom:** Building `docker/Dockerfile.detector` (torch + ultralytics on top of the
`perception` layer, for the YOLO-World detector) hit three separate, unrelated failures, each
only visible after fixing the one before it: (1) `pip install torch` timed out mid-download
with a default 120s timeout, (2) after fixing that, `import pycuda.autoinit` failed with
`AttributeError: _ARRAY_API not found`, (3) after fixing that, the detector node failed with
`RuntimeError: Failed to deserialize one of the engines.`

**Root cause:** Three independent issues, not one:
1. torch's CUDA 13 wheels are large; pip's default read timeout can trip mid-stream well
   before anything is actually wrong (same as `Dockerfile.perception`'s own tensorrt install
   already documents) — needs a much longer `--timeout`, not just `--retries`.
2. `ultralytics` declares `numpy` unconstrained, so pip resolved `numpy==2.2.6` — silently
   overriding `perception`'s deliberate `numpy==1.26.4` pin (there specifically so `pycuda`,
   compiled against numpy 1.x's C-API, works). numpy 2.x breaks pycuda outright, not just a
   warning.
3. `git+https://github.com/ultralytics/CLIP.git` "succeeded" but built a wheel named literally
   `UNKNOWN-0.0.0` — the base image's stock pip 22.0.2/setuptools 59.6.0 mis-resolve that
   repo's `pyproject.toml` metadata, so no `clip` module was actually importable despite a
   clean-looking install log. Separately, once TensorRT engines finally loaded, they were
   built on the HOST (TensorRT 10.16.1.11) and failed to deserialize in the container
   (TensorRT 10.15.1.29) — `Dockerfile.perception` already documents that engines are locked
   to the exact TRT build that produced them, not just a major-version match.

**Fix / workaround:** In order: raised the pip timeout to 600s/8 retries for the torch
install; added an explicit `pip3 install numpy==1.26.4` as the LAST install step (after
ultralytics/CLIP, so nothing downstream re-upgrades it) — confirmed `cv2` (a *different*
package, pip's own `opencv-python`, not `perception`'s apt `python3-opencv`) tolerates the
downgrade fine despite pip flagging it as a dependency conflict; upgraded
`pip`/`setuptools`/`wheel` before the CLIP install so it resolves the real package name; and
rebuilt just the TensorRT engine (not the ONNX export — that's portable) inside the container
that will actually run it.

**Don't:** Don't trust a "Successfully installed X" pip log line as proof X actually works —
check the real package name landed (not `UNKNOWN`) and that nothing else silently got
up/downgraded as a side effect. Don't assume a TensorRT engine built on one machine/container
works in another with a different TensorRT version, even the same GPU — rebuild the engine
(cheap, seconds) inside wherever it will actually run, and check `Dockerfile.perception`'s own
comments before re-debugging a version-mismatch class already documented there.

---

## 2026-07-29 — stale `ros2` CLI daemon kept showing zero R1 topics after switching networks/domains back and forth

**Symptom:** After going real-drone (`ROS_DOMAIN_ID=1`, CycloneDDS pointed at the real
network) and then switching back to Sphera (`ROS_DOMAIN_ID=9`, CycloneDDS reverted to
`172.16.17.10`), `ros2 topic list`/`ros2 topic echo` inside `it` showed ZERO `/R1/...`
topics, even though R1's own container logs showed a perfectly healthy boot (FCU
connected, position valid, no CDR/NIC errors) and both `it`'s and R1's cyclonedds
configs correctly agreed on domain 9 / `enp129s0`/`172.16.17.10`.

**Root cause:** `ros2` (Foxy) CLI commands are served by a long-lived background
`_ros2_daemon` process that caches graph/discovery state and does NOT restart or
rebind when `CYCLONEDDS_URI`/`ROS_DOMAIN_ID` change in a later shell — it keeps
running with whatever network state it had when it first started. A daemon that had
been running since earlier in the session (started during the real-drone/domain-1
work) kept serving stale results to every later `ros2 topic list` call, even ones that
correctly exported `ROS_DOMAIN_ID=9` — the CLI just asks the existing daemon, it
doesn't re-discover itself. `ros2 daemon stop` itself hung (never completed) rather
than fixing it.

**Fix / workaround:** `docker exec it kill -9 <daemon-pid>` (find via `ps aux | grep
daemon` inside `it`) — a fresh daemon spins up automatically on the next `ros2` CLI
call and immediately sees the correct, current graph.

**Don't:** Don't assume "both configs agree and the container's own logs look healthy"
rules out a networking/discovery problem — the `ros2` CLI's own daemon is a separate,
stateful layer that can lag behind a config change made after it started. Any time
`ROS_DOMAIN_ID`/`CYCLONEDDS_URI` changes mid-session (e.g. switching between Sphera and
a real drone), kill and let the daemon restart before trusting a "no topics" result.

---

## 2026-07-27 — duplicate method definition silently shadowed the real navigation output

**Symptom:** FALCON's `waypoint_follower` logged `nav=RUN done=False` continuously and
appeared to be actively navigating after a BEV click, but the drone never moved.
`cmd_vel_gate`'s "commands passed" counter was frozen at a fixed number instead of
climbing — meaning `/cmd_vel_raw` was never actually being published on any tick.

**Root cause:** `waypoint_follower_node.py` defined `_publish_twist_multi` TWICE in the
same class. The correct version (with `vz` altitude-hold support, yaw-pitch-bias,
`cmd_vy_sign`) came first; a much older, simpler version (no `vz` parameter at all) was
defined later in the file and silently overrode it — that's just how Python class bodies
work, no error at import time. Every tick called
`self._publish_twist_multi(cmd.vx, cmd.vy, cmd.wz, vz=self._alt_vz)`, which raised
`TypeError: _publish_twist_multi() got an unexpected keyword argument 'vz'` against the
live (second) definition — right before it would have reached `.publish()`. The
exception was swallowed by rospy's timer callback error handling (logged, not fatal),
so the node stayed "alive" and kept logging as if navigating.

**Fix / workaround:** Deleted the older, shadowing definition. Confirmed via
`grep -n "_publish_twist_multi" *.py` before assuming a single definition existed.

**Don't:** Don't trust a log line saying `nav=RUN` as proof that output is actually being
published — check the downstream gate/counter (`cmd_vel_gate`'s passed count) for real
movement, and grep for duplicate method names in a file before deep-diagnosing a "looks
alive but doesn't work" node.

---

## 2026-07-27 — half-applied coordinate handedness fix (yaw negated, position not)

**Symptom:** The drone's reported heading looked correct at low yaw, but flying it
(pure `forward` command) produced a real-world displacement whose Y sign didn't match
`tan(yaw)`'s sign — i.e. the map/localization built in the mirrored lateral direction
from the drone's actual physical motion.

**Root cause:** `rooster_ground_truth_localization.py` converts Sphera/Unreal telemetry
(left-handed, clockwise-positive yaw) to ROS (right-handed, counter-clockwise-positive).
A previous fix (documented inline, dated earlier) negated yaw to fix a "map built behind
the drone instead of in front" bug — but a full handedness conversion needs ONE LINEAR
AXIS negated too, and that half was never done. `position.y` was passed straight
through unflipped, leaving rotation and translation using inconsistent conventions.

**Fix / workaround:** Verified quantitatively before touching code: commanded a pure
`forward` move, recorded ground-truth `(dx, dy)` before/after, and compared
`dy/dx` against `tan(yaw)` from the reported orientation. The sign only matched after
mentally flipping `dy`, confirming which axis and which fix. Negated `position.y` in
`rooster_ground_truth_localization.py`.

**Don't:** Don't assume a rotation-only handedness fix (negating yaw alone) is complete —
check whether a corresponding linear axis needs the same treatment. And don't guess which
axis without a live before/after ground-truth measurement; a wrong guess flips it a
different, still-wrong way.

---

## 2026-07-27 — fixing the Y-axis sign broke the map/BEV bounds (expected, but easy to miss)

**Symptom:** Immediately after the Y-axis localization fix above, the drone appeared to
spawn "outside the BEV click map" and RViz's 3D voxel view showed no drone marker at all.

**Root cause:** `maps/sphera_jail.yaml` (`init_y`, `map_min_y`/`map_max_y`, `box_min_y`/
`box_max_y`, `vbox_min_y`/`vbox_max_y`) and the launch args documented in the
`fly-rooster-sphera` skill (`bev_ymin`/`bev_ymax`/`goal_y`) were all tuned against the
drone's OLD (unflipped) reported Y position. The drone's real, physical spawn point
never moved — but the sign fix above changed which number `/R1/localization` reports for
it (`+14.66` → `-14.66`), which took every Y bound out of range at once. The map yaml's
own comments already document that being out of bounds FATAL-aborts `exploration_node`
with an out-of-range voxel-array index.

**Fix / workaround:** Negated AND min/max-swapped every Y bound in both files (negating
a range reverses which end is the min and which is the max).

**Don't:** Don't change a world-frame axis sign convention in one place without auditing
every spatial config tuned against the old convention (map bounds, BEV click bounds,
default goal coordinates) — they're a matched pair, not independent.

---

## 2026-07-27 — twist-control adapter's stop-watchdog cancels any in-progress takeoff/land

**Symptom:** Sending `arm` then `takeoff` armed the drone, but the climb got cancelled
(`"Climb cancelled - holding z=..."`) within ~50-100ms every time, regardless of
throttle/altitude-hold tuning. Same thing happened trying to `land` while airborne.

**Root cause:** `rooster_twist_control_adapter.py` runs a 20Hz watchdog that publishes
`{"action": "stop"}` on `/R1/cmd_nav` whenever it hasn't seen a `/cmd_vel` message in the
last `cmd_timeout_sec` (default 0.4s) — reasonable as a "planner went silent, stop
moving" safety net. But `rooster_command_unit.py`'s `RoosterUnit.stop()` unconditionally
cancels ANY in-progress `takeoff`/`land` sequence (`busy_action in ("takeoff", "land")`),
with no distinction between "the planner's own no-op stop" and "the user wants to abort
a takeoff." Since FALCON only publishes `/cmd_vel` while actively driving toward a goal,
this adapter spams stop essentially constantly outside of active navigation — killing
any manual arm/takeoff/land test within one or two 50ms timer ticks.

**Fix / workaround:** Kill (or don't yet start) the twist-control adapter before any
manual arm/takeoff/land test; only start/restart it once the drone is confirmed hovering
and you're about to test click-to-fly.

**Don't:** Don't leave the twist-control adapter running "just in case" during a manual
flight test — it will silently sabotage takeoff/land and the failure looks like a
throttle/timing bug in `rooster_command_unit.py`, not a second process fighting it.

---

## 2026-07-28 — camera rig/mount visible in its own FOV, fused as a permanent phantom wall

**Symptom:** FALCON's A* planner reported "boxed in - no A* route" almost immediately
after a BEV click, falling back to NavDP, which then searched/spun erratically — and
the resulting map looked noisy/speckled (a "V"-shaped or cauliflower-like blob rather
than clean walls).

**Root cause:** The bottom ~25% of every single RGB frame showed a near-constant
`~0.17-0.35m` depth reading, completely stable across consecutive frames regardless of
drone motion — confirmed by sampling `/tmp/rooster_depth/*.npy` directly with numpy.
That's the drone's own camera rig/mount, visible in its own field of view, not real
environment. `cam_min_depth` was `0.1` — below that artifact's range — so it passed
the near-depth filter and got fused into the map as a permanent phantom wall directly
in front of the drone on every single frame, regardless of where the drone actually
was or which way it was facing.

**Fix / workaround:** Raised `cam_min_depth` from `0.1` to `0.45` (comfortably above
the observed `~0.35m` ceiling of the artifact, similar order of margin to
`astar_planner`'s own `inflate_radius_m=0.4m`). Confirmed occupied-cell counts dropped
substantially in a comparable flight window after the fix (889 → 315).

**Don't:** Don't assume a "boxed in" planner result or a noisy map is necessarily a
planner/mapping-logic bug — check the raw depth data FIRST (a quick numpy stat check
across a handful of `.npy` frames caught this in minutes) before chasing coordinate
transforms or freeze-timing theories. The freeze-timing (turning) theory investigated
first was real but secondary — this was the dominant cause.

---

## 2026-07-28 — vbox/box bounds set exactly equal to map bounds crashed exploration_node

**Symptom:** Raised `box_max_z`/`vbox_max_z` from `1.8` to `4.0` (to stop RViz truncating
the room below its real ceiling) and set `map_max_z` to the same `4.0` — `exploration_node`
crashed within seconds of the next `falcon` restart: `voxel_mapping::ESDF::getDistance`
→ glog FATAL "Address out of range", from `BsplineOptimizer::calcDistanceCost` during
the exploration planner's own internal trajectory optimization.

**Root cause:** ESDF distance/gradient queries read neighboring cells near a queried
point. With `vbox_max_z` exactly equal to `map_max_z`, there's no room left inside the
map's own allocated grid for those neighbor reads near the top boundary — an
out-of-range access is inevitable, not just possible. The documented ordering rule
(`map ⊇ vbox ⊇ box`) technically allows equality, but "supports the ordering" and
"leaves a working margin" are not the same thing, and the code that enforces this
never checks for the margin.

**Fix / workaround:** Raised `map_max_z` to `5.0` instead of leaving it at `4.0`,
restoring a real 1.0m margin between `vbox_max_z` (4.0) and `map_max_z` (5.0) — smaller
than the original config's 2.2m gap, but comfortably more than the handful of 0.1m grid
cells any real stencil actually reads.

**Don't:** Don't set `vbox_max_z`/`box_max_z` equal to `map_max_z` (or `_min` bounds
equal to each other) even though the ordering comment technically permits it — always
leave real headroom. If this crash class recurs (`ESDF::getDistance`/`getDistanceAndGradient`
→ glog FATAL "Address out of range"), check the map/vbox/box margins first, not just
whether the ordering constraint holds.

---

## 2026-07-28 — RViz showed a completely empty scene after the Y-axis localization fix

**Symptom:** After negating `position.y` in `rooster_ground_truth_localization.py`
(see the handedness-fix lesson above), RViz's 3D view showed absolutely nothing — no
error, no warning, just an empty gray scene. The 2D BEV (matplotlib) window meanwhile
showed real, live, correctly-updating data (drone pose, A* planning), proving the
underlying pipeline was fine.

**Root cause:** `maps/sphera_jail.rviz`'s saved "Current View" camera had a hardcoded
`Focal Point: X: -54.75, Y: 14.66` — captured in an earlier session, before the Y-axis
sign fix. The drone's real, physical spawn point never moved, but the sign reported
for it flipped (`+14.66` → `-14.66`), so the saved camera was now pointed at the
mirror-image empty location where the room used to be under the old convention.

**Fix / workaround:** Updated the saved Focal Point to `Y: -14.66` to match. RViz must
be restarted (not just have the `.rviz` file edited) to pick up a changed saved view,
since it only reads that file at startup.

**Don't:** When fixing a world-frame axis sign, don't assume you've found every place
it's baked in once the code and config bounds are fixed — RViz's own saved viewport
state is invisible until you actually open it, and "no error" does not mean "nothing
is wrong." Check any saved camera/view files for the same axis too.

---

## 2026-07-28 — mapping_sync's rotation freeze got permanently stuck once it started working

**Symptom:** After adding `rooster_demo_mode_manager.py` (a Rooster-only demo-mode
arbiter that didn't exist before), the map stopped updating entirely — even with the
drone sitting disarmed on the floor. `mapping_sync`'s heartbeat showed `gate=FROZEN`
with the `frozen` counter climbing every single cycle, never recovering. Restarting
`mapping_sync` itself (re-arming its warm-up sequence, which force-fuses the first 2
pairs and calls `reset_freeze()`) only unstuck it for a few frames before it froze
again immediately.

**Root cause:** `waypoint_follower_node.py`'s own rotation supervisor was found to be
continuously, on every single tick, requesting `"turning"` mode — while in its normal
`RUNNING` navigation state, targeting the default startup goal, with the drone either
disarmed on the floor or being flown manually (bypassing FALCON's own `/cmd_vel`
entirely). Its "turn complete" condition apparently never resolves without real flight
dynamics to confirm against. Before `rooster_demo_mode_manager.py` existed, nothing
ever echoed this request onto `/R1/demo_mode`, so the freeze silently never engaged —
masking this bug entirely. Fixing the arbiter (correct behavior) exposed a real,
pre-existing bug in the rotation supervisor (incorrect behavior) that had never
mattered before.

**Fix / workaround:** No real fix yet. Set `freeze_on_turning_mode:=false` on
`mapping_sync` in `sphera_drone.launch`, reverting to the freeze-less behavior
everything had already been verified against. This gives up the turning-smear
protection the freeze exists for (confirmed real and needed — see the original
ring-artifact map from 2026-07-27) until the supervisor bug itself is found and fixed.

**Don't:** Don't assume "the demo mode value looks right when I check it manually" (it
did — `rostopic pub` confirmed `/R1/demo_mode` genuinely read `"fly_straight"`) means
the freeze will clear — the freeze-clearing logic in `depth_fusion_gate.py` re-evaluates
on every NEW mode message, so if something else keeps re-requesting `"turning"` a moment
later, a manual override is immediately overwritten. Check what's actually driving the
requests (`rosnode info /waypoint_follower`, or grep its own status log for `mode=`)
before assuming a one-off manual fix will stick.

**Update 2026-07-28 (later same day):** the actual supervisor bug above was found and
fixed (`MultiAxisCommand.yaw_engaged`, see `multi_axis_follower/types.py` +
`waypoint_follower_node.py:_supervisor_cmd_wz()`), and `freeze_on_turning_mode` was
re-enabled (`true`) in `sphera_drone.launch`. It recurred anyway during a pure manual
flight test (drone driven directly via `/R1/cmd_nav`, `waypoint_follower_node.py` never
consulted at all) — but this time the mechanism was different and worth telling apart:

- `mapping_sync`'s heartbeat showed `gate=FROZEN` with `frozen` climbing, exactly like
  before. But `sensor_gate`'s own heartbeat log showed `state=FUSING`,
  `mode_turning=False` the entire time — a red herring. `sensor_gate_node.py` and
  `mapping_sync_node_sphera.py` each hold their **own separate** `DepthFusionGate`
  instance (per `sensor_gate_node.py`'s own docstring: sensor_gate's copy "does not
  freeze" the authoritative one in mapping_sync). Checking sensor_gate's health proves
  nothing about mapping_sync's gate — always check `mapping_sync`'s own heartbeat.
- Publishing to `/sensor_gate/reset_mode_freeze` (the documented manual-recovery topic)
  did nothing, because it only resets sensor_gate's copy, not mapping_sync's.
- The real cause this time: `rooster_demo_mode_manager.py`'s mode publisher is
  `latch=True` *and* re-published on a 1 s timer from its own `self.current_mode`
  — so a single stale `"turning"` request from earlier (before the manual-flight test
  even started) stayed latched indefinitely with nothing to overwrite it, once the node
  that sent it was gone.

**Actual fix:** publish a fresh non-turning string straight to the *source*,
`/R1/demo_mode_request` (not `/R1/demo_mode` itself, and not
`/sensor_gate/reset_mode_freeze`) — e.g. `rostopic pub -1 /R1/demo_mode_request
std_msgs/String 'data: fly_straight'` (`fly_straight` is the manager's own
`~initial_mode` default). This updates the manager's internal `current_mode`, so the
1 s timer starts re-publishing the correct value instead of re-latching `"turning"`.
Confirmed: `mapping_sync`'s `gate` flipped to `FUSING` and `emit` resumed climbing
within one heartbeat cycle, `frozen` stopped growing.

## 2026-07-29 — Jetson's `agency_ws` isn't a git-tracked clone of this repo

**Symptom:** Wired a new `mission_control.py` service to run
`dir_watch_path_publisher.py` from `{JETSON_REPO}/sparx_agency/robots/common/` on the
Jetson — the file is committed in this repo (`647701b9`) and clearly present locally,
but `ls` on the Jetson showed the directory without it at all.

**Root cause:** `{JETSON_REPO}` (`/home/user/agency_ws`) is not kept in sync via
`git pull`. `git status`/`git log` there show an empty, commit-less `master` branch
with unrelated home-directory dotfiles as untracked — its `sparx_agency/` tree was
manually copied over at some point in the past and never touched since for files
outside whatever was actively being used. Any file committed after that copy simply
isn't there, regardless of how long ago it was committed here.

**Fix / workaround:** `scp`'d the one needed file directly to the matching path on the
Jetson. No general sync mechanism exists — don't assume one does.

**Don't:** Don't assume `{JETSON_REPO}` matches this repo just because paths/service
definitions reference it as if it does. Before wiring any new `mission_control.py`
Jetson-side service, verify the specific file(s) it needs actually exist there first
(`ssh ... ls <path>`), rather than debugging a confusing `FileNotFoundError` after the
fact.

---

## 2026-07-29 — testing `mission_control.py`'s orchestration logic without driving the UI

**Symptom:** Needed to verify `start_service`/`stop_service`/`get_all_states` actually
work for two new services, but no browser-automation tool was available to click
through the Streamlit UI, and the script has no `if __name__ == "__main__":` guard —
it's UI calls top-to-bottom, seemingly not meant to be imported as a plain module.

**Root cause / finding:** Streamlit tolerates being imported outside a real
`streamlit run` session. Every `st.*` call just logs "missing ScriptRunContext! This
warning can be ignored when running in bare mode." and no-ops, rather than raising —
this is documented Streamlit behavior ("bare mode"), not a hack specific to this file.

**Fix / workaround:** `import sparx_agency.tools.mission_control as mc` directly (via
`importlib`, with the repo root on `sys.path`) and call `mc.start_service(svc)` /
`mc.stop_service(svc)` / `mc.get_all_states()` against real `Service` objects pulled
from `mc.ALL_SERVICES`. Output is drowned in bare-mode warnings — grep for your own
`print()` markers rather than trying to read the raw output.

**Don't:** Don't assume a Streamlit app needs a real browser session (or refactoring
into a separate importable module) just to unit-test its orchestration logic — try a
plain import first.

## 2026-07-29 — broken torch import: orphaned cu12 packages + two corrupted cu13 installs

**Symptom:** `import torch` failed with `undefined symbol: ncclCommWindowDeregister` in
`libtorch_cuda.so`. After removing the conflicting packages (see Root cause), the same import
then failed with `libcudnn.so.9: cannot open shared object file`, then after fixing that,
`libnccl.so.2: cannot open shared object file` — three distinct failures in sequence, each
looking like it could be "the" bug.

**Root cause:** Two independent problems layered on top of each other in the project venv:
1. Both cu12 and cu13 variants of nearly every NVIDIA package (`nvidia-cublas`,
   `nvidia-cudnn`, `nvidia-nccl`, etc.) were installed side by side. `torch==2.11.0` declares
   `nvidia-nccl-cu13` etc. as its actual dependency, but the orphaned cu12 cluster (verified via
   `pip show <pkg>` — `Required-by:` was empty for every one of them) was shadowing/conflicting
   at import time, producing symbol mismatches against the older cu12 library.
2. Independently, `nvidia-cudnn-cu13` and `nvidia-nvshmem-cu13` had their `.dist-info` metadata
   present (`pip list`/`pip show` reported them installed) but their actual `.so` library files
   were missing from disk entirely — an interrupted or corrupted earlier install, unrelated to
   the cu12/cu13 conflict. Fixing #1 alone still left torch broken because of #2.

**Fix / workaround:** Diagnosed systematically rather than guessing package-by-package:
`pip show -f <pkg>` lists every file a package installed; checked each installed nvidia-* cu13
package's files against disk (`[[ -f "$location/$f" ]]`) to find exactly which ones were
actually missing (`nccl`: 1/1 missing, `nvshmem`: 12/12 missing — everything else was fine).
Uninstalled the entire orphaned cu12 cluster (`pip uninstall -y nvidia-*-cu12` for every one
with an empty `Required-by`), then `pip install --force-reinstall --no-deps` the two genuinely
broken cu13 packages specifically (not a blanket torch reinstall).

**Don't:** Don't assume the first import error is the only problem — each fix can unmask a
different, independent failure underneath. Don't `pip uninstall`/reinstall packages just
because they look suspicious; check `Required-by` first so a shared venv (many things in this
project depend on it) doesn't lose something another part of the codebase actually needs.
Don't reinstall a big package (torch) wholesale to fix what's actually a missing-file problem
in one of its dependencies — `pip show -f` + a file-existence check finds the precise culprit
in seconds and avoids re-downloading everything.

<!--
Example, in the style already proven useful in project-specific skills:

## 2026-07-22 — hover_z drift at 560

**Symptom:** Drone gets airborne and holds briefly at hover_z=560, then slowly drifts to
the ceiling over 1-2 minutes. Lower values (550) sink to the floor instead; higher values
(575+) drift up faster.

**Root cause:** Unresolved as of this writing — shortening climb_duration_sec (tested at
1.5s) did NOT fix the drift, which rules out simple integral windup as the sole cause.

**Fix / workaround:** No real fix yet. Use hover_z=560 and budget for manual landing before
~1 minute on any longer test.

**Don't:** Don't assume climb_duration_sec is the whole story — already tested and ruled out.
-->
