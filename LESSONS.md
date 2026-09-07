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

**Update 2026-07-30 — same root cause, a much more misleading second symptom:** left the
twist-control adapter running from an earlier FALCON click-to-fly test, then moved on to a
manual-flight session via the Tkinter `ui.py`. The drone took off and immediately flew hard
into a wall; it LOOKED exactly like a yaw/turn-direction bug (see the "BEV click turned the
wrong way" investigation earlier the same day) — plausible enough that real time went into
checking bearing math and body-frame conventions before the actual cause surfaced. A
command+pose logger (subscribing to `/R1/cmd_nav` and `/R1/localization`, no
publishing) showed the real story: continuous `{"action": "move", "axes": {"x": 599, "y":
0, "r": 0}}` commands at ~50ms intervals, present in **6672 of 6765** logged commands
(only 93 had `x==0`) — i.e. an almost-constant ~60%-forward push with zero steering,
starting before `arm` and continuing straight through takeoff. `ui.py` itself only ever
sends discrete named actions (`forward`/`turn_left`/etc., grepped — no `"move"` string
anywhere in it), so this wasn't the UI's own doing at all: `{"action": "move", "axes":
{...}}` is `rooster_twist_control_adapter.py`'s own publish format, and it was still alive
from earlier, faithfully translating FALCON's `waypoint_follower` (still cruising toward
its own leftover default goal) into a continuous forward push that fought every manual
input the whole session.

**Don't (extended):** A drone that "drifts/turns into a wall immediately on takeoff" is
NOT proof of a yaw-sign or turn-direction bug — check `ps aux | grep
rooster_twist_control_adapter` (or any other `/R1/cmd_nav` publisher) FIRST, before
re-deriving bearing/handedness math again. A cheap command+pose logger (subscribe-only,
never publish) that records every `cmd_nav` message alongside ground-truth pose settles
this class of question in seconds instead of live-flight-testing guesswork — keep using
one whenever "what actually got commanded" is in doubt.

**Update 2026-08-02 — third occurrence, this time looked like a "turning left instead of
right" planning/direction bug across THREE separate takeoffs.** After building a live
telemetry dashboard for click-to-fly debugging, the twist-control adapter was left running
(needed for that testing) and never stopped before the user switched to flying manually via
`ui.py`. All three flights showed the drone hovering nearly in place with yaw oscillating
50-100°+ repeatedly right after takeoff — this time the interference wasn't a constant
forward push (2026-07-30's symptom) but an alignment fight, because FALCON's
`waypoint_follower` always has a non-empty `self.goal` from the moment it launches
(`~goal_x`/`~goal_y` args are read at init, before any click ever arrives — see
`astar_planner_node.py`), so the adapter kept trying to turn the drone toward that static
default goal while the user tried to fly a different direction by hand. The command log
confirmed it instantly: 300 of 301 logged commands in each flight were `{"action": "move",
...}` (the adapter's format), only 1 was the actual `takeoff` action — zero named actions
(`forward`/`turn_left`/etc.) from the UI made it through uncontested.

**Don't (extended further):** Having a live dashboard that shows this process's up/down
status does NOT prevent this bug by itself — the status was accurate and visible the whole
time, but nothing prompted anyone to actually stop the process when the testing mode
switched from click-to-fly to manual flight. Whenever handing control back to a human pilot
after any autonomous-navigation testing, explicitly kill every `/R1/cmd_nav` publisher that
isn't the UI first — don't rely on remembering, and don't treat "the dashboard would show it
as running" as equivalent to "someone will notice and stop it."

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

## 2026-07-30 — Rooster twist-control adapter's max_yaw_rate was ~4x too low

**Symptom:** No live crash or error — found via a logged manual flight, not a reported bug.
`rooster_twist_control_adapter.py` scales an incoming Twist to a `/R1/cmd_nav` axis value as
`axis = twist_component / max_component * 1000`, with `max_yaw_rate` defaulting to 0.5 rad/s
(never live-validated — same "assumed from doc convention" class of issue already seen once
with `turn_left`/`turn_right`'s r-axis sign, see the earlier stop-watchdog entry above).

**Root cause:** A subscribe-only command+pose logger (`manual_flight_logger.py`, joins
`/R1/cmd_nav` and `/R1/localization` by timestamp) captured a full manual flight and let the
actual axis->rate relationship be measured directly instead of assumed. 8 isolated
`turn_right` segments (axis r=500, each bounded by the next command so no averaging across
unrelated motion) gave a consistent ~55 deg/s (~0.96 rad/s), extrapolating to axis 1000 (full
deflection) -> ~1.9 rad/s. The configured 0.5 rad/s max meant any planner (FALCON's
`waypoint_follower_node.py` via `rooster_twist_control_adapter.py`) requesting even a modest
angular.z was actually commanding a turn ~4x faster than it thought it was asking for.
3 `turn_left` segments gave a lower ~42 deg/s (~1.5 rad/s at full scale) — a real left/right
asymmetry, or noise from the small sample, not yet resolved (see
`docs/progress/entries/007-rooster-velocity-controller.md`).

**Fix / workaround:** Recalibrated `max_yaw_rate`'s default from 0.5 to 1.8 rad/s (both the
constructor default and the CLI `--max-yaw-rate` default) in `rooster_twist_control_adapter.py`,
with the derivation documented inline. `max_linear_x`/`max_linear_y` were left unchanged —
this same flight's forward/lateral segments were too short and interleaved with adjacent turns
(leftover momentum contaminated each segment; several "forward" segments even showed net
*negative* displacement) to derive a trustworthy number. A dedicated calibration flight
(isolated single-axis moves, no interleaving) is the planned follow-up for those and to
confirm whether the yaw asymmetry is real.

**Don't:** Don't assume a `max_*` "rate at full deflection" constant is correct just because
no one has complained — it can be silently wrong for a long time if nothing downstream
saturates it in an obviously visible way. Don't try to calibrate axis->rate gains from a
flight that wasn't designed for calibration (this one mixed forward/turn commands back to
back) unless the segment is long/isolated enough that adjacent-command momentum can't
contaminate it — turn-rate segments here were trustworthy because each was bounded cleanly by
stops; translational segments were not, for the same reason.

## 2026-07-30 — flew after an R1 crash without restarting falcon, map was poisoned

**Symptom:** After `rooster_manager` crashed inside `R1` and Sphera was restarted, only
`ros1_bridge` and `video_trigger.py` were restarted (the documented fix for those two going
stale on R1 recreation) — `falcon` itself, running continuously since well before the crash,
was left alone on the reasoning "the drone respawned at the same spawn point, so the map
should still be valid." The next click-to-fly test got repeated `astar_planner` "boxed in" /
"PLAN FAILED ... pinched at (x,y) — thinner than the airframe, so most likely a mis-detected
voxel" warnings, the drone jumped several meters at a time while nominally stopped, and landed
~11m from the intended goal.

**Root cause:** `exploration_node`'s voxel map is long-lived state for the whole `falcon`
container's lifetime, and FALCON's TSDF/ESDF mapping has no exposed decay/forgetting or
hit/miss-probability knob at all (confirmed by reading the vendored source directly) — bad
fused data isn't temporally erased, it just sits there until enough new correct observations
outweigh it, if that happens at all before the bad region matters for planning. During the
`rooster_manager` crash window, `video_trigger.py` kept running and (per the existing
"does NOT reconnect after R1 respawn" gotcha above) very likely froze on its last decoded
frame and kept feeding stale/close-up depth into the still-running map as if it were real
geometry. "Same spawn point" says nothing about whether the map's accumulated voxels are
still trustworthy — that reasoning was the actual mistake.

**Fix / workaround:** Elevated to a first-rule item in the `fly-rooster-sphera` skill: any
time `R1` crashes or gets recreated, fully restart `falcon` (not just `ros1_bridge`/
`video_trigger.py`) before flying again. A full container restart costs under a minute and
guarantees a clean map, which is far cheaper than diagnosing "is the occupancy map buggy" as
if it were a deep, novel bug every time this happens.

**Don't:** Don't reason about map validity from drone *position* alone — the map is a
function of everything that got fused into it since the container started, not of where the
drone currently is. Don't skip restarting `falcon` after an `R1`-side crash just because the
immediately-obvious staleness fixes (bridge, video) are already in the routine — the map
itself is the thing most silently damaged by exactly that kind of disruption.

## 2026-07-30 — a directional cmd_nav action doesn't clear the other axes

**Symptom:** Drone got pinned against a wall during an autonomous click-to-fly test. Killed
`rooster_twist_control_adapter.py` (the competing publisher) and immediately sent a single
`{"action": "backward", "value": 300}` to back it away. Instead of retreating cleanly, the
drone's position and the altitude-hold ranger both went wild (ranger jumped 1.7m -> 3.1m ->
4.06m within 2s, velocity swinging -1.84 -> +0.94 -> -0.17 m/s) and it ended up far from where
a straight retreat should have put it.

**Root cause:** `rooster_command_unit.py`'s `_MOVE_ACTIONS` handling (`_on_cmd_nav`) only
overrides the ONE axis a named action maps to (`"backward": dict(x=-1)` only ever touches
`x`) — `y` and `r` are explicitly preserved from `unit.axes`'s current value, by design, so
that turning while flying doesn't zero the drone's throttle/altitude-hold `z` axis. But that
same preservation applies to `y`/`r` too: whatever `rooster_twist_control_adapter.py` had
last written (it had been actively commanding a nonzero yaw rate while fighting the stuck-
against-wall situation, right up until the moment its process was killed) stayed latched in
`RoosterUnit.axes` and got carried straight into the `backward` command. The drone was very
likely retreating correctly relative to its own nose (confirmed via `rooster_unit.py:164-168`
publish_manual - `x/y/r` pass straight into the vendor `ManualControl` message with no frame
transform in our code, and FCU `ManualControl` axes are body-relative on essentially every
FCU convention) while ALSO still spinning from the stale `r` - a body-relative retreat plus
continuous yaw looks, from the world frame, exactly like "backward went somewhere wrong."

**Fix / workaround:** Send `stop` first (a distinct action, not part of `_MOVE_ACTIONS` -
zeroes x/y/r fully) to clear ALL latched axes, THEN send the desired directional action.
Never assume killing a competing publisher process also clears the state it already wrote
into `RoosterUnit` - the axes it set are still sitting there until something explicitly
overrides or zeroes them.

**Don't:** Don't reach for a single named directional cmd_nav action as an emergency override
without a preceding `stop` when a different controller (autonomous or otherwise) was just
active - it inherits whatever that controller left on every axis except the one you're
setting. Don't assume "the drone moved in a weird direction" means the axis convention itself
is wrong (body- vs world-frame) before checking what was actually latched on the OTHER axes
first - the frame convention here was correct; the bug was leftover state.

---

## 2026-08-02 — ros1_bridge crashed with std::bad_alloc after every Sphera restart, no voxels in RViz

**Symptom:** After restarting Sphera/R1, `ros1_bridge` (launched via `run_bridge.sh` with only
`ROS_DOMAIN_ID=9` set) crashed within ~1s of the first real messages flowing (`terminate called
after throwing an instance of 'std::bad_alloc'`), right after logging "Passing message from ROS 2
geometry_msgs/msg/PoseStamped..." or a plain `std_msgs/String`. `falcon`'s `mapping_sync` heartbeat
stayed at `gate=WARMUP`, `last_pose` growing forever, so no voxels ever appeared in RViz. Recreating
`R1`, restarting `it`, and retrying the bridge repeatedly all failed to fix it.

**Root cause:** `run_bridge.sh`'s own header comment says Rooster/R1 requires
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` plus `CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml`
(R1 is Jazzy/CycloneDDS; XTEND, the script's default, is Foxy/FastRTPS) - only `ROS_DOMAIN_ID=9` was
being passed. The bridge ran on `rmw_fastrtps_cpp` trying to deserialize CDR from a CycloneDDS
publisher; the cross-vendor CDR mismatch is what threw `bad_alloc`, not the previously-documented
Sphera CycloneDDS interface corruption bug ([[project_sphera_cyclonedds_interface]] in memory) - that
one only applies when both sides already agree on CycloneDDS.

**Fix / workaround:** Always launch the bridge for Rooster/R1 with all three overrides together:
`ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///home/user1/rqs_iai_ws/src/cyclonedds.xml ./run_bridge.sh`.
`ROS_DOMAIN_ID` alone is not enough. Confirm by checking the bridge's own startup banner prints
`RMW (ROS2): rmw_cyclonedds_cpp`, and confirm real flow with `rostopic hz /R1/localization` inside
`falcon` (not by grepping the bridge log for "Passing message" - that line only ever prints once per
topic type, so it can't tell you whether the stream later stalled).

**Don't:** Don't assume a repeat `bad_alloc` after a Sphera restart is the known CycloneDDS-corruption
bug and reach straight for "restart Sphera again" - check the bridge's actual RMW banner first. Don't
trust `mapping_sync`'s heartbeat alone to mean "no bridge running" - `pose=2, buf=1` stuck and aging is
what it looks like when the bridge greeted the topic once and then silently dropped it, distinct from
`pose=0, last_pose=-1.0s` (never connected at all).

---

## 2026-08-03 — Rooster's published yaw had TWO independent sign/reference bugs, not one

**Symptom:** BEV map/mapped geometry appeared on the wrong side of the drone from the way it was
actually facing in Sphera (e.g. drone visually facing +Y, mapped area showing up toward -Y). Separately,
`turn_right`/`turn_left` from the manual UI always worked correctly, but a controller computing a
specific relative turn (e.g. "turn +30deg") sometimes went the wrong way.

**Root cause:** Two distinct, independent bugs in `rooster_ground_truth_localization.py`'s `_on_state`,
both hiding behind similar-looking symptoms:
1. **Rotational sense.** The code negated yaw (`-msg.rotation.yaw`) based on a comment claiming
   Sphera/Unreal yaw is left-handed/clockwise-positive. Confirmed live via a real commanded `turn_right`
   while airborne (compare raw `msg.rotation.yaw` against published `/R1/localization` yaw at the same
   timestamps): raw yaw DECREASED during the real right/clockwise turn, i.e. it was already standard
   REP103 CCW-positive. The negation was taking an already-correct value and flipping it.
2. **Zero-reference axis offset (a completely separate degree of freedom from #1).** Confirmed on a
   fresh, untouched spawn (no commands sent since the last Sphera restart): the drone visually faces
   world +Y there, but its yaw read ~-94deg, not the +90deg that FALCON's own camera-to-world depth
   integration assumes (yaw=0 means forward aligns with world's own +X, CCW-positive) for a drone facing
   +Y. Off by ~180deg - independently explains the BEV map symptom, since FALCON would then integrate
   depth data toward almost the exact opposite of where the drone was really looking.
   `sphera_jail.yaml`'s `init_yaw: 0.0` for this same spawn point is consequently also wrong, but that's
   a repo-wide placeholder default used identically on every map file, not something calibrated
   per-map, so it was left alone (real localization data supersedes it within milliseconds anyway).

**Fix / workaround:** `yaw = atan2(sin(raw_yaw + pi), cos(raw_yaw + pi))` - no negation (fixes #1), plus
a `+pi` (180deg) offset before wrapping (fixes #2). Verified: the corrected value on that same fresh
spawn came out to 85.6deg, matching the expected +90deg closely (small residual because the real spawn
heading isn't perfectly axis-aligned - not a bug).

**Don't:** Don't assume "yaw problem" is one bug just because both symptoms involve yaw and both got
fixed by edits to the same function - these were two independently-derived, independently-confirmed
fixes with completely different evidence (a live turn test vs. a static fresh-spawn quaternion reading).
Fixing #1 alone did not fix #2, and there was no a-priori reason #2 had to exist at all once #1 was
fixed. Also don't put a fix like this inside `core/planning/trackers/waypoint_follower` (tried and
reverted mid-session) - that package is deliberately drone-agnostic and assumes standard REP103 yaw;
the correct place is the Rooster-specific localization boundary that produces the yaw in the first
place, so every consumer (core's own bearing math, FALCON's C++ mapping, the BEV click arrow) gets
fixed at once without needing its own compensating hack.

---

## 2026-08-03 — computed turns went backwards while manual turn_left/turn_right always worked

**Symptom:** The manual UI's `turn_right`/`turn_left` buttons always turned the drone the correct way.
But any controller-computed turn (waypoint, drift_pid, ...) that needed a specific relative correction
(e.g. "turn +30deg") sometimes executed in the opposite direction, and turns driven by `/cmd_vel` never
settled - repeatedly overshooting past the target and correcting back the other way, forming a sustained
pendulum that never completed.

**Root cause:** Two separate, additive problems, both inside `rooster_twist_control_adapter.py` (the
one, shared conversion point every controller's `/cmd_vel` funnels through before reaching the FCU -
manual UI actions never go through this file at all, straight to `rooster_command_unit.py`'s named-
action dict instead, which is exactly why manual always worked):
1. **Sign mismatch.** Standard REP103 (which `waypoint_follower`/`drift_pid` correctly assume, since
   `core/` is deliberately drone-agnostic) has positive `angular.z` = CCW = left. This drone's FCU axis
   convention has positive `r` = right (confirmed live, same convention already baked into
   `rooster_command_unit.py`'s `turn_left: r=-1` / `turn_right: r=+1`). The adapter was doing a direct,
   unflipped `r = angular.z / max_yaw_rate * 1000` - so a controller correctly computing "turn left"
   executed as "turn right", the opposite of intended, for every controller.
2. **No damping on top of an undamped FCU loop.** PX4's own yaw-rate control loop has zero derivative
   gain (`MC_YAWRATE_D=0.0`, confirmed via `/R1/fcu/param/get_float` - P/I only). The adapter was
   forwarding whatever r-axis value a controller requested as an instantaneous step change every tick,
   exactly the kind of abrupt input that excites a P/I-only loop into sustained oscillation. Per the
   user's explicit preference, this was NOT fixed by touching the PX4 parameter itself.

**Fix / workaround:** (1) Negate: `target_r = -angular.z / max_yaw_rate * 1000`. (2) Added a
`slew()` rate-limiter (new helper in `robots/common/math_utils.py`) so the adapter ramps its own r-axis
output toward the target by at most `max_yaw_axis_step_per_sec` (2500, a deliberately conservative
first guess - live-test and retune) per second, instead of snapping to it - smoothing the input without
touching PX4. `stop_motion()` resets the ramp state directly so a stop is still immediate.

**Don't:** Don't assume a shared symptom across every controller means the bug is in one specific
controller's tuning - check what code path is actually COMMON to all of them first (here, the single
Twist->cmd_nav converter) before hunting through per-controller PID gains. Don't conflate "the turn
never settles" (damping/oscillation) with "the turn goes the wrong way" (sign) - they were reported
and diagnosed as one thing at first, but are two independent bugs with two independent fixes; a
sign-flipped, undamped loop produces a much more violent, sustained oscillation than an undamped-but-
correctly-signed one, so a report of severe swinging is a hint to check the sign FIRST, not just reach
for damping.

---

## 2026-08-10 — `robotican_dev` container: wrong name + missing engine mount broke three Rooster services

**Symptom:** After the `robotican_dev` container exited (stale, 3 days old) and was brought back up with
`docker compose -f docker-compose.robotican.yml up -d`, mission_control's Rooster Frame Capture, Depth
Processor, and Twist Control Adapter all failed identically: `Error response from daemon: container
<hex-id> is not running` — quoting the ID of the OLD, exited container, even though a freshly-started
one was clearly running. Once that was fixed, Depth Processor then failed with `FileNotFoundError` for
the DA3 TensorRT engine, despite the file existing fine on the host.

**Root cause:** Two independent bugs in `docker-compose.robotican.yml`, both invisible until something
tried to recreate the container from scratch: (1) no `container_name:` was set, so `docker compose up`
named it `theagency-robotican-1` (the default `<project>-<service>-<index>` pattern) instead of the
literal `robotican_dev` every `run_*.sh` wrapper and `mission_control.py`'s `proc_container` hardcode —
the previous, long-lived container only had the right name because it predated this compose file /
was started some other way, so the mismatch never surfaced until it needed recreating. (2)
`~/depth_anything_ws` (where the DA3 engine + ONNX live) was never in the compose file's volume list at
all, so the path doesn't exist inside the container regardless of naming.

**Fix / workaround:** Added `container_name: robotican_dev` to the compose file's service block, and
added a `${HOME}/depth_anything_ws:${HOME}/depth_anything_ws` volume (read-write, not `:ro` — a TRT
version mismatch has needed rebuilding the engine from inside this exact container before, see the
2026-07-29 entry above).

**Don't:** Don't assume "container is not running" means the container just needs restarting — check
`docker ps -a` for the EXACT name being `docker exec`'d into first. A compose-managed container can
come up successfully under a completely different name than every script expects, and the error message
alone won't tell you that.

---

## 2026-08-10 — FALCON `exploration_node` stuck in a "no path to next viewpoint" loop with the drone still on the ground

**Symptom:** After wiring `nav_mode:=exploration` (`exploration_node` → `traj_server` →
`falcon_exploration_follower_node.py`), the drone never moved. `exploration_node` was clearly alive and
computing (HGrid/TSP/SOP all logging every ~30-50ms, correctly picking a next-viewpoint target), but
every attempt ended in `[ExplorationManager] planTrajToView: No path to next viewpoint using default
A*` / `... using coarse A*` / `[FSM] Plan fail`, repeating in a tight loop — with the map otherwise
healthy (`mapping_sync` reporting `gate=FUSING`, pose and depth both flowing).

**Root cause:** The drone was armed but had never taken off — sitting at ground level. Exploration's
frontier viewpoints are chosen in free 3D space at flight altitude; A* has no reachable route from a
ground-level start pose to one of those. `falcon_exploration_follower_node.py` correctly recognized it
had no valid trajectory and held station the whole time (by design, via `ReferenceTracker3D`'s own
stale-reference handling) rather than fabricating motion — so nothing in the follower's own logs pointed
at "not flying yet" as the cause.

**Fix / workaround:** Take off and get the drone hovering first, *then* switch to `nav_mode:=exploration`.
Confirmed live: within seconds of takeoff, `exploration_node` started finding paths, `traj_server`
reported real flight-time/path-length progress, and coverage climbed past 97%.

**Don't:** Don't debug this as a planner/map bug (checking safety clearance, map connectivity, frontier
placement, etc.) before confirming the aircraft is actually airborne — the log signature is identical
either way, and chasing the map/planner side first wastes time on a non-issue.

---

## 2026-08-10 — Rooster Frame Capture: `PermissionError` writing into `/tmp/rooster_frames`

**Symptom:** `rooster_frame_dir_publisher.py` (Rooster Frame Capture, runs inside
`robotican_dev`) crashed on its very first frame with `PermissionError: [Errno 13]
Permission denied: '/tmp/rooster_frames/frame_00000001.tmp'`.

**Root cause:** `/tmp/rooster_frames` and `/tmp/rooster_depth` are shared host directories
bind-mounted (at the same path) into `robotican_dev`, `falcon`, and `detector_dev`. Docker
auto-creates a bind-mount's host-side directory as `root:root` mode `0755` if it doesn't
already exist when a container starts — and since `/tmp` is typically cleared on reboot,
whichever of those containers happened to start first ended up creating the directory as
root. `robotican_dev` (the one that actually needs to WRITE the frame/depth files) runs as
non-root `uid 1000`, so it couldn't write into a root-owned 0755 directory. `falcon` and
`detector_dev` both mount these paths read-only, so neither could fix it from inside
either — the fix had to come from the host.

**Fix / workaround:**
```bash
sudo chmod 777 /tmp/rooster_frames /tmp/rooster_depth
```
Do this any time `/tmp` has been freshly cleared (reboot) before the first bring-up of the
day.

**Don't:** Don't assume a `PermissionError` here means something wrong with
`rooster_frame_dir_publisher.py` itself — check `ls -ld` on the two shared dirs first;
whichever container starts first on a clean `/tmp` decides the ownership for everyone else.

---

## 2026-08-13 — `rooster_ground_truth_localization` silently received nothing after Sphera restart

**Symptom:** After restarting Sphera, `/R1/localization` published nothing (confirmed via a
real subscriber script, not `ros2 topic hz` — see the entry below on why that tool lied).
Restarting `rooster_ground_truth_localization` didn't fix it either.

**Root cause:** Sphera's own engine publishes `/R1/sphera/state` at `BEST_EFFORT` reliability
from a bare DDS participant (not rclcpp). `rooster_ground_truth_localization.py`'s subscriber
used a plain integer queue depth (`create_subscription(..., 10)`), which rclpy expands to the
default `RELIABLE` profile — incompatible with a `BEST_EFFORT` publisher, so it silently
received zero messages. `ros2 topic info --verbose` showed the exact warning: `New publisher
discovered on this topic, offering incompatible QoS... RELIABILITY_QOS_POLICY`.

**Fix / workaround:** Subscribe with an explicit `QoSProfile(depth=10,
reliability=ReliabilityPolicy.BEST_EFFORT)` instead of a bare int. Confirmed live: ~140Hz
real pose data end-to-end afterward. The same two-publishers-different-QoS pattern also
shows up on `/R1/state` (ranger) — one `BEST_EFFORT` bare-DDS publisher, one `RELIABLE`
`rooster_manager` publisher — but that one has always been benign since the real consumer
subscribes `RELIABLE` (compatible with `rooster_manager`), so don't chase the warning there.

**Don't:** Don't assume "the process is alive and logs 'ready'" means it's receiving data.
Check `ros2 topic info --verbose` for a QoS mismatch warning before assuming the bug is
somewhere else (Sphera itself, the bridge, network).

---

## 2026-08-13 — `ros2 topic hz` / `rostopic hz` report nothing on `BEST_EFFORT`-only topics

**Symptom:** Repeatedly checked "is data flowing?" with `ros2 topic hz` / `rostopic hz` and
got zero output, even when a real, independently-confirmed consumer (a node's own log
showing incrementing counts) proved the topic was very much alive.

**Root cause:** Both tools subscribe with default (`RELIABLE`) QoS. Any topic whose only
publisher is `BEST_EFFORT` (Sphera's own sim-engine topics are, throughout this stack) is
QoS-incompatible with that default subscription, so the tool silently sees nothing. This
ROS2 distro (Foxy) doesn't even have a `--qos-reliability` override flag for `ros2 topic hz`.

**Fix / workaround:** Don't trust `hz`/`rostopic hz` as ground truth on any Sphera-originated
topic. Write a one-off subscriber script with explicit `BEST_EFFORT` QoS instead (a few lines
of rclpy/rospy) — that's what actually caught the real data rate every time this came up today.

**Don't:** Don't conclude a topic is dead from `hz` alone. Cross-check against a real
consumer's own log, or a hand-rolled subscriber with matching QoS, before restarting things.

---

## 2026-08-13 — Altitude-hold PD loop ran at 1Hz, ~10x slower than its own sensor

**Symptom:** `rooster_command_unit`'s altitude hold would drift 0.5-1m past target and
either never recover, or oscillate in a stuck band for minutes — confirmed live across many
flights, at multiple `altitude_hold_max_correction` values (200, then 380).

**Root cause:** Two separate bugs stacked on each other.
(1) `altitude_hold_interval_sec` defaulted to `1.0`, but `/R1/state` (the ranger source)
actually updates at ~10Hz — the loop was reacting to only 1 in 10 fresh readings on a true
double-integrator plant (throttle → accel → velocity → position), way too slow to converge.
(2) Naively raising the loop rate to match (~10Hz) then aliased against the sensor's own
~10Hz rate: on ticks where no new sample had arrived yet, `ranger == prev_ranger` computed a
false `velocity=0`, producing an undamped P-only correction that alternated with properly
-damped ticks every other cycle — visible directly in the log as a correction value bouncing
between wildly different numbers tick to tick.

**Fix / workaround:** `altitude_hold_interval_sec` raised to `0.1` (still runs the ROS timer
at 10Hz), AND `_altitude_hold_tick` now skips entirely when `ranger` hasn't actually changed
since the last real sample, using the true elapsed wall-clock time (`time.monotonic()`) for
the velocity term instead of the fixed loop interval. Confirmed live: a clean, monotonic
convergence to within 2cm of target in under 5 seconds on one flight (vs. minutes of
oscillation before) — though a later flight still didn't converge cleanly, and turned out to
be physically wedged against a door frame at the time, not a control-loop failure.

**Don't:** Don't just widen the correction clamp again if this recurs — check `/R1/state`'s
actual publish rate first (a hand-rolled subscriber, not `hz` — see the entry above) before
assuming the gains or the clamp are the problem.

---

## 2026-08-13 — Voxel map looks fragmented/floating; root cause is altitude instability, not FALCON

**Symptom:** RViz's Voxel Mapping display repeatedly showed disconnected chunks of occupied
cells floating above the main wall/floor structure, with visible gaps — reproduced fresh on
multiple flights, on a completely reset map each time, so not stale/leftover data.

**Root cause:** Confirmed via direct query (subscribing to `/voxel_mapping/occupancy_grid_occupied`
and comparing X/Y footprints across height bands): the vast majority of "floating" cells had
zero occupied structure directly below them, while every low cell also had structure above
it. This matches a coverage gap, not corrupted data: the drone scans real walls at whatever
height it happens to be flying, but the still-imperfect altitude hold means that height
varies flight to flight (and even within one flight), so the same X/Y spot's lower wall
section sometimes never gets scanned from a floor-hugging vantage. FALCON's own exploration
logic doesn't force a second, different-height pass over an already-"seen" frontier.

**Fix / workaround:** None yet — the aliasing fix above improves *convergence* but the hold
still isn't perfectly steady through a whole flight, and that residual variance is enough to
leave gaps. The real fix is making our own altitude hold hold a genuinely constant height
throughout exploration, not a FALCON-side change (FALCON's frontier logic just faithfully
maps whatever height band the camera was actually pointed at).

**Don't:** Don't reach for FALCON's own exploration/frontier code to fix this — the root
cause is in our altitude-hold stability (`rooster_unit.py`), not upstream FALCON logic.

---

## 2026-08-13 — Launching Sphera + Rooster has no CLI/service hook; it's GUI-only past the container restart

**Symptom:** Wanted to script/automate the full Sphera + Rooster bring-up. Found and confirmed
`~/.sphera/sphera-restart.sh` handles the container restart itself cleanly (`docker compose
down` + `up -d run_sphera`), but the drone never actually became controllable after just that.

**Root cause:** Everything past the container restart is a click inside Sphera's own rendered
(Unreal) window, with zero corresponding shell command, ROS topic, or file change to key off
of: a "Welcome to SPHERA" screen needing a "Continue" click, a role-select screen ("Manager" is
the only enabled option in this build), a "Manager mode" scenario picker (`Start` →
`Standalone`), then assigning the drone to an operator and pressing the green ▶ Play button.
`ros2 service list`/binary `strings` turned up no scenario-control service or CLI flag.
**Play is also what spawns the separate `R1` backend container** (`sphera-backend:rooster`,
`docker run --name R1 ...`) — it's not a distinct manual step, just an automatic side effect of
that one click. If `R1` dies mid-scenario (vs. the container never having existed), Sphera does
NOT auto-respawn it — replaying the exact captured `docker run` command by hand works fine.

**Fix / workaround:** No automation exists for the GUI portion. `xdotool` could theoretically
click through it (proportional coordinates within the window's current geometry, not absolute
screen position — Sphera's window isn't pinned to one place or size), but that was explicitly
decided against for now given the fragility of blind GUI-coordinate clicking. Manual for now.

**Don't:** Don't assume "restart the container" is equivalent to "the drone is flyable again" —
always confirm ROS2 nodes/topics under `/R1/*` and `/Rooster_1` actually reappear before moving
on to restarting the falcon-side stack.

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

---

## 2026-08-26 — InternVLA-N1 on SJTU: three knobs that were read by nobody, and a wall that rotation could not escape

**Symptom:** With everything documented and configured, the hospital flights produced
almost no continuous trajectory: 18 of 22 committed routes were 0.25 m two-point stubs,
the drone spent 43% of a flight stationary, five separate twelve-second stalls in ninety
seconds, and one run sat 0.45–0.70 m from a wall for seventy seconds re-committing the same
forward step. From the outside it looked like the model producing short plans.

**Root cause:** Four independent things, none of which raised anything.

1. **The agent hands the curve over and then queues its own discretisation.** One System-1
   pass renders twice — the continuous path *and* the list of 0.25 m / 15° steps that
   approximates it. Upstream returns `idx[0]` now and queues `idx[1:]` for the next three
   calls, which is right for a discrete client and double-counting for one flying the curve:
   it had already covered that ground, and three of every four decisions carried no curve at
   all. Server-side, needs a patch (PATCH 7, `sys1_continuous_only`).
2. **`commit.expected_speed_mps` and `commit.commit_grace_s` were in the YAML and never read
   by the node.** Every commitment therefore fell back to the flat `max_commit_s: 12` — the
   exact twelve-second stalls the YAML comment claims those two knobs removed. The config
   file described a fix that did not exist. Found by an offline dry run, not by a flight.
3. **A discrete TURN action was flown as a bent 0.25 m waypoint.** Upstream a turn is a pure
   rotation (`trajectory_to_discrete_actions_close_to_goal` advances `pos` only on a forward
   action), but everything downstream consumes polylines, so it was rendered as a step — and
   a *holonomic* pursuit satisfies that by crabbing sideways. The model asked to look
   somewhere and the aircraft shuffled 0.25 m without changing where it was pointing.
4. **`/agent/init` against a live agent is a server-side no-op**, so changed intrinsics,
   `sys2_max_forward_step` and the new flag all reached a server started before the change
   and were silently ignored — the flight behaves like the old configuration while every
   file on disk says otherwise.

**Fix / workaround:** PATCH 7 in the vendored agent (opt-in, defaults to upstream behaviour);
pass `expected_speed_mps`/`commit_grace_s` through to `CommitSpec`; fly a turn as a rotation
via `core/planning/vlas/common/turn_in_place.py` on a `/n1/yaw_goal` topic; warn loudly when
`init_agent` short-circuits with settings in hand, and restart the server after changing any.
Also: hold the aircraft still while System 2 thinks (`hold_to_think`) — System 2 takes 2.6–10 s
and the route is anchored at the pose the *frame* was taken from, so a moving aircraft
anchors its next route metres behind itself.

**Then the failure moved.** With turns flown as real rotations, an aircraft that got inside
its own stopping distance of a wall could no longer escape: the wall stays inside the depth
corridor across most of the arc, so every heading reads blocked. Measured: thirteen rotations,
zero metres. **Rotating is only half an escape** — it has to break contact first
(`core/planning/recovery/escape_maneuver.EscapeManeuver`: brake, 0.5 m of reverse, settle,
*then* rotate). Holding still to think makes this worse, not better: a blocked aircraft asking
from a stationary frame gets a byte-identical answer for ever.

**Don't:**
- Don't trust a config comment that describes a fix. Grep for the key in the code that reads
  it. Two of the four above were knobs nobody had ever passed.
- Don't debug this from flights. `tasks/planning/sjtu_internvla_n1/scripts/dry_run.py` flies
  the real nodes against a fake drone and a scripted server in ninety seconds, with no Gazebo,
  no GPU and no model, and it found (2) and the incomplete escape before a run was spent.
- Don't run `record_campaign.sh` from a shell that has not sourced ROS — until this was fixed
  it sourced nothing itself, so `ros2` failed on an importlib traceback, the odom wait burned
  its full sixty seconds, and every run was written off as "could not reach the area" with
  nothing anywhere naming the cause. It only ever worked by accident.
