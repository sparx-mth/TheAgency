# Lessons

Errors, gotchas, and failure modes that cost real debugging time — kept here so they're
never re-derived from scratch. Check here before debugging anything that feels familiar.

Format per entry:
- **Symptom** — what you actually observed
- **Root cause** — what was really going on (often different from the first suspect)
- **Fix / workaround** — what resolved it, or the current best mitigation if unresolved
- **Don't** — anything that looked like a fix but wasn't, or made it worse

---

## 2026-08-18 — FALCON declared the jail "explored" after 372s with 12 clusters retired behind one; the flight was fine, the frontier tests were not

**Symptom:** A mapping run in `sphera_jail` looked healthy by every control metric — reference
tracking error 0.02–0.10 m, zero `Collision detected before publishing`, roll p90 0.6°,
altitude spread ±0.04 m, voxel count climbing 7k → 43.7k — and then stopped dead. The
follower reported `holding=True` for minutes on end and `/planning/bspline` went silent. The
first instinct was a control or follower bug, because that is where every previous stall had
been.

**Root cause:** FALCON had quit on purpose, and said so in `exploration_node`'s stdout (which
goes to the container's roslaunch log, NOT to any `/root/.ros/log/*/` file — it was invisible
in the place we habitually look):

```
[ExplorationManager] Frontier number: 1, dormant frontier number: 12
[UniformGrid] Cell 15 has 1 frontiers, but no free subspace
[ExplorationManager] No frontier
[FSM] Transit state from EXEC_TRAJ to FINISH by FSM
[FSM] Finish exploration: No frontier detected
[TrajServer] Task finished, traj server shutdown
```

Twelve of the thirteen clusters were retired, not absent. Three separate tests in
`frontier_finder.cpp` did the retiring:
1. `countVisibleCells()` treats `UNKNOWN` as an occluder, exactly like a wall. But a frontier
   cell **is** the boundary of unknown space, so a ray arriving at one crosses that boundary
   by construction — the test reports "you cannot see this frontier" *because* it is a
   frontier.
2. `min_visib_num` is an absolute count of frontier cells, while the most any viewpoint *can*
   see is the cluster's own size. The bar is therefore hardest exactly where the cluster is
   smallest, and small leftover clusters are what remains late in a mission.
3. `grantFinishAmnesty()` already existed and was already wired — but it is checked *before*
   categorisation, guarded on `frontiers_` **and** `tmp_frontiers_` both being empty. In this
   failure the last surviving cluster was sitting in `tmp_frontiers_` and went dormant
   *during* categorisation, so the amnesty never fired at all.

**Fix / workaround:** `patches/fix_falcon_frontier_visibility.sh` — a bounded unknown-crossing
budget per ray (`/frontier_finder/visib_unknown_tolerance=2`), a cluster-relative visibility
bar applied only to viewpoints that are not `isNearOccupied`
(`open_visib_fraction=0.5`, `open_visib_floor=4`), and a second categorisation pass that
re-checks the amnesty *after* the loop. All three default to upstream behaviour in the source
and are switched on in `nav_stack.launch`.

Written as an anchor-based script, not a `.patch`: falcon_sjtu's `falcon_vp_audit`,
`falcon_visib_unknown_tolerance` and `falcon_open_visib_bar` all target the same regions of
`frontier_finder.cpp` that our own ports (`falcon_deadend_guard`,
`falcon_blocked_region_ttl/widen`, `falcon_publish_fail_blacklist`) already rewrote, and fail
to apply in *any* order — verified by trying all four against the live tree.

**Don't:** Don't debug a stalled mapping run from the follower's heartbeat alone —
`holding=True` with a low `pos_err` is what perfect tracking of a *stopped plan* looks like,
and it is indistinguishable from perfect tracking of a moving one. Check
`/planning/bspline` liveness and the exploration FSM state first. And don't grep only
`/root/.ros/log/*/` for FALCON's own reasoning; `exploration_node` and `traj_server` write to
the roslaunch stdout redirect instead.

## 2026-08-19 — a sentinel that gates autonomy must expire, and `pkill -f` matches the shell that runs it

**What happened:** an autonomous campaign sat idle for **13.5 hours**. Nothing crashed. A
`runs/PAUSE` sentinel had been created to let a calibration flight own the aircraft alone,
the calibration finished normally twenty minutes later, and nothing ever removed the file.
The supervisor did exactly what it was told and logged `PAUSE sentinel present -- holding`
1,600 times.

**The design lesson, which is the important one:** a pause protects an edit or a manual
flight — minutes of work. The thing it gates is a loop meant to run for days unattended.
Those two lifetimes are wildly mismatched, so the sentinel has to carry an expiry: past
`CAMPAIGN_PAUSE_MAX_AGE_S` (30 min) the supervisor now treats it as forgotten, says so
loudly, and resumes. `STOP` deliberately does **not** expire — stopping is intent, pausing
is a temporary courtesy, and only one of them should survive being forgotten. Any state that
can silently disable an autonomous system should be asked the same question: what happens if
whoever set it never comes back?

**Second trap, hit while fixing the first:** `pkill -f "falcon_campaign/supervisor.sh"` killed
the shell that ran it — the pattern matches that shell's own command line. This file already
documents the same self-match for `pgrep -f` inside a container watchdog; it is exactly as
true for `pkill` on the host, and it kills the process doing the cleanup, so the rest of the
command (restarting the thing) never runs. Match on something the caller cannot contain, or
read the pid from a lockfile.

**Don't:** don't leave a "temporary" flag as the only thing standing between an autonomous
loop and running. And when a turn ends with such a flag set, the *same* turn must schedule
whatever will come back to clear it — that scheduling step, not the flag, was the real
omission here.

## 2026-08-18 — a bare `import` broke every automated Sphera restart, and three layers of "success" hid it

**Symptom:** an autonomous campaign refused to fly for six consecutive cycles, each one
correctly reporting `battery_ok: false` and then correctly asking for a Sphera restart. The
restart reported success every time. `R1` stayed up for two hours with `voltage: 0.0,
percentage: 0.0`.

**Root cause:** `sphera_battery_watchdog.py` did `import sphera_gui_automation` — a bare
import that resolves only when the file is run as a script from its own directory. Invoked
the module way (`python3 -m sparx_agency.tools.sphera_battery_watchdog`, which is how any
programmatic caller will do it) it died instantly with `ModuleNotFoundError` before executing
a single line of its own logic — including the `docker rm -f R1` that the whole procedure
depends on.

**Why it stayed hidden is the more useful half.** Three separate things each turned a failure
into a "success":
1. the caller ran the restart with `sh(...)` and **ignored the return code**;
2. it then treated **"the drone container exists"** as proof the restart worked — but a
   still-running *old* container satisfies that just as well as a fresh one;
3. the thing actually wanted (a charged battery) was never checked, even though it is one
   call away and is the entire reason for restarting.

**Fix:** make the import work both ways (`try: from sparx_agency.tools import ... except
ImportError: import ...`), check the return code and print the stderr when it is non-zero,
and — the general lesson — **verify the goal, not a proxy for it**. `_fresh_drone_ready()`
now returns true only when the container is up *and* the battery has actually reset.

**Don't:** don't let a subprocess wrapper discard a non-zero exit; and when writing a
"did it work?" check, ask what the operation was *for* rather than what it *touched*. A
proxy that the pre-existing state already satisfies cannot detect failure at all.

## 2026-08-18 — a stale docker image made ten committed FALCON fixes silently inert, and exploration quit after 26 seconds

**Symptom:** FALCON "froze and never replanned". `/planning/replan` sat at `2`
(EXPLORATION_FINISHED) for 6.5 hours while its own frontier finder still reported 560
frontier points and 736 dormant ones. `traj_server` was gone, so `/planning/pos_cmd` had
**zero publishers** and the reference follower held station forever — which reads exactly
like a control bug.

**Root cause:** `falcon-ros:noetic` was built at 11:17; the commit adding the finish-grace
fix landed at **14:36**. The container had no `/catkin_ws` bind mount, so the image content
was authoritative and three hours of committed C++ patches were simply not in it.
`nav_stack.launch` dutifully set `/fsm/finish_grace_*` and the frontier-visibility params,
roslaunch echoed them, and **nothing read them**: `strings
/catkin_ws/devel/lib/exploration_manager/exploration_node | grep -c finish_grace` returned
`0`. FALCON's `FINISH` state is absorbing, and it is entered on the *first* cycle where
`updateFrontierStruct()` returns 0 — so without the grace patch, 26 seconds of exploration
was enough to end the mission permanently.

**Fix:** rebuild, then *verify the artifact rather than the source*. `bringup.py`'s
`assert_falcon_patches()` now greps each compiled binary for a marker string per patch and
refuses to fly if any is missing. Checked at **bring-up**, not build time: a build-time gate
only protects the build that runs it, while this catches any stale image that later gets
started.

**Don't:** don't infer from "the launch file sets the parameter" that the parameter does
anything — for a vendored C++ node, confirm the string is in the binary. And don't debug a
"planner froze" symptom in the follower until you have checked `/planning/replan`'s value and
whether `/planning/pos_cmd` has a publisher at all; both are one `rostopic` call.

## 2026-08-18 — `SpheraPawnState.velocity` exists, is documented as m/s, and is all-zero in this build

**Symptom / temptation:** `rooster_ground_truth_localization.py` differentiates the truth
pose and low-passes it with `velocity_filter_tau_s=0.25` to get a usable velocity. That lag
lands straight on the velocity servo's proportional term and is why `servo_kp` had to be cut
from 220 to 90 after a ~1.15 Hz limit cycle. `SpheraPawnState` carries a `velocity` field
(`geometry_msgs/Vector3 velocity #m/s`), and `manual_flight_logger.py` even documents it as
"true physics-engine linear velocity (not derived by differencing position)" — so switching
to it looks like a free removal of 250 ms of feedback lag.

**It is all-zero.** Measured live on `sphera-backend:rooster`: `ros2 topic echo
/R1/sphera/state` shows `velocity: {x: 0.0, y: -0.0, z: 0.0}` while the aircraft is flying at
0.5 m/s and its position is visibly changing. The field is declared and never populated.

**Why it matters more than "the optimisation is unavailable":** a controller that closes on
that field sees a permanent zero, reads it as "not moving", and winds its integrator to full
deflection while the aircraft is already at speed. The guard that caught this treats an
all-zero field as suspicious only when the differentiated position disagrees, waits 25
samples to avoid tripping on a real standstill, then logs an ERROR and falls back to
differentiation for the rest of the run.

**Don't:** don't trust a message field because it is declared and documented — check that it
is populated, against motion you can independently see. And when adding a feedback source,
make "the source went dead" a distinguishable state rather than a plausible zero.

## 2026-08-18 — a calibrated axis curve is still open loop, and it under-delivered by 3x in flight

**Symptom:** With the measured ManualControl axis curve in place (forward dead until ~620
counts, ~1.25 m/s at 1000), a full FALCON exploration leg commanded a mean 0.30 m/s and
achieved a mean 0.11 m/s against Sphera ground truth — ratio 0.31, i.e. the aircraft could
not keep up with a plan that had been deliberately slowed down for it. Achieved speed was
also spiky (max 0.99 m/s against a 0.60 m/s ceiling).

**Root cause:** Two compounding things. The curve is a fit of the *steady-state* response to
an isolated axis step, and an exploring aircraft is almost never settled — it is accelerating,
turning, or being braked, and every one of those states needs more deflection than the fit
predicts. Separately, the follower's `max_measured_speed_xy` backstop **zeroed** translation
outright the instant measured speed touched the cap, so the loop was bang-bang: full command,
overspeed, zero, coast, full command. That is where the 0.99 m/s spikes came from.

**Fix / workaround:** Close the loop. `core/control/axis_velocity_servo.py` adds a PI
correction on top of the curve (which becomes feed-forward), with dead-band-aware anti-windup,
a bounded correction, and a reset on stop; the velocity it closes on is differentiated from
Sphera ground truth and published by `rooster_ground_truth_localization.py` as
`/R1/velocity_truth`. The follower's hard cutoff became a linear taper that only bites past
the cap.

**Don't:** Don't tune the curve harder instead — the number it is missing is not a constant,
which is exactly why an integrator finds it and a re-measurement doesn't. And don't close this
loop on the autopilot's own velocity estimate; PX4's was measured drifting convincingly while
the aircraft sat still (see the 2026-08-17 offboard entry).

## 2026-08-18 — `ros2 topic echo | grep -q` inside a short `timeout` hangs a bring-up script for 13 minutes

**Symptom:** A bring-up script's "wait for the drone to come back" loop
(`docker exec it ... timeout -s KILL 4 ros2 topic echo /R1/state | grep -q "^armable"`)
never satisfied its condition and sat spinning for 13 minutes, even though running the same
`ros2 topic echo` by hand printed `armable: true` immediately.

**Root cause:** `ros2 topic echo`'s stdout is block-buffered when it is a pipe rather than a
TTY, and `timeout -s KILL` discards whatever is still sitting in that buffer. Whether the
condition is ever met depends on whether the message rate happens to fill a buffer inside the
timeout window — so the same loop passes some runs and hangs others, which is worse than
failing outright.

**Fix / workaround:** Redirect to a file inside the container and read the file, never pipe
`ros2 topic echo` into a short-lived `timeout`:
`timeout -s KILL 6 ros2 topic echo /R1/state > /tmp/state_probe.txt; grep -m1 '^percentage:' /tmp/state_probe.txt`.
`sparx_agency/tools/sphera_battery_watchdog.py` already does the equivalent — that is why the
watchdog's own battery poll has always been reliable while ad-hoc loops in shell scripts are
not.

**Don't:** Don't "fix" it by lengthening the timeout or adding `stdbuf` — the buffering is on
the far side of a `docker exec`, so a host-side `stdbuf` does nothing, and a longer timeout
only makes the hang less frequent, not absent.

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

---

## 2026-08-17 — `sphera-restart.sh` alone does not reset the drone's battery

**Symptom:** Built `sphera_battery_watchdog.py` to auto-restart Sphera on low battery via
`~/.sphera/sphera-restart.sh`. Live test: `drone_simulator` cleanly recreated (confirmed via
`docker inspect -f '{{.State.StartedAt}}'`), but `/R1/state`'s battery stayed flat at 97%
instead of resetting to ~99-100% as `fly-rooster-sphera/SKILL.md` documents.

**Root cause:** `R1` is a sibling container Sphera spawns via the host Docker socket
(bind-mounted into `drone_simulator`, not a child of its cgroup) — bouncing the
`drone_simulator` engine container alone does not kill or recreate it, confirmed via
`docker inspect`'s `StartedAt` predating the restart. Matches the 2026-08-13 entry above:
Sphera's green ▶ Play button only spawns a new `R1` "if R1 dies mid-scenario" — it does NOT
replace a still-alive one. A real full quit-and-relaunch of the Sphera app apparently kills
`R1` as a side effect (which is why the user's manual ritual "just works"); the lighter
`docker compose down`/`up` cycle `sphera-restart.sh` does does not.

**Fix / workaround:** Force-remove `R1` explicitly (`docker rm -f R1`) before running
`sphera-restart.sh`, guaranteeing the next Play spawns a genuinely fresh instance. Confirmed
live: with this fix, `R1` came back with a fresh container (`StartedAt` after the restart)
and battery read 99%.

Also confirmed live (2026-08-17): entering the scenario after a restart requires clicking
the specific drone (`Rooster_1`) in the "Drones Assignment" panel *before* pressing Play —
skipping straight to Play raises a "Not all drones have operators associated with them, do
you want to continue anyway?" dialog. Answering "Yes" still seems to work for this stack
(control goes through ROS2 to `it`, not Sphera's own Operator role), but the clean/intended
order is Continue → choose role (Manager) → Start → Standalone → click `Rooster_1` in Drones
Assignment → Play.

**Don't:** Don't assume "the engine container came back up cleanly" means the drone's state
reset too — `drone_simulator` and `R1` have independent lifecycles; always check `R1`
specifically (`docker inspect ... StartedAt`, or just the battery reading itself).

---

## 2026-08-17 — automating the post-restart GUI walkthrough: window exists before Sphera is interactive

**Symptom:** Extended `sphera_battery_watchdog.py` to drive the whole post-restart GUI
sequence itself (`sphera_gui_automation.py`, new) instead of leaving it for a human, since
the user won't be at the keyboard when this fires. First live end-to-end test (real battery
trigger -> `sphera-restart.sh` -> immediately drive the GUI): failed silently — no exception,
no error, `enter_scenario()` returned `True`, but `R1` never appeared and battery stayed
unreadable. Called `enter_scenario()` again moments later on the exact same window/
coordinates and it worked immediately.

**Root cause:** The X11 window that `xdotool search --name Sphera` matches exists — and is
therefore found by `wait_for_window()` — well before Sphera/Unreal is actually rendering the
"Welcome to SPHERA" screen and accepting input. Clicking `Continue` the moment the window is
found lands on nothing (a still-loading/black frame), and every click after that keeps
computing coordinates for screens that never actually advance — the whole sequence silently
misfires against screen 1, with no error at any step to catch it. Confirmed by taking a
screenshot right after the failed run: still exactly on "Welcome to SPHERA", not stuck
mid-sequence somewhere later.

**Fix:** Two changes to `sphera_gui_automation.enter_scenario()`: (1) a fixed 10s warmup
sleep after the window is first found, before clicking anything — window-*exists* is not
window-*interactive*, and there's no cheap way to detect Unreal's actual render-readiness
without image-diffing, which was judged not worth the complexity here; (2) an idempotent
redundant second `Continue` click 1s after the first (harmless no-op if the first already
landed — it hits empty background on the next screen). Re-tested twice more after the fix:
both fully unattended runs succeeded, ~38s total (trigger to verified-fresh-R1-and-battery).

**Don't:** Don't trust "the window exists" (`xdotool search` returning a hit) as "the app is
ready for input" for any Unreal/game-engine window — window creation and first-interactive-
frame are two different milestones, and the gap between them was large enough here to break
every click after the first. Also don't assume a GUI-automation function returning normally
means it worked — this is exactly why `restart_and_reenter()` verifies success independently
afterward (`docker ps` for `R1` + a real battery reading), not by trusting
`enter_scenario()`'s return value alone.

## 2026-08-17 — Rooster's ManualControl x/y axes have a ~65%-of-stick DEADZONE; this is why trajectory tracking was jerky all day

**Symptom:** Every attempt at smooth low-speed trajectory following produced jerky, unpredictable horizontal motion, whatever the follower or control law. FALCON's exploration mode had already been "fixed" by pegging commands to a constant magnitude (`force_mode=fixed` bang-bang) rather than tracking proportionally — which always looked like a workaround for something unexplained.

**Root cause, measured against SPHERA GROUND TRUTH while hovering in Posctl:**

| ManualControl axis | forward (x) m/s | lateral (y) m/s |
|---|---|---|
| 150 / 300 / 450 | 0.002 / 0.011 / 0.001 | ~0 |
| 600 | 0.000 | ~0 |
| 700 | **0.261** | ~0.00 |
| 1000 | **1.248** | **-1.023** |

So the horizontal axes are **dead until roughly 600-650 counts (65% of full stick)**, then ramp steeply and non-linearly. Lateral is even deader than forward (still ~0 at 700). Yaw, by contrast, is well behaved with only a small deadzone: axis 200 -> -0.278 rad/s, axis 400 -> -0.845 rad/s (~-1.75 to -2.1 rad/s extrapolated to full stick, negative for positive `r`, consistent with the documented sign flip).

**Why that produced exactly the observed jerkiness:** FALCON cruises at 0.15-0.2 m/s. `rooster_twist_control_adapter.py` converts that with `max_linear_x/y = 0.25 m/s @ axis 1000`, so a 0.15 m/s request becomes axis ~600 and 0.2 m/s becomes ~800 — i.e. every normal command lands **right on the deadzone edge**, where the real response swings between 0.00 and 0.26 m/s for a few counts of change. That is not a tuning problem; it is an actuator-resolution problem.

**Consequences that matter for design:**
- **Minimum controllable horizontal speed is ~0.26 m/s.** Anything slower cannot be commanded continuously at all — only approximated by pulsing, which is precisely what `PulseShaper`'s `force_mode=fixed` was doing. That bang-bang default was a rational response to this hardware, not a mistake.
- Therefore **"fly slower for better tracking" is backwards here.** Proportional control only exists above the deadzone, so FALCON's cruise should be raised into the responsive band (~0.4-0.8 m/s) if smooth tracking is the goal.
- Usable resolution is only ~350 of 1000 counts, and the x and y axes are asymmetric, so any velocity->axis conversion must be a **calibrated inverse (deadzone offset + measured gain), per axis** — a single linear `v/max*1000` scale factor cannot work.
- `max_linear_x/y = 0.25` is wrong in both directions depending on regime (it under-reads full-scale ~1.25 m/s while over-promising that small values do anything). It was never live-validated; its own docstring says so.

**Don't:** Don't calibrate these axes from PX4's `MPC_VEL_MANUAL` (2.0 m/s) — the vendor's `rooster_manager` sits in between with its own `manual_ctrl_deadband` (150) and scaling, so the PX4 param is not the effective end-to-end gain (measured full-scale was ~1.25 m/s forward, not 2.0). And don't measure any of this from PX4's own `UAVState.position/velocity` — use `/R1/sphera/state` ground truth (see the offboard entry below for how PX4's estimator drifts convincingly while the aircraft is motionless). Also avoid measuring while battery is low (<~25%): samples taken at 19% had a "forward" command produce mostly lateral motion as thrust authority and yaw drift corrupted the body-frame projection.

## 2026-08-17 — Rooster's native PX4 offboard setpoints do NOT actuate Sphera at all (and how a convincing false positive nearly hid that)

**Symptom / question:** After a day of poor Rooster control through the RC-style `ManualControl` axes, the question was whether ROBOTICAN's native `fcu_driver` setpoint topics (`local/velocity`, `body/velocity`, `attitude`, `local/position`) are real working control interfaces — the plan being to use one of them the way the SJTU stack uses its Gazebo velocity plugin.

**First answer was WRONG, and the way it was wrong matters.** Streaming `LocalVelocityCommand` in Offboard produced beautiful numbers: commanded +0.50 m/s vs PX4-measured +0.478 (ratio 0.96), commanded +0.40 vs +0.419 (1.05), hold drift only -0.11 m over 5 s, ±2% spread, plus `position` deltas that matched the commanded distance. It was written up as proven. **It was measured entirely from `/R1/fcu/state` (`UAVState`) — PX4's own estimator — which was agreeing with itself while the aircraft never moved.**

**The truth, from Sphera ground truth (`/R1/sphera/state`):** displacement was **0.00 m on every axis** across multiple offboard runs. PX4 engaged Offboard (`fcu_mode == "Offboard"`, held 19 s+), even logged `FCU [Info]: Takeoff detected`, while the drone sat on the ground with `ranger` pinned at 0.13 m. In the same session the **`ManualControl` path flew for real** — ground truth `dz = +1.53 m`, then holding **TRUTH z = 1.75 m / ranger 1.77 m rock-steady for 10+ s**.

**Root cause (architectural):** Sphera's `sphera_physical_rooster_backend_node` drives the simulated physics from the **vendor's `ManualControl` → `rooster_manager` → backend** pipeline, *not* from PX4's motor/actuator outputs. PX4 runs only as an estimator and mode manager. So every PX4-native setpoint interface is inert **in Sphera**, no matter how correct the messages are. (On real hardware they may work fine — this is simulator-specific, and there is no way to validate them in Sphera.)

**Consequence:** in Sphera, the ManualControl axis interface (-1000..1000) is the only actuation route, so trajectory tracking has to close the loop on our side.

**Don't:** Don't validate a control interface using the same autopilot's own state estimate — it will happily confirm your command while the vehicle is motionless. Use an independent ground truth (here `/R1/sphera/state`). Don't trust `set_offboard`/`force_arm` returning `success=True`, or `fcu_mode == "Offboard"`, or even PX4's `Takeoff detected`, as evidence the aircraft moved. And `body/velocity` specifically satisfies the driver's "setpoint present" gate (Offboard *will* engage on it) while tracking nothing — it is not a usable interface even before the actuation issue.

**Sequencing rules that DO engage/hold Offboard** (kept because they're correct and needed for real hardware): `ManualControl` must stream continuously or PX4 raises `Failsafe enabled: No manual control stick input`; `KeepAlive.requested_flight_mode` is enforced by `rooster_manager` and drags PX4 out of Offboard after ~2.0 s unless set to `NONE (0)`; the setpoint stream must never go stale (>~1 s) or `set_offboard` fails with `'Failed to start Offboard mode: No Setpoint Set'` (publish it from its own timer, not inline in a phase loop); minimise armed-on-ground time or PX4 fires `Disarmed by auto preflight disarming`. Note `rooster_manager` only relays `ManualControl` to the FCU while `KeepAlive` requests a real flight mode, so `NONE` triggers `Manual control lost` — break that catch-22 with PX4 `COM_RCL_EXCEPT = 4` (bit 2 excepts Offboard from RC-loss failsafe; default 0, with `NAV_RCL_ACT = 3` = Land). Set params via rclpy: `ros2 service call` crashes with `munmap_chunk(): invalid pointer` on this Foxy/CycloneDDS build.

**Also, better telemetry:** `/R1/fcu/state` (`UAVState`) beats `/R1/state` for control work — `fcu_mode` as a readable string plus PX4 `position`/`velocity` — but treat its position/velocity as **estimates, not truth** (they drift; its local origin is also re-initialised per boot and drifted ~2 m when a run ended still armed/airborne, so never use `position.z` as absolute altitude — use `ranger`, and always force-disarm after landing). `RoosterState.flight_mode` reports 0 (`NONE`) during Offboard because the vendor enum has no Offboard value.

## 2026-08-17 — a vendor example script left running since the morning fought every flight-control test run that day

**Symptom:** A full day of Rooster flight-control investigation (altitude-hold PD retuning, switching flight modes, per-axis characterization) kept producing inconsistent, sometimes-wildly-unstable results that didn't match earlier clean tests on the same code -- including the FCU's own log repeatedly showing `FCU flight mode change detected: Posctl -> Altctl -> Posctl -> Altctl...` every 2-3s, even though only one process (`rooster_command_unit`, requesting ALTITUDE) should have been asking for a mode at all.

**Root cause:** `ros2 topic info /R1/keep_alive --verbose` showed **3 publishers**, not 1: `rooster_command_unit` (ours, correct) plus a node named `position_fly_controller` plus a bare/unnamed DDS participant. `position_fly_controller.py` (`sparx_agency/robots/ROBOTICAN/examples/src/`, a vendor-example-style script, launchable as a `mission_control.py` service) had been started via mission_control at **10:49 that morning** -- hours before any of the day's testing began -- and was never stopped. It independently publishes its own `ManualControl` and `KeepAlive` (requesting `FLIGHT_MODE_POSITION`, per its name) at its own rate, on the exact same topics `rooster_command_unit` owns. Two processes were flying the same drone the entire day, each periodically overwriting the other's flight-mode request and z-axis command -- this is the same *class* of bug already documented for `RoosterManualControl`/`PathRunnerNode` (see `[[project_falcon_robotican_bridging]]`-adjacent memory: "single owner" is asserted in code comments but never enforced), just a third, previously-uncatalogued instance of it, and the most damaging one found yet because it ran silently for hours before anyone thought to check `ros2 topic info --verbose` for publisher count.

**Fix:** Killed `position_fly_controller.py`. Immediately confirmed live: `flight_mode` held at 4 (Altitude) with zero flickering across a full takeoff+hold, roll/pitch stayed under 0.4° the whole time, and ranger climbed smoothly and monotonically (no more wild swings). This single process being alive explained essentially all of that day's instability -- more than any PD gain, slew limit, or flight-mode choice investigated before finding it.

**Don't:** Don't trust that only the processes *you* started are the only things talking to a drone's control topics, especially on a shared host with a long-running `mission_control.py` (services started hours or days earlier persist across unrelated sessions/conversations). **Before any flight-control debugging session, check `ros2 topic info /<id>/manual_control --verbose` and `.../keep_alive --verbose` for publisher count > 1** -- this is now the first thing to check, ahead of gain tuning, flight-mode selection, or plant characterization, all of which are meaningless while a second, uncoordinated publisher is fighting on the same topic.

## 2026-08-17 — altitude "rises and falls with no control": the z-axis response is a narrow step, not a smooth thrust curve, and the PD loop's correction range was 5-40x wider than it

**Symptom:** Every Rooster flight this session (and, per the user, prior sessions) shows altitude climbing and sinking with no apparent control -- ranger swinging anywhere from ground level to 6+ meters, never settling, independent of which follower or control law was driving x/y/yaw.

**Root cause:** Measured directly with `altitude_hold_kp`/`kd` forced to 0 (pure open-loop, one fixed z value held constant from a ground start, no correction at all): `z<=690` (tested at 550, 625, 670, 690, each held 16-20s) produced **literally zero climb** -- ranger frozen at ground level to 3 decimal places the entire time, despite `armed`/`airborne` telemetry both reading true throughout. `z=700` produced a fast, sustained ~1.6 m/s climb that did NOT level off on its own -- it climbed straight into the room's real physical ceiling (~3.4-3.5m, matching `sphera_jail.yaml`'s own documented ceiling height) and sat pinned there, stable, for the rest of the test. The entire *effective* control band sits inside a ~10-unit window near 700, out of the axis's full 2000-unit range -- consistent with a discrete ground-idle/landing-detector lockout gate (the FCU refusing to leave "landed" state below some commanded-rate threshold) rather than a continuous thrust-vs-lift curve; `rooster_unit.py`'s own history already contains a matching data point (`climb_z` raised 600->1000 specifically "to escape PX4's landing-detector confirm window"). `_altitude_hold_tick`'s `altitude_hold_max_correction=380` let ONE tick's correction move `z` across virtually the entire span between "zero lift" and "full climb to ceiling" -- every apparent "instability" was this loop's own correction saturating past that narrow band in either direction, not a tuning-quality problem with the PD gains themselves.

**Fix:** Added a separate, tighter slew-rate limit (`altitude_hold_max_step`, default 15 units/tick at the existing 10Hz loop rate) on top of the existing correction clamp -- `_altitude_hold_tick` now moves the actual commanded `z` toward the PD-computed target by at most this much per tick, regardless of how large the instantaneous correction is. Verified live: with real gains restored (kp=500, kd=600, max_correction=380 unchanged) plus `altitude_hold_max_step=15`, `target_ranger_m=1.5`, ranger held **1.33-1.62m for the full 15s test** -- tight, converging, no runaway climb, no floor-sink. Compare to the exact same gains without the step limit, which produced the original unbounded 0.13-6m swings.

**Don't:** Don't assume a fresh round of PD gain-tuning (different kp/kd) fixes this -- the problem is that ANY gain, applied without a per-tick step limit, can compute a `wanted_z` correction that overshoots the narrow effective band in one tick; the axis-level slew limit is the fix, not the gains. Also don't trust an "armed: true, airborne: true" telemetry pair as proof the vehicle is actually climbing when testing z-axis response in isolation -- cross-check `docker logs R1 | grep uav_state_subscriber_cb` for the vendor CDR-corruption signature (see `project_sphera_cyclonedds_interface` memory) before reading a flat ranger as a real physics measurement; this session hit that corruption at least twice mid-investigation and it produced misleading "no response at any z value" results until a fresh Sphera restart cleared it.

## 2026-08-17 — first live flight of a new Rooster follower capsized the drone; neither follower had any attitude awareness

**Symptom:** ~90s into the first live test of `rooster_bspline_follower_node.py` (new
`core.control.velocity_servo`-based follower, swapped in mid-flight for
`falcon_exploration_follower_node.py` to A/B tracking quality), `/R1/state` showed `ranger`
jump to 6.09m (the flight box is 4.8m tall) and `roll` at 0.88 rad (~51deg), worsening to
-1.07 rad (~61deg) over the next ~25s even after commanding `stop` via `/R1/cmd_nav` and
killing the follower node entirely — nothing published to `cmd_vel_raw` and the tilt still
grew. Battery drained abnormally fast alongside it (92% -> 52% in under 3 minutes).

**Root cause:** A capsize, matching this repo's own prior finding (`b4ec96a0`, "the capsizes
are not a control problem" — a contact tips the airframe between monitor samples, with no
threshold-crossing window a reflex could catch, and past the physical recoverability ceiling
no command can right it). What made THIS one findable as more than "just bad luck": **neither
`falcon_exploration_follower_node.py` (the production follower) nor the new
`rooster_bspline_follower_node.py` read roll/pitch at all** — both tracked yaw only. Every
other follower already in this codebase (`falcon_sjtu`'s `bspline_follower_node.py`) has a
tilt-cutoff reflex specifically because of this; Rooster's followers never got one, so nothing
ever cut horizontal drive once the aircraft was already past recoverable tilt — the follower
kept commanding translation into a tip that was already unrecoverable, for as long as it kept
running.

**Fix:** Ported the tilt-cutoff reflex (`~tilt_limit_deg`, default 15deg, same default as
`falcon_sjtu`) to both `falcon_exploration_follower_node.py` and
`rooster_bspline_follower_node.py`: the instant `|roll|` or `|pitch|` crosses the limit, cut
to zero Twist and reset the tracker/servo's integrators, every control tick, ahead of the
demo-mode gate. This does not prevent a capsize (nothing can, per `b4ec96a0`) — it stops the
follower from fighting the recovery attempt or continuing to drive translation into an already-
tipped airframe. Recovery from the capsize itself was via
`sphera_battery_watchdog.py --once` (force an immediate Sphera restart+GUI-re-entry cycle,
regardless of battery) — ~36s to a fresh, level, 99%-battery `R1`.

**Don't:** Don't add a NEW follower (or copy an existing one) for this airframe without this
reflex — it's not exotic safety machinery, it's the one thing standing between "aircraft tips
over" and "aircraft tips over AND keeps getting horizontal-drive commands while it happens."
Also don't assume commanding `stop` (a `cmd_nav` action) or killing the follower node stops an
already-capsized aircraft from getting worse — once physically past the recoverability
ceiling, only a sim restart (or, on hardware, a real recovery/kill) fixes it; the tilt-cutoff's
job is to stop BEFORE that point, not to undo it after.

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
