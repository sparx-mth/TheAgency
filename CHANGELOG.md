# Changelog

All notable changes to this project are logged here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/) — one line per change, written for
future-you, not for a commit log.

## [Unreleased]
### Added
- nav_debug replay now shows the **velocity-loop input vector** explicitly: the target linear
  `(vx, vy, vz)` and angular `wz` handed to the velocity-closing block, in m/s and rad/s (and
  deg/s), beside the gauges that were previously the only view of it. The tracker's own
  `command`/`command_requested` records — written on ~99% of control ticks and read by nothing —
  now reach the frame, so a tick the pulse shaper or the GO gate changed is flagged
  (`SHAPER CLIPPED` / `GO GATE BLOCKED`) instead of being invisible.
- nav_debug replay now shows the drone's **actual state beside the reference**: a
  target/actual/error table over position, velocity and heading (world frame, heading error
  wrapped), replacing a reference display that gave only a scalar speed and left the outcome
  to be read off the map. The measured state comes from the follower's own odometry, else
  Sphera ground truth, else a centred difference of the recorded pose spine — and the lane
  names which, because the three are not interchangeable. New `nav_debug/state_source.py`.
- `falcon_exploration_follower_node.py` records a `state` section (its own odometry pose,
  velocity and yaw) in the nav_debug control trace, so future runs carry a measured velocity
  without needing the second recorder. Additive: it runs after the command is published,
  inside the existing swallow-everything guard, and `~nav_debug_trace:=false` removes it.
- `nav_debug/why.py` narrates **commanded-but-not-moving** (>0.15 m/s asked, <0.05 m/s
  measured) — 15.5% of frames on run `nav_debug_20260906_235809`. No commanded-vs-commanded
  comparison could ever have detected it.

- nav_debug draws the joystick command as the **transmitter**: two Mode-2 sticks (throttle
  and yaw on the left, forward and lateral on the right) showing the deflection a pilot's
  hands would hold to send the same command. They replace three needle gauges that covered
  roll, pitch and yaw only — the throttle, which the altitude hold writes underneath the
  planner, previously had no gauge at all and was the least visible axis on the screen.

### Fixed
- **The exploration follower no longer flies backward.** A demand pointing behind the
  nose is now answered by turning in place first (`scripts/heading_gate.py`, on by
  default, with a hard reverse clamp at the publish boundary that only the escape
  reflex may bypass). The align gate that was supposed to prevent this lived inside
  the follower's `not use_lateral` branch, so a lateral-enabled flight skipped it
  entirely. Measured on `nav_debug_20260906_235809`: 9.5% of airborne driving ticks
  commanded backward flight, and those ticks sat within 0.30 m of mapped geometry 33%
  of the time against 9% otherwise (p10 clearance 0.00 m vs 0.30 m). The only camera
  faces forward, so backward flight is blind flight. Replaying the run through the new
  logic: a backward demand still arises on 10.2% of ticks, **85% of it answered by a turn
  and 15% by the clamp**, and 0% of it reaches the wire. The split matters — it is how a
  future run tells a working gate from a clamp quietly papering over one.
- **A finished trajectory is no longer flown as if it were live.** `traj_server`
  republishes its frozen endpoint with a fresh stamp at 100 Hz and never changes
  `trajectory_flag` (READY on 6627/6627 samples), so neither of the follower's two
  usability tests could ever fire. `ReferenceTracker3D` now classifies the reference
  stream (`plan_state.py`): a plan that has commanded no motion — translation *or*
  yaw — for longer than a grace period is finished, and the aircraft flies to its last
  waypoint and holds there instead of chasing it indefinitely. Note what this is *not*:
  the approach is not slower (commanded speed over finished-plan ticks is p50 0.054, p90
  0.729, max 1.33 m/s — the position loop still pulls at ~1 m/s from 1.7 m out). The win
  is that the aircraft **arrives and stops**, and that beyond `endpoint_reach_m` (3 m) it
  refuses the approach entirely and holds position — a case the recorded flight spent
  11.3 s in, drifting from 1.78 m to 9.41 m from a dead endpoint while still driving.
- **The position loop no longer brakes backward onto a point the aircraft has passed.**
  Being ahead of schedule is bounded (`max_lead_error_m`, 0.25 m) instead of flown out;
  cross-track, the component that keeps the aircraft off walls, is untouched. The
  error split now takes an explicit direction of travel and remembers the last real
  one, so a finished plan no longer reports its whole lag as cross-track.
- `max_yaw_rate_deg` in the follower defaulted to 45 while `nav_stack.launch` has
  passed 90 all along — an ad hoc launch silently flew at half the measured ceiling.
- `falcon_replan_from_pose.patch` now anchors a replan at
  `odom_pos_ + odom_vel_*replan_duration_` rather than the pose at solve time. Without
  that lead the guard planted each new curve *behind* a moving aircraft, which is why
  the 2026-09-02 attempt at a 0.6 m threshold "forgave the deviation" and was reverted;
  with it, `fsm_replan_from_pose_drift` drops to 0.05 (effectively always). **Needs a
  docker image rebuild and a flight — the C++ half is unverified from the host.**

- nav_debug's `TO DRONE (cmd_nav)` block was empty on every Sphera run ever recorded, for two
  independent reasons: `drone_cmd` could only be filled from the XTEND certainty CSV (Sphera
  writes none), and the branch that drew it was taken only when the ROS2 half was *absent* —
  i.e. exactly when there were no counts to draw. It is now filled from the actuator lane's
  `ManualControl`, drawn unconditionally as a per-axis table (forward/lateral/vertical/yaw,
  each with requested counts, counts sent and % of full scale), and the servo internals moved
  to their own `AXES` section instead of replacing it.
- nav_debug's `speed` history strip fell back to the *commanded* velocity whenever the ground-
  truth lane was missing, plotting the command against itself — a flat, perfect-looking trace
  on exactly the runs where nothing had been measured. It is now a measurement or nothing.
- nav_debug drew Sphera runs recorded without the ROS2 half on the **XTEND gauge envelope**,
  mis-scaling every command gauge by ~3.5x (0.45 vs 1.566 m/s full scale) on the runs with the
  least other information. FALCON's exploration lanes now count as Rooster evidence.
- A missing or mismatched `ros2/` half is now reported instead of rendering as a blank panel:
  `NavSession` warns (naming `run_nav_debug_recorder.sh`), warns separately when a `--ros2`
  directory loaded but belongs to a different flight, and the player prints `join_report()` —
  which existed, was documented, and was called by nothing.
- `rooster_twist_control_adapter.py`'s `max_yaw_rate` recalibrated from a never-validated
  0.5 rad/s to 1.8 rad/s, derived from a logged manual flight's actual turn-rate behavior
  (~4x too low previously — any planner-requested yaw rate was executed much faster than
  intended). See LESSONS.md and `docs/progress/entries/007-rooster-velocity-controller.md`.
- Broken `torch` import in the project venv (`venv/`): removed an orphaned cu12 NVIDIA package
  cluster conflicting with `torch==2.11.0`'s actual cu13 requirement, and force-reinstalled two
  cu13 packages (`nvidia-cudnn-cu13`, `nvidia-nvshmem-cu13`) whose library files were missing
  despite being reported installed (corrupted/interrupted earlier install). See LESSONS.md.

### Added
- New `detector` container (`docker/Dockerfile.detector`, `bake.hcl` sibling target off
  `perception`, `docker-compose.detector.yml`, started persistently as `detector_dev`) running
  the YOLO-World detector sidecar — torch/ultralytics/CLIP kept out of `perception`/`robotican`
  since `yolo_world_trt` is shared task infra, not ROBOTICAN-specific. `run_object_mission_sphera.sh`
  now launches the sidecar via `docker exec` into it instead of the bare host venv, and passes
  `rgb_topic:=/R1/rgb_frame_path` explicitly (previously unset, silently defaulting to XTEND's
  topic). See LESSONS.md for three build issues hit and fixed along the way (numpy pin conflict,
  a mis-built CLIP wheel, TensorRT engine/version lock).
- Built the YOLO-World-`s` TensorRT engine (`yolo_world_trt/build_all.sh s`) end-to-end for
  the first time on this PC (installed `ultralytics`, downloaded `yolov8s-worldv2.pt`) —
  `./run_object_mission_sphera.sh --detector-only` now actually starts the detector sidecar.
  `m`/`l`/`x` are not built yet.
- Sphera/Rooster fork of the "pick an object, then fly to it and land" mission stack:
  `adapter/launch/object_mission_sphera.launch`, `config/mission_sphera.yaml`,
  `run_object_mission_sphera.sh`, and a placeholder `objects_sphera.json` (no real object
  catalog exists for `sphera_jail` yet). Mirrors XTEND's existing `object_mission.launch` stack
  the same way `sphera_drone.launch` mirrors `real_drone.launch` — additive, XTEND's originals
  untouched. Dry-validated (`mission_config.py` + `--help`); not yet flown live. See
  `docs/progress/entries/005-yolo-object-navigation.md`.
- Two new `mission_control.py` services (`rooster_jetson` group): `Rooster Frame Relay ->
  Jetson (R1)` and `Rooster Jetson Frame Watcher (R1)`, wiring the existing
  `dir_push_relay.py`/`dir_watch_path_publisher.py` mechanism (previously XTEND-only, never
  tested for Rooster) so captured frames can be forwarded to the Jetson over rsync/SSH,
  additive to (doesn't change) the existing local Rooster/Falcon vision pipeline. Verified
  end-to-end through the orchestrator's own start/stop/status code, not just ad-hoc commands —
  see `docs/progress/entries/006-rooster-frame-jetson-relay.md`.
- Closed-loop PD altitude hold for the ROBOTICAN Rooster (`rooster_unit.py`), replacing
  the previous open-loop throttle constant that reliably drifted to floor or ceiling.
- `demo_mode_topic`/`demo_mode_request_topic` params on `nav_stack.launch`, letting
  `sphera_drone.launch` route Rooster's demo-mode handshake through `/R1/...` topics
  instead of the XTEND-shaped `/xtend/...` defaults.
- `rooster_demo_mode_manager.py` (new, `falcon_adapter` ROS1 node): a minimal Rooster
  equivalent of `xtend_drone_demo_manager.py` — echoes a requested demo mode back as
  the authoritative current mode, with no other side effects (deliberately does not
  auto-land on FINISH the way the XTEND version does).
- `sparx_agency/tools/sphera_battery_watchdog.py`: polls R1's battery inside `it` and, once
  it drops to 10% (re-arms above 80%), force-removes `R1` and runs
  `~/.sphera/sphera-restart.sh` — replacing the manual "exit and relaunch the simulator"
  action. The `R1` removal is required, not optional: confirmed live that
  `sphera-restart.sh` alone doesn't reset battery, since `R1` is a sibling container that
  survives a `drone_simulator` bounce untouched (see LESSONS.md's 2026-08-17 entry).
  Deliberately scoped to just that — no Falcon/bridge/node reconnect logic — wired into
  `mission_control.py` as a new "Sphera Battery Watchdog" service card (`rooster_watchdog`
  group). See `docs/progress/entries/008-sphera-battery-watchdog.md` and the
  `sphera-battery-watchdog` Claude Code skill.
- `sparx_agency/tools/sphera_gui_automation.py`: drives Sphera's post-restart GUI walkthrough
  (Continue -> Manager -> Standalone -> select Rooster_1 -> Play) via `xdotool`, using
  proportional (fraction-of-window) coordinates so it doesn't depend on Sphera's window
  staying at one exact position/size. Wired into `sphera_battery_watchdog.py` as the new
  default (`--no-gui-reentry` opts back out) so the whole recovery — container restart AND
  getting a flyable drone back — runs unattended, ~38s end-to-end, confirmed live twice in a
  row. Caught and fixed a real race live (window exists before Sphera is actually
  interactive — see LESSONS.md's second 2026-08-17 entry) via a structural post-Play check
  (`R1` exists + reports a real battery reading) rather than trusting the click sequence
  blindly. See `docs/progress/entries/009-sphera-gui-reentry-automation.md`.
- `sphera_battery_watchdog.py --once`: runs the full restart+re-entry cycle immediately
  regardless of current battery, then exits with a real 0/1 exit code (was previously only
  reachable via the continuous polling loop). Powers a new "⚡ Restart Sphera Now" button on
  `mission_control.py`'s Watchdog card (two-click confirm, blocks with a spinner for the
  ~40s the cycle takes). `restart_and_reenter()` now returns a real success `bool` instead
  of `None` to support this.

### Changed
- `sphera_drone.launch` now overrides FALCON's navigation controller to `multi_axis`
  for Rooster (a genuinely holonomic platform), instead of the default `roll_assist`.
- Every Y-axis bound in `maps/sphera_jail.yaml` (`init_y`, `map_min_y`/`map_max_y`,
  `box_min_y`/`box_max_y`, `vbox_min_y`/`vbox_max_y`) and the Y-axis launch args
  documented in the `fly-rooster-sphera` skill (`bev_ymin`/`bev_ymax`/`goal_y`) —
  negated and min/max-swapped to match the corrected localization sign (see Fixed).
- `map_max_z` raised `4.0` → `5.0` and `box_max_z`/`vbox_max_z` raised `1.8` → `4.0`
  in `maps/sphera_jail.yaml`, so RViz shows real room geometry up to the actual
  ceiling instead of truncating at an artificially low height, while keeping a
  real (1.0m) margin between the map and box/vbox bounds (see Fixed for why the
  margin matters).
- `cam_min_depth` raised `0.1` → `0.45` in `sphera_drone.launch` (see Fixed).
- `mapping_sync`'s `freeze_on_turning_mode` set to `false` in `sphera_drone.launch`
  (see Fixed) — turning-smear protection is temporarily disabled pending a real fix
  to the rotation-supervisor bug it exposed.
- Stale saved RViz camera position in `maps/sphera_jail.rviz` (`Focal Point Y: 14.66`)
  updated to `Y: -14.66` to match the corrected localization sign.

### Fixed
- `waypoint_follower_node.py`: `_publish_twist_multi` was defined twice in the same
  class; the second (older) definition silently shadowed the first and didn't accept
  the `vz` keyword the caller passed, so every navigation tick threw a `TypeError`
  before ever calling `.publish()`. FALCON's internal state showed `nav=RUN` the whole
  time, but `/cmd_vel` was never actually emitted — the drone never moved toward a
  clicked BEV goal regardless of controller/topic configuration.
- `rooster_ground_truth_localization.py`: `position.y` was passed straight through
  from Sphera/Unreal telemetry while yaw was already negated for the left-handed →
  right-handed conversion, leaving position and rotation handedness inconsistent.
  Completed the conversion by negating `position.y` too (see LESSONS.md for how this
  was verified).
- Click-to-fly deadlock for Rooster: the default `roll_assist` controller's demo-mode
  confirmation handshake never resolves because nothing publishes to `/xtend/demo_mode`
  for this platform — fixed via the `multi_axis` controller switch above (Rooster is
  holonomic and doesn't need the turn-then-forward handshake `roll_assist` requires).
- Noisy/speckled voxel map: the bottom ~25% of every RGB frame showed a near-constant
  `~0.17-0.35m` depth reading (the camera rig/mount itself, visible in its own FOV,
  not real environment) that was below `cam_min_depth` and so got fused as a permanent
  phantom wall directly ahead of the drone on every frame — very likely the real cause
  of "boxed in, no A* route" seen the same day. Fixed by raising `cam_min_depth`
  (see Changed).
- `exploration_node` crashing (`voxel_mapping::ESDF::getDistance` → glog FATAL
  "Address out of range") introduced by setting `vbox_max_z` exactly equal to
  `map_max_z` — ESDF's neighbor-cell queries need a real margin near a boundary, not
  just `map ⊇ vbox` ordering. Fixed by raising `map_max_z` instead (see Changed).
- RViz appearing completely empty (no error, just nothing rendered): a stale saved
  camera focal point in `sphera_jail.rviz` from before the Y-axis fix pointed the
  camera at the mirror-image empty location where the room used to be under the old
  sign convention (see Changed).

### Known issues (not yet fixed)
- `waypoint_follower_node.py`'s rotation supervisor gets stuck permanently requesting
  `"turning"` mode while its navigation loop targets the default startup goal with no
  real flight dynamics ever confirming a turn is complete — this permanently froze
  `mapping_sync`'s rotation-freeze mechanism once `rooster_demo_mode_manager.py` made
  it actually engage. Worked around by disabling `freeze_on_turning_mode` for now
  (see Changed); the supervisor bug itself is unresolved.
- Possible left/right (lateral) mirroring in the built map: reported once during a
  BEV-driven flight (drone next to the real left wall in Sphera; map showed it next
  to the right wall). Forward/back and altitude are both confirmed correct via
  quantitative ground-truth tests, so this is a different bug from the Y-axis fix
  above — most likely in how the camera's local lateral axis projects into world-frame
  points, not the drone's own tracked pose. Not yet conclusively confirmed or
  root-caused; a physical landmark was added to the test hallway to check this
  properly next session.

<!--
Example of a real entry once you have one:

## [Unreleased]
### Fixed
- Hover z-axis drift at hover_z=560 traced to accumulated integral windup in the altitude
  controller, not the sim's ranger noise as first suspected. See LESSONS.md.
-->
