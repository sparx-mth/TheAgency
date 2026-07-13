# Next session: connect FALCON to ROBOTICAN

**Also see `MISSION_CONTROL_INTEGRATION_PLAN.md`** — a separate, self-contained
task (adding the sensing/dome pipeline to `sparx_agency/tools/mission_control.py`'s
Streamlit launcher) with every path/command/gotcha needed, written for
picking up from a different machine.

## Status as of 2026-07-13 (superseding everything below this point — kept for history)

The Sphera CycloneDDS networking issue is resolved (pragmatically — see
`DOME_CAPTURE_README.md`'s "CycloneDDS network interface" section and
`project_sphera_cyclonedds_interface.md` in auto-memory, local to the PC
this was debugged on). The full sensing pipeline was validated live against
real Sphera, twice, end-to-end:

- Gateway (`rooster_command_unit.py`): arm/disarm/takeoff/hover/rotate/land
  all confirmed via real telemetry. Tuned climb parameters for Sphera:
  `climb_z=700, hover_z=550, climb_duration_sec=5.0` (code defaults are too
  weak/short to actually leave the ground in this sim). Left/right axis
  sign was inverted — fixed.
- `rooster_frame_dir_publisher.py`: real video frames flowing to disk.
- `rooster_depth_processor.py` (DA3-TRT): real-time (~7-9ms/frame), sane
  metric depth (0.2-8m range typical). Needed `pip install tensorrt==10.16.1.11`
  in the main venv (version-matched to `da3_venv`'s engine — the generic
  `pip install tensorrt` resolves to an incompatible newer major version).
  Cage-mask inpainting (`bar_inpainter.py`) is currently **disabled**
  (all-zero mask) — the committed mask was built at the wrong resolution,
  and a rebuild attempt from real diverse frames showed the fixed-brightness-
  threshold approach is too fragile (max persistence landed just under the
  80% cutoff even for the real cage). Needs a better algorithm, not just more
  calibration frames — separate task, not blocking.
- `rooster_ground_truth_localization.py` (**new**): republishes Sphera's own
  ground-truth pose (`sphera_common_interfaces/SpheraPawnState`, only
  importable inside the `it` container) as `/R1/localization` — no AprilTag
  needed for sim testing.
- `rooster_dome_main.py`: two full autonomous 360° dome-sweep missions
  completed end-to-end (video_on -> arm -> takeoff -> rotate -> capture ->
  land -> disarm), each producing 49 complete, correctly-paired
  `.jpg`/`.json`/`.npy` triples with real RGB, real DA3 depth, and real
  ground-truth pose. Also fixed a shutdown-time race (`spin_thread` never
  joined before `node.destroy_node()`, causing an intermittent C++ abort at
  teardown) — verified clean on both live runs after the fix.
- `rooster_offline_frame_dir_publisher.py` (**new**): replays a recorded
  session back onto the live pipeline's exact topics
  (`/R1/rgb_frame_path`/`/R1/depth_frame_path`/`/R1/localization`), so
  downstream consumers can be tested without Sphera at all. Verified against
  a real recorded session.
- Two full recorded sessions available for offline work:
  `~/rooster_dome_capture/latest` and `~/rooster_dome_capture_2/latest`
  (49 triples each, all with valid non-zero ground-truth pose).
- `core/mapping/depth/__init__.py` had the same eager-import problem as the
  earlier `robots/ROBOTICAN/adapters/__init__.py` fix (pulling in a broken
  torch/NCCL install just to import `DA3TensorRTModel`, which doesn't need
  torch at all) — fixed the same way.

## Goal for the FALCON session

Get FALCON (`sparx_agency/tasks/planning/falcon/`, ROS1 Noetic, dockerized)
running against ROBOTICAN — the "second scenario" from the original
XTEND-parity investigation: given a point/object, navigate to it avoiding
obstacles. Not yet started. What's known from earlier research (see the
session that produced this repo's ROBOTICAN work, if resumable) that's
still relevant:

- FALCON's ROS1<->ROS2 bridge (`tasks/planning/falcon/bridge/`) bridges RGB/
  depth as file-path strings (`std_msgs/String`), not raw `sensor_msgs/Image`
  — exactly the convention `rooster_frame_dir_publisher.py`/
  `rooster_depth_processor.py`/`rooster_offline_frame_dir_publisher.py`
  already produce. The bridge's topic list (`bridge.yaml`) and
  `run_falcon.sh`'s frame-dir mounts (`/tmp/xtend_frames`,
  `/tmp/xtend_depth`) are currently XTEND-specific — will need a ROBOTICAN
  variant (or just repointed at `/tmp/rooster_frames`/`/tmp/rooster_depth`,
  which the ROBOTICAN nodes already default to).
- `combination_planner_node.py`'s default camera intrinsics/resolution
  (504x294) won't match ROBOTICAN's native 540x360 — needs the ROBOTICAN
  calibration (`config/camera_rooster_calib_540_360.yaml`) and matching
  `--cam-width`/`--cam-height`/intrinsics launch args.
- Can develop/test this entirely offline today using
  `rooster_offline_frame_dir_publisher.py` against the two recorded
  sessions above — no need to have Sphera running to get the FALCON side
  wired up and talking to the bridge; only actually driving the (simulated)
  drone requires live Sphera.

---

## (Historical, mostly superseded) Original plan from 2026-07-08

Live testing against Sphera was blocked at the time: `R1` (spawned
internally by the closed-source Sphera app, no config controllable from
outside) always landed on the wrong network interface. Now resolved — see
above. The mock-frame/mock-gateway approach originally planned here was
superseded once the real pipeline was confirmed working live; kept only in
case a lighter-weight mock is ever needed again for a specific reason (e.g.
testing without any Docker/Sphera dependency at all).
