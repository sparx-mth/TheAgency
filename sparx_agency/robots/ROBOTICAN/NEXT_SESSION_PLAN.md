# Next session: mock Sphera, wire up DA3 + the ROBOTICAN pipeline

## Context

Live testing against Sphera is blocked: `R1` (spawned internally by the closed-source
Sphera app, no config we control) always lands on the wrong network interface
(`enp129s0` instead of the required `enx00e04c680602`/192.168.131.x), causing
CycloneDDS discovery to fail/crash. Full root-cause trail, everything tried, and
why each attempt didn't work is in this machine's memory at
`project_sphera_cyclonedds_interface.md` (auto-memory, local to this PC — may not
be visible from a different machine, hence this file). Waiting on the Sphera
vendor's answer on how they expect multi-NIC hosts to be configured.

**Reminder if this is being read on a different PC**: check that the prior
session's commit (`rooster_frame_dir_publisher.py`, `rooster_dome_main.py`,
`DOME_CAPTURE_README.md`, the `run_*.sh` wrapper updates, the
`chessboard_camera_calibration.py` rectification/projection fix,
`camera_rooster_calib_540_360.yaml`) was actually pushed and pulled here — it was
only committed locally as of the prior session (commit `6faf024` on
`robotican_mode_daphna`), not pushed.

## Goal

Get the sparx_agency-owned half of the ROBOTICAN pipeline (frame → DA3 depth →
localization → dome-mission capture logic) fully wired and verified end-to-end
using **mocked** video/telemetry, so progress isn't blocked on the Sphera
networking issue. Swap the mocks back out for the real `rooster_frame_dir_publisher.py`
+ `rooster_command_unit.py` once that's resolved.

## Plan

### 1. Mock the frame source — reuse, don't rewrite

`sparx_agency/robots/XTEND/offline_frame_dir_publisher.py` is already generic
(topic names are CLI args, not hardcoded to `/xtend/...`). Run it pointed at
ROBOTICAN's topics instead of writing a new mock:
```bash
python3 sparx_agency/robots/XTEND/offline_frame_dir_publisher.py \
  --path-topic /R1/rgb_frame_path \
  --depth-path-topic /R1/depth_frame_path \
  --out-dir /tmp/rooster_frames
```
It expects a "take session" directory (see its docstring / `take_xtend_da3_frames.py`
for the expected input layout) — use any existing captured session, or a handful
of real indoor photos arranged into that layout, so DA3 gets meaningful images
rather than synthetic noise.

### 2. Mock the command gateway — new, small

No existing equivalent (XTEND's dome_main talks directly to XTEND's own virtual
controller, not through a separate gateway process). Needs to be a small new
ROS2 node, e.g. `sparx_agency/robots/ROBOTICAN/mock_rooster_command_unit.py`:
- Subscribes `/R1/cmd_nav` (same JSON schema `RoosterCommandUnitNode` uses:
  `arm`/`disarm`/`takeoff`/`land`/`turn_left`/`turn_right`/`stop`/`video_on`/`video_off`).
- Tracks a fake `armed`/`airborne`/`video_on` state machine with plausible delays
  (mirror `RoosterUnit`'s climb/land timing loosely, doesn't need to be exact).
- Publishes `/R1/rooster_status` JSON on a timer, same shape as the real gateway.
- Single purpose, no FCU/video/AprilTag logic — mirrors the SRP reasoning already
  applied to `rooster_frame_dir_publisher.py`.

This unblocks running `rooster_dome_main.py`'s full mission logic (arm → takeoff →
rotate → capture → land → disarm) end-to-end without Sphera/R1 at all.

### 3. Verify DA3 against the new engine with real images

With mocked frames flowing, run `depth_processor_node.py` with the
`DA3METRIC-LARGE_fp16_546x364.engine` and `camera_rooster_calib_540_360.yaml`
(both already in place from the prior session) and sanity-check the depth output
on a real (not synthetic) indoor image — confirm reasonable metric depth values,
not NaN/garbage.

### 4. Verify localization

Run `localization_node.py` (apriltag provider) against a real image containing a
visible AprilTag if one is available in test data — at minimum confirm it runs
without crashing against ROBOTICAN's topics/calib.

### 5. Full mock-stack dome-mission run

Run `rooster_dome_main.py` against the mocked gateway + mocked frame source and
confirm it produces a complete, well-formed session folder — matching
`.jpg`/`.json`/`.npy` triples, working `latest` symlink — exactly like a real
capture would, fully exercising the capture/sidecar/symlink logic in isolation
from Sphera.

### 6. (Stretch) Feed a mock session into the room mapper

`sparx_agency/demos/Demo_No4_XTEND_MapRoom/room_mapper/run_room_mapper.py`
against the mock-produced ROBOTICAN session dir, to confirm the capture format
is fully compatible with the existing offline mapping pipeline (this was the
original point of matching XTEND's output layout).

### 7. Once Sphera vendor responds

Swap the mocks back out for the real `rooster_frame_dir_publisher.py` and
`rooster_command_unit.py`, redo the live end-to-end test per
`DOME_CAPTURE_README.md`'s run order.
