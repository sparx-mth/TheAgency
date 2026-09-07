# ROBOTICAN Orin-NX acceptance tests

Five tests to hand to Robotican before accepting the new Orin-NX drone,
plus a sixth (`orin_nx_mavlink_direct_test.py`) that talks straight to the
FCU over MAVLink instead of through ROS2 -- see item 6 below.
**Every script here is deliberately standalone** -- no `sparx_agency`
import, nothing that requires our repo or environment beyond whatever ROS2
packages/interfaces Robotican already has of their own. We hand these over
as individual files, not the whole codebase, so that's a hard constraint,
not a style preference -- check `grep -n "^from sparx_agency\|^import sparx_agency"`
on any file here before adding to it.

For day-to-day debugging tools (not acceptance-specific), see `../debug/`.

## The five tests (2026-09)

Robotican is delivering a new drone with a single Jetson Orin NX mounted
onboard, replacing the previous split (backend on a host PC, VLM/detector
models on a separate Jetson AGX Orin).

1. **Get stream** -- `orin_nx_camera_capture_test.py`. Whether the camera
   ends up direct-USB to the Orin or still arrives via Rooster's own video
   relay isn't fully settled -- the script supports both (`--mode v4l2` /
   `--mode gst`), so it answers the question rather than assuming one.
2. **Get IMU** -- `orin_nx_mavlink_direct_test.py` (item 6), not the ROS2
   layer. There is no IMU subscriber in `orin_nx_flight_interface_test.py`
   -- its original `/{id}/imu/data` (`sensor_msgs/Imu`) default turned out
   to be unconfirmed against anything findable: traced it back to a
   Sphera-simulator example script, then checked the whole local Sphera
   workspace directly and found no message anywhere with gyro/accel/
   angular_velocity fields, nor anything publishing `sensor_msgs/Imu`. Not
   "unproven on hardware, fine in sim" -- unproven full stop. None of
   Robotican's actual vendored ROS2 interfaces carry IMU fields either. So
   IMU is item 6's job: real IMU almost certainly only exists in the raw
   MAVLink stream, read directly with no topic name to guess. Checked
   independently of the camera either way -- no cross-stream sync
   validation (that's the mapping pipeline's job on our side, not this
   acceptance pass's).
3. **Arm/disarm + service/topic check** -- `orin_nx_flight_interface_test.py`.
   Confirms `cmd_nav` still works post-architecture-change, and that both
   `/{id}/state` (RoosterState) and `/{id}/rooster_status` (battery_pct,
   battery_voltage, armed, airborne, busy_action) keep delivering. Bench
   mode (props off) by default -- pass `--real-flight` for an actual
   takeoff / hover in [1.0, 1.5]m / 360 turn / land sequence. That mode
   requires props installed, a clear area, and a safety pilot present with
   a manual override ready; the script requires a typed `FLY` confirmation
   (or `--yes` for non-interactive use) before it will arm, and a
   continuous ceiling guard (`--ceiling-abort-m`, default 2.5m) commands
   an immediate land if altitude ever exceeds it. See the script's own
   docstring for why hover altitude is verified but not corrected live
   (it's a `RoosterCommandUnitNode` launch parameter, `target_ranger_m`,
   not something `cmd_nav` can set), and why the 360 turn is closed-loop
   via `UAVState.azimuth` when available, with an explicitly-flagged
   open-loop fallback if that field never arrives. Also reports `fcu_mode`
   and subscribes to `StatusText` (controller-reported errors/warnings,
   MAVLink-STATUSTEXT-style) -- informational, since it's event-driven and
   silence during a clean run is expected, not a failure. Its topic name is
   an inferred guess (`/{id}/status_text`), never confirmed against the
   real system -- see the script's docstring. `--check-link-dropout` is an
   interactive add-on: physically disconnect/reconnect the FCU link when
   prompted, and it confirms `is_fcu_connected` reports both the fault and
   the recovery -- the closest thing here to an actual error-condition test.
4. **Run DA3, including building the engine file** -- `da3_acceptance_summary.py`.
   Builds the engine from ONNX via `trtexec` on-device if it doesn't exist
   yet (not a pre-built engine handed over -- TensorRT engines are
   hardware/version-locked, so building it here is part of what's being
   tested), then runs inference and reports latency + sanity of the depth
   values. No camera calibration needed -- this only checks the engine
   itself, not a projected point cloud.
5. **Run vLLM with a model** -- `vllm_orin_nx_acceptance_test.sh`. Runs a
   vLLM container with a real VLM (Qwen3-VL-4B AWQ), sends one real
   inference call, reports memory used.
6. **Direct MAVLink check** -- `orin_nx_mavlink_direct_test.py`. Complements
   #2/#3: talks straight to the FCU over the direct USB/MAVLink link via
   `pymavlink`, with no ROS2 and no dependency on Robotican's own bridge
   software (`fcu_driver`/`rooster_manager`/`rooster_handler`) being wired
   correctly. Checks HEARTBEAT, IMU message rate, and STATUSTEXT (the real
   standard MAVLink error/status message, 8-level `MAV_SEVERITY` -- unlike
   #3's `StatusText`, nothing here needs Robotican to confirm a topic name,
   it's part of the protocol). If this passes but #3's ROS2-level checks
   fail, that isolates the problem to their bridge software, not the link.

Also worth knowing for #3: the Orin NX module has no onboard eMMC
(confirmed against NVIDIA's spec) -- everything boots and lives on an
external NVMe, so a correctly-provisioned unit needs one.

| Script | Checks | Needs ROS2? |
|---|---|---|
| `orin_nx_camera_capture_test.py` | Camera opens (USB or relay), sustains target FPS/resolution, saves sample frames | No |
| `orin_nx_flight_interface_test.py` | Bench: `cmd_nav`/`state`/`rooster_status` topics work the same (arm/disarm, telemetry, battery), image topic rate, `StatusText`/`fcu_mode`. `--real-flight`: actual takeoff/hover/360-turn/land | Yes |
| `da3_acceptance_summary.py` | DA3 engine builds from ONNX and infers on this device, at what latency, with sane (finite) depth values | No |
| `vllm_orin_nx_acceptance_test.sh` | vLLM + Qwen3-VL-4B AWQ container starts, model loads, one real inference call returns a valid response | No |
| `orin_nx_mavlink_direct_test.py` | HEARTBEAT, IMU rate, STATUSTEXT -- straight over MAVLink, no ROS2, no Robotican bridge software involved | No (needs `pymavlink`) |

**Run order on a freshly-arrived unit:**

```bash
# 1. Camera, before anything else is even wired up
python3 orin_nx_camera_capture_test.py --device /dev/video0 --duration-sec 10
# if that finds nothing, try the relay path instead:
python3 orin_nx_camera_capture_test.py --mode gst --host-ip 127.0.0.1 --port 5001

# 2. DA3 -- builds the engine from ONNX itself if it doesn't exist yet
python3 da3_acceptance_summary.py \
    --rgb-dir /path/to/sample_images \
    --onnx-path /path/to/DA3-METRIC.onnx \
    --engine-path /path/to/DA3-METRIC.engine

# 3a. Flight interface, bench-only, PROPS REMOVED
python3 orin_nx_flight_interface_test.py --rooster-id R1 --observe-sec 8

# 3b. Real flight (separate step, only once 3a passes and a safety pilot is present):
# RoosterCommandUnitNode must already be launched with target_ranger_m in [1.0,1.5]
python3 orin_nx_flight_interface_test.py --rooster-id R1 --real-flight \
    --hover-min-m 1.0 --hover-max-m 1.5 --turn-direction left

# 4. vLLM -- needs the model weights already in place
./vllm_orin_nx_acceptance_test.sh --model-dir ~/my_models/qwen3-vl-4b --image /path/to/test.jpg

# 5. Direct MAVLink, no ROS2 -- confirm the connection string with Robotican first
python3 orin_nx_mavlink_direct_test.py --connection /dev/ttyACM0
```

All scripts print a final `PASS`/`FAIL` line and exit non-zero on failure.
None of them are a contractual gate on their own -- the first real run's
job is to produce numbers to act on, not to enforce thresholds nobody has
agreed to yet.
