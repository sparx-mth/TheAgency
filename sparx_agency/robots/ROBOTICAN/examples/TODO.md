# TODO — rotate_360_example.py (R3 flight tests)

State (2026-10-13): `rotate_360_example.py` (arm -> takeoff -> Position mode -> 3-second
settle -> 360 turn -> hover; camera capture runs throughout) has been flight-tested twice on
R3. Companion vision-pipeline tooling lives in `NanoLLM_VILA_and_OWL/orin_nx/
run_rotate_scan_test.sh` (pre-flight container health check, post-flight
`comm_manager_vllm.py` + display run on the captured frames).

1. **Does `MAV_CMD_CONDITION_YAW` actually produce rotation in Position mode here?** Still
   unresolved. First test: ~90 deg achieved in a window a clean 360 deg at 30 deg/s should
   have taken (estimated, not measured). Second test (after switching to real azimuth-feedback
   completion detection): accumulated azimuth oscillated near zero for ~25 clean seconds
   while the command was being correctly re-issued - no measurable rotation at all. That
   second test was also contaminated partway through: DDS/multicast dropouts (see item 2) hit
   at the same time the operator intervened and landed early (drone had drifted near a wall),
   so the apparent "jump" in accumulated azimuth right after is likely dropout/landing noise,
   not real rotation. Also treat `UAVState.azimuth` itself with some skepticism - it's
   probably magnetometer-derived, and this is an indoor environment (steel structure, the
   drone's own metal cage) where compass interference is plausible. Cross-check against video
   next time, not just telemetry.
   - A 3-second settle period (`YAW_SETTLE_TIME_S`) was added between Position-mode
     confirmation and the yaw command, since the drone was seen drifting against a wall for
     the whole second flight (optical-flow-based position hold settling in?) - untested
     whether this actually helps.
   - A conservative pitch-down (`CAPTURE_PITCH_DOWN`) was added during the capture window so
     the body-mounted (no gimbal) camera looks more toward the floor/room instead of straight
     ahead at wall height - sign/magnitude unconfirmed, watch closely on the next flight.
2. **DDS/multicast dropouts seen mid-flight**: `ddsi_udp_conn_write to udp/239.255.0.1:8151
   failed with retcode -1`, in bursts, appearing in both `rotate_360_example.py`'s and
   `land_and_disarm_example.py`'s logs at the same wall-clock moment during the second test -
   a real, shared link/network problem, not a bug in either script. Takeoff was also much
   slower and more erratic in that test (`flight_mode` flickering 7/8 for ~17s instead of a
   clean ~6s climb) right alongside the first dropout burst. **Not yet root-caused** - check
   WiFi/RF link quality between the drone and the RPi/ground network; a flaky link would
   corrupt both command delivery and telemetry in ways that look exactly like a flight-control
   bug. A safety check (`comms_stale()`/`STATE_STALE_TIMEOUT_S`) now stops automated commands
   and holds if `RoosterState`/`UAVState` haven't updated in 2s, rather than acting on data
   that's gone silent - this makes dropouts safer but doesn't fix the underlying link issue.
3. **Raw MAVLink/IMU telemetry (`telemetry_example.py`-style) — flagged, not implemented.**
   The ask was to connect flight logic more directly to IMU/attitude data, inspired by
   `telemetry_example.py`'s direct pymavlink connection to `/dev/FCU`. **Caution before doing
   this literally**: `telemetry_example.py` opens a *separate* raw MAVLink connection to the
   same serial device (`/dev/FCU`) that `fcu_driver` (the ROS2 node our script already talks
   to) already owns and is actively using. Two processes both reading/writing and sending
   heartbeats on the same physical link risks destabilizing the *existing* connection -
   plausibly making dropouts (item 2) worse, not better. `UAVState` already exposes `azimuth`,
   `roll`, `pitch` via the proper ROS2 abstraction (no serial contention) - what was
   implemented instead, serving the same "trust the sensors, not a blind timer/guess" goal
   without that risk, is the comms-staleness check in item 2. If a literal second MAVLink
   connection is still wanted (e.g. for faster-rate IMU data `UAVState` doesn't expose), confirm
   first whether the serial port allows a second concurrent client safely on this hardware.
4. **R2/R3 camera** vs this flow: NanoOWL's baked-in TensorRT engine (R1) and now this flight
   capture path both touch the same kind of "built/tuned on one board, assumed to work on
   others" risk NanoLLM_VILA_and_OWL's TODO already tracks for DA3 - if this script gets
   copied to R1/R2, re-verify `/dev/HD_CAMERA` and the `iai` container's frame mount
   (`docker inspect iai --format '{{range .Mounts}}...'`) per board rather than assuming.
5. Package as a proper `ros2 run examples rotate_360_example` entry (add an `install()` line
   to `CMakeLists.txt` matching the existing `RENAME` pattern, then `colcon build`) once the
   above is validated - deliberately held off per the operator's request to test with plain
   `python3` first.
