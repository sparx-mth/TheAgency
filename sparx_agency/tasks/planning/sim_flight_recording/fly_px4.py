"""Fly a PEGASUS Iris under a real PX4 autopilot and record the flight.

This is the PX4-in-the-loop path: PX4 SITL does the actual flying (attitude,
position control, arming logic, landing) and this script only streams offboard
position setpoints to it, exactly as a companion computer would on real
hardware. Contrast :mod:`fly_direct`, which bypasses the autopilot and applies
forces itself.

What makes it work, and what used to stop it: Isaac Sim 6.0.1 silently stops
dispatching physics callbacks a couple of steps after ``world.reset()``, and
Pegasus drives *everything* off those callbacks -- including the one that sends
``HIL_SENSOR`` to PX4. PX4 is built with ``ENABLE_LOCKSTEP_SCHEDULER``, so with
no sensor data its clock never advances and it never even boots far enough to
emit a heartbeat. :class:`manual_physics_driver.ManualPhysicsDriver` calls those
callbacks by hand each step, which restores the whole chain.

Everything else here is non-blocking for the same lockstep reason: any blocking
MAVLink wait would stop the loop that produces the data PX4 needs to answer.

To watch it live, connect NVIDIA's Isaac Sim WebRTC Streaming Client to
``<this machine>:49100`` once the log prints ``STREAMING_READY``.

Must run under Isaac Sim's own Python::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/fly_px4.py \\
        --pegasus-root /tmp/dev/PegasusSimulator/extensions/pegasus.simulator \\
        --px4-dir /tmp/dev/PX4-Autopilot \\
        --scene office --out-dir /tmp/dev/recordings/office_px4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sparx_agency.tasks.planning.sim_flight_recording import flight_session, px4_launch
from sparx_agency.tasks.planning.sim_flight_recording.manual_physics_driver import ManualPhysicsDriver
from sparx_agency.tasks.planning.sim_flight_recording.px4_offboard import PX4Offboard
from sparx_agency.tasks.planning.sim_flight_recording.px4_vision_pose import (
    VISION_EKF_PARAMS, VisionPoseSender,
)
from sparx_agency.tasks.planning.sim_flight_recording.waypoint_mission import WaypointMission

# Physics runs every step; rendering every step would cap the whole simulation
# at render speed, which starves PX4's lockstep clock (it logs
# "simulator_mavlink: poll timeout"). Nothing depends on rendering every step
# any more -- the manual driver, not a render-driven callback, is what advances
# the vehicle. 10 renders per 100 physics steps gives exactly the 10 Hz the
# recording wants.
RENDER_EVERY_N_STEPS = 10
BOOT_TIMEOUT_S = 120.0              # simulated seconds to wait for PX4's first heartbeat
SETPOINT_PRIME_S = 2.0              # offboard needs a setpoint stream flowing before it engages
ARM_RETRY_S = 1.0                   # PX4 rejects arming until its own pre-flight checks pass
ARM_TIMEOUT_S = 60.0                # simulated seconds of rejected arming before giving up
POST_LAND_S = 8.0
STATUS_EVERY_S = 2.0                # how often to print a state line
# A multirotor held past this tilt is not flying, it is lying against something.
# Bailing out beats silently recording minutes of a nose-down drone scraping a wall.
CRASH_TILT_DEG = 60.0
CRASH_HOLD_S = 3.0


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pegasus-root", type=Path, required=True)
    ap.add_argument("--px4-dir", type=Path, required=True)
    ap.add_argument("--scene", default="simple_room")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--altitude", type=float, default=1.5, help="cruise altitude, metres")
    ap.add_argument("--legs", type=int, default=None,
                    help="number of route legs to fly (default: the whole scene route)")
    ap.add_argument("--vision", action="store_true",
                    help="switch PX4's estimator from simulated GPS to a ground-truth "
                         "vision pose. INCOMPLETE -- PX4 still reports 'ekf2 missing data' "
                         "and refuses to arm; see px4_vision_pose.py")
    ap.add_argument("--no-stream", action="store_true", help="skip WebRTC livestream")
    ap.add_argument("--video-out", type=Path, default=None, help="also write an MP4")
    ap.add_argument("--video-source", choices=["chase", "onboard"], default="chase")
    return ap.parse_args()


def _pace_to_realtime(wall_start: float, sim_time: float) -> None:
    """Throttle the loop to real time so a live viewer can follow the flight.

    With warm GPU caches ``world.step()`` runs faster than the sim time it
    advances, so an unthrottled mission finishes before anyone can connect.
    """
    behind = wall_start + sim_time - time.monotonic()
    if behind > 0:
        time.sleep(behind)


def _attitude_deg(vehicle):
    """The vehicle's current roll, pitch and yaw in degrees, for logging."""
    from scipy.spatial.transform import Rotation

    qx, qy, qz, qw = vehicle.state.attitude
    return Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=True)


def _status_line(vehicle, px4, sim_time: float, target=None) -> str:
    position = vehicle.state.position
    x, y, z = position
    roll, pitch, yaw = _attitude_deg(vehicle)
    dx, dy, _ = px4.frame_offset(position)
    line = (f"t={sim_time:6.1f}s pos=({x:6.2f},{y:6.2f},{z:5.2f}) "
            f"rpy=({roll:6.1f},{pitch:6.1f},{yaw:6.1f}) "
            f"armed={px4.armed} frame_off=({dx:5.2f},{dy:5.2f})")
    if target is not None:
        line += f" -> ({target[0]:6.2f},{target[1]:6.2f},{target[2]:5.2f})"
    return line


def _advance(world, driver, px4, dt: float, step: int, vision=None, vehicle=None,
             sim_time: float = 0.0) -> bool:
    """Advance one physics step, driving the vehicle and PX4 by hand.

    The vision pose goes out *after* ``driver.step``, which is what runs the
    backend's own update and stamps the sensor clock the pose must agree with.

    Returns:
        True if this step also rendered (and so produced a fresh camera frame).
    """
    render = step % RENDER_EVERY_N_STEPS == 0
    world.step(render=render)
    driver.step(dt)
    if vision is not None:
        vision.send(vehicle, sim_time)
    px4.poll()
    return render


def _wait_for_heartbeat(world, driver, px4, dt: float) -> float:
    """Step the sim, feeding PX4 sensor data, until it answers with a heartbeat.

    Returns:
        The simulated time at which the heartbeat arrived.

    Raises:
        RuntimeError: If PX4 never responded within :data:`BOOT_TIMEOUT_S`.
    """
    sim_time = 0.0
    step = 0
    while sim_time < BOOT_TIMEOUT_S:
        _advance(world, driver, px4, dt, step)
        step += 1
        sim_time += dt
        if px4.heartbeat_seen:
            print(f"PX4 heartbeat after {sim_time:.1f} s of simulated time", flush=True)
            return sim_time
    raise RuntimeError(
        f"no PX4 heartbeat after {BOOT_TIMEOUT_S:.0f} simulated seconds -- PX4 SITL did not boot"
    )


def _flight_plan(scene: str, spawn_xyz, altitude: float, legs):
    """Build the waypoint list for ``scene``, clipped to ``legs`` legs."""
    from sparx_agency.robots.PEGASUS.adapters.scene import scene_route

    route = scene_route(scene, altitude)
    if legs is not None:
        route = route[:max(legs, 1)]
    # Always climb straight up from the spawn point first, and land back on it.
    x, y, _ = spawn_xyz
    return [(x, y, altitude, 0.0)] + route + [(x, y, altitude, 0.0)]


def main() -> None:
    args = _parse_args()

    from sparx_agency.robots.PEGASUS.adapters.scene import scene_spawn

    simulation_app = flight_session.boot_isaac(stream=not args.no_stream)

    spawn_xyz = scene_spawn(args.scene)
    px4_process = px4_launch.launch_px4(args.px4_dir)
    try:
        world, adapter = flight_session.build_scene(
            simulation_app, args.scene, args.pegasus_root, spawn_xyz,
            use_px4=True, px4_dir=args.px4_dir,
        )
        vehicle = adapter.vehicle

        chase_camera = None
        want_chase_video = args.video_out is not None and args.video_source == "chase"
        if want_chase_video or not args.no_stream:
            from sparx_agency.tasks.planning.sim_flight_recording.chase_camera import make_chase_camera
            chase_camera = make_chase_camera()

        world.reset()
        driver = ManualPhysicsDriver(vehicle)
        driver.ensure_started()

        if chase_camera is not None and not args.no_stream:
            # Point the streamed viewport -- what the WebRTC livestream
            # actually shows -- at the chase camera. Otherwise a live viewer
            # sees Kit's unaimed default camera, not the drone. Must happen
            # *after* world.reset(), which otherwise resets the viewport back
            # to its default camera. get_active_viewport() can return nothing
            # useful in a --no-window headless app; look the primary viewport
            # up by its known window name instead.
            from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name

            viewport = get_viewport_from_window_name("Viewport") or get_active_viewport()
            if viewport is not None:
                viewport.camera_path = "/World/ChaseCamera"
                print(f"live viewport camera set to /World/ChaseCamera on {viewport}", flush=True)
            else:
                print("WARNING: no viewport found to point at the chase camera "
                      "-- live stream will show the default unaimed camera", flush=True)

        px4 = PX4Offboard()
        dt = flight_session.PHYSICS_DT
        sim_time = _wait_for_heartbeat(world, driver, px4, dt)

        px4.set_indoor_limits()
        vision = None
        if args.vision:
            px4.set_params(VISION_EKF_PARAMS)
            vision = VisionPoseSender(px4, vehicle._backends[0])
            print("feeding PX4 a vision pose (GPS fusion disabled)", flush=True)

        recorder = flight_session.FlightRecorder(
            adapter, args.out_dir, rate_hz=args.rate_hz,
            video_out=args.video_out, video_source=args.video_source,
            chase_camera=chase_camera,
        )
        try:
                _fly(world, driver, px4, adapter, recorder, args, spawn_xyz, sim_time,
                 chase_camera, vision)
        finally:
            # Write whatever was captured even if the flight aborted -- a partial
            # recording of a flight that hit something is still worth inspecting.
            recorder.finish()
            px4.close()
    finally:
        px4_launch.terminate_px4(px4_process)
        simulation_app.close()


def _fly(world, driver, px4, adapter, recorder, args, spawn_xyz, sim_time, chase_camera,
         vision) -> None:
    """Prime offboard, arm, fly the scene route, land -- stepping the sim throughout."""
    from sparx_agency.tasks.planning.sim_flight_recording.chase_camera import aim_chase_camera

    vehicle = adapter.vehicle
    dt = flight_session.PHYSICS_DT
    plan = _flight_plan(args.scene, spawn_xyz, args.altitude, args.legs)
    mission = WaypointMission(plan)
    print(f"flight plan ({len(plan)} waypoints): "
          + " -> ".join(f"({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})" for w in plan), flush=True)

    warmup_left = flight_session.CAMERA_WARMUP_RENDER_TICKS
    arming_started = None
    last_arm_request = 0.0
    last_status = 0.0
    tilted_since = None
    landing_started = None
    step = 0
    wall_start = time.monotonic()
    stream_announced = args.no_stream

    while True:
        rendered = _advance(world, driver, px4, dt, step, vision, vehicle, sim_time)
        step += 1
        sim_time += dt

        position = vehicle.state.position
        if chase_camera is not None and rendered:
            aim_chase_camera(chase_camera, position)

        # Stream a setpoint every step: PX4 refuses to enter offboard mode
        # without one already flowing, and drops out of it if the stream stops.
        target = mission.current() or (spawn_xyz[0], spawn_xyz[1], args.altitude, 0.0)
        if landing_started is None:
            px4.send_setpoint_world(*target, vehicle_enu=position)

        if warmup_left > 0:
            if rendered:
                warmup_left -= 1
                if warmup_left == 0 and not stream_announced:
                    print("STREAMING_READY -- connect the Isaac Sim WebRTC Streaming "
                          "Client to this machine, port 49100", flush=True)
                    stream_announced = True
            continue

        _pace_to_realtime(wall_start, sim_time)

        if sim_time - last_status >= STATUS_EVERY_S:
            print(_status_line(vehicle, px4, sim_time, target), flush=True)
            last_status = sim_time

        roll, pitch, _ = _attitude_deg(vehicle)
        if max(abs(roll), abs(pitch)) > CRASH_TILT_DEG:
            tilted_since = tilted_since if tilted_since is not None else sim_time
            if sim_time - tilted_since >= CRASH_HOLD_S:
                raise RuntimeError(
                    f"vehicle has been tilted past {CRASH_TILT_DEG:.0f} deg for "
                    f"{CRASH_HOLD_S:.0f} s at ({position[0]:.1f}, {position[1]:.1f}, "
                    f"{position[2]:.1f}) -- it has hit something. Check the scene's "
                    f"route in robots/PEGASUS/adapters/scene.py."
                )
        else:
            tilted_since = None

        if not px4.armed and landing_started is None:
            if arming_started is None:
                arming_started = sim_time + SETPOINT_PRIME_S
            elif sim_time >= arming_started and sim_time - last_arm_request >= ARM_RETRY_S:
                px4.set_offboard_mode()
                px4.arm()
                last_arm_request = sim_time
                if sim_time - arming_started > ARM_TIMEOUT_S:
                    raise RuntimeError(
                        f"PX4 refused to arm for {ARM_TIMEOUT_S:.0f} simulated seconds -- "
                        f"check the PX4 console output above for the pre-flight check that failed"
                    )
        elif px4.armed and landing_started is None:
            if mission.update(position, sim_time):
                print(f"reached waypoint {mission.index}/{len(plan)} at "
                      f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})", flush=True)
            if mission.finished:
                px4.land()
                landing_started = sim_time
                print("route complete -- landing", flush=True)

        if landing_started is not None and sim_time - landing_started >= POST_LAND_S:
            break

        if rendered:
            recorder.capture()


if __name__ == "__main__":
    main()
