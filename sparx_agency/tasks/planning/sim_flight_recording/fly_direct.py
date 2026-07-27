"""Fly a PEGASUS Iris with direct Python force control -- no autopilot at all.

A fallback and a debugging tool, not the main path: it bypasses PX4 entirely
and applies forces to the airframe from a world-frame PD controller. Useful
when you want a guaranteed-moving drone without an autopilot in the loop (for
example to check a scene loads, the camera streams, and the recorder writes),
or when PX4 refuses to arm and you need to isolate whether the problem is the
autopilot or the simulator.

For real flights and for collecting data use :mod:`collect`, which puts an
actual PX4 autopilot in the loop and plans a wall-avoiding route across a
surveyed map.

Because it commands forces directly, this script does no planning at all: it
flies a fixed climb / forward-and-back cruise / descend pattern around the
spawn point, with no obstacle checking whatsoever, so in a furnished scene it
can and does fly into things.

Must run under Isaac Sim's own Python::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/fly_direct.py \\
        --pegasus-root /tmp/dev/PegasusSimulator/extensions/pegasus.simulator \\
        --scene simple_room --out-dir /tmp/dev/recordings/simple_room_direct \\
        --altitude 2.0 --cruise-s 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sparx_agency.tasks.planning.sim_flight_recording import flight_session
from sparx_agency.tasks.planning.sim_flight_recording.sim_loop import SimLoop

G = 9.81
CLIMB_S = 5.0
DESCEND_S = 5.0
CRUISE_AMPLITUDE_M = 3.0

# Altitude (Z) and horizontal (X, Y) PD gains, plus an attitude-levelling PD that
# keeps the body roughly flat while the position PD pushes it around. Tuned
# empirically for the Iris's ~1.6 kg mass on simple_room; expect to retune elsewhere.
KP_Z, KD_Z = 40.0, 14.0
KP_XY, KD_XY = 6.0, 5.0
KP_ATT, KD_ATT = 4.0, 0.6


def _total_mass(vehicle) -> float:
    parts = ["/body"] + [f"/rotor{i}" for i in range(4)]
    return float(sum(vehicle.get_rigid_prim(p).get_masses()[0] for p in parts))


def _body_rotation(vehicle):
    from scipy.spatial.transform import Rotation

    qx, qy, qz, qw = vehicle.state.attitude
    return Rotation.from_quat([qx, qy, qz, qw])


def _target_position(t: float, spawn_xy, altitude: float, cruise_s: float):
    """Scripted mission: climb, cruise forward and back, descend.

    Returns:
        World-frame ``(x, y, z)`` to hold at time ``t``.
    """
    x0, y0 = spawn_xy
    if t < CLIMB_S:
        return x0, y0, altitude * (t / CLIMB_S)

    t_cruise = t - CLIMB_S
    if t_cruise < cruise_s:
        x = x0 + CRUISE_AMPLITUDE_M * np.sin(2 * np.pi * t_cruise / cruise_s)
        return x, y0, altitude

    t_land = t_cruise - cruise_s
    if t_land < DESCEND_S:
        return x0, y0, max(altitude * (1.0 - t_land / DESCEND_S), 0.05)
    return x0, y0, 0.05


def _apply_control(vehicle, mass: float, target) -> None:
    """One step of world-frame position PD plus attitude levelling."""
    tx, ty, tz = target
    position = vehicle.state.position
    velocity = vehicle.state.linear_velocity

    force_world = np.array([
        KP_XY * (tx - position[0]) - KD_XY * velocity[0],
        KP_XY * (ty - position[1]) - KD_XY * velocity[1],
        mass * G + KP_Z * (tz - position[2]) - KD_Z * velocity[2],
    ])
    rotation = _body_rotation(vehicle)
    vehicle.apply_force(list(rotation.inv().apply(force_world)), body_part="/body")

    roll, pitch, _ = rotation.as_euler("xyz")
    angular = vehicle.state.angular_velocity
    vehicle.apply_torque([
        -KP_ATT * roll - KD_ATT * angular[0],
        -KP_ATT * pitch - KD_ATT * angular[1],
        -KD_ATT * angular[2],
    ], body_part="/body")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pegasus-root", type=Path, required=True)
    ap.add_argument("--scene", default="simple_room")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--altitude", type=float, default=2.0, help="cruise altitude, metres")
    ap.add_argument("--cruise-s", type=float, default=15.0, help="cruise duration, seconds")
    ap.add_argument("--no-stream", action="store_true", help="skip WebRTC livestream")
    ap.add_argument("--video-out", type=Path, default=None, help="also write an MP4")
    ap.add_argument("--video-source", choices=["chase", "onboard"], default="chase")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    simulation_app = flight_session.boot_isaac(stream=not args.no_stream)
    try:
        from sparx_agency.robots.PEGASUS.adapters.scene import scene_spawn
        from sparx_agency.tasks.planning.sim_flight_recording.chase_camera import make_chase_camera

        spawn_xyz = scene_spawn(args.scene)
        world, adapter = flight_session.build_scene(
            simulation_app, args.scene, args.pegasus_root, spawn_xyz, use_px4=False,
        )
        vehicle = adapter.vehicle

        chase_camera = None
        want_chase_video = args.video_out is not None and args.video_source == "chase"
        if want_chase_video or not args.no_stream:
            chase_camera = make_chase_camera()
            if not args.no_stream:
                # Point the interactive viewport -- what the WebRTC livestream
                # actually shows -- at the chase camera. Otherwise a live
                # viewer sees Kit's unaimed default camera, not the drone.
                from omni.kit.viewport.utility import get_active_viewport
                get_active_viewport().camera_path = "/World/ChaseCamera"

        world.reset()
        # state_only: this script applies its own forces, so it must not let the
        # driver run Pegasus's rotor/backend update as well.
        loop = SimLoop(world, vehicle, px4=None, rate_hz=args.rate_hz,
                       realtime=not args.no_stream, state_only=True)
        loop.start()
        mass = _total_mass(vehicle)
        print(f"vehicle mass: {mass:.3f} kg", flush=True)

        loop.warmup_camera()
        if not args.no_stream:
            print("STREAMING_READY -- connect the Isaac Sim WebRTC Streaming Client "
                  "to this machine, port 49100", flush=True)

        recorder = flight_session.FlightRecorder(
            adapter, args.out_dir, rate_hz=args.rate_hz, camera_height_m=args.altitude,
            video_out=args.video_out, video_source=args.video_source,
            chase_camera=chase_camera,
        )
        try:
            _fly(loop, vehicle, mass, recorder, args, spawn_xyz)
        finally:
            recorder.finish({"scene": args.scene, "controller": "direct_pd"})
    finally:
        simulation_app.close()


def _fly(loop, vehicle, mass, recorder, args, spawn_xyz) -> None:
    """Step the sim through the scripted mission, controlling and recording."""
    mission_end_s = CLIMB_S + args.cruise_s + DESCEND_S + 3.0
    start = loop.sim_time

    while loop.sim_time - start < mission_end_s:
        _apply_control(vehicle, mass,
                       _target_position(loop.sim_time - start, spawn_xyz[:2],
                                        args.altitude, args.cruise_s))
        if loop.step():
            recorder.capture(stamp_s=loop.sim_time - start)


if __name__ == "__main__":
    main()
