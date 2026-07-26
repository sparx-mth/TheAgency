"""Fly a PEGASUS Iris with direct Python force control (no PX4) -- watchable live
over WebRTC -- while recording the flight.

Root-cause context (see robots/PEGASUS/README.md): Isaac Sim 6.0.1 stops
dispatching ``World.add_physics_callback()``-registered callbacks after ~2
calls following ``world.reset()``. Confirmed directly -- PhysX itself keeps
integrating the rigid body fine (it free-fell from spawn height straight
through the floor over 200 steps in a debug probe), but Pegasus's own
``Vehicle.update_state``/``Multirotor.update`` (registered exactly that way)
never run again to refresh state or (in the PX4 path) sensor data, which is
also why ``record_flight.py``/``fly_and_watch.py``'s recordings were always
flat/stationary and PX4 never got a live sensor stream.

This script sidesteps the callback bug entirely: it drives
``Vehicle.update_state`` and force application **manually**, once per step,
from its own loop -- which we've confirmed keeps running reliably. A simple
world-frame PD controller (altitude hold + a slow forward/back cruise, plus
attitude leveling) takes off, flies, and lands. No PX4, no MAVLink.

To watch it: connect NVIDIA's Isaac Sim WebRTC Streaming Client to
``<this machine>:49100`` once the log prints ``STREAMING_READY`` (see
``robots/PEGASUS/README.md``).

Must run under Isaac Sim's own Python, e.g.::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/fly_direct.py \\
        --pegasus-root /tmp/dev/PegasusSimulator/extensions/pegasus.simulator \\
        --scene simple_room --out-dir /tmp/dev/recordings/simple_room_direct \\
        --altitude 2.0 --cruise-s 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STAGE_SETTLE_STEPS = 20
CAMERA_WARMUP_RENDER_TICKS = 110
CAPTURE_EVERY_N_STEPS = 10
PHYSICS_DT = 1.0 / 100.0  # matches Pegasus's default SimulationCfg physics dt
G = 9.81

# Altitude (Z) PD gains and horizontal (X, Y) PD gains -- tuned empirically for the
# Iris's mass; see README for how these were arrived at.
KP_Z, KD_Z = 40.0, 14.0
KP_XY, KD_XY = 6.0, 5.0
# Attitude-leveling PD (keeps the body roughly level while position PD pushes it around)
KP_ATT, KD_ATT = 4.0, 0.6


def _total_mass(vehicle) -> float:
    parts = ["/body"] + [f"/rotor{i}" for i in range(4)]
    return float(sum(vehicle.get_rigid_prim(p).get_masses()[0] for p in parts))


def _quat_to_rot(vehicle):
    from scipy.spatial.transform import Rotation
    qx, qy, qz, qw = vehicle.state.attitude
    return Rotation.from_quat([qx, qy, qz, qw])


def _make_watch_camera(prim_path: str, resolution=(960, 540)):
    """An external camera watching the drone -- not the drone's own onboard view.

    Its pose is set every frame by :func:`_aim_watch_camera` to chase the vehicle
    at a fixed world-frame offset, rather than sitting at a fixed world position.
    A fixed position risks clipping into unknown room geometry (walls, an
    exterior opening) since the room's actual interior bounds aren't known in
    advance; chasing the vehicle is safe by construction -- the vehicle's own
    position is always valid, collision-free airspace (that's what "it's
    flying" means).
    """
    from isaacsim.sensors.camera.camera import Camera

    camera = Camera(prim_path=prim_path, resolution=resolution)
    camera.initialize()
    return camera


def _aim_watch_camera(camera, vehicle_position, world_offset=(0.0, 1.2, 0.6)):
    """Point ``camera`` at ``vehicle_position`` from a fixed world-frame offset."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    vehicle_position = np.array(vehicle_position, dtype=np.float64)
    position = vehicle_position + np.array(world_offset, dtype=np.float64)
    forward = vehicle_position - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rot_matrix = np.column_stack([forward, left, up])  # local FLU axes in world frame
    qx, qy, qz, qw = Rotation.from_matrix(rot_matrix).as_quat()
    camera.set_world_pose(position=position, orientation=[qw, qx, qy, qz], camera_axes="world")


def _target_position(t: float, spawn_xy, altitude: float, cruise_s: float):
    """Scripted mission: climb, cruise forward/back, descend. Returns (x, y, z)."""
    x0, y0 = spawn_xy
    climb_s = 5.0
    descend_s = 5.0
    if t < climb_s:
        z = altitude * (t / climb_s)
        return x0, y0, z
    t_cruise = t - climb_s
    if t_cruise < cruise_s:
        # slow forward-and-back translation, 3 m amplitude
        x = x0 + 3.0 * np.sin(2 * np.pi * t_cruise / cruise_s)
        return x, y0, altitude
    t_land = t_cruise - cruise_s
    if t_land < descend_s:
        z = altitude * (1.0 - t_land / descend_s)
        return x0, y0, max(z, 0.05)
    return x0, y0, 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pegasus-root", type=Path, required=True)
    ap.add_argument("--scene", default="simple_room")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--altitude", type=float, default=2.0, help="cruise altitude, metres")
    ap.add_argument("--cruise-s", type=float, default=15.0, help="cruise duration, seconds")
    ap.add_argument("--no-stream", action="store_true", help="skip WebRTC livestream")
    ap.add_argument("--video-out", type=Path, default=None,
                     help="also write an MP4 of the flight "
                          "(reliable fallback when WebRTC streaming isn't available/practical)")
    ap.add_argument("--video-source", choices=["chase", "onboard"], default="chase",
                     help="'chase' = external camera following the drone (default); "
                          "'onboard' = the drone's own forward-facing camera (first-person view)")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("pegasus.simulator")
    simulation_app.update()
    if not args.no_stream:
        enable_extension("omni.kit.livestream.app")
        simulation_app.update()

    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

    from sparx_agency.robots.PEGASUS.adapters.scene import load_indoor_scene
    from sparx_agency.robots.PEGASUS.adapters.vehicle import PegasusIrisVehicle
    from sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract import export_flight

    pg = PegasusInterface()
    pg.initialize_world()
    world = pg.world

    load_indoor_scene(args.scene)
    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()

    spawn_xyz = (0.0, 0.0, 0.05)
    adapter = PegasusIrisVehicle(
        pegasus_extension_root=args.pegasus_root,
        init_pos=spawn_xyz,
        use_px4=False,  # this script drives forces directly; no PX4/MAVLink involved
    )
    vehicle = adapter.vehicle

    watch_camera = None
    video_writer = None
    if args.video_out is not None and args.video_source == "chase":
        watch_camera = _make_watch_camera("/World/WatchCamera")

    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()

    world.reset()
    mass = _total_mass(vehicle)
    print(f"vehicle mass: {mass:.3f} kg", flush=True)

    # Camera warmup (render-driven, same as fly_and_watch.py).
    step = 0
    render_ticks = 0
    while render_ticks < CAMERA_WARMUP_RENDER_TICKS:
        world.step(render=True)
        step += 1
        render_ticks += 1

    if not args.no_stream:
        print("STREAMING_READY -- connect the Isaac Sim WebRTC Streaming Client to this machine, port 49100")

    if args.video_out is not None:
        import cv2
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        video_size = (960, 540) if args.video_source == "chase" else (adapter.intrinsics.width, adapter.intrinsics.height)
        video_writer = cv2.VideoWriter(
            str(args.video_out), cv2.VideoWriter_fourcc(*"mp4v"), args.rate_hz, video_size,
        )
        print(f"VIDEO_RECORDING ({args.video_source}) -- writing to {args.video_out}", flush=True)

    frames = []
    t = 0.0
    mission_end_s = 5.0 + args.cruise_s + 5.0 + 3.0  # climb + cruise + descend + settle
    # Pace the loop to real time when someone might be watching live: with warm GPU
    # caches, world.step() runs far faster than the simulated dt it advances, so an
    # unthrottled mission finishes in a fraction of its "real" duration -- nobody has
    # time to connect. Recording-only runs (--no-stream) skip this and just run flat out.
    wall_start = time.monotonic()
    while t < mission_end_s:
        world.step(render=True)
        step += 1
        t += PHYSICS_DT

        if not args.no_stream:
            behind = wall_start + t - time.monotonic()
            if behind > 0:
                time.sleep(behind)

        vehicle.update_state(PHYSICS_DT)  # manually refresh state (see module docstring)

        tx, ty, tz = _target_position(t, spawn_xyz[:2], args.altitude, args.cruise_s)
        pos = vehicle.state.position
        if watch_camera is not None:
            _aim_watch_camera(watch_camera, pos)  # takes effect next render
        vel = vehicle.state.linear_velocity
        force_world = np.array([
            KP_XY * (tx - pos[0]) - KD_XY * vel[0],
            KP_XY * (ty - pos[1]) - KD_XY * vel[1],
            mass * G + KP_Z * (tz - pos[2]) - KD_Z * vel[2],
        ])
        rot = _quat_to_rot(vehicle)
        force_body = rot.inv().apply(force_world)
        vehicle.apply_force(list(force_body), body_part="/body")

        # Simple attitude leveling: drive roll/pitch toward zero, hold yaw.
        roll, pitch, yaw = rot.as_euler("xyz")
        ang_vel = vehicle.state.angular_velocity
        torque_body = np.array([
            -KP_ATT * roll - KD_ATT * ang_vel[0],
            -KP_ATT * pitch - KD_ATT * ang_vel[1],
            -KD_ATT * ang_vel[2],
        ])
        vehicle.apply_torque(list(torque_body), body_part="/body")

        if step % CAPTURE_EVERY_N_STEPS == 0:
            frame = adapter.capture_frame()
            frames.append(frame)
            if video_writer is not None:
                rgb = frame.rgb if args.video_source == "onboard" else watch_camera.get_rgb()
                if rgb is not None:
                    video_writer.write(rgb[:, :, ::-1])  # RGB -> BGR for cv2

    if video_writer is not None:
        video_writer.release()
        print(f"wrote video to {args.video_out}", flush=True)

    stats = export_flight(
        frames, args.out_dir, adapter.intrinsics,
        rate_hz=args.rate_hz, camera_height_m=1.0, pitch_deg=0.0,
    )
    print(f"wrote {stats['frames']} frames to {args.out_dir}")

    simulation_app.close()


if __name__ == "__main__":
    main()
