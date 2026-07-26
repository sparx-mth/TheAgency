"""Arm, takeoff, hover, and land a PEGASUS Iris in an Isaac Sim indoor scene --
watchable live over WebRTC -- while recording the flight.

Unlike ``record_flight.py`` (a stationary infra smoke test), this actually
flies: it commands PX4 over MAVLink directly (arm -> NAV_TAKEOFF -> hover ->
NAV_LAND), against the "offboard/companion" MAVLink link PX4 SITL opens per
vehicle instance (instance 0 -> UDP 14540, see
``PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink``).

This script launches PX4 itself (``_launch_px4``) rather than letting Pegasus's
``PX4LaunchTool`` do it (``px4_autolaunch=False``) -- that tool runs PX4 from a
fresh empty temp directory, which fails to boot for this PX4 version (PX4
sources ``$PWD/etc/init.d/rc.vehicle_setup`` at startup, which only exists
under the real build output). See ``robots/PEGASUS/README.md``.

To watch it: connect NVIDIA's Isaac Sim WebRTC Streaming Client to
``<this machine>:49100`` once the log prints ``STREAMING_READY`` (see
``robots/PEGASUS/README.md`` for the download link and why -- this build has
no bundled browser client).

Must run under Isaac Sim's own Python, e.g.::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/fly_and_watch.py \\
        --pegasus-root /tmp/dev/PegasusSimulator/extensions/pegasus.simulator \\
        --px4-dir /tmp/dev/PX4-Autopilot \\
        --scene simple_room --out-dir /tmp/dev/recordings/simple_room_flight \\
        --altitude 2.0 --hover-s 20
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CAMERA_WARMUP_RENDER_TICKS = 110  # MonocularCamera.update ignores its first ~100 render callbacks
STAGE_SETTLE_STEPS = 20  # let async USD reference loads compose before prim queries
MAVLINK_OFFBOARD_PORT = 14540  # PX4 SITL instance 0's companion-computer link
MAX_HEARTBEAT_WAIT_STEPS = 800  # generous bound on how long we'll step waiting for PX4 (now rendered, so real time per step is much higher than the old render=False budget)
RENDER_EVERY_N_STEPS = 10  # physics runs every step; only render (and capture) every Nth
# Rendering every physics step stalls PX4's sensor link somewhat, but the real bug this
# guards against is worse: PX4 SITL's internal clock is entirely driven by the physics
# ticks Pegasus feeds it. Any *blocking* pymavlink call (wait_heartbeat, motors_armed_wait)
# stops world.step() from running while it waits -- which stops PX4's clock -- which means
# it can never finish starting up and send the heartbeat we're blocked waiting for. Real
# deadlock. So this script never blocks: it polls MAVLink non-blockingly from inside the
# step loop instead (see main()).

PX4_VEHICLE_MODEL = "gazebo-classic_iris"  # must match PegasusIrisVehicle's backend config


def _launch_px4(px4_dir: Path):
    """Launch PX4 SITL directly, with the working directory Pegasus's own
    ``PX4LaunchTool`` gets wrong for this PX4 version (see the module
    docstring / ``robots/PEGASUS/README.md``): PX4 sources
    ``$PWD/etc/init.d/rc.vehicle_setup`` at boot, which only exists under the
    real build output (``build/px4_sitl_default``), not an arbitrary empty
    temp directory.
    """
    rootfs_dir = px4_dir / "build" / "px4_sitl_default"  # has etc/, created by the build itself
    rc_script = px4_dir / "ROMFS" / "px4fmu_common" / "init.d-posix" / "rcS"
    env = dict(os.environ)
    env["PX4_SIM_MODEL"] = PX4_VEHICLE_MODEL
    return subprocess.Popen(
        [str(rootfs_dir / "bin" / "px4"), str(px4_dir / "ROMFS" / "px4fmu_common") + "/",
         "-s", str(rc_script), "-i", "0", "-d"],
        cwd=str(rootfs_dir), env=env,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pegasus-root", type=Path, required=True)
    ap.add_argument("--px4-dir", type=Path, required=True)
    ap.add_argument("--scene", default="simple_room")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--altitude", type=float, default=2.0, help="takeoff altitude, metres")
    ap.add_argument("--hover-s", type=float, default=20.0, help="hover duration before landing")
    ap.add_argument("--no-stream", action="store_true", help="skip WebRTC livestream (headless-only)")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from isaacsim.core.utils.extensions import enable_extension
    if not args.no_stream:
        enable_extension("omni.kit.livestream.app")
    enable_extension("pegasus.simulator")
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

    px4_process = _launch_px4(args.px4_dir)
    vehicle = PegasusIrisVehicle(
        pegasus_extension_root=args.pegasus_root,
        init_pos=(0.0, 0.0, 0.2),
        px4_dir=str(args.px4_dir),
        px4_autolaunch=False,  # we launched it ourselves above, with a working cwd
    )
    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()

    world.reset()

    from pymavlink import mavutil
    master = mavutil.mavlink_connection(f"udpin:0.0.0.0:{MAVLINK_OFFBOARD_PORT}")

    # Phase 1: render on every step. Counterintuitively, world.step(render=False) does
    # NOT reliably keep dispatching physics callbacks in this Isaac Sim build -- confirmed
    # by instrumenting Multirotor.update() itself: it fires exactly twice, then stops,
    # with zero exceptions, when stepped with render=False in a tight loop. Rendering is
    # what actually paces/drives the simulation forward here.
    step = 0
    heartbeat_received = False
    for step in range(MAX_HEARTBEAT_WAIT_STEPS):
        world.step(render=True)
        if step <= 10 or step % 50 == 0:
            print(f"PEGASUS_DEBUG phase1 step {step}: position={vehicle.vehicle.state.position}, "
                  f"is_playing={world.is_playing()}", flush=True)
        msg = master.recv_match(blocking=False)
        if msg is not None and msg.get_type() == "HEARTBEAT":
            heartbeat_received = True
            print(f"heartbeat from system {master.target_system} component {master.target_component}")
            break
    if not heartbeat_received:
        print("WARNING: never received a PX4 heartbeat -- recording without flight control")

    # Phase 2: now bring up rendering -- camera warmup, then periodic capture.
    render_ticks = 0
    while render_ticks < CAMERA_WARMUP_RENDER_TICKS:
        render = (step % RENDER_EVERY_N_STEPS == 0)
        world.step(render=render)
        step += 1
        if render:
            render_ticks += 1

    if not args.no_stream:
        print("STREAMING_READY -- connect the Isaac Sim WebRTC Streaming Client to this machine, port 49100")

    frames = []
    flight_commanded = False
    flight_start = None
    landed_at = None
    while True:
        render = (step % RENDER_EVERY_N_STEPS == 0)
        world.step(render=render)
        step += 1
        if render:
            frames.append(vehicle.capture_frame())

        if heartbeat_received and not flight_commanded:
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0,
            )
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, args.altitude,
            )
            print(f"arm + takeoff to {args.altitude} m commanded")
            flight_commanded = True
            flight_start = time.monotonic()

        if not flight_commanded:
            if landed_at is None and step > MAX_HEARTBEAT_WAIT_STEPS + CAMERA_WARMUP_RENDER_TICKS * RENDER_EVERY_N_STEPS + 300:
                break  # no PX4 heartbeat ever arrived; stop after a short recording-only window
            continue

        elapsed = time.monotonic() - flight_start
        if landed_at is None and elapsed >= args.hover_s:
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0,
            )
            print("land commanded")
            landed_at = elapsed
        if landed_at is not None and elapsed >= landed_at + 10.0:
            break

    stats = export_flight(
        frames, args.out_dir, vehicle.intrinsics,
        rate_hz=args.rate_hz, camera_height_m=1.0, pitch_deg=0.0,
    )
    print(f"wrote {stats['frames']} frames to {args.out_dir}")

    px4_process.terminate()
    try:
        px4_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        px4_process.kill()

    simulation_app.close()


if __name__ == "__main__":
    main()
