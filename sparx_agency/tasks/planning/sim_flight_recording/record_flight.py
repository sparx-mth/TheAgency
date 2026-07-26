"""Smoke-test harness: spawn a PEGASUS Iris in an Isaac Sim indoor scene, step the
sim, and write a ``recording.py``-compatible flight recording.

This is infrastructure validation, **not** a piloted flight -- it proves the
scene/vehicle/camera/recorder chain produces a directory that
``sim_extract.py`` and ``flight_dataset.py`` agree on, ahead of any actual
NavDP fine-tuning or scripted mission logic (both future work; see
``robots/PEGASUS/README.md``). The vehicle is spawned and the sim is stepped
so PX4 and the camera come up; no arm/takeoff command is sent.

Must run under Isaac Sim's own Python, with the patched Pegasus extension
installed (``robots/PEGASUS/setup/install.sh``), e.g.::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/record_flight.py \\
        --pegasus-root /tmp/dev/PegasusSimulator/extensions/pegasus.simulator \\
        --px4-dir /tmp/dev/PX4-Autopilot \\
        --scene simple_room --out-dir /tmp/dev/recordings/simple_room_smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CAMERA_WARMUP_STEPS = 110  # MonocularCamera.update ignores its first ~100 calls
CAPTURE_STEPS = 30
STAGE_SETTLE_STEPS = 20  # let async USD reference loads compose before prim queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pegasus-root", type=Path, required=True,
                     help="Path to the patched PegasusSimulator/extensions/pegasus.simulator checkout")
    ap.add_argument("--px4-dir", type=Path, required=True,
                     help="Path to a PX4-Autopilot checkout built with 'make px4_sitl_default none'")
    ap.add_argument("--scene", default="simple_room")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from isaacsim.core.utils.extensions import enable_extension
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
    # Let the (async-loaded) scene reference fully compose before spawning the
    # vehicle on top of it -- otherwise Multirotor's own prim queries can race
    # the stage still populating.
    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()

    vehicle = PegasusIrisVehicle(
        pegasus_extension_root=args.pegasus_root,
        init_pos=(0.0, 0.0, 1.0),
        px4_dir=str(args.px4_dir),
    )
    # Same reasoning for the vehicle's own referenced USD (rotor/body sub-prims)
    # before world.reset() walks the articulation tree.
    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()

    world.reset()
    for _ in range(CAMERA_WARMUP_STEPS):
        world.step(render=True)

    frames = []
    for _ in range(CAPTURE_STEPS):
        world.step(render=True)
        frames.append(vehicle.capture_frame())

    stats = export_flight(
        frames, args.out_dir, vehicle.intrinsics,
        rate_hz=args.rate_hz, camera_height_m=1.0, pitch_deg=0.0,
    )
    print(f"wrote {stats['frames']} frames to {args.out_dir}")

    simulation_app.close()


if __name__ == "__main__":
    main()
