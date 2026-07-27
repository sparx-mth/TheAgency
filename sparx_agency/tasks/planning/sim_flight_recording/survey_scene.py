"""Survey an indoor scene's flyable space and cache it as a map. Run once per scene.

Produces the :class:`OccupancyGrid2D` that every later flight plans against, and
writes it to ``robots/PEGASUS/maps/``. Surveying takes minutes; loading the
result takes milliseconds, which is what lets a collection campaign plan
hundreds of routes without touching the simulator again.

**A map is only valid at the altitude it was surveyed at** -- clearance at head
height and clearance at desk height are different buildings -- so the altitude
is part of the filename. Re-run this for every altitude you intend to fly.

``--preview`` also writes a PNG of the map. Look at it before trusting a new
scene: a survey that came out mostly empty, or that clearly is not the shape of
the building, is obvious in a picture and invisible in a cell count.

Must run under Isaac Sim's own Python::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/survey_scene.py \\
        --scene office --altitude 1.5 --preview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Steps to run after world.reset() so the physics scene is live. PhysX scene
# queries return nothing at all against a stopped timeline, silently, which
# would survey the entire building as open space.
PHYSICS_SETTLE_STEPS = 20


def _parse_args():
    from sparx_agency.robots.PEGASUS.adapters.occupancy_survey import (
        DEFAULT_BODY_HALF_HEIGHT_M, DEFAULT_RADIUS_M, DEFAULT_RESOLUTION_M,
    )
    from sparx_agency.robots.PEGASUS.adapters.vehicle import AIRFRAME_RADIUS_M

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="a key of scene.INDOOR_SCENES")
    ap.add_argument("--altitude", type=float, default=1.5,
                    help="survey height above the floor, metres")
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M,
                    help="map cell size, metres")
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M,
                    help="half-width of the surveyed area around the origin, metres")
    ap.add_argument("--robot-radius", type=float, default=AIRFRAME_RADIUS_M,
                    help="airframe half-width used for the overlap test, metres. "
                         "The planner adds its own standoff on top of this.")
    ap.add_argument("--body-half-height", type=float, default=DEFAULT_BODY_HALF_HEIGHT_M,
                    help="vertical half-extent of the overlap test, metres")
    ap.add_argument("--map-dir", type=Path, default=None,
                    help="override where the map is written")
    ap.add_argument("--preview", action="store_true",
                    help="also write a PNG of the map next to it")
    return ap.parse_args()


def write_preview(grid, path: Path, landing_region=None) -> None:
    """Save a human-readable picture of a surveyed map.

    Obstacles black, unsurveyed (outside the building) grey, flyable-but-not-
    landable white, and landable green -- which makes it obvious at a glance
    whether the scene has enough floor to run a campaign on, or only airspace
    over furniture. Flipped vertically so +y is up, i.e. a plan view.

    Args:
        grid: The surveyed :class:`OccupancyGrid2D`.
        path: Destination PNG.
        landing_region: Optional boolean mask of cells clear to the floor.
    """
    import cv2
    import numpy as np

    values = grid.values
    image = np.full(grid.grid.shape, 128, dtype=np.uint8)   # unknown
    image[grid.grid == values.free] = 255
    image[grid.grid == values.occupied] = 0
    colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if landing_region is not None:
        colour[np.asarray(landing_region, dtype=bool)] = (120, 220, 120)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.flipud(colour))


def main() -> None:
    args = _parse_args()

    from sparx_agency.tasks.planning.sim_flight_recording import flight_session

    simulation_app = flight_session.boot_isaac(stream=False)
    try:
        from sparx_agency.core.planning.mission import largest_region
        from sparx_agency.robots.PEGASUS.adapters.occupancy_survey import survey_scene
        from sparx_agency.robots.PEGASUS.adapters.scene_map import LANDABLE_LAYER, save_scene_map

        world = flight_session.build_world(simulation_app, args.scene)
        world.reset()
        for _ in range(PHYSICS_SETTLE_STEPS):
            world.step(render=False)

        print(f"surveying '{args.scene}' at {args.altitude:.2f} m, "
              f"{args.resolution:.2f} m cells...", flush=True)
        grid, metadata, layers = survey_scene(
            altitude_m=args.altitude,
            robot_radius_m=args.robot_radius,
            resolution_m=args.resolution,
            radius_m=args.radius,
            body_half_height_m=args.body_half_height,
        )

        # Report the airspace an episode would actually be drawn from, not just
        # the raw cell count: a map that is 90% free but shattered into thirty
        # disconnected pockets cannot produce a single long flight, and one with
        # no landable cells cannot produce any.
        region = largest_region(grid, clearance_m=args.robot_radius)
        cell_area = args.resolution ** 2
        landing_region = region & layers[LANDABLE_LAYER]

        path = save_scene_map(args.scene, args.altitude, grid, metadata, args.map_dir,
                              layers=layers)
        print(f"SURVEY {args.scene}: {metadata['shape'][1]}x{metadata['shape'][0]} cells, "
              f"{metadata['indoor_cells']} indoor "
              f"({metadata['free_cells']} free, {metadata['occupied_cells']} occupied)",
              flush=True)
        print(f"LARGEST_REGION {args.scene}: {int(region.sum())} cells = "
              f"{int(region.sum()) * cell_area:.0f} m^2 of contiguous flyable space "
              f"at {args.robot_radius:.2f} m clearance", flush=True)
        print(f"LANDABLE {args.scene}: {int(landing_region.sum())} cells = "
              f"{int(landing_region.sum()) * cell_area:.0f} m^2 of that is also clear "
              f"to the floor, so an episode can start or finish there", flush=True)
        print(f"wrote map to {path}", flush=True)

        if args.preview:
            preview = path.with_suffix(".png")
            write_preview(grid, preview, landing_region)
            print(f"wrote preview to {preview}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
