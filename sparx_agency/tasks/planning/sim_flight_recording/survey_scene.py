"""Survey a scene once in 3D, and derive from it every map a flight needs.

One sweep, at 10 cm, over the whole building. Out of it come three artefacts:

``<scene>_voxels.npz``
    The ground-truth 3D occupancy grid. What the 3D planners read, and the
    source of everything else here.
``<scene>_voxels.ply``
    The same occupied voxels as a point cloud, for opening in Open3D and
    actually looking at.
``<scene>_alt<NNN>cm.npz``
    The 2D map flights plan against: a horizontal slab of the voxel grid at the
    cruise altitude, plus a ``landable`` layer saying which cells are clear all
    the way down to the floor.

Deriving the 2D map from the 3D one, rather than sweeping again, is what makes a
second altitude free -- and it guarantees the two cannot disagree.

``--preview`` writes a PNG of the 2D map. Look at it before trusting a new
scene: a survey that came out mostly empty, or that is clearly not the shape of
a building, is obvious in a picture and invisible in a cell count.

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
# Vertical half-thickness of the slab projected into the 2D map. The airframe is
# flat, but a route planned at exactly the cruise altitude would happily skim a
# desk edge below it, and the autopilot does not hold height to the centimetre.
DEFAULT_SLAB_HALF_HEIGHT_M = 0.3


def _parse_args():
    from sparx_agency.robots.PEGASUS.adapters.voxel_survey import (
        DEFAULT_CEILING_M, DEFAULT_FLOOR_M, DEFAULT_RADIUS_M, DEFAULT_RESOLUTION_M,
    )
    from sparx_agency.robots.PEGASUS.adapters.vehicle import AIRFRAME_RADIUS_M

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="a key of scene.INDOOR_SCENES")
    ap.add_argument("--altitude", type=float, default=1.5,
                    help="cruise height the 2D map is sliced at, metres")
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M,
                    help="voxel edge length, metres. Also the 2D map's cell size")
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M,
                    help="half-width of the swept area around the origin, metres")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR_M,
                    help="lowest world z to sweep, metres")
    ap.add_argument("--ceiling", type=float, default=DEFAULT_CEILING_M,
                    help="highest world z to sweep, metres")
    ap.add_argument("--slab-half-height", type=float, default=DEFAULT_SLAB_HALF_HEIGHT_M,
                    help="half-thickness of the slab projected into the 2D map, metres")
    ap.add_argument("--max-ceiling", type=float, default=8.0,
                    help="how far above the flight altitude to look for a ceiling, "
                         "metres. A column with none is outdoors")
    ap.add_argument("--floor-z", type=float, default=0.0,
                    help="world z of the floor, metres, for the landability test")
    ap.add_argument("--robot-radius", type=float, default=AIRFRAME_RADIUS_M,
                    help="airframe half-width, used only to report usable airspace")
    ap.add_argument("--map-dir", type=Path, default=None,
                    help="override where the maps are written")
    ap.add_argument("--sweep-origin", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="world point inside the building to flood the sweep "
                         "from, for a scene with no entry in SCENE_SPAWNS")
    ap.add_argument("--preview", action="store_true",
                    help="also write a PNG of the 2D map next to it")
    ap.add_argument("--no-ply", action="store_true",
                    help="skip the point cloud (it is the largest artefact)")
    return ap.parse_args()


def write_preview(grid, path: Path, landing_region=None) -> None:
    """Save a human-readable picture of the 2D map.

    Obstacles black, unsurveyed (outside the building) grey, flyable-but-not-
    landable white, and landable green -- which makes it obvious at a glance
    whether the scene has enough floor to run a campaign on, or only airspace
    over furniture. Flipped vertically so +y is up, i.e. a plan view.

    Args:
        grid: The 2D :class:`OccupancyGrid2D`.
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


def _sweep_origin(scene: str, altitude_m: float, override=None):
    """A point inside the building for the sweep to flood outward from.

    The generator marks everything it cannot reach as UNKNOWN, so this choice is
    what separates the interior from the kilometre of open ground the asset sits
    on. The hand-measured spawn is a known-open spot by construction.

    A scene with no recorded spawn is a chicken-and-egg problem -- the spawn is
    measured off the map and the map needs the spawn -- so ``--sweep-origin``
    breaks it: survey once from a guess, look at the preview, and record the
    spot in ``SCENE_SPAWNS`` once it is known good.
    """
    if override is not None:
        return (float(override[0]), float(override[1]), altitude_m)

    from sparx_agency.robots.PEGASUS.adapters.scene import scene_spawn

    x, y, _z = scene_spawn(scene)
    return (x, y, altitude_m)


def main() -> None:
    args = _parse_args()

    from sparx_agency.tasks.planning.sim_flight_recording import flight_session

    simulation_app = flight_session.boot_isaac(stream=False)
    try:
        import numpy as np

        from sparx_agency.core.planning.environment.voxel_grid_3d import (
            landable_mask, project_to_occupancy_2d, restrict_to_indoor, save_voxel_grid,
        )
        from sparx_agency.core.planning.mission import largest_region
        from sparx_agency.robots.PEGASUS.adapters.scene_map import (
            LANDABLE_LAYER, save_scene_map, voxel_map_path,
        )
        from sparx_agency.robots.PEGASUS.adapters.voxel_survey import (
            survey_voxels, trim_to_content,
        )
        from sparx_agency.tasks.planning.sim_flight_recording.voxel_export import (
            export_voxel_grid, viewer_snippet,
        )

        world = flight_session.build_world(simulation_app, args.scene)
        world.reset()
        for _ in range(PHYSICS_SETTLE_STEPS):
            world.step(render=False)

        print(f"surveying '{args.scene}' in 3D at {args.resolution:.2f} m...", flush=True)
        voxels, metadata = survey_voxels(
            _sweep_origin(args.scene, args.altitude, args.sweep_origin),
            resolution_m=args.resolution, radius_m=args.radius,
            floor_m=args.floor, ceiling_m=args.ceiling,
        )
        # The flood fill escapes over the roof, so outdoors comes back FREE.
        # A ceiling test is what actually separates the building from the field
        # it stands in; applying it to the grid makes every derived map inherit it.
        voxels = trim_to_content(restrict_to_indoor(voxels, args.altitude,
                                                    args.max_ceiling))
        # Re-read the counts AFTER restricting and trimming: the ones survey_voxels
        # returned describe the raw sweep, which still had the car park in it.
        metadata.update(voxels.stats())
        metadata.update({"scene": args.scene, "shape": list(voxels.voxels.shape),
                         "max_ceiling_m": args.max_ceiling})
        print(f"VOXELS {args.scene}: {voxels}", flush=True)
        print(f"   {metadata['occupied']} occupied, {metadata['free']} free, "
              f"{metadata['unknown']} unknown", flush=True)

        voxel_path = voxel_map_path(args.scene, args.map_dir)
        save_voxel_grid(voxel_path, voxels, metadata)
        print(f"wrote voxel map to {voxel_path} "
              f"({voxel_path.stat().st_size / 1e6:.1f} MB)", flush=True)

        if not args.no_ply:
            ply = voxel_path.with_suffix(".ply")
            export_voxel_grid(voxels, ply)
            print(f"wrote point cloud to {ply} "
                  f"({ply.stat().st_size / 1e6:.1f} MB)", flush=True)
            print(viewer_snippet(ply, args.resolution), flush=True)

        # The 2D map is a slice of the 3D one, never a second sweep.
        grid = project_to_occupancy_2d(voxels, args.altitude, args.slab_half_height)
        landable = landable_mask(voxels, args.altitude, floor_z_m=args.floor_z)
        region = largest_region(grid, clearance_m=args.robot_radius)
        landing_region = region & landable
        cell_area = args.resolution ** 2

        map_metadata = dict(metadata)
        map_metadata.update({
            "slab_half_height_m": args.slab_half_height,
            "derived_from": voxel_path.name,
            "landable_cells": int(landable.sum()),
        })
        path = save_scene_map(args.scene, args.altitude, grid, map_metadata,
                              args.map_dir, layers={LANDABLE_LAYER: landable})
        print(f"SURVEY {args.scene}: {grid.width}x{grid.height} cells @ "
              f"{args.resolution:.2f} m at {args.altitude:.2f} m", flush=True)
        print(f"LARGEST_REGION {args.scene}: {int(region.sum())} cells = "
              f"{int(region.sum()) * cell_area:.0f} m^2 of contiguous flyable space "
              f"at {args.robot_radius:.2f} m clearance", flush=True)
        print(f"LANDABLE {args.scene}: {int(landing_region.sum())} cells = "
              f"{int(landing_region.sum()) * cell_area:.0f} m^2 of that is also clear "
              f"to the floor, so an episode can start or finish there", flush=True)
        print(f"wrote 2D map to {path}", flush=True)

        if args.preview:
            preview = path.with_suffix(".png")
            write_preview(grid, preview, landing_region)
            print(f"wrote preview to {preview}", flush=True)
    except Exception:
        # Kit's fast shutdown in the finally below tears the process down
        # without letting the traceback reach a terminal, so a survey that
        # raised looks exactly like one that simply finished. Print it first.
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
