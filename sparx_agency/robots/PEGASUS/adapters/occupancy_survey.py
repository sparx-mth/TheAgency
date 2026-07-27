"""Measure where a loaded Isaac Sim scene is actually flyable.

The stock Isaac environments are furnished buildings whose interior layout is
not described anywhere machine-readable. Nothing can plan a route through one
until something has looked. This sweeps a horizontal grid at flight altitude and
asks two questions per cell, both directly of PhysX:

* **Is this cell inside the building?** A floor below it and a ceiling above it.
  Without the ceiling test the sweep accepts open field: these assets sit on a
  kilometre-wide ground plane, and a floor-only test once accepted 2.9 million
  cells, nearly all of them outdoors.
* **Is there anything in the way?** A box overlap the size of the airframe,
  covering its vertical extent. An overlap answers "is geometry *here*" exactly,
  which a ray fired in eight directions only approximates -- the old survey
  measured the distance to the nearest wall and inferred the rest.

Cells that are not inside the building come back UNKNOWN rather than free, and
the planner is run with ``unknown_blocked=True``, so a route can never leave
through a wall the sweep never looked at.

Must run inside a live Isaac Sim process **with the timeline playing** -- PhysX
scene queries return nothing against a stopped simulation, silently, which would
survey the whole building as empty space.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D, occupancy_from_mask
from sparx_agency.robots.PEGASUS.adapters.scene_map import LANDABLE_LAYER

MAX_FLOOR_DROP_M = 4.0    # no hit below this and the cell is not over walkable floor
MAX_CEILING_M = 8.0       # no hit above this and the cell is outdoors, not in a room
DEFAULT_RESOLUTION_M = 0.25
DEFAULT_RADIUS_M = 40.0
# Vertical half-extent of the overlap box: the airframe is flat, but a route
# planned at exactly the cruise altitude would happily skim a desk edge 15 cm
# below it, and the autopilot does not hold height to the centimetre.
DEFAULT_BODY_HALF_HEIGHT_M = 0.3
# How much shallower than the full drop to the floor still counts as landable.
# Enough for a rug or a doorway threshold; far less than a desk.
DEFAULT_LANDING_TOLERANCE_M = 0.25


def scene_bounds(prim_path: str = "/World/Scene") -> Tuple[float, float, float, float]:
    """World-space XY bounds of the loaded environment.

    Args:
        prim_path: Stage path the environment was referenced at.

    Returns:
        ``(min_x, min_y, max_x, max_y)``.
    """
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache

    aabb = compute_aabb(create_bbox_cache(), prim_path, include_children=True)
    return float(aabb[0]), float(aabb[1]), float(aabb[3]), float(aabb[4])


def _grid_axes(bounds, radius_m: float, resolution_m: float):
    """Cell-centre coordinate arrays for the surveyed area.

    The asset's own bounding box is often kilometre-scale, so it is clipped to
    ``radius_m`` around the origin -- the building, not the ground plane it
    stands on.
    """
    min_x, min_y, max_x, max_y = bounds
    min_x, max_x = max(min_x, -radius_m), min(max_x, radius_m)
    min_y, max_y = max(min_y, -radius_m), min(max_y, radius_m)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError(
            f"the scene's bounds {bounds} do not intersect a {radius_m:.0f} m "
            f"radius around the origin -- is the environment loaded?"
        )
    xs = np.arange(min_x, max_x, resolution_m) + resolution_m / 2.0
    ys = np.arange(min_y, max_y, resolution_m) + resolution_m / 2.0
    return xs, ys, min_x, min_y


def survey_scene(
    altitude_m: float,
    robot_radius_m: float,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    radius_m: float = DEFAULT_RADIUS_M,
    body_half_height_m: float = DEFAULT_BODY_HALF_HEIGHT_M,
    landing_tolerance_m: float = DEFAULT_LANDING_TOLERANCE_M,
    prim_path: str = "/World/Scene",
    progress_every: int = 20,
) -> Tuple[OccupancyGrid2D, dict, dict]:
    """Sweep the loaded scene and return a flyable-space map.

    Args:
        altitude_m: Height above the world origin to survey at, metres. A map is
            only valid at the altitude it was made at.
        robot_radius_m: Horizontal half-extent of the overlap box, metres. The
            airframe's radius; the planner adds its own standoff on top, so this
            should be the aircraft, not the aircraft plus margin.
        resolution_m: Cell size, metres.
        radius_m: Half-width of the surveyed area around the origin, metres.
        body_half_height_m: Vertical half-extent of the overlap box, metres.
        landing_tolerance_m: How much closer than the full drop to the floor
            still counts as landable, metres. Covers a rug, a threshold, or a
            slightly uneven floor -- not a desk.
        prim_path: Stage path the environment was referenced at.
        progress_every: Print a progress line every this many grid rows. 0 = silent.

    Returns:
        ``(grid, metadata, layers)``. The grid uses FREE/OCCUPIED/UNKNOWN with
        UNKNOWN meaning "not inside the building"; ``layers`` carries the
        ``landable`` mask.

    Raises:
        RuntimeError: If the sweep found no indoor cells at all, which almost
            always means the timeline was not playing.
    """
    import carb
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    xs, ys, origin_x, origin_y = _grid_axes(
        scene_bounds(prim_path), radius_m, resolution_m)

    half_extent = carb.Float3(float(robot_radius_m), float(robot_radius_m),
                              float(body_half_height_m))
    identity = carb.Float4(0.0, 0.0, 0.0, 1.0)  # PhysX wants XYZW here, unlike isaacsim.core

    occupied = np.zeros((len(ys), len(xs)), dtype=bool)
    known = np.zeros((len(ys), len(xs)), dtype=bool)
    landable = np.zeros((len(ys), len(xs)), dtype=bool)

    for gy, y in enumerate(ys):
        for gx, x in enumerate(xs):
            origin = (float(x), float(y), float(altitude_m))
            floor = query.raycast_closest(origin, (0.0, 0.0, -1.0), MAX_FLOOR_DROP_M)
            if not floor["hit"]:
                continue  # nothing underneath -- not over a floor
            if not query.raycast_any(origin, (0.0, 0.0, 1.0), MAX_CEILING_M):
                continue  # open sky above -- outdoors, not in a room
            known[gy, gx] = True
            occupied[gy, gx] = query.overlap_box_any(
                half_extent, carb.Float3(*origin), identity)
            # What is directly below decides whether the aircraft can be *put
            # down* here, which is a different question from whether it can fly
            # through. Anything closer than the full drop to the floor is
            # furniture, and a landing on furniture ends with the airframe on
            # its side.
            landable[gy, gx] = (
                float(floor["distance"]) >= altitude_m - landing_tolerance_m)
        if progress_every and (gy + 1) % progress_every == 0:
            print(f"  surveyed row {gy + 1}/{len(ys)} "
                  f"({int(known.sum())} indoor cells so far)", flush=True)

    if not known.any():
        raise RuntimeError(
            "the survey found no indoor cells. The most likely cause by far is "
            "that the timeline was not playing -- PhysX scene queries return "
            "nothing against a stopped simulation, with no error. Otherwise try "
            "a different --altitude or --radius."
        )

    grid = occupancy_from_mask(
        occupied, resolution_m, origin_x, origin_y, frame_id="world", known=known)
    landable &= known & ~occupied
    metadata = {
        "resolution_m": resolution_m,
        "robot_radius_m": robot_radius_m,
        "body_half_height_m": body_half_height_m,
        "landing_tolerance_m": landing_tolerance_m,
        "radius_m": radius_m,
        "indoor_cells": int(known.sum()),
        "occupied_cells": int((occupied & known).sum()),
        "free_cells": int((~occupied & known).sum()),
        "landable_cells": int(landable.sum()),
        "shape": [int(len(ys)), int(len(xs))],
    }
    return grid, metadata, {LANDABLE_LAYER: landable}
