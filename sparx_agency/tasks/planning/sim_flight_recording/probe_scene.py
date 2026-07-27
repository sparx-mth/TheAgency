"""Survey an indoor scene's free space, so flights can be planned into it.

The stock Isaac environments are furnished buildings whose interior layout
isn't described anywhere machine-readable -- spawning at the origin and flying
a fixed pattern gets you a drone wedged against a wall, which is exactly what
the first ``office`` run did. This script raycasts the scene at flight altitude
and reports where the open space actually is.

Two things it has to get right, both learned the hard way:

* **"Over floor" is not enough to mean indoors.** These assets sit on a huge
  ground plane -- ``office``'s bounding box is over a kilometre across, and a
  naive floor-only test accepted 2.9 million cells, most of them open field
  outside the building. A cell only counts as indoor if it *also* has a ceiling
  within :data:`MAX_CEILING_M`.
* **A waypoint being in open space does not make the leg to it flyable.** Every
  leg of the generated route is raycast along its length, three rays abreast,
  before it is accepted.

Run it once per scene; paste the result into
``robots/PEGASUS/adapters/scene.py``. Must run under Isaac Sim's own Python::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/probe_scene.py \\
        --scene office --altitude 1.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GRID_STEP_M = 0.5
MAX_RAY_M = 30.0
MAX_FLOOR_DROP_M = 4.0    # no hit below this and the cell is not over walkable floor
MAX_CEILING_M = 8.0       # no hit above this and the cell is outdoors, not in a room
LEG_MARGIN_M = 0.5        # half-width of the corridor each route leg is checked through
# Tried widening this to 2.0 (and min_clearance below to 3.5) on 2026-07-27 to
# cover PX4's documented 2-5 m GPS-grade position drift (px4_vision_pose.py),
# since routes "verified clear" at the old margin still ended against
# furniture. Reverted: 3/3 flights on wider-margin routes crashed from a
# compass/accelerometer-bias attitude-estimator divergence (not from hitting
# anything) at inconsistent times (52s, 96s, 140s) regardless of route length
# or width, so the change couldn't be shown to help and wasn't kept. Office's
# unreliability looks like it's this estimator issue more than route
# clearance -- see the incomplete vision-pose fusion work in
# px4_vision_pose.py, which is the more promising angle to continue from.
HORIZONTAL_DIRS = [
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, 0.7071), (-0.7071, -0.7071),
]


def _scene_bounds(prim_path: str = "/World/Scene"):
    """World-space XY bounds of the loaded environment.

    Returns:
        ``(min_x, min_y, max_x, max_y)``.
    """
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache

    aabb = compute_aabb(create_bbox_cache(), prim_path, include_children=True)
    return float(aabb[0]), float(aabb[1]), float(aabb[3]), float(aabb[4])


def _probe_cell(query, x: float, y: float, z: float):
    """Clearance, enclosure, floor distance and ceiling distance at one point.

    ``walls`` (how many of the eight horizontal rays hit anything) is what
    separates "inside a room" from "in open air just outside the building": a
    cell under an overhang passes the ceiling test but has open sky in some
    horizontal direction, so fewer than eight walls.

    Returns:
        ``(clearance_m, walls, floor_m, ceiling_m)``; ``floor_m``/``ceiling_m``
        are ``inf`` when nothing was hit in that direction.
    """
    origin = (x, y, z)
    clearance = MAX_RAY_M
    walls = 0
    for dx, dy in HORIZONTAL_DIRS:
        hit = query.raycast_closest(origin, (dx, dy, 0.0), MAX_RAY_M)
        if hit["hit"]:
            walls += 1
            clearance = min(clearance, float(hit["distance"]))

    down = query.raycast_closest(origin, (0.0, 0.0, -1.0), MAX_FLOOR_DROP_M)
    up = query.raycast_closest(origin, (0.0, 0.0, 1.0), MAX_CEILING_M)
    return (clearance, walls,
            float(down["distance"]) if down["hit"] else float("inf"),
            float(up["distance"]) if up["hit"] else float("inf"))


def leg_is_clear(query, start, end, z: float, margin: float = LEG_MARGIN_M) -> bool:
    """Whether a straight flight from ``start`` to ``end`` at ``z`` is unobstructed.

    Casts three parallel rays -- centreline plus one offset either side by
    ``margin`` -- so the drone's width is accounted for, not just a hairline
    path between two obstacles.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return True
    ux, uy = dx / length, dy / length
    px, py = -uy * margin, ux * margin  # perpendicular offset

    for ox, oy in ((0.0, 0.0), (px, py), (-px, -py)):
        hit = query.raycast_closest((start[0] + ox, start[1] + oy, z), (ux, uy, 0.0), length)
        if hit["hit"]:
            return False
    return True


def survey(altitude: float, radius: float, prim_path: str = "/World/Scene"):
    """Raycast a horizontal grid over the scene at ``altitude``.

    Args:
        altitude: Height above the world origin to survey at, metres.
        radius: Half-width of the grid around the origin. Bounds the search to
            the building rather than the asset's full (often kilometre-scale)
            bounding box.
        prim_path: Stage path the environment was referenced at.

    Returns:
        A dict with the grid bounds and a ``cells`` list of indoor cells, each
        ``{"x", "y", "clearance", "floor", "ceiling"}``. A cell is indoor only
        if it has floor below, ceiling above, *and* is enclosed on all eight
        horizontal directions.
    """
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    min_x, min_y, max_x, max_y = _scene_bounds(prim_path)
    min_x, max_x = max(min_x, -radius), min(max_x, radius)
    min_y, max_y = max(min_y, -radius), min(max_y, radius)

    cells = []
    for x in np.arange(min_x, max_x + GRID_STEP_M, GRID_STEP_M):
        for y in np.arange(min_y, max_y + GRID_STEP_M, GRID_STEP_M):
            clearance, walls, floor, ceiling = _probe_cell(query, float(x), float(y), altitude)
            if floor == float("inf") or ceiling == float("inf") or walls < len(HORIZONTAL_DIRS):
                continue  # open below, open above, or open to the side -- not an indoor cell
            cells.append({"x": round(float(x), 2), "y": round(float(y), 2),
                          "clearance": round(clearance, 2),
                          "floor": round(floor, 2), "ceiling": round(ceiling, 2)})
    return {"bounds": [min_x, min_y, max_x, max_y], "altitude": altitude, "cells": cells}


def best_spawn(result):
    """The most open indoor cell in the survey -- the safest place to spawn.

    Returns:
        ``(x, y, clearance)``.

    Raises:
        RuntimeError: If the survey found no indoor cells at all.
    """
    cells = result["cells"]
    if not cells:
        raise RuntimeError(
            "survey found no indoor cells -- try a different --altitude or --radius"
        )
    best = max(cells, key=lambda c: c["clearance"])
    return best["x"], best["y"], best["clearance"]


def _spread_points(candidates, start, hops: int):
    """Farthest-point sampling: ``hops`` cells spread as widely as possible.

    Picking the single furthest cell each time (the obvious greedy choice) just
    ping-pongs between the two ends of the building, which revisits the same
    two views over and over -- poor training data. Maximising the distance to
    the *nearest already-chosen* point instead spreads waypoints over distinct
    areas.
    """
    chosen = [start]
    remaining = [(c["x"], c["y"]) for c in candidates]
    for _ in range(hops):
        if not remaining:
            break
        furthest = max(remaining, key=lambda p: min(math.dist(p, q) for q in chosen))
        chosen.append(furthest)
        remaining.remove(furthest)
    return chosen[1:]


def plan_route(query, result, altitude: float, hops: int = 5, min_clearance: float = 1.0,
               min_leg_m: float = 2.0):
    """Order well-spread open cells into a route whose every leg is verified clear.

    Waypoints are chosen for coverage (:func:`_spread_points`) and then visited
    nearest-neighbour-first, dropping any target the drone cannot fly straight
    to from where it currently is. Sprawling routes past varied geometry are
    what makes good training data.

    Args:
        query: The PhysX scene-query interface.
        result: A :func:`survey` result.
        altitude: Flight altitude, metres.
        hops: How many waypoints to aim for.
        min_clearance: Minimum obstacle clearance for a cell to be a waypoint.
        min_leg_m: Reject legs shorter than this, so the route actually travels.

    Returns:
        A list of ``(x, y)`` waypoints, starting from the spawn cell. May be
        shorter than ``hops`` if the scene has no clear leg to the rest.
    """
    candidates = [c for c in result["cells"] if c["clearance"] >= min_clearance]
    spawn_x, spawn_y, _ = best_spawn(result)
    spawn = (spawn_x, spawn_y)

    # Oversample: some spread points will turn out to be unreachable in a
    # straight line, and dropping them shouldn't shorten the route.
    targets = _spread_points(candidates, spawn, hops * 3)

    route = [spawn]
    while targets and len(route) <= hops:
        current = route[-1]
        reachable = [t for t in targets
                     if math.dist(t, current) >= min_leg_m
                     and leg_is_clear(query, current, t, altitude)]
        if not reachable:
            break
        nearest = min(reachable, key=lambda t: math.dist(t, current))
        route.append(nearest)
        targets.remove(nearest)
    return route


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--altitude", type=float, default=1.5)
    ap.add_argument("--radius", type=float, default=40.0,
                    help="half-width of the surveyed area around the origin, metres")
    ap.add_argument("--hops", type=int, default=5, help="waypoints to chain into the route")
    ap.add_argument("--out", type=Path, default=None, help="write the full survey as JSON")
    args = ap.parse_args()

    from sparx_agency.tasks.planning.sim_flight_recording import flight_session

    simulation_app = flight_session.boot_isaac(stream=False)
    try:
        from omni.physx import get_physx_scene_query_interface
        from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

        from sparx_agency.robots.PEGASUS.adapters.scene import load_indoor_scene

        pg = PegasusInterface()
        pg.initialize_world()
        world = pg.world
        load_indoor_scene(args.scene)
        for _ in range(flight_session.STAGE_SETTLE_STEPS):
            simulation_app.update()
        world.reset()
        for _ in range(10):
            world.step(render=False)

        result = survey(args.altitude, args.radius)
        spawn = best_spawn(result)
        route = plan_route(get_physx_scene_query_interface(), result, args.altitude, args.hops)

        print(f"SURVEY {args.scene}: bounds={[round(b, 1) for b in result['bounds']]} "
              f"indoor_cells={len(result['cells'])}", flush=True)
        print(f"SPAWN {args.scene}: x={spawn[0]} y={spawn[1]} clearance={spawn[2]} m", flush=True)
        print(f"ROUTE {args.scene}: {json.dumps(route)}", flush=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({**result, "spawn": spawn, "route": route}, indent=1))
            print(f"wrote survey to {args.out}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
