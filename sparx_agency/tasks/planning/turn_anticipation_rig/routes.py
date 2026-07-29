"""The routes the rig flies: hand-built corners, and real ones off the survey.

Two sources, and both are needed for the comparison to mean anything.

**Hand-built corners** are the controlled experiment. One turn, of a known angle,
in a known place, with nothing else going on — so a difference in the flight is
attributable to the turn and not to the route. The set covers what the
anticipation is claimed to handle: a single 90, two corners a metre apart (the
case that must not be anticipated as one), an S-bend, and a gentle bend it is
supposed to leave alone.

**Survey routes** are the reality check. The same weighted A* the FALCON stack
plans with, on the committed 10 cm survey of the office the PEGASUS aircraft
flies (``robots/PEGASUS/maps/office_alt0150cm.npz``), put through the same
trajectory simplifier the stack runs before the follower ever sees a waypoint.
Whatever corner distribution that chain produces is the one the controller meets
in the air, including the ones nobody would think to hand-build.
"""
from __future__ import annotations

import pathlib
from typing import List, Optional, Tuple

from sparx_agency.core.common.types import Pose2D

#: Committed 10 cm survey of the office, sliced at the 150 cm cruise height.
OFFICE_MAP = (pathlib.Path(__file__).resolve().parents[3]
              / "robots" / "PEGASUS" / "maps" / "office_alt0150cm.npz")

#: ``(name, waypoints)`` — the controlled experiments.
CORRIDORS = (
    ("right turn", [Pose2D(0.0, 0.0), Pose2D(5.0, 0.0), Pose2D(5.0, -3.0)]),
    ("left turn", [Pose2D(0.0, 0.0), Pose2D(5.0, 0.0), Pose2D(5.0, 3.0)]),
    ("right then right, 1 m apart",
     [Pose2D(0.0, 0.0), Pose2D(4.0, 0.0), Pose2D(4.0, -1.0), Pose2D(7.0, -1.0)]),
    ("S-bend, left then right",
     [Pose2D(0.0, 0.0), Pose2D(3.0, 0.0), Pose2D(3.0, 1.5), Pose2D(6.0, 1.5)]),
    ("gentle 30 degree bend",
     [Pose2D(0.0, 0.0), Pose2D(3.0, 0.0), Pose2D(6.0, 1.73)]),
    ("hairpin", [Pose2D(0.0, 0.0), Pose2D(4.0, 0.0), Pose2D(1.5, -2.0)]),
)


def survey_routes(count=4, seed=7, min_separation_m=8.0, max_separation_m=25.0):
    # type: (int, int, float, float) -> List[Tuple[str, List[Pose2D]]]
    """Plan ``count`` routes across the surveyed office, simplified as flown.

    Uses the stack's own chain — weighted A* at the aircraft's standoff, then
    the trajectory simplifier — so the corners are the ones the follower would
    actually be handed. Deterministic for a given seed.

    Args:
        count: How many routes to plan.
        seed: Seed for the start/goal sampler.
        min_separation_m: Shortest route worth flying (m).
        max_separation_m: Longest (m).

    Returns:
        ``(name, waypoints)`` pairs. Empty if the map is missing.

    Raises:
        RuntimeError: If the map is present but no route could be planned,
            which means the chain is broken rather than the map unlucky.
    """
    if not OFFICE_MAP.exists():
        return []
    # Imported here, not at module scope: this pulls numpy, the OMPL bindings
    # (via core.planning.planners) and the simplifier, none of which the
    # hand-built corridors need -- and the corridors are what runs in CI.
    import numpy as np
    from sparx_agency.core.planning.environment import load_occupancy_grid
    from sparx_agency.core.planning.interfaces.planner import PlanRequest
    from sparx_agency.core.planning.mission import largest_region, sample_start_goal
    from sparx_agency.core.planning.path_simplification import (
        TrajectorySimplifier2D, TrajectorySimplifierConfig,
    )
    from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
    from sparx_agency.core.planning.planners.astar.weighted_planner_2d import (
        WeightedAStarPlanner2D,
    )

    grid, _, _ = load_occupancy_grid(OFFICE_MAP)
    planner = WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=0.45, inflate_floor_m=0.30,
        waypoint_spacing_m=0.30, unknown_blocked=True, heading_penalty_m=0.0))
    simplifier = TrajectorySimplifier2D(TrajectorySimplifierConfig())
    region = largest_region(grid, 0.6)
    rng = np.random.default_rng(seed)

    routes = []                    # type: List[Tuple[str, List[Pose2D]]]
    for attempt in range(count * 20):
        if len(routes) >= count:
            break
        pair = sample_start_goal(grid, rng, clearance_m=0.6,
                                 min_separation_m=min_separation_m,
                                 max_separation_m=max_separation_m,
                                 region=region)
        if pair is None:
            continue
        result = planner.plan(
            PlanRequest(start=pair.start, goal=pair.goal,
                        frame_id=grid.frame_id), grid)
        if not result.ok or result.path is None:
            continue
        points = simplifier.simplify(list(result.path.points)).points
        if len(points) < 3:
            continue               # no corner to anticipate: nothing to compare
        routes.append(("survey %d (%d waypoints)" % (len(routes) + 1,
                                                     len(points)),
                       [Pose2D(p.x, p.y, 0.0) for p in points]))
    if not routes:
        raise RuntimeError(
            "the survey map loaded but no route survived planning + "
            "simplification -- the chain, not the map, is what to look at")
    return routes


def corner_angles(waypoints):
    # type: (List[Pose2D]) -> List[float]
    """Signed turn at every interior vertex of a route (rad), for reporting."""
    from math import atan2

    from sparx_agency.core.common.types import normalize_angle

    turns = []
    for i in range(1, len(waypoints) - 1):
        a, v, b = waypoints[i - 1], waypoints[i], waypoints[i + 1]
        incoming = atan2(v.y - a.y, v.x - a.x)
        outgoing = atan2(b.y - v.y, b.x - v.x)
        turns.append(normalize_angle(outgoing - incoming))
    return turns
