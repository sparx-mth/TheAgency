from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AStarParams:
    """
    Parameters for A* planning on grids.

    Attributes:
        allow_unknown: If True, planner may traverse UNKNOWN cells.
                       Default False (classic exploration planning behavior).
        connectivity: 4 or 8 for 2D. (3D uses 6/18/26 separately in AStar3DParams)
        max_expansions: Safety cap; None means unlimited.
    """
    allow_unknown: bool = False
    connectivity: int = 4
    max_expansions: int | None = 200_000


@dataclass(frozen=True)
class WeightedAStarParams:
    """
    Parameters for the weighted 2D A* planner on an OccupancyGrid2D.

    Geometry-aware extension of :class:`AStarParams` that builds a float cost
    map (inflation + UNKNOWN weighting), restricts the search to a bounding
    box, and post-processes the path (line-of-sight smoothing + corner-
    preserving resample + start-prefix trim). Distances are in meters.

    Attributes:
        connectivity: 4 or 8 (8 = diagonals at sqrt(2) cost, octile heuristic).
        inflate_radius_m: Obstacle inflation (robot radius). 0 disables.
        unknown_blocked: If True, UNKNOWN cells are impassable.
        unknown_cost: Traversal cost multiplier for UNKNOWN cells when not
            blocked (1.0 = same as free; >1 prefers known-free routes).
        search_margin_m: Bounding-box margin around start/goal. Smaller =
            faster but may miss paths that detour far outside the box.
        turn_penalty: Extra cost on a direction change (suppresses staircasing
            before smoothing). 0 disables.
        los_smoothing: Run greedy line-of-sight smoothing on the raw A* cells.
        waypoint_spacing_m: Max output segment length; corners are preserved,
            only longer legs are split. <= 0 disables splitting.
        goal_snap_radius_m: If the goal cell is blocked, snap to the nearest
            free cell within this radius. 0 disables.
        start_skip_m: Drop leading waypoints within this distance of the start
            so a follower is never pointed at a point it already occupies.
        max_expansions: Safety cap on A* node pops; None means unlimited.
    """
    connectivity: int = 8
    inflate_radius_m: float = 0.4
    unknown_blocked: bool = False
    unknown_cost: float = 1.0
    search_margin_m: float = 3.0
    turn_penalty: float = 0.0
    los_smoothing: bool = True
    waypoint_spacing_m: float = 3.0
    goal_snap_radius_m: float = 2.0
    start_skip_m: float = 0.4
    max_expansions: int | None = 200_000


@dataclass(frozen=True)
class AStar3DParams:
    """
    Parameters for voxel-grid A* (3D).

    Attributes:
        allow_unknown: If True, planner may traverse UNKNOWN voxels (if map supports it).
        connectivity: 6, 18, or 26.
        max_expansions: Safety cap; None means unlimited.
    """
    allow_unknown: bool = False
    connectivity: int = 6
    max_expansions: int | None = 800_000
