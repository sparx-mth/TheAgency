from __future__ import annotations
from dataclasses import dataclass
from math import radians


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
        corner_round: Run corner rounding (merge near-collinear + chamfer
            moderate corners) so a stop-and-turn follower needs fewer/smaller
            in-place yaws. Applied after LOS smoothing, before segment splitting.
        corner_merge_rad: Interior vertices that turn less than this are dropped
            as near-collinear (rad).
        corner_max_turn_rad: Corners gentler than this are already glide-able and
            left untouched (rad). Pair with the follower's ``skip_yaw_thresh``.
        corner_chamfer_max_rad: Corners between ``corner_max_turn_rad`` and this
            are chamfered into two half-angle turns; sharper corners are kept as
            genuine turns (rad). Keep near ``2 * corner_max_turn_rad`` so each
            half is glide-able.
        corner_chamfer_dist_m: How far back from a corner to cut it (m); clamped
            to half the shorter adjacent leg.
        corner_min_runup_m: Minimum leg length on both sides for a corner to be
            chamfered (m); shorter corners are left sharp.
    """
    connectivity: int = 8
    inflate_radius_m: float = 0.4
    unknown_blocked: bool = False
    unknown_cost: float = 1.0
    search_margin_m: float = 3.0
    turn_penalty: float = 0.3
    los_smoothing: bool = True
    waypoint_spacing_m: float = 3.0
    goal_snap_radius_m: float = 2.0
    start_skip_m: float = 0.4
    max_expansions: int | None = 200_000

    # Corner rounding (gentler paths for a stop-and-turn follower).
    corner_round: bool = True
    corner_merge_rad: float = radians(8.0)
    corner_max_turn_rad: float = radians(14.0)
    corner_chamfer_max_rad: float = radians(28.0)
    corner_chamfer_dist_m: float = 0.5
    corner_min_runup_m: float = 0.6


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
