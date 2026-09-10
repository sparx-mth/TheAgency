"""Draw random, genuinely-flyable start/goal pairs out of an occupancy grid.

Autonomous data collection needs a fresh A-to-B mission every episode, and the
two ends have to satisfy three things that are easy to get wrong:

1. **Both ends must be clear.** A goal 30 cm from a wall is a goal the aircraft
   cannot hold position at -- an estimator that is a few centimetres off puts it
   *in* the wall. Ends are drawn only from cells whose distance to the nearest
   obstacle is at least ``clearance_m``.
2. **A route between them must exist.** Picking two clear cells says nothing
   about whether they are in the same room. Sampling from a single connected
   component of the clear space makes an unreachable goal structurally
   impossible, instead of something the planner discovers and the caller has to
   retry around.
3. **The mission must be worth flying.** Two cells a metre apart produce a
   recording of a drone hovering. ``min_separation_m`` sets the floor.

Everything here is pure numpy on an :class:`OccupancyGrid2D` -- no simulator, no
planner, no ROS -- so it is fast, seedable and unit-testable. UNKNOWN cells are
treated as obstacles throughout: an unsurveyed cell is not a cell anything
should be sent to.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot
from typing import List, Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D, normalize_angle
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.environment.grid_regions import connected_regions
from sparx_agency.core.planning.planners.common.clearance_2d import clearance_field


@dataclass(frozen=True)
class StartGoal:
    """One sampled mission.

    Attributes:
        start: Start pose. ``yaw`` faces the goal (plus any requested jitter).
        goal: Goal pose. ``yaw`` is 0 -- a point goal has no meaningful heading.
        separation_m: Straight-line distance between the two, metres.
        clearance_m: Obstacle clearance both ends were required to have.
    """

    start: Pose2D
    goal: Pose2D
    separation_m: float
    clearance_m: float


def traversable_mask(grid: OccupancyGrid2D, clearance_m: float) -> np.ndarray:
    """Cells that are FREE and at least ``clearance_m`` from any obstacle.

    Args:
        grid: The map. UNKNOWN counts as an obstacle.
        clearance_m: Required distance to the nearest obstacle, metres.

    Returns:
        ``(H, W)`` boolean array indexed ``[gy, gx]``.
    """
    values = grid.values
    cells = grid.grid
    free = cells == values.free
    if clearance_m <= 0.0:
        return free

    blocked = ~free
    clearance = clearance_field(
        blocked, grid.resolution, max_clearance_m=clearance_m + grid.resolution,
    )
    return free & (clearance >= clearance_m)


def largest_region(grid: OccupancyGrid2D, clearance_m: float) -> np.ndarray:
    """The biggest block of contiguous clear space in ``grid``.

    This is the airspace an episode is drawn from: any two of its cells are
    reachable from one another at ``clearance_m`` of standoff, so a planner
    asked for a route between them cannot come back with NO_PATH for
    connectivity reasons.

    Args:
        grid: The map.
        clearance_m: Required obstacle clearance, metres.

    Returns:
        ``(H, W)`` boolean mask.

    Raises:
        ValueError: If no cell in the grid has that much clearance.
    """
    regions = connected_regions(traversable_mask(grid, clearance_m))
    if not regions:
        raise ValueError(
            f"no cell in the map has {clearance_m:.2f} m of clearance -- the map is "
            f"either empty, fully unknown, or the clearance is larger than the building"
        )
    return regions[0]


def _cell_world(grid: OccupancyGrid2D, cells: np.ndarray) -> np.ndarray:
    """Centre-of-cell world coordinates for an ``(N, 2)`` array of ``[gy, gx]``."""
    resolution = grid.resolution
    xs = (cells[:, 1] + 0.5) * resolution + grid.origin_x
    ys = (cells[:, 0] + 0.5) * resolution + grid.origin_y
    return np.stack([xs, ys], axis=1)


def sample_start_goal(
    grid: OccupancyGrid2D,
    rng: np.random.Generator,
    clearance_m: float = 0.6,
    min_separation_m: float = 4.0,
    max_separation_m: Optional[float] = None,
    start_yaw_jitter_rad: float = 0.0,
    region: Optional[np.ndarray] = None,
) -> StartGoal:
    """Draw one random, reachable, worth-flying mission from ``grid``.

    Args:
        grid: The map.
        rng: Seeded numpy generator -- the only source of randomness, so a
            campaign is reproducible from its seed.
        clearance_m: Obstacle clearance both ends must have, metres.
        min_separation_m: Reject pairs closer than this, metres.
        max_separation_m: Reject pairs further than this, metres. None = no cap.
        start_yaw_jitter_rad: Half-width of a uniform perturbation added to the
            start heading. 0 points the aircraft straight at its goal; a large
            value makes it turn before it can fly, which is what a navigation
            policy has to learn to do.
        region: Precomputed traversable region (see :func:`largest_region`), to
            avoid re-running the clearance transform for every episode of a
            campaign. Must be co-registered with ``grid``.

    Returns:
        The sampled :class:`StartGoal`.

    Raises:
        ValueError: If no pair in the region satisfies the separation limits.
    """
    if region is None:
        region = largest_region(grid, clearance_m)

    cells = np.argwhere(region)
    if cells.shape[0] < 2:
        raise ValueError("the traversable region has fewer than two cells")
    world = _cell_world(grid, cells)

    start_idx = int(rng.integers(cells.shape[0]))
    # One vectorised pass over every candidate rather than rejection sampling:
    # a region can be long and thin, where a randomly drawn partner is almost
    # never far enough away and rejection sampling stalls.
    offsets = world - world[start_idx]
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    eligible = distances >= min_separation_m
    if max_separation_m is not None:
        eligible &= distances <= max_separation_m

    if not eligible.any():
        # This start is in a cul-de-sac of the region; the region as a whole may
        # still be fine, so report what was actually measured.
        raise ValueError(
            f"no cell in the traversable region is between {min_separation_m:.1f} m "
            f"and {max_separation_m if max_separation_m is not None else float('inf'):.1f} m "
            f"from the sampled start (region spans {distances.max():.1f} m at most)"
        )

    candidates = np.flatnonzero(eligible)
    goal_idx = int(rng.choice(candidates))

    sx, sy = float(world[start_idx, 0]), float(world[start_idx, 1])
    gx, gy = float(world[goal_idx, 0]), float(world[goal_idx, 1])
    yaw = atan2(gy - sy, gx - sx)
    if start_yaw_jitter_rad > 0.0:
        yaw = normalize_angle(
            yaw + float(rng.uniform(-start_yaw_jitter_rad, start_yaw_jitter_rad))
        )

    return StartGoal(
        start=Pose2D(sx, sy, yaw),
        goal=Pose2D(gx, gy, 0.0),
        separation_m=hypot(gx - sx, gy - sy),
        clearance_m=clearance_m,
    )


def sample_goal_from(
    grid: OccupancyGrid2D,
    rng: np.random.Generator,
    start: Pose2D,
    clearance_m: float = 0.6,
    min_separation_m: float = 4.0,
    max_separation_m: Optional[float] = None,
    region: Optional[np.ndarray] = None,
) -> Tuple[Pose2D, float]:
    """Draw a goal reachable from a start the caller already has.

    The chained form of :func:`sample_start_goal`: a drone that has just landed
    is already standing at a perfectly good start, and flying on from there
    avoids a teleport (and the estimator reset a teleport forces).

    Args:
        grid: The map.
        rng: Seeded numpy generator.
        start: Where the aircraft actually is. Snapped to the nearest cell of
            the traversable region, so a landing that ended slightly off is
            still usable.
        clearance_m: Obstacle clearance the goal must have, metres.
        min_separation_m: Minimum distance from ``start``, metres.
        max_separation_m: Maximum distance from ``start``, metres. None = no cap.
        region: Precomputed traversable region, see :func:`sample_start_goal`.

    Returns:
        ``(goal, separation_m)``.

    Raises:
        ValueError: If the region holds no cell at an acceptable distance.
    """
    if region is None:
        region = largest_region(grid, clearance_m)

    cells = np.argwhere(region)
    if cells.shape[0] == 0:
        raise ValueError("the traversable region is empty")
    world = _cell_world(grid, cells)

    offsets = world - np.array([start.x, start.y])
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    eligible = distances >= min_separation_m
    if max_separation_m is not None:
        eligible &= distances <= max_separation_m
    if not eligible.any():
        raise ValueError(
            f"no traversable cell is between {min_separation_m:.1f} m and "
            f"{max_separation_m if max_separation_m is not None else float('inf'):.1f} m "
            f"from ({start.x:.1f}, {start.y:.1f})"
        )

    candidates = np.flatnonzero(eligible)
    goal_idx = int(rng.choice(candidates))
    gx, gy = float(world[goal_idx, 0]), float(world[goal_idx, 1])
    return Pose2D(gx, gy, 0.0), float(distances[goal_idx])


def snap_to_region(grid: OccupancyGrid2D, region: np.ndarray, x: float, y: float) -> Pose2D:
    """The cell of ``region`` nearest to ``(x, y)``, as a world pose.

    Used to pull a real vehicle position -- which drifts, and may sit a few
    centimetres inside a cell the survey called occupied -- back onto a cell the
    planner will accept as a start.

    Args:
        grid: The map.
        region: Traversable mask, co-registered with ``grid``.
        x: World x, metres.
        y: World y, metres.

    Returns:
        The nearest traversable cell centre, ``yaw`` 0.

    Raises:
        ValueError: If ``region`` is empty.
    """
    cells = np.argwhere(region)
    if cells.shape[0] == 0:
        raise ValueError("the traversable region is empty")
    world = _cell_world(grid, cells)
    nearest = int(np.argmin(np.hypot(world[:, 0] - x, world[:, 1] - y)))
    return Pose2D(float(world[nearest, 0]), float(world[nearest, 1]), 0.0)
