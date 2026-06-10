"""2D path planning utilities."""
from __future__ import annotations

from math import ceil, hypot
from typing import List, TYPE_CHECKING

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import Costmap2D

from .ompl_imports import ob, OMPL_AVAILABLE

if TYPE_CHECKING:
    from ompl import base as ob


def interpolate_path_2d(points: List[Pose2D], spacing: float) -> List[Pose2D]:
    """Interpolate 2D path at uniform spacing."""
    if len(points) < 2 or spacing <= 0:
        return points

    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        dist = hypot(dx, dy)

        if dist > spacing:
            n_segments = int(dist / spacing)
            for i in range(1, n_segments + 1):
                t = i / (n_segments + 1)
                result.append(Pose2D(a.x + t * dx, a.y + t * dy))
        result.append(b)
    return result


def split_long_segments_2d(points: List[Pose2D], max_seg: float) -> List[Pose2D]:
    """Corner-preserving resample: keep every vertex, split only long legs.

    Unlike :func:`interpolate_path_2d`, this never moves or drops an input
    vertex. It keeps every corner exactly where it is (e.g. the corners left by
    line-of-sight smoothing) and only inserts evenly-spaced intermediate points
    on segments longer than ``max_seg``. A straight 6 m leg with ``max_seg=3``
    therefore yields just its two endpoints plus one midpoint — so a follower
    yaws only at genuine corners, not at every grid step.

    Args:
        points: Ordered path vertices.
        max_seg: Maximum segment length in meters (<= 0 disables splitting).

    Returns:
        A new list starting at ``points[0]`` and ending at ``points[-1]``.
    """
    if len(points) < 2 or max_seg <= 0:
        return list(points)
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        dist = hypot(dx, dy)
        if dist > max_seg:
            n = int(ceil(dist / max_seg))  # n sub-segments -> n-1 inserts
            for k in range(1, n):
                t = k / n
                out.append(Pose2D(a.x + t * dx, a.y + t * dy))
        out.append(b)
    return out


def reduce_path_2d(si, costmap: Costmap2D, states: List, min_clearance: float) -> List:
    """Adaptive waypoint reduction for 2D."""
    if len(states) < 3:
        return [si.cloneState(s) for s in states]

    kept = [si.cloneState(states[0])]
    for i in range(1, len(states) - 1):
        x, y = states[i][0], states[i][1]
        clearance = costmap.world_clearance(x, y)
        can_skip = si.checkMotion(kept[-1], states[i + 1])

        if clearance < min_clearance or not can_skip:
            kept.append(si.cloneState(states[i]))

    kept.append(si.cloneState(states[-1]))
    return kept


def make_clearance_objective_2d(si, costmap: Costmap2D, weight: float):
    """Create 2D clearance objective."""
    if not OMPL_AVAILABLE:
        raise RuntimeError("OMPL not available")

    class ClearanceObjective2D(ob.StateCostIntegralObjective):
        def __init__(self, si, costmap: Costmap2D, weight: float) -> None:
            super().__init__(si, True)
            self._costmap = costmap
            self._weight = weight

        def stateCost(self, state) -> ob.Cost:
            clearance = self._costmap.world_clearance(state[0], state[1])
            return ob.Cost(self._weight / (clearance + 1.0))

    return ClearanceObjective2D(si, costmap, weight)


__all__ = [
    "interpolate_path_2d",
    "split_long_segments_2d",
    "reduce_path_2d",
    "make_clearance_objective_2d",
]
