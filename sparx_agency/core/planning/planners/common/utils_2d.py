"""2D path planning utilities."""
from __future__ import annotations

from math import hypot
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


def split_long_segments_2d(points: List[Pose2D], target_seg: float) -> List[Pose2D]:
    """Corner-preserving resample: keep every vertex, even out the long legs.

    Unlike :func:`interpolate_path_2d`, this never moves or drops an input
    vertex. It keeps every corner exactly where it is (e.g. the corners left by
    line-of-sight smoothing) and only inserts intermediate points on the long
    legs between corners.

    Each leg is divided into the *nearest whole number* of equal sub-segments,
    ``round(dist / target_seg)`` (at least one), so the resulting legs land as
    close as possible to ``target_seg`` rather than always below it. A 6 m leg
    with ``target_seg=2.5`` becomes 2 legs of 3.0 m (round(2.4)=2), and a 2.6 m
    leg stays whole (round(1.04)=1) instead of being halved into 1.3 m steps.
    The trade-off of targeting the spacing this way is that a leg may exceed
    ``target_seg`` by up to ~50 % (a leg up to 1.5x is left whole); use a smaller
    ``target_seg`` if a hard ceiling matters. Legs shorter than ~0.5x (e.g. the
    close vertices a chamfered corner leaves) are kept as-is.

    Args:
        points: Ordered path vertices.
        target_seg: Desired segment length in meters (<= 0 disables resampling).

    Returns:
        A new list starting at ``points[0]`` and ending at ``points[-1]``.
    """
    if len(points) < 2 or target_seg <= 0:
        return list(points)
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        dist = hypot(dx, dy)
        n = max(1, int(round(dist / target_seg)))  # nearest count -> n-1 inserts
        for k in range(1, n):
            t = k / n
            out.append(Pose2D(a.x + t * dx, a.y + t * dy))
        out.append(b)
    return out


def decimate_min_spacing_2d(points: List[Pose2D], min_spacing: float) -> List[Pose2D]:
    """Thin a path so consecutive kept waypoints are at least ``min_spacing`` apart.

    The inverse of :func:`split_long_segments_2d`: instead of inserting points on
    long legs, it DROPS points that crowd together. Keeps the first point, then
    greedily keeps a waypoint only when it is at least ``min_spacing`` from the
    previously kept one, and always ends exactly at the last point (replacing the
    final kept interior point if the endpoint would otherwise sit too close).

    A very dense path (e.g. NavDP's ~24 tightly-spaced waypoints) makes a
    downstream lateral corrector push near-coincident neighbours in opposite
    directions, producing a visible zig-zag; thinning to a roughly uniform spacing
    first removes that. A path already spaced wider than ``min_spacing`` (e.g. A*'s
    metre-scale waypoints) is returned unchanged.

    Args:
        points: Ordered path vertices.
        min_spacing: Minimum distance (m) between kept waypoints (<= 0 disables it).

    Returns:
        A new list starting at ``points[0]`` and ending at ``points[-1]`` (>= 2
        points when the input has >= 2).
    """
    n = len(points)
    if min_spacing <= 0.0 or n < 3:
        return list(points)
    out = [points[0]]
    for p in points[1:-1]:
        if hypot(p.x - out[-1].x, p.y - out[-1].y) >= min_spacing:
            out.append(p)
    last = points[-1]
    if len(out) >= 2 and hypot(last.x - out[-1].x, last.y - out[-1].y) < min_spacing:
        out[-1] = last        # endpoint too close to the last kept -> replace it
    else:
        out.append(last)
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
    "decimate_min_spacing_2d",
    "reduce_path_2d",
    "make_clearance_objective_2d",
]
