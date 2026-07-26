"""Scalar path metrics used by the replanning adopt/keep hysteresis.

ROS-free and 3.8-compatible. When a route-relevant map change triggers a replan,
the policy must decide whether the freshly planned candidate is *worth adopting*
or whether the drone should keep flying its current committed route. Swapping to
a route that is not meaningfully better is exactly what makes a slow stop-and-turn
platform ping-pong between near-equal alternatives, so adoption is gated on a real
length improvement over the *remaining* committed route.

  * :func:`polyline_length` -- geometric length of a route.
  * :func:`remaining_polyline` -- the tail of the committed route from where the
    drone actually is now (forward-monotone projection), so the candidate and the
    old route are compared over the same remaining span rather than counting the
    distance the drone has already flown.

Length is the right comparator here because the planner's cost grid is effectively
binary (free = 1, blocked = inf) with a flat gray weight, so an A*-valid route's
cost is proportional to its Euclidean length; validity (does it cross an inflated
obstacle) is a separate boolean handled by ``WeightedAStarPlanner2D.path_collides``.
Using length also avoids the ~29% axis/diagonal anisotropy of a cell-count metric.
"""
from __future__ import annotations

from math import hypot, inf
from typing import List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import Pose2D


def polyline_length(points: Sequence[Pose2D]) -> float:
    """Total Euclidean length of the polyline (0.0 for < 2 points)."""
    return float(sum(hypot(b.x - a.x, b.y - a.y)
                     for a, b in zip(points[:-1], points[1:])))


def point_at_arclength_2d(
    points: Sequence[Pose2D], s: float
) -> Optional[Pose2D]:
    """Point at arclength ``s`` along the polyline (clamped to the endpoints).

    ``s <= 0`` returns the first vertex and ``s`` past the total length returns the
    last, so callers can measure a chord over a fixed span near the ends without
    running off the polyline. Returns ``None`` only for an empty input. Shared by
    the route-difficulty turn analysis and the hard-turn corner scan so both walk
    arclength identically.
    """
    if not points:
        return None
    if s <= 0.0:
        return points[0]
    acc = 0.0
    for a, b in zip(points[:-1], points[1:]):
        seg = hypot(b.x - a.x, b.y - a.y)
        if acc + seg >= s:
            t = (s - acc) / seg if seg > 1e-9 else 0.0
            return Pose2D(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
        acc += seg
    return points[-1]


def remaining_polyline(
    points: Sequence[Pose2D], pose: Pose2D, min_index: int = 0
) -> Tuple[List[Pose2D], int]:
    """Return the committed route from the drone's current position onward.

    Projects ``pose`` onto the nearest segment **at or after** ``min_index`` and
    returns a new polyline starting at that projection and continuing through the
    remaining vertices. Restricting the search to ``min_index`` onward makes the
    projection forward-monotone: on an A* detour that passes near itself the
    global-nearest segment could be a much later one, which would truncate the
    route and grossly under-estimate the remaining length; feeding back the
    previous segment index each tick prevents that backward/forward snapping.

    Args:
        points: Committed world waypoints.
        pose: Current drone position (yaw ignored).
        min_index: Lowest segment index to consider (the last progress index).
            Clamped into range; 0 searches the whole path.

    Returns:
        ``(remaining, seg_index)`` where ``remaining`` is the tail polyline
        (>= 1 point, starting at the projection) and ``seg_index`` is the segment
        the projection landed on (feed back as ``min_index`` next call). For < 2
        input points, returns ``(copy, min_index)``.
    """
    pts = list(points)
    if len(pts) < 2:
        return pts, min_index
    lo = min(max(0, int(min_index)), len(pts) - 2)
    best_d2 = inf
    best_i = lo
    best_pt = pts[lo]
    for i in range(lo, len(pts) - 1):
        ax, ay = pts[i].x, pts[i].y
        bx, by = pts[i + 1].x, pts[i + 1].y
        vx, vy = bx - ax, by - ay
        seg2 = vx * vx + vy * vy
        if seg2 <= 1e-12:
            t = 0.0
        else:
            t = ((pose.x - ax) * vx + (pose.y - ay) * vy) / seg2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        px, py = ax + t * vx, ay + t * vy
        d2 = (pose.x - px) ** 2 + (pose.y - py) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
            best_pt = Pose2D(px, py)
    # Projection point, then every vertex strictly after the segment it lies on.
    return [best_pt] + pts[best_i + 1:], best_i
