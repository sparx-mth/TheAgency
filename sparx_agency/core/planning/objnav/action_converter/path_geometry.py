"""Pure path geometry the action converter measures progress with: length, projection and the point a distance along.

Every function maps numbers and ``(x, y)`` tuples to numbers, so each fact the
converter's progress rests on is testable on its own. Two quiet failures are
designed out here rather than patched in the converter:

* **Losing progress.** A projection onto the globally nearest segment snaps
  back to an earlier leg where a path doubles back near itself. Subtler, at a
  corner that nearly reverses, the aim point a lookahead ahead lies on the
  next leg *behind* the agent: walking toward it runs the projection back
  along the current segment, the aim slides back in front, and the agent
  paces between two poses forever (a 600-case random sweep livelocked five
  times on a segment bound alone). :func:`project_onto_path` therefore bounds
  its search by segment *and* by arc length -- the caller's progress -- and
  gives a tie to the earlier segment.
* **Jumping ahead.** The opposite failure: a later leg that passes nearer
  than the one being walked -- the return of an out-and-back a few
  centimetres beside its way out, a tour's last leg crossing its first --
  takes a nearest-wins projection there, and everything between is skipped.
  So the caller may give a band round the point: the first leg the point
  stands within it of, and has not walked past the end of, keeps it.
* **An aim on the agent's own foot.** Where a path folds back exactly onto
  itself, the point a lookahead past the progress lands on the return leg at
  the agent's own position, and the heading to it is the angle of a rounding
  error: the agent turns ninety degrees and steps sideways off the path (a
  path of quarter-metre reversals paced so forever). :func:`arclength_beyond`
  finds where the path leaves a disc round the agent, so the aim can be moved
  out of it.

These are deliberate tuple versions of helpers found elsewhere.
``core.planning.replanning.path_metrics`` has ``Pose2D`` versions, but importing
it runs ``replanning/__init__``, which imports the planners and the OMPL
bindings (heavy, and they corrupt the heap at interpreter exit). The segment
projection in ``core.planning.trackers.roll_assist_follower.algorithm`` loads
the whole tracker registry, 28 modules, for eight lines of arithmetic.

Python 3.8 syntax; numpy arrives only through the types it imports.
"""
from __future__ import annotations

import math
import numbers
from typing import List, Optional, Sequence, Tuple

from sparx_agency.core.planning.objnav.action_converter.checks import (
    is_finite_real,
    require_finite,
    require_positive,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.command import Waypoint


def _vertex(points: Sequence[Waypoint], index: int) -> Tuple[float, float]:
    """Waypoint ``index`` as a float pair, or raise naming it."""
    try:
        x, y = points[index]
    except (TypeError, ValueError):
        raise ObjNavError("waypoint %d must be an (x, y) pair, got %r"
                          % (index, points[index]))
    if not (is_finite_real(x) and is_finite_real(y)):
        raise ObjNavError("waypoint %d must be finite, got %r"
                          % (index, points[index]))
    return float(x), float(y)


def _cumulative(points: Sequence[Waypoint]
                ) -> Tuple[List[Tuple[float, float]], List[float]]:
    """The vertices as float pairs and the arc length at each, or raise.

    The one accumulation every arc length here comes from, so a projection's
    arc never exceeds the path length by a rounding error.
    """
    if len(points) == 0:
        raise ObjNavError("a path needs at least one waypoint, got none")
    vertices = [_vertex(points, 0)]
    lengths = [0.0]
    for index in range(1, len(points)):
        x, y = _vertex(points, index)
        px, py = vertices[-1]
        lengths.append(lengths[-1] + math.hypot(x - px, y - py))
        vertices.append((x, y))
    return vertices, lengths


def path_length(points: Sequence[Waypoint]) -> float:
    """Length of the polyline through ``points``, metres.

    Args:
        points: World ``(x, y)`` waypoints; one point has length 0.

    Returns:
        The sum of the segment lengths.

    Raises:
        ObjNavError: On an empty path or a malformed waypoint.
    """
    return _cumulative(points)[1][-1]


def project_onto_path(points: Sequence[Waypoint], x: float, y: float,
                      from_segment: int = 0, from_arc_m: float = 0.0,
                      keep_within_m: float = 0.0
                      ) -> Tuple[int, float, float, Tuple[float, float]]:
    """Where ``(x, y)`` projects onto the path, looking only forward.

    Candidates are the segments from ``from_segment`` on, and on them only the
    part at or after arc length ``from_arc_m``: the caller's progress, which
    therefore never runs backwards. The nearest candidate wins, a tie going
    to the lowest index, so a point equally near the current leg and a later
    one stays on the current leg. With a band, the first segment whose
    candidate lies within ``keep_within_m`` of the point wins outright -- a
    later leg that passes nearer, such as the return of an out-and-back a few
    centimetres beside its way out, is a pass not yet reached -- unless that
    candidate is the segment's end vertex: the point has walked past it, and
    the next segment, which starts there, decides. The defaults search the
    whole path for its nearest point.

    Args:
        points: World ``(x, y)`` waypoints; segment ``i`` joins points ``i``
            and ``i + 1``.
        x: Query point east, metres.
        y: Query point north, metres.
        from_segment: The lowest segment to consider, clamped into the path.
        from_arc_m: The lowest arc length to consider, metres. At or past the
            end only the last waypoint remains.
        keep_within_m: The band round the point, metres: the first segment
            the point stands within it of, and not past the end of, wins. 0
            leaves the nearest-wins rule alone.

    Returns:
        ``(segment_index, arc_length_m, cross_track_m, projection_xy)``. A
        one-point path gives ``(0, 0.0, distance to the point, the point)``.

    Raises:
        ObjNavError: On an empty path, a malformed waypoint, a non-finite
            query point, arc bound or band, a negative band, or a non-integer
            ``from_segment``.
    """
    vertices, lengths = _cumulative(points)
    require_finite("x", x)
    require_finite("y", y)
    require_finite("from_arc_m", from_arc_m)
    require_finite("keep_within_m", keep_within_m)
    if keep_within_m < 0.0:
        raise ObjNavError("keep_within_m must be at least 0, got %r"
                          % (keep_within_m,))
    if (not isinstance(from_segment, numbers.Integral)
            or isinstance(from_segment, bool)):
        raise ObjNavError("from_segment must be an integer segment index, "
                          "got %r" % (from_segment,))
    if len(vertices) == 1:
        px, py = vertices[0]
        return 0, 0.0, math.hypot(x - px, y - py), (px, py)
    last = len(vertices) - 2
    best = None
    for index in range(min(max(int(from_segment), 0), last), last + 1):
        # A segment wholly behind the progress mark is no candidate; the last
        # one always is, so something always wins.
        if lengths[index + 1] < from_arc_m and index < last:
            continue
        candidate = _project_onto_segment(vertices, lengths, index, x, y,
                                          from_arc_m)
        if (keep_within_m > 0.0 and candidate[2] <= keep_within_m
                and (index == last or candidate[1] < lengths[index + 1])):
            # The first leg the point stands beside and not past the end of;
            # a later leg that passes nearer is a pass not yet reached. (A
            # candidate at its segment's end vertex is passed over: the next
            # segment starts there, so nothing nearer is lost.)
            return candidate
        # Strict: on a tie the earlier segment, already held, keeps it.
        if best is None or candidate[2] < best[2]:
            best = candidate
    return best


def _project_onto_segment(vertices: List[Tuple[float, float]],
                          lengths: List[float], index: int, x: float,
                          y: float, from_arc_m: float
                          ) -> Tuple[int, float, float, Tuple[float, float]]:
    """:func:`project_onto_path`'s answer restricted to segment ``index``."""
    (ax, ay), (bx, by) = vertices[index], vertices[index + 1]
    ex, ey = bx - ax, by - ay
    length = math.hypot(ex, ey)
    t = 0.0
    if length > 0.0:
        lowest = 0.0
        if from_arc_m > lengths[index]:
            lowest = min(1.0, (from_arc_m - lengths[index]) / length)
        along = ((x - ax) * ex + (y - ay) * ey) / (length * length)
        t = min(1.0, max(lowest, along))
    qx, qy = ax + t * ex, ay + t * ey
    # t * length <= length, so the arc never passes the next vertex's.
    return (index, lengths[index] + t * length, math.hypot(x - qx, y - qy),
            (qx, qy))


def point_at_arclength(points: Sequence[Waypoint], s: float
                       ) -> Tuple[float, float]:
    """The point ``s`` metres along the path from its first waypoint.

    Args:
        points: World ``(x, y)`` waypoints. At least one.
        s: Arc length, metres, clamped to ``[0, path_length(points)]``. At or
            past the end the answer is exactly the last waypoint, which is how
            a caller tells that it aims at the goal.

    Returns:
        The point, as a float pair.

    Raises:
        ObjNavError: On an empty path, a malformed waypoint, or a NaN ``s``.
    """
    vertices, lengths = _cumulative(points)
    if (not isinstance(s, numbers.Real) or isinstance(s, bool)
            or math.isnan(s)):
        raise ObjNavError("s must be a number of metres, got %r" % (s,))
    if s <= 0.0:
        return vertices[0]
    if s >= lengths[-1]:
        return vertices[-1]
    # 0 < s < total: a later vertex has reached s, and the segment ending
    # there has a positive length.
    index = 1
    while lengths[index] < s:
        index += 1
    (ax, ay), (bx, by) = vertices[index - 1], vertices[index]
    t = (s - lengths[index - 1]) / (lengths[index] - lengths[index - 1])
    return ax + t * (bx - ax), ay + t * (by - ay)


def arclength_beyond(points: Sequence[Waypoint], x: float, y: float,
                     radius_m: float, from_arc_m: float = 0.0) -> float:
    """The first arc length at or after ``from_arc_m`` whose path point is at least ``radius_m`` from ``(x, y)``.

    Args:
        points: World ``(x, y)`` waypoints. At least one.
        x: Disc centre east, metres.
        y: Disc centre north, metres.
        radius_m: Disc radius, metres.
        from_arc_m: Where along the path the search starts, metres, clamped
            to the path.

    Returns:
        The arc length, metres; the path length when the rest of the path
        stays inside the disc.

    Raises:
        ObjNavError: On an empty path, a malformed waypoint, a non-finite
            centre or start, or a radius that is not positive and finite.
    """
    vertices, lengths = _cumulative(points)
    require_finite("x", x)
    require_finite("y", y)
    require_finite("from_arc_m", from_arc_m)
    require_positive("radius_m", radius_m)
    total = lengths[-1]
    start = min(max(float(from_arc_m), 0.0), total)
    for index in range(len(vertices) - 1):
        if lengths[index + 1] < start:
            continue
        found = _leave_disc_on_segment(vertices, lengths, index, x, y,
                                       radius_m, start)
        if found is not None:
            return min(found, total)
    return total


def _leave_disc_on_segment(vertices: List[Tuple[float, float]],
                           lengths: List[float], index: int, x: float,
                           y: float, radius_m: float, from_arc_m: float
                           ) -> Optional[float]:
    """:func:`arclength_beyond` on segment ``index`` alone; None if it stays inside."""
    (ax, ay), (bx, by) = vertices[index], vertices[index + 1]
    length = math.hypot(bx - ax, by - ay)
    if length == 0.0:
        return None
    ux, uy = (bx - ax) / length, (by - ay) / length
    t0 = min(max(from_arc_m - lengths[index], 0.0), length)
    # The start relative to the centre; |p + tau u|^2 = r^2 is a quadratic.
    px, py = ax + t0 * ux - x, ay + t0 * uy - y
    c = px * px + py * py - radius_m * radius_m
    if c >= 0.0:
        return lengths[index] + t0
    b = ux * px + uy * py
    # Inside at the start (c < 0), so the exit root is the positive one.
    tau = -b + math.sqrt(b * b - c)
    if t0 + tau > length:
        return None
    return lengths[index] + t0 + tau
