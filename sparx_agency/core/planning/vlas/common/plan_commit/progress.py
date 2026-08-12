"""Where a point sits on a polyline: arc length along it, and distance from it.

Two numbers, one pass. A committed plan needs both and needs them to agree:
*how far along* decides when the commitment has been flown, and *how far off*
decides whether it is still being flown at all.

``core/planning/planners/common/utils_2d.arclength_fraction_2d`` already answers
the first question, but it takes ``Pose2D`` and lives in a module that imports
OMPL -- which a VLA runtime, and the Noetic container it has to import inside,
must not pull in. This is the same projection on plain ``(N, 2)`` arrays,
returning metres rather than a fraction because ``min_commit_m`` and
``max_deviation_m`` are both distances.

numpy only, Python 3.8 idioms, numpy 1.17 API: this module is on the FALCON
import path.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

DEGENERATE_M = 1e-9
"""A segment shorter than this has no direction and is skipped when projecting."""

CURSOR_WINDOW = 4
"""How many segments ahead of the cursor a projection may jump in one step.

A lower bound alone does not make progress trustworthy. Nearest-point projection
is a *global* argmin, so on a route that comes back near itself the return leg is
a candidate from the very first tick: an aircraft 4 cm from the anchor of a
hairpin projects onto the way back, reads 4.8 m of progress against a 2.4 m
commitment, and is declared finished having flown nothing -- while the carrot
points backwards. Refusing to consider segments more than a few ahead of the
cursor makes that impossible: to reach the return leg the cursor must first walk
every segment of the way out, each of which requires the aircraft to actually be
nearest to it.

Four is generous. A prediction's waypoints are ~0.2 m apart, so this permits
~0.8 m of advance per call -- 26 m/s at the FALCON node's 33 Hz, and far more
than that in the 250 Hz simulator loop."""


def cumulative_arc(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length of an ``(N, 2)`` polyline, starting at ``0``.

    Args:
        points: ``(N, 2)`` polyline vertices in order.

    Returns:
        ``(N,)`` arc length at each vertex; ``[0.0]`` for a single point.
    """
    path = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if path.shape[0] < 2:
        return np.zeros(max(path.shape[0], 1), dtype=np.float64)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([np.zeros(1), np.cumsum(steps)])


def project(points: np.ndarray, x: float, y: float, from_segment: int = 0,
            window: int = CURSOR_WINDOW) -> Tuple[float, float, int]:
    """Project ``(x, y)`` onto a polyline, within a window ahead of a cursor.

    Every candidate segment is projected onto independently and the nearest
    projection wins. A plan is two dozen points long, so the whole sweep is one
    vectorised expression and there is no reason to walk it.

    The cursor and its window are what make the answer usable as *progress*
    rather than merely as a nearest point. Both bounds are load-bearing: the
    lower one stops an aircraft blown backwards from un-flying a route, and the
    upper one stops a route that doubles back from being declared flown from a
    standing start. See :data:`CURSOR_WINDOW`.

    Args:
        points: ``(N, 2)`` polyline vertices in order.
        x, y: The query point, same frame as ``points``.
        from_segment: Ignore segments before this index. Clamped into range, so
            a stale cursor on a shorter plan degrades to the last segment rather
            than raising.
        window: How many segments beyond ``from_segment`` may be considered.
            ``None`` or a negative value searches to the end of the polyline --
            the unbounded behaviour, which is only safe when the caller knows
            the route cannot approach itself.

    Returns:
        ``(arc_m, lateral_m, segment)`` -- arc length from the polyline's start
        to the nearest projection, the distance from ``(x, y)`` to that
        projection, and the index of the segment it landed on. ``segment`` lies
        in ``[from_segment, from_segment + window]``.
    """
    path = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if path.shape[0] < 2:
        origin = path[0] if path.shape[0] else np.zeros(2)
        return 0.0, float(np.hypot(x - origin[0], y - origin[1])), 0

    first = max(0, min(int(from_segment), path.shape[0] - 2))
    last = (path.shape[0] - 1 if window is None or window < 0
            else min(path.shape[0] - 1, first + int(window) + 1))
    start, end = path[first:last], path[first + 1:last + 1]
    delta = end - start
    length = np.linalg.norm(delta, axis=1)
    usable = length > DEGENERATE_M
    if not usable.any():
        arc = cumulative_arc(path)
        return (float(arc[first]),
                float(np.hypot(x - path[first, 0], y - path[first, 1])), first)

    safe = np.where(usable, length, 1.0)
    along = ((x - start[:, 0]) * delta[:, 0]
             + (y - start[:, 1]) * delta[:, 1]) / (safe * safe)
    along = np.clip(along, 0.0, 1.0)
    foot = start + along[:, None] * delta
    offset = np.hypot(foot[:, 0] - x, foot[:, 1] - y)
    # A zero-length segment projects onto its own vertex from every direction and
    # would win ties against the real geometry it sits on.
    offset = np.where(usable, offset, np.inf)

    local = int(np.argmin(offset))
    segment = first + local
    arc = cumulative_arc(path)
    return (float(arc[segment] + along[local] * length[local]),
            float(offset[local]), segment)
