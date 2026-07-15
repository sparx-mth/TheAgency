"""Locate the next hard turn on the committed route ahead (ROS-free, 3.8-safe).

The FALCON ``hybrid`` navigation mode flies A* on the straight, open stretches and
hands control to the learned NavDP local policy for a **hard turn** -- a corridor
corner into a room, a sharp bend a slow stop-and-turn follower tracks poorly. The
operator's model is literally *"fly A* up to the turn, switch to NavDP a moment
before it, take A* back once the turn is behind"*. That switch needs one clean
number: **how far ahead the next hard turn is**. This module is that pure geometric
primitive -- given the committed route and the drone pose it returns the along-route
distance to the nearest vertex sharp enough to matter, so the arbiter can stay on A*
while the turn is still far and engage NavDP once it comes within an engage range.

Why a *distance to the turn* and not a single "is the window curvy" verdict: the
distance shrinks monotonically as the drone approaches, so the switch stays asserted
for the whole approach (many consecutive ticks -> the debounce reliably confirms)
instead of firing only inside a razor-thin band of positions and being skipped by a
fast drone.

The turn magnitude at a vertex is the **net heading change** measured over a short
``span_m`` of arclength on each side of it (:func:`net_turn_at_arclength_2d`), NOT
the bare three-point vertex angle. A genuine corner the planner discretized into two
nearby shallow vertices still reads its full angle across the span, while single-cell
grid jitter -- which cancels between entry and exit -- does not inflate into a false
turn. This mirrors the weave-immunity of
:func:`sparx_agency.core.planning.replanning.route_difficulty.net_turn_deg`, applied
per candidate corner rather than once over a whole window.

Python 3.8 compatible (the FALCON Noetic adapter imports ``core`` under 3.8): no PEP
604 unions, no ``match``/``case``. Pure Python; no numpy, no scipy.
"""
from __future__ import annotations

from math import atan2, degrees, hypot, inf, radians
from typing import List, Sequence, Tuple

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.planners.common.corner_rounding_2d import (
    merge_collinear_2d,
    turn_angle_2d,
)

from .path_metrics import point_at_arclength_2d, remaining_polyline


def net_turn_at_arclength_2d(
    pts: Sequence[Pose2D],
    cum: Sequence[float],
    i: int,
    span_m: float,
    total_m: float,
) -> float:
    """Net heading change (deg) at vertex ``i``, measured over ``+/- span_m``.

    The angle between the direction entering ``pts[i]`` (from the point ``span_m``
    of arclength before it) and the direction leaving it (to the point ``span_m``
    after), each chord clamped to the polyline ends. Measuring over a span rather
    than at the immediate neighbours makes a corner split across two nearby vertices
    read its full turn while local jitter cancels. Falls back to the bare three-point
    :func:`turn_angle_2d` when a chord degenerates (the vertex sits on an endpoint).

    Args:
        pts: The polyline (typically the remaining route ahead of the drone).
        cum: Cumulative arclength at each vertex (``cum[i]`` = distance to ``pts[i]``).
        i: Interior vertex index (``0 < i < len(pts) - 1``).
        span_m: Arclength of the entry/exit chords (m).
        total_m: Total polyline arclength (``cum[-1]``).

    Returns:
        Net heading change at the vertex in degrees (0 for a straight run).
    """
    v = pts[i]
    s = cum[i]
    p_in = point_at_arclength_2d(pts, max(0.0, s - span_m))
    p_out = point_at_arclength_2d(pts, min(total_m, s + span_m))
    ix, iy = v.x - p_in.x, v.y - p_in.y            # entry direction
    ox, oy = p_out.x - v.x, p_out.y - v.y          # exit direction
    if hypot(ix, iy) < 1e-9 or hypot(ox, oy) < 1e-9:
        return degrees(turn_angle_2d(pts[i - 1], pts[i], pts[i + 1]))
    return degrees(abs(atan2(ix * oy - iy * ox, ix * ox + iy * oy)))


def scan_hard_turn_ahead(
    points: Sequence[Pose2D],
    pose: Pose2D,
    turn_thresh_deg: float,
    min_index: int = 0,
    skip_m: float = 0.0,
    max_scan_m: float = inf,
    span_m: float = 0.7,
    merge_collinear_deg: float = 0.0,
) -> Tuple[float, float, float, int]:
    """Scan the route ahead of the drone for the next hard turn.

    Projects ``pose`` onto the committed route (forward-monotone) and walks the
    remaining vertices, reporting the nearest one whose net turn
    (:func:`net_turn_at_arclength_2d`) reaches ``turn_thresh_deg`` within the range
    ``[skip_m, max_scan_m]`` measured ahead of the drone. ``skip_m`` drops the
    footprint the drone already occupies so a turn it has just flown does not keep
    reading as "ahead".

    Args:
        points: Committed world-frame route waypoints.
        pose: Current drone position (yaw ignored; the route supplies direction).
        turn_thresh_deg: Net heading change (deg) at/above which a corner is "hard".
        min_index: Forward-monotone projection hint (feed back the returned index).
        skip_m: Near cutoff -- ignore corners closer than this ahead (m).
        max_scan_m: Far cutoff -- ignore corners farther than this ahead (m); this is
            the "engage range" (the arbiter engages NavDP once a hard turn is within).
        span_m: Entry/exit chord length for the per-vertex net turn (m).
        merge_collinear_deg: Drop route vertices turning less than this (deg) before
            scanning, de-noising a raw/jagged route. 0 = off.

    Returns:
        ``(hard_dist_m, hard_turn_deg, max_turn_deg, seg_index)``:
          * ``hard_dist_m`` -- along-route distance to the NEAREST hard corner in
            range, or ``inf`` when there is none.
          * ``hard_turn_deg`` -- that corner's net turn (deg), or 0.0 when none is
            hard.
          * ``max_turn_deg`` -- the largest net turn of ANY corner in range (deg),
            for a live readout even while below threshold.
          * ``seg_index`` -- projection segment index (feed back as ``min_index``).
    """
    remaining, seg = remaining_polyline(points, pose, min_index)
    if merge_collinear_deg > 0.0 and len(remaining) > 2:
        remaining = merge_collinear_2d(remaining, radians(merge_collinear_deg))
    if len(remaining) < 3:
        return inf, 0.0, 0.0, seg

    cum = [0.0]  # type: List[float]
    for a, b in zip(remaining[:-1], remaining[1:]):
        cum.append(cum[-1] + hypot(b.x - a.x, b.y - a.y))
    total = cum[-1]
    lo = max(0.0, skip_m)

    hard_dist = inf
    hard_deg = 0.0
    max_deg = 0.0
    for i in range(1, len(remaining) - 1):
        s = cum[i]
        if s < lo:
            continue
        if s > max_scan_m:
            break                                  # vertices are ordered by arclength
        deg = net_turn_at_arclength_2d(remaining, cum, i, span_m, total)
        if deg > max_deg:
            max_deg = deg
        if deg >= turn_thresh_deg and s < hard_dist:
            hard_dist = s                          # nearest hard corner (first in range)
            hard_deg = deg                         # keep scanning: max_deg spans the range,
            #                                        the s<hard_dist guard keeps the nearest
    return hard_dist, hard_deg, max_deg, seg
