"""Detect a *difficult maneuver* on the route ahead (ROS-free, 3.8-compatible).

The FALCON ``hybrid`` navigation mode flies A* on the easy stretches and hands
control to the learned NavDP local policy exactly where a slow stop-and-turn
follower struggles: a **hard turn** (a corridor corner into a room) or a **narrow
passage** (a doorway, a tight gap). This module is the pure, testable decision
input for that switch -- given the committed A* route, the drone pose and (for the
narrowness test) a query into the occupancy map, it answers "is the stretch just
ahead difficult, and why?".

Two independent signals, each over a short forward window of the route:

* **Turn** -- the along-route DISTANCE to the next hard corner ahead
  (:func:`sparx_agency.core.planning.replanning.corner_scan_2d.scan_hard_turn_ahead`),
  where a corner's magnitude is the net heading change over a short span each side of
  it. A distance shrinks monotonically as the drone approaches, so the switch stays
  asserted for the whole approach (the arbiter's debounce reliably confirms) and the
  hand-off point is simply "the turn is within the engage range". The per-corner net
  turn keeps the weave-immunity of :func:`net_turn_deg`: a grid A* route jittering
  down a straight corridor cancels to ~0, a genuine 90 deg corner reads its full
  angle. (Limitation, as for net turn: a symmetric S-bend nets to ~0 per corner and
  is NOT flagged -- an accepted trade for weave-immunity.)
* **Narrowness** -- the free passage WIDTH measured *perpendicular* to the route
  (:func:`passage_free_width_2d`): at each sample it marches left and right until
  it meets an obstacle, so the width is the true gap the drone must thread. This
  is deliberately not the nearest-wall clearance: an A* route that clips a convex
  corner in an open room is near one wall but open on the other side (wide gap,
  not narrow), whereas a doorway is tight on *both* sides. Compared against a
  width threshold.

No hysteresis, no state, no ROS here -- the arbiter node owns the confirm-streak
debounce and the actual A*<->NavDP switch. This mirrors the rest of
:mod:`sparx_agency.core.planning.replanning`: pure primitives the node composes.

Python 3.8 compatible (the FALCON Noetic adapter imports ``core`` under 3.8): no
PEP 604 unions, no ``match``/``case``, frozen dataclass without ``slots=``. Pure
Python + :func:`turn_angle_2d` + :func:`remaining_polyline`; no numpy, no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, radians
from typing import Callable, List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.planners.common.corner_rounding_2d import (
    merge_collinear_2d,
    turn_angle_2d,
)

from .corner_scan_2d import scan_hard_turn_ahead
from .path_metrics import point_at_arclength_2d, remaining_polyline

# A query into the occupancy map: True iff the world point (x, y) is a KNOWN
# obstacle. Off-map / unknown cells return False (not a wall), matching the rest
# of the planner -- so an unmapped side reads as open and never false-triggers a
# "narrow" (the switch stays conservative until the wall is actually observed).
OccupiedQuery = Callable[[float, float], bool]


@dataclass(frozen=True)
class RouteDifficulty:
    """Difficulty verdict for the route stretch just ahead of the drone.

    Attributes:
        turn_deg: Net heading change of the relevant corner ahead (deg) -- the sharp
            corner being engaged when ``hard_turn``, else the sharpest sub-threshold
            corner in range (0 for a straight stretch).
        turn_dist_m: Along-route distance from the drone to the next hard turn (m);
            ``inf`` when none is within the turn scan range ("how close is the turn",
            the signal the arbiter engages / disengages on).
        passage_width_m: Narrowest free passage width over the window (m);
            ``inf`` when no occupancy query was supplied (narrowness not tested).
        hard_turn: A corner reaching the turn threshold lies within the scan range.
        narrow: ``passage_width_m`` fell below the width threshold.
        reason: One of ``"clear"`` / ``"turn"`` / ``"narrow"`` / ``"turn+narrow"``.
    """

    turn_deg: float
    turn_dist_m: float
    passage_width_m: float
    hard_turn: bool
    narrow: bool
    reason: str

    @property
    def is_difficult(self) -> bool:
        """True if either the turn or the narrowness signal fired."""
        return self.hard_turn or self.narrow


def _slice_polyline_by_arclength(
    pts: Sequence[Pose2D], start_m: float, end_m: float
) -> List[Pose2D]:
    """Sub-polyline of ``pts`` spanning arclength ``[start_m, end_m]``.

    The window endpoints are interpolated onto the segments they land on, while
    every ORIGINAL interior vertex inside the span is preserved (so a corner's
    turn angle is measured at the true vertex, not smoothed away). Returns just
    the final vertex when the span starts past the end of ``pts``.
    """
    if len(pts) < 2 or end_m <= start_m:
        return [pts[0]] if pts else []
    out = []  # type: List[Pose2D]
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        seg = hypot(dx, dy)
        if seg < 1e-9:
            continue
        seg_start, seg_end = acc, acc + seg
        lo, hi = max(seg_start, start_m), min(seg_end, end_m)
        if hi > lo:  # this segment overlaps the window with non-zero length
            # (strict >: a zero-length touch exactly at start_m/end_m would emit a
            #  duplicate boundary vertex; the real vertices are kept as segment ends)
            t0, t1 = (lo - seg_start) / seg, (hi - seg_start) / seg
            if not out:
                out.append(Pose2D(a.x + t0 * dx, a.y + t0 * dy))
            out.append(Pose2D(a.x + t1 * dx, a.y + t1 * dy))
        acc = seg_end
        if acc >= end_m:
            break
    return out if out else [pts[-1]]


def forward_window_2d(
    points: Sequence[Pose2D],
    pose: Pose2D,
    lookahead_m: float,
    min_index: int = 0,
    skip_m: float = 0.0,
) -> Tuple[List[Pose2D], int]:
    """The route stretch ahead of the drone: ``[skip_m, skip_m + lookahead_m]``.

    Projects ``pose`` onto the committed route (forward-monotone, via
    :func:`remaining_polyline`), then returns the sub-polyline from ``skip_m``
    past that projection out to ``lookahead_m`` further. ``skip_m`` drops the
    immediate footprint the drone already occupies so a maneuver it has just
    finished does not keep reading as "ahead".

    Args:
        points: Committed world-frame route waypoints.
        pose: Current drone position (yaw ignored).
        lookahead_m: Length of the window measured ahead of ``skip_m`` (m).
        min_index: Lowest route segment to project onto (feed back the returned
            index each tick for a forward-monotone projection).
        skip_m: Arclength ahead of the projection at which the window starts (m).

    Returns:
        ``(window, seg_index)``: the window polyline and the projection segment
        index to feed back as ``min_index`` next call.
    """
    remaining, seg = remaining_polyline(points, pose, min_index)
    if len(remaining) < 2 or lookahead_m <= 0.0:
        return remaining, seg
    start = max(0.0, skip_m)
    return _slice_polyline_by_arclength(remaining, start, start + lookahead_m), seg


def windowed_turn_deg(window: Sequence[Pose2D]) -> float:
    """Total absolute heading change over ``window`` (degrees, 0 for a straight run).

    Sums :func:`turn_angle_2d` at every interior vertex, so both a single sharp
    corner and a distributed multi-vertex bend register their full turn. NOTE this
    CUMULATIVE sum inflates on a straight-but-jagged route (a grid A* path weaving
    around occupancy speckle sums many small jogs into a large false turn); the
    difficulty gate uses the per-corner net turn of
    :func:`sparx_agency.core.planning.replanning.corner_scan_2d.scan_hard_turn_ahead`
    instead. Kept for reference/tests.
    """
    if len(window) < 3:
        return 0.0
    total = sum(turn_angle_2d(window[i - 1], window[i], window[i + 1])
                for i in range(1, len(window) - 1))
    return degrees(total)


def net_turn_deg(window: Sequence[Pose2D], edge_span_m: float = 0.6) -> float:
    """Net heading change across ``window`` (degrees): the angle between its ENTRY
    direction (over the first ``edge_span_m`` of arclength) and its EXIT direction
    (over the last ``edge_span_m``).

    Unlike :func:`windowed_turn_deg` (the raw cumulative sum), this is ROBUST to the
    intermediate weaving/jogs a grid A* route accumulates dodging obstacles or
    occupancy speckle -- those cancel between entry and exit -- while a genuine
    corner (a corridor turning into a room) shows its full turn. It also grows only
    as a real corner enters the window, so the hand-off happens near the turn, not
    on the straight approach. Kept for reference/tests: the live hard-turn gate now
    scans the per-corner net turn (:func:`net_turn_at_arclength_2d` via
    :func:`scan_hard_turn_ahead`) to report the DISTANCE to the next hard corner.
    """
    if len(window) < 2:
        return 0.0
    total = sum(hypot(b.x - a.x, b.y - a.y)
                for a, b in zip(window[:-1], window[1:]))
    if total < 1e-6:
        return 0.0
    # Cap the entry/exit chords well under half the window so a corner still reads
    # its full angle as long as it sits clear of the very ends -- keeps the >=thresh
    # detection band wide even at a short lookahead (a 0.5*total cap collapses it to
    # a single tick for short windows and a fast drone can skip past the peak).
    span = min(max(edge_span_m, 1e-3), 0.35 * total)
    p_in = point_at_arclength_2d(window, span)
    p_out = point_at_arclength_2d(window, total - span)
    ix, iy = p_in.x - window[0].x, p_in.y - window[0].y          # entry direction
    ox, oy = window[-1].x - p_out.x, window[-1].y - p_out.y      # exit direction
    if hypot(ix, iy) < 1e-9 or hypot(ox, oy) < 1e-9:
        return 0.0
    return degrees(abs(atan2(ix * oy - iy * ox, ix * ox + iy * oy)))


def _free_distance(
    occupied: OccupiedQuery, x: float, y: float, dx: float, dy: float,
    step_m: float, max_m: float,
) -> float:
    """Free distance from ``(x, y)`` along the unit ray ``(dx, dy)`` to the nearest
    obstacle, capped at ``max_m`` (returned when the ray stays clear).

    The ray is sampled every ``step_m``. The first sample that reads occupied is at
    ``d``, and the last clear sample at ``d - step_m``, so the true wall lies in
    ``(d - step_m, d]``; the free distance is reported as its midpoint estimate
    ``d - step_m/2`` (unbiased, error <= step_m/2) rather than ``d`` -- returning
    ``d`` would systematically over-report clearance by up to ``step_m`` and let a
    genuine doorway read as passable. Use ``step_m`` no larger than the map cell so
    no occupied cell is stepped over.
    """
    d = step_m
    while d <= max_m:
        if occupied(x + dx * d, y + dy * d):
            return max(0.0, d - 0.5 * step_m)
        d += step_m
    return max_m


def passage_widths_2d(
    window: Sequence[Pose2D],
    occupied: OccupiedQuery,
    step_m: float,
    max_half_m: float,
) -> List[float]:
    """Free passage width at each sample along ``window`` (m), measured perpendicular.

    At each sample (evenly spaced along the window at ~``step_m``, both segment
    endpoints included), marches left and right of the route heading until it meets
    an obstacle (each side capped at ``max_half_m``) and sums the two free distances
    -- the true width of the gap the drone must fly through THERE. A doorway is tight
    on both sides (small sum); an open room, or a route that merely clips one convex
    corner, has at least one open (capped) side (large sum). Returns one width per
    sample (empty if nothing to sample); ``2 * max_half_m`` = fully open at a point.
    """
    widths = []  # type: List[float]
    if len(window) < 2 or step_m <= 0.0 or max_half_m <= 0.0:
        return widths
    for a, b in zip(window[:-1], window[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        seg = hypot(dx, dy)
        if seg < 1e-9:
            continue
        hx, hy = dx / seg, dy / seg          # unit route heading
        lx, ly = -hy, hx                     # unit left perpendicular
        n = max(1, int(round(seg / step_m)))
        for k in range(n + 1):
            t = seg * k / n
            sx, sy = a.x + hx * t, a.y + hy * t
            left = _free_distance(occupied, sx, sy, lx, ly, step_m, max_half_m)
            right = _free_distance(occupied, sx, sy, -lx, -ly, step_m, max_half_m)
            widths.append(left + right)
    return widths


def passage_free_width_2d(
    window: Sequence[Pose2D],
    occupied: OccupiedQuery,
    step_m: float,
    max_half_m: float,
) -> float:
    """Narrowest single-sample free passage width along ``window`` (m).

    Convenience min over :func:`passage_widths_2d`; ``2 * max_half_m`` when nothing
    constrains the route. This bare single-sample minimum is sensitive to one-cell
    occupancy speckle, so :func:`assess_route_difficulty` gates "narrow" on a
    *sustained* run of narrow samples (``min_narrow_span_m``), not on this minimum.
    """
    widths = passage_widths_2d(window, occupied, step_m, max_half_m)
    return min(widths) if widths else 2.0 * max(max_half_m, 0.0)


def _has_run_below(values, thresh, min_run):
    """True if ``values`` holds a run of ``>= min_run`` consecutive entries < ``thresh``."""
    run = 0
    for v in values:
        if v < thresh:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def assess_route_difficulty(
    points: Sequence[Pose2D],
    pose: Pose2D,
    lookahead_m: float,
    turn_thresh_deg: float,
    passage_width_thresh_m: float,
    occupied: Optional[OccupiedQuery] = None,
    min_index: int = 0,
    skip_m: float = 0.0,
    sample_step_m: float = 0.1,
    merge_collinear_deg: float = 0.0,
    min_narrow_span_m: float = 0.0,
    turn_span_m: float = 0.7,
    turn_scan_m: Optional[float] = None,
) -> Tuple[RouteDifficulty, int]:
    """Assess whether the route just ahead holds a hard turn or a narrow passage.

    Two independent signals over the stretch just ahead of the drone:

    * **Hard turn** -- via :func:`scan_hard_turn_ahead`: the along-route distance to
      the NEAREST corner turning at least ``turn_thresh_deg`` within the engage range
      (``[skip_m, turn_scan_m]``). A distance rather than a single window verdict, so
      the switch stays asserted for the whole approach (reliable debounce) and yields
      "how close is the turn" directly; the per-corner net turn is robust to the grid
      A* weave (jitter cancels across the span, a real corner does not).
    * **Narrow passage** -- via :func:`passage_widths_2d` over
      :func:`forward_window_2d` (unchanged): perpendicular free width, tight on both
      sides = a doorway. Skipped when no ``occupied`` query is given.

    Args:
        points: Committed world-frame route waypoints.
        pose: Current drone position.
        lookahead_m: Forward window length for the NARROWNESS test (m). Also the
            default turn scan range when ``turn_scan_m`` is unset.
        turn_thresh_deg: Net heading change (deg) at/above which a corner is a hard
            turn.
        passage_width_thresh_m: Passage width (m) below which the route is narrow.
            Also caps each perpendicular march (so an open side contributes exactly
            this much and a mere corner-clip never reads as narrow).
        occupied: Occupancy query for the narrowness test; ``None`` = turn only.
        min_index: Forward-monotone projection hint (feed back the returned index).
        skip_m: Arclength ahead of the drone at which both windows start (m).
        sample_step_m: Perpendicular-march / sampling step (m); ~one cell.
        merge_collinear_deg: Drop route vertices turning less than this (deg) before
            measuring difficulty, de-noising a raw/jagged route. 0 = off.
        min_narrow_span_m: A passage counts as narrow only where it stays below the
            width threshold for at least this arclength (m) -- so a single occupancy
            speckle cell can't trip a false doorway. 0 = any single sample suffices.
        turn_span_m: Entry/exit chord length for the per-corner net turn (m); larger
            averages out more local jog noise and merges a split corner.
        turn_scan_m: How far ahead to scan for a hard turn (m) -- the engage range
            ("close enough to the turn"). Defaults to ``lookahead_m`` when ``None``.

    Returns:
        ``(RouteDifficulty, seg_index)``.
    """
    scan_m = lookahead_m if turn_scan_m is None else max(0.0, turn_scan_m)
    turn_dist_m, hard_deg, max_deg, seg = scan_hard_turn_ahead(
        points, pose, turn_thresh_deg, min_index=min_index, skip_m=skip_m,
        max_scan_m=scan_m, span_m=turn_span_m,
        merge_collinear_deg=merge_collinear_deg)
    hard_turn = turn_dist_m != float("inf")
    turn_deg = hard_deg if hard_turn else max_deg

    window, _ = forward_window_2d(points, pose, lookahead_m, min_index, skip_m)
    if merge_collinear_deg > 0.0 and len(window) > 2:
        window = merge_collinear_2d(window, radians(merge_collinear_deg))
    if occupied is not None:
        widths = passage_widths_2d(
            window, occupied, sample_step_m, max(passage_width_thresh_m, 0.0))
        width = min(widths) if widths else float("inf")
        need = max(1, int(round(min_narrow_span_m / sample_step_m))
                   if min_narrow_span_m > 0.0 and sample_step_m > 0.0 else 1)
        narrow = _has_run_below(widths, passage_width_thresh_m, need)
    else:
        width = float("inf")
        narrow = False

    if hard_turn and narrow:
        reason = "turn+narrow"
    elif hard_turn:
        reason = "turn"
    elif narrow:
        reason = "narrow"
    else:
        reason = "clear"
    return RouteDifficulty(turn_deg=turn_deg, turn_dist_m=turn_dist_m,
                           passage_width_m=width, hard_turn=hard_turn,
                           narrow=narrow, reason=reason), seg
