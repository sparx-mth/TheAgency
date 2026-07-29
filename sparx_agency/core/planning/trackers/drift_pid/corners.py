"""Where the route next changes direction, and which way it leaves.

The yaw lookahead needs one fact about the route that line tracking never had to
supply: *how far ahead is the next real corner, how sharp is it, and what
heading does the route hold once it is round?* Everything else the controller
knows about the path is local — the foot, the carrot, the cross-track offset —
and none of it can see a turn coming.

Two things make this harder than "measure the angle at each vertex":

  * **A grid A\\* route weaves.** A 90-degree corner can arrive as three 30-degree
    vertices 15 cm apart, and a straight corridor can carry 10-degree jitter. So
    a candidate corner is the first vertex whose *outgoing* leg has swung far
    enough from the leg being flown — accumulated, not per-vertex — and the
    heading it leaves on is measured over a **run** of path rather than off the
    single next segment.
  * **Corners come in pairs.** Turn right, fly a metre, turn right again. The run
    that defines the outgoing heading therefore stops at the next direction
    change, so the anticipation for the first corner can never be contaminated
    by the second. The drone lines up with the first turn, flies it, and only
    then starts looking into the second.

Why this is not imported from somewhere else. ``core/planning/replanning/
corner_scan_2d.py`` answers a very similar question for the replanner
(``scan_hard_turn_ahead``) and ``core/planning/planners/common/
corner_rounding_2d.py`` owns the canonical ``turn_angle_2d``. Neither may be
used here: importing either drags in ``planners.common`` and with it the OMPL
bindings, which do not exist in the ROS1/Noetic container the FALCON nodes
import ``core`` inside — and which corrupt the heap at interpreter exit where
they do. ``path_simplification/simplifier_2d.py`` re-implements the same
primitive for the same reason. Both of those also answer a slightly different
question: they return an *unsigned* turn, and a controller that must decide
which way to rotate needs the sign.

Body frame is REP-103: ``+x`` forward, ``+y`` left, ``+yaw`` counter-clockwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot
from typing import Optional, Sequence, Tuple

from sparx_agency.core.common.types import normalize_angle

from .geometry import active_segment, project_point_on_segment

XY = Tuple[float, float]

#: Segment shorter than this is treated as a duplicated point and skipped.
_DEGENERATE_M = 1e-9


@dataclass(frozen=True)
class Corner:
    """The next real direction change on the route ahead of the drone.

    Attributes:
        index: Index in the active path of the vertex the route turns at. The
            heading carrot is clamped here while the corner is being
            anticipated: with the nose already round the turn and the body
            still on the leg, a carrot that walked past the corner would aim
            the *travel* vector across the inside of the turn and the drone
            would cut it.
        distance_m: Arc length from the drone's foot on the trajectory to that
            vertex, walked ALONG the path. Never the straight line — the
            straight line to a corner two turns away goes through a wall.
        turn_rad: Signed heading change from the leg being flown to the heading
            the route holds after the corner (+ = left / counter-clockwise).
            Measured to the confirmed outgoing heading, not to the single next
            segment, so a jog that comes straight back reads as no turn.
        heading_out: That outgoing heading in the path frame (rad).
    """

    index: int
    distance_m: float
    turn_rad: float
    heading_out: float


def find_corner(path, wp_idx, px, py, max_distance, min_turn_rad, confirm_m,
                min_run_m=0.0):
    # type: (Sequence[XY], int, float, float, float, float, float, float) -> Optional[Corner]
    """The first real corner within reach ahead of the drone, or None.

    Walks the polyline forward from the drone's foot on the active segment,
    comparing each vertex's outgoing direction against the leg being flown. The
    first vertex that has swung at least ``min_turn_rad`` away from it is the
    corner — so a run of small vertices that adds up to a turn is caught, while
    a straight leg's weave is not.

    The final waypoint is never a corner: the route does not continue past it,
    so there is no heading to anticipate and the drone should simply arrive.

    Args:
        path: The active (re-anchored) waypoints, at least two.
        wp_idx: Index of the waypoint currently being pursued.
        px: Drone x in the path frame (m).
        py: Drone y in the path frame (m).
        max_distance: Stop looking this far along the path (m). A corner beyond
            it is not yet worth anticipating.
        min_turn_rad: Smallest heading change that counts as a turn rather than
            as route noise (rad, > 0).
        confirm_m: How much path past the corner is used to establish the
            outgoing heading (m). See :func:`run_heading`.
        min_run_m: Shortest run the route must hold in the new direction for it
            to be a direction at all (m). A vertex that turns hard and turns
            straight back is a jog, not a corner: there is nothing there to line
            the nose up with, and the search continues past it. 0 accepts any
            run.

    Returns:
        The corner, or None when the route runs straight (or straight enough)
        for ``max_distance``.
    """
    n = len(path)
    if n < 3 or max_distance <= 0.0 or min_turn_rad <= 0.0:
        return None
    ax, ay, bx, by = active_segment(path, wp_idx)
    heading_in = atan2(by - ay, bx - ax)
    qx, qy, _ = project_point_on_segment(px, py, ax, ay, bx, by)

    index = wp_idx if wp_idx >= 1 else 1
    if index > n - 1:
        index = n - 1
    distance = hypot(bx - qx, by - qy)

    while index <= n - 2 and distance <= max_distance:
        leaving = _segment_heading(path, index)
        if (leaving is not None
                and abs(normalize_angle(leaving - heading_in)) >= min_turn_rad):
            heading_out, run_m = run_heading(path, index, confirm_m,
                                             min_turn_rad)
            if run_m >= min_run_m:
                return Corner(index=index, distance_m=distance,
                              turn_rad=normalize_angle(heading_out - heading_in),
                              heading_out=heading_out)
        distance += _segment_length(path, index)
        index += 1
    return None


def run_heading(path, index, confirm_m, min_turn_rad):
    # type: (Sequence[XY], int, float, float) -> Tuple[float, float]
    """Heading the route holds for a run of path leaving vertex ``index``.

    The chord from the corner to wherever the route stops going that way — which
    is either ``confirm_m`` of arc later, the next direction change, or the end
    of the path, whichever comes first. Taking a chord rather than the single
    next segment averages out the weave a grid route leaves on the far side of a
    corner; stopping it at the next direction change is what refuses to look past
    a *second* corner, so a turn-then-turn is anticipated one turn at a time.

    Args:
        path: The active waypoints.
        index: Index of the corner vertex.
        confirm_m: Arc length budget for the run (m).
        min_turn_rad: Heading change at which the run is considered over (rad).

    Returns:
        ``(heading, length)`` — the outgoing heading in the path frame (rad) and
        how much path actually held it (m). A short length is how a jog gives
        itself away. Falls back to the single outgoing segment when the run
        degenerates to a point.
    """
    n = len(path)
    base = _segment_heading(path, index)
    if base is None:
        return 0.0, 0.0
    end = index + 1
    walked = _segment_length(path, index)
    while end <= n - 2 and walked < confirm_m:
        leaving = _segment_heading(path, end)
        if (leaving is None
                or abs(normalize_angle(leaving - base)) >= min_turn_rad):
            break
        walked += _segment_length(path, end)
        end += 1
    cx, cy = float(path[index][0]), float(path[index][1])
    ex, ey = float(path[end][0]), float(path[end][1])
    chord = hypot(ex - cx, ey - cy)
    if chord < _DEGENERATE_M:
        return base, walked
    return atan2(ey - cy, ex - cx), walked


def _segment_heading(path, index):
    # type: (Sequence[XY], int) -> Optional[float]
    """Direction of ``path[index] -> path[index + 1]``, or None if degenerate."""
    ax, ay = float(path[index][0]), float(path[index][1])
    bx, by = float(path[index + 1][0]), float(path[index + 1][1])
    if hypot(bx - ax, by - ay) < _DEGENERATE_M:
        return None
    return atan2(by - ay, bx - ax)


def _segment_length(path, index):
    # type: (Sequence[XY], int) -> float
    """Length of ``path[index] -> path[index + 1]`` (m)."""
    ax, ay = float(path[index][0]), float(path[index][1])
    bx, by = float(path[index + 1][0]), float(path[index + 1][1])
    return hypot(bx - ax, by - ay)
