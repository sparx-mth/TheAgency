"""Path geometry the drift-PID controller needs: the line, the carrot, the errors.

Pure functions, no state. The heavier primitives (segment projection, body-frame
offsets, saturation, deadbands) already exist in the roll-assist and multi-axis
packages and are reused rather than reimplemented; this module adds only what is
specific to tracking a *line* with a *lookahead*.

Body frame is REP-103: ``+x`` forward, ``+y`` left, ``+yaw`` counter-clockwise.
"""
from __future__ import annotations

from math import atan2, hypot
from typing import Sequence, Tuple

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.trackers.roll_assist_follower.algorithm import (
    active_segment,
    body_offset_to_point,
    project_point_on_segment,
)

XY = Tuple[float, float]

__all__ = [
    "active_segment",
    "body_offset_to_point",
    "project_point_on_segment",
    "cross_track_error",
    "lookahead_point",
    "leg_heading",
    "bearing_error",
]


def cross_track_error(path, wp_idx, px, py, yaw):
    # type: (Sequence[XY], int, float, float, float) -> Tuple[float, float, XY]
    """Body-frame offset from the drone to its place on the trajectory.

    Args:
        path: The active (re-anchored) waypoints, at least two.
        wp_idx: Index of the waypoint currently being pursued.
        px: Drone x in the path frame (m).
        py: Drone y in the path frame (m).
        yaw: Drone heading (rad).

    Returns:
        ``(e_fwd, e_lat, foot)`` — the forward and left components of the vector
        from the drone to the closest point on the active segment, and that point.
        ``e_lat`` positive means the trajectory is to the drone's left, so a
        positive lateral (left) command closes it.
    """
    ax, ay, bx, by = active_segment(path, wp_idx)
    qx, qy, _ = project_point_on_segment(px, py, ax, ay, bx, by)
    e_fwd, e_lat = body_offset_to_point(px, py, yaw, qx, qy)
    return e_fwd, e_lat, (qx, qy)


def lookahead_point(path, wp_idx, px, py, distance):
    # type: (Sequence[XY], int, float, float, float) -> XY
    """A point ``distance`` further along the path than the drone's foot on it.

    This is the heading setpoint — aiming at it rather than at the next corner is
    what keeps a turn from being a stop-and-spin, and what stops the controller
    weaving when the pose is noisy. Walking the polyline (rather than taking the
    straight-line intersection of a circle) means the carrot follows the route
    round a corner instead of cutting across whatever is inside it.

    Args:
        path: The active waypoints, at least two.
        wp_idx: Index of the waypoint currently being pursued.
        px: Drone x in the path frame (m).
        py: Drone y in the path frame (m).
        distance: How far ahead to look along the path (m).

    Returns:
        The carrot point. Clamped to the final waypoint, so approaching the goal
        the carrot stops moving and the drone converges on it.
    """
    n = len(path)
    ax, ay, bx, by = active_segment(path, wp_idx)
    qx, qy, t = project_point_on_segment(px, py, ax, ay, bx, by)
    remaining = float(distance)

    seg_end = wp_idx if wp_idx >= 1 else 1
    if seg_end >= n:
        seg_end = n - 1

    # Walk out the rest of the current segment first, then whole segments.
    cx, cy = qx, qy
    ex, ey = bx - cx, by - cy
    step = hypot(ex, ey)
    while remaining > step:
        remaining -= step
        seg_end += 1
        if seg_end >= n:
            return float(path[n - 1][0]), float(path[n - 1][1])
        cx, cy = float(path[seg_end - 1][0]), float(path[seg_end - 1][1])
        ex = float(path[seg_end][0]) - cx
        ey = float(path[seg_end][1]) - cy
        step = hypot(ex, ey)
    if step <= 1e-9:
        return cx, cy
    frac = remaining / step
    return cx + ex * frac, cy + ey * frac


def leg_heading(path, wp_idx):
    # type: (Sequence[XY], int) -> float
    """Direction of the active segment (rad), i.e. the way the leg points."""
    ax, ay, bx, by = active_segment(path, wp_idx)
    return atan2(by - ay, bx - ax)


def bearing_error(px, py, yaw, tx, ty):
    # type: (float, float, float, float, float) -> float
    """Signed heading error from the drone's yaw to the bearing of ``(tx, ty)``.

    Returns 0 when the target is on top of the drone, so a degenerate carrot can
    never spin the drone on numerical noise.
    """
    dx, dy = tx - px, ty - py
    if hypot(dx, dy) < 1e-6:
        return 0.0
    return normalize_angle(atan2(dy, dx) - yaw)
