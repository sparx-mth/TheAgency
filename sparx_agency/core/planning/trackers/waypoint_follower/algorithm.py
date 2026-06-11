"""Pure geometry helpers for the waypoint follower (ROS-free, math-only)."""
from __future__ import annotations

from math import hypot
from typing import List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import Pose2D

XY = Tuple[float, float]


def saturate(value: float, limit: float) -> float:
    """Clamp ``value`` to the symmetric interval ``[-limit, limit]``."""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def slew(target: float, current: float, max_step: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_step`` (rate limit)."""
    delta = target - current
    if delta > max_step:
        return current + max_step
    if delta < -max_step:
        return current - max_step
    return target


def yaw_brake_distance(wz: float, yaw_accel_limit: float, dt: float) -> float:
    """Angle still swept while decelerating ``wz`` to zero under the slew limit.

    Used to decide when to stop commanding yaw so the rotation coasts to a
    stop on the desired heading rather than overshooting.
    """
    return (wz * wz) / (2.0 * yaw_accel_limit) + abs(wz) * dt


def yaw_lead_offset(eyaw: float, yaw_lead_pct: float) -> float:
    """Absolute heading offset to stop short by, given the initial sweep.

    ``yaw_lead_pct`` is clamped to [0, 40]; above that it would cancel most of
    the rotation.
    """
    pct = min(max(yaw_lead_pct, 0.0), 40.0)
    return abs(eyaw) * (pct / 100.0)


def reanchor_path(
    points: Sequence[XY],
    pose: Optional[Pose2D],
    pos_radius: float,
) -> List[XY]:
    """Drop waypoints the robot has already passed, given its current pose.

    Projects the pose onto each path segment, finds the closest one and keeps
    the path from the next waypoint onward (plus one more if it is already
    within ``pos_radius``). With no pose or a degenerate path the points are
    returned unchanged. Never returns an empty list: at minimum the final
    waypoint is kept.
    """
    pts = list(points)
    if pose is None or len(pts) < 2:
        return pts

    cx, cy = pose.x, pose.y
    best_i, best_d = 0, float("inf")
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        ex, ey = bx - ax, by - ay
        seg_sq = ex * ex + ey * ey
        if seg_sq < 1e-9:
            px, py = ax, ay
        else:
            t = ((cx - ax) * ex + (cy - ay) * ey) / seg_sq
            t = min(max(t, 0.0), 1.0)
            px, py = ax + t * ex, ay + t * ey
        d = hypot(cx - px, cy - py)
        if d < best_d:
            best_d, best_i = d, i

    drop = best_i + 1
    if drop < len(pts):
        ex, ey = pts[drop]
        if hypot(ex - cx, ey - cy) < pos_radius:
            drop += 1
    if drop >= len(pts):
        return pts[-1:]
    return pts[drop:]
