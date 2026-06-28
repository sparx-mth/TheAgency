"""Pure geometry helpers for the waypoint follower (ROS-free, math-only)."""
from __future__ import annotations

from math import atan2, cos, hypot, sin
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


def sweep_floor(yaw_rate: float, dt: float, min_ticks: int) -> float:
    """Smallest burst angle that reliably overcomes the yaw deadband (rad).

    A burst shorter than ``min_ticks`` control ticks may not break static
    friction/inertia, so the platform never actually turns. This is the floor a
    burst is clamped up to.
    """
    return max(0.0, float(min_ticks)) * abs(yaw_rate) * abs(dt)


def yaw_accept_floor(
    yaw_rate: float, dt: float, min_ticks: int, yaw_coast: float
) -> float:
    """Heading error below which alignment is accepted without rotating (rad).

    With a deadband floor (``sweep_floor``) and physical coast (``yaw_coast``),
    the smallest turn the platform can make is ``sweep_floor + yaw_coast``.
    Accepting anything within *half* of that is the tightest tolerance that
    cannot oscillate: a residual smaller than this can never be reduced by
    another burst (the burst would overshoot back past zero by more than it
    removed). Aligning tighter than this is physically impossible for the
    platform, so we stop trying — which is exactly the "58 deg is good enough
    for a 70 deg target" behaviour.
    """
    return 0.5 * (sweep_floor(yaw_rate, dt, min_ticks) + abs(yaw_coast))


def burst_target_angle(
    eyaw: float, yaw_coast: float, floor: float, ceil: float
) -> float:
    """Open-loop angle to *command* this burst so the platform lands on target.

    Aims ``yaw_coast`` short of the measured error (the coast fills the rest),
    clamped to ``[floor, ceil]`` so the burst always moves yet never commits to
    a huge open-loop turn before re-measuring. Always positive (magnitude).
    """
    return min(max(abs(eyaw) - abs(yaw_coast), floor), ceil)


def advance_gate(
    eyaw: float,
    dist: float,
    capture_tol: float,
    accept_floor: float,
    acquire_max: float,
) -> bool:
    """Whether the drone is aligned *enough* to start advancing (predictive).

    Returns True when either the heading error is within the un-improvable
    ``accept_floor``, or — looking ahead — driving straight on the current
    heading would still pass within ``capture_tol`` of the waypoint (the
    cross-track miss ``dist * sin|eyaw|``), provided the waypoint is ahead
    (``cos eyaw > 0``) and the error is within ``acquire_max``. The closer the
    waypoint, the larger the heading error tolerated — so the drone moves
    forward instead of fussing over an exact heading it cannot hold anyway.
    """
    e = abs(eyaw)
    if e <= accept_floor:
        return True
    if e <= acquire_max and cos(eyaw) > 0.0 and dist * abs(sin(eyaw)) <= capture_tol:
        return True
    return False


def circular_mean(angles: Sequence[float]) -> float:
    """Mean of angles via unit vectors (rad), robust to the +/-pi wrap.

    Used to fuse the heading samples collected during a YAW_SETTLE dwell into
    one estimate, so a single localization jump does not skew the decision.
    """
    if not angles:
        return 0.0
    s = sum(sin(a) for a in angles)
    c = sum(cos(a) for a in angles)
    if s == 0.0 and c == 0.0:
        return float(angles[-1])
    return atan2(s, c)


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
