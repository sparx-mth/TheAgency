"""Pure cross-track math for the roll-assist corrector (ROS-free, math-only).

These helpers turn a path and a pose into the body-frame offset the drone must
close to sit back on its trajectory, plus the small shaping primitives the
control law needs. They are deliberately side-effect-free so the correction law
can be reasoned about and unit-tested in isolation; the stateful glue (slew
memory) lives in ``corrector.py``.

Body frame is REP-103: ``+forward (vx)``, ``+left (vy)``, ``+CCW (wz)``.
"""
from __future__ import annotations

from math import copysign, cos, hypot, sin
from typing import Sequence, Tuple

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


def deadband(value: float, width: float) -> float:
    """Continuous deadband: 0 within ``+-width``, else the excess past it.

    Subtracting the width (rather than a hard cut) keeps the response continuous
    across the threshold, so the correction eases in from zero instead of
    snapping on. ``width <= 0`` passes the value through unchanged.
    """
    if width <= 0.0:
        return value
    if value > width:
        return value - width
    if value < -width:
        return value + width
    return 0.0


def shape_axis(cmd: float, min_mag: float, release_frac: float, zero_eps: float) -> float:
    """Per-axis minimum-force deadband-with-snap (mirrors the multi-axis follower).

    A command below ``max(zero_eps, release_frac*min_mag)`` is dropped to zero
    (too weak to be worth a pulse the motors would ignore); between there and
    ``min_mag`` it is snapped up to ``min_mag`` so it actually moves the platform;
    above ``min_mag`` it passes through. Shaping ``0 -> 0``.
    """
    a = abs(cmd)
    drop = max(zero_eps, release_frac * min_mag)
    if a <= drop:
        return 0.0
    if a < min_mag:
        return copysign(min_mag, cmd)
    return cmd


def active_segment(path: Sequence[XY], wp_idx: int) -> Tuple[float, float, float, float]:
    """Endpoints ``(ax, ay, bx, by)`` of the segment the drone is tracking.

    The trajectory the drone should sit on for the current leg is the polyline
    segment that *ends* at the active waypoint: ``[path[wp_idx-1], path[wp_idx]]``.
    For the first leg (``wp_idx == 0``) there is no predecessor, so the first real
    segment ``[path[0], path[1]]`` is used instead. Indices are clamped so a
    stale ``wp_idx`` can never index out of range. Requires ``len(path) >= 2``.
    """
    n = len(path)
    end = wp_idx if wp_idx >= 1 else 1
    if end >= n:
        end = n - 1
    if end < 1:
        end = 1
    ax, ay = path[end - 1]
    bx, by = path[end]
    return float(ax), float(ay), float(bx), float(by)


def project_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float]:
    """Closest point on segment ``[A, B]`` to ``P``, clamped to the endpoints.

    Returns ``(qx, qy, t)`` where ``t`` in ``[0, 1]`` is the normalized position
    along the segment (0 at ``A``, 1 at ``B``). A degenerate (zero-length)
    segment returns ``A`` at ``t = 0``.
    """
    ex, ey = bx - ax, by - ay
    seg_sq = ex * ex + ey * ey
    if seg_sq < 1e-12:
        return ax, ay, 0.0
    t = ((px - ax) * ex + (py - ay) * ey) / seg_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return ax + t * ex, ay + t * ey, t


def body_offset_to_point(
    px: float, py: float, yaw: float, qx: float, qy: float
) -> Tuple[float, float]:
    """Vector from ``P`` to ``Q`` expressed in the body frame.

    Returns ``(e_fwd, e_lat)``: the forward (``+x``) and left (``+y``) components
    of ``Q - P`` after rotating world into the body frame by ``-yaw``. When the
    drone is flying along the trajectory, ``Q`` is its perpendicular foot on the
    line, so ``e_fwd`` is near zero and ``e_lat`` is the signed cross-track drift
    (positive when the trajectory is to the drone's left).
    """
    dx = qx - px
    dy = qy - py
    e_fwd = dx * cos(yaw) + dy * sin(yaw)
    e_lat = -dx * sin(yaw) + dy * cos(yaw)
    return e_fwd, e_lat
