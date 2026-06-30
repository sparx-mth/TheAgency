"""Pure axis-allocation math for the multi-axis follower (ROS-free, math-only).

These helpers turn a target waypoint and the current pose into per-axis velocity
commands. They are deliberately small and side-effect-free so the control law can
be reasoned about and unit-tested in isolation; the stateful glue (yaw hysteresis
latch, waypoint sequencing, slew memory) lives in ``follower.py``.

Body frame is REP-103: ``+forward (vx)``, ``+left (vy)``, ``+CCW (wz)``.
"""
from __future__ import annotations

from math import atan2, copysign, cos, hypot, sin
from typing import Tuple

from sparx_agency.core.common.types import normalize_angle


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


def body_error(
    px: float, py: float, yaw: float, tx: float, ty: float
) -> Tuple[float, float, float, float]:
    """Error to the target expressed in the body frame.

    Returns ``(e_fwd, e_lat, dist, eyaw)`` where ``e_fwd`` is the forward
    component, ``e_lat`` the left component, ``dist`` the range and ``eyaw`` the
    heading error (bearing minus current yaw, normalized). Note
    ``e_fwd = dist*cos(eyaw)`` and ``e_lat = dist*sin(eyaw)``, so the body angle
    to the target is exactly ``eyaw``.
    """
    dx = tx - px
    dy = ty - py
    dist = hypot(dx, dy)
    eyaw = normalize_angle(atan2(dy, dx) - yaw)
    e_fwd = dx * cos(yaw) + dy * sin(yaw)
    e_lat = -dx * sin(yaw) + dy * cos(yaw)
    return e_fwd, e_lat, dist, eyaw


def approach_speed(
    dist: float, pos_radius: float, slow_radius: float,
    cruise: float, arrive_min: float,
) -> float:
    """Desired translation speed for the current range (m/s).

    Cruises at ``cruise`` until ``slow_radius``, then ramps linearly down to
    ``arrive_min`` at ``pos_radius`` for a gentle, controlled arrival. Never
    drops below ``arrive_min`` while outside ``pos_radius`` (so the drone keeps
    moving rather than crawling to a noisy stop); returns 0 once captured. Passing
    ``slow_radius <= pos_radius`` disables the ramp (glide through at cruise).
    """
    if dist <= pos_radius:
        return 0.0
    if dist >= slow_radius or slow_radius <= pos_radius:
        return cruise
    frac = (dist - pos_radius) / (slow_radius - pos_radius)
    return arrive_min + (cruise - arrive_min) * frac


def alignment_gate(
    eyaw: float, cone: float, suppress_rad: float, floor: float
) -> float:
    """Translation-speed scale (0..1) that throttles travel when mis-pointed.

    Full speed within the travel ``cone``; ramps linearly down to ``floor`` at
    ``suppress_rad`` and stays at ``floor`` beyond it, so a grossly mis-aligned
    drone mostly yaws to face the target before charging toward it. Degenerate
    ``suppress_rad <= cone`` disables the throttle.
    """
    a = abs(eyaw)
    if a <= cone or suppress_rad <= cone:
        return 1.0
    if a >= suppress_rad:
        return floor
    frac = (suppress_rad - a) / (suppress_rad - cone)
    return floor + (1.0 - floor) * frac


def clamp_travel_angle(eyaw: float, cone: float) -> float:
    """Clamp the body-frame travel direction to ``[-cone, cone]`` (rad)."""
    if eyaw > cone:
        return cone
    if eyaw < -cone:
        return -cone
    return eyaw


def allocate_translation(
    speed: float, travel_angle: float, lateral_max: float
) -> Tuple[float, float]:
    """Split a translation speed along a body direction into ``(vx, vy)``.

    ``vy`` (lateral) is capped at ``lateral_max`` — the residual offset it cannot
    cover is what keeps the heading error large enough to engage yaw.
    """
    vx = speed * cos(travel_angle)
    vy = speed * sin(travel_angle)
    if vy > lateral_max:
        vy = lateral_max
    elif vy < -lateral_max:
        vy = -lateral_max
    return vx, vy


def yaw_engaged(prev: bool, eyaw: float, engage: float, release: float) -> bool:
    """Hysteresis latch for the yaw axis.

    Engages once ``|eyaw|`` exceeds ``engage`` and stays engaged until it falls
    below ``release`` (``release < engage``), so the yaw command does not chatter
    on noise around the threshold.
    """
    a = abs(eyaw)
    if prev:
        return a > release
    return a > engage


def yaw_setpoint(eyaw: float, engaged: bool, kp: float, yaw_rate_max: float) -> float:
    """Proportional yaw rate toward the target, saturated; 0 when not engaged."""
    if not engaged:
        return 0.0
    return saturate(kp * eyaw, yaw_rate_max)


def shape_axis(
    cmd: float, min_mag: float, release_frac: float, zero_eps: float
) -> float:
    """Apply the per-axis minimum-force deadband-with-snap.

    A command below ``max(zero_eps, release_frac*min_mag)`` is dropped to zero (too
    weak to be worth a pulse); between there and ``min_mag`` it is snapped up to
    ``min_mag`` (so it actually moves the platform); above ``min_mag`` it passes
    through. The caller is responsible for commanding exactly 0 on an axis it does
    not want to move, so this never injects motion the controller did not intend.
    """
    a = abs(cmd)
    drop = max(zero_eps, release_frac * min_mag)
    if a <= drop:
        return 0.0
    if a < min_mag:
        return copysign(min_mag, cmd)
    return cmd


def saturate_translation(
    vx: float, vy: float, limit: float
) -> Tuple[float, float]:
    """Scale a ``(vx, vy)`` vector down so its magnitude is at most ``limit``."""
    mag = hypot(vx, vy)
    if mag > limit and mag > 0.0:
        scale = limit / mag
        return vx * scale, vy * scale
    return vx, vy
