"""Pure functions of the visual-servo control law (stateless, numpy scalar math).

Each maps an image/geometry error to one body-velocity component with an explicit
saturation. The stateful glue (mode/hysteresis, EMA smoothing, limit fusion) lives
in :mod:`sparx_agency.core.planning.visual_servo.controller`. Keeping these pure
makes the sign conventions and saturations unit-testable in isolation.

Sign conventions: see :mod:`sparx_agency.core.planning.visual_servo.params`.
"""
from __future__ import annotations


def saturate(v: float, lim: float) -> float:
    """Clamp ``v`` to ``[-lim, lim]`` (``lim`` assumed >= 0)."""
    if v > lim:
        return lim
    if v < -lim:
        return -lim
    return float(v)


def clamp01(v: float) -> float:
    """Clamp ``v`` to ``[0, 1]``."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def yaw_command(ox: float, kp: float, max_yaw_rate: float, deadband: float) -> float:
    """Yaw rate to centre a target at normalised x-offset ``ox``.

    Target right of centre (``ox > 0``) => yaw CW (negative). Inside ``deadband``
    the yaw is zero (anti-jitter).
    """
    if abs(ox) <= deadband:
        return 0.0
    return saturate(-kp * ox, max_yaw_rate)


def lateral_command(ox: float, kp: float, max_speed: float, deadband: float) -> float:
    """Lateral (crab) speed to centre a target at ``ox``.

    Target right of centre (``ox > 0``) => crab right (``vy < 0``, since ``+vy`` is
    left). Zero inside ``deadband``.
    """
    if abs(ox) <= deadband:
        return 0.0
    return saturate(-kp * ox, max_speed)


def vertical_command(oy: float, kp: float, max_speed: float, deadband: float) -> float:
    """Vertical speed to centre a target at normalised y-offset ``oy``.

    Target above centre (``oy < 0``, image ``+y`` is down) => climb (``vz > 0``).
    Zero inside ``deadband``.
    """
    if abs(oy) <= deadband:
        return 0.0
    return saturate(-kp * oy, max_speed)


def centering_gain(ox: float, advance_offset_max: float) -> float:
    """Forward-speed scale in ``[0, 1]`` from how centred the target is.

    1 when perfectly centred, ramping linearly to 0 at ``|ox| >= advance_offset_max``
    so the drone does not charge forward while the target is far off-axis (which
    would swing it out of frame).
    """
    if advance_offset_max <= 0.0:
        return 0.0 if abs(ox) > 0.0 else 1.0
    return clamp01(1.0 - abs(ox) / advance_offset_max)


def forward_from_range(range_m: float, target_range_m: float, slowdown_range_m: float,
                       vx_max: float, kp: float) -> float:
    """Forward speed from a metric range-to-target.

    Full speed beyond ``slowdown_range_m``; a P-ramp (gain ``kp``) inside it that
    reaches 0 at ``target_range_m``; never negative (we don't back up).
    """
    if range_m <= target_range_m:
        return 0.0
    if range_m >= slowdown_range_m:
        return float(vx_max)
    return clamp01(kp * (range_m - target_range_m)) * vx_max


def forward_from_area(area_frac: float, target_area_frac: float,
                      slowdown_area_frac: float, vx_max: float) -> float:
    """Forward speed from the box area fraction (proximity proxy, no depth).

    Full speed below ``slowdown_area_frac``; a linear ramp to 0 as the area grows
    to ``target_area_frac``; 0 at/above the target (close enough).
    """
    if area_frac >= target_area_frac:
        return 0.0
    if area_frac < slowdown_area_frac:
        return float(vx_max)
    span = max(1e-6, target_area_frac - slowdown_area_frac)
    return vx_max * clamp01((target_area_frac - area_frac) / span)


def ema(prev: float, target: float, blend: float) -> float:
    """Exponential-moving-average step: ``prev + blend*(target - prev)``."""
    return float(prev + blend * (target - prev))
