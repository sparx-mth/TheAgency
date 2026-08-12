"""Bound a commanded velocity, and how fast it may change.

Two clamps, both applied in the **world** frame and both before the command is
rotated into the body. That ordering is the whole content of this module and it
is not a matter of taste.

Saturating per body axis turns a speed limit into a steering error: an aircraft
asked for 1.4 m/s diagonally, clipped to 1.0 on each axis, still flies
diagonally -- but one asked for (1.4, 0.4) and clipped per axis flies 8 degrees
off where it was pointed, and the error grows with how saturated it is. So the
horizontal pair is scaled **together**, preserving direction, and only the
vertical axis is clipped on its own.

Slew limiting has the same trap with a longer fuse. The physical limit is on
``dv/dt`` of the aircraft, which is a world-frame quantity; a body-frame slew
limit applied to a rotating vehicle throttles a command that is not changing at
all -- a steady world velocity, seen from a yawing body, is a body velocity
going round in a circle at the yaw rate. The previous stack limited in the body
frame and paid for it in every turn. Both clamps here run in world, and the
single rotation into the body happens afterwards, where it is a pure change of
basis that cannot violate either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityLimits:
    """Ceilings on a commanded velocity and on its rate of change.

    Attributes:
        max_speed_xy: Largest horizontal speed, m/s. The airframe's own ceiling,
            not the mission's -- the plan's speed is respected separately, and
            relative to the plan, so this is only the backstop.
        max_speed_up: Largest climb rate, m/s.
        max_speed_down: Largest descent rate, m/s. Smaller than climb on
            purpose: descending fast into ground effect is how a flight ends.
        max_accel_xy: Largest horizontal rate of change of the *command*,
            m/s^2. Not the airframe's acceleration limit -- the autopilot
            underneath owns that -- but a bound on how large a step this
            controller may put into it, which is what keeps a replan from
            arriving as a jolt.
        max_accel_z: The same for the vertical axis.
        max_yaw_rate: Largest commanded heading rate, rad/s.
        max_yaw_accel: Largest rate of change of that command, rad/s^2.
    """

    max_speed_xy: float = 1.5
    max_speed_up: float = 1.0
    max_speed_down: float = 0.7
    max_accel_xy: float = 2.0
    max_accel_z: float = 2.0
    max_yaw_rate: float = math.radians(90.0)
    max_yaw_accel: float = math.radians(180.0)

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the clamps rely on."""
        for name in ("max_speed_xy", "max_speed_up", "max_speed_down",
                     "max_accel_xy", "max_accel_z", "max_yaw_rate", "max_yaw_accel"):
            if getattr(self, name) <= 0.0:
                raise ValueError("%s must be > 0, got %r" % (name, getattr(self, name)))


def limit_velocity(velocity, limits, max_speed_xy=None):
    # type: (object, VelocityLimits, float) -> tuple
    """Clamp a world velocity command, preserving horizontal direction.

    Args:
        velocity: Wanted world ``(vx, vy, vz)``, m/s.
        limits: The airframe's ceilings.
        max_speed_xy: A tighter horizontal ceiling for this tick, overriding
            ``limits.max_speed_xy``. This is where the plan's own speed goes:
            FALCON checks its trajectory against the map at the speed it
            planned, with a fixed clearance around it, and flying the same curve
            faster spends that clearance on stopping distance. None uses the
            airframe ceiling alone.

    Returns:
        ``(limited, saturated)`` -- the clamped ``(3,)`` array and whether
        anything was actually reduced.
    """
    wanted = np.asarray(velocity, dtype=float).reshape(3)
    ceiling = limits.max_speed_xy if max_speed_xy is None else min(limits.max_speed_xy,
                                                                  float(max_speed_xy))
    ceiling = max(ceiling, 0.0)
    saturated = False

    horizontal = wanted[:2].copy()
    speed = float(math.hypot(horizontal[0], horizontal[1]))
    if speed > ceiling:
        # Scaled as a pair, never clipped per axis: the direction of travel is
        # the part of this command that must survive.
        horizontal *= (ceiling / speed) if speed > 0.0 else 0.0
        saturated = True

    vertical = float(wanted[2])
    if vertical > limits.max_speed_up:
        vertical, saturated = limits.max_speed_up, True
    elif vertical < -limits.max_speed_down:
        vertical, saturated = -limits.max_speed_down, True

    return np.array([horizontal[0], horizontal[1], vertical], dtype=float), saturated


def slew_velocity(previous, wanted, limits, dt):
    # type: (object, object, VelocityLimits, float) -> tuple
    """Bound how far the command may move in one tick, in the world frame.

    Horizontal is limited as a **vector step**, so the direction of the change
    is preserved for the same reason the magnitude clamp preserves the direction
    of the command.

    Args:
        previous: Last tick's world command, ``(vx, vy, vz)``. None skips the
            limit, which is what the first tick after a reset wants.
        wanted: This tick's world command.
        limits: The ceilings on rate of change.
        dt: Seconds since the previous command.

    Returns:
        ``(limited, rate_limited)`` -- the slewed ``(3,)`` array and whether the
        step was actually shortened.
    """
    target = np.asarray(wanted, dtype=float).reshape(3)
    if previous is None or dt <= 0.0:
        return target, False
    last = np.asarray(previous, dtype=float).reshape(3)
    step = target - last
    limited = False

    horizontal = step[:2]
    allowed_xy = limits.max_accel_xy * float(dt)
    magnitude = float(math.hypot(horizontal[0], horizontal[1]))
    if magnitude > allowed_xy and magnitude > 0.0:
        horizontal = horizontal * (allowed_xy / magnitude)
        limited = True

    allowed_z = limits.max_accel_z * float(dt)
    vertical = max(-allowed_z, min(allowed_z, float(step[2])))
    if abs(vertical - float(step[2])) > 1e-12:
        limited = True

    return last + np.array([horizontal[0], horizontal[1], vertical], dtype=float), limited
