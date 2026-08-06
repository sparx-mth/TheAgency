"""Bound a requested acceleration to something the airframe can actually produce.

Three ceilings, applied in an order that matters:

1. **Vertical first.** Climb and descent acceleration are clamped on their own,
   because the vertical axis is the one that ends a flight against a floor.
2. **Then tilt.** The horizontal acceleration a multirotor can produce is
   ``total_vertical * tan(tilt)``, so the tilt ceiling is a limit on the *ratio*,
   not on the horizontal alone. Horizontal is scaled as a pair so a capped
   diagonal keeps its direction -- clipping per axis would turn a speed limit
   into a steering error.
3. **Then total thrust**, with **horizontal giving way**. When the airframe
   cannot deliver everything asked of it, holding altitude and cornering less is
   survivable; the reverse is not.

The tilt ceiling is the interesting one. It is not a comfort setting: it is what
keeps the commanded attitude in the region where the small-angle behaviour of
the layer below still holds, and it is what stops a large transient position
error from being answered with a 60-degree lean that unloads the rotors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sparx_agency.core.control.constants import GRAVITY_MPS2


@dataclass(frozen=True)
class AccelerationLimits:
    """Ceilings on a commanded acceleration, in m/s^2 unless stated.

    Attributes:
        max_tilt_rad: Largest angle between the commanded thrust axis and
            vertical. 35 degrees is a sane indoor ceiling: it allows about
            0.7 g of horizontal acceleration, far more than an exploration
            trajectory asks for, while staying well clear of the angles at
            which a multirotor starts trading altitude for translation.
        max_accel_up: Largest upward acceleration, excluding the acceleration
            needed to hold station against gravity.
        max_accel_down: Largest downward acceleration. Smaller than upward on
            purpose -- descending fast into ground effect is how a flight ends.
        min_specific_thrust: Floor on total specific thrust. Must stay above
            zero: a multirotor at zero thrust is in free fall and its attitude
            command means nothing.
        max_specific_thrust: Ceiling on total specific thrust, i.e. the
            airframe's thrust-to-weight times gravity, with margin. The Iris
            this flies has a thrust-to-weight near 2, so 1.6 g leaves headroom
            for the attitude loop underneath to still have authority.
    """

    max_tilt_rad: float = math.radians(35.0)
    max_accel_up: float = 4.0
    max_accel_down: float = 3.0
    min_specific_thrust: float = 0.35 * GRAVITY_MPS2
    max_specific_thrust: float = 1.6 * GRAVITY_MPS2

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the clamp relies on."""
        if not 0.0 < self.max_tilt_rad < math.pi / 2.0:
            raise ValueError("max_tilt_rad must be in (0, pi/2), got %r" % (self.max_tilt_rad,))
        for name in ("max_accel_up", "max_accel_down", "min_specific_thrust"):
            if getattr(self, name) <= 0.0:
                raise ValueError("%s must be > 0, got %r" % (name, getattr(self, name)))
        if self.max_specific_thrust <= self.min_specific_thrust:
            raise ValueError("max_specific_thrust must exceed min_specific_thrust")


def limit_acceleration(acceleration, limits):
    # type: (object, AccelerationLimits) -> tuple
    """Clamp a desired world acceleration to what the airframe can produce.

    Args:
        acceleration: Desired world ``(ax, ay, az)``, m/s^2, **excluding**
            gravity -- the acceleration the vehicle should have, not the thrust
            it should produce.
        limits: The ceilings to apply.

    Returns:
        ``(limited_acceleration, saturated)`` -- the clamped ``(3,)`` array and
        whether anything was actually reduced.
    """
    wanted = np.asarray(acceleration, dtype=float).reshape(3)
    saturated = False

    vertical = wanted[2]
    if vertical > limits.max_accel_up:
        vertical, saturated = limits.max_accel_up, True
    elif vertical < -limits.max_accel_down:
        vertical, saturated = -limits.max_accel_down, True

    # The thrust axis has to carry gravity as well as the wanted acceleration,
    # so it is the SUM that gets tilted and limited, not the acceleration alone.
    thrust_z = vertical + GRAVITY_MPS2
    thrust_z = max(thrust_z, limits.min_specific_thrust * math.cos(limits.max_tilt_rad))

    horizontal = wanted[:2].copy()
    magnitude = float(math.hypot(horizontal[0], horizontal[1]))
    allowed = thrust_z * math.tan(limits.max_tilt_rad)
    if magnitude > allowed and magnitude > 0.0:
        horizontal *= allowed / magnitude
        magnitude = allowed
        saturated = True

    # Total thrust last. Horizontal yields, because altitude is not negotiable.
    total = float(math.sqrt(magnitude * magnitude + thrust_z * thrust_z))
    if total > limits.max_specific_thrust:
        room = limits.max_specific_thrust ** 2 - thrust_z ** 2
        if room <= 0.0:
            # Even hovering exceeds the ceiling: keep the vertical axis and give
            # up horizontal entirely rather than pretending either can be met.
            horizontal *= 0.0
            thrust_z = min(thrust_z, limits.max_specific_thrust)
        else:
            horizontal *= math.sqrt(room) / magnitude
        saturated = True
    elif total < limits.min_specific_thrust:
        # Too little thrust to keep the attitude command meaningful. Raise the
        # vertical component rather than the horizontal one, so the direction of
        # travel survives.
        thrust_z = math.sqrt(max(limits.min_specific_thrust ** 2 - magnitude ** 2, 0.0))
        saturated = True

    limited = np.array([horizontal[0], horizontal[1], thrust_z - GRAVITY_MPS2], dtype=float)
    return limited, saturated
