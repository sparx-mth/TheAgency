"""Split the gap between aircraft and plan into "late" and "sideways".

One displacement, decomposed once, against an explicitly supplied direction of
travel. Pure functions with no state and no configuration, so both control
backends can report the same numbers and a test can pin them exactly.

**Measured in space, deliberately.** The obvious way to find the schedule lag is
``elapsed - projected_time``, a difference of two times on the current curve,
and it is wrong in exactly the case the catch-up term exists for. FALCON does
not plan the next curve from the aircraft; it plans it from its **own previous
curve**, at ``now + replan_duration``. So a lagging aircraft is behind the new
curve's *start*, a curve has no negative time, the projection clamps at zero,
and the deficit disappears. Measured on a real flight: a true 1.30 m of lag read
as 0.03 m.

The distance to the *nearest* point on the curve is the other tempting
definition of cross-track, and it fails the same way: with the aircraft directly
behind the start of a straight curve it reports the whole gap as cross-track,
because the nearest point really is that far away. Honest about the curve,
useless as a measure of being off the path.

**Why the direction is an argument.** The tempting source for it is the sampled
reference velocity, and that is a trap at the one moment it matters most:
``BsplineTrajectory.sample`` zeroes every derivative past the end of the curve,
so an aircraft trailing a finished trajectory has a zero reference velocity,
no direction of travel, and its entire along-track lag is reported as
cross-track -- the number this module exists to keep honest, inverted precisely
when the aircraft is furthest behind. The caller passes the curve's own tangent
instead, which is defined everywhere.
"""
from __future__ import annotations

import numpy as np

_STATIONARY = 1e-6
"""Below this speed a direction of travel is not meaningful, in m/s."""


def decompose_error(offset, direction):
    # type: (object, object) -> tuple
    """Resolve a displacement into along-track and cross-track components.

    Args:
        offset: World ``(dx, dy, dz)`` from the aircraft **to** the point the
            plan says it should occupy. Metres.
        direction: The plan's direction of travel there, as a world vector. Need
            not be normalised; its magnitude is ignored. A zero vector means the
            reference is not moving.

    Returns:
        ``(gap_m, along_track_lag_m, cross_track_error_m)``, satisfying
        ``along**2 + cross**2 == gap**2``. Positive ``along`` means late. With a
        stationary reference the whole gap is reported as cross-track, which is
        the safe reading: an offset from a hover point is not being late.
    """
    displacement = np.asarray(offset, dtype=float).reshape(3)
    heading = np.asarray(direction, dtype=float).reshape(3)
    gap = float(np.linalg.norm(displacement))
    speed = float(np.linalg.norm(heading))
    if speed <= _STATIONARY:
        return gap, 0.0, gap
    unit = heading / speed
    along = float(np.dot(displacement, unit))
    cross = float(np.linalg.norm(displacement - along * unit))
    return gap, along, cross
