"""Resolve the gap to the plan along the direction of travel, and bound the
correction that pulls the aircraft *backward* along it.

Two jobs, both about the same axis.

**Splitting the error honestly.** Being late is benign; being sideways is what
hits walls, so the two are reported separately. The tempting direction to
measure against is the reference's own velocity, and that is a trap at the one
moment it matters most: past the end of a curve the reference velocity is
exactly zero, so an aircraft trailing a finished trajectory has no direction of
travel and its entire along-track lag is reported as cross-track -- the number
this split exists to keep honest, inverted precisely when the aircraft is
furthest behind. Measured on one flight, 2015 of 2035 ticks at a finished plan
reported a lag of exactly 0.0 and the whole gap as cross-track. So the direction
is an argument, and the caller passes the last direction the plan was actually
travelling when the current one is stopped.

:mod:`sparx_agency.core.control.reference.diagnosis` makes the same choice for
the trajectory-object branch of the stack, for the same reason, and the two
agree by construction: positive along-track means late, and a stationary
direction reports the whole gap as cross-track.

**Bounding the retard.** An aircraft *ahead* of its reference is the ordinary
consequence of flying faster than planned, and the planner will catch up on its
own. The position loop cannot tell that from being off the path: it pulls
straight at the reference, so it brakes the aircraft backward onto a point it
has already passed. :func:`limit_lead` shrinks only the component of the error
that points backward along travel, leaving cross-track -- the component that
actually keeps the aircraft off walls -- untouched.
"""
from __future__ import annotations

import math

_STATIONARY = 1e-6
"""Below this speed a direction of travel is not meaningful, in m/s."""


def unit(vector):
    # type: (tuple) -> tuple
    """The vector's direction, or None when it is too short to have one.

    Args:
        vector: Any 3-tuple.

    Returns:
        A unit 3-tuple, or None if the magnitude is below :data:`_STATIONARY`.
    """
    magnitude = math.sqrt(sum(float(v) * float(v) for v in vector))
    if magnitude <= _STATIONARY:
        return None
    return tuple(float(v) / magnitude for v in vector)


def split_error(error, direction):
    # type: (tuple, tuple) -> tuple
    """Resolve a position error into along-track lag and cross-track offset.

    Args:
        error: World ``(dx, dy, dz)`` from the aircraft **to** the reference.
        direction: The plan's direction of travel, as a world vector. Its
            magnitude is ignored. None, or a zero vector, means there is no
            direction of travel.

    Returns:
        ``(along_track_lag_m, cross_track_error_m)``, satisfying
        ``along**2 + cross**2 == |error|**2``. Positive ``along`` means the
        aircraft is *behind* the reference. With no direction the whole gap is
        reported as cross-track, which is the safe reading: an offset from a
        hover point is not lateness.
    """
    magnitude = math.sqrt(sum(component * component for component in error))
    heading = None if direction is None else unit(direction)
    if heading is None:
        return 0.0, magnitude
    along = sum(error[i] * heading[i] for i in range(3))
    cross_squared = max(magnitude * magnitude - along * along, 0.0)
    return along, math.sqrt(cross_squared)


def limit_lead(error, direction, max_lead_m):
    # type: (tuple, tuple, float) -> tuple
    """Bound how far *ahead* of the reference the position loop is allowed to see.

    An aircraft ahead of schedule has an error pointing backward along travel.
    Left alone the position loop turns that into a backward command, which on a
    forward-facing airframe is blind flight toward a point it has already
    cleared. Shrinking that one component bounds the brake without touching the
    cross-track pull that keeps the aircraft on the path.

    Args:
        error: World ``(dx, dy, dz)`` from the aircraft to the reference.
        direction: The plan's direction of travel. None leaves the error
            untouched -- with no direction there is no "ahead" to bound.
        max_lead_m: How much of a backward-pointing along-track component
            survives, metres. Must be >= 0; 0 removes the retard entirely.

    Returns:
        The error with its backward along-track component limited.

    Raises:
        ValueError: If ``max_lead_m`` is negative.
    """
    if max_lead_m < 0.0:
        raise ValueError("max_lead_m must be >= 0, got %r" % (max_lead_m,))
    heading = None if direction is None else unit(direction)
    if heading is None:
        return tuple(float(component) for component in error)
    along = sum(error[i] * heading[i] for i in range(3))
    if along >= -max_lead_m:
        return tuple(float(component) for component in error)
    # Only the excess is removed, so the error keeps pointing the same way.
    excess = along + max_lead_m
    return tuple(error[i] - excess * heading[i] for i in range(3))
