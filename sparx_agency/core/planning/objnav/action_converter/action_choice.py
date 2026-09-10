"""Which one action closes a heading or a pitch error: the converter's turn and tilt choices.

Every function maps numbers to a number or to one action, so each of the
converter's turn and tilt decisions is testable as one geometric fact. Two
quiet failures are designed out here rather than patched in the converter:

* **Dithering.** A heading dead band narrower than half a turn makes the agent
  turn left, overshoot and turn right forever. :func:`turn_action`'s is half a
  turn, so under exact execution one turn always lands inside it. With
  AI2-THOR's turn noise (each turn off by N(0, 0.5) degrees) a turn from just
  outside the band can overshoot past its far edge and be undone at once --
  occasionally, never a livelock: the next true pose decides afresh.
* **Refused LOOKs.** AI2-THOR fails a LOOK past its pitch limits: the step is
  spent and the camera does not move. :func:`pitch_action` never asks for one,
  judging with :func:`pitch_within_limits`, the predicate
  ``transition.apply_action`` executes with.
* **Legal LOOKs refused over float noise.** A simulator reports its camera
  angle with float32 noise, so a pitch on the tilt lattice can read a hair
  past it. The limits are judged with :data:`PITCH_LIMIT_EPS_RAD`, not with
  the dead band's last-bit slack.

Python 3.8 syntax; numpy arrives only through ``core.common.types``.
"""
from __future__ import annotations

import math
from typing import Optional

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.checks import (
    require_finite,
    require_positive,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction

#: Slack on the dead-band comparisons, radians. ``radians(30) / 2`` and the
#: error left after one 30-degree turn can differ in the last bit; without the
#: slack an error of exactly half a turn could land on either side of the dead
#: band.
ANGLE_EPS_RAD = 1e-9

#: Slack on the pitch limits, radians (about 0.006 degrees). Far above float32
#: noise, far below one tilt. After LOOK_UP from 30 degrees, Unity may report
#: the horizon as -1e-7 degrees; judged with :data:`ANGLE_EPS_RAD`, the next
#: LOOK_UP would land 1e-7 degrees past the -30-degree limit and be refused, and
#: the converter would idle with the camera short of its commanded pitch --
#: while Unity, whose check rounds the angle, would have executed it.
PITCH_LIMIT_EPS_RAD = 1e-4


def heading_error(x: float, y: float, yaw: float, tx: float, ty: float
                  ) -> float:
    """Signed turn from heading ``yaw`` at ``(x, y)`` to face ``(tx, ty)``.

    Args:
        x: Agent position east, metres.
        y: Agent position north, metres.
        yaw: Agent heading, radians, counter-clockwise from world ``+x``.
        tx: Target east, metres.
        ty: Target north, metres.

    Returns:
        Radians in ``(-pi, pi]``, positive to the left. A target straight
        behind reads as ``+pi``, so it is turned toward on the left.

    Raises:
        ObjNavError: On a non-finite argument.
    """
    for name, value in (("x", x), ("y", y), ("yaw", yaw), ("tx", tx),
                        ("ty", ty)):
        require_finite(name, value)
    return normalize_angle(math.atan2(ty - y, tx - x) - yaw)


def turn_action(error_rad: float, turn_angle_rad: float
                ) -> Optional[DiscreteAction]:
    """The one turn that reduces a heading error, or None inside the dead band.

    The dead band is half a turn (plus :data:`ANGLE_EPS_RAD`): under exact
    execution one turn moves an error just outside it to just inside it on
    the other side, so the next call returns None rather than the opposite
    turn. (A noisy turn can overshoot the far edge; see the module notes.)

    Args:
        error_rad: Signed heading error, radians, positive to the left,
            wrapped to ``[-pi, pi]``.
        turn_angle_rad: What one TURN rotates, radians.

    Returns:
        None when ``|error| <= turn / 2``; else TURN_LEFT for a positive error
        and TURN_RIGHT for a negative one.

    Raises:
        ObjNavError: On a non-finite or unwrapped error (it would turn the
            long way round) or a non-positive turn angle.
    """
    require_finite("error_rad", error_rad)
    require_positive("turn_angle_rad", turn_angle_rad)
    if abs(error_rad) > math.pi + ANGLE_EPS_RAD:
        raise ObjNavError(
            "error_rad=%r is not wrapped to [-pi, pi]; wrap it with "
            "normalize_angle, or the agent turns the long way round"
            % (error_rad,))
    if abs(error_rad) <= turn_angle_rad / 2.0 + ANGLE_EPS_RAD:
        return None
    if error_rad > 0:
        return DiscreteAction.TURN_LEFT
    return DiscreteAction.TURN_RIGHT


def _require_limits(min_pitch: Optional[float],
                    max_pitch: Optional[float]) -> None:
    """Raise unless each limit is None or finite and the range is not inverted."""
    for name, value in (("min_pitch", min_pitch), ("max_pitch", max_pitch)):
        if value is not None:
            require_finite(name, value)
    if (min_pitch is not None and max_pitch is not None
            and min_pitch >= max_pitch):
        raise ObjNavError(
            "pitch range is inverted: min_pitch=%r is not below max_pitch=%r "
            "(REP-103: up is negative)" % (min_pitch, max_pitch))


def pitch_within_limits(pitch: float, min_pitch: Optional[float] = None,
                        max_pitch: Optional[float] = None) -> bool:
    """Whether a camera pitch lies inside the simulator's limits.

    Args:
        pitch: Camera pitch, radians, REP-103 (positive looks down).
        min_pitch: The most upward pitch allowed, or None for no limit.
        max_pitch: The most downward pitch allowed, or None for no limit.

    Returns:
        Whether ``min_pitch <= pitch <= max_pitch`` within
        :data:`PITCH_LIMIT_EPS_RAD`, each bound applying only when set.

    Raises:
        ObjNavError: On a non-finite pitch or limit, or an inverted range.
    """
    require_finite("pitch", pitch)
    _require_limits(min_pitch, max_pitch)
    if min_pitch is not None and pitch < min_pitch - PITCH_LIMIT_EPS_RAD:
        return False
    return max_pitch is None or pitch <= max_pitch + PITCH_LIMIT_EPS_RAD


def pitch_action(current: float, desired: float, tilt_rad: float,
                 min_pitch: Optional[float] = None,
                 max_pitch: Optional[float] = None
                 ) -> Optional[DiscreteAction]:
    """The one LOOK that brings the camera toward ``desired``, or None.

    A desired pitch past a limit is aimed at the limit instead -- as far as
    the camera goes; asking for "all the way down" is not an error. A LOOK the
    simulator would refuse is never returned.

    Args:
        current: Camera pitch now, radians, REP-103 (positive looks down).
        desired: Commanded camera pitch, radians, REP-103.
        tilt_rad: What one LOOK tilts, radians.
        min_pitch: The most upward pitch allowed, or None for no limit.
        max_pitch: The most downward pitch allowed, or None for no limit.

    Returns:
        None when the limited desired pitch is within half a tilt of
        ``current`` or the one LOOK toward it would leave the limits; else
        LOOK_DOWN to increase the pitch and LOOK_UP to decrease it.

    Raises:
        ObjNavError: On a non-finite angle, a non-positive tilt, or an
            inverted pitch range.
    """
    require_finite("current", current)
    require_finite("desired", desired)
    require_positive("tilt_rad", tilt_rad)
    _require_limits(min_pitch, max_pitch)
    target = desired
    if max_pitch is not None:
        target = min(target, max_pitch)
    if min_pitch is not None:
        target = max(target, min_pitch)
    delta = target - current
    if abs(delta) <= tilt_rad / 2.0 + ANGLE_EPS_RAD:
        return None
    if delta > 0:
        action, after = DiscreteAction.LOOK_DOWN, current + tilt_rad
    else:
        action, after = DiscreteAction.LOOK_UP, current - tilt_rad
    if not pitch_within_limits(after, min_pitch, max_pitch):
        return None
    return action
