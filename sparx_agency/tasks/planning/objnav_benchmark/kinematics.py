"""Hold a simulator to the action geometry its episode advertises, one action at a time.

The action converter plans every step with the episode's
:class:`DiscreteActionSpec`, so a simulator that executes another step, turn
or tilt corrupts every decision the same way, every step, and nothing else
notices. An AI2-THOR controller built without ``rotateStepDegrees=30`` turns
90 degrees, and the converter dithers left and right until the budget runs
out; habitat-sim configured without the benchmark's config turns 10 degrees
and tilts 15, which yields a normal-looking SR under another protocol and a
camera that stops at half the pitch it was asked for. A pose stream in the
wrong frame is the other quiet failure: a Y-up axis left unconverted, or a
handedness left mirrored (AI2-THOR is left-handed), keeps every path length
-- a sum of chords is the same in a rotated or mirrored frame, so
``cross_checks.py`` cannot see it -- while every map the agent builds from
those poses is rotated or mirrored.

So :func:`check_motion` sets each action's realised motion, from the pose
before it to the pose after, beside the motion the spec describes:

* TURN_LEFT / TURN_RIGHT turn by the spec's angle, left positive, in place;
* LOOK_UP / LOOK_DOWN tilt by the spec's angle, down positive (REP-103), or
  not at all -- a LOOK the simulator refused at its pitch limit -- in place;
* MOVE_FORWARD advances at most one step along the heading it started with (a
  shorter step is a blocked or clipped one), climbs no more than a stair
  allows, and neither turns nor tilts;
* STOP changes nothing.

The slack (:class:`KinematicTolerance`) is sized for real simulators, not for
rounding: AI2-THOR's documented actuation noise (a turn N(0, 0.5) deg, the
heading of a move N(0, 0.25) deg, its length N(0.001, 0.005) m) and Habitat's
partial no-sliding steps and stairs (up to 0.2 m up one step, slopes up to 45
degrees).

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.checks import is_length
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError

_A = DiscreteAction
#: The sign of each turn's yaw change: counter-clockwise from above is positive.
_TURN_SIGN = {_A.TURN_LEFT: 1.0, _A.TURN_RIGHT: -1.0}
#: The sign of each LOOK's pitch change: REP-103, looking down is positive.
_LOOK_SIGN = {_A.LOOK_DOWN: 1.0, _A.LOOK_UP: -1.0}

# The likely cause of each symptom, named where the symptom is detected.
_CAUSE_TURN = ("a simulator configured with another turn than the spec's %g "
               "deg (AI2-THOR's rotateStepDegrees, habitat-sim's turn_angle)")
_CAUSE_MIRROR = ("a mirrored handedness (AI2-THOR is left-handed: negate y "
                 "and yaw at the adapter)")
_CAUSE_TILT = ("a simulator configured with another tilt than the spec's %g "
               "deg (habitat-sim's tilt_angle)")
_CAUSE_PITCH_SIGN = ("a camera pitch signed the other way (REP-103: down is "
                     "positive; Habitat's sensor pitch is positive up)")
_CAUSE_STEP = ("a simulator configured with another step than the spec's %g "
               "m (habitat-sim's forward_step_size, AI2-THOR's gridSize), or "
               "positions in another unit than metres")
_CAUSE_COURSE = ("a pose left in the simulator's Y-up frame, a mirrored "
                 "handedness, or a simulator that slides along walls "
                 "(habitat-sim's allow_sliding; the ObjectNav protocol does "
                 "not slide)")
_CAUSE_CLIMB = ("a pose left in the simulator's Y-up frame (a step along the "
                "floor read as a change of height)")
_CAUSE_IN_PLACE = ("a pose left in the simulator's Y-up frame, or a position "
                   "read somewhere other than the agent's base")
_CAUSE_YAW = ("a yaw that is not the heading about world +z, or actions "
              "mapped to the wrong simulator actions")
_CAUSE_PITCH = ("a pitch read about the wrong axis, or actions mapped to the "
                "wrong simulator actions")


@dataclass(frozen=True)
class KinematicTolerance:
    """How far a simulator's realised motion may stray from its action spec before it is refused.

    Attributes:
        position_m: How far a turn, a LOOK or STOP may move the agent, metres.
        turn_deg: How far a turn may miss the spec's angle, degrees: six
            sigmas of AI2-THOR's turn noise. Five sigmas would refuse a sound
            simulator about once in 1.8 million turns -- a one-in-ten chance
            of aborting a full RoboTHOR sweep -- while six make that
            negligible and still refuse a 10- or 90-degree simulator against
            a 30-degree spec at its first turn.
        pitch_deg: How far a LOOK may miss its tilt (or no tilt at all), and
            how far any other action may change the pitch, degrees.
        yaw_deg: How far anything but a turn may change the heading, degrees:
            six sigmas of the heading noise AI2-THOR adds to a move, a blocked
            one included.
        forward_overshoot_m: How far past the spec's step a MOVE_FORWARD may
            advance, metres: AI2-THOR's step-length noise with room to spare.
        min_heading_check_m: A MOVE_FORWARD that advances less than this,
            metres, has no direction worth checking: a blocked step moves by
            noise alone.
        heading_deg: How far from the heading it started with a MOVE_FORWARD
            may advance, degrees.
        climb_m: How far a MOVE_FORWARD may change the height, metres, when it
            advances less than this; a longer one may climb its own length
            (Habitat's stairs: 0.2 m up a step, slopes up to 45 degrees).

    Raises:
        HarnessError: On a tolerance that is not a positive finite number.
    """

    position_m: float = 0.02
    turn_deg: float = 3.0
    pitch_deg: float = 0.5
    yaw_deg: float = 1.5
    forward_overshoot_m: float = 0.03
    min_heading_check_m: float = 0.05
    heading_deg: float = 10.0
    climb_m: float = 0.2

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not (is_length(value) and value > 0.0):
                raise HarnessError(
                    "KinematicTolerance.%s must be a positive finite number, "
                    "got %r" % (item.name, value))


def _wrapped(angle: float) -> float:
    """``angle`` in ``(-pi, pi]``; ``fmod`` first, since ``normalize_angle`` loops once per 2 pi."""
    return normalize_angle(math.fmod(angle, 2.0 * math.pi))


@dataclass(frozen=True)
class _Motion:
    """What one action did to the pose. Private: built only by :func:`_motion`."""

    moved_m: float
    advanced_m: float
    climbed_m: float
    course_rad: float
    turned_rad: float
    tilted_rad: float

    def describe(self) -> str:
        return ("moved %.3f m (%.3f m across the floor at %+.1f deg from its "
                "heading, %+.3f m up), turned %+.2f deg and tilted %+.2f deg"
                % (self.moved_m, self.advanced_m, math.degrees(self.course_rad),
                   self.climbed_m, math.degrees(self.turned_rad),
                   math.degrees(self.tilted_rad)))


def _motion(before: AgentPose, after: AgentPose) -> _Motion:
    dx, dy, dz = after.x - before.x, after.y - before.y, after.z - before.z
    advanced = math.hypot(dx, dy)
    course = _wrapped(math.atan2(dy, dx) - before.yaw) if advanced > 0.0 else 0.0
    return _Motion(moved_m=math.sqrt(dx * dx + dy * dy + dz * dz),
                   advanced_m=advanced, climbed_m=dz, course_rad=course,
                   turned_rad=_wrapped(after.yaw - before.yaw),
                   tilted_rad=after.camera_pitch - before.camera_pitch)


def _refuse(action: DiscreteAction, expected: str, motion: _Motion,
            cause: str) -> EnvContractError:
    return EnvContractError("%s should %s, but the agent %s; likely %s"
                            % (action.name, expected, motion.describe(), cause))


def _check_in_place(action, motion, tolerance, what: str) -> None:
    if motion.moved_m > tolerance.position_m:
        raise _refuse(action, "%s (move at most %.3f m)"
                      % (what, tolerance.position_m), motion, _CAUSE_IN_PLACE)


def _check_heading_kept(action, motion, tolerance) -> None:
    if abs(motion.turned_rad) > math.radians(tolerance.yaw_deg):
        raise _refuse(action, "keep the heading (within %.2f deg)"
                      % tolerance.yaw_deg, motion, _CAUSE_YAW)


def _check_pitch_kept(action, motion, tolerance) -> None:
    if abs(motion.tilted_rad) > math.radians(tolerance.pitch_deg):
        raise _refuse(action, "keep the camera pitch (within %.2f deg)"
                      % tolerance.pitch_deg, motion, _CAUSE_PITCH)


def _check_turn(action, motion, spec, tolerance) -> None:
    expected = _TURN_SIGN[action] * spec.turn_angle_rad
    slack = math.radians(tolerance.turn_deg)
    if abs(_wrapped(motion.turned_rad - expected)) > slack:
        mirrored = abs(_wrapped(motion.turned_rad + expected)) <= slack
        raise _refuse(action, "turn %+.1f deg (within %.1f deg)"
                      % (math.degrees(expected), tolerance.turn_deg), motion,
                      _CAUSE_MIRROR if mirrored
                      else _CAUSE_TURN % spec.turn_angle_deg)
    _check_in_place(action, motion, tolerance, "turn in place")
    _check_pitch_kept(action, motion, tolerance)


def _check_look(action, motion, spec, tolerance) -> None:
    expected = _LOOK_SIGN[action] * spec.tilt_angle_rad
    slack = math.radians(tolerance.pitch_deg)
    if not (abs(motion.tilted_rad) <= slack
            or abs(motion.tilted_rad - expected) <= slack):
        flipped = abs(motion.tilted_rad + expected) <= slack
        raise _refuse(action, "tilt %+.1f deg, or not at all at a pitch limit "
                      "(within %.2f deg)" % (math.degrees(expected),
                                             tolerance.pitch_deg), motion,
                      _CAUSE_PITCH_SIGN if flipped
                      else _CAUSE_TILT % spec.tilt_angle_deg)
    _check_in_place(action, motion, tolerance, "tilt in place")
    _check_heading_kept(action, motion, tolerance)


def _check_forward(action, motion, spec, tolerance) -> None:
    reach = spec.forward_step_m + tolerance.forward_overshoot_m
    if motion.advanced_m > reach:
        raise _refuse(action, "advance at most %.3f m" % reach, motion,
                      _CAUSE_STEP % spec.forward_step_m)
    if (motion.advanced_m >= tolerance.min_heading_check_m
            and abs(motion.course_rad) > math.radians(tolerance.heading_deg)):
        raise _refuse(action, "advance along its heading (within %.1f deg)"
                      % tolerance.heading_deg, motion, _CAUSE_COURSE)
    climb = max(tolerance.climb_m, motion.advanced_m)
    if abs(motion.climbed_m) > climb:
        raise _refuse(action, "change its height by at most %.3f m" % climb,
                      motion, _CAUSE_CLIMB)
    _check_heading_kept(action, motion, tolerance)
    _check_pitch_kept(action, motion, tolerance)


def check_motion(action: DiscreteAction, before: AgentPose, after: AgentPose,
                 spec: DiscreteActionSpec, tolerance: KinematicTolerance) -> None:
    """Refuse a realised motion that the episode's action spec cannot explain.

    Args:
        action: The action the simulator was sent.
        before: The agent's pose before it: the previous observation's.
        after: The pose after it.
        spec: The episode's action spec, which the motion must match.
        tolerance: How far the motion may stray from the spec's.

    Raises:
        TypeError: If an argument has the wrong type.
        EnvContractError: When the motion is not the one ``spec`` describes;
            the message names the action, the expected and the realised
            motion, and the likely cause.
    """
    for name, value, kind in (("action", action, DiscreteAction),
                              ("before", before, AgentPose),
                              ("after", after, AgentPose),
                              ("spec", spec, DiscreteActionSpec),
                              ("tolerance", tolerance, KinematicTolerance)):
        if not isinstance(value, kind):
            raise TypeError("%s must be a %s, got %r"
                            % (name, kind.__name__, value))
    motion = _motion(before, after)
    if action in _TURN_SIGN:
        _check_turn(action, motion, spec, tolerance)
    elif action in _LOOK_SIGN:
        _check_look(action, motion, spec, tolerance)
    elif action == _A.MOVE_FORWARD:
        _check_forward(action, motion, spec, tolerance)
    else:
        _check_in_place(action, motion, tolerance, "leave the pose unchanged")
        _check_heading_kept(action, motion, tolerance)
        _check_pitch_kept(action, motion, tolerance)
