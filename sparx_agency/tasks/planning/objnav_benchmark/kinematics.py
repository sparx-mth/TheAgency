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
degrees). Habitat's navmesh step filter is the one more thing a protocol may
have to size for (``reproject_m``): it starts every body action -- a turn
included -- from the agent's projection onto the mesh, so an agent it finds a
few centimetres off its mesh is carried back onto it by that action, a
placement the action did not ask for.

An environment that can *report* that projection -- habitat-sim's
``pathfinder.snap_point`` of the agent's pose is the very point its step
filter starts the next body action from -- passes it to :func:`check_motion`
as ``placed``, and the action is then judged the way the filter executed it:
from ``placed`` toward where the action aimed from the reported pose, with
the realised end anywhere on that segment (a blocked no-sliding step stops
at the wall) or at the pose itself (a step the filter refused outright moves
nothing). That explains a placement of any size exactly, where a scalar
``reproject_m`` large enough for it would also excuse a backward stride
(since 2026-10-27; see :func:`check_motion`).

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Optional

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
        settle_m: How much vertical motion the FIRST TRANSLATION of an
            episode may carry beyond the bounds above, metres: a published
            start that lies off the navmesh surface is settled onto it by
            the simulator on the first action that moves the agent (in
            habitat-sim 0.2.4 that is the first *body* action, a turn
            included -- ``turn_left`` is registered as one, so the step
            filter's ``try_step(start, start)`` places the agent; Gibson
            val starts lie up to 0.28 m below the mesh, Darden/000000
            0.10 m), which is the simulator placing the agent, not the
            action moving it. The excuse is additive -- that much of the
            height change is taken off before the climb and in-place
            bounds are judged, so a settle plus a step up a tread passes --
            and the horizontal bounds still apply unless ``settle_xy_m``
            says otherwise. 0 (the default) excuses nothing; the protocol
            that knows its data sets it.
        settle_xy_m: The horizontal counterpart of ``settle_m``, metres: how
            far ACROSS THE FLOOR the first action that moves the agent may
            carry it beyond the bounds above. A published start that lies
            outside the navmesh *horizontally* -- in a sliver the mesh's
            agent radius pruned -- is pulled onto the nearest mesh edge by
            habitat-sim's step filter on the first body action, a turn
            included (``try_step(start, start)`` returns the nearest point
            of the mesh, not ``start``): Gibson val Wiconisco/000118 lies
            0.033 m outside and its warm-up TURN_LEFT moved it that far,
            which refused the episode -- and, through the runner, the whole
            1,000-episode run, at episode 919, on 2026-10-10. Additive like
            ``settle_m``: that much of the horizontal displacement is taken
            off (toward zero, along its own direction) before the in-place,
            step and heading bounds are judged, once, on the first action
            that moves the agent. 0 (the default) excuses nothing.
        reproject_m: How far the navmesh step filter may carry the agent on
            ANY turn or MOVE_FORWARD beyond the bounds above, metres (since
            2026-10-26). habitat-sim 0.2.4's ``try_step`` starts every body
            action from ``findNearestPoly(start)`` -- the agent's projection
            onto the mesh -- and returns a point on the mesh, so an agent it
            finds a few centimetres off its mesh (HM3D's recomputed mesh,
            its stairs) is pulled back onto it by the next body action, a
            turn included: the 2026-10-22 v1 sweep lost 00878-XB4GS9ShBRE
            at action 156 of its second episode to a TURN_LEFT that "moved"
            0.071 m, and 00873-bxsVRursffK at action 315 of its 69th to a
            MOVE_FORWARD that went 0.094 m across the floor 53 degrees off
            its heading and 0.057 m up -- two actions in some half a million,
            each ending a 99-episode scene. On a turn that much of the
            displacement (3-D, along its own direction) is taken off before
            the in-place bound; on a MOVE_FORWARD that much of what lies
            ACROSS the heading, behind the start, or past the step's reach,
            each separately, before the step, heading and climb bounds --
            the step itself is still judged along the heading it began with.
            LOOKs and STOP, which habitat executes on the sensors alone, get
            none of it. 0 (the default) excuses nothing; the protocol that
            knows its simulator sets it.

    Raises:
        HarnessError: On a tolerance that is not a positive finite number
            (``settle_m``, ``settle_xy_m`` and ``reproject_m`` may be 0).
    """

    position_m: float = 0.02
    turn_deg: float = 3.0
    pitch_deg: float = 0.5
    yaw_deg: float = 1.5
    forward_overshoot_m: float = 0.03
    min_heading_check_m: float = 0.05
    heading_deg: float = 10.0
    climb_m: float = 0.2
    settle_m: float = 0.0
    settle_xy_m: float = 0.0
    reproject_m: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            lowest_allowed = 0.0 if item.name in ("settle_m", "settle_xy_m", "reproject_m") else None
            if not (is_length(value) and (value > 0.0 or value == lowest_allowed)):
                raise HarnessError(
                    "KinematicTolerance.%s must be a %sfinite number, "
                    "got %r" % (item.name, "non-negative " if lowest_allowed == 0.0 else "positive ", value))


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


_CAUSE_OFF_SEGMENT = ("a simulator that slides along walls (habitat-sim's allow_sliding; the ObjectNav "
                      "protocol does not slide), a mirrored handedness, or a placement the environment "
                      "misreported: a no-sliding step filter ends every body action on the straight line "
                      "from where it placed the agent to where the action aimed")


def _check_from_placement(action: DiscreteAction, before: AgentPose, placed: AgentPose, after: AgentPose,
                          spec: DiscreteActionSpec, tolerance: KinematicTolerance) -> None:
    """Judge a body action the way a navmesh step filter executed it: from ``placed`` toward where it aimed.

    habitat-sim's ``tryStep`` projects the agent onto the mesh first (``placed``), then walks the straight
    line from there to the point the action asked for from the *reported* pose -- ``before`` plus a stride
    for MOVE_FORWARD, ``before`` itself for a turn -- and, with sliding off, stops where that line meets a
    wall; or it returns the unprojected pose untouched when the two projections are not connected or no
    path exists. So the realised end either lies on that segment -- its advance along it at most the
    segment's length (plus the overshoot slack), its course the segment's (within ``heading_deg``), its
    height judged from the mesh point the filter started at, not from a pose that may have been left off
    the surface -- or is the pose itself, a blocked step. The yaw and pitch bounds of the action are the
    ordinary ones either way.
    """
    stayed = _motion(before, after)
    if stayed.moved_m <= tolerance.position_m:
        # The filter returned the pose it was given: the body did not move, the step was blocked where it stood.
        if action in _TURN_SIGN:
            _check_turn(action, stayed, spec, tolerance)
        else:
            _check_heading_kept(action, stayed, tolerance)
            _check_pitch_kept(action, stayed, tolerance)
        return
    aim_x, aim_y = before.x, before.y
    if action == _A.MOVE_FORWARD:
        aim_x += spec.forward_step_m * math.cos(before.yaw)
        aim_y += spec.forward_step_m * math.sin(before.yaw)
    ex, ey = aim_x - placed.x, aim_y - placed.y
    reach = math.hypot(ex, ey)
    # The motion from the filter's start, its course measured against the segment's direction.
    heading = math.atan2(ey, ex) if reach > 0.0 else before.yaw
    motion = _motion(replace(placed, yaw=heading, camera_pitch=before.camera_pitch), after)
    motion = replace(motion, turned_rad=_wrapped(after.yaw - before.yaw))
    placement = math.hypot(placed.x - before.x, placed.y - before.y)
    where = " (the step filter placed it %.3f m from its pose first)" % placement
    slack = tolerance.forward_overshoot_m if action == _A.MOVE_FORWARD else tolerance.position_m
    if motion.advanced_m > reach + slack:
        raise _refuse(action, "advance at most %.3f m toward where it aimed%s" % (reach + slack, where), motion,
                      _CAUSE_STEP % spec.forward_step_m if action == _A.MOVE_FORWARD else _CAUSE_IN_PLACE)
    if (motion.advanced_m >= tolerance.min_heading_check_m
            and abs(motion.course_rad) > math.radians(tolerance.heading_deg)):
        raise _refuse(action, "advance toward where it aimed (within %.1f deg)%s" % (tolerance.heading_deg, where),
                      motion, _CAUSE_OFF_SEGMENT)
    climb = max(tolerance.climb_m, motion.advanced_m) if action == _A.MOVE_FORWARD else tolerance.climb_m
    if abs(motion.climbed_m) > climb:
        raise _refuse(action, "change its height by at most %.3f m from the mesh%s" % (climb, where), motion,
                      _CAUSE_CLIMB)
    if action in _TURN_SIGN:
        expected = _TURN_SIGN[action] * spec.turn_angle_rad
        if abs(_wrapped(motion.turned_rad - expected)) > math.radians(tolerance.turn_deg):
            mirrored = abs(_wrapped(motion.turned_rad + expected)) <= math.radians(tolerance.turn_deg)
            raise _refuse(action, "turn %+.1f deg (within %.1f deg)" % (math.degrees(expected), tolerance.turn_deg),
                          motion, _CAUSE_MIRROR if mirrored else _CAUSE_TURN % spec.turn_angle_deg)
    else:
        _check_heading_kept(action, motion, tolerance)
    _check_pitch_kept(action, motion, tolerance)


def _without_reprojection(action: DiscreteAction, before: AgentPose, after: AgentPose,
                          spec: DiscreteActionSpec, tolerance: KinematicTolerance) -> AgentPose:
    """``after`` with up to ``tolerance.reproject_m`` of the step filter's placement taken off (see :class:`KinematicTolerance`)."""
    slack = tolerance.reproject_m
    dx, dy, dz = after.x - before.x, after.y - before.y, after.z - before.z
    if action == _A.MOVE_FORWARD:
        # The step is judged along and across the heading it began with: the placement is what lies
        # across it, behind its start, or past its reach -- each taken off separately, up to the bound.
        cos_yaw, sin_yaw = math.cos(before.yaw), math.sin(before.yaw)
        along, across = dx * cos_yaw + dy * sin_yaw, dy * cos_yaw - dx * sin_yaw
        across = math.copysign(max(0.0, abs(across) - slack), across)
        if along < 0.0:
            along = min(0.0, along + slack)
        reach = spec.forward_step_m + tolerance.forward_overshoot_m
        if along > reach:
            along = max(reach, along - slack)
        return replace(after, x=before.x + along * cos_yaw - across * sin_yaw,
                       y=before.y + along * sin_yaw + across * cos_yaw)
    # A turn asked for no displacement: whatever there is, up to the bound, is the placement.
    moved = math.sqrt(dx * dx + dy * dy + dz * dz)
    if moved <= 0.0:
        return after
    kept = max(0.0, moved - slack) / moved
    return replace(after, x=before.x + dx * kept, y=before.y + dy * kept, z=before.z + dz * kept)


def check_motion(action: DiscreteAction, before: AgentPose, after: AgentPose,
                 spec: DiscreteActionSpec, tolerance: KinematicTolerance,
                 first: bool = False, placed: Optional[AgentPose] = None) -> None:
    """Refuse a realised motion that the episode's action spec cannot explain.

    Args:
        action: The action the simulator was sent.
        before: The agent's pose before it: the previous observation's.
        after: The pose after it.
        spec: The episode's action spec, which the motion must match.
        tolerance: How far the motion may stray from the spec's.
        first: Whether the agent has not moved yet this episode, so this
            action may be its first translation, on which up to
            ``tolerance.settle_m`` of vertical and ``tolerance.settle_xy_m``
            of horizontal motion is the simulator placing the agent on the
            mesh and is taken off before the bounds are judged (see
            :attr:`KinematicTolerance.settle_m` and
            :attr:`KinematicTolerance.settle_xy_m`). On any turn or
            MOVE_FORWARD up to ``tolerance.reproject_m`` of the navmesh step
            filter's placement is taken off likewise
            (:attr:`KinematicTolerance.reproject_m`).
        placed: Where the simulator's step filter started the action, when
            the environment can report it (since 2026-10-27): ``before``
            with its position projected onto the navmesh, the way
            habitat-sim's ``tryStep`` begins every body action
            (``pathfinder.snap_point`` of the agent's pose is that very
            point). When it differs from ``before``, a turn or MOVE_FORWARD
            is judged from ``placed`` toward where the action aimed from
            ``before`` -- a stride ahead, or ``before`` itself for a turn --
            with the realised end anywhere on that segment, as a no-sliding
            filter leaves it, or at ``before`` itself (the filter returned
            the pose untouched: a step blocked where it stood), and its
            height judged from the mesh point the filter started at;
            ``reproject_m`` is then not needed and not applied. LOOKs and
            STOP, which habitat executes on the sensors alone, ignore it.
            None, or a ``placed`` at ``before``'s position, is the ordinary
            check.

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
    if placed is not None and not isinstance(placed, AgentPose):
        raise TypeError("placed must be an AgentPose or None, got %r" % (placed,))
    if (placed is not None and (action in _TURN_SIGN or action == _A.MOVE_FORWARD)
            and (placed.x, placed.y, placed.z) != (before.x, before.y, before.z)):
        _check_from_placement(action, before, placed, after, spec, tolerance)
        return
    if first and tolerance.settle_m > 0.0:
        # The settle is the simulator's, not the action's: take it off, keep the rest.
        dz = after.z - before.z
        excused = math.copysign(min(abs(dz), tolerance.settle_m), dz)
        after = replace(after, z=after.z - excused)
    if first and tolerance.settle_xy_m > 0.0:
        # Likewise across the floor: the pull onto the mesh edge, along its own direction, up to the bound.
        dx, dy = after.x - before.x, after.y - before.y
        across = math.hypot(dx, dy)
        if across > 0.0:
            scale = min(across, tolerance.settle_xy_m) / across
            after = replace(after, x=after.x - dx * scale, y=after.y - dy * scale)
    if tolerance.reproject_m > 0.0 and (action in _TURN_SIGN or action == _A.MOVE_FORWARD):
        # The step filter's placement is the simulator's too, on any body action: take it off, keep the rest.
        after = _without_reprojection(action, before, after, spec, tolerance)
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
