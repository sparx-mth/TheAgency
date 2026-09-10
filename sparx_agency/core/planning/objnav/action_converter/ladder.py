"""The converter's decision ladder: the one action a step takes, given where the agent is on its path.

Stateless. :class:`~sparx_agency.core.planning.objnav.action_converter.converter.DiscreteActionConverter`
keeps the path, the progress along it and the last action, answers a stop
command itself, and hands everything else here with one
:class:`~sparx_agency.core.planning.objnav.action_converter.path_progress.PathSample`
per step. The rungs, first match wins:

1. A camera pitch is commanded and one LOOK brings the camera toward it: that
   LOOK.
2. No path: face ``final_yaw`` if set and a turn is needed; else the command
   is done -- ``empty`` if it asked for nothing.
3. Arrived: the aim point is the last waypoint and the agent is within the
   effective goal tolerance of it, ``max(goal_tolerance_m, reach_floor)``.
   The arrival branch: face ``final_yaw`` if needed, else the command is done.
4. The aim point is outside the heading dead band: turn.
5. Aligned: MOVE_FORWARD, unless the aim point is the goal and a step would
   not bring the agent closer to it -- then the arrival branch.

A command that is done yields ``arrived`` and no action -- or STOP, when the
policy asked for ``stop_on_arrival``.

The quiet failures it is built against:

* **A STOP nobody asked for.** A done or empty command yields *no* action (an
  idle status): a ladder that stopped whenever a path ran out would end
  episodes a metre short, and the caller decides what an idle step is spent
  on. The one STOP here is the one a policy asked for in advance with
  ``stop_on_arrival``, on the very step its command completes -- the idle turn
  a step later would undo a facing.
* **An arrival on the way out.** Arrival needs the aim point to be the goal --
  the progress within a lookahead of the path's end -- not merely the agent
  near the last waypoint: an out-and-back, or a tour whose last stop lies
  beside an earlier leg, passes near its end long before it is walked.
* **Facing and turning back, forever.** Within :func:`reach_floor` of the goal
  no step aligned within the dead band is sure to close in. A ladder that kept
  aiming there would turn toward the goal, face ``final_yaw``, turn back, until
  the step budget ran out. So arrival is absorbing: turning in place changes
  neither the distance nor the progress.
* **Dithering.** The heading dead band is half a turn, so under exact
  execution a turn never needs undoing. With AI2-THOR's turn noise (each turn
  off by N(0, 0.5) degrees) a turn from just outside the band occasionally
  overshoots past its far edge and is undone at once -- never a livelock: the
  next true pose decides afresh.
* **Actions the benchmark lacks.** A pitch command on a benchmark without
  LOOK actions is a :class:`CommandError`, not a silently ignored request, and
  a LOOK past the simulator's pitch limits is never chosen.

The effective goal tolerance is never below :func:`reach_floor` -- 0.129 m for
Habitat's 0.25 m steps and 30-degree turns. A smaller ``goal_tolerance_m``
cannot be honoured by discrete steps: that close, the agent is as close as a
step aligned within half a turn can promise to take it.

Python 3.8 syntax; numpy arrives only through ``core.common.types``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.action_choice import (
    ANGLE_EPS_RAD,
    pitch_action,
    turn_action,
)
from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.action_converter.path_progress import (
    PathSample,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUS_ARRIVED,
    STATUS_EMPTY,
    STATUS_FACE,
    STATUS_FORWARD,
    STATUS_PITCH,
    STATUS_STOP,
    STATUS_TURN,
    ConversionResult,
)
from sparx_agency.core.planning.objnav.errors import CommandError, ObjNavError
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: The largest half-turn :func:`reach_floor` and :func:`parallel_offset` use,
#: radians. At a 180-degree turn the dead band is a quarter turn wide, its
#: cosine 0, and the floor and the offset infinite; 89 degrees keeps that
#: degenerate spec's finite.
MAX_FLOOR_HALF_TURN_RAD = math.radians(89.0)


def reach_floor(spec: DiscreteActionSpec) -> float:
    """The distance inside which a forward step aligned with a point may not close in on it.

    From distance ``d`` at heading error ``e``, one step of ``s`` metres
    shortens the distance exactly when ``d * cos(e) > s / 2``. The turn rung
    aligns the agent only to within half a turn, so beyond
    ``s / (2 * cos(turn / 2))`` an aligned step always closes in, and inside
    it nothing does for sure. That is the tightest goal tolerance discrete
    steps can honour: 0.129 m for Habitat's 0.25 m and 30 degrees.

    Args:
        spec: The benchmark's action geometry.

    Returns:
        The floor, metres.
    """
    # + ANGLE_EPS_RAD: turn_action's dead band carries the same slack, so an
    # error a hair past half a turn counts as aligned and must be covered too.
    half_turn = min(spec.turn_angle_rad / 2.0 + ANGLE_EPS_RAD,
                    MAX_FLOOR_HALF_TURN_RAD)
    return spec.forward_step_m / (2.0 * math.cos(half_turn))


def parallel_offset(spec: DiscreteActionSpec, lookahead_m: float) -> float:
    """How far off a leg the agent may walk along it without being turned back to it.

    Walking parallel to a leg at offset ``d``, the aim ``lookahead_m`` along
    it lies ``atan(d / lookahead_m)`` off the heading, and the turn rung acts
    only past half a turn: any ``d`` up to ``lookahead_m * tan(turn / 2)`` is
    walked as it is -- 0.134 m for Habitat's 30 degrees and a 0.5 m
    lookahead. That near a leg the agent is walking it, so the progress keeps
    that leg however near a later one passes.

    Args:
        spec: The benchmark's action geometry.
        lookahead_m: How far along the path the agent aims, metres.

    Returns:
        The offset, metres.
    """
    # + ANGLE_EPS_RAD, and the cap, as in reach_floor.
    half_turn = min(spec.turn_angle_rad / 2.0 + ANGLE_EPS_RAD,
                    MAX_FLOOR_HALF_TURN_RAD)
    return lookahead_m * math.tan(half_turn)


def check_executable(command: NavigationCommand,
                     spec: DiscreteActionSpec) -> None:
    """Raise if the benchmark cannot execute what ``command`` asks for.

    Args:
        command: A policy's command.
        spec: The benchmark's action geometry and action set.

    Raises:
        CommandError: If the command sets a camera pitch and the spec has no
            LOOK actions.
    """
    if command.camera_pitch is not None and not spec.has_camera_tilt:
        raise CommandError(
            "the command asks for camera pitch %r rad but this benchmark "
            "has no LOOK actions; plan without camera tilt"
            % (command.camera_pitch,))


def choose_action(pose: AgentPose, command: NavigationCommand,
                  sample: Optional[PathSample], blocked: bool,
                  spec: DiscreteActionSpec, params: ActionConverterParams
                  ) -> ConversionResult:
    """Climb the ladder for one step.

    Args:
        pose: The agent's true pose now.
        command: This step's command. Not a stop: that is answered before
            the ladder.
        sample: Where the agent is on the command's path this step, or None
            when the command has no path.
        blocked: Whether the last MOVE_FORWARD failed to move the agent;
            reported on the result.
        spec: The benchmark's action geometry and action set.
        params: The conversion tuning.

    Returns:
        The decision and its geometry; path fields are filled whenever there
        is a path. The heading error is to ``final_yaw`` at the end of a
        command (a hold, or the arrival branch) when one is set, else to the
        aim point.

    Raises:
        ObjNavError: If ``command`` is a stop.
        CommandError: As :func:`check_executable`.
    """
    if command.stop:
        raise ObjNavError("choose_action() is the ladder below STOP; answer "
                          "a stop command with STOP before climbing it")
    check_executable(command, spec)
    if command.camera_pitch is not None:
        look = pitch_action(pose.camera_pitch, command.camera_pitch,
                            spec.tilt_angle_rad, spec.min_pitch_rad,
                            spec.max_pitch_rad)
        if look is not None:
            return _result(look, STATUS_PITCH, sample, blocked)
    if sample is None or _arrived(sample, spec, params):
        return _finish(pose, command, sample, blocked, spec)
    turn = turn_action(sample.heading_error_rad, spec.turn_angle_rad)
    if turn is not None:
        return _result(turn, STATUS_TURN, sample, blocked)
    if not sample.aims_at_goal or _step_closes_in(pose, sample.target_xy,
                                                  spec):
        return _result(DiscreteAction.MOVE_FORWARD, STATUS_FORWARD, sample,
                       blocked)
    return _finish(pose, command, sample, blocked, spec)


def _arrived(sample: PathSample, spec: DiscreteActionSpec,
             params: ActionConverterParams) -> bool:
    """Whether the path is walked: the aim is the goal and the agent within the effective tolerance."""
    tolerance = max(params.goal_tolerance_m, reach_floor(spec))
    return sample.aims_at_goal and sample.distance_to_goal_m <= tolerance


def _finish(pose: AgentPose, command: NavigationCommand,
            sample: Optional[PathSample], blocked: bool,
            spec: DiscreteActionSpec) -> ConversionResult:
    """The end of a command: face ``final_yaw`` if needed, else it is done.

    Done is ``empty`` for a command that asked for nothing, STOP for one that
    asked for ``stop_on_arrival``, and ``arrived`` otherwise.
    """
    error = None
    if command.final_yaw is not None:
        error = normalize_angle(command.final_yaw - pose.yaw)
        turn = turn_action(error, spec.turn_angle_rad)
        if turn is not None:
            return _result(turn, STATUS_FACE, sample, blocked, error)
    if (sample is None and command.camera_pitch is None
            and command.final_yaw is None):
        return _result(None, STATUS_EMPTY, None, blocked)
    if command.stop_on_arrival:
        return _result(DiscreteAction.STOP, STATUS_STOP, sample, blocked,
                       error)
    return _result(None, STATUS_ARRIVED, sample, blocked, error)


def _step_closes_in(pose: AgentPose, target: Tuple[float, float],
                    spec: DiscreteActionSpec) -> bool:
    """Whether one forward step strictly shortens the distance to ``target``."""
    after = apply_action(pose, DiscreteAction.MOVE_FORWARD, spec)
    tx, ty = target
    return (math.hypot(tx - after.x, ty - after.y)
            < math.hypot(tx - pose.x, ty - pose.y))


def _result(action: Optional[DiscreteAction], status: str,
            sample: Optional[PathSample], blocked: bool,
            heading_error_rad: Optional[float] = None) -> ConversionResult:
    """A :class:`ConversionResult`, with the path geometry when there is a path.

    ``heading_error_rad`` None means the error to the aim point, or 0 without
    a path.
    """
    if sample is None:
        return ConversionResult(action=action, status=status,
                                heading_error_rad=heading_error_rad or 0.0,
                                forward_blocked=blocked)
    if heading_error_rad is None:
        heading_error_rad = sample.heading_error_rad
    return ConversionResult(
        action=action, status=status,
        segment_index=sample.segment_index,
        target_xy=sample.target_xy,
        heading_error_rad=heading_error_rad,
        distance_to_goal_m=sample.distance_to_goal_m,
        remaining_path_m=sample.remaining_path_m,
        cross_track_m=sample.cross_track_m,
        forward_blocked=blocked)
