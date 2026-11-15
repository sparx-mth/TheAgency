"""What one discrete action does to the agent's pose, if it executes perfectly.

This is the converter's model of the simulator: what a rollout predicts with,
and the definition of "one forward step" the converter decides with. It
follows the simulators at their edges, because a model that is kinder than
the simulator predicts poses the agent never reaches:

* MOVE_FORWARD moves along the heading and never slides. A step into a wall
  is not modelled here at all -- it is the environment's to execute, and the
  converter notices it afterwards (``forward_blocked``).
* TURN_LEFT is counter-clockwise from above, as in Habitat; yaw is wrapped to
  ``(-pi, pi]``.
* LOOK_DOWN adds the tilt (REP-103: positive pitch looks down) and LOOK_UP
  subtracts it. A LOOK that would leave the spec's pitch limits leaves the
  pitch where it was -- AI2-THOR *fails* it, the step spent -- rather than
  clamping, which would predict a pitch the camera never reaches. The limits
  are judged by ``action_choice.pitch_within_limits``, with its float-noise
  slack. Habitat has no limits, and neither does a spec whose limits are None.
* STOP changes nothing, and nothing ever changes ``z``: the floor height is
  the environment's to report.

Python 3.8 syntax; numpy arrives only through ``core.common.types``.
"""
from __future__ import annotations

import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.action_choice import (
    pitch_within_limits,
)
from sparx_agency.core.planning.objnav.errors import CommandError
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def apply_action(pose: AgentPose, action: DiscreteAction,
                 spec: DiscreteActionSpec) -> AgentPose:
    """The pose after ``action``, executed perfectly.

    Args:
        pose: The pose before the action.
        action: The action to execute. A :class:`DiscreteAction` member,
            never a bare integer.
        spec: The benchmark's action geometry and action set.

    Returns:
        The new pose. ``pose`` itself when nothing changes (STOP, or a LOOK
        the pitch limits refuse).

    Raises:
        TypeError: If an argument has the wrong type.
        CommandError: If ``spec`` does not allow ``action``.
    """
    if not isinstance(pose, AgentPose):
        raise TypeError("pose must be an AgentPose, got %r" % (pose,))
    if not isinstance(action, DiscreteAction):
        raise TypeError(
            "action must be a DiscreteAction member, got %r -- a bare integer "
            "means a different action to each simulator" % (action,))
    if not isinstance(spec, DiscreteActionSpec):
        raise TypeError("spec must be a DiscreteActionSpec, got %r" % (spec,))
    if not spec.allows(action):
        raise CommandError(
            "%s is not in this benchmark's action set (%s)"
            % (action.name, ", ".join(a.name for a in spec.actions)))
    if action == DiscreteAction.MOVE_FORWARD:
        step = spec.forward_step_m
        return AgentPose(pose.x + step * math.cos(pose.yaw),
                         pose.y + step * math.sin(pose.yaw),
                         pose.z, pose.yaw, pose.camera_pitch)
    if action in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT):
        turn = spec.turn_angle_rad
        if action == DiscreteAction.TURN_RIGHT:
            turn = -turn
        return AgentPose(pose.x, pose.y, pose.z,
                         normalize_angle(pose.yaw + turn), pose.camera_pitch)
    if action in (DiscreteAction.LOOK_UP, DiscreteAction.LOOK_DOWN):
        tilt = spec.tilt_angle_rad
        if action == DiscreteAction.LOOK_UP:
            tilt = -tilt
        pitch = pose.camera_pitch + tilt
        if not pitch_within_limits(pitch, spec.min_pitch_rad,
                                   spec.max_pitch_rad):
            return pose
        return AgentPose(pose.x, pose.y, pose.z, pose.yaw, pitch)
    return pose
