"""What the action converter decided, and why.

Every decision carries its reason (``status``) and the geometry behind it,
returned with the action: the headless agent hands both to the caller of
``act()`` in ``AgentDecision.info``. The harness stores only the agent's
``episode_info()``, so a caller that wants a per-step trace keeps the
decisions itself. A converter with nothing left to do returns *no* action
rather than STOP: ending the episode is the policy's decision, and a converter
that defaulted to STOP would end episodes a metre short whenever a path ran
out. (A command that asked for ``stop_on_arrival`` gets STOP then -- the
policy's decision, made in advance.)

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose

# -- why this action ----------------------------------------------------------

#: The command asked to stop: a stop command, or a ``stop_on_arrival`` command
#: on the step it is fully executed.
STATUS_STOP = "stop"
#: Tilting the camera toward the commanded pitch.
STATUS_PITCH = "pitch"
#: Rotating toward the path.
STATUS_TURN = "turn"
#: Stepping along the path.
STATUS_FORWARD = "forward"
#: At the end of the path (or holding), rotating toward the commanded heading.
STATUS_FACE = "face"
#: The command is fully executed: nothing left to do, so no action.
STATUS_ARRIVED = "arrived"
#: The command asked for nothing at all, so no action.
STATUS_EMPTY = "empty"

#: Statuses that carry an action.
ACTING_STATUSES = (STATUS_STOP, STATUS_PITCH, STATUS_TURN, STATUS_FORWARD,
                   STATUS_FACE)
#: Statuses that carry none: the caller decides what the step is spent on.
IDLE_STATUSES = (STATUS_ARRIVED, STATUS_EMPTY)
#: Every status.
STATUSES = ACTING_STATUSES + IDLE_STATUSES

# -- how a rollout ended ------------------------------------------------------

#: The rollout used its whole action budget before the command was done.
ROLLOUT_LIMIT = "limit"
#: Every way a rollout ends.
ROLLOUT_ENDINGS = (STATUS_STOP, STATUS_ARRIVED, STATUS_EMPTY, ROLLOUT_LIMIT)


@dataclass(frozen=True)
class ConversionResult:
    """One step's decision.

    Attributes:
        action: The action to execute, or None exactly when ``status`` is
            idle. Never STOP unless the command asked to stop (a stop, or
            ``stop_on_arrival`` once it is done).
        status: One of :data:`STATUSES`.
        segment_index: The path segment the agent's progress projects onto;
            0 without a path.
        target_xy: The point being aimed at on the path, or None when not
            following one.
        heading_error_rad: Signed error to the aim -- the lookahead point, or
            the commanded heading at the end of a command (facing, or arrived
            with ``final_yaw`` set) -- wrapped to ``(-pi, pi]``; 0 when there
            is no aim.
        distance_to_goal_m: Straight-line distance to the last waypoint; 0
            without a path.
        remaining_path_m: Path length from the agent's projection to the last
            waypoint; 0 without a path.
        cross_track_m: Distance from the agent to its projection on the path.
        forward_blocked: The previous action was MOVE_FORWARD and the agent
            moved less than ``blocked_epsilon_m`` -- something the
            perfect-execution model could not know about. Reported on every
            result, STOP included. The converter's only response is to aim one
            forward step ahead instead of ``lookahead_m`` until a MOVE_FORWARD
            moves again; it never plans a detour. That is the policy's job,
            through its optional ``notify_blocked`` hook.

    Raises:
        ObjNavError: On an unknown status, an action on an idle status or none
            on an acting one, or a STOP on anything but a stop.
    """

    action: Optional[DiscreteAction]
    status: str
    segment_index: int = 0
    target_xy: Optional[Tuple[float, float]] = None
    heading_error_rad: float = 0.0
    distance_to_goal_m: float = 0.0
    remaining_path_m: float = 0.0
    cross_track_m: float = 0.0
    forward_blocked: bool = False

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ObjNavError(
                "ConversionResult.status must be one of %r, got %r"
                % (STATUSES, self.status))
        if (self.action is None) != (self.status in IDLE_STATUSES):
            raise ObjNavError(
                "ConversionResult carries an action exactly when its status "
                "is acting; got action=%r with status=%r"
                % (self.action, self.status))
        if self.action is not None and not isinstance(self.action,
                                                      DiscreteAction):
            raise ObjNavError(
                "ConversionResult.action must be a DiscreteAction, got %r"
                % (self.action,))
        if (self.action == DiscreteAction.STOP) != (self.status == STATUS_STOP):
            raise ObjNavError(
                "ConversionResult emits STOP only for a stop command; got "
                "action=%r with status=%r" % (self.action, self.status))

    @property
    def idle(self) -> bool:
        """Whether the converter had nothing to do this step."""
        return self.action is None


@dataclass(frozen=True)
class Rollout:
    """A command executed to completion under perfect execution.

    Attributes:
        actions: Every action emitted, in order.
        poses: The start pose, then the pose after each action.
        status: How it ended: one of :data:`ROLLOUT_ENDINGS`.

    Raises:
        ObjNavError: If ``poses`` is not one longer than ``actions``, or the
            ending is unknown.
    """

    actions: Tuple[DiscreteAction, ...]
    poses: Tuple[AgentPose, ...]
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "poses", tuple(self.poses))
        if len(self.poses) != len(self.actions) + 1:
            raise ObjNavError(
                "Rollout needs the start pose plus one pose per action: %d "
                "actions, %d poses" % (len(self.actions), len(self.poses)))
        if self.status not in ROLLOUT_ENDINGS:
            raise ObjNavError(
                "Rollout.status must be one of %r, got %r"
                % (ROLLOUT_ENDINGS, self.status))

    @property
    def final_pose(self) -> AgentPose:
        """Where the rollout left the agent."""
        return self.poses[-1]
