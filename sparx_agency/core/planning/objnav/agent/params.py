"""Tuning of the headless agent.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction

#: What an idle step may be spent on: a turn in place, which every action set
#: has (``DiscreteActionSpec`` refuses one without both turns).
IDLE_ACTIONS = (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT)


@dataclass(frozen=True)
class HeadlessAgentParams:
    """How the headless agent turns a policy's commands into actions.

    Attributes:
        converter: Tuning of the waypoint-to-action conversion.
        idle_action: What a step is spent on when the converter has nothing to
            do -- the policy's path is complete, or it asked for nothing -- and
            the policy did not ask to stop. TURN_LEFT by default, TURN_RIGHT
            the only alternative: turning in place is the one move that always
            brings new views, never hits anything and is never refused. An
            idle MOVE_FORWARD into a wall would never be reported blocked (the
            converter emitted nothing to check), and an idle LOOK asks, at the
            pitch limit, for one AI2-THOR refuses. Never STOP: ending the
            episode is the policy's decision alone.

    Raises:
        ObjNavError: On a converter that is not :class:`ActionConverterParams`,
            or an idle action that is STOP, not a :class:`DiscreteAction`, or
            not a turn.
    """

    converter: ActionConverterParams = field(
        default_factory=ActionConverterParams)
    idle_action: DiscreteAction = DiscreteAction.TURN_LEFT

    def __post_init__(self) -> None:
        if not isinstance(self.converter, ActionConverterParams):
            raise ObjNavError(
                "HeadlessAgentParams.converter must be ActionConverterParams, "
                "got %r" % (self.converter,))
        if not isinstance(self.idle_action, DiscreteAction):
            raise ObjNavError(
                "HeadlessAgentParams.idle_action must be a DiscreteAction, got "
                "%r" % (self.idle_action,))
        if self.idle_action == DiscreteAction.STOP:
            raise ObjNavError(
                "HeadlessAgentParams.idle_action cannot be STOP: an idle step "
                "would silently end the episode")
        if self.idle_action not in IDLE_ACTIONS:
            raise ObjNavError(
                "HeadlessAgentParams.idle_action must be TURN_LEFT or "
                "TURN_RIGHT, got %s: an idle MOVE_FORWARD into a wall is never "
                "reported blocked, and an idle LOOK at the pitch limit is one "
                "the simulator refuses" % (self.idle_action.name,))
