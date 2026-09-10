"""What an agent returns for one observation: the action, and why.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field
from typing import Any, Dict

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction


@dataclass(frozen=True)
class AgentDecision:
    """One step's output of an ObjectNav agent.

    Attributes:
        action: The discrete action to execute. A :class:`DiscreteAction`
            member, never a bare integer -- the integer order is Habitat's,
            and a bare ``3`` means something else to another simulator.
        info: Per-step diagnostics (which converter status, which room, ...):
            a mapping, stored as a dict copy. Returned to the caller of
            ``act()`` and never interpreted. The harness does not store it:
            only the agent's ``episode_info()`` reaches the results row (its
            ``agent_info``), so a caller that wants a per-step trace keeps
            these itself.

    Raises:
        ObjNavError: If ``action`` is not a :class:`DiscreteAction`, or
            ``info`` is not a mapping.
    """

    action: DiscreteAction
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, DiscreteAction):
            raise ObjNavError(
                "AgentDecision.action must be a DiscreteAction, got %r"
                % (self.action,))
        if not isinstance(self.info, collections.abc.Mapping):
            raise ObjNavError(
                "AgentDecision.info must be a mapping of diagnostics, got %r"
                % (self.info,))
        object.__setattr__(self, "info", dict(self.info))
