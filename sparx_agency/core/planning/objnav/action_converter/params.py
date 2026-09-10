"""Tuning of the waypoint-to-action conversion.

The benchmark fixes the *geometry* of an action
(:class:`~sparx_agency.core.planning.objnav.types.DiscreteActionSpec`); these
decide which action to take. Two tolerances are deliberately *not* here,
because they follow from the geometry and a separate knob could only make them
wrong: the heading dead band is half a turn, and the pitch dead band is half a
tilt. A dead band narrower than half a turn would make the agent turn left,
overshoot, turn right, forever.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from sparx_agency.core.planning.objnav.errors import ObjNavError


@dataclass(frozen=True)
class ActionConverterParams:
    """How the converter follows a path. Metres.

    Attributes:
        lookahead_m: How far ahead along the path, from the agent's projection
            onto it, the heading target sits. At least one forward step (the
            converter checks this against the spec): a target closer than half
            a step is overshot by the next step, and the agent turns back.
            Longer lookaheads cut corners. Two steps by default.
        goal_tolerance_m: The path counts as complete once the aim point is
            its last waypoint and the agent is within this distance of it.
            The effective tolerance is never below the ladder's
            ``reach_floor``, ``step / (2 cos(turn / 2))`` (0.129 m for
            Habitat's 0.25 m and 30 degrees): nearer than that no step aligned
            within the half-turn dead band is sure to close in, and insisting
            would turn toward the goal, face ``final_yaw`` and turn back
            forever. One step by default: the policy places the last waypoint
            where it wants to stand, and 0.25 m is well inside every
            benchmark's success radius.
        blocked_epsilon_m: A MOVE_FORWARD that moved the agent less than this
            is reported as blocked.

    Raises:
        ObjNavError: If any value is not a positive finite number.
    """

    lookahead_m: float = 0.5
    goal_tolerance_m: float = 0.25
    blocked_epsilon_m: float = 0.01

    def __post_init__(self) -> None:
        for name in ("lookahead_m", "goal_tolerance_m", "blocked_epsilon_m"):
            value = getattr(self, name)
            if (not isinstance(value, numbers.Real) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ObjNavError(
                    "ActionConverterParams.%s must be a positive finite "
                    "number, got %r" % (name, value))
