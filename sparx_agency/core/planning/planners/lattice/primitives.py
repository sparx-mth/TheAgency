"""The motion primitives of a discrete-action benchmark: one step, one turn each way, nothing else.

The lattice is the benchmark's own kinematics. Habitat's agent moves
``forward_step_m`` along its heading or rotates ``turn_angle_rad`` in place,
exactly, so every pose it can ever reach from a start pose lies on a lattice
of headings ``yaw0 + k * turn`` and positions that are sums of steps along
them. A planner that searches that lattice emits a plan the agent executes
verbatim -- no controller quantising a smooth path into turns it did not
plan for.

Python 3.8 syntax; numpy only for the sample offsets.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction, DiscreteActionSpec

#: The three body primitives of the lattice, in the order successors are generated.
BODY_PRIMITIVES = (DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT)


class MotionPrimitives:
    """The step and the turn of a :class:`DiscreteActionSpec`, as lattice geometry.

    Args:
        spec: The benchmark's action geometry.
        sample_spacing_m: Spacing of the collision samples along one step.

    Attributes:
        step_m: Length of one MOVE_FORWARD.
        turn_rad: Angle of one TURN.
        headings: How many distinct lattice headings a full circle holds
            (12 at 30 degrees). The lattice wraps only when the turn divides
            the circle; at 30 degrees it does.
        samples: Fractions of a step, ``(0, 1]``, at which a forward
            primitive is collision-checked. The start pose was checked as
            the previous state, so 0 is not among them.
    """

    def __init__(self, spec: DiscreteActionSpec, sample_spacing_m: float = 0.05):
        if not isinstance(spec, DiscreteActionSpec):
            raise TypeError("spec must be a DiscreteActionSpec, got %r" % (spec,))
        for action in BODY_PRIMITIVES:
            if not spec.allows(action):
                raise ValueError("the lattice needs %s in the action set" % action.name)
        self.spec = spec
        self.step_m = float(spec.forward_step_m)
        self.turn_rad = float(spec.turn_angle_rad)
        self.headings = max(1, int(round(2.0 * math.pi / self.turn_rad)))
        self.wraps = abs(self.headings * self.turn_rad - 2.0 * math.pi) < 1e-6
        count = max(1, int(math.ceil(self.step_m / float(sample_spacing_m))))
        self.samples = tuple(float(f) for f in np.linspace(1.0 / count, 1.0, count))

    def heading(self, yaw0: float, k: int) -> float:
        """The lattice heading ``k`` turns to the left of ``yaw0``, wrapped."""
        return normalize_angle(yaw0 + k * self.turn_rad)

    def index(self, yaw0: float, yaw: float) -> int:
        """The lattice index of ``yaw`` relative to ``yaw0`` (the nearest one, wrapped when the lattice wraps)."""
        k = int(round(normalize_angle(yaw - yaw0) / self.turn_rad))
        return k % self.headings if self.wraps else k

    def forward(self, x: float, y: float, yaw: float) -> Tuple[float, float]:
        """Where one MOVE_FORWARD from ``(x, y, yaw)`` ends."""
        return x + self.step_m * math.cos(yaw), y + self.step_m * math.sin(yaw)

    def turned(self, k: int, action: DiscreteAction) -> int:
        """The heading index after a TURN from heading index ``k``."""
        if action == DiscreteAction.TURN_LEFT:
            k += 1
        elif action == DiscreteAction.TURN_RIGHT:
            k -= 1
        else:
            raise ValueError("%s is not a turn" % (action,))
        return k % self.headings if self.wraps else k
