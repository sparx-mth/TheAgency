"""How the fake's agent moves: Habitat's transition, except that a forward step off the navigable floor does not happen at all.

A **test rig, not a benchmark.** Turns, looks and STOP are
:func:`~sparx_agency.core.planning.objnav.action_converter.transition.apply_action`'s,
the same prediction the action converter makes, so a LOOK past the pitch
limits fails without moving the camera, as AI2-THOR's does. Only MOVE_FORWARD
can differ from the prediction -- that difference is how a benchmark refuses
a step -- and the quiet failures of modelling it are designed out here:

* **A step that slides.** A blocked MOVE_FORWARD does not move the agent at
  all, AI2-THOR's atomic failure. A sliding fake would credit path never
  walked, and hide the blocked step the converter and the policy must see.
* **A step through a corner.** The step is checked at points a quarter cell
  apart, so it cannot pass through a non-navigable cell between two navigable
  ends.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld

#: MOVE_FORWARD is checked at points this many to a cell apart, so a step
#: cannot pass through a non-navigable cell between its two ends.
SAMPLES_PER_CELL = 4


def move(world: GridWorld, navigable: np.ndarray, pose: AgentPose,
         action: DiscreteAction, spec: DiscreteActionSpec) -> AgentPose:
    """The pose ``action`` leaves the agent in.

    Args:
        world: The building.
        navigable: Mask of the cells the agent can stand on.
        pose: The agent's pose before the action.
        action: The action; one ``spec`` allows.
        spec: The action geometry.

    Returns:
        :func:`apply_action`'s pose, or ``pose`` itself when a MOVE_FORWARD
        would leave the navigable floor.
    """
    after = apply_action(pose, action, spec)
    if (action == DiscreteAction.MOVE_FORWARD
            and not _step_is_clear(world, navigable, pose, after)):
        return pose
    return after


def _step_is_clear(world: GridWorld, navigable: np.ndarray, before: AgentPose,
                   after: AgentPose) -> bool:
    """Whether every point a quarter cell apart on the step is navigable."""
    spacing = world.resolution_m / SAMPLES_PER_CELL
    length = math.hypot(after.x - before.x, after.y - before.y)
    count = max(1, int(math.ceil(length / spacing)))
    for index in range(count + 1):
        t = index / count
        row, col = world.cell_of(before.x + t * (after.x - before.x),
                                 before.y + t * (after.y - before.y))
        if not (world.in_bounds(row, col) and navigable[row, col]):
            return False
    return True
