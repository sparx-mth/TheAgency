"""Any reasonable path is followed to its end, promptly, and without dithering.

Three hundred seeded random polylines of 2 to 6 waypoints in a 10 m box, each
started from a random pose anywhere in the same box -- on the path or far off
it, facing anywhere -- and rolled out under perfect execution with Habitat's
action geometry and the default tuning. Random corners include every sharp,
near-reversing one a hand-built case would miss, and those are where a
lookahead follower livelocks: the aim wraps round the corner to behind the
agent, the agent walks back toward it, and the aim slides back in front. So
the properties a benchmark run depends on are checked over all of them:

* every rollout ends ARRIVED, within ``4 * path_length / step + 60`` actions,
  with ``path_length`` measured from the start pose through the waypoints.
  The approach counts because a start anywhere in the box can be 14 m from a
  short path, which the + 60 alone does not cover (a 3000-case sweep found
  one rollout one action over a waypoints-only budget);
* it ends within ``goal_tolerance_m`` of the last waypoint;
* it never turns left and then straight back right, or the reverse;
* it never emits STOP (ending the episode is the policy's decision) nor a
  LOOK (the command asked for no pitch).
"""
from __future__ import annotations

import math
import random

import pytest

from sparx_agency.core.planning.objnav.action_converter.converter import (
    DiscreteActionConverter,
)
from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.action_converter.path_geometry import (
    path_length,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUS_ARRIVED,
)
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose

SPEC = DiscreteActionSpec()
PARAMS = ActionConverterParams()
CASES = 300
SEED = 20260910
BOX_M = 10.0
#: How far past the budget a rollout may run, so a livelock and a merely slow
#: rollout fail different tests.
OVERRUN = 10


def budget(route) -> int:
    """The action budget of a route: ``4 * path_length / step + 60``."""
    return int(4.0 * path_length(route) / SPEC.forward_step_m) + 60


@pytest.fixture(scope="module")
def cases():
    """``(index, points, route, rollout)`` per seeded case, rolled out once.

    ``route`` is the start position followed by the waypoints: what the
    budget is measured along.
    """
    rng = random.Random(SEED)
    converter = DiscreteActionConverter(SPEC, PARAMS)
    result = []
    for index in range(CASES):
        points = tuple((rng.uniform(0.0, BOX_M), rng.uniform(0.0, BOX_M))
                       for _ in range(rng.randint(2, 6)))
        start = AgentPose(rng.uniform(0.0, BOX_M), rng.uniform(0.0, BOX_M),
                          yaw=rng.uniform(-math.pi, math.pi))
        route = ((start.x, start.y),) + points
        rollout = converter.rollout(start, NavigationCommand.follow(points),
                                    OVERRUN * budget(route))
        result.append((index, points, route, rollout))
    return result


def test_every_rollout_arrives(cases):
    """A rollout that never arrives is a livelock, typically at a reversing corner."""
    failed = [index for index, _, _, rollout in cases
              if rollout.status != STATUS_ARRIVED]
    assert not failed, "cases that did not arrive: %r" % failed


def test_every_rollout_fits_its_action_budget(cases):
    """Arriving by a detour of hundreds of steps would still fail a 500-step episode."""
    over = [(index, len(rollout.actions), budget(route))
            for index, _, route, rollout in cases
            if len(rollout.actions) > budget(route)]
    assert not over, "(case, actions, budget) over budget: %r" % over


def test_every_rollout_ends_within_tolerance_of_the_last_waypoint(cases):
    """ARRIVED must mean at the goal, not merely out of ideas."""
    far = []
    for index, points, _, rollout in cases:
        end = rollout.final_pose
        miss = math.hypot(end.x - points[-1][0], end.y - points[-1][1])
        if miss > PARAMS.goal_tolerance_m:
            far.append((index, miss))
    assert not far, "(case, metres from the goal): %r" % far


def test_no_rollout_turns_one_way_and_straight_back(cases):
    """A left immediately undone by a right is two wasted steps, and the start of a dither."""
    opposite = {DiscreteAction.TURN_LEFT: DiscreteAction.TURN_RIGHT,
                DiscreteAction.TURN_RIGHT: DiscreteAction.TURN_LEFT}
    dithered = []
    for index, _, _, rollout in cases:
        pairs = zip(rollout.actions, rollout.actions[1:])
        if any(opposite.get(first) == second for first, second in pairs):
            dithered.append(index)
    assert not dithered, "cases that turned straight back: %r" % dithered


def test_a_follow_command_emits_only_moves_and_turns(cases):
    """STOP is the policy's alone, and nothing here asked for the camera to move."""
    allowed = {DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT,
               DiscreteAction.TURN_RIGHT}
    stray = [index for index, _, _, rollout in cases
             if not set(rollout.actions) <= allowed]
    assert not stray, "cases with a STOP or LOOK: %r" % stray
