"""A rollout predicts a command's perfect execution without touching the converter it came from.

Each test rolls a command out and checks one promise a planner's what-if
relies on: the converter's own next step is unchanged, a subclass predicts
with its own decisions, every pose follows from its action, STOP and the
action budget end the rollout as documented, and the prediction is the
unobstructed one even when the converter is recovering from a blocked step.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.objnav.action_converter.converter import (
    DiscreteActionConverter,
)
from sparx_agency.core.planning.objnav.action_converter.tests.helpers import (
    FORWARD,
    HABITAT,
    HAIRPIN,
    LEFT,
    ORIGIN,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    ROLLOUT_LIMIT,
    STATUS_ARRIVED,
    STATUS_EMPTY,
    STATUS_STOP,
    ConversionResult,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def test_rollout_does_not_touch_the_converter():
    """A planner's what-if must not change the step the agent is about to take."""
    command = NavigationCommand.follow(HAIRPIN)
    used, twin = (DiscreteActionConverter(HABITAT),
                  DiscreteActionConverter(HABITAT))
    for converter in (used, twin):
        converter.step(AgentPose(2.0, 1.0, yaw=math.pi), command)
    used.rollout(ORIGIN, NavigationCommand.follow([(5.0, 5.0)]), 50)
    pose = AgentPose(2.0, 0.4, yaw=math.pi)
    assert used.step(pose, command) == twin.step(pose, command)


def test_a_subclass_rollout_predicts_with_the_subclass_decisions():
    """A rollout that silently used the base class would predict a different agent."""
    class Idle(DiscreteActionConverter):
        def step(self, pose, command):
            return ConversionResult(None, STATUS_EMPTY)

    rollout = Idle(HABITAT).rollout(
        ORIGIN, NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)]), 10)
    assert (rollout.actions, rollout.status) == ((), STATUS_EMPTY)


def test_rollout_poses_follow_the_actions():
    """Each pose is the previous one after its action, executed perfectly."""
    rollout = DiscreteActionConverter(HABITAT).rollout(
        ORIGIN, NavigationCommand.follow(HAIRPIN), 200)
    for index, action in enumerate(rollout.actions):
        assert rollout.poses[index + 1] == apply_action(
            rollout.poses[index], action, HABITAT)


def test_rollout_ends_on_stop_and_includes_it():
    """STOP is an action the environment executes, so it is part of the rollout."""
    rollout = DiscreteActionConverter(HABITAT).rollout(
        ORIGIN, NavigationCommand.stop_here(), 5)
    assert rollout.actions == (DiscreteAction.STOP,)
    assert rollout.poses == (ORIGIN, ORIGIN)
    assert rollout.status == STATUS_STOP


def test_rollout_stops_at_its_budget():
    """An unfinished command reports the limit rather than pretending to arrive."""
    rollout = DiscreteActionConverter(HABITAT).rollout(
        ORIGIN, NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)]), 4)
    assert rollout.actions == (FORWARD,) * 4
    assert rollout.status == ROLLOUT_LIMIT


def test_a_rollout_done_at_exactly_its_budget_reports_arrival():
    """The limit means the budget ran out first; here it did not."""
    rollout = DiscreteActionConverter(HABITAT).rollout(
        ORIGIN, NavigationCommand.follow([(0.0, 0.0), (1.0, 0.0)]), 3)
    assert rollout.actions == (FORWARD,) * 3
    assert rollout.status == STATUS_ARRIVED


@pytest.mark.parametrize("budget", [0, -1, True, 2.5])
def test_rollout_refuses_a_budget_that_is_not_a_positive_integer(budget):
    """A zero budget would report the limit for every command, done or not."""
    with pytest.raises(ObjNavError):
        DiscreteActionConverter(HABITAT).rollout(
            ORIGIN, NavigationCommand.hold(), budget)


def test_a_rollout_predicts_the_unobstructed_execution_not_the_recovery():
    """Perfect execution never blocks a step, so a recovering converter predicts as a fresh one does."""
    corner = NavigationCommand.follow([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)])
    before_corner = AgentPose(1.62, 0.0)
    ahead = NavigationCommand.follow([(1.62, 0.0), (5.0, 0.0)])
    recovering = DiscreteActionConverter(HABITAT)
    recovering.step(before_corner, ahead)
    recovering.step(before_corner, ahead)
    assert recovering.recovering
    rollout = recovering.rollout(before_corner, corner, 60)
    assert rollout == DiscreteActionConverter(HABITAT).rollout(
        before_corner, corner, 60)
    assert rollout.actions[0] == LEFT
    # Its own next step still recovers: one step ahead is straight on.
    assert recovering.step(before_corner, corner).action == FORWARD
