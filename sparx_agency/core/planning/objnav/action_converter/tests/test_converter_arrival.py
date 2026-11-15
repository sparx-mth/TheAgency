"""A path is arrived at only once it is walked, and an arrival stays an arrival.

Arrival used to mean "within goal_tolerance_m of the last waypoint", whatever
the progress. An out-and-back that ends on its way out was arrived at before
it was walked, and a tour whose last stop lies beside an earlier leg was cut
short there. Below the reach floor, a tight tolerance with a final heading
turned toward the goal, faced the heading and turned back until the step
budget ran out. Each test drives a real converter under perfect execution
through one of those paths.
"""
from __future__ import annotations

import math
import random

import pytest

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.converter import (
    DiscreteActionConverter,
)
from sparx_agency.core.planning.objnav.action_converter.ladder import (
    parallel_offset,
    reach_floor,
)
from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.action_converter.tests.helpers import (
    HABITAT,
    LEFT,
    ORIGIN,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUS_ARRIVED,
)
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: Below Habitat's reach floor (0.129 m): its own 0.1 m success radius.
TIGHT = ActionConverterParams(goal_tolerance_m=0.1)
#: The goal the review found: 1.12 m out, 3 cm off the line.
NEAR_MISS_GOAL = (1.12, 0.03)
#: A stop counts as visited within the default lookahead: the agent aims that
#: far ahead, so it cuts a corner by up to about that much. A skipped stop
#: stays metres away.
VISITED_M = 0.5


def miss(pose, point) -> float:
    """How far ``pose`` is from ``point``, metres."""
    return math.hypot(pose.x - point[0], pose.y - point[1])


def closest_approach(poses, point) -> float:
    """How near any of ``poses`` came to ``point``, metres."""
    return min(miss(pose, point) for pose in poses)


@pytest.mark.parametrize("path", [
    ((0.0, 0.0), (3.0, 0.0), (0.0, 0.0)),
    ((0.0, 0.0), (3.0, 0.0), (1.0, 0.0)),
], ids=["back-to-the-start", "back-onto-the-way-out"])
def test_an_out_and_back_ending_on_its_outbound_leg_is_walked_to_the_end(path):
    """The agent passes near the end on the way out; arriving there would skip the walk."""
    rollout = DiscreteActionConverter(HABITAT).rollout(
        ORIGIN, NavigationCommand.follow(path), 200)
    assert rollout.status == STATUS_ARRIVED
    assert closest_approach(rollout.poses, path[1]) <= 0.25
    assert miss(rollout.final_pose, path[-1]) <= 0.25


def test_a_tour_whose_last_stop_is_passed_early_is_walked_in_full():
    """RPT*'s room order sent as one path: every stop visited, not cut short where the end is first near."""
    tour = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 3.0), (2.0, 0.2))
    rollout = DiscreteActionConverter(HABITAT).rollout(
        ORIGIN, NavigationCommand.follow(tour), 300)
    assert rollout.status == STATUS_ARRIVED
    for stop in tour[1:-1]:
        assert closest_approach(rollout.poses, stop) <= 0.25, stop
    assert miss(rollout.final_pose, tour[-1]) <= 0.25


@pytest.mark.parametrize("start", [AgentPose(0.0, 0.04), AgentPose(0.0, 0.08)],
                         ids=["4cm-off", "8cm-off"])
def test_an_out_and_back_returning_just_beside_its_way_out_is_walked_to_its_far_end(start):
    """An A* return leg 5 cm beside the way out took the progress from an agent a few cm off it: arrived without a step."""
    path = ((0.0, 0.0), (5.0, 0.0), (5.0, 0.05), (0.0, 0.05))
    rollout = DiscreteActionConverter(HABITAT).rollout(
        start, NavigationCommand.follow(path), 200)
    assert rollout.status == STATUS_ARRIVED
    assert closest_approach(rollout.poses, path[1]) <= VISITED_M
    assert (miss(rollout.final_pose, path[-1])
            <= ActionConverterParams().goal_tolerance_m)


@pytest.mark.parametrize("yaw_deg", [5.0, 10.0, 14.0])
def test_a_tour_started_a_little_off_its_heading_still_visits_every_stop(yaw_deg):
    """Inside the dead band the agent walks beside the hallway, nearer its return leg; the room used to be skipped."""
    tour = ((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (4.95, 0.05), (0.0, 0.05))
    start = AgentPose(0.0, 0.0, yaw=math.radians(yaw_deg))
    rollout = DiscreteActionConverter(HABITAT).rollout(
        start, NavigationCommand.follow(tour), 300)
    assert rollout.status == STATUS_ARRIVED
    for stop in tour[1:-1]:
        assert closest_approach(rollout.poses, stop) <= VISITED_M, stop
    assert (miss(rollout.final_pose, tour[-1])
            <= ActionConverterParams().goal_tolerance_m)


@pytest.mark.parametrize("start", [AgentPose(0.0, -0.75, yaw=math.pi),
                                   AgentPose(0.1, -0.7, yaw=0.3)],
                         ids=["on-the-path", "off-it-facing-away"])
def test_quarter_metre_reversals_are_walked_to_their_end_from_on_or_off_the_path(start):
    """Inside the band the reversals are one stretch: progress held at a walked leg's end vertex paced the agent forever."""
    path = ((0.0, -0.75), (-0.5, -0.75), (-0.75, -0.75), (-0.5, -0.75),
            (-0.75, -0.75), (-1.0, -0.75), (-2.0, -0.75))
    rollout = DiscreteActionConverter(HABITAT).rollout(
        start, NavigationCommand.follow(path), 200)
    assert rollout.status == STATUS_ARRIVED
    assert len(rollout.actions) <= 60
    assert (miss(rollout.final_pose, path[-1])
            <= ActionConverterParams().goal_tolerance_m)


def test_the_parallel_offset_is_the_widest_a_leg_is_walked_beside_without_a_turn():
    """The band that keeps the progress on a leg must be exactly as wide as the ladder lets the agent stray from it."""
    offset = parallel_offset(HABITAT, 0.5)
    assert offset == pytest.approx(0.5 * math.tan(math.radians(15.0)))
    leg = NavigationCommand.follow([(-1.0, 0.0), (10.0, 0.0)])
    inside = DiscreteActionConverter(HABITAT).step(
        AgentPose(0.0, 0.99 * offset), leg)
    outside = DiscreteActionConverter(HABITAT).step(
        AgentPose(0.0, 1.05 * offset), leg)
    assert inside.action == DiscreteAction.MOVE_FORWARD
    assert outside.action == DiscreteAction.TURN_RIGHT


def test_a_tolerance_below_the_reach_floor_with_a_final_heading_arrives_instead_of_looping():
    """0.1 m with final_yaw used to alternate TURN_LEFT and TURN_RIGHT until the budget ran out."""
    command = NavigationCommand.follow([(0.0, 0.0), NEAR_MISS_GOAL],
                                       final_yaw=math.pi / 2)
    rollout = DiscreteActionConverter(HABITAT, TIGHT).rollout(ORIGIN, command,
                                                              200)
    assert rollout.status == STATUS_ARRIVED
    assert len(rollout.actions) <= 10
    end = rollout.final_pose
    assert miss(end, NEAR_MISS_GOAL) <= reach_floor(HABITAT)
    assert abs(normalize_angle(end.yaw - math.pi / 2)) <= math.radians(15.0)


def test_an_arrival_survives_the_idle_turns_spent_after_it():
    """Each idle TURN_LEFT used to un-arrive the agent and draw a TURN_RIGHT straight back."""
    converter = DiscreteActionConverter(HABITAT, TIGHT)
    command = NavigationCommand.follow([(0.0, 0.0), NEAR_MISS_GOAL])
    pose, statuses = ORIGIN, []
    for _ in range(20):
        result = converter.step(pose, command)
        statuses.append(result.status)
        pose = apply_action(pose, LEFT if result.idle else result.action,
                            HABITAT)
    first = statuses.index(STATUS_ARRIVED)
    assert statuses[first:] == [STATUS_ARRIVED] * (len(statuses) - first)


def test_a_tight_tolerance_with_any_final_heading_always_arrives_within_the_reach_floor():
    """The review's sweep at 0.1 m: 62 of 300 random paths with a heading never ended."""
    rng = random.Random(20260910)
    converter = DiscreteActionConverter(
        HABITAT, ActionConverterParams(goal_tolerance_m=0.05))
    failed = []
    for index in range(150):
        points = [(rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0))
                  for _ in range(rng.randint(2, 5))]
        start = AgentPose(rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0),
                          yaw=rng.uniform(-math.pi, math.pi))
        command = NavigationCommand.follow(
            points, final_yaw=rng.uniform(-math.pi, math.pi))
        rollout = converter.rollout(start, command, 1000)
        if (rollout.status != STATUS_ARRIVED
                or miss(rollout.final_pose, points[-1]) > reach_floor(HABITAT)):
            failed.append(index)
    assert not failed, "cases that did not arrive within the floor: %r" % failed
