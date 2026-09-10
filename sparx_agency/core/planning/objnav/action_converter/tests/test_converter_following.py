"""A path is walked with exactly the turns its corners need, and progress along it only ever moves forward.

Each test drives a real converter through one scripted path and checks the
decision that path is about: forward steps when it is dead ahead, three turns
one way for a right angle, no U-turn for a waypoint already passed, arrival
rather than overshoot, a cursor that keeps its progress through a path that
doubles back near itself, nearly reverses at a corner, or retraces itself
exactly, and an aim that never sits on the agent's own foot at a fold.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.objnav.action_converter.converter import (
    DiscreteActionConverter,
)
from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.action_converter.tests.helpers import (
    FORWARD,
    HABITAT,
    HAIRPIN,
    LEFT,
    ORIGIN,
    RIGHT,
    drive,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUS_ARRIVED,
    STATUS_TURN,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose


# -- following a path -------------------------------------------------------------

def test_a_straight_path_is_walked_with_forward_steps_until_within_tolerance():
    """Nothing but MOVE_FORWARD when the path is dead ahead; no wasted turns."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)])
    results, final = drive(converter, ORIGIN, command)
    assert [r.action for r in results[:-1]] == [FORWARD] * 11
    assert results[-1].status == STATUS_ARRIVED
    assert math.hypot(final.x - 3.0, final.y) <= 0.25


def test_every_geometry_field_is_filled_while_following():
    """A bad episode is read back from these numbers, not re-simulated."""
    converter = DiscreteActionConverter(HABITAT)
    result = converter.step(AgentPose(1.0, 0.5),
                            NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)]))
    assert result.status == STATUS_TURN
    assert result.segment_index == 0
    assert result.target_xy == (1.5, 0.0)
    assert result.heading_error_rad == pytest.approx(-math.pi / 4)
    assert result.distance_to_goal_m == pytest.approx(math.hypot(2.0, 0.5))
    assert result.remaining_path_m == pytest.approx(2.0)
    assert result.cross_track_m == pytest.approx(0.5)
    assert result.forward_blocked is False


@pytest.mark.parametrize("corner_y, turn, other", [
    (2.0, LEFT, RIGHT),
    (-2.0, RIGHT, LEFT),
])
def test_a_right_angle_corner_takes_exactly_three_turns_one_way(
        corner_y, turn, other):
    """90 degrees is three 30-degree turns; a fourth, or any the other way, is waste."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (2.0, 0.0),
                                        (2.0, corner_y)])
    results, final = drive(converter, ORIGIN, command)
    actions = [r.action for r in results]
    assert actions.count(turn) == 3
    assert actions.count(other) == 0
    assert results[-1].status == STATUS_ARRIVED
    assert math.hypot(final.x - 2.0, final.y - corner_y) <= 0.25


def test_a_path_straight_behind_is_turned_toward_on_the_left():
    """An error of exactly pi must choose a side, the same side every time."""
    converter = DiscreteActionConverter(HABITAT)
    result = converter.step(ORIGIN,
                            NavigationCommand.follow([(0.0, 0.0), (-3.0, 0.0)]))
    assert (result.action, result.status) == (LEFT, STATUS_TURN)
    assert result.heading_error_rad == math.pi


def test_a_first_waypoint_behind_the_agent_is_not_turned_back_for():
    """A path planned from a cell centre just behind the agent must not cost a U-turn."""
    converter = DiscreteActionConverter(HABITAT)
    rollout = converter.rollout(AgentPose(1.0, 0.0),
                                NavigationCommand.follow([(0.0, 0.0),
                                                          (4.0, 0.0)]), 50)
    assert set(rollout.actions) == {FORWARD}
    assert rollout.status == STATUS_ARRIVED


def test_a_one_waypoint_path_is_walked_to():
    """A lone goal point is a path: turn toward it, walk, arrive."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(2.0, 1.0)])
    first = converter.step(ORIGIN, command)
    assert (first.action, first.target_xy, first.segment_index) == (
        LEFT, (2.0, 1.0), 0)
    results, final = drive(DiscreteActionConverter(HABITAT), ORIGIN, command)
    assert results[-1].status == STATUS_ARRIVED
    assert math.hypot(final.x - 2.0, final.y - 1.0) <= 0.25


def test_an_aim_point_within_half_a_step_short_of_the_goal_is_moved_on_along_the_path():
    """The path folds back at the agent's feet and goes on; aimed half a step out, the agent acts on a real heading."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (0.3, 0.0), (0.1, 0.0),
                                        (0.1, 3.0)])
    result = converter.step(ORIGIN, command)
    assert result.target_xy == pytest.approx((0.1, 0.075))
    assert math.hypot(*result.target_xy) == pytest.approx(0.125)
    assert (result.action, result.status) == (LEFT, STATUS_TURN)


def test_a_goal_within_half_a_step_is_arrived_at_rather_than_overshot():
    """With a tight tolerance, one more step would only land farther away."""
    converter = DiscreteActionConverter(
        HABITAT, ActionConverterParams(goal_tolerance_m=0.05))
    result = converter.step(AgentPose(0.9, 0.0),
                            NavigationCommand.follow([(0.0, 0.0), (1.0, 0.0)]))
    assert (result.action, result.status) == (None, STATUS_ARRIVED)
    assert result.distance_to_goal_m == pytest.approx(0.1)


# -- the cursor -------------------------------------------------------------------

def test_a_point_equally_near_two_legs_does_not_jump_ahead():
    """On a tie the earlier leg wins, or the agent would skip the hairpin."""
    converter = DiscreteActionConverter(HABITAT)
    result = converter.step(AgentPose(2.0, 0.5),
                            NavigationCommand.follow(HAIRPIN))
    assert result.segment_index == 0


def test_the_same_path_keeps_the_cursor_and_a_new_path_resets_it():
    """Re-sent every step, the same path must not lose progress; a new one must."""
    converter = DiscreteActionConverter(HABITAT)
    converter.step(AgentPose(2.0, 1.0, yaw=math.pi),
                   NavigationCommand.follow(HAIRPIN))
    # Nearer the outbound leg now, but already on the return leg.
    same = converter.step(AgentPose(2.0, 0.4, yaw=math.pi),
                          NavigationCommand.follow(list(HAIRPIN)))
    assert same.segment_index == 2
    assert same.target_xy[0] < 2.0 and same.target_xy[1] == 1.0
    fresh = converter.step(AgentPose(2.0, 0.4, yaw=math.pi),
                           NavigationCommand.follow(HAIRPIN + ((-1.0, 1.0),)))
    assert fresh.segment_index == 0


def test_a_hold_in_between_starts_the_same_path_over():
    """The cursor belongs to one path; a hold is a different (empty) one."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow(HAIRPIN)
    converter.step(AgentPose(2.0, 1.0, yaw=math.pi), command)
    converter.step(AgentPose(2.0, 1.0, yaw=math.pi), NavigationCommand.hold())
    again = converter.step(AgentPose(2.0, 0.4, yaw=math.pi), command)
    assert again.segment_index == 0


def test_a_path_that_doubles_back_near_itself_is_walked_in_order():
    """The return leg, 0.4 m away, must neither be jumped to early nor lose progress."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (4.0, 0.0), (4.0, 0.4),
                                        (-1.0, 0.4)])
    results, final = drive(converter, ORIGIN, command)
    segments = [r.segment_index for r in results]
    assert segments == sorted(segments)
    assert set(segments) == {0, 1, 2}
    poses = [ORIGIN]
    for result in results[:-1]:
        poses.append(apply_action(poses[-1], result.action, HABITAT))
    first_on_return = segments.index(2)
    assert poses[first_on_return].x > 3.5
    assert results[-1].status == STATUS_ARRIVED
    assert math.hypot(final.x + 1.0, final.y - 0.4) <= 0.25


def test_a_corner_that_nearly_reverses_is_walked_round_not_paced_before():
    """The livelock the random sweep found, reduced to round numbers."""
    # 0.134 m short of the corner, the aim half a metre along the path lies on
    # the return leg behind the agent. With only a segment cursor, walking
    # back toward it ran the projection back too, the aim slid in front
    # again, and the agent paced between x = 2.616 and 2.866 forever.
    rollout = DiscreteActionConverter(HABITAT).rollout(
        AgentPose(0.116, 0.0), NavigationCommand.follow(
            [(0.0, 0.0), (3.0, 0.0), (-1.0, 0.05)]), 100)
    assert rollout.status == STATUS_ARRIVED
    assert len(rollout.actions) <= 40


def test_a_path_that_retraces_itself_exactly_still_reaches_its_end():
    """Out and back on one line: every point of the way back ties with the way out."""
    rollout = DiscreteActionConverter(HABITAT).rollout(
        ORIGIN, NavigationCommand.follow([(0.0, 0.0), (2.0, 0.0),
                                          (-1.0, 0.0)]), 200)
    assert max(pose.x for pose in rollout.poses) >= 1.75
    assert rollout.status == STATUS_ARRIVED
    end = rollout.final_pose
    assert math.hypot(end.x + 1.0, end.y) <= 0.25


# -- the aim at a fold ---------------------------------------------------------

#: Quarter-metre reversals on one line, then on west: paced forever before.
REVERSALS = ((0.0, -0.75), (-0.5, -0.75), (-0.75, -0.75), (-0.5, -0.75),
             (-0.75, -0.75), (-1.0, -0.75), (-2.0, -0.75))


@pytest.mark.parametrize("final_yaw", [None, 0.0])
def test_a_path_of_exact_reversals_is_walked_to_its_end(final_yaw):
    """With the aim on the agent's own foot, the agent paced across the path until the budget ran out."""
    rollout = DiscreteActionConverter(HABITAT).rollout(
        AgentPose(0.0, -0.75, yaw=math.pi),
        NavigationCommand.follow(REVERSALS, final_yaw=final_yaw), 200)
    assert rollout.status == STATUS_ARRIVED
    assert len(rollout.actions) <= 30
    end = rollout.final_pose
    assert math.hypot(end.x + 2.0, end.y + 0.75) <= 0.25


def test_an_exact_fold_is_walked_along_the_path_not_stepped_off_sideways():
    """At the fold the heading to the aim was a rounding error's; in a dead-end corridor that sideways step hits the wall."""
    rollout = DiscreteActionConverter(HABITAT).rollout(
        AgentPose(0.0, 0.0, yaw=math.pi / 2),
        NavigationCommand.follow([(0.0, 0.0), (0.0, 2.0), (0.0, -1.0)]), 100)
    assert rollout.status == STATUS_ARRIVED
    assert max(pose.y for pose in rollout.poses) >= 1.75
    assert max(abs(pose.x) for pose in rollout.poses) < 1e-9
