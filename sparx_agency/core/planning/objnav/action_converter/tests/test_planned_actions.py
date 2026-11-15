"""A command's planned actions are executed verbatim while the agent stands where the plan expects it.

Python 3.8 syntax; numpy arrives only through the types.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.action_converter.tests.helpers import FORWARD, HABITAT, LEFT, ORIGIN, RIGHT
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.action_converter.types import STATUS_FORWARD, STATUS_PITCH, STATUS_TURN
from sparx_agency.core.planning.objnav.errors import CommandError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: A dog-leg the ladder would walk with turns of its own choosing.
PATH = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def planned(pose, actions, **kwargs):
    return NavigationCommand.follow(PATH, actions=actions, plan_pose=(pose.x, pose.y, pose.yaw), **kwargs)


def test_the_first_planned_action_is_executed_as_planned_and_flagged():
    converter = DiscreteActionConverter(HABITAT)
    # The ladder facing east with the aim 0.5 m along the first leg would step forward; the plan says turn.
    result = converter.step(ORIGIN, planned(ORIGIN, (LEFT, FORWARD)))
    assert result.action == LEFT and result.status == STATUS_TURN and result.planned
    assert result.remaining_path_m == pytest.approx(2.0), "the path geometry is still reported"


def test_a_stale_plan_pose_falls_back_to_the_waypoint_ladder():
    converter = DiscreteActionConverter(HABITAT)
    elsewhere = AgentPose(0.10, 0.0, 0.0, 0.0, 0.0)              # 10 cm from where the plan starts
    result = converter.step(elsewhere, planned(ORIGIN, (LEFT, FORWARD)))
    assert result.action == FORWARD and not result.planned, "the ladder decides, not the stale plan"
    turned = AgentPose(0.0, 0.0, 0.0, math.radians(30.0), 0.0)  # at the position, off the heading
    result = converter.step(turned, planned(ORIGIN, (LEFT, FORWARD)))
    assert not result.planned


def test_the_plan_is_executed_step_by_step_as_the_policy_sends_the_suffix():
    converter = DiscreteActionConverter(HABITAT)
    pose, actions = ORIGIN, [FORWARD, FORWARD, FORWARD, FORWARD, LEFT, LEFT, LEFT, FORWARD, FORWARD, FORWARD, FORWARD]
    executed = []
    while actions:
        result = converter.step(pose, planned(pose, tuple(actions)))
        assert result.planned and result.action == actions[0]
        executed.append(result.action)
        pose = apply_action(pose, result.action, HABITAT)
        actions.pop(0)
    assert executed.count(FORWARD) == 8 and executed.count(LEFT) == 3
    assert (pose.x, pose.y) == pytest.approx((1.0, 1.0), abs=1e-9)
    assert abs(pose.yaw - math.pi / 2) < 1e-9


def test_a_camera_pitch_is_reached_before_the_plan_continues():
    converter = DiscreteActionConverter(HABITAT)
    command = planned(ORIGIN, (FORWARD,), camera_pitch=math.radians(30.0))
    first = converter.step(ORIGIN, command)
    assert first.action == DiscreteAction.LOOK_DOWN and first.status == STATUS_PITCH and not first.planned
    pitched = AgentPose(0.0, 0.0, 0.0, 0.0, math.radians(30.0))
    second = converter.step(pitched, command)
    assert second.action == FORWARD and second.planned and second.status == STATUS_FORWARD


def test_a_stop_command_still_wins_and_the_blocked_flag_is_reported():
    converter = DiscreteActionConverter(HABITAT)
    converter.step(ORIGIN, planned(ORIGIN, (FORWARD,)))
    # The agent did not move: the step was blocked. The flag is on the planned result too.
    result = converter.step(ORIGIN, planned(ORIGIN, (RIGHT,)))
    assert result.forward_blocked and result.planned and result.action == RIGHT
    assert converter.step(ORIGIN, NavigationCommand.stop_here()).action == DiscreteAction.STOP


def test_plan_tolerances_are_the_converters_knobs():
    loose = DiscreteActionConverter(HABITAT, ActionConverterParams(plan_position_tolerance_m=0.2))
    result = loose.step(AgentPose(0.10, 0.0, 0.0, 0.0, 0.0), planned(ORIGIN, (LEFT,)))
    assert result.planned and result.action == LEFT
    with pytest.raises(Exception):
        ActionConverterParams(plan_position_tolerance_m=0.0)


def test_the_command_refuses_a_malformed_plan():
    with pytest.raises(CommandError, match="MOVE_FORWARD, TURN_LEFT or TURN_RIGHT"):
        NavigationCommand.follow(PATH, actions=(DiscreteAction.LOOK_DOWN,), plan_pose=(0.0, 0.0, 0.0))
    with pytest.raises(CommandError, match="plan_pose"):
        NavigationCommand.follow(PATH, actions=(FORWARD,))
    with pytest.raises(CommandError, match="plan_pose"):
        NavigationCommand.follow(PATH, plan_pose=(0.0, 0.0, 0.0))
    with pytest.raises(CommandError, match="waypoints"):
        NavigationCommand(actions=(FORWARD,), plan_pose=(0.0, 0.0, 0.0))
    with pytest.raises(CommandError, match="finite"):
        NavigationCommand.follow(PATH, actions=(FORWARD,), plan_pose=(0.0, float("nan"), 0.0))
    command = NavigationCommand.follow(PATH, actions=[FORWARD, LEFT], plan_pose=[1, 2, 0.5])
    assert command.actions == (FORWARD, LEFT) and command.plan_pose == (1.0, 2.0, 0.5)
    assert NavigationCommand.follow(PATH).actions == () and NavigationCommand.follow(PATH).plan_pose is None


# -- strict plan tracking ------------------------------------------------------------------------------------

STRICT = ActionConverterParams(strict_plan_tracking=True)


def test_strict_tracking_turns_at_the_kink_instead_of_cutting_the_corner():
    """A dog-leg walked by the ladder alone: the strict ladder reaches the kink before it turns."""
    from sparx_agency.core.planning.objnav.action_converter.tests.helpers import drive
    loose, _ = drive(DiscreteActionConverter(HABITAT), ORIGIN, NavigationCommand.follow(PATH))
    strict, _ = drive(DiscreteActionConverter(HABITAT, STRICT), ORIGIN, NavigationCommand.follow(PATH))
    poses = [ORIGIN]
    for result in strict:
        if result.action is not None:
            poses.append(apply_action(poses[-1], result.action, HABITAT))
    first_turn = next(i for i, r in enumerate(strict) if r.action in (LEFT, RIGHT))
    assert poses[first_turn].x >= 0.75, "strict: the agent reaches the kink (within one step) before turning"
    loose_first_turn = next(i for i, r in enumerate(loose) if r.action in (LEFT, RIGHT))
    assert loose_first_turn < first_turn, "the 0.5 m lookahead turns early and cuts the corner"


def test_strict_tracking_realigns_to_the_plans_heading_before_walking_on():
    converter = DiscreteActionConverter(HABITAT, STRICT)
    displaced = AgentPose(0.10, 0.0, 0.0, math.radians(60.0), 0.0)         # off the plan's start, 60 degrees off its heading
    command = NavigationCommand.follow(PATH, actions=(FORWARD, FORWARD), plan_pose=(0.0, 0.0, 0.0))
    result = converter.step(displaced, command)
    assert result.action == RIGHT and result.status == STATUS_TURN and not result.planned
    assert result.heading_error_rad == pytest.approx(-math.radians(60.0))
    aligned = AgentPose(0.10, 0.0, 0.0, 0.0, 0.0)                         # heading right, position off: the one-step aim
    result = converter.step(aligned, command)
    assert result.action == FORWARD and not result.planned
    loose = DiscreteActionConverter(HABITAT).step(displaced, command)
    assert loose.action != RIGHT or loose.status == STATUS_TURN, "without strict tracking the ladder decides by its aim"


def test_strict_tracking_disables_the_heading_hysteresis():
    hysteretic = ActionConverterParams(heading_hysteresis_deg=2.0, strict_plan_tracking=True)
    converter = DiscreteActionConverter(HABITAT, hysteretic)
    path = ((0.0, 0.0), (3.0, 0.0))
    converter.step(ORIGIN, NavigationCommand.follow(path))                 # forward
    skewed = AgentPose(0.25, 0.0, 0.0, math.radians(16.0), 0.0)             # a hair past half a turn
    result = converter.step(skewed, NavigationCommand.follow(path))
    assert result.action == RIGHT, "strict: the dead band is exactly half a turn, no hysteresis"
    with pytest.raises(Exception):
        ActionConverterParams(strict_plan_tracking="yes")
