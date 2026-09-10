"""A blocked forward step is reported, and the corner is not cut into the same wall again.

``forward_blocked`` is the one definition of a blocked step, so its truth table
is pinned here. The recovery is tested in a world defined in this file: an
L-shaped path whose inside corner is a wall 10 cm from both legs, and a step
loop in which a MOVE_FORWARD that would end inside the wall leaves the pose
where it was. The default lookahead cuts that corner into the wall; without
the recovery the same pose gives the same decision, the same step into the
same wall, until the step budget runs out. With it the converter aims one
step ahead until a forward step moves, and completes the path.
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
    NO_TILT,
    ORIGIN,
    RIGHT,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUS_ARRIVED,
)
from sparx_agency.core.planning.objnav.errors import CommandError, ObjNavError
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: Three metres east from the origin.
EAST = NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)])
#: An L: 2 m east, then 2 m north round a left-hand corner.
CORNER = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0))
#: Where the corner walk starts: an approach that reaches the turn at a phase
#: where the default lookahead's cut is deep enough to hit the wall.
CORNER_START = AgentPose(0.12, 0.0)
#: How far the policy's path keeps from the wall inside the corner, metres.
CLEARANCE_M = 0.1


def in_wall(pose) -> bool:
    """Whether ``pose`` is inside the wall filling the inside of :data:`CORNER`."""
    return pose.x < 2.0 - CLEARANCE_M and pose.y > CLEARANCE_M


def walk_with_walls(converter, pose, command, blocked_by, limit=60):
    """Step under perfect execution, except that a MOVE_FORWARD ending in ``blocked_by`` does not move.

    Returns:
        ``(results, poses)``: ``poses[i]`` is the pose ``results[i]`` was
        decided at.
    """
    results, poses = [], []
    for _ in range(limit):
        result = converter.step(pose, command)
        results.append(result)
        poses.append(pose)
        if result.idle or result.action == DiscreteAction.STOP:
            return results, poses
        after = apply_action(pose, result.action, converter.spec)
        if not (result.action == FORWARD and blocked_by(after)):
            pose = after
    raise AssertionError("the converter did not finish in %d steps" % limit)


# -- reporting a blocked step -------------------------------------------------------

def test_a_forward_step_that_did_not_move_is_reported_blocked():
    """A wall the model could not know about; reported, and left to the policy."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)])
    assert converter.step(ORIGIN, command).action == FORWARD
    blocked = converter.step(ORIGIN, command)
    assert blocked.forward_blocked is True
    assert blocked.action == FORWARD
    moved = converter.step(AgentPose(0.25, 0.0), command)
    assert moved.forward_blocked is False


def test_a_step_shorter_than_the_epsilon_counts_as_blocked():
    """Habitat realises a partial move into a wall; a few millimetres is not progress."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)])
    converter.step(ORIGIN, command)
    assert converter.step(AgentPose(0.005, 0.0), command).forward_blocked


def test_a_turn_is_never_reported_blocked():
    """Turning in place does not move the agent, and is not supposed to."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (-3.0, 0.0)])
    assert converter.step(ORIGIN, command).action == LEFT
    assert converter.step(ORIGIN, command).forward_blocked is False


def test_reset_forgets_the_path_and_the_last_action():
    """A new episode must not inherit a blocked flag or a cursor."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow(HAIRPIN)
    converter.step(AgentPose(2.0, 1.0, yaw=math.pi), command)
    converter.reset()
    result = converter.step(AgentPose(2.0, 0.4, yaw=math.pi), command)
    assert result.segment_index == 0
    assert result.forward_blocked is False


#: What each ``last`` in the truth table does: the pose, the command, the action.
LAST_DECISIONS = {
    "forward": (ORIGIN, EAST, FORWARD),
    "turn": (ORIGIN, NavigationCommand.follow([(0.0, 0.0), (-3.0, 0.0)]),
             LEFT),
    "look": (ORIGIN, NavigationCommand.hold(camera_pitch=0.5),
             DiscreteAction.LOOK_DOWN),
    "stop": (ORIGIN, NavigationCommand.stop_here(), DiscreteAction.STOP),
    "idle": (AgentPose(3.0, 0.0), EAST, None),
}


@pytest.mark.parametrize("last, moved, blocked", [
    ("nothing", (0.0, 0.0, 0.0), False),
    ("forward", (0.0, 0.0, 0.0), True),
    ("forward", (0.005, 0.0, 0.0), True),
    ("forward", (0.004, 0.0, 0.003), True),
    ("forward", (0.0, 0.0, 0.02), False),
    ("forward", (0.25, 0.0, 0.0), False),
    ("turn", (0.0, 0.0, 0.0), False),
    ("look", (0.0, 0.0, 0.0), False),
    ("stop", (0.0, 0.0, 0.0), False),
    ("idle", (0.0, 0.0, 0.0), False),
], ids=["before-any-action", "forward-did-not-move", "forward-moved-5mm",
        "forward-moved-5mm-in-3d", "forward-climbed-2cm",
        "forward-moved-a-step", "after-a-turn", "after-a-look", "after-stop",
        "after-an-idle-step"])
def test_forward_blocked_holds_only_after_a_forward_step_that_moved_less_than_epsilon(
        last, moved, blocked):
    """The one definition the converter recovers with and the agent notifies with; a move counts in 3-D."""
    converter = DiscreteActionConverter(HABITAT)
    pose = ORIGIN
    if last != "nothing":
        pose, command, action = LAST_DECISIONS[last]
        assert converter.step(pose, command).action == action
    dx, dy, dz = moved
    now = AgentPose(pose.x + dx, pose.y + dy, pose.z + dz, pose.yaw,
                    pose.camera_pitch)
    assert converter.forward_blocked(now) is blocked


def test_asking_forward_blocked_first_changes_nothing():
    """The agent asks before planning; the question must not alter the step that follows."""
    asked, unasked = (DiscreteActionConverter(HABITAT),
                      DiscreteActionConverter(HABITAT))
    for converter in (asked, unasked):
        converter.step(ORIGIN, EAST)
    assert asked.forward_blocked(ORIGIN) and asked.forward_blocked(ORIGIN)
    assert asked.step(ORIGIN, EAST) == unasked.step(ORIGIN, EAST)


def test_forward_blocked_refuses_a_pose_of_the_wrong_type():
    """A tuple pose would fail on an attribute, far from the caller's mistake."""
    with pytest.raises(TypeError):
        DiscreteActionConverter(HABITAT).forward_blocked((0.0, 0.0))


def test_a_stop_right_after_a_blocked_step_still_reports_it():
    """forward_blocked is a fact about the last action, whatever this step's command."""
    converter = DiscreteActionConverter(HABITAT)
    converter.step(ORIGIN, EAST)
    stop = converter.step(ORIGIN, NavigationCommand.stop_here())
    assert (stop.action, stop.forward_blocked) == (DiscreteAction.STOP, True)


@pytest.mark.parametrize("spec, params", [
    (HABITAT, ActionConverterParams(blocked_epsilon_m=0.25)),
    (HABITAT, ActionConverterParams(blocked_epsilon_m=0.3)),
    (DiscreteActionSpec(forward_step_m=0.005), ActionConverterParams()),
], ids=["epsilon-one-step", "epsilon-past-a-step", "step-below-epsilon"])
def test_a_blocked_epsilon_not_shorter_than_a_step_is_refused(spec, params):
    """Every unobstructed step would read as blocked, and the policy be told of a wall every step."""
    with pytest.raises(ObjNavError, match="blocked_epsilon_m"):
        DiscreteActionConverter(spec, params)


# -- recovering from it -------------------------------------------------------------

def test_a_corner_cut_into_a_wall_is_recovered_from_and_the_path_completed():
    """The default lookahead cuts this corner into the wall; aiming one step ahead gets round it."""
    converter = DiscreteActionConverter(HABITAT)
    results, poses = walk_with_walls(converter, CORNER_START,
                                     NavigationCommand.follow(CORNER), in_wall)
    assert any(result.forward_blocked for result in results), (
        "the default lookahead never hit the wall; the scenario tests nothing")
    assert not any(first.forward_blocked and second.forward_blocked
                   for first, second in zip(results, results[1:])), (
        "the same blocked step was taken twice in a row")
    assert results[-1].status == STATUS_ARRIVED
    end = poses[-1]
    assert (math.hypot(end.x - 2.0, end.y - 2.0)
            <= converter.params.goal_tolerance_m)
    assert len(results) <= 30


def test_without_the_recovery_the_same_blocked_step_would_repeat():
    """The livelock the recovery breaks: a converter that forgot the block steps into the same wall."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow(CORNER)
    results, poses = walk_with_walls(converter, CORNER_START, command, in_wall)
    first = next(index for index, result in enumerate(results)
                 if result.forward_blocked)
    stuck = poses[first]
    assert (results[first - 1].action, poses[first - 1]) == (FORWARD, stuck)
    forgetful = DiscreteActionConverter(HABITAT).step(stuck, command)
    assert forgetful.action == FORWARD
    assert in_wall(apply_action(stuck, FORWARD, HABITAT))
    assert results[first].action != FORWARD


def test_the_recovery_aims_one_step_ahead_until_a_forward_step_moves():
    """Blocked again keeps the short aim; the first forward step that moves restores the lookahead."""
    converter = DiscreteActionConverter(HABITAT)
    first = converter.step(ORIGIN, EAST)
    assert first.target_xy == pytest.approx((0.5, 0.0))
    assert not converter.recovering
    for _ in range(2):
        blocked = converter.step(ORIGIN, EAST)
        assert blocked.forward_blocked and converter.recovering
        assert blocked.target_xy == pytest.approx((0.25, 0.0))
        assert blocked.action == FORWARD
    moved = converter.step(AgentPose(0.25, 0.0), EAST)
    assert not moved.forward_blocked and not converter.recovering
    assert moved.target_xy == pytest.approx((0.75, 0.0))


def test_a_turn_does_not_end_the_recovery():
    """Only a forward step that moves is evidence the wall is behind; turning in place is not."""
    converter = DiscreteActionConverter(HABITAT)
    start = AgentPose(0.0, 0.1)
    assert converter.step(start, EAST).action == FORWARD
    blocked = converter.step(start, EAST)
    assert blocked.forward_blocked and blocked.action == RIGHT
    turned = apply_action(start, RIGHT, HABITAT)
    after_turn = converter.step(turned, EAST)
    assert converter.recovering
    assert (after_turn.action, after_turn.target_xy) == (
        FORWARD, pytest.approx((0.25, 0.0)))
    moved = apply_action(turned, FORWARD, HABITAT)
    released = converter.step(moved, EAST)
    assert not converter.recovering
    assert released.target_xy == pytest.approx((moved.x + 0.5, 0.0))


def test_a_new_path_does_not_end_the_recovery():
    """A re-plan usually cuts the same corner, so a new path keeps the short aim."""
    converter = DiscreteActionConverter(HABITAT)
    start = AgentPose(0.0, 0.1)
    converter.step(start, EAST)
    assert converter.step(start, EAST).action == RIGHT
    turned = apply_action(start, RIGHT, HABITAT)
    longer = NavigationCommand.follow([(0.0, 0.0), (4.0, 0.0)])
    result = converter.step(turned, longer)
    assert converter.recovering
    assert result.target_xy == pytest.approx((0.25, 0.0))


def test_reset_ends_the_recovery():
    """Reset forgets everything: a new episode starts elsewhere, and a kept recovery would make results depend on episode order."""
    converter = DiscreteActionConverter(HABITAT)
    converter.step(ORIGIN, EAST)
    assert converter.step(ORIGIN, EAST).forward_blocked
    assert converter.recovering
    converter.reset()
    assert not converter.recovering
    result = converter.step(ORIGIN, EAST)
    assert not result.forward_blocked
    assert result.target_xy == pytest.approx((0.5, 0.0))
    assert result == DiscreteActionConverter(HABITAT).step(ORIGIN, EAST)


def test_a_rejected_command_does_not_end_the_recovery():
    """A refused command changes nothing, so the forward step before it still decides the next one."""
    converter = DiscreteActionConverter(NO_TILT)
    converter.step(ORIGIN, EAST)
    converter.step(ORIGIN, EAST)
    moved = AgentPose(0.25, 0.0)
    with pytest.raises(CommandError):
        converter.step(moved, NavigationCommand.hold(camera_pitch=0.3))
    assert converter.recovering
    assert converter.step(moved, EAST).forward_blocked is False
    assert not converter.recovering
