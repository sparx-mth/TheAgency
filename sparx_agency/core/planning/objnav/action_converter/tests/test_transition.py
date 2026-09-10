"""What one action does to the pose: the converter's model of the simulator.

The model has to be exactly as unkind as the simulators it stands for: a LOOK
past the limits fails rather than clamps (AI2-THOR), TURN_LEFT is
counter-clockwise (Habitat), STOP moves nothing and no action changes the
floor height. A kinder model would predict poses the agent never reaches.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.errors import CommandError
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS,
    NAVIGATION_ACTIONS,
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.pose import AgentPose

HABITAT = DiscreteActionSpec()
THOR = DiscreteActionSpec(min_pitch_deg=-30.0, max_pitch_deg=30.0)


def test_move_forward_steps_along_the_heading():
    """Facing north, a step moves north and changes nothing else."""
    pose = AgentPose(1.0, 2.0, z=0.5, yaw=math.pi / 2, camera_pitch=0.2)
    after = apply_action(pose, DiscreteAction.MOVE_FORWARD, HABITAT)
    assert after.x == pytest.approx(1.0)
    assert after.y == pytest.approx(2.25)
    assert (after.z, after.yaw, after.camera_pitch) == (0.5, math.pi / 2, 0.2)


def test_move_forward_uses_the_spec_step():
    """A spec that disagrees with the simulator is wrong by this much every step."""
    spec = DiscreteActionSpec(forward_step_m=0.5)
    after = apply_action(AgentPose(0.0, 0.0), DiscreteAction.MOVE_FORWARD,
                         spec)
    assert (after.x, after.y) == (0.5, 0.0)


def test_turn_left_is_counter_clockwise_and_wraps():
    """Habitat's TURN_LEFT increases yaw; the result stays in (-pi, pi]."""
    pose = AgentPose(0.0, 0.0, yaw=math.radians(170.0))
    after = apply_action(pose, DiscreteAction.TURN_LEFT, HABITAT)
    assert after.yaw == pytest.approx(math.radians(-160.0))


def test_turn_right_is_clockwise():
    """The mirror image: yaw decreases by one turn."""
    after = apply_action(AgentPose(0.0, 0.0), DiscreteAction.TURN_RIGHT,
                         HABITAT)
    assert after.yaw == pytest.approx(math.radians(-30.0))


def test_look_down_adds_the_tilt_and_look_up_subtracts_it():
    """REP-103: positive pitch looks down."""
    pose = AgentPose(0.0, 0.0)
    assert apply_action(pose, DiscreteAction.LOOK_DOWN, HABITAT) \
        .camera_pitch == pytest.approx(math.radians(30.0))
    assert apply_action(pose, DiscreteAction.LOOK_UP, HABITAT) \
        .camera_pitch == pytest.approx(math.radians(-30.0))


def test_a_look_past_the_limit_leaves_the_pitch_unchanged():
    """AI2-THOR fails it: the step is spent and the camera does not move."""
    down = AgentPose(0.0, 0.0, camera_pitch=math.radians(30.0))
    up = AgentPose(0.0, 0.0, camera_pitch=math.radians(-30.0))
    assert apply_action(down, DiscreteAction.LOOK_DOWN, THOR) is down
    assert apply_action(up, DiscreteAction.LOOK_UP, THOR) is up


def test_a_look_to_exactly_the_limit_is_executed():
    """Refusing the one LOOK that lands on the limit would make it unreachable."""
    after = apply_action(AgentPose(0.0, 0.0), DiscreteAction.LOOK_DOWN, THOR)
    assert after.camera_pitch == pytest.approx(math.radians(30.0))


def test_without_limits_looks_keep_going():
    """Habitat does not clamp LOOK at all, so neither may the model."""
    pose = AgentPose(0.0, 0.0)
    for _ in range(3):
        pose = apply_action(pose, DiscreteAction.LOOK_DOWN, HABITAT)
    assert pose.camera_pitch == pytest.approx(math.radians(90.0))


def test_stop_changes_nothing():
    """STOP ends the episode where the agent stands."""
    pose = AgentPose(1.0, 2.0, z=0.3, yaw=0.4, camera_pitch=0.1)
    assert apply_action(pose, DiscreteAction.STOP, HABITAT) is pose


@pytest.mark.parametrize("action", ALL_ACTIONS)
def test_no_action_changes_the_floor_height(action):
    """z is the environment's to report; the model never invents a floor."""
    pose = AgentPose(1.0, 2.0, z=3.0, yaw=0.4)
    assert apply_action(pose, action, HABITAT).z == 3.0


def test_an_action_the_benchmark_lacks_is_refused():
    """A LOOK on a benchmark without camera tilt is a planning bug, not a no-op."""
    spec = DiscreteActionSpec(actions=NAVIGATION_ACTIONS)
    with pytest.raises(CommandError):
        apply_action(AgentPose(0.0, 0.0), DiscreteAction.LOOK_UP, spec)


def test_a_bare_integer_is_not_an_action():
    """1 is MOVE_FORWARD in Habitat's order only; the model refuses to guess."""
    with pytest.raises(TypeError):
        apply_action(AgentPose(0.0, 0.0), 1, HABITAT)


def test_wrong_argument_types_are_refused():
    """A Pose2D or a plain dict would otherwise fail later, somewhere less clear."""
    with pytest.raises(TypeError):
        apply_action((0.0, 0.0), DiscreteAction.STOP, HABITAT)
    with pytest.raises(TypeError):
        apply_action(AgentPose(0.0, 0.0), DiscreteAction.STOP, {"step": 0.25})
