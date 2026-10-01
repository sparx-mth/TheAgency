"""Small path-only hysteresis prevents reversing turn noise without bypassing safety."""
import math
from dataclasses import replace

import pytest

from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.actions import ALL_ACTIONS, DiscreteAction, DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def rig(margin=1.0):
    spec = DiscreteActionSpec(forward_step_m=.25, turn_angle_deg=30, tilt_angle_deg=30,
                              min_pitch_deg=-30, max_pitch_deg=60, actions=ALL_ACTIONS)
    converter = DiscreteActionConverter(spec, ActionConverterParams(heading_hysteresis_deg=margin))
    return converter, NavigationCommand.follow([(0, 0), (3, 0)])


def test_reversing_turn_within_hysteresis_prefers_forward():
    converter, command = rig()
    assert converter.step(AgentPose(0, 0, 0, math.radians(16)), command).action == DiscreteAction.TURN_RIGHT
    result = converter.step(AgentPose(0, 0, 0, math.radians(-15.2)), command)
    assert result.action == DiscreteAction.MOVE_FORWARD and result.status == "forward"
    assert math.degrees(result.heading_error_rad) == pytest.approx(15.2)


def test_forward_progress_keeps_forward_inside_wider_band():
    converter, command = rig()
    assert converter.step(AgentPose(0, 0, 0, 0), command).action == DiscreteAction.MOVE_FORWARD
    assert converter.step(AgentPose(.25, 0, 0, math.radians(-15.5)), command).action == DiscreteAction.MOVE_FORWARD


@pytest.mark.parametrize("mode", ["disabled", "large_error", "new_path", "final_yaw"])
def test_hysteresis_never_overrides_real_heading_changes(mode):
    converter, command = rig(0 if mode == "disabled" else 1)
    converter.step(AgentPose(0, 0, 0, math.radians(16)), command)
    if mode == "new_path":
        command = NavigationCommand.follow([(0, 0), (2, 0), (3, 0)])
    if mode == "final_yaw":
        command = NavigationCommand.hold(final_yaw=0)
    angle = -17 if mode == "large_error" else -15.2
    assert converter.step(AgentPose(0, 0, 0, math.radians(angle)), command).action == DiscreteAction.TURN_LEFT


def test_blocked_forward_and_pitch_and_stop_keep_priority():
    converter, command = rig()
    converter.step(AgentPose(0, 0, 0, 0), command)
    pose = AgentPose(0, 0, 0, math.radians(-15.2))
    result = converter.step(pose, command)
    assert result.forward_blocked and result.action == DiscreteAction.TURN_LEFT
    assert converter.step(pose, replace(command, camera_pitch=math.pi / 6)).action == DiscreteAction.LOOK_DOWN
    assert converter.step(pose, NavigationCommand.stop_here()).action == DiscreteAction.STOP


@pytest.mark.parametrize("margin", [-1, 3, float("nan"), True])
def test_hysteresis_margin_is_bounded(margin):
    with pytest.raises(ValueError):
        ActionConverterParams(heading_hysteresis_deg=margin)

