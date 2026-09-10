"""What each kind of command becomes: STOP only on request, a tilt before walking, a heading faced once there.

Each test drives a real converter through one scripted situation and checks
the decision that situation is about: a tuning that cannot work is refused at
construction, STOP comes from anywhere and only on request, the camera tilts
before the walk and never past the simulator's limits, a heading is faced
after arrival, and nothing but the benchmark's own actions is ever emitted.
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
    THOR,
    drive,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUS_ARRIVED,
    STATUS_EMPTY,
    STATUS_FACE,
    STATUS_FORWARD,
    STATUS_PITCH,
    STATUS_STOP,
)
from sparx_agency.core.planning.objnav.errors import CommandError, ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose


# -- construction -------------------------------------------------------------

def test_a_lookahead_shorter_than_one_step_is_refused():
    """The step would overshoot the aim point, and the agent would turn back."""
    with pytest.raises(ObjNavError):
        DiscreteActionConverter(HABITAT, ActionConverterParams(lookahead_m=0.2))


def test_wrong_argument_types_are_refused():
    """A dict of params or a tuple pose would otherwise fail somewhere less clear."""
    with pytest.raises(TypeError):
        DiscreteActionConverter("habitat")
    with pytest.raises(TypeError):
        DiscreteActionConverter(HABITAT, {"lookahead_m": 0.5})
    converter = DiscreteActionConverter(HABITAT)
    with pytest.raises(TypeError):
        converter.step((0.0, 0.0), NavigationCommand.hold())
    with pytest.raises(TypeError):
        converter.step(ORIGIN, "stop")


def test_the_spec_and_default_params_are_exposed():
    """The agent reads the spec back to check every action it is handed."""
    converter = DiscreteActionConverter(HABITAT)
    assert converter.spec is HABITAT
    assert converter.params == ActionConverterParams()


# -- stop -----------------------------------------------------------------------

@pytest.mark.parametrize("pose", [
    ORIGIN,
    AgentPose(7.0, -3.0, z=1.5, yaw=2.0, camera_pitch=0.4),
])
def test_stop_is_emitted_from_anywhere(pose):
    """Ending the episode is the policy's call; the converter never second-guesses it."""
    converter = DiscreteActionConverter(HABITAT)
    converter.step(ORIGIN, NavigationCommand.follow(HAIRPIN))
    result = converter.step(pose, NavigationCommand.stop_here())
    assert (result.action, result.status) == (DiscreteAction.STOP, STATUS_STOP)


def test_a_finished_path_is_idle_never_stop():
    """A converter that stopped when a path ran out would end episodes a metre short."""
    converter = DiscreteActionConverter(HABITAT)
    result = converter.step(AgentPose(3.0, 0.0),
                            NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)]))
    assert (result.action, result.status) == (None, STATUS_ARRIVED)


# -- camera pitch ----------------------------------------------------------------

def test_a_pitch_command_looks_down_before_walking():
    """Tilting first means the walk is seen through the camera the policy asked for."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)],
                                       camera_pitch=math.radians(30.0))
    first = converter.step(ORIGIN, command)
    assert (first.action, first.status) == (DiscreteAction.LOOK_DOWN,
                                            STATUS_PITCH)
    assert first.distance_to_goal_m == pytest.approx(3.0)
    tilted = apply_action(ORIGIN, first.action, HABITAT)
    assert converter.step(tilted, command).action == FORWARD


def test_a_pitch_past_the_limit_stops_at_the_limit():
    """AI2-THOR would fail a LOOK past +30 degrees; the converter never asks for it."""
    converter = DiscreteActionConverter(THOR)
    rollout = converter.rollout(ORIGIN, NavigationCommand.hold(
        camera_pitch=math.radians(60.0)), 10)
    assert rollout.actions == (DiscreteAction.LOOK_DOWN,)
    assert rollout.status == STATUS_ARRIVED
    assert rollout.final_pose.camera_pitch == pytest.approx(math.radians(30.0))


def test_look_up_raises_the_camera():
    """Negative pitch looks up (REP-103)."""
    converter = DiscreteActionConverter(HABITAT)
    result = converter.step(ORIGIN, NavigationCommand.hold(
        camera_pitch=math.radians(-30.0)))
    assert (result.action, result.status) == (DiscreteAction.LOOK_UP,
                                              STATUS_PITCH)


def test_a_pitch_already_reached_is_arrived_not_empty():
    """The command asked for something and got it; empty means it asked for nothing."""
    converter = DiscreteActionConverter(HABITAT)
    result = converter.step(ORIGIN, NavigationCommand.hold(camera_pitch=0.0))
    assert (result.action, result.status) == (None, STATUS_ARRIVED)


def test_a_pitch_command_without_look_actions_is_a_command_error():
    """Silently ignoring it would leave the policy believing the camera moved."""
    converter = DiscreteActionConverter(NO_TILT)
    path = NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)])
    converter.step(ORIGIN, path)
    with pytest.raises(CommandError):
        converter.step(ORIGIN, NavigationCommand.hold(camera_pitch=0.3))
    # Refused before any state changed: the last MOVE_FORWARD is remembered.
    assert converter.step(ORIGIN, path).forward_blocked is True


# -- facing a heading ------------------------------------------------------------

def test_final_yaw_is_faced_after_arrival():
    """Walk first, then turn: facing the heading early would walk the path sideways."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (1.0, 0.0)],
                                       final_yaw=math.pi / 2)
    results, final = drive(converter, ORIGIN, command)
    statuses = [r.status for r in results]
    assert statuses == [STATUS_FORWARD] * 3 + [STATUS_FACE] * 3 + [
        STATUS_ARRIVED]
    assert all(r.action == LEFT for r in results if r.status == STATUS_FACE)
    assert results[3].heading_error_rad == pytest.approx(math.pi / 2)
    assert abs(final.yaw - math.pi / 2) <= math.radians(15.0)


def test_a_hold_turns_to_the_commanded_heading():
    """A hold with a heading turns in place, the short way, and then idles."""
    converter = DiscreteActionConverter(HABITAT)
    rollout = converter.rollout(ORIGIN, NavigationCommand.hold(
        final_yaw=-math.pi / 2), 10)
    assert rollout.actions == (RIGHT, RIGHT, RIGHT)
    assert rollout.status == STATUS_ARRIVED


def test_follow_with_a_heading_and_stop_on_arrival_stops_right_after_facing():
    """The idle turn a step later would undo the facing, and RoboTHOR needs the object in view in the STOP frame."""
    converter = DiscreteActionConverter(HABITAT)
    command = NavigationCommand.follow([(0.0, 0.0), (1.0, 0.0)],
                                       final_yaw=math.pi / 2,
                                       stop_on_arrival=True)
    results, final = drive(converter, ORIGIN, command)
    assert [r.status for r in results] == [STATUS_FORWARD] * 3 + [
        STATUS_FACE] * 3 + [STATUS_STOP]
    assert results[-1].action == DiscreteAction.STOP
    assert abs(final.yaw - math.pi / 2) <= math.radians(15.0)


@pytest.mark.parametrize("command, before_stop", [
    (NavigationCommand.hold(final_yaw=-math.pi / 2, stop_on_arrival=True),
     (RIGHT, RIGHT, RIGHT)),
    (NavigationCommand.hold(camera_pitch=math.radians(30.0),
                            stop_on_arrival=True),
     (DiscreteAction.LOOK_DOWN,)),
], ids=["heading", "pitch"])
def test_a_hold_with_stop_on_arrival_reaches_its_target_then_stops(command,
                                                                   before_stop):
    """STOP comes on the step the hold is satisfied: the policy's own decision, made in advance."""
    rollout = DiscreteActionConverter(HABITAT).rollout(ORIGIN, command, 10)
    assert rollout.actions == before_stop + (DiscreteAction.STOP,)
    assert rollout.status == STATUS_STOP


def test_an_empty_command_is_idle_with_status_empty():
    """No opinion from the policy is not arrival, and never STOP."""
    converter = DiscreteActionConverter(HABITAT)
    result = converter.step(ORIGIN, NavigationCommand.hold())
    assert (result.action, result.status, result.idle) == (
        None, STATUS_EMPTY, True)
    rollout = converter.rollout(ORIGIN, NavigationCommand.hold(), 5)
    assert (rollout.actions, rollout.status) == ((), STATUS_EMPTY)


# -- the action set ---------------------------------------------------------------

def test_only_the_benchmark_actions_are_ever_emitted():
    """Without camera tilt, a path with a corner and a heading uses only the four."""
    converter = DiscreteActionConverter(NO_TILT)
    rollout = converter.rollout(ORIGIN, NavigationCommand.follow(
        HAIRPIN, final_yaw=1.0), 200)
    assert set(rollout.actions) <= {FORWARD, LEFT, RIGHT}
    assert rollout.status == STATUS_ARRIVED
