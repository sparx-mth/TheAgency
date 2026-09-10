"""Each step the headless agent returns the converter's action, validates the frame, and never stops on its own.

Each test drives a real agent -- a real converter inside -- with the scripted
policy and the two-category mapper of
:mod:`~sparx_agency.core.planning.objnav.agent.tests.helpers`, and checks one
promise the harness relies on: the converter's action comes back with its
reasons, STOP only when the policy asks, an idle step is spent on the idle
action, and a frame from another episode or out of order is refused.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import math

import pytest

import sparx_agency.core.planning.objnav.agent.headless_agent as headless_agent_module
from sparx_agency.core.planning.objnav.action_converter.converter import (
    DiscreteActionConverter,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUS_ARRIVED,
    STATUS_EMPTY,
    STATUS_FORWARD,
    STATUS_PITCH,
    STATUS_STOP,
    ConversionResult,
)
from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams
from sparx_agency.core.planning.objnav.agent.tests.helpers import (
    AHEAD,
    FORWARD,
    HABITAT,
    LEFT,
    NO_TILT,
    ORIGIN,
    RIGHT,
    STOP,
    agent_with,
    camera,
    episode,
    observation,
    started,
)
from sparx_agency.core.planning.objnav.errors import (
    CommandError,
    ObjNavError,
    ObjNavInternalError,
    ObservationError,
)
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: The keys of every decision's ``info``, and nothing else.
STEP_INFO_KEYS = {"status", "idle", "segment_index", "heading_error_rad",
                  "distance_to_goal_m", "remaining_path_m", "cross_track_m",
                  "forward_blocked", "policy"}


def test_the_converters_action_is_returned_with_its_reasons():
    """The agent adds validation and bookkeeping, never a different decision."""
    agent = started(AHEAD)
    decision = agent.act(observation())
    expected = DiscreteActionConverter(HABITAT).step(ORIGIN, AHEAD)
    assert decision.action == expected.action == FORWARD
    assert set(decision.info) == STEP_INFO_KEYS
    assert decision.info["status"] == STATUS_FORWARD
    assert decision.info["idle"] is False
    assert decision.info["distance_to_goal_m"] == pytest.approx(3.0)
    assert decision.info["remaining_path_m"] == pytest.approx(3.0)
    turned = started(NavigationCommand.follow([(0.0, 3.0)])).act(observation())
    assert turned.action == LEFT
    assert turned.info["heading_error_rad"] == pytest.approx(math.pi / 2)


def test_the_policys_diagnostics_ride_along_as_a_copy():
    """Logged per step; a shared dict would let the log rewrite the command."""
    command = NavigationCommand.follow([(3.0, 0.0)], info={"room": "kitchen"})
    decision = started(command).act(observation())
    assert decision.info["policy"] == {"room": "kitchen"}
    decision.info["policy"]["room"] = "hall"
    assert command.info == {"room": "kitchen"}


def test_stop_is_returned_when_the_policy_asks():
    """Ending the episode is the policy's call, from wherever the agent stands."""
    decision = started(NavigationCommand.stop_here()).act(
        observation(pose=AgentPose(4.0, -2.0, yaw=1.0)))
    assert decision.action == STOP
    assert decision.info["status"] == STATUS_STOP


def test_stop_on_arrival_stops_on_the_step_the_heading_is_faced_with_no_idle_turn_between():
    """An idle TURN_LEFT after the facing would swing the object out of RoboTHOR's STOP frame."""
    agent = started(NavigationCommand.hold(final_yaw=math.radians(60.0),
                                           stop_on_arrival=True))
    pose, actions = ORIGIN, []
    for step in range(3):
        decision = agent.act(observation(step=step, pose=pose))
        actions.append(decision.action)
        pose = apply_action(pose, decision.action, HABITAT)
    assert actions == [LEFT, LEFT, STOP]
    assert decision.info["status"] == STATUS_STOP
    assert pose.yaw == pytest.approx(math.radians(60.0))


@pytest.mark.parametrize("idle", [LEFT, RIGHT])
def test_an_exhausted_path_spends_the_idle_action_never_stop(idle):
    """A path that ran out is not the policy saying stop; STOP there ends episodes short."""
    agent = started(AHEAD, params=HeadlessAgentParams(idle_action=idle))
    decision = agent.act(observation(pose=AgentPose(3.0, 0.0)))
    assert decision.action == idle
    assert (decision.info["status"], decision.info["idle"]) == (
        STATUS_ARRIVED, True)


def test_an_empty_command_spends_the_idle_action():
    """No opinion from the policy still spends the step, and still not on STOP."""
    decision = started(NavigationCommand.hold()).act(observation())
    assert decision.action == LEFT
    assert (decision.info["status"], decision.info["idle"]) == (
        STATUS_EMPTY, True)


def test_act_before_reset_is_refused():
    """Without an episode there is no target, camera or action set to check against."""
    with pytest.raises(ObjNavError):
        agent_with().act(observation())


def test_something_that_is_not_an_observation_is_refused():
    """A raw simulator dict would reach the policy unvalidated."""
    with pytest.raises(ObservationError):
        started().act({"rgb": None, "depth": None})


def test_an_observation_of_another_target_is_refused():
    """A frame from another episode would steer the search toward the wrong object."""
    with pytest.raises(ObservationError, match="chair"):
        started().act(observation(target_category="chair"))


def test_an_observation_from_another_camera_is_refused():
    """Every depth pixel would be back-projected with the wrong geometry."""
    with pytest.raises(ObservationError):
        started().act(observation(camera=camera(height_m=1.5)))


@pytest.mark.parametrize("second", [7, 3])
def test_a_step_that_does_not_advance_is_refused(second):
    """A repeated or older frame means the environment did not execute the last action."""
    agent = started()
    agent.act(observation(step=7))
    with pytest.raises(ObservationError):
        agent.act(observation(step=second))
    assert agent.act(observation(step=8)).action == LEFT


def test_a_policy_that_returns_a_non_command_is_a_type_error():
    """A policy emitting a bare action would bypass the converter and the spec check."""
    with pytest.raises(TypeError, match="NavigationCommand"):
        started(DiscreteAction.MOVE_FORWARD).act(observation())


def test_a_pitch_command_on_a_benchmark_without_tilt_is_a_command_error():
    """Silently dropping it would leave the policy believing the camera moved."""
    agent = started(NavigationCommand.hold(camera_pitch=0.5))
    agent.reset(episode(action_spec=NO_TILT))
    with pytest.raises(CommandError):
        agent.act(observation())


def test_an_action_outside_the_action_set_never_leaves_the_agent(monkeypatch):
    """The last guard before the simulator, which would refuse it mid-run."""
    class Rogue(DiscreteActionConverter):
        def step(self, pose, command):
            return ConversionResult(DiscreteAction.LOOK_DOWN, STATUS_PITCH)

    monkeypatch.setattr(headless_agent_module, "DiscreteActionConverter", Rogue)
    agent = agent_with()
    agent.reset(episode(action_spec=NO_TILT))
    with pytest.raises(ObjNavInternalError, match="LOOK_DOWN"):
        agent.act(observation())


# -- a whole episode --------------------------------------------------------------

def test_a_whole_episode_walks_the_path_idles_at_its_end_and_stops_on_request():
    """The loop the harness runs, under perfect execution, end to end."""
    path = NavigationCommand.follow([(0.0, 0.0), (1.0, 0.0)])
    agent = started(path, path, path, path, path, NavigationCommand.stop_here())
    pose, actions, statuses = ORIGIN, [], []
    for step in range(6):
        decision = agent.act(observation(step=step, pose=pose))
        actions.append(decision.action)
        statuses.append(decision.info["status"])
        pose = apply_action(pose, decision.action, HABITAT)
    assert actions == [FORWARD, FORWARD, FORWARD, LEFT, LEFT, STOP]
    assert statuses == [STATUS_FORWARD] * 3 + [STATUS_ARRIVED] * 2 + [
        STATUS_STOP]
    assert math.hypot(pose.x - 1.0, pose.y) <= 0.25
