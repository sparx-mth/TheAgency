"""InternVLA-N1 behind the uniform NavigationPolicy contract."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.vlas.interfaces.goals import LanguageGoal, PointGoal
from sparx_agency.core.planning.vlas.interfaces.policy import PolicyObservation
from sparx_agency.core.planning.vlas.internvla_n1.policy import InternVLAN1Policy
from sparx_agency.core.planning.vlas.internvla_n1.types import StepResponse


class _FakeClient:
    """Stand-in for ModelClient: records what it was sent, returns a canned step."""

    def __init__(self, response):
        self._response = response
        self.last_rgb = None
        self.last_instruction = None
        self.last_depth = None
        self.init_called = False
        self.reset_called = False

    def init_agent(self, **kwargs):
        self.init_called = True
        return True

    def reset(self, reset_index=None):
        self.reset_called = True
        return True

    def step(self, rgb, instruction, depth=None):
        self.last_rgb = rgb
        self.last_instruction = instruction
        self.last_depth = depth
        return self._response


def _policy(response):
    policy = InternVLAN1Policy(url="http://127.0.0.1:8087")
    policy.client = _FakeClient(response)
    return policy


def _rgb():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_step_returns_the_continuous_trajectory_when_the_server_gives_one():
    raw = {"trajectory": [[[0.0, 0.0], [0.25, 0.0], [0.5, 0.1]]]}
    policy = _policy(StepResponse(action="MOVE_FORWARD", action_index=1, raw_response=raw))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.ok
    assert result.trajectory.shape == (3, 3)
    assert result.metadata["action"] == "MOVE_FORWARD"


def test_step_falls_back_to_the_action_when_no_curve_is_returned():
    policy = _policy(StepResponse(action="TURN_LEFT", action_index=2, raw_response={}))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.ok
    # a left turn bends the single step to the left (+y)
    assert result.trajectory[-1, 1] > 0.0


def test_step_flies_the_continuous_trajectory_from_the_patched_server():
    # The trajectory-patched InternNav server returns the continuous curve inside
    # action[0]; the policy must prefer it over the discrete action.
    raw = {
        "action": [{"action": [1], "ideal_flag": True,
                    "trajectory": [[0.0, 0.0], [0.25, 0.0], [0.5, 0.1], [0.75, 0.25]]}],
        "pixel_goal": None, "pixel_goal_step": -1,
    }
    policy = _policy(StepResponse(action="MOVE_FORWARD", action_index=1, raw_response=raw))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.ok
    # a 4-point curve, not the 2-point action step — the continuous trajectory won
    assert result.trajectory.shape == (4, 3)


def test_step_stop_yields_a_not_ok_result_asking_to_stop():
    policy = _policy(StepResponse(action="STOP", action_index=0, raw_response={}))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert not result.ok
    assert result.stop is True


def test_step_surfaces_system1_and_system2_timings():
    raw = {"action": [{"action": [1], "s1_ms": 43.4, "s2_ms": 707.9,
                       "trajectory": [[0.0, 0.0], [0.25, 0.0]]}]}
    policy = _policy(StepResponse(action="MOVE_FORWARD", action_index=1, raw_response=raw))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.metadata["s1_ms"] == pytest.approx(43.4)
    assert result.metadata["s2_ms"] == pytest.approx(707.9)


def test_step_transport_failure_is_a_not_ok_result_not_an_exception():
    policy = _policy(StepResponse(success=False, error="timeout"))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert not result.ok
    assert result.metadata["transport_failed"] is True


def test_step_converts_rgb_to_bgr_for_the_server():
    policy = _policy(StepResponse(action="STOP", action_index=0, raw_response={}))
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 10   # R
    rgb[..., 2] = 200  # B
    policy.step(PolicyObservation(rgb=rgb), LanguageGoal(instruction="go"))
    sent = policy.client.last_rgb
    assert sent[0, 0, 0] == 200 and sent[0, 0, 2] == 10  # channels flipped


def test_step_rejects_a_non_language_goal():
    policy = _policy(StepResponse(action="STOP", action_index=0))
    with pytest.raises(TypeError):
        policy.step(PolicyObservation(rgb=_rgb()), PointGoal(forward_m=1.0))


def test_step_requires_an_rgb_frame():
    policy = _policy(StepResponse(action="STOP", action_index=0))
    with pytest.raises(ValueError):
        policy.step(PolicyObservation(rgb=None), LanguageGoal(instruction="go"))


def test_reset_initialises_and_clears_the_server_agent():
    policy = _policy(StepResponse())
    assert policy.reset() is True
    assert policy.client.init_called and policy.client.reset_called


def test_registry_creates_the_policy_by_name():
    from sparx_agency.core.planning.vlas.registry import default_vla_registry
    registry = default_vla_registry()
    assert "internvla_n1" in registry.names()
    policy = registry.create("internvla_n1", url="http://127.0.0.1:8087")
    assert isinstance(policy, InternVLAN1Policy)

