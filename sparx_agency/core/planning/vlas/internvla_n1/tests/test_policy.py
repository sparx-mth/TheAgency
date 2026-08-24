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



# ── an idle tick is not a stop ───────────────────────────────────────────
#
# The agent reports action index -1 while System 2 is looking down, and again
# whenever System 1 returns no actions. Neither means the task is finished, but
# a plain `INDEX_TO_ACTION.get(idx, "STOP")` turns both into "STOP" -- and a
# runner that believes it throws away the route it is halfway through flying.
# Measured over five hospital flights: System 2 emitted a real STOP zero times
# and the agent emitted -1 seventeen times, so every one of those was a route
# abandoned for nothing.


def test_look_down_is_idle_not_a_stop():
    policy = _policy(StepResponse(action="LOOK_DOWN", action_index=5, raw_response={}))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.metadata["idle"] is True
    assert result.stop is False


def test_no_action_is_idle_not_a_stop():
    policy = _policy(StepResponse(action="NO_ACTION", action_index=-1, raw_response={}))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.metadata["idle"] is True
    assert result.stop is False
    assert result.trajectory is None, "an idle tick carries no route of its own"


def test_a_real_stop_is_still_a_stop():
    """Index 0 is System 2 literally answering STOP; that one is terminal."""
    policy = _policy(StepResponse(action="STOP", action_index=0, raw_response={}))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.stop is True
    assert result.metadata["idle"] is False


def test_a_moving_action_is_neither_idle_nor_a_stop():
    policy = _policy(StepResponse(action="TURN_RIGHT", action_index=3, raw_response={}))
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.metadata["idle"] is False
    assert result.stop is False
    assert result.ok


def test_from_curve_distinguishes_the_two_producers():
    raw = {"trajectory": [[[0.0, 0.0], [0.3, 0.0]]]}
    curve = _policy(StepResponse(action="MOVE_FORWARD", action_index=1, raw_response=raw))
    step = _policy(StepResponse(action="MOVE_FORWARD", action_index=1, raw_response={}))
    goal = LanguageGoal(instruction="go")
    assert curve.step(PolicyObservation(rgb=_rgb()), goal).metadata["from_curve"] is True
    assert step.step(PolicyObservation(rgb=_rgb()), goal).metadata["from_curve"] is False


def test_the_indices_the_agent_can_emit_all_have_names():
    from sparx_agency.core.planning.vlas.internvla_n1.types import INDEX_TO_ACTION
    for index in (-1, 0, 1, 2, 3, 5):
        assert index in INDEX_TO_ACTION, "index %d would decode as STOP" % index


# ── depth goes on the wire normalised, not in metres ─────────────────────
#
# The agent's System-1 path does `depth * 10.0` and clips at 5, with the source
# comment `# should be 0-10m`. Sending metres does not produce a wrong number,
# it destroys the channel: a 3 m wall arrives as 30 and clips to 5, and so does
# everything else past 0.5 m, so System 1 plans against a flat plane.


def test_depth_is_normalised_over_ten_metres():
    from sparx_agency.core.planning.vlas.internvla_n1.policy import DEPTH_RANGE_M
    metres = np.array([[0.0, 0.5, 3.0, 9.9]], dtype=np.float32)
    wire = InternVLAN1Policy._to_depth(metres)[..., 0].ravel()
    assert wire == pytest.approx((metres / DEPTH_RANGE_M).ravel(), abs=1e-6)
    # What System 1 then sees is the original metres back again, its own 5 m
    # threshold aside.
    assert np.clip(wire * 10.0, 0.0, 5.0) == pytest.approx([0.0, 0.5, 3.0, 5.0])


def test_depth_beyond_the_range_saturates_at_one_not_beyond():
    wire = InternVLAN1Policy._to_depth(np.array([[25.0, 1e6]], dtype=np.float32))
    assert (wire <= 1.0).all() and (wire >= 0.0).all()


def test_non_finite_depth_reads_as_far_away_not_as_nan():
    """A Gazebo depth miss is +inf; a NaN would propagate into the latents."""
    wire = InternVLAN1Policy._to_depth(np.array([[np.inf, np.nan, 2.0]], dtype=np.float32))
    assert np.isfinite(wire).all()
    assert wire[0, 0, 0] == pytest.approx(1.0)
    assert wire[0, 1, 0] == pytest.approx(1.0)
    assert wire[0, 2, 0] == pytest.approx(0.2)


def test_depth_keeps_the_hxwx1_shape_the_client_sends():
    assert InternVLAN1Policy._to_depth(np.zeros((4, 6), np.float32)).shape == (4, 6, 1)
    assert InternVLAN1Policy._to_depth(np.zeros((4, 6, 1), np.float32)).shape == (4, 6, 1)
    assert InternVLAN1Policy._to_depth(None) is None


# ── an unreadable answer is a transport failure, not "we have arrived" ───


@pytest.mark.parametrize("body", [
    {},                              # no action key at all
    {"action": [{"action": []}]},    # the inner list is empty
    {"action": [None]},              # a null element
])
def test_an_unparseable_body_is_not_a_stop(body):
    from sparx_agency.core.planning.vlas.internvla_n1.client import ModelClient
    parsed = ModelClient()._parse_response(body, 0.0)
    assert parsed.success is False, "an unreadable body decoded as a decision"

    policy = _policy(parsed)
    result = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go"))
    assert result.metadata.get("transport_failed") is True
    assert result.stop is False


# ── a pixel goal has an age, and the age is the point ────────────────────
#
# The patched agent never clears the last goal, so `pixel_goal` is non-null on
# almost every step and says nothing about whether it is current. It is a pixel
# in the frame System 2 saw; once the aircraft has moved it no longer points
# where it meant. Drawn identically fresh or eight steps stale, it reads as a
# live target lock, which is exactly what it is not.


def _goal_response(step, index=1):
    return StepResponse(action="MOVE_FORWARD", action_index=index,
                        waypoint=(300, 200), waypoint_step=step, raw_response={})


def test_a_pixel_goal_is_fresh_only_on_the_decision_that_chose_it():
    policy = InternVLAN1Policy(url="http://127.0.0.1:8087")
    obs, goal = PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go")
    seen = []
    for step in (7, 7, 7, 12, 12):
        policy.client = _FakeClient(_goal_response(step))
        md = policy.step(obs, goal).metadata
        seen.append((md["waypoint_fresh"], md["waypoint_age_steps"]))
    assert [f for f, _ in seen] == [True, False, False, True, False]
    assert [a for _, a in seen] == [0, 1, 2, 0, 1]


def test_no_pixel_goal_reports_no_age():
    policy = _policy(StepResponse(action="TURN_LEFT", action_index=2, raw_response={}))
    md = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go")).metadata
    assert md["waypoint_fresh"] is False
    assert md["waypoint_age_steps"] is None


def test_reset_forgets_the_previous_episodes_goal():
    """The agent's step counter restarts, so a new goal can reuse an old number."""
    policy = InternVLAN1Policy(url="http://127.0.0.1:8087")
    policy.client = _FakeClient(_goal_response(3))
    obs, goal = PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go")
    assert policy.step(obs, goal).metadata["waypoint_fresh"] is True
    assert policy.step(obs, goal).metadata["waypoint_fresh"] is False
    policy.reset()
    assert policy.step(obs, goal).metadata["waypoint_fresh"] is True, \
        "after a reset, step 3 is a NEW goal, not the one from the last episode"


# ── the look-down is a request this stack has to act on ──────────────────


def test_look_down_is_surfaced_from_the_wire():
    from sparx_agency.core.planning.vlas.internvla_n1.client import ModelClient
    parsed = ModelClient()._parse_response(
        {"action": [{"action": [-1]}], "look_down": True}, 0.0)
    assert parsed.look_down is True
    policy = _policy(parsed)
    md = policy.step(PolicyObservation(rgb=_rgb()), LanguageGoal(instruction="go")).metadata
    assert md["look_down"] is True
    assert md["idle"] is True, "a look-down carries no motion of its own"


def test_an_unpatched_server_simply_reports_no_look_down():
    from sparx_agency.core.planning.vlas.internvla_n1.client import ModelClient
    parsed = ModelClient()._parse_response({"action": [{"action": [1]}]}, 0.0)
    assert parsed.look_down is False
    assert parsed.waypoint_step is None
