"""The fake environment keeps the ObjNavEnv contract: every action counts, a blocked step does not move, success is STOP in the goal region.

The fake is the reference every benchmark adapter is compared against and the
rig every pipeline test stands on, so each clause is pinned here: every action
counts, a blocked step does not move the agent, STOP and the budget end the
episode, success is STOP inside the goal region, and the runner's own checks
all pass on it. Which episodes it serves is pinned in ``test_fake_episodes``,
its frames in ``test_fake_rendering``, and its oracle in
``test_oracle_policy``.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.objnav.errors import (
    EnvContractError,
    ObjNavError,
)
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.types.actions import (
    NAVIGATION_ACTIONS,
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_episode
from sparx_agency.tasks.planning.objnav_benchmark.tests.fake_rooms import (
    INSIDE,
    OUTSIDE,
    OUTSIDE_TO_GOAL_M,
    play,
    room_env,
)

A = DiscreteAction


# -- actions ------------------------------------------------------------------

def test_move_forward_steps_one_forward_length_along_the_heading():
    """The converter predicts with this step; a different one makes every prediction wrong."""
    env = room_env(starts=[("north", OUTSIDE, math.pi / 2)])
    before = env.reset("north")[1].pose
    after = env.step(A.MOVE_FORWARD).pose
    assert after.x == pytest.approx(before.x) and after.y == pytest.approx(before.y + 0.25)
    assert after.yaw == before.yaw


def test_a_step_into_the_wall_band_does_not_move_the_agent_but_counts_and_adds_no_path():
    """AI2-THOR fails a blocked step atomically; a sliding fake would credit path never walked."""
    env = room_env(starts=[("west", (4, 2), math.pi)])
    start = env.reset("west")[1].pose
    blocked = env.step(A.MOVE_FORWARD)
    assert blocked.pose == start and blocked.step == 1
    env.step(A.TURN_LEFT)
    env.step(A.TURN_LEFT)
    env.step(A.TURN_LEFT)
    env.step(A.TURN_LEFT)
    env.step(A.TURN_LEFT)
    env.step(A.TURN_LEFT)            # facing east now
    env.step(A.MOVE_FORWARD)
    env.step(A.STOP)
    measurement = env.measure()
    assert measurement.steps == 9
    assert measurement.path_length_m == pytest.approx(0.25)


def test_turns_rotate_by_the_turn_angle_and_a_look_past_the_pitch_limit_fails():
    """AI2-THOR spends a refused LOOK without moving the camera; clamping would invent a pitch."""
    spec = DiscreteActionSpec(min_pitch_deg=-30.0, max_pitch_deg=30.0)
    env = room_env(action_spec=spec)
    env.reset("out")
    assert env.step(A.TURN_LEFT).pose.yaw == pytest.approx(math.radians(30))
    assert env.step(A.TURN_RIGHT).pose.yaw == pytest.approx(0.0)
    assert env.step(A.LOOK_DOWN).pose.camera_pitch == pytest.approx(math.radians(30))
    refused = env.step(A.LOOK_DOWN)
    assert refused.pose.camera_pitch == pytest.approx(math.radians(30)) and refused.step == 4
    assert env.step(A.LOOK_UP).pose.camera_pitch == pytest.approx(0.0)


def test_stop_ends_the_episode_and_the_next_step_is_refused():
    """Stepping on after STOP would move an agent the benchmark has already scored."""
    env = room_env()
    observation = play(env, "out", [A.STOP])
    assert env.episode_over and observation.step == 1
    with pytest.raises(EnvContractError, match="over"):
        env.step(A.TURN_LEFT)


def test_the_budget_ends_an_episode_that_never_stops_as_a_step_limit():
    """The environment owns the budget; every action counts toward it."""
    env = room_env(max_steps=3)
    play(env, "out", [A.TURN_LEFT, A.TURN_LEFT])
    assert not env.episode_over
    env.step(A.TURN_LEFT)
    measurement = env.measure()
    assert env.episode_over and measurement.steps == 3
    assert (measurement.termination, measurement.stop_called, measurement.success) == (
        "step_limit", False, False)


def test_an_action_outside_the_spec_or_a_bare_integer_is_refused():
    """A LOOK on a benchmark without tilt, or a Habitat index, means something else elsewhere."""
    env = room_env(action_spec=DiscreteActionSpec(actions=NAVIGATION_ACTIONS))
    env.reset("out")
    with pytest.raises(EnvContractError, match="not in this episode's action set"):
        env.step(A.LOOK_DOWN)
    with pytest.raises(EnvContractError, match="bare integer"):
        env.step(1)
    assert not env.episode_over and env.step(A.STOP).step == 1


# -- measurement ------------------------------------------------------------------

def test_success_is_stop_inside_the_goal_region_measured_by_the_geodesic():
    """Success, l, d0 and dT are the fake's whole scoring contract with the harness."""
    env = room_env(starts=[("out", OUTSIDE, 0.0), ("in", INSIDE, 0.0)])
    play(env, "out", [A.MOVE_FORWARD, A.STOP])
    short = env.measure()
    assert (short.success, short.stop_called, short.termination) == (False, True, "stop")
    assert short.shortest_path_m == short.start_distance_to_goal_m == OUTSIDE_TO_GOAL_M
    assert short.final_distance_to_goal_m == OUTSIDE_TO_GOAL_M - 0.25
    assert short.native_metrics == {"success": 0.0, "distance_to_goal": 1.0}
    play(env, "in", [A.STOP])
    found = env.measure()
    assert found.success and found.final_distance_to_goal_m == 0.0
    assert found.shortest_path_m == 0.0 and found.path_length_m == 0.0
    assert found.native_metrics == {"success": 1.0, "distance_to_goal": 0.0}


def test_measure_is_refused_while_the_episode_runs_and_before_any_reset():
    """A measurement of a running episode is a score of an episode that has not ended."""
    env = room_env()
    assert env.episode_over
    with pytest.raises(EnvContractError, match="none was started"):
        env.measure()
    with pytest.raises(EnvContractError, match="before reset"):
        env.step(A.STOP)
    env.reset("out")
    with pytest.raises(EnvContractError, match="still running"):
        env.measure()


def test_the_privileged_accessors_are_read_only_and_describe_the_scored_region():
    """The oracle reads these; a writable mask edited by it would change the score."""
    env = room_env(starts=[("out", OUTSIDE, 0.0), ("in", INSIDE, 0.0)])
    goal = env.privileged_goal_mask("chair")
    assert goal[INSIDE] and not goal[OUTSIDE]
    assert not goal.flags.writeable and not env.privileged_navigable_mask().flags.writeable
    assert (env.agent_radius_m, env.success_distance_m) == (0.18, 1.0)
    with pytest.raises(ObjNavError, match="no episode looks for 'bed'"):
        env.privileged_goal_mask("bed")


# -- the runner's contract checks ------------------------------------------------------

class ScriptedAgent(ObjNavAgent):
    """Plays a fixed action script."""

    name = "scripted"

    def __init__(self, script):
        self.script = list(script)

    def reset(self, episode):
        self.index = 0

    def act(self, observation):
        self.index += 1
        return AgentDecision(self.script[self.index - 1])


def test_the_runner_holds_the_fake_to_every_clause_of_the_contract():
    """The fake is the adapters' reference: the runner's own checks must all pass on it."""
    spec = DiscreteActionSpec(min_pitch_deg=-30.0, max_pitch_deg=30.0)
    env = room_env(action_spec=spec)
    script = [A.TURN_LEFT, A.MOVE_FORWARD, A.LOOK_DOWN, A.LOOK_DOWN, A.LOOK_UP,
              A.TURN_RIGHT, A.MOVE_FORWARD, A.MOVE_FORWARD, A.STOP]
    record = run_episode(env, ScriptedAgent(script), "out")
    assert record.steps == 9 and record.termination == "stop"
    assert record.path_length_m == pytest.approx(0.75)
    assert record.observed_path_length_m == pytest.approx(record.path_length_m, abs=1e-12)
