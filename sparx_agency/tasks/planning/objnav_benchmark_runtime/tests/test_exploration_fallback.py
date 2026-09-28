"""The exploration fallback: a failed plan, model or decision keeps the agent moving, and says so.

Before this, a room LLM that timed out ended the episode as an agent error,
and a loop with nothing to work in returned an empty hold that the headless
agent spent as one idle turn per action -- the "camera spinning in place" of
the Ranchester recordings.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_fallback import (
    DETECTOR, ROOM_LLM, FallbackSettings)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, RES, VALUES, loop_policy, obs_at)


def moving(command):
    return bool(command.waypoints) and not command.stop


# -- the room LLM ---------------------------------------------------------------
def test_a_failed_room_llm_hands_the_action_to_the_nearest_frontier_and_backs_off():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    calls = []

    def broken(world_, target, step):
        calls.append(step)
        raise RuntimeError("Room LLM failed: read timed out")

    policy.graph.reason = broken
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier"
    assert command.info["fallback"].startswith("room LLM unavailable")
    assert policy.fallback.stats[ROOM_LLM + "_failures"] == 1 and policy.fallback.failures[0]["kind"] == ROOM_LLM
    assert policy.fallback.retry_step[ROOM_LLM] == FallbackSettings().service_backoff_actions
    assert policy.loop.events[-1]["event"] == "llm_failure" and policy.loop.stats["llm_fallbacks"] == 1
    for step in range(1, 5):                                 # inside the back-off: no model call, still moving
        assert moving(policy.loop.plan(obs_at(episode, step, IN_A), world))
    assert calls == [0]
    command = policy.loop.plan(obs_at(episode, 25, IN_A), world)   # the back-off ends: asked again, fails again
    assert calls == [0, 25] and moving(command)
    assert policy.fallback.retry_step[ROOM_LLM] == 25 + 2 * FallbackSettings().service_backoff_actions


def test_a_recovered_room_llm_resets_the_back_off_and_the_loop_resumes_its_steps():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    real = policy.graph.reason
    failures = {"left": 1}

    def flaky(world_, target, step):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("once")
        real(world_, target, step)

    policy.graph.reason = flaky
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    command = policy.loop.plan(obs_at(episode, 25, IN_A), world)
    assert reasoned == [25] and ROOM_LLM not in policy.fallback.retry_step
    assert command.info["kind"] == "transit/1", "steps 2-6 ran on the action the model came back"


# -- the planner ---------------------------------------------------------------------
def test_a_goal_the_planner_refuses_is_dropped_for_one_it_accepts_not_spun_on():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    real = policy._plan_to
    refused = []

    def near_only(obs, world_, goal):
        if math.dist((obs.pose.x, obs.pose.y), goal) > 4.0:
            refused.append(goal)
            return None
        return real(obs, world_, goal)

    policy._plan_to = near_only
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert refused, "the transit into room B was asked for and refused"
    assert moving(command)
    assert math.dist(IN_A, command.waypoints[-1]) <= 4.0


def test_a_dead_planner_is_one_hold_per_action_with_the_reason_recorded_not_an_exception():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy._plan_to = lambda obs, world_, goal: None
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert not command.waypoints and command.info["kind"] == "fallback_hold"
    assert policy.fallback.stats["hold"] == 1 and policy.fallback.stats["invocations"] == 1


# -- nothing left ---------------------------------------------------------------------
def no_rooms(policy):
    policy.graph.registry.rooms = {}
    policy.graph.labels = np.zeros(policy.graph.labels.shape, np.int32)
    policy.graph.options, policy.graph.probs, policy.graph.facts = [], {}, {}


def test_a_swept_floor_relocates_to_a_new_vantage_point_instead_of_turning_in_place():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    g = np.zeros((60, 120), np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    known = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, IN_A), known)
    assert moving(command) and command.info["fallback_stage"] == "relocation"
    assert math.dist(IN_A, command.waypoints[-1]) >= FallbackSettings().relocation_min_m
    assert command.info["kind"] == "relocate"
    second = policy.loop.plan(obs_at(episode, 1, command.waypoints[-1]), known)
    assert moving(second) and math.dist(command.waypoints[-1], second.waypoints[-1]) >= 1.0, "not the same spot again"


def test_a_retired_frontier_is_still_unknown_space_when_nothing_else_is_left():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    first = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert first.info["fallback_stage"] == "frontier"
    from sparx_agency.core.planning.exploration.frontier_ranking import ranked_frontier_goals
    from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
    cost = assemble_cost_grid(policy.planner.fields_for(world), policy.planner_params, policy.settings.body_radius_m)[0]
    everywhere = ranked_frontier_goals(world, cost, np.ones(world.grid.shape, bool), IN_A, 0.0, policy.sweep.settings.ranking)
    policy._visited_frontiers = [g.xy for g in everywhere]       # every frontier retired
    policy.route_memory.clear("test")
    policy._route = policy._goal = None
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier_retired"


def test_boxed_in_the_hold_names_itself():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    g = np.full((60, 120), 100, np.int8)
    g[29:32, 29:32] = 0
    boxed = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, IN_A), boxed)
    assert not command.waypoints and command.info["kind"] == "fallback_hold"
    assert "passable map" in command.info["reason"]


# -- the decision and the detector ------------------------------------------------------
def test_an_exception_in_the_decision_is_recorded_and_the_agent_keeps_exploring():
    policy, episode = setup_policy()
    policy._search = lambda *args: (_ for _ in ()).throw(KeyError("a bug in the loop"))
    command = policy.plan(observation(episode, 0, depth=3))
    assert moving(command) and command.info["fallback"].startswith("decision failure: KeyError")
    [failure] = policy.fallback.failures
    assert failure["kind"] == "decision" and failure["step"] == 0 and "KeyError" in failure["error"]
    assert failure["origin"].startswith("test_exploration_fallback.py:")
    assert policy.episode_info()["exploration_fallback"]["stats"]["failures"] == 1


def test_a_failing_fallback_is_one_recorded_hold_never_a_dead_episode():
    policy, episode = setup_policy()
    policy._search = lambda *args: (_ for _ in ()).throw(RuntimeError("decision"))
    policy.fallback.plan = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fallback too"))
    command = policy.plan(observation(episode, 0, depth=3))
    assert not command.stop and command.info["kind"] == "fallback_hold"
    assert [f["kind"] for f in policy.fallback.failures] == ["decision", "fallback"]


def test_a_failed_detector_frame_is_skipped_and_backed_off_not_waited_on_every_action():
    policy, episode = setup_policy()
    calls = []

    def dead(rgb):
        calls.append(len(calls))
        raise RuntimeError("HTTP 503 from the detector")

    policy.detector.detect = dead
    command = policy.plan(observation(episode, 0, depth=3))
    assert not command.stop and calls == [0]
    assert policy.fallback.stats[DETECTOR + "_failures"] == 1 and policy.perception.counts["detector_failures"] == 1
    assert policy.perception.detector_evidence["retry_step"] == FallbackSettings().service_backoff_actions
    for step in range(1, 4):
        policy.plan(observation(episode, step, depth=3))
    assert calls == [0], "no detector call inside the back-off"
    assert policy.perception.counts["detector_skipped"] == 3 and policy.perception.detections == ()
    assert policy.landmarks.all_landmarks() == []
    policy.plan(observation(episode, 25, depth=3))
    assert calls == [0, 1] and policy.fallback.retry_step[DETECTOR] == 25 + 50


@pytest.mark.parametrize("kwargs", [{"frontier_attempts": 0}, {"relocation_min_m": -1.0},
                                    {"service_backoff_actions": 50, "service_backoff_max_actions": 10}])
def test_fallback_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        FallbackSettings(**kwargs)


def test_the_configuration_and_the_episode_record_carry_the_fallback():
    policy, episode = setup_policy()
    assert policy.configuration()["exploration_fallback"]["frontier_attempts"] == 6
    policy.plan(observation(episode, 0, depth=3))
    record = policy.episode_info()["exploration_fallback"]
    assert set(record) == {"settings", "stats", "failures", "retry_step"}

