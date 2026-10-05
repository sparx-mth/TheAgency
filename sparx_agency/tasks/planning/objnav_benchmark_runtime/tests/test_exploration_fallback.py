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
    FREE, IN_A, OCC, RES, UNK, VALUES, loop_policy, obs_at)


def moving(command):
    return bool(command.waypoints) and not command.stop


# -- the room LLM ---------------------------------------------------------------
def test_a_failed_room_llm_hands_the_action_to_the_nearest_frontier_and_backs_off():
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0))
    calls = []

    def broken(world_, target, step, **kwargs):
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

    def flaky(world_, target, step, **kwargs):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("once")
        real(world_, target, step, **kwargs)

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
    """The Gibson protocol has no LOOK actions: boxed in, the hold is the one move left."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    g = np.full((60, 120), 100, np.int8)
    g[29:32, 29:32] = 0
    boxed = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, IN_A), boxed)
    assert not command.waypoints and command.info["kind"] == "fallback_hold"
    assert "passable map" in command.info["reason"]
    assert policy.fallback.stats["footing"] == 0


def test_boxed_in_with_a_camera_that_tilts_the_fallback_maps_its_footing_before_it_holds():
    """An agent that has not moved stands on a disk of unknown under the camera's blind radius
    (Hanson/000002, 2026-10-05); one footing sweep maps it, and only then is the hold the last move."""
    from dataclasses import replace
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    episode = replace(episode, action_spec=MULTIFLOOR_PROTOCOL.actions())
    policy.episode = episode
    policy.camera_control.actions = episode.action_spec
    g = np.full((60, 120), -1, np.int8)                  # the spawn: the footprint known, everything around it unknown
    g[29:32, 29:32] = 0
    boxed = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, IN_A), boxed)
    assert not command.waypoints and command.info["kind"] == "footing_sweep" and command.info["fallback_stage"] == "footing"
    assert policy.fallback.stats["footing"] == 1 and policy.fallback.stats["hold"] == 0
    assert policy.camera_control.footing, "the camera controller carries the sweep from here, one action at a time"
    # The sweep done at this spot, the hold is what is left.
    policy.camera_control.inspection = None
    command = policy.loop.plan(obs_at(episode, 20, IN_A), boxed)
    assert command.info["kind"] == "fallback_hold" and policy.fallback.stats["hold"] == 1
    # Walled in by KNOWN cells there is nothing a sweep would map: the hold at once.
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    episode = replace(episode, action_spec=MULTIFLOOR_PROTOCOL.actions())
    policy.episode = episode
    policy.camera_control.actions = episode.action_spec
    g = np.full((60, 120), 100, np.int8)
    g[29:32, 29:32] = 0
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, IN_A), OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES))
    assert command.info["kind"] == "fallback_hold" and policy.fallback.stats["footing"] == 0


# -- the decision and the detector ------------------------------------------------------
def test_an_exception_in_the_decision_is_recorded_and_the_agent_keeps_exploring():
    policy, episode = setup_policy(label="bed")
    policy._search = lambda *args: (_ for _ in ()).throw(KeyError("a bug in the loop"))
    command = policy.plan(observation(episode, 0, depth=3))
    assert moving(command) and command.info["fallback"].startswith("decision failure: KeyError")
    [failure] = policy.fallback.failures
    assert failure["kind"] == "decision" and failure["step"] == 0 and "KeyError" in failure["error"]
    assert failure["origin"].startswith("test_exploration_fallback.py:")
    assert policy.episode_info()["exploration_fallback"]["stats"]["failures"] == 1


def test_a_failing_fallback_is_one_recorded_hold_never_a_dead_episode():
    policy, episode = setup_policy(label="bed")
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
                                    {"service_backoff_actions": 50, "service_backoff_max_actions": 10},
                                    {"exits_first": 1}, {"shadow_margin_m": 0.0}, {"shadow_length_factor": -1.0},
                                    {"near_blind_m": 0.0}, {"goal_switch_gain": 0.5}, {"goal_match_m": float("inf")}])
def test_fallback_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        FallbackSettings(**kwargs)


def test_the_configuration_and_the_episode_record_carry_the_fallback():
    policy, episode = setup_policy()
    assert policy.configuration()["exploration_fallback"]["frontier_attempts"] == 6
    assert policy.configuration()["exploration_fallback"]["exits_first"] is True
    policy.plan(observation(episode, 0, depth=3))
    record = policy.episode_info()["exploration_fallback"]
    assert set(record) == {"settings", "stats", "failures", "retry_step", "last_demoted"}


# -- exits before shadows (Ranchester 2026-10-04, actions 61-102) -----------------------
def shadow_world():
    """One finished room, 6 x 6 m, with a bed whose shadow is a 1.6 m strip of unknown
    behind it, and a 1 m doorway in the east wall with the next room unknown beyond."""
    h, w = 60, 90
    g = np.full((h, w), FREE, np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = OCC
    g[:, 60] = OCC                        # the east wall of the room, at x = 6.0
    g[25:35, 60] = FREE                   # the doorway: y 2.5 .. 3.5
    g[:, 61:] = UNK                       # the next room: unknown beyond the wall
    g[25:35, 61:66] = FREE                # ...except the half-metre the camera saw through the door
    g[22:38, 36:42] = OCC                 # the bed: 1.6 x 0.6 m, centred at (3.9, 3.0)
    g[22:38, 42:46] = UNK                 # its shadow: the strip the camera never saw behind it
    world = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    return world


def bed_landmark(xy=(3.9, 3.0), radius=0.8):
    from sparx_agency.core.mapping.objects.landmarks import ObjectLandmark
    return ObjectLandmark(id=0, class_name="bed", xy=xy, count=9, votes={"bed": 9}, radius_m=radius)


def test_a_beds_shadow_is_not_an_exit_and_the_doorway_beyond_it_is_chosen():
    policy, episode, _, rooms, _ = loop_policy(order=(1, 0))
    world = shadow_world()
    no_rooms(policy)
    policy.landmarks.confirmed = lambda: [bed_landmark()]
    here = (3.0, 3.0)                                       # the strip is 1.2 m away, the door 3 m
    command = policy.loop.plan(obs_at(episode, 0, here), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier"
    assert command.waypoints[-1][0] > 5.0, "the doorway in the east wall, not the strip behind the bed"
    demoted = policy.fallback.last_demoted
    assert demoted and all(d["why"].startswith("shadow of bed 0") for d in demoted)
    assert all(4.0 < d["xy"][0] < 5.0 for d in demoted), "the demoted goals sit beside the bed"
    assert policy.fallback.stats["shadows_demoted"] >= 1
    assert policy.episode_info()["exploration_fallback"]["last_demoted"] == demoted


def test_without_exits_first_the_nearer_shadow_wins_as_before():
    policy, episode, _, rooms, _ = loop_policy(order=(1, 0))
    policy.fallback.settings = FallbackSettings(exits_first=False)
    world = shadow_world()
    no_rooms(policy)
    policy.landmarks.confirmed = lambda: [bed_landmark()]
    command = policy.loop.plan(obs_at(episode, 0, (3.0, 3.0)), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier"
    assert 4.0 < command.waypoints[-1][0] < 5.0 and not policy.fallback.last_demoted


def test_a_long_frontier_passing_an_object_is_an_opening_not_its_shadow():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    # Room A's whole west end is unknown: a 5.8 m frontier. A plant standing beside it does not own it.
    policy.landmarks.confirmed = lambda: [bed_landmark(xy=(1.4, 3.0), radius=0.3)]
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier"
    assert command.waypoints[-1][0] < 2.0 and not policy.fallback.last_demoted


def test_a_frontier_of_a_room_the_target_cannot_be_in_waits_behind_the_exits():
    from sparx_agency.core.planning.exploration.frontier_ranking import accessible_frontiers
    from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    labels = policy.graph.labels.copy()
    no_rooms(policy)
    cost = assemble_cost_grid(policy.planner.fields_for(world), policy.planner_params, policy.settings.body_radius_m)[0]
    policy.graph.frontier_inventory = accessible_frontiers(world, cost, labels, IN_A, 0.0, policy.sweep.settings.ranking)
    assert 1 in policy.graph.frontier_inventory.by_room, "room A's west frontier is credited to label 1 = pid 0"
    policy.loop._excluded = {0: "type:bathroom"}            # room A (pid 0) cannot hold the target
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier"
    assert command.waypoints[-1][0] > 6.0, "room B's frontier through the door, not the nearer bathroom's"
    assert policy.fallback.last_demoted and policy.fallback.last_demoted[0]["why"] == "room 0 is type:bathroom"
    assert policy.fallback.stats["type_demoted"] >= 1


def test_demoted_frontiers_are_still_unknown_space_once_the_exits_are_spent():
    policy, episode, _, rooms, _ = loop_policy(order=(1, 0))
    world = shadow_world()
    world.grid[25:35, 60:66] = OCC                           # brick up the doorway: the shadow is all that is left
    no_rooms(policy)
    policy.landmarks.confirmed = lambda: [bed_landmark()]
    command = policy.loop.plan(obs_at(episode, 0, (3.0, 3.0)), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier_demoted"
    assert 4.0 < command.waypoints[-1][0] < 5.0
    assert policy.fallback.stats["frontier_demoted"] == 1


def test_a_frontier_inside_the_cameras_blind_radius_is_demoted():
    policy, episode, _, rooms, _ = loop_policy(order=(1, 0))
    world = shadow_world()
    world.grid[22:38, 36:46] = FREE                          # no bed, no shadow...
    world.grid[28:32, 26:28] = UNK                           # ...but a speck of unknown 0.4 m from the agent
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, (3.0, 3.0)), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier"
    assert command.waypoints[-1][0] > 5.0, "the doorway, not the speck under the agent's feet"
    assert any("blind radius" in d["why"] for d in policy.fallback.last_demoted)
    assert policy.fallback.stats["blind_demoted"] >= 1


# -- commitment: the goal in force is kept until gone, refused or clearly outranked ---------
def test_the_goal_in_force_is_kept_while_the_agent_turns_toward_it():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    first = policy.loop.plan(obs_at(episode, 0, IN_A, yaw=0.0), world)
    goal = first.waypoints[-1]
    assert policy.fallback.goal is not None and policy.fallback.goal_stage == "frontier"
    # Facing the other way now: the greedy order would discount this goal and may prefer another;
    # the commitment keeps it, and the route is not replaced.
    second = policy.loop.plan(obs_at(episode, 1, IN_A, yaw=math.pi), world)
    assert second.waypoints[-1] == goal
    assert second.info.get("route_replaced") is None
    assert policy.fallback.stats["goal_kept"] == 1 and policy.fallback.stats["goal_switched"] == 0


def test_a_commitment_does_not_survive_an_action_the_loop_owned():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.fallback.goal_step == 0
    policy.loop.plan(obs_at(episode, 5, IN_A), world)       # four actions elsewhere in between
    assert policy.fallback.goal_step == 5 and policy.fallback.stats["goal_switched"] == 0, "a fresh choice, not a switch"


def test_a_goal_worth_twice_as_much_takes_over_a_lesser_one_in_force():
    from sparx_agency.core.planning.exploration.frontier_ranking import FrontierGoal
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    fallback = policy.fallback
    weak = FrontierGoal(xy=(1.0, 1.0), cell=(10, 10), size_cells=4, geodesic_m=2.0, heading_error_rad=0.0, utility=1.0)
    strong = FrontierGoal(xy=(5.0, 5.0), cell=(50, 50), size_cells=64, geodesic_m=2.0, heading_error_rad=0.0, utility=2.5)
    fallback.goal, fallback.goal_stage = (1.0, 1.0), "frontier"
    assert fallback._committed([strong, weak], "frontier") == [weak, strong] or fallback.stats["goal_outranked"] == 1
    # 2.5 >= 2 x 1.0: the strong goal outranks the one in force; 1.5 would not.
    assert fallback._committed([strong, weak], "frontier")[0] is strong
    middling = FrontierGoal(xy=(5.0, 5.0), cell=(50, 50), size_cells=16, geodesic_m=2.0, heading_error_rad=0.0, utility=1.5)
    assert fallback._committed([middling, weak], "frontier")[0] is weak
    assert fallback._committed([middling, weak], "frontier_demoted")[0] is middling, "another rung: no goal in force"


def test_the_step_record_carries_the_fallback_snapshot():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    snap = policy.fallback.snapshot()
    assert snap["stage"] == "frontier" and snap["goal_stage"] == "frontier" and snap["goal_step"] == 0
    assert len(snap["goal"]) == 2 and isinstance(snap["demoted"], list)
    assert {"frontier", "goal_kept", "blind_demoted"} <= set(snap["stats"])

