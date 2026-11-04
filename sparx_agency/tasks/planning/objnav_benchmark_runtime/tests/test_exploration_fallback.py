"""The exploration fallback: a failed plan, model or decision keeps the agent moving, and says so.

Before this, a room LLM that timed out ended the episode as an agent error,
and a loop with nothing to work in returned an empty hold that the headless
agent spent as one idle turn per action -- the "camera spinning in place" of
the Ranchester recordings.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.object_search.exploration.exploration_fallback import (
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


def test_a_dead_planner_is_one_probing_step_per_action_with_the_reason_recorded_not_an_exception():
    """Every plan refused on open floor: the escape probe's forward step is the move left (before 2026-10-13, a hold)."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    policy._plan_to = lambda obs, world_, goal: None
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert command.info["kind"] == "escape" and command.info["fallback"] == "no room in force"
    assert policy.fallback.stats["escape"] == 1 and policy.fallback.stats["hold"] == 0
    assert policy.fallback.stats["invocations"] == 1
    policy.escape.settings = replace(policy.escape.settings, enabled=False)
    command = policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert not command.waypoints and command.info["kind"] == "fallback_hold"
    assert policy.fallback.stats["hold"] == 1 and policy.fallback.stats["invocations"] == 2


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
    # The sweep done at this spot, a probing step into the unknown is what is left (the hold, before 2026-10-13).
    policy.camera_control.inspection = None
    command = policy.loop.plan(obs_at(episode, 20, IN_A), boxed)
    assert command.info["kind"] == "escape" and policy.fallback.stats["escape"] == 1 and policy.fallback.stats["hold"] == 0
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
    policy.loop.plan(obs_at(episode, 20, IN_A), world)      # nineteen actions elsewhere in between (> goal_memory_actions)
    assert policy.fallback.goal_step == 20 and policy.fallback.stats["goal_switched"] == 0, "a fresh choice, not a switch"


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



# -- the end-to-end review of 2026-10-05 (evening): the fallback over consecutive actions with motion ------
def walk(policy, episode, world, start, steps, yaw=0.0):
    """Drive the fallback ``steps`` actions, moving 0.25 m along the returned route each time; return the goals taken."""
    here, goals = tuple(start), []
    for step in range(steps):
        command = policy.loop.plan(obs_at(episode, step, here, yaw=yaw), world)
        assert moving(command) or command.info.get("kind") == "fallback_hold", command.info
        if command.waypoints:
            goal = tuple(round(float(v), 2) for v in command.waypoints[-1])
            goals.append(goal)
            dx, dy = goal[0] - here[0], goal[1] - here[1]
            span = math.hypot(dx, dy)
            if span > 1e-6:
                here = (here[0] + 0.25 * dx / span, here[1] + 0.25 * dy / span)
    return goals


def test_a_blind_radius_demotion_is_sticky_so_the_fallback_does_not_turn_between_two_exits_for_ever():
    """Approaching exit A, the step that brings it under 1.2 m demotes it; the step toward exit B puts A at
    1.2 m again and, nearer, it out-utilities B -- two actions a cycle, indefinitely, 0.25 m each so the
    route watchdog never fires. A demoted goal is retired and the goal in force cleared."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    goals = walk(policy, episode, world, (2.6, 3.0), 12)
    switches = sum(1 for a, b in zip(goals, goals[1:]) if math.dist(a, b) > policy.fallback.settings.goal_match_m)
    assert switches <= 2, "a goal chosen, driven and resolved -- not alternated: %r" % (goals,)
    assert policy.fallback.stats["blind_demoted"] >= 1, "the near frontier was under the blind radius at some point"
    assert policy._visited_frontiers, "the demoted goal was retired, not left to be promoted again"


def test_the_relocation_target_is_kept_until_reached():
    """The farthest reachable cell was excluded while still being walked to, so the next action took the
    opposite far cell, and the agent shuffled a quarter metre back and forth between the two."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    world.grid[:, :] = FREE                                  # an empty hall: no frontier anywhere, relocation only
    world.grid[0, :] = world.grid[-1, :] = world.grid[:, 0] = world.grid[:, -1] = OCC
    no_rooms(policy)
    goals = walk(policy, episode, world, (6.0, 3.0), 8)
    assert len({g for g in goals}) == 1, "one far cell, walked to action after action: %r" % (goals,)
    assert policy.fallback.stats["relocation"] == 8


def test_a_vantage_point_already_walked_to_is_not_relocated_to_again_so_a_closed_pocket_reaches_the_escape_rung():
    """00835/000064 (2026-10-25): a 2 m strip beside a bed, fully known, no frontier; the farthest cell flipped
    between the strip's two ends and the fallback relocated 458 times without once reaching the escape probe
    that would have tried the strip's one thin exit. A vantage point of the current run of fallback actions is
    never a target again; with both ends walked, the rungs below relocation get their turn."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    g = np.full(world.grid.shape, OCC, np.int8)
    g[27:34, 10:50] = FREE                                   # a 0.7 m wide, 4 m long strip, closed everywhere
    g[27:34, 50] = OCC                                       # ... by one occupied cell at its east end, free beyond
    g[27:34, 51:60] = FREE
    strip = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    no_rooms(policy)
    here, goals, stages = (3.0, 3.0), [], []
    for step in range(60):
        command = policy.loop.plan(obs_at(episode, step, here, yaw=0.0), strip)
        stages.append(command.info.get("fallback_stage") or command.info.get("kind"))
        if command.info.get("fallback_stage") == "relocation":
            goal = tuple(round(float(v), 2) for v in command.waypoints[-1])
            goals.append(goal)
            dx, dy = goal[0] - here[0], goal[1] - here[1]
            span = math.hypot(dx, dy)
            if span > 1e-6:
                here = (here[0] + 0.25 * dx / span, here[1] + 0.25 * dy / span)
        elif command.info.get("fallback_stage") == "escape":
            break
    assert "escape" in stages, "the escape rung is reached once the strip's vantage points are spent: %r" % (stages,)
    vantage = {g for g in goals}
    assert 1 <= len(vantage) <= 3, "each end of the strip at most once: %r" % (sorted(vantage),)
    assert policy.fallback.stats["relocation"] < 40
    with pytest.raises(ValueError):
        FallbackSettings(relocation_revisit_m=0.0)


def test_a_frontier_behind_a_gap_the_floor_rung_refuses_is_reached_by_the_squeeze_rung_before_any_relocation():
    """00835/000064 (2026-10-25): the strip beside the bed opened onto the rest of the flat through a doorway the
    map drew 0.4 m wide -- walked on the way in, impassable at the planner's 0.25 m floor rung, so the inventory
    held no reachable frontier and the fallback relocated and probed inside the strip for 480 actions. Nothing
    reachable at the rung: the goals reachable at the body radius are routed through the squeeze planner."""
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    g = np.full(world.grid.shape, OCC, np.int8)
    g[27:34, 20:40] = FREE                                   # the agent's strip: 0.7 m wide, 2 m long
    g[29:33, 40:46] = FREE                                   # a 0.4 m wide, 0.6 m long gap out of it ...
    g[20:40, 46:70] = FREE                                   # ... onto open floor ...
    g[20:40, 70:80] = UNK                                    # ... with unknown beyond: the only frontier
    pocket = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, IN_A, yaw=0.0), pocket)
    assert command.info["fallback_stage"] == "frontier_squeeze" and command.info["kind"] == "frontier"
    assert command.info.get("squeeze") is True and command.waypoints
    assert command.waypoints[-1][0] > 4.6, "the goal lies on the open floor beyond the gap"
    assert any(4.0 <= x <= 4.6 for x, _ in command.waypoints), "the route threads the gap"
    assert policy.fallback.stats["frontier_squeeze"] >= 1 and policy.fallback.stats["relocation"] == 0
    # With the gap a cell wider the ordinary rungs reach the frontier themselves and the squeeze rung is not asked.
    g[28:34, 40:46] = FREE
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, IN_A, yaw=0.0), OccupancyGrid2D(g, pocket.params, values=VALUES))
    assert command.info["fallback_stage"] == "frontier" and "squeeze" not in command.info


def test_the_snapshot_names_this_actions_rung_including_a_hold():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.fallback.snapshot()["stage"] == "frontier"
    boxed = OccupancyGrid2D(np.full(world.grid.shape, UNK, np.int8), world.params, values=world.values)
    command = policy.loop.plan(obs_at(episode, 1, IN_A), boxed)
    stage = {"fallback_hold": "hold", "footing_sweep": "footing", "escape": "escape"}[command.info["kind"]]
    assert policy.fallback.snapshot()["stage"] == stage


# -- 2026-10-14: a sliver is not an exit, and a glance does not forget the goal ----------------------------
def test_a_sliver_that_reveals_too_little_is_not_an_exit_and_the_exits_are_ranked_by_what_they_reveal():
    """The shadow world WITHOUT the bed on the landmark map, plus a four-cell notch of unknown in the room's far
    corner. Until 2026-10-14 the inventory's size-based order made the 1.2 m strip behind the bed the first exit
    (Collierville/000036, actions 99-113: one sliver after another, each resolved by the step toward it, the next
    behind the agent). Now the exits are ranked by the unknown a look from each reveals: the doorway, whose rays
    reach the whole next room, comes first; the strip, which still sees through the door, stays an exit; and the
    notch, which reveals nothing but itself, is demoted as a sliver."""
    policy, episode, _, rooms, _ = loop_policy(order=(1, 0))
    base = shadow_world()
    g = base.grid.copy()
    g[1:3, 1:3] = UNK                                       # the notch: 0.04 m2 in the south-west corner
    world = OccupancyGrid2D(g, base.params, values=base.values)
    no_rooms(policy)
    here = (3.0, 3.0)
    command = policy.loop.plan(obs_at(episode, 0, here), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier"
    assert command.waypoints[-1][0] > 5.0, "the doorway in the east wall, not the strip behind the bed"
    demoted = policy.fallback.last_demoted
    assert demoted and all(d["why"].startswith("reveals under") for d in demoted), demoted
    assert all(d["xy"][0] < 1.0 and d["xy"][1] < 1.0 for d in demoted), "the notch, not the strip"
    assert policy.fallback.stats["slivers_demoted"] == len(demoted) >= 1
    # Switched off (0): the inventory's size-based order stands and the nearer strip is the exit, as before.
    policy.fallback.settings = replace(policy.fallback.settings, min_gain_m2=0.0)
    policy.fallback.goal = None
    command = policy.loop.plan(obs_at(episode, 1, here), world)
    assert command.waypoints[-1][0] < 5.0 and policy.fallback.stats["slivers_demoted"] == len(demoted)
    with pytest.raises(ValueError):
        FallbackSettings(min_gain_m2=-1.0)


def test_a_commitment_survives_a_glance_in_between_but_not_a_long_absence():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    first = policy.loop.plan(obs_at(episode, 0, IN_A, yaw=0.0), world)
    goal = first.waypoints[-1]
    # Ten turns of a glance in between, facing the other way at the end of it: the goal in force stands.
    second = policy.loop.plan(obs_at(episode, 10, IN_A, yaw=math.pi), world)
    assert second.waypoints[-1] == goal and policy.fallback.stats["goal_kept"] == 1
    # Twenty actions elsewhere: forgotten, and the fresh choice is not counted as a switch.
    policy.loop.plan(obs_at(episode, 31, IN_A, yaw=0.0), world)
    assert policy.fallback.goal_step == 31 and policy.fallback.stats["goal_switched"] == 0
    with pytest.raises(ValueError):
        FallbackSettings(goal_memory_actions=0)


# -- 2026-10-14: what casts a shadow, and what is too much unknown to be one ------------------------------
def test_a_clock_casts_no_shadow_and_neither_does_a_small_measured_footprint():
    """Collierville/000023, action 27: a doorway frontier 0.5 m from a confirmed clock vanished as 'the shadow of
    clock 24'. A wall or surface object hides no floor; nor does a landmark whose measured radius is under
    ``shadow_min_radius_m``."""
    from sparx_agency.core.mapping.objects.landmarks import ObjectLandmark
    from sparx_agency.core.planning.exploration.frontier_ranking import FrontierGoal
    from sparx_agency.core.planning.object_search.exploration.exploration_fallback import object_shadow
    goal = FrontierGoal(xy=(4.0, 3.0), cell=(40, 30), size_cells=10, geodesic_m=3.0, heading_error_rad=0.0, utility=1.0)
    clock = ObjectLandmark(id=24, class_name="clock", xy=(4.3, 3.2), count=3, votes={"clock": 3}, radius_m=0.6)
    vase = ObjectLandmark(id=25, class_name="vase", xy=(4.3, 3.2), count=3, votes={"vase": 3}, radius_m=None)
    small = ObjectLandmark(id=26, class_name="chair", xy=(4.3, 3.2), count=3, votes={"chair": 3}, radius_m=0.15)
    cabinet = ObjectLandmark(id=27, class_name="cabinet", xy=(4.3, 3.2), count=3, votes={"cabinet": 3}, radius_m=0.6)
    args = (0.1, 0.30, 0.50, 3.0)
    assert object_shadow(goal, [clock, vase, small], *args, min_radius_m=0.25) is None
    assert object_shadow(goal, [cabinet], *args, min_radius_m=0.25) is cabinet, "a cabinet beside a 1 m frontier: its shadow"
    assert object_shadow(goal, [small], *args, min_radius_m=0.0) is small, "the floor is the knob"
    with pytest.raises(ValueError):
        FallbackSettings(shadow_min_radius_m=0.0)


def test_a_frontier_with_a_rooms_floor_right_behind_it_is_nobodys_shadow():
    """Collierville/000023, action 24: the 2.9 m frontier onto the room the toilet stood in, a refrigerator 1 m beside
    it, dropped as 'the shadow of refrigerator 0' -- and the search re-explored the house for 180 actions. The
    shadow world with the bed removed from the landmark map and a FRIDGE put beside the doorway: the strip behind
    the bed stays a shadow of the bed (the wall is right behind it); the doorway, with the next room's floor from
    the threshold on, is an exit whatever stands beside it."""
    from sparx_agency.core.mapping.objects.landmarks import ObjectLandmark
    policy, episode, _, rooms, _ = loop_policy(order=(1, 0))
    world = shadow_world()
    no_rooms(policy)
    fridge = ObjectLandmark(id=1, class_name="refrigerator", xy=(6.3, 3.7), count=5, votes={"refrigerator": 5}, radius_m=0.7)
    policy.landmarks.confirmed = lambda: [bed_landmark(), fridge]
    here = (3.0, 3.0)
    command = policy.loop.plan(obs_at(episode, 0, here), world)
    assert moving(command) and command.info["fallback_stage"] == "frontier"
    assert command.waypoints[-1][0] > 5.0, "the doorway beside the fridge is still the exit"
    whys = [d["why"] for d in policy.fallback.last_demoted]
    assert whys and all(w.startswith("shadow of bed 0") for w in whys), whys
    # The exemption off: the doorway is the fridge's shadow and the strip behind the bed wins by distance.
    policy.fallback.settings = replace(policy.fallback.settings, shadow_exempt_m2=0.0)
    policy.fallback.goal = None
    command = policy.loop.plan(obs_at(episode, 1, here), world)
    assert command.waypoints[-1][0] < 5.0 and any(w.startswith("shadow of refrigerator 1") for w in
                                                  (d["why"] for d in policy.fallback.last_demoted))
    with pytest.raises(ValueError):
        FallbackSettings(shadow_exempt_m2=-1.0)
