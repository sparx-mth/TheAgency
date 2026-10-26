"""The multi-resolution prescan: goal-free FALCON on a coarse copy of the map, then the room reasoning."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.object_search_supervisor import EXHAUSTED, SEARCH
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.object_search.falcon.prescan import PrescanSettings
from sparx_agency.core.planning.object_search.falcon.room_falcon import room_scope
from sparx_agency.core.planning.object_search.rooms.discovery import discover
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.core.planning.object_search.world_model.coarse_map import downsample_world
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, cell_in, loop_policy, obs_at)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_explore_visit import resolve_around, walk
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_opening_nodes import refresh

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


def small_world(cells):
    grid = np.zeros((10, 10), np.int16)
    for (row, column), value in cells.items():
        grid[row, column] = value
    return OccupancyGrid2D(grid, OccupancyGrid2DParams(0.1, 1.0, 2.0), values=VALUES)


def prescan_policy(steps=3):
    policy, episode, world, rooms, _ = loop_policy(settings={"prescan": {"steps": steps}})
    policy.warmup_actions, policy._warmup_pending = policy.warmup_turns, False
    return policy, episode, world


def test_a_block_is_occupied_if_any_cell_is_else_unknown_if_any_is_else_free():
    coarse = downsample_world(small_world({(0, 0): 100, (0, 1): -1, (4, 4): -1}), 3)
    assert coarse.grid[0, 0] == 100 and coarse.grid[1, 1] == -1 and coarse.grid[0, 1] == 0


def test_the_coarse_copy_keeps_the_frame_and_scales_the_cell_size():
    fine = small_world({})
    coarse = downsample_world(fine, 3)
    assert coarse.resolution == pytest.approx(0.3)
    assert (coarse.params.origin_x, coarse.params.origin_y) == (1.0, 2.0)
    assert coarse.grid.shape == (4, 4)
    assert downsample_world(fine, 1) is fine


def test_a_factor_below_one_is_refused():
    with pytest.raises(ValueError):
        downsample_world(small_world({}), 0)


def test_the_prescan_settings_are_validated_and_default_off():
    assert RPTSettings().prescan.steps == 0
    assert RPTSettings(prescan={"steps": 50}).prescan == PrescanSettings(steps=50)
    with pytest.raises(ValueError, match="resolution_factor"):
        PrescanSettings(steps=5, resolution_factor=0)


def test_the_prescan_is_off_unless_asked_for():
    policy, _, _, _, _ = loop_policy()
    assert policy.prescan is None


def test_the_prescan_explores_on_the_coarse_map_and_ends_at_its_budget():
    policy, episode, world = prescan_policy(steps=2)
    kinds = []
    for step in range(2):
        command = policy.prescan.plan(obs_at(episode, step, IN_A), world)
        assert command is not None and command.waypoints
        kinds.append(command.info["kind"])
    assert set(kinds) <= {"falcon", "relocate"}
    assert policy.prescan.plan(obs_at(episode, 2, IN_A), world) is None
    assert policy.prescan.ended == "budget" and policy.prescan.actions == 2
    assert policy._route is None and policy._goal is None


def test_the_prescan_turns_in_place_while_falcon_is_patient_and_ends_when_it_stops():
    policy, episode, world = prescan_policy(steps=50)
    world.grid[world.grid == world.values.unknown] = world.values.free
    first = policy.prescan.plan(obs_at(episode, 0, IN_A), world)
    assert first is not None and first.info["kind"] == "exploration_exhausted_turn" and not first.waypoints
    step = 1
    while policy.prescan.active:
        policy.prescan.plan(obs_at(episode, step, IN_A), world)
        step += 1
        assert step < 60
    assert policy.prescan.ended == "exhausted" and policy.prescan.actions >= policy.warmup_turns


def test_a_falcon_warmup_that_gives_up_early_hands_over_to_the_rotation():
    policy, _, _ = prescan_policy(steps=50)
    policy.settings = replace(policy.settings, scan_kind="falcon")
    policy.warmup_actions, policy._warmup_pending = policy.warmup_turns, False
    policy.prescan.actions = 3
    policy.prescan._end("exhausted")
    assert policy.warmup_actions == 0 and policy._warmup_pending and policy.prescan.rotation_fallback


def test_a_falcon_warmup_that_explored_enough_does_not_rotate_again():
    policy, _, _ = prescan_policy(steps=50)
    policy.settings = replace(policy.settings, scan_kind="falcon")
    policy.warmup_actions, policy._warmup_pending = policy.warmup_turns, False
    policy.prescan.actions = policy.warmup_turns
    policy.prescan._end("exhausted")
    assert policy.warmup_actions == policy.warmup_turns and not policy.prescan.rotation_fallback



def test_discovery_hands_the_action_to_the_prescan_then_to_the_room_loop():
    policy, episode, world = prescan_policy(steps=2)
    cost = policy.navigation_cost(world)
    for step in range(2):
        command = discover(policy, obs_at(episode, step, IN_A), world, cost)
        assert command is not None and policy._action_owner == "prescan"
    assert discover(policy, obs_at(episode, 2, IN_A), world, cost) is None
    assert not policy.prescan.active and policy.prescan.ended == "budget"


def test_a_blocked_step_during_the_prescan_rejects_its_viewpoint():
    policy, episode, world = prescan_policy(steps=5)
    policy.prescan.plan(obs_at(episode, 0, IN_A), world)
    assert policy.prescan.explorer.current is not None
    policy._action_owner = "prescan"
    policy.notify_blocked(obs_at(episode, 1, IN_A))
    assert policy.prescan.explorer.current is None and policy.prescan.explorer.stats["blocked"] == 1


def test_the_full_policy_runs_the_warmup_then_the_prescan_without_error():
    policy, episode = setup_policy("bed", discovery=True, prescan={"steps": 3}, doorway_peek={"enabled": False})
    for step in range(policy.warmup_turns + 8):
        obs = observation(episode, step, depth=3)
        policy.plan(obs)
        policy.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert policy.prescan.ended in ("budget", "exhausted")
    assert policy.episode_info()["prescan"]["ended"] == policy.prescan.ended


def test_the_unknown_rule_decides_how_much_unseen_makes_a_block_unknown():
    cells = {(0, 0): -1, (0, 1): -1}
    for rule, unknown in (("any", True), ("majority", False), ("all", False)):
        assert (downsample_world(small_world(cells), 3, rule).grid[0, 0] == -1) == unknown
    wholly_unseen = {(r, c): -1 for r in range(3) for c in range(3)}
    assert all(downsample_world(small_world(wholly_unseen), 3, rule).grid[0, 0] == -1 for rule in ("any", "majority", "all"))


def test_an_unknown_rule_outside_the_three_is_refused():
    with pytest.raises(ValueError, match="unknown_rule"):
        downsample_world(small_world({}), 3, "most")
    with pytest.raises(ValueError, match="unknown_rule"):
        PrescanSettings(steps=5, unknown_rule="most")


def test_the_block_under_the_agent_is_free_even_when_a_wall_pixel_shares_it():
    world = small_world({(0, 0): 100})
    assert downsample_world(world, 3).grid[0, 0] == 100
    assert downsample_world(world, 3, agent_xy=(1.05, 2.05)).grid[0, 0] == 0


def test_falcon_as_the_warmup_needs_a_prescan_budget():
    with pytest.raises(ValueError, match="prescan.steps"):
        RPTSettings(scan_kind="falcon")
    assert RPTSettings(scan_kind="falcon", prescan={"steps": 20}).scan_kind == "falcon"


def test_falcon_as_the_warmup_replaces_the_start_rotation():
    policy, episode = setup_policy("bed", discovery=True, scan_kind="falcon", prescan={"steps": 4},
                                   doorway_peek={"enabled": False})
    assert policy.warmup_actions == policy.warmup_turns and not policy._warmup_pending
    kinds = []
    for step in range(6):
        obs = observation(episode, step, depth=3)
        kinds.append(policy.plan(obs).info.get("kind"))
        policy.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert "warmup" not in kinds and policy.scans.records == []
    assert policy.prescan.ended in ("budget", "exhausted") or policy.prescan.actions > 0


def falcon_room_policy():
    from dataclasses import replace
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0), visit="explore", local_explorer="falcon")
    policy.sight.settings = replace(policy.sight.settings, enabled=False)
    refresh(policy, world, IN_A)
    return policy, episode, world, rooms


def test_the_falcon_room_visit_walks_falcon_viewpoints_inside_the_room_then_finishes_it():
    policy, episode, world, rooms = falcon_room_policy()
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    entry = policy.route_memory.goal
    inside = (entry[0] - 0.3, entry[1]) if cell_in(world, rooms[1], (entry[0] - 0.3, entry[1])) else entry
    resolve_around(world, inside, 1.2)
    refresh(policy, world, inside)
    policy.sight.observe(obs_at(episode, 1, inside), world)
    first = policy.loop.plan(obs_at(episode, 1, inside), world)
    assert policy.supervisor.state == SEARCH and first.info["kind"] == "falcon" and first.waypoints
    policy.loop.charge()
    commands = walk_facing(policy, episode, world, 2, inside)
    assert {c.info.get("kind") for c in commands[:-1]} <= {"falcon"}
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    assert policy.room_falcon.visits == 1 and policy.loop.stats["explore_exhausted"] == 1


def test_the_room_scope_is_the_room_plus_a_margin():
    mask = np.zeros((10, 10), bool)
    mask[4:6, 4:6] = True
    scope = room_scope(small_world({}), mask, margin_m=0.2)
    assert scope[4:6, 4:6].all() and scope[3, 4] and not scope[1, 1]


def walk_facing(policy, episode, world, start_step, start_xy, max_actions=60):
    """Like ``walk``, but the agent also ends each route facing the command's final heading, as the follower does."""
    commands, xy, yaw = [], start_xy, 0.0
    for step in range(start_step, start_step + max_actions):
        obs = obs_at(episode, step, xy, yaw=yaw)
        policy.sight.observe(obs, world)
        command = policy.loop.plan(obs, world)
        commands.append(command)
        if policy.supervisor.state != SEARCH or policy.loop.room_id is None:
            return commands
        policy.loop.charge()
        if command.waypoints:
            end = command.waypoints[-1]
            xy = (float(end[0]), float(end[1]))
        if command.final_yaw is not None:
            yaw = float(command.final_yaw)
        resolve_around(world, xy, 1.5)
        refresh(policy, world, xy)
    raise AssertionError("the visit never ended")


def test_falcon_plans_from_a_level_camera_at_the_fixed_body_height():
    from sparx_agency.core.planning.object_search.falcon.ground_height import level_camera
    from dataclasses import replace as dc_replace
    from sparx_agency.core.planning.objnav.types.pose import AgentPose
    tilted = dc_replace(obs_at(setup_policy()[1], 0, IN_A), pose=AgentPose(3.0, 3.0, 0.0, 0.5, camera_pitch=0.4))
    level = level_camera(tilted)
    assert level.pose.camera_pitch == 0.0
    assert (level.pose.x, level.pose.y, level.pose.z, level.pose.yaw) == (3.0, 3.0, 0.0, 0.5)
    assert tilted.pose.camera_pitch == 0.4


def test_a_tilted_camera_does_not_change_what_falcon_is_asked():
    policy, episode, world = prescan_policy(steps=3)
    from dataclasses import replace as dc_replace
    from sparx_agency.core.planning.objnav.types.pose import AgentPose
    seen = []
    original = policy.prescan.explorer.explorer.plan
    policy.prescan.explorer.explorer.plan = lambda w, c, s, request, h: (seen.append(request.pose.camera_pitch), original(w, c, s, request, h))[1]
    tilted = dc_replace(obs_at(episode, 0, IN_A), pose=AgentPose(IN_A[0], IN_A[1], 0.0, 0.0, camera_pitch=0.4))
    policy.prescan.plan(tilted, world)
    assert seen and set(seen) == {0.0}
