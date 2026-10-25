"""The multi-resolution prescan: goal-free FALCON on a coarse copy of the map, then the room reasoning."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.object_search.falcon.prescan import PrescanSettings
from sparx_agency.core.planning.object_search.rooms.discovery import discover
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.core.planning.object_search.world_model.coarse_map import downsample_world
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, loop_policy, obs_at)

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


def test_the_prescan_ends_when_falcon_has_nothing_left_to_explore():
    policy, episode, world = prescan_policy(steps=50)
    world.grid[world.grid == world.values.unknown] = world.values.free
    assert policy.prescan.plan(obs_at(episode, 0, IN_A), world) is None
    assert policy.prescan.ended == "exhausted"


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
