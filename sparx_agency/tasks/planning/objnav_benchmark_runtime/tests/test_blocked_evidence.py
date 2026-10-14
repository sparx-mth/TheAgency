"""A blocked forward step is evidence the body gathered: it marks the map outright and it does not outlive a look.

Corozal/000142 (``51eff3fa1637``, 2026-10-13): the agent pushed into the same
wall ten actions running because one hit's increment could not lift a
ray-cleared cell over the occupied threshold; and one blocked step followed
by a ten-turn glance released the next four transits ``blocked`` an action in,
because the blocked clock ran on through the look.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridConfig, LogOddsGridCostmap
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.observed_map import ObservedMap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy


def test_mark_occupied_lifts_a_ray_cleared_cell_over_the_threshold_at_once():
    grid = LogOddsGridCostmap(LogOddsGridConfig(size_m=4.0, resolution_m=0.1))
    cleared = np.zeros((grid.height, grid.width), bool)
    cleared[20, 22] = True
    for _ in range(10):
        grid.apply_free_mask(cleared)                      # the floor rays have driven the cell to l_min
    assert grid._L[20, 22] == grid.cfg.l_min
    hit = np.array([[0.25, 0.05]])                          # the cell (22, 20) in a grid whose origin is (-2, -2)
    grid.update_from_points_xy(np.repeat(hit, 20, axis=0))
    _, probabilities = grid.get_grid()
    assert probabilities[20, 22] < 65, "twenty points in one update are one hit: not enough to flip the cell"
    grid.mark_occupied(hit)
    _, probabilities = grid.get_grid()
    assert probabilities[20, 22] >= 65 and grid._L[20, 22] == grid.cfg.l_max
    grid.mark_occupied(np.array([[50.0, 50.0]]))           # off the grid: ignored, never an exception
    grid.mark_occupied(np.zeros((0, 2)))


def test_a_blocked_forward_step_marks_the_cell_ahead_occupied_on_the_planning_world():
    policy, episode = setup_policy()
    mapping = ObservedMap(size_m=20)
    obs = observation(episode, 0, depth=3)                  # a wall 3 m ahead: the floor between is ray-cleared
    world = mapping.update(obs)
    step = episode.action_spec.forward_step_m
    gx, gy = world.world_to_grid(obs.pose.x + step * math.cos(obs.pose.yaw), obs.pose.y + step * math.sin(obs.pose.yaw))
    assert world.grid[gy, gx] == world.values.free
    mapping.notify_blocked(obs, step)
    world = mapping.update(observation(episode, 1, depth=3))
    assert world.grid[gy, gx] == world.values.occupied, "one blocked step is a wall the planner routes round"


def test_the_blocked_clock_pauses_through_a_look_and_runs_through_the_followers_own_actions():
    policy, episode = setup_policy(label="bed")               # no target in view: the search, not the closing, owns the actions
    obs = observation(episode, 0)
    policy.plan(obs)
    dt = policy.settings.action_time_s
    policy._blocked_since = policy._floor_time
    since = policy._blocked_since
    policy._action_owner = "glance"                          # a suspended phase: the supervisor's deadlines pause
    policy.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert policy._blocked_since == since + dt, "the look did not count as time spent blocked"
    assert policy._floor_time - policy._blocked_since == 0.0
    policy._action_owner = "search"
    policy.notify_action(observation(episode, 1), DiscreteAction.MOVE_FORWARD)
    assert policy._blocked_since == since + dt, "the follower's own action lets the clock run"
    assert policy._floor_time - policy._blocked_since == dt
