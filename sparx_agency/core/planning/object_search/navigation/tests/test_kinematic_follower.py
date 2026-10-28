"""The follower attaches the lattice's actions to the policy's follow commands and keeps them in step with the agent."""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.object_search.navigation.tests.rig import (DOOR_X, DOOR_Y, doorway_policy,
                                                                              doorway_world, observe, planned_route)
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand

FORWARD, LEFT, RIGHT = DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT


def follow(policy, ep, world, step, pose, route):
    """What ``_navigate`` returns once the route memory holds ``route``: the command with the plan attached."""
    obs = observe(ep, step, *pose)
    policy.route_memory.adopt(route, (route.points[-1].x, route.points[-1].y), "frontier", obs)
    policy._route, policy._goal, policy.last_world = route, policy.route_memory.goal, world   # as _navigate leaves them
    command = NavigationCommand.follow(route, info={"kind": "frontier"})
    return obs, policy.kinematic.command(obs, world, policy.planner, command)


def test_the_command_carries_the_plan_from_the_agents_pose_and_the_converter_walks_it():
    policy, ep = doorway_policy()
    world = doorway_world()
    start, goal = (3.0, 4.2, math.radians(10)), (9.0, 7.5)
    route = planned_route(policy, world, start[:2], goal)
    obs, command = follow(policy, ep, world, 1, start, route)
    assert command.actions and command.plan_pose == pytest.approx(start)
    assert command.info["kinematic"]["status"] == "planned" and command.info["kinematic"]["reason"] == "no_plan"
    converter = DiscreteActionConverter(ep.action_spec, policy.converter_params)
    pose, step = obs.pose, 1
    while command.actions:
        result = converter.step(pose, command)
        assert result.planned and result.action == command.actions[0]
        policy.notify_action(obs, result.action)
        pose = apply_action(pose, result.action, ep.action_spec)
        step += 1
        obs, command = follow(policy, ep, world, step, (pose.x, pose.y, pose.yaw), route)
    assert command.info["kinematic"]["status"] == "complete"
    assert math.dist((pose.x, pose.y), goal) <= policy.converter_params.goal_tolerance_m + 1e-6
    assert policy.kinematic.stats["planned"] == 1 and policy.kinematic.stats["continued"] >= 20
    assert policy.kinematic.stats["executed"] == policy.kinematic.stats["continued"] + 1


def test_an_agent_off_the_plan_gets_a_fresh_plan_from_where_it_is():
    policy, ep = doorway_policy()
    world = doorway_world()
    start, goal = (3.0, 4.2, math.radians(10)), (9.0, 7.5)
    route = planned_route(policy, world, start[:2], goal)
    obs, command = follow(policy, ep, world, 1, start, route)
    policy.notify_action(obs, command.actions[0])
    elsewhere = (start[0] + 0.3, start[1] - 0.2, start[2])            # not where the plan expected the agent
    obs, command = follow(policy, ep, world, 2, elsewhere, route)
    assert command.info["kinematic"]["reason"] == "off_plan" and command.plan_pose == pytest.approx(elsewhere)
    assert policy.kinematic.stats["replans"] == 2


def test_a_refused_step_is_remembered_as_a_pose_and_the_next_plan_turns_first():
    policy, ep = doorway_policy()
    world = doorway_world()
    start, goal = (DOOR_X - 1.0, DOOR_Y, 0.0), (DOOR_X + 2.5, DOOR_Y)
    route = planned_route(policy, world, start[:2], goal)
    obs, command = follow(policy, ep, world, 1, start, route)
    assert command.actions[0] == FORWARD
    policy.notify_blocked(obs)                                         # the simulator refused the step
    assert policy.kinematic.lattice.refused and policy.kinematic.stats["refused"] == 1
    assert policy.scrape_guard.stats["scrapes"] == 1, "aligned with the doorway: a scrape, not a wall"
    assert not policy.mapping._bumps.get(policy.mapping.floor_id), "nothing was written on the map"
    obs, command = follow(policy, ep, world, 2, start, route)
    assert command.actions and command.actions[0] in (LEFT, RIGHT)
    assert command.info["kinematic"]["reason"] in ("no_plan", "route_changed")


def test_an_obstacle_on_the_plan_re_plans_it():
    policy, ep = doorway_policy()
    world = doorway_world()
    start, goal = (3.0, DOOR_Y, 0.0), (9.0, DOOR_Y)
    route = planned_route(policy, world, start[:2], goal)
    obs, command = follow(policy, ep, world, 1, start, route)
    policy.notify_action(obs, command.actions[0])
    pose = apply_action(obs.pose, command.actions[0], ep.action_spec)
    blocked = doorway_world()
    grid = blocked.grid.copy()
    grid[58:63, 45:47] = 100                                          # a box appears on the straight run, 1.5 m ahead
    blocked = type(blocked)(grid, blocked.params, values=blocked.values)
    obs, command = follow(policy, ep, world and blocked, 2, (pose.x, pose.y, pose.yaw), route)
    assert command.info["kinematic"]["reason"] == "plan_obstructed"


def test_disabled_the_follower_leaves_the_command_alone():
    policy, ep = doorway_policy(kinematic={"enabled": False})
    world = doorway_world()
    route = planned_route(policy, world, (3.0, DOOR_Y), (9.0, DOOR_Y))
    _, command = follow(policy, ep, world, 1, (3.0, DOOR_Y, 0.0), route)
    assert command.actions == () and command.plan_pose is None and "kinematic" not in command.info


def test_the_configuration_and_the_episode_info_carry_the_layer():
    policy, _ = doorway_policy()
    config = policy.configuration()
    assert config["kinematic"]["enabled"] and "rule" in config["kinematic"]
    assert config["planner"]["centreline_weight"] == 4.0 and config["planner"]["channel_half_width_m"] == 0.6
    assert config["doorway_scrape"]["max_heading_deg"] == 40.0 and config["doorway_look_down"]["pitch_deg"] == 30.0
    info = policy.episode_info()
    assert set(info["kinematic"]) == {"enabled", "stats", "lattice", "events"}
    assert "stats" in info["doorway_scrape"] and "stats" in info["doorway_look_down"]


def test_a_centimetre_displacement_at_the_planned_heading_keeps_the_plan_re_anchored():
    policy, ep = doorway_policy()
    world = doorway_world()
    start, goal = (3.0, DOOR_Y, 0.0), (9.0, DOOR_Y)
    route = planned_route(policy, world, start[:2], goal)
    obs, command = follow(policy, ep, world, 1, start, route)
    actions = command.actions
    policy.notify_action(obs, actions[0])
    pose = apply_action(obs.pose, actions[0], ep.action_spec)
    snapped = (pose.x - 0.04, pose.y + 0.03, pose.yaw)                 # the navmesh snap moved the step's end 5 cm
    obs, command = follow(policy, ep, world, 2, snapped, route)
    assert command.info["kinematic"]["status"] == "resynced" and command.actions == actions[1:]
    assert command.plan_pose == pytest.approx(snapped), "the same actions, from where the agent really is"
    assert policy.kinematic.stats["resynced"] == 1 and policy.kinematic.stats["replans"] == 1
    far = (pose.x - 0.3, pose.y + 0.2, pose.yaw)                         # beyond resync_m: a fresh plan
    obs, command = follow(policy, ep, world, 3, far, route)
    assert command.info["kinematic"]["reason"] == "off_plan"
    turned = (pose.x, pose.y, pose.yaw + math.radians(30.0))             # a heading the plan did not predict
    obs, command = follow(policy, ep, world, 4, turned, route)
    assert command.info["kinematic"]["reason"] == "off_plan"


def test_a_route_the_lattice_cannot_walk_at_the_planners_floor_is_planned_at_the_lattice_floor():
    """A diagonal passage whose centre cells clear the walls by 0.224 m: lethal at 0.25 m, walkable at the lattice floor."""
    from sparx_agency.core.common.types import Pose2D, Path2D
    from sparx_agency.core.planning.object_search.navigation.tests.rig import RES, VALUES
    from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
    import numpy as np
    policy, ep = doorway_policy()
    g = np.full((120, 120), 100, np.int8)
    ii = np.arange(120)
    free = np.abs(ii[None, :] - ii[:, None]) <= 2                     # five cells wide across the diagonal
    free[:10, :] = free[110:, :] = free[:, :10] = free[:, 110:] = False
    g[free] = 0
    world = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    fields = policy.planner.fields_for(world)
    assert 0.2 < fields.clearance[50, 50] < 0.25, "the centre line clears the walls by sqrt(5) cells"
    along = tuple(Pose2D(2.05 + 0.25 * i / math.sqrt(2), 2.05 + 0.25 * i / math.sqrt(2)) for i in range(30))
    route = Path2D(along, "world", {"inflate_used_m": 0.25})
    start = (along[0].x, along[0].y, math.pi / 4)
    obs, command = follow(policy, ep, world, 1, start, route)
    assert command.actions, command.info["kinematic"]
    assert command.info["kinematic"]["radius_m"] == pytest.approx(0.20) and policy.kinematic.stats["squeezed"] == 1
    for x, y, _ in policy.kinematic._plan["poses"]:
        gx, gy = world.world_to_grid(x, y)
        assert fields.clearance[gy, gx] > 0.20
    strict, _ = doorway_policy(kinematic={"lattice_floor_m": 0.25})      # no rung below the floor: the ladder walks it
    _, command = follow(strict, ep, world, 1, start, route)
    assert command.actions == () and command.info["kinematic"]["status"] == "unplanned"


def test_strict_tracking_sets_the_converters_params_and_can_be_turned_off():
    strict, _ = doorway_policy()
    assert strict.converter_params.strict_plan_tracking and strict.converter_params.heading_hysteresis_deg == 0.0
    loose, _ = doorway_policy(kinematic={"strict_tracking": False})
    assert not loose.converter_params.strict_plan_tracking and loose.converter_params.heading_hysteresis_deg == 1.0
    with pytest.raises(ValueError, match="lattice_floor_m"):
        doorway_policy(kinematic={"lattice_floor_m": 0.1})
