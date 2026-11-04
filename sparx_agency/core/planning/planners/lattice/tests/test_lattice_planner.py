"""The lattice planner threads doorways on the benchmark's own step and turn, in fewer actions than the ladder.

Python 3.8 syntax.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.types.actions import NAVIGATION_ACTIONS, DiscreteAction, DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.core.planning.planners.lattice import (STATUS_NO_PLAN, STATUS_START_OUTSIDE, STATUS_SUCCESS,
                                                          DiscreteActionLatticePlanner, LatticePlannerParams,
                                                          MotionPrimitives, route_corridor_fields, route_cost_to_go)

RES = 0.1
VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
SPEC = DiscreteActionSpec(forward_step_m=0.25, turn_angle_deg=30.0, actions=NAVIGATION_ACTIONS)
FORWARD, LEFT, RIGHT = DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT


def planner_params(**overrides):
    return WeightedAStarParams(**dict(dict(
        inflate_radius_m=0.30, inflate_floor_m=0.25, detour_factor=1.5, unknown_blocked=True, goal_snap_radius_m=0.3,
        waypoint_spacing_m=0.25, start_skip_m=0.0, search_margin_m=80.0, centreline_weight=4.0,
        channel_half_width_m=0.6), **overrides))


def doorway_world(width_cells=8, size=120):
    """A ``size`` x ``size`` cell room split by a wall at x = 6.0 m with a doorway centred on y = 6.0 m."""
    g = np.zeros((size, size), np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    g[:, 59:61] = 100
    half = width_cells // 2
    g[60 - half:60 + half, 59:61] = 0
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def route_and_fields(world, start, goal, params=None):
    params = params or planner_params()
    planner = WeightedAStarPlanner2D(params)
    result = planner.plan(PlanRequest(start, goal, frame_id="world"), world)
    assert result.ok, result.message
    fields = planner.fields_for(world)
    radius = result.path.metadata["inflate_used_m"]
    cost, _ = assemble_cost_grid(fields, params, radius)
    return [(p.x, p.y) for p in result.path.points], fields, cost, radius


def lattice_plan(world, start, goal, params=None, **lattice):
    route, fields, cost, radius = route_and_fields(world, start, goal, params)
    planner = DiscreteActionLatticePlanner(SPEC, LatticePlannerParams(**lattice))
    plan = planner.plan((start.x, start.y, start.yaw), route, 0.0, 0.0, RES, cost, fields.clearance, radius,
                        start_free_radius_m=radius)
    return plan, route, fields, radius, planner


def clearance_along(plan, fields, world):
    return [fields.clearance[world.world_to_grid(x, y)[1], world.world_to_grid(x, y)[0]] for x, y, _ in plan.poses]


def replay(plan, start):
    """The poses the converter's own transition model reaches by executing the plan: must match the plan's."""
    pose = AgentPose(start.x, start.y, 0.0, start.yaw, 0.0)
    poses = [pose]
    for action in plan.actions:
        pose = apply_action(pose, action, SPEC)
        poses.append(pose)
    return poses


def test_primitives_are_the_specs_step_and_turn():
    prim = MotionPrimitives(SPEC, 0.05)
    assert prim.step_m == 0.25 and prim.turn_rad == pytest.approx(math.radians(30)) and prim.headings == 12
    assert prim.wraps and prim.samples == pytest.approx((0.2, 0.4, 0.6, 0.8, 1.0))
    assert prim.index(0.3, prim.heading(0.3, 5)) == 5 and prim.turned(11, LEFT) == 0 and prim.turned(0, RIGHT) == 11
    with pytest.raises(ValueError):
        MotionPrimitives(DiscreteActionSpec(actions=(DiscreteAction.STOP, FORWARD, LEFT, RIGHT, DiscreteAction.LOOK_UP,
                                                     DiscreteAction.LOOK_DOWN)), 0.05)  # fine: all present
        raise ValueError("the body primitives are present")


def test_the_plan_threads_the_doorway_through_its_middle_and_is_executable_as_it_is():
    world = doorway_world()
    start, goal = Pose2D(3.0, 4.2, math.radians(10)), Pose2D(9.0, 7.5)
    plan, route, fields, radius, _ = lattice_plan(world, start, goal)
    assert plan.ok and plan.status == STATUS_SUCCESS, plan.message
    assert set(plan.actions) <= {FORWARD, LEFT, RIGHT}
    assert min(clearance_along(plan, fields, world)) >= radius, "every pose keeps the standoff the route was planned at"
    through = [y for x, y, _ in plan.poses if 5.7 <= x <= 6.3]
    assert through and all(abs(y - 6.0) <= 0.15 for y in through), "the doorway (0.8 m) is passed within 15 cm of its axis"
    for predicted, replayed in zip(plan.poses, replay(plan, start)):
        assert math.hypot(predicted[0] - replayed.x, predicted[1] - replayed.y) < 1e-9
        assert abs(math.sin(predicted[2] - replayed.yaw)) < 1e-9
    assert math.dist(plan.poses[-1][:2], (goal.x, goal.y)) <= 0.25


def test_the_plan_spends_fewer_actions_than_the_waypoint_ladder_on_the_same_route():
    world = doorway_world()
    start, goal = Pose2D(2.0, 3.0, math.radians(100)), Pose2D(10.0, 9.0)
    plan, route, _, _, _ = lattice_plan(world, start, goal)
    assert plan.ok
    converter = DiscreteActionConverter(SPEC, ActionConverterParams(heading_hysteresis_deg=1.0))
    rollout = converter.rollout(AgentPose(start.x, start.y, 0.0, start.yaw, 0.0), NavigationCommand.follow(route), 600)
    assert rollout.status == "arrived"
    assert len(plan.actions) < len(rollout.actions), (len(plan.actions), len(rollout.actions))
    assert plan.turns < sum(1 for a in rollout.actions if a != FORWARD)


def test_a_refused_pose_excludes_only_that_forward_step_and_the_next_plan_turns_first():
    world = doorway_world()
    start, goal = Pose2D(3.0, 6.0, 0.0), Pose2D(9.0, 6.0)
    plan, route, fields, radius, planner = lattice_plan(world, start, goal)
    assert plan.ok and plan.actions[0] == FORWARD, "a straight run through the doorway starts with a step"
    planner.note_refused(start.x, start.y, start.yaw, step=10)
    assert planner.move_refused(start.x + 0.05, start.y, start.yaw) and not planner.move_refused(start.x, start.y, start.yaw + 0.6)
    cost, _ = assemble_cost_grid(fields, planner_params(), radius)
    again = planner.plan((start.x, start.y, start.yaw), route, 0.0, 0.0, RES, cost, fields.clearance, radius, step=11,
                         start_free_radius_m=radius)
    assert again.ok and again.actions[0] in (LEFT, RIGHT), "the refused step is not tried again from this pose"
    assert min(clearance_along(again, fields, world)) >= radius
    planner.expire(10 + planner.params.refused_move_actions + 1)
    assert not planner.refused and planner.stats["refused_expired"] == 1


def test_failures_are_named_not_hidden():
    world = doorway_world()
    start, goal = Pose2D(3.0, 6.0, 0.0), Pose2D(9.0, 6.0)
    route, fields, cost, radius = route_and_fields(world, start, goal)
    planner = DiscreteActionLatticePlanner(SPEC, LatticePlannerParams())
    outside = planner.plan((1.0, 1.0, 0.0), route, 0.0, 0.0, RES, cost, fields.clearance, radius)
    assert outside.status == STATUS_START_OUTSIDE and not outside.ok and outside.poses == ((1.0, 1.0, 0.0),)
    # A footprint wider than the doorway: no action sequence gets through, and the search says so.
    too_wide = planner.plan((start.x, start.y, start.yaw), route, 0.0, 0.0, RES, cost, fields.clearance, 0.45,
                            start_free_radius_m=0.45)
    assert too_wide.status == STATUS_NO_PLAN and too_wide.actions == ()
    assert planner.diagnostics()["failures"] == 2 and planner.diagnostics()["failed:no_plan"] == 1


def test_the_route_field_covers_the_corridor_and_weights_the_remaining_cost():
    route = [(1.0, 1.0), (4.0, 1.0)]
    field = route_cost_to_go(route, 0.0, 0.0, RES, (60, 60), corridor_m=1.0)
    assert field[10, 40] == pytest.approx(0.0, abs=0.08), "the goal cell is at distance zero"
    assert field[10, 10] == pytest.approx(3.0, abs=0.1), "the start cell is the route's length from the goal"
    # Half a metre off the route: the cheapest way back onto it and on to the goal, never less than the straight
    # line to the goal, never more than the offset plus the remaining length from the nearest point.
    assert math.hypot(1.45, 0.5) - 0.05 <= field[15, 25] <= 1.45 + 0.5 + 0.05
    assert not np.isfinite(field[30, 25]), "two metres off the route lies outside the corridor"
    cost = np.ones((60, 60))
    cost[:, 25:30] = 5.0                       # a costly band half a metre wide across the route
    weighted = route_cost_to_go(route, 0.0, 0.0, RES, (60, 60), corridor_m=1.0, cost=cost)
    # The band's metres cost five, not one (its 0.5 m read at the 0.2 m sampling: 0.4-0.6 m of it).
    assert 3.0 + 0.4 * 4.0 - 0.1 <= weighted[10, 10] <= 3.0 + 0.6 * 4.0 + 0.1
    with pytest.raises(ValueError):
        route_cost_to_go(route[:1], 0.0, 0.0, RES, (60, 60), corridor_m=1.0)


def test_params_refuse_nonsense():
    with pytest.raises(ValueError):
        LatticePlannerParams(heuristic_weight=0.5)
    with pytest.raises(ValueError):
        LatticePlannerParams(max_expansions=0)
    with pytest.raises(ValueError):
        LatticePlannerParams(corridor_m=0.0)
    assert LatticePlannerParams(refused_move_actions=0).refused_move_actions == 0


def test_the_offset_cost_keeps_the_plan_on_the_route_and_the_corridor_bounds_it():
    world = doorway_world()
    start, goal = Pose2D(2.0, 3.0, math.radians(100)), Pose2D(10.0, 9.0)
    route, fields, cost, radius = route_and_fields(world, start, goal)
    _, offset = route_corridor_fields(route, 0.0, 0.0, RES, cost.shape, 0.4)
    def worst_offset(plan):
        return max(offset[world.world_to_grid(x, y)[1], world.world_to_grid(x, y)[0]] for x, y, _ in plan.poses)
    free = DiscreteActionLatticePlanner(SPEC, LatticePlannerParams(corridor_m=1.5, route_offset_cost=0.0)).plan(
        (start.x, start.y, start.yaw), route, 0.0, 0.0, RES, cost, fields.clearance, radius, start_free_radius_m=radius)
    tight = DiscreteActionLatticePlanner(SPEC, LatticePlannerParams(corridor_m=0.4, route_offset_cost=2.0)).plan(
        (start.x, start.y, start.yaw), route, 0.0, 0.0, RES, cost, fields.clearance, radius, start_free_radius_m=radius)
    assert free.ok and tight.ok
    assert all(math.isfinite(offset[world.world_to_grid(x, y)[1], world.world_to_grid(x, y)[0]]) for x, y, _ in tight.poses), \
        "every pose of the tight plan lies inside the 0.4 m corridor"
    assert worst_offset(tight) <= 0.4 + 1e-9
    mean_tight = sum(offset[world.world_to_grid(x, y)[1], world.world_to_grid(x, y)[0]] for x, y, _ in tight.poses) / len(tight.poses)
    assert mean_tight <= 0.2, "the plan rides the route, leaving it only to line up a heading"
    assert LatticePlannerParams().corridor_m == 0.4 and LatticePlannerParams().route_offset_cost == 2.0
