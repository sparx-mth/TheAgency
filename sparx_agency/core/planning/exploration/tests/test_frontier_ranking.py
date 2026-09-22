"""Tests for the utility order of frontier goals.

Every world here is small enough to reason about by hand, so a wrong order
names itself: a boundary ahead must beat a bigger one behind the robot, a
boundary behind a wall must cost the walk around it, and a boundary with no
known-free path to it must not be offered at all -- offering it costs a failed
A* and an idle action, which is exactly what the Ranchester recording spent
three of its first thirty actions on.

Grids are strings: ``.`` free, ``#`` occupied, ``?`` unknown; row 0 is minimum
y, and a cell ``(col, row)`` sits at world ``((col + 0.5) * RES, (row + 0.5) * RES)``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.exploration.frontier_ranking import (
    FrontierGoal, FrontierRankingParams, ranked_frontier_goals)
from sparx_agency.core.planning.exploration.room_costs import (
    in_room_frontier_goals, passable_graph)

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
FREE, OCC, UNK = 0, 100, -1
RES = 0.5


def world_from(rows):
    table = {".": FREE, "#": OCC, "?": UNK}
    g = np.array([[table[c] for c in row] for row in rows], dtype=np.int16)
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def flat_cost(world):
    cost = np.ones(world.grid.shape, dtype=np.float64)
    cost[world.grid != FREE] = np.inf
    return cost


def at(col, row):
    """World centre of a cell."""
    return ((col + 0.5) * RES, (row + 0.5) * RES)


def test_a_small_boundary_ahead_beats_a_larger_nearer_one_behind():
    """Facing +x from cell (6, 3): three frontier cells ahead, five behind and nearer."""
    hall = world_from([
        "############",
        "#??.......?#",
        "#??.......?#",
        "#??........#",
        "#??........#",
        "#??........#",
        "############",
    ])
    mask = hall.grid != OCC
    goals = ranked_frontier_goals(hall, flat_cost(hall), mask, origin_xy=at(6, 3), yaw=0.0,
                                  min_cluster_cells=1)
    assert len(goals) == 2
    ahead, behind = goals
    assert ahead.cell[0] == 9 and abs(ahead.heading_error_rad) < 0.4
    assert behind.cell[0] == 3 and abs(behind.heading_error_rad) > 2.5
    assert behind.size_cells > ahead.size_cells and behind.geodesic_m < ahead.geodesic_m, (
        "the test only means something if the loser is both bigger and nearer")
    # The pose-free order is the opposite one -- which is the point.
    size_first = in_room_frontier_goals(hall, flat_cost(hall), mask, min_cluster_cells=1)
    assert hall.world_to_grid(*size_first[0])[0] == 3


def test_cost_is_the_walk_around_a_wall_not_the_distance_through_it():
    corridor = world_from([
        "###########",
        "#?...#....#",
        "#....#....#",
        "#....#....#",
        "#.........#",
        "###########",
    ])
    mask = corridor.grid != OCC
    origin = at(6, 1)
    goals = ranked_frontier_goals(corridor, flat_cost(corridor), mask, origin, yaw=math.pi,
                                  min_cluster_cells=1)
    assert len(goals) == 1
    goal = goals[0]
    assert goal.cell[0] <= 2, "the boundary is on the far side of the wall"
    assert goal.geodesic_m > math.dist(origin, goal.xy) + RES, (
        "the geodesic must go around the wall through the gap, not through it")


def test_an_unreachable_boundary_is_dropped_not_offered():
    """Free cells seen through a gap but not connected by known free space."""
    islands = world_from([
        "###########",
        "#....#..?##",
        "#....#..?##",
        "#....######",
        "#....######",
        "###########",
    ])
    mask = np.ones(islands.grid.shape, dtype=bool)
    goals = ranked_frontier_goals(islands, flat_cost(islands), mask, origin_xy=at(2, 2), yaw=0.0,
                                  min_cluster_cells=1)
    assert goals == []
    assert in_room_frontier_goals(islands, flat_cost(islands), mask, min_cluster_cells=1), (
        "the pose-free list still offers it, so the ranking is what protects the planner")


def test_a_robot_off_the_passable_graph_gets_no_goals_rather_than_a_crash():
    room = world_from([
        "#######",
        "#....?#",
        "#.....#",
        "#######",
    ])
    mask = room.grid != OCC
    goals = ranked_frontier_goals(room, flat_cost(room), mask, origin_xy=(20.0, 20.0), yaw=0.0,
                                  min_cluster_cells=1, params=FrontierRankingParams(snap_radius_m=0.5))
    assert goals == []


def test_facing_is_a_discount_not_a_veto():
    """Same size, same distance: ahead first. Bigger and nearer behind: behind first."""
    symmetric = world_from([
        "###########",
        "#?.......?#",
        "#?.......?#",
        "#?.......?#",
        "###########",
    ])
    mask = symmetric.grid != OCC
    goals = ranked_frontier_goals(symmetric, flat_cost(symmetric), mask, origin_xy=at(5, 2), yaw=0.0,
                                  min_cluster_cells=1)
    assert [g.cell[0] for g in goals] == [8, 2]
    assert goals[0].geodesic_m == pytest.approx(goals[1].geodesic_m)
    ignoring_facing = ranked_frontier_goals(symmetric, flat_cost(symmetric), mask, at(5, 2), 0.0,
                                            min_cluster_cells=1, params=FrontierRankingParams(heading_weight=0.0))
    assert ignoring_facing[0].utility == pytest.approx(ignoring_facing[1].utility)

    lopsided = world_from([
        "###########",
        "#??......?#",
        "#??.......#",
        "#??.......#",
        "#??.......#",
        "#??.......#",
        "###########",
    ])
    mask = lopsided.grid != OCC
    goals = ranked_frontier_goals(lopsided, flat_cost(lopsided), mask, origin_xy=at(5, 3), yaw=0.0,
                                  min_cluster_cells=1)
    assert goals[0].cell[0] == 3, "five cells one metre behind outweigh two cells ahead"
    assert goals[0].utility > goals[1].utility


def test_prebuilt_graph_gives_the_same_order_and_half_a_graph_is_refused():
    hall = world_from([
        "###########",
        "#?.......?#",
        "#?.......?#",
        "###########",
    ])
    mask = hall.grid != OCC
    cost = flat_cost(hall)
    ids, graph = passable_graph(cost)
    fresh = ranked_frontier_goals(hall, cost, mask, at(5, 1), 0.0, min_cluster_cells=1)
    reused = ranked_frontier_goals(hall, cost, mask, at(5, 1), 0.0, min_cluster_cells=1,
                                   ids=ids, graph=graph)
    assert [g.cell for g in fresh] == [g.cell for g in reused]
    assert all(isinstance(g, FrontierGoal) for g in fresh)
    with pytest.raises(ValueError):
        ranked_frontier_goals(hall, cost, mask, at(5, 1), 0.0, ids=ids)
    with pytest.raises(ValueError):
        ranked_frontier_goals(hall, cost, mask, (float("nan"), 0.75), 0.0)


def test_a_far_boundary_beyond_the_horizon_is_withheld_this_tick():
    hall = world_from([
        "###############",
        "#............?#",
        "###############",
    ])
    mask = hall.grid != OCC
    near_horizon = FrontierRankingParams(max_geodesic_m=2.0)
    assert ranked_frontier_goals(hall, flat_cost(hall), mask, at(1, 1), 0.0,
                                 min_cluster_cells=1, params=near_horizon) == []
    assert ranked_frontier_goals(hall, flat_cost(hall), mask, at(1, 1), 0.0, min_cluster_cells=1)


@pytest.mark.parametrize("kwargs", [{"gain_exponent": -0.1}, {"distance_floor_m": 0.0},
                                    {"heading_weight": float("inf")}, {"max_geodesic_m": -1.0},
                                    {"snap_radius_m": 0.0}])
def test_ranking_params_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        FrontierRankingParams(**kwargs)

