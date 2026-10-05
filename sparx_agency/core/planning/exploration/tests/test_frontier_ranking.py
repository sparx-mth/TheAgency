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
    FrontierGoal, FrontierRankingParams, frontier_goals_by_room, ranked_frontier_goals)
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


def test_an_arc_shaped_boundary_gets_a_goal_on_the_boundary_not_at_its_centroid():
    """The edge of the depth range bows around the robot. Its centroid sits
    in the middle of the seen floor, where a goal has no unknown cell near
    it; the goal must be one of the arc's own cells."""
    arc = world_from([
        "#############",
        "#.....??????#",
        "#........???#",
        "#..........?#",
        "#...........#",
        "#..........?#",
        "#........???#",
        "#.....??????#",
        "#############",
    ])
    mask = arc.grid != OCC
    goals = ranked_frontier_goals(arc, flat_cost(arc), mask, at(2, 4), 0.0, min_cluster_cells=1)
    assert len(goals) == 1
    gx, gy = goals[0].cell
    assert arc.grid[gy, gx] == FREE
    around = arc.grid[gy - 1:gy + 2, gx - 1:gx + 2]
    assert (around == UNK).any(), "the goal touches unknown space: it is on the boundary"
    frontier_cells = {(4, 1), (5, 1), (7, 2), (8, 2), (9, 3), (10, 3), (10, 4), (11, 4), (10, 5), (9, 5),
                      (8, 6), (7, 6), (5, 7), (4, 7)}
    assert (gx, gy) in frontier_cells or any(abs(gx - x) + abs(gy - y) <= 1 for x, y in frontier_cells)
    assert (gx, gy) != (7, 4), "not the centroid, which is a cell of seen floor two columns from any boundary"


@pytest.mark.parametrize("kwargs", [{"gain_exponent": -0.1}, {"distance_floor_m": 0.0},
                                    {"heading_weight": float("inf")}, {"max_geodesic_m": -1.0},
                                    {"snap_radius_m": 0.0}])
def test_ranking_params_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        FrontierRankingParams(**kwargs)


# -- goals for every room at once -------------------------------------------
TWO_ROOMS = [
    "#############",
    "#?....#....?#",
    "#?....#.....#",
    "#......#....#",
    "#.....#.....#",
    "#############",
]
"""Two rooms joined by a one-cell gap in row 3; a frontier column in each."""


def labels_for(world, split_col=6):
    """Room 1 left of ``split_col``, room 2 right of it; walls and the gap are 0."""
    lbl = np.zeros(world.grid.shape, dtype=np.int32)
    free = world.grid == FREE
    cols = np.arange(world.grid.shape[1])[None, :]
    lbl[free & (cols < split_col)] = 1
    lbl[free & (cols > split_col)] = 2
    return lbl


def test_goals_are_keyed_by_the_room_label_and_ranked_within_each_room():
    world = world_from(TWO_ROOMS)
    labels = labels_for(world)
    by_room = frontier_goals_by_room(world, flat_cost(world), labels, at(3, 3), 0.0,
                                     min_cluster_cells=1)
    assert set(by_room) == {1, 2}
    assert all(g.cell[0] <= 2 for g in by_room[1]), "room 1's goals lie in room 1"
    assert all(g.cell[0] >= 10 for g in by_room[2]), "room 2's goals lie in room 2"
    for goals in by_room.values():
        assert [g.utility for g in goals] == sorted((g.utility for g in goals), reverse=True)
    assert min(by_room[1], key=lambda g: g.geodesic_m).geodesic_m < \
        min(by_room[2], key=lambda g: g.geodesic_m).geodesic_m, (
            "from inside room 1 its own boundary is the nearer one")


def test_by_room_matches_the_single_room_ranking_and_the_scene_graph_count():
    """One pass over the floor must agree with a pass per room, and with the count."""
    from sparx_agency.core.mapping.topology.room_stats import count_frontier_clusters
    world = world_from(TWO_ROOMS)
    labels = labels_for(world)
    cost = flat_cost(world)
    by_room = frontier_goals_by_room(world, cost, labels, at(3, 3), 0.0, min_cluster_cells=1)
    for room in (1, 2):
        alone = ranked_frontier_goals(world, cost, labels == room, at(3, 3), 0.0, min_cluster_cells=1)
        assert [g.cell for g in by_room[room]] == [g.cell for g in alone]
    counts = count_frontier_clusters(world.grid.astype(np.int8), labels, min_cluster_cells=1)
    assert {room: len(goals) for room, goals in by_room.items()} == \
        {room: n for room, n in counts.items() if n}


def test_a_straddling_cluster_goes_to_the_room_holding_most_of_it():
    """Three of its five cells are labelled 2, so the count credits room 2 -- and so do we."""
    world = world_from([
        "#########",
        "#.......#",
        "#.......#",
        "#.......#",
        "#?????..#",
        "#########",
    ])
    labels = np.zeros(world.grid.shape, dtype=np.int32)
    labels[world.grid == FREE] = 1
    labels[1:5, 3:8][world.grid[1:5, 3:8] == FREE] = 2     # room 2 owns columns 3..7
    by_room = frontier_goals_by_room(world, flat_cost(world), labels, at(6, 2), 0.0,
                                     min_cluster_cells=1)
    assert set(by_room) == {2}
    from sparx_agency.core.mapping.topology.room_stats import count_frontier_clusters
    counts = count_frontier_clusters(world.grid.astype(np.int8), labels, min_cluster_cells=1)
    assert counts == {1: 0, 2: 1}


def test_a_cluster_in_unlabelled_space_is_dropped_and_shape_is_checked():
    world = world_from([
        "#######",
        "#....?#",
        "#.....#",
        "#######",
    ])
    nowhere = np.zeros(world.grid.shape, dtype=np.int32)
    assert frontier_goals_by_room(world, flat_cost(world), nowhere, at(2, 2), 0.0,
                                  min_cluster_cells=1) == {}
    with pytest.raises(ValueError):
        frontier_goals_by_room(world, flat_cost(world), np.zeros((2, 2), np.int32), at(2, 2), 0.0)
    off_graph = FrontierRankingParams(snap_radius_m=0.5)
    labels = (world.grid == FREE).astype(np.int32)
    assert frontier_goals_by_room(world, flat_cost(world), labels, (20.0, 20.0), 0.0,
                                  min_cluster_cells=1, params=off_graph) == {}




# -- distances the way the planner flies them (2026-10-05) ------------------------------
def squeeze_world():
    """A wall with a one-cell squeeze near the robot and a wide gap far from it.

    The frontier lies on the far side of the wall. At the body radius the
    squeeze at row 1 is passable and the boundary is two cells away; at
    the preferred standoff only the gap at row 4 is, and the walk is round
    through it. Returns ``(world, body_cost, preferred_cost)``.
    """
    world = world_from([
        "###########",
        "#?........#",
        "#?...#....#",
        "#?...#....#",
        "#?...#....#",
        "#.........#",
        "###########",
    ])
    body = flat_cost(world)
    preferred = flat_cost(world)
    preferred[1, 5] = np.inf                           # the squeeze at row 1: lethal at the preferred standoff
    return world, body, preferred


def test_distances_are_measured_at_the_preferred_standoff_where_it_reaches():
    from sparx_agency.core.planning.exploration.frontier_ranking import accessible_frontiers
    world, body, preferred = squeeze_world()
    labels = np.ones(world.grid.shape, dtype=np.int32)
    origin = at(6, 1)
    through = accessible_frontiers(world, body, labels, origin, math.pi)
    around = accessible_frontiers(world, body, labels, origin, math.pi, preferred_cost=preferred)
    assert len(through.goals) == len(around.goals) == 1
    assert through.goals[0].geodesic_m < 6 * RES, "through the squeeze: a few cells"
    assert around.goals[0].geodesic_m > through.goals[0].geodesic_m + 3 * RES, (
        "the route A* will fly goes round through the gap, and that is what the goal is charged")
    assert around.goals[0].cell == through.goals[0].cell, "the same boundary, measured differently"
    gx, gy = through.goals[0].cell
    assert around.distance_m[gy, gx] == pytest.approx(around.goals[0].geodesic_m)
    # A cell reachable only through the squeeze keeps its body-radius distance rather than reading unreachable.
    world2, body2, preferred2 = squeeze_world()
    preferred2[5, 5] = np.inf                           # the gap at row 5 is a squeeze too
    only_squeeze = accessible_frontiers(world2, body2, labels, origin, math.pi, preferred_cost=preferred2)
    assert len(only_squeeze.goals) == 1 and only_squeeze.goals[0].geodesic_m == pytest.approx(through.goals[0].geodesic_m)
    with pytest.raises(ValueError):
        accessible_frontiers(world, body, labels, origin, math.pi, preferred_cost=preferred[:3])
