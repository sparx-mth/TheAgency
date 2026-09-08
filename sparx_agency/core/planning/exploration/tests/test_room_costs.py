"""Tests for the room-centre arc weights and the HPP-PT instance.

Two kinds of fact here, and the split matters.

The **synthetic** tests use a three-room corridor small enough to reason about
by hand, so a wrong answer names itself: two rooms four metres apart across a
wall have an arc weight that goes the long way round, and a centroid dropped in
a wall snaps to the floor beside it rather than being silently unreachable.

The **fixture** tests replay the real captured hospital BEV, because the three
properties RPT* actually depends on are properties of real geometry and cannot
be reproduced at toy scale:

* the matrix is **symmetric**, or the solver is optimising an asymmetric problem
  it was never proved correct on;
* the matrix is **metric** -- worst triangle-inequality violation 0.0 -- because
  RPT*'s dominance pruning consumes the triangle inequality directly, and a
  violation makes it discard optimal orders with no error anywhere;
* the whole build fits in a **control tick**, which is the entire reason this
  module is a Dijkstra sweep rather than N^2 A*.

The fixture is the same ``live_bev_hospital.npz`` the watershed regression uses:
413x200 at 0.15 m, 29 rooms, 5 of whose centroids land on a blocked cell.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_watershed import (
    segment_rooms_watershed)
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.exploration.room_costs import (
    HppPtInstance, RoomNode, build_instance, cost_matrix,
    in_room_frontier_goals, passable_graph, room_nodes, snap_cell)

FIXTURE = (
    "sparx_agency/core/mapping/topology/tests/fixtures/live_bev_hospital.npz")
BEV_VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
RES = 0.5

FREE, OCC, UNK = 0, 100, -1

#: Room centres of :func:`corridor_world`, in world metres. The world is
#: 4.5 m x 3.0 m, so a coordinate outside that is off the map entirely.
ROOM_A = (1.0, 2.0)
ROOM_B = (2.5, 2.0)
ROOM_C = (4.0, 2.0)
CORRIDOR = (2.5, 0.75)


def corridor_world():
    """Three 2 m rooms off a corridor, walls between them, at 0.5 m/cell.

        row 0 is minimum y

        col   0 1 2 3 4 5 6 7 8
        row 5  # # # # # # # # #
        row 4  # A A # B B # C C
        row 3  # A A # B B # C C
        row 2  # . . . . . . . .    <- the corridor, the only way between
        row 1  # . . . . . . . .
        row 0  # # # # # # # # #
    """
    g = np.full((6, 9), OCC, dtype=np.int8)
    g[1:3, 1:9] = FREE                  # corridor
    g[3:5, 1:3] = FREE                  # room A
    g[3:5, 4:6] = FREE                  # room B
    g[3:5, 7:9] = FREE                  # room C
    return OccupancyGrid2D(
        g.astype(np.int16),
        OccupancyGrid2DParams(RES, 0.0, 0.0, "world"),
        values=BEV_VALUES)


def walled_world():
    """Solid wall everywhere but one cell in the far corner."""
    g = np.full((6, 9), OCC, dtype=np.int8)
    g[5, 8] = FREE
    return OccupancyGrid2D(g.astype(np.int16),
                           OccupancyGrid2DParams(RES, 0.0, 0.0, "world"),
                           values=BEV_VALUES)


def walled_cost():
    """The cost array of :func:`walled_world`."""
    return flat_cost(walled_world())


def flat_cost(world, blocked_value=OCC):
    """A cost array that is 1.0 on free cells and inf on everything else.

    Stands in for ``WeightedAStarPlanner2D.cost_for`` without importing the
    planner: this module never looks at the cost VALUES, only at which cells
    are finite, and keeping the planner out keeps these tests fast.
    """
    cost = np.full(world.grid.shape, np.inf, dtype=np.float64)
    cost[world.grid == FREE] = 1.0
    return cost


# -- the graph and the snapper -------------------------------------------
def test_passable_graph_indexes_only_free_cells():
    world = corridor_world()
    ids, graph = passable_graph(flat_cost(world))
    free = world.grid == FREE
    assert int(free.sum()) == int((ids >= 0).sum())
    assert (ids[~free] == -1).all()
    assert graph.shape == (int(free.sum()), int(free.sum()))
    # Forward edges only; directed=False supplies the rest.
    assert graph.nnz < 4 * int(free.sum())


def test_snap_cell_returns_the_cell_itself_when_already_passable():
    world = corridor_world()
    ids, _ = passable_graph(flat_cost(world))
    assert snap_cell(ids, 1, 1, 4) == (1, 1)


def test_snap_cell_finds_the_nearest_free_cell_from_inside_a_wall():
    world = corridor_world()
    ids, _ = passable_graph(flat_cost(world))
    # (3, 3) is the wall between rooms A and B. Every free neighbour is one
    # cell away, so any of them is a correct answer -- what matters is that
    # the result is ON the graph and adjacent, not which tie it broke.
    got = snap_cell(ids, 3, 3, 4)
    assert got is not None
    assert ids[got[1], got[0]] >= 0
    assert max(abs(got[0] - 3), abs(got[1] - 3)) == 1


def test_snap_cell_gives_up_rather_than_reaching_across_the_map():
    ids, _ = passable_graph(walled_cost())
    assert snap_cell(ids, 1, 1, 3) is None


def test_room_nodes_snaps_a_walled_centroid_and_flags_it():
    world = corridor_world()
    cost = flat_cost(world)
    # A centroid deliberately dropped in the wall column 3.
    nodes, dropped = room_nodes(world, cost, {7: (1.75, 1.75)}, snap_radius_m=2.0)
    assert dropped == []
    assert len(nodes) == 1
    assert nodes[0].pid == 7
    assert nodes[0].snapped is True
    assert np.isfinite(cost[nodes[0].cell[1], nodes[0].cell[0]])


def test_room_nodes_drops_a_room_with_no_passable_cell_in_radius():
    world = walled_world()
    nodes, dropped = room_nodes(world, walled_cost(),
                                {3: (0.75, 0.75)}, snap_radius_m=1.0)
    assert nodes == []
    assert dropped == [3]


def test_room_nodes_are_in_ascending_pid_order():
    world = corridor_world()
    cents = {9: ROOM_A, 2: ROOM_B, 5: ROOM_C}
    nodes, _ = room_nodes(world, flat_cost(world), cents)
    assert [n.pid for n in nodes] == [2, 5, 9]


# -- the matrix -----------------------------------------------------------
def test_cost_matrix_routes_around_the_wall_not_through_it():
    """Rooms A and B are 1.5 m apart in a straight line and 4 m apart to walk."""
    world = corridor_world()
    ids, graph = passable_graph(flat_cost(world))
    a = (2, 4)   # room A, top-right cell
    b = (4, 4)   # room B, top-left cell
    C = cost_matrix(world, graph, ids, [a, b])
    straight = RES * np.hypot(b[0] - a[0], b[1] - a[1])
    assert C[0, 1] > straight * 1.5
    assert C[0, 1] == pytest.approx(C[1, 0])
    assert C[0, 0] == 0.0


def test_cost_matrix_is_infinite_for_an_unreachable_cell():
    world = corridor_world()
    g = world.grid.copy()
    g[1:3, 6] = OCC                      # seal room C off from the corridor
    sealed = OccupancyGrid2D(g, world.params, values=BEV_VALUES)
    ids, graph = passable_graph(flat_cost(sealed))
    C = cost_matrix(sealed, graph, ids, [(1, 1), (8, 4)])
    assert not np.isfinite(C[0, 1])


# -- the instance ---------------------------------------------------------
def three_room_instance(**kw):
    world = corridor_world()
    cents = {1: ROOM_A, 2: ROOM_B, 3: ROOM_C}
    probs = {1: 0.6, 2: 0.3, 3: 0.1}
    return build_instance(world, flat_cost(world), cents, probs, **kw)


def test_build_instance_shape_and_index_map():
    inst, dropped = three_room_instance()
    assert isinstance(inst, HppPtInstance)
    assert dropped == []
    assert inst.C.shape == (3, 3)
    assert inst.p.shape == (3,)
    assert inst.index_to_pid == (1, 2, 3)
    assert all(isinstance(n, RoomNode) for n in inst.nodes)
    assert 0 <= inst.depot < inst.n


def test_build_instance_appends_the_aircraft_as_its_own_depot():
    inst, _ = three_room_instance(depot_xy=CORRIDOR)
    assert inst.n == 4
    assert inst.depot == 3
    assert inst.index_to_pid[-1] == -1
    assert inst.p[inst.depot] == 0.0


def test_probabilities_are_never_exactly_one():
    """RPT*'s heuristic divides by ``1 - p``; a lone room would divide by zero."""
    world = corridor_world()
    inst, _ = build_instance(world, flat_cost(world),
                             {1: ROOM_A}, {1: 1.0})
    assert inst.p[0] < 1.0
    assert 1.0 - inst.p[0] > 0.0


def test_an_unranked_room_scores_zero_rather_than_disappearing():
    world = corridor_world()
    inst, dropped = build_instance(world, flat_cost(world),
                                   {1: ROOM_A, 2: ROOM_B}, {1: 0.9})
    assert dropped == []
    assert inst.index_to_pid == (1, 2)
    assert inst.p[1] == 0.0


def test_probabilities_are_not_normalised_to_sum_to_one():
    """Normalising over mapped rooms asserts the target is certainly in one."""
    inst, _ = three_room_instance()
    assert inst.p.sum() == pytest.approx(1.0)   # these happen to sum to 1
    world2 = corridor_world()
    inst2, _ = build_instance(world2, flat_cost(world2),
                              {1: ROOM_A, 2: ROOM_B}, {1: 0.2, 2: 0.1})
    assert inst2.p.sum() == pytest.approx(0.3)  # left alone, not rescaled


def test_metres_by_default_and_seconds_once_a_speed_is_given():
    metres, _ = three_room_instance(cruise_speed_mps=0.0)
    assert metres.units == "metres"
    seconds, _ = three_room_instance(cruise_speed_mps=0.5)
    assert seconds.units == "seconds"
    assert seconds.C[0, 1] == pytest.approx(metres.C[0, 1] / 0.5)


def test_the_search_budget_is_charged_on_arcs_entering_a_room():
    plain, _ = three_room_instance(cruise_speed_mps=0.5)
    budgeted, _ = three_room_instance(cruise_speed_mps=0.5, search_time_s=30.0)
    depot = budgeted.depot
    for j in range(budgeted.n):
        if j == depot:
            continue
        assert budgeted.C[depot, j] == pytest.approx(plain.C[depot, j] + 30.0)
    assert np.allclose(np.diag(budgeted.C), 0.0)


def test_the_budgeted_matrix_is_still_metric():
    """Folding a constant onto entering arcs must not break the solver."""
    inst, _ = three_room_instance(cruise_speed_mps=0.5, search_time_s=45.0)
    C = inst.C
    for i in range(inst.n):
        for j in range(inst.n):
            for k in range(inst.n):
                assert C[i, j] <= C[i, k] + C[k, j] + 1e-9


def test_an_unreachable_room_is_dropped_not_given_a_fake_weight():
    world = corridor_world()
    g = world.grid.copy()
    g[1:3, 6] = OCC                      # seal room C
    sealed = OccupancyGrid2D(g, world.params, values=BEV_VALUES)
    inst, dropped = build_instance(
        sealed, flat_cost(sealed),
        {1: ROOM_A, 2: ROOM_B, 3: ROOM_C},
        {1: 0.5, 2: 0.3, 3: 0.2}, depot_xy=CORRIDOR)
    assert 3 in dropped
    assert 3 not in inst.index_to_pid
    assert np.isfinite(inst.C).all()


def test_the_frontier_blend_lifts_an_unexplored_room():
    world = corridor_world()
    cents = {1: ROOM_A, 2: ROOM_B}
    probs = {1: 0.5, 2: 0.5}
    blended, _ = build_instance(world, flat_cost(world), cents, probs,
                                frontier_weight=0.5,
                                frontier_counts={1: 0, 2: 4})
    assert blended.p[1] > blended.p[0]


def test_build_instance_refuses_an_empty_room_set():
    world = corridor_world()
    with pytest.raises(ValueError):
        build_instance(world, flat_cost(world), {}, {})


# -- in-room sweep goals --------------------------------------------------
def test_in_room_frontier_goals_stay_inside_the_mask():
    world = corridor_world()
    g = world.grid.copy()
    g[3:5, 7:9] = UNK                     # room C becomes unscanned
    part = OccupancyGrid2D(g, world.params, values=BEV_VALUES)
    mask = np.zeros(g.shape, dtype=bool)
    mask[1:3, 1:9] = True                 # the corridor only
    goals = in_room_frontier_goals(part, flat_cost(part), mask,
                                   min_cluster_cells=1)
    assert goals
    for x, y in goals:
        gx, gy = part.world_to_grid(x, y)
        assert mask[gy, gx], "a goal escaped the room mask"


def test_in_room_frontier_goals_are_empty_for_a_fully_scanned_room():
    world = corridor_world()
    mask = np.zeros(world.grid.shape, dtype=bool)
    mask[3:5, 1:3] = True                 # room A, no unknown anywhere near
    assert in_room_frontier_goals(world, flat_cost(world), mask) == []


def test_in_room_frontier_goals_are_largest_cluster_first():
    """A five-cell unscanned boundary must be offered before a two-cell one."""
    g = np.full((7, 12), OCC, dtype=np.int8)
    g[1:6, 1:11] = FREE                    # one open hall
    g[1, 1] = UNK                          # boundary of 2 free cells
    g[1:6, 8:11] = UNK                     # boundary of 5, down column 7
    hall = OccupancyGrid2D(g.astype(np.int16),
                           OccupancyGrid2DParams(RES, 0.0, 0.0, "world"),
                           values=BEV_VALUES)
    mask = np.zeros(g.shape, dtype=bool)
    mask[1:6, 1:11] = True
    goals = in_room_frontier_goals(hall, flat_cost(hall), mask,
                                   min_cluster_cells=1)
    assert len(goals) == 2
    first = hall.world_to_grid(*goals[0])
    assert first[0] == 7, "the long boundary at column 7 should come first"


# -- the real hospital ----------------------------------------------------
@pytest.fixture(scope="module")
def hospital():
    """The captured hospital BEV, segmented into rooms, as the mission sees it."""
    data = np.load(FIXTURE)
    grid = data["grid"]
    res = float(data["res"])
    world = OccupancyGrid2D(
        grid.astype(np.int16),
        OccupancyGrid2DParams(res, float(data["ox"]), float(data["oy"]), "world"),
        values=BEV_VALUES)
    doors = [(int(a), int(b)) for a, b in data["door_cells"]]
    _, _, stats = segment_rooms_watershed(grid == FREE, res, door_cells=doors)
    cents = {}
    for st in stats:
        cx, cy = st.centroid_cells
        cents[int(st.id)] = world.grid_to_world(int(round(cx)), int(round(cy)))
    return world, cents


def test_hospital_matrix_is_symmetric_and_metric(hospital):
    """The two properties RPT*'s optimality proof consumes."""
    world, cents = hospital
    cost = flat_cost(world)
    inst, _ = build_instance(world, cost, cents,
                             {pid: 1.0 / len(cents) for pid in cents},
                             cruise_speed_mps=0.0)
    C = inst.C
    assert C.shape[0] >= 20, "the fixture should segment into many rooms"
    assert np.allclose(C, C.T), "asymmetric arc weights"
    assert np.isfinite(C).all(), "the instance must be a complete graph"
    # Triangle inequality over every triple, vectorised over the middle vertex.
    viol = 0.0
    for k in range(C.shape[0]):
        viol = max(viol, float(np.max(C - (C[:, [k]] + C[[k], :]))))
    assert viol <= 1e-9, "worst triangle violation %.9f m" % viol


def test_hospital_snaps_the_centroids_that_land_in_walls(hospital):
    world, cents = hospital
    nodes, dropped = room_nodes(world, flat_cost(world), cents)
    assert dropped == [], "no room should be unreachable on the captured map"
    assert sum(1 for n in nodes if n.snapped) >= 1, (
        "some hospital centroids land on blocked cells and must snap")
    for n in nodes:
        assert np.isfinite(flat_cost(world)[n.cell[1], n.cell[0]])


def test_hospital_instance_builds_inside_a_control_tick(hospital):
    """The whole reason this is a Dijkstra sweep and not N^2 A*."""
    world, cents = hospital
    inst, _ = build_instance(world, flat_cost(world), cents,
                             {pid: 1.0 / len(cents) for pid in cents})
    assert inst.build_ms < 1000.0, (
        "arc weights took %.0f ms; the replan tick is 1 s" % inst.build_ms)
