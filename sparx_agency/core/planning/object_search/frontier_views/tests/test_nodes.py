"""Frontier nodes: the inventory's clusters merged into the places a picture is asked about, each with an id that stays.

The clusters come from the method's own inventory
(``frontier_ranking.accessible_frontiers``) over small plans, so what is
tested is the seam the nodes are built on: the inventory hands over the
cells of every goal's cluster, and the nodes are those cells -- merged a
doorway apart, floored at half a metre, named by a registry that knows the
place again on the next action.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.exploration.frontier_ranking import accessible_frontiers
from sparx_agency.core.planning.exploration.room_costs import FrontierCluster
from sparx_agency.core.planning.object_search.frontier_views.nodes import NodeSettings, frontier_nodes
from sparx_agency.core.planning.object_search.frontier_views.tests.rig import ORIGIN, RESOLUTION, VALUES, Plan
from sparx_agency.core.planning.object_search.rooms.opening_nodes import merge_near_seed


def inventory_of(plan, here=(0.55, 0.05)):
    """The frontier inventory of ``plan`` from ``here``, every free cell passable."""
    world = plan.world()
    cost = np.where(world.grid == VALUES.free, 1.0, np.inf)
    labels = np.zeros(world.grid.shape, dtype=np.int32)
    return world, accessible_frontiers(world, cost, labels, here, 0.0)


def cluster_at(x, y, cells=8, along="y"):
    """A cluster of ``cells`` cells in a row starting at the world point ``(x, y)``, its goal at its middle."""
    col, row = int((x - ORIGIN) / RESOLUTION), int((y - ORIGIN) / RESOLUTION)
    rows = row + (np.arange(cells) if along == "y" else np.zeros(cells, int))
    cols = col + (np.arange(cells) if along == "x" else np.zeros(cells, int))
    return FrontierCluster(cells, (int(cols[cells // 2]), int(rows[cells // 2])), rows, cols)


def two_openings() -> Plan:
    """A 6 x 4 m room open at its east end (a 4 m boundary) and at a 0.8 m gap in its north wall, 4.5 m away."""
    plan = Plan().free(0.0, -2.0, 6.0, 2.0)
    plan.wall(-0.1, -2.1, 0.0, 2.1).wall(-0.1, -2.1, 6.0, -2.0).wall(-0.1, 2.0, 6.0, 2.1)
    plan.free(1.0, 2.0, 1.8, 2.5)
    return plan


def test_the_inventory_hands_over_the_cells_of_every_goals_cluster_in_the_goals_order():
    world, inventory = inventory_of(two_openings())
    assert len(inventory.goals) == len(inventory.clusters) >= 2
    for goal, cluster in zip(inventory.goals, inventory.clusters):
        assert cluster.cell == goal.cell and cluster.size == goal.size_cells == len(cluster.rows) == len(cluster.cols)
        assert (world.grid[cluster.rows, cluster.cols] == VALUES.free).all()
    covered = np.zeros(world.grid.shape, bool)
    for cluster in inventory.clusters:
        assert not covered[cluster.rows, cluster.cols].any()             # no cell in two clusters
        covered[cluster.rows, cluster.cols] = True
    assert (covered == inventory.cells).all()                             # the clusters ARE the inventory's cells
    assert accessible_frontiers(world, np.full(world.grid.shape, np.inf), np.zeros(world.grid.shape, np.int32),
                                (0.55, 0.05), 0.0).clusters == ()


def test_the_nodes_are_the_inventorys_boundaries_each_with_its_cells_and_a_goal_point():
    world, inventory = inventory_of(two_openings())
    registry = NodeSettings().registry()
    nodes = frontier_nodes(world, inventory.clusters, registry, 0)
    assert len(nodes) == 2 and [node.size_cells for node in nodes] == sorted((n.size_cells for n in nodes), reverse=True)
    east, north = nodes
    assert east.size_cells == 40 and east.floor_id == 0 and east.xy[0] == pytest.approx(5.95, abs=0.06)
    assert north.xy == pytest.approx((1.4, 2.45), abs=0.11) and north.size_cells >= 8
    assert {east.node_id, north.node_id} == {0, 1}
    for node in nodes:
        assert (world.grid[node.rows, node.cols] == VALUES.free).all()
        assert world.grid_to_world(*node.cell) == pytest.approx(node.xy)
        assert (node.rows == node.cell[1]).any() or node.parts > 1          # the goal is a cell of the seed cluster


def test_an_id_stays_while_the_boundary_moves_less_than_the_match_distance():
    plan = two_openings()
    registry = NodeSettings().registry()
    world, inventory = inventory_of(plan)
    before = {node.node_id: node for node in frontier_nodes(world, inventory.clusters, registry, 0)}
    plan.free(6.0, -2.0, 6.5, 2.0)                                # the east boundary recedes half a metre
    world, inventory = inventory_of(plan)
    after = {node.node_id: node for node in frontier_nodes(world, inventory.clusters, registry, 0)}
    assert set(after) == set(before)
    east = max(before, key=lambda node_id: before[node_id].size_cells)
    assert after[east].xy[0] == pytest.approx(before[east].xy[0] + 0.5, abs=0.11)
    plan.free(6.5, -2.0, 9.0, 2.0)                                # ... and then two and a half more: another place
    world, inventory = inventory_of(plan)
    moved = frontier_nodes(world, inventory.clusters, registry, 0)
    assert east not in {node.node_id for node in moved} and max(node.node_id for node in moved) == 2
    # The same place on another storey is another node.
    world, inventory = inventory_of(two_openings())
    upstairs = frontier_nodes(world, inventory.clusters, registry, 1)
    assert not {node.node_id for node in upstairs} & {0, 1, 2} and all(node.floor_id == 1 for node in upstairs)


def test_clusters_a_doorway_apart_are_one_node_seeded_by_the_longest():
    world = Plan().free(0.0, -2.0, 6.0, 2.0).world()
    long, short, far = cluster_at(3.0, 0.0, 12), cluster_at(3.9, 0.3, 6), cluster_at(3.0, 5.0, 9)
    for order in ([long, short, far], [far, short, long], [short, far, long]):
        nodes = frontier_nodes(world, order, NodeSettings().registry(), 0)
        assert [(node.size_cells, node.parts) for node in nodes] == [(18, 2), (9, 1)]
        assert nodes[0].cell == long.cell                           # the longest cluster's goal, whatever the order
        assert set(zip(nodes[0].rows, nodes[0].cols)) == set(zip(long.rows, long.cols)) | set(zip(short.rows, short.cols))
    # Nothing merges at a merge distance under their separation.
    apart = frontier_nodes(world, [long, short, far], NodeSettings(merge_m=0.5, match_m=0.5).registry(), 0,
                           NodeSettings(merge_m=0.5, match_m=0.5))
    assert [node.parts for node in apart] == [1, 1, 1]


def test_a_boundary_under_half_a_metre_is_no_node_unless_its_neighbours_make_it_one():
    world = Plan().free(0.0, -2.0, 6.0, 2.0).world()
    speck = cluster_at(1.0, 0.0, 4)
    assert frontier_nodes(world, [speck], NodeSettings().registry(), 0) == []
    assert len(frontier_nodes(world, [speck], NodeSettings(min_cells=4).registry(), 0, NodeSettings(min_cells=4))) == 1
    pair = frontier_nodes(world, [speck, cluster_at(1.5, 0.0, 3)], NodeSettings().registry(), 0)
    assert [(node.size_cells, node.parts) for node in pair] == [(7, 2)]
    # A dropped group claims no id: the next node is still number 0.
    registry = NodeSettings().registry()
    frontier_nodes(world, [speck], registry, 0)
    assert frontier_nodes(world, [cluster_at(4.0, 0.0, 9)], registry, 0)[0].node_id == 0


def test_no_two_nodes_of_one_action_share_an_id():
    world = Plan().free(-9.0, -9.0, 9.0, 9.0).world()
    registry = NodeSettings().registry()
    rng = np.random.RandomState(5)
    for _ in range(30):
        clusters = [cluster_at(rng.uniform(-8, 8), rng.uniform(-8, 7), int(rng.randint(5, 12))) for _ in range(12)]
        ids = [node.node_id for node in frontier_nodes(world, clusters, registry, 0)]
        assert len(ids) == len(set(ids))


def test_merge_near_seed_groups_by_the_first_item_of_each_group():
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.4, 0.0), (2.9, 0.0)]
    groups = merge_near_seed(points, 1.2, xy=lambda point: point)
    # (2, 0) is 2 m from the first seed and seeds its own group, though it is a metre from a member of the first.
    assert groups == [[(0.0, 0.0), (1.0, 0.0), (0.4, 0.0)], [(2.0, 0.0), (2.9, 0.0)]]
    assert merge_near_seed([], 1.0) == []

    class Goal:
        def __init__(self, xy):
            self.xy = xy

    goals = [Goal((0.0, 0.0)), Goal((0.5, 0.0)), Goal((5.0, 0.0))]
    assert [[goal.xy for goal in group] for group in merge_near_seed(goals, 1.5)] == [[(0.0, 0.0), (0.5, 0.0)], [(5.0, 0.0)]]


@pytest.mark.parametrize("override", [{"min_cells": 0}, {"merge_m": 0.0}, {"match_m": -1.0}, {"merge_m": 1.0, "match_m": 1.2},
                                      {"min_cells": 2.5}])
def test_node_settings_refuse_values_outside_their_range(override):
    with pytest.raises(ValueError):
        NodeSettings(**override)
