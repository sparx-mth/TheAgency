"""The unknown behind a room's frontiers: one united figure per room, from the live inventory or the map alone."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.view_gain import UnknownView, ViewCone
from sparx_agency.core.planning.object_search.rooms.frontier_unknown import (
    coarse_unknown_words, room_frontier_seeds, rooms_unknown_behind, unknown_behind)

RES = 0.1
VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
CONE = ViewCone(half_fov_rad=math.radians(39.5), max_range_m=5.0, min_range_m=0.0)


def world_with_room():
    """A 4 x 4 m room (x 2-6, y 2-6) walled on three sides, open to the unknown on its +x side; the hallway beyond
    the -x wall's door is room 2 and leads nowhere unknown."""
    grid = np.full((120, 120), -1, np.int8)
    labels = np.zeros((120, 120), np.int32)
    grid[20:60, 20:60] = 0
    labels[20:60, 20:60] = 1
    grid[19, 19:61] = 100
    grid[60, 19:61] = 100
    grid[19:61, 19] = 100                                      # the -x wall ...
    grid[36:44, 19] = 0                                        # ... with a doorway
    grid[36:44, 10:19] = 0
    labels[36:44, 10:19] = 2                                   # the hallway behind it, walled
    grid[35, 9:19] = 100
    grid[44, 9:19] = 100
    grid[35:45, 9] = 100
    world = OccupancyGrid2D(grid, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    return world, labels


def test_the_seeds_come_one_per_frontier_cluster_of_each_room_off_the_map_alone():
    world, labels = world_with_room()
    seeds = room_frontier_seeds(world, labels)
    assert set(seeds) == {0}, "only the room's open +x side borders unknown; the hallway is walled"
    assert len(seeds[0]) == 1
    gx, gy = seeds[0][0]
    assert gx == 59 and 20 <= gy < 60, "the seed sits on the open edge"
    # A settled strip along the edge is no frontier.
    resolved = np.zeros_like(labels, dtype=bool)
    resolved[20:60, 60:64] = True
    assert room_frontier_seeds(world, labels, resolved=resolved) == {}
    # Two separate openings in a wall are two clusters.
    grid = world.grid.copy()
    grid[19:61, 60] = 100
    grid[22:26, 60] = -1
    grid[50:54, 60] = -1
    two = OccupancyGrid2D(grid, world.params, values=VALUES)
    assert len(room_frontier_seeds(two, labels)[0]) == 2


def test_the_unknown_behind_a_frontier_is_the_cone_into_the_unknown_and_several_cones_are_united():
    world, labels = world_with_room()
    view = UnknownView(world, CONE)
    m2, mask = unknown_behind(view, world, [(59, 40)])
    assert 8.0 < m2 < 20.0, m2                                 # a 5 m cone of 79 degrees holds about 17 m2 at most
    assert mask[40, 60] and not mask[40, 50], "it starts beyond the edge; the known floor is not unknown"
    # Two seeds a cell apart see nearly the same unknown: the union is barely more than one, a sum would double it.
    both, _ = unknown_behind(view, world, [(59, 40), (59, 41)])
    assert both < 1.15 * m2
    # Two seeds at opposite ends of the open side see different unknown: the union grows.
    apart, _ = unknown_behind(view, world, [(59, 22), (59, 57)])
    assert apart > 1.3 * m2
    assert rooms_unknown_behind(view, world, {0: [(59, 40)], 1: [(14, 40)]}) == {0: pytest.approx(m2), 1: 0.0}, (
        "the hallway's cell has no unknown heading: nothing behind it")
    assert unknown_behind(view, world, [(500, 500)])[0] == 0.0, "off the map: nothing"


def test_the_words_are_a_count_with_the_unknown_in_brackets():
    assert coarse_unknown_words(30.4, 2) == "2 (~30m2 unknown behind them)"
    assert coarse_unknown_words(7.6, 1) == "1 (~8m2 unknown behind it)"
    assert coarse_unknown_words(0.2, 3) == "3 (under 1m2 unknown behind them)"
    assert coarse_unknown_words(12.0, 0) == "0" and coarse_unknown_words(None, 2) == "2"
