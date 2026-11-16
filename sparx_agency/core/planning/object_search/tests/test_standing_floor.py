"""The standing floor: observed free, room for the body, joined to the agent -- and never anything unobserved.

``world_model/standing_floor.py`` is everything the closing knows about where
the agent may stand since the simulator's navigation mesh was taken away from
it (2026-11-15). These tests pin the three rules of its docstring on small
hand-drawn grids, and then on random ones as properties: whatever the map
looks like, a standing point is a cell observed free whose whole body disc is
observed free, reached through cells like it.

numpy only; no simulator, no policy.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.object_search.world_model.standing_floor import (
    StandingFloor, body_disc, obstacle_skirt, standable_cells, standing_floor)

FREE, OCCUPIED, UNKNOWN = 0, 100, -1
VALUES = OccupancyValues(free=FREE, occupied=OCCUPIED, unknown=UNKNOWN)
RES = 0.1
BODY = 0.18          # the agent's radius: the 3 x 3 block at 0.1 m
ROOMY = 0.25         # the default planner's floor standoff: the 5 x 5 block without its corners


def grid(data) -> OccupancyGrid2D:
    """``data[gy, gx]`` as the agent's map, cell (0, 0) at the world origin."""
    return OccupancyGrid2D(np.asarray(data, dtype=np.int16), OccupancyGrid2DParams(RES, 0.0, 0.0), values=VALUES)


def centre(gx, gy):
    return ((gx + 0.5) * RES, (gy + 0.5) * RES)


def unknown_map(h=40, w=40):
    return np.full((h, w), UNKNOWN, dtype=np.int16)


def room(h=40, w=40, x0=5, x1=35, y0=5, y1=35):
    """A free rectangle ``[y0, y1) x [x0, x1)`` in an unobserved map."""
    data = unknown_map(h, w)
    data[y0:y1, x0:x1] = FREE
    return data


# -- rule 2's disc -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("clearance, cells, corner", [(0.18, 9, True), (0.25, 21, False), (0.30, 29, False), (0.0, 1, True)])
def test_the_body_disc_is_the_cells_whose_centres_lie_within_the_clearance(clearance, cells, corner):
    disc = body_disc(clearance, RES)
    assert disc.shape[0] == disc.shape[1] and disc.shape[0] % 2 == 1
    assert int(disc.sum()) == cells and bool(disc[0, 0]) is corner
    n = disc.shape[0] // 2
    assert disc[n, n], "the cell itself"
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            assert bool(disc[dy + n, dx + n]) == (math.hypot(dx, dy) * RES <= clearance + 1e-9)


@pytest.mark.parametrize("clearance, resolution", [(-0.1, 0.1), (float("nan"), 0.1), (float("inf"), 0.1), (0.18, 0.0), (0.18, -1.0)])
def test_the_body_disc_refuses_a_radius_or_a_resolution_that_means_nothing(clearance, resolution):
    with pytest.raises(ValueError):
        body_disc(clearance, resolution)


# -- rule 1: observed free ---------------------------------------------------------------------------------
def test_nothing_is_standing_floor_on_a_map_that_is_all_unknown_or_all_occupied():
    for value in (UNKNOWN, OCCUPIED):
        world = grid(np.full((30, 30), value))
        assert not standable_cells(world, BODY).any()
        floor = standing_floor(world, centre(15, 15), BODY)
        assert not floor.reachable.any() and not floor.seeds.any()
        assert floor.nearest(centre(15, 15), 5.0) is None and not floor.contains(centre(15, 15))


def test_only_cells_observed_free_are_standing_floor_and_unknown_is_not_floor():
    world = grid(room())
    standable = standable_cells(world, BODY)
    assert standable.any()
    assert (world.grid[standable] == FREE).all()
    assert not standable[world.grid == UNKNOWN].any()


# -- rule 2: room for the body, all of it observed ---------------------------------------------------------
def test_a_free_cell_beside_unknown_or_occupied_cells_is_not_standing_floor():
    data = room(x0=5, x1=35, y0=5, y1=35)
    world = grid(data)
    at_body, at_roomy = standable_cells(world, BODY), standable_cells(world, ROOMY)
    # The outermost ring of the free rectangle touches unknown cells: the body would stand half on unseen floor.
    assert not at_body[5, 5:35].any() and not at_body[5:35, 5].any()
    assert at_body[6, 6] and at_body[33, 33]
    assert int(at_body.sum()) == 28 * 28
    # At 0.25 m two rings go, and the corner cell of what is left keeps its corner neighbour (0.28 m away).
    assert not at_roomy[6, 6:34].any() and at_roomy[7, 7]
    assert int(at_roomy.sum()) == 26 * 26
    # One obstacle in the open floor takes its whole body disc with it, and one unknown cell does exactly the same.
    for value in (OCCUPIED, UNKNOWN):
        holed = data.copy()
        holed[20, 20] = value
        hole = standable_cells(grid(holed), BODY)
        assert not hole[19:22, 19:22].any() and hole[18, 20] and hole[20, 22]
        assert int(hole.sum()) == 28 * 28 - 9
        wide = standable_cells(grid(holed), ROOMY)
        assert not wide[20, 18:23].any() and not wide[19, 19] and wide[18, 18]
        assert int(wide.sum()) == 26 * 26 - 21


def test_the_edge_of_the_grid_is_unobserved_too():
    world = grid(np.zeros((12, 12)))             # free right up to the edge
    at_body, at_roomy = standable_cells(world, BODY), standable_cells(world, ROOMY)
    assert not at_body[0].any() and not at_body[:, -1].any() and at_body[1:-1, 1:-1].all()
    assert not at_roomy[:2].any() and not at_roomy[:, -2:].any() and at_roomy[2:-2, 2:-2].all()


# -- rule 3: joined to the agent ---------------------------------------------------------------------------
def two_rooms(door_cells, wall=OCCUPIED):
    """Two free rooms either side of a one-cell wall at column 20, joined by a doorway ``door_cells`` wide."""
    data = room(h=40, w=44, x0=4, x1=40, y0=4, y1=36)
    data[:, 20] = wall
    data[20:20 + door_cells, 20] = FREE
    return data


def test_a_doorway_joins_two_rooms_only_when_the_body_fits_through_it_on_the_map():
    here, there = centre(10, 20), centre(30, 20)
    for door, body_fits, roomy_fits in ((2, False, False), (3, True, False), (4, True, False), (5, True, True)):
        world = grid(two_rooms(door))
        tight, wide = standing_floor(world, here, BODY), standing_floor(world, here, ROOMY)
        assert tight.contains(here) and wide.contains(here)
        assert tight.contains(there) is body_fits, "a %d-cell doorway at the body radius" % door
        assert wide.contains(there) is roomy_fits, "a %d-cell doorway at 0.25 m" % door
        assert tight.standable_at(there) and wide.standable_at(there), "the far room is floor either way; it is the way in that is missing"


def test_unknown_floor_between_two_observed_rooms_is_not_a_way_through():
    """The toilet seen through a doorway whose threshold the level camera never saw: floor on both sides, nothing joining them."""
    here, there = centre(10, 20), centre(30, 20)
    data = room(h=40, w=44, x0=4, x1=40, y0=4, y1=36)
    data[:, 19:22] = UNKNOWN                         # a band of unseen floor the whole height of the map
    floor = standing_floor(grid(data), here, BODY)
    assert floor.contains(here) and floor.standable_at(there) and not floor.contains(there)
    assert not floor.reachable[:, 18:].any(), "nothing on or beyond the unseen band, and not the free cells that touch it"
    assert floor.nearest(there, 0.5) is None, "the far floor is seen, clear and not reachable: no standing point"
    # One row of the band observed is not a way through either: the body is three cells wide.
    data[20, 19:22] = FREE
    assert not standing_floor(grid(data), here, BODY).contains(there)
    data[19:22, 19:22] = FREE                        # three rows are
    assert standing_floor(grid(data), here, BODY).contains(there)


def test_one_unobserved_cell_in_a_corridor_three_cells_wide_closes_it():
    """A sampling hole is unobserved space like any other: the corridor is closed until the cell has been seen."""
    data = unknown_map(20, 60)
    data[8:11, 5:55] = FREE                          # a corridor exactly as wide as the body
    here, end = centre(8, 9), centre(50, 9)
    assert standing_floor(grid(data), here, BODY).contains(end)
    data[9, 30] = UNKNOWN
    floor = standing_floor(grid(data), here, BODY)
    assert not floor.contains(end) and floor.contains(centre(28, 9)) and not floor.contains(centre(29, 9))


def test_the_chain_starts_under_the_agents_own_body_so_a_wall_at_its_shoulder_does_not_strand_it():
    data = room()
    data[:, 12] = OCCUPIED                            # a wall one cell from where the agent stands
    world = grid(data)
    here = centre(13, 20)
    floor = standing_floor(world, here, BODY)
    assert not floor.standable_at(here) and not floor.contains(here), "the agent's own cell is in the wall's skirt"
    assert floor.seeds[20, 14] and floor.contains(centre(30, 20)), "the cell beside it is not, and the room follows"
    assert not floor.reachable[:, :12].any(), "never through the wall"
    # Walled in on the map on every side within its body: no chain starts, and nothing is reachable.
    boxed = room()
    boxed[17:24, 17:24] = OCCUPIED
    boxed[20, 20] = FREE
    stranded = standing_floor(grid(boxed), centre(20, 20), BODY)
    assert not stranded.seeds.any() and not stranded.reachable.any() and stranded.standable.any()
    assert stranded.nearest(centre(30, 30), 1.0) is None


def test_a_seed_is_never_on_the_far_side_of_a_wall():
    """Stepping out of a skirt is allowed; stepping through a one-cell wall is not, however near the far side is."""
    data = room()
    data[:, 20] = OCCUPIED
    for clearance in (BODY, ROOMY, 0.30):
        floor = standing_floor(grid(data), centre(19, 20), clearance)
        assert not floor.reachable[:, 20:].any() and not floor.seeds[:, 20:].any()
        assert floor.reachable[:, :20].any()


# -- the questions the closing asks ------------------------------------------------------------------------
def test_nearest_moves_a_point_to_standing_floor_within_the_radius_and_refuses_beyond_it():
    data = room()
    data[18:23, 25:30] = OCCUPIED                     # a cabinet: columns 25-29, rows 18-22
    world = grid(data)
    floor = standing_floor(world, centre(10, 20), BODY)
    # A point that is standing floor comes back as its own cell's centre.
    assert floor.nearest((1.03, 2.08), 0.5) == pytest.approx(centre(10, 20))
    # A point inside the cabinet is moved out to the floor in front of it -- two cells clear of its face.
    inside = centre(26, 20)
    spot = floor.nearest(inside, 0.5)
    assert spot is not None and floor.contains(spot) and math.dist(spot, inside) <= 0.5
    gx, gy = world.world_to_grid(*spot)
    assert (world.grid[gy - 1:gy + 2, gx - 1:gx + 2] == FREE).all()
    # ... and refused when the nearest floor lies beyond the radius, or the point is out in the unknown.
    assert floor.nearest(centre(27, 20), 0.2) is None
    assert floor.nearest(centre(2, 2), 0.2) is None and floor.nearest((-3.0, -3.0), 0.5) is None
    assert floor.nearest(centre(2, 20), 0.5) == pytest.approx(centre(6, 20)), "four cells in from the unseen edge"


def test_geodesic_is_the_walk_over_standing_floor_not_the_straight_line():
    data = room(h=50, w=50, x0=4, x1=46, y0=4, y1=46)
    data[4:36, 25] = OCCUPIED                         # a wall from the top, open for the last ten rows
    world = grid(data)
    here, beyond = centre(15, 10), centre(35, 10)     # two metres apart through the wall
    floor = standing_floor(world, here, BODY)
    walk = floor.geodesic_m()
    assert walk[10, 15] == pytest.approx(0.0, abs=1e-6) and math.isinf(walk[10, 25]) and math.isinf(walk[0, 0])
    gx, gy = world.world_to_grid(*beyond)
    assert floor.contains(beyond) and walk[gy, gx] > 2.5 * math.dist(here, beyond), "round the wall's end"
    assert walk[10, 20] == pytest.approx(0.5), "five cells along a row, measured from where the agent stands"
    assert floor.seeds[11, 16] and walk[11, 16] == pytest.approx(RES * math.hypot(1, 1)), "under the body: the straight step onto it"
    assert floor.geodesic_m() is walk, "computed once"
    xs, ys = floor.cells_xy()
    assert len(xs) == int(floor.reachable.sum()) == int(np.isfinite(walk).sum())
    assert np.allclose(walk[floor.reachable], walk[np.floor(ys / RES).astype(int), np.floor(xs / RES).astype(int)])


# -- one floor, and the standoff of the planner that is to stop on it --------------------------------------
def test_a_planner_standoff_leaves_out_the_standing_cells_nearer_an_obstacle_and_nothing_else():
    """The default planner cannot end a route within its standoff of an obstacle: those standing cells are not
    offered to it. They stay standing floor -- the same floor, asked without the standoff, has them."""
    data = room()
    data[18:23, 25:30] = OCCUPIED                     # a cabinet: columns 25-29, rows 18-22
    world = grid(data)
    here = centre(10, 20)
    whole, roomy = standing_floor(world, here, BODY), standing_floor(world, here, BODY, standoff_m=ROOMY)
    assert whole.standoff_m == BODY and roomy.standoff_m == ROOMY and roomy.clearance_m == BODY
    assert (roomy.standable == whole.standable).all() and (roomy.joined == whole.reachable).all()
    assert (roomy.reachable <= whole.reachable).all() and roomy.reachable.any()
    # In front of the cabinet's face (column 25): two cells off is the body's floor, three cells off the planner's.
    assert whole.contains(centre(23, 20)) and not roomy.contains(centre(23, 20))
    assert whole.contains(centre(22, 20)) and roomy.contains(centre(22, 20))
    # Exactly the cells whose centre lies within the standoff of an occupied cell's centre, the slow way.
    skirt = np.zeros(data.shape, dtype=bool)
    for gy, gx in np.argwhere(data == OCCUPIED):
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if math.hypot(dx, dy) * RES <= ROOMY + 1e-9:
                    skirt[gy + dy, gx + dx] = True
    assert (obstacle_skirt(world, ROOMY) == skirt).all()
    assert (roomy.reachable == (whole.reachable & ~skirt)).all()
    # A point inside the cabinet is moved to the floor each planner can stop at.
    assert whole.nearest(centre(26, 20), 0.5) == pytest.approx(centre(23, 20))
    assert roomy.nearest(centre(26, 20), 0.5) == pytest.approx(centre(22, 20))
    # A standoff that is not above the clearance leaves the floor whole; one that means nothing is refused.
    for standoff in (None, 0.0, BODY):
        assert (standing_floor(world, here, BODY, standoff_m=standoff).reachable == whole.reachable).all()
    for standoff in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            standing_floor(world, here, BODY, standoff_m=standoff)


def test_unknown_cells_cast_no_standoff_beyond_the_body_nothing_has_to_have_been_observed():
    """The standoff is from what was SEEN to be in the way. A corridor four cells wide through unseen floor has
    no cell with a 0.25 m disc observed free, and every cell of its middle is where the default planner may stop."""
    data = unknown_map(20, 60)
    data[8:12, 5:55] = FREE
    world = grid(data)
    here, end = centre(8, 9), centre(50, 10)
    strict = standing_floor(world, here, ROOMY)
    assert not strict.reachable.any() and not strict.standable.any(), "asked for the whole 0.25 m disc: nothing"
    roomy, whole = standing_floor(world, here, BODY, standoff_m=ROOMY), standing_floor(world, here, BODY)
    assert roomy.contains(end) and (roomy.reachable == whole.reachable).all()
    assert not obstacle_skirt(world, ROOMY).any()
    # ... while an obstacle on the same map does cast one, however much unseen floor lies round it.
    data[9, 30] = OCCUPIED
    walled = standing_floor(grid(data), here, BODY, standoff_m=ROOMY)
    assert walled.contains(centre(26, 9)) and not walled.contains(centre(28, 9)) and not walled.contains(end)


def test_the_walk_is_measured_over_the_whole_floor_and_the_far_room_is_the_planners_to_reach():
    """A three-cell doorway: the body fits, the default planner's standoff does not. The far room's open floor is
    on the roomy floor all the same -- a body-wide observed way leads there -- with the walk through the doorway
    measured; whether the default planner has a route is asked of the planner, and the squeeze planner has one."""
    world = grid(two_rooms(3))
    here, there, door = centre(10, 20), centre(30, 20), centre(20, 21)
    whole, roomy = standing_floor(world, here, BODY), standing_floor(world, here, BODY, standoff_m=ROOMY)
    assert whole.contains(door) and not roomy.contains(door) and roomy.joined[21, 20]
    assert whole.contains(there) and roomy.contains(there)
    assert roomy.geodesic_m()[20, 30] == pytest.approx(whole.geodesic_m()[20, 30]) and math.isfinite(roomy.geodesic_m()[21, 20])
    assert np.isfinite(roomy.geodesic_m()[roomy.reachable]).all()


# -- the properties, on maps nobody drew ---------------------------------------------------------------------
def random_world(rng, h=48, w=48):
    """Blobs of free floor in an unknown map, with obstacles and unseen cells sprinkled through them."""
    data = unknown_map(h, w)
    for _ in range(int(rng.integers(2, 6))):
        y0, x0 = int(rng.integers(1, h - 12)), int(rng.integers(1, w - 12))
        data[y0:y0 + int(rng.integers(6, 22)), x0:x0 + int(rng.integers(6, 22))] = FREE
    noise = rng.random((h, w))
    data[(noise < 0.03) & (data == FREE)] = OCCUPIED
    data[(noise > 0.97) & (data == FREE)] = UNKNOWN
    return data


def flooded(standable, seeds):
    """The cells joined to ``seeds`` through ``standable``, by plain 8-connected breadth-first search."""
    out = np.zeros_like(standable)
    stack = [tuple(c) for c in np.argwhere(seeds)]
    for cell in stack:
        out[cell] = True
    while stack:
        gy, gx = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < out.shape[0] and 0 <= nx < out.shape[1] and standable[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    stack.append((ny, nx))
    return out


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("clearance", [BODY, ROOMY])
def test_on_any_map_a_standing_point_is_observed_free_with_its_body_disc_observed_free_and_joined_to_the_agent(seed, clearance):
    rng = np.random.default_rng(1000 + seed)
    data = random_world(rng)
    world = grid(data)
    free_cells = np.argwhere(data == FREE)
    gy, gx = (int(v) for v in free_cells[int(rng.integers(len(free_cells)))])
    floor = standing_floor(world, centre(gx, gy), clearance)
    assert isinstance(floor, StandingFloor)
    h, w = data.shape
    disc = body_disc(clearance, RES)
    n = disc.shape[0] // 2
    # Rules 1 and 2, cell by cell, the slow way.
    expected = np.zeros((h, w), dtype=bool)
    for y in range(n, h - n):
        for x in range(n, w - n):
            expected[y, x] = bool((data[y - n:y + n + 1, x - n:x + n + 1][disc] == FREE).all())
    assert (floor.standable == expected).all()
    # Rule 3: exactly the cells a breadth-first search over standable cells finds from the cells under the body.
    assert (floor.seeds <= floor.standable).all()
    for y, x in np.argwhere(floor.seeds):
        assert math.hypot(x - gx, y - gy) <= max(1, int(round(clearance / RES)))
    assert (floor.reachable == flooded(floor.standable, floor.seeds)).all()
    # No reachable cell is unobserved, touches anything unobserved within the clearance, or is occupied.
    assert (data[floor.reachable] == FREE).all()
    # The walk to a reachable cell is finite and at least the straight line from the nearest cell under the body.
    walk = floor.geodesic_m()
    assert (np.isfinite(walk) == floor.reachable).all()
    # The same floor for a planner with a wider standoff: the cells an observed obstacle is not that near, no more.
    wider = standing_floor(world, centre(gx, gy), clearance, standoff_m=clearance + 0.1)
    near = np.zeros((h, w), dtype=bool)
    reach = int(math.floor((clearance + 0.1) / RES + 1e-9))
    for y, x in np.argwhere(data == OCCUPIED):
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if 0 <= y + dy < h and 0 <= x + dx < w and math.hypot(dx, dy) * RES <= clearance + 0.1 + 1e-9:
                    near[y + dy, x + dx] = True
    assert (wider.joined == floor.reachable).all() and (wider.reachable == (floor.reachable & ~near)).all()
    assert (np.isfinite(wider.geodesic_m()) == wider.joined).all()
    # Every answer of ``nearest`` is a reachable standing cell's centre within the radius asked.
    for _ in range(30):
        point = (float(rng.uniform(-0.5, w * RES + 0.5)), float(rng.uniform(-0.5, h * RES + 0.5)))
        within = float(rng.choice([0.0, 0.15, 0.5, 1.0]))
        spot = floor.nearest(point, within)
        if spot is not None:
            assert floor.contains(spot) and math.dist(spot, point) <= within + 1e-9
            sx, sy = world.world_to_grid(*spot)
            assert spot == pytest.approx(centre(sx, sy))


@pytest.mark.parametrize("seed", range(20))
def test_unobserved_space_counts_for_exactly_as_much_as_a_wall(seed):
    """Paint every UNKNOWN cell OCCUPIED and the standing floor does not change: it never leaned on one."""
    rng = np.random.default_rng(2000 + seed)
    data = random_world(rng)
    walled = np.where(data == UNKNOWN, OCCUPIED, data)
    free_cells = np.argwhere(data == FREE)
    gy, gx = (int(v) for v in free_cells[int(rng.integers(len(free_cells)))])
    for clearance in (BODY, ROOMY):
        seen, solid = standing_floor(grid(data), centre(gx, gy), clearance), standing_floor(grid(walled), centre(gx, gy), clearance)
        assert (seen.standable == solid.standable).all() and (seen.reachable == solid.reachable).all()
    # ... and observing more floor can only ever add standing floor, never take it away.
    more = np.where(data == UNKNOWN, FREE, data)
    assert (standable_cells(grid(data), BODY) <= standable_cells(grid(more), BODY)).all()
