"""``closing/wall_normal.py``: the direction from a mounted target's memory into the room, read from the map."""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.object_search.closing.wall_normal import MIN_CELLS, wall_normal


def room(occupied=(), unknown_where=None):
    """A 10 x 10 m map of free floor with ``occupied`` cells (world x, y) and the half-plane ``unknown_where`` unknown."""
    world = OccupancyGrid2D(np.zeros((100, 100), np.int8), OccupancyGrid2DParams(.1, -5, -5))
    if unknown_where is not None:
        for gy in range(100):
            for gx in range(100):
                if unknown_where(*world.grid_to_world(gx, gy)):
                    world.grid[gy, gx] = world.values.unknown
    for x, y in occupied:
        gx, gy = world.world_to_grid(x, y)
        world.grid[gy, gx] = world.values.occupied
    return world


def wall_along_x(y=1.0, x_from=-3.0, x_to=3.0):
    """A wall at ``y`` with the outside beyond it unknown, as a mapped wall reads."""
    return room([(x, y) for x in np.arange(x_from, x_to + 1e-9, 0.1)], unknown_where=lambda x, yy: yy > y)


def test_a_picture_on_a_wall_faces_into_the_room():
    normal = wall_normal(wall_along_x(), (1.0, 0.95))
    assert normal is not None and normal[0] == pytest.approx(0.0, abs=0.05) and normal[1] == pytest.approx(-1.0, abs=0.01)
    assert math.hypot(*normal) == pytest.approx(1.0)


def test_a_picture_above_a_sideboard_still_faces_into_the_room():
    # A 0.4 m deep sideboard under the picture: a blob beside the wall that defeats a line fit.
    world = wall_along_x()
    for x in np.arange(0.5, 1.5, 0.1):
        for y in np.arange(0.6, 0.95, 0.1):
            gx, gy = world.world_to_grid(x, y)
            world.grid[gy, gx] = world.values.occupied
    normal = wall_normal(world, (1.0, 0.95))
    assert normal is not None and normal[1] < -0.95


def test_a_diagonal_wall_and_a_corner_give_their_own_directions():
    diagonal = room([(x, x) for x in np.arange(-3.0, 3.0, 0.07)], unknown_where=lambda x, y: y > x)
    normal = wall_normal(diagonal, (1.0, 1.0))
    assert normal[0] == pytest.approx(math.sqrt(0.5), abs=0.1) and normal[1] == pytest.approx(-math.sqrt(0.5), abs=0.1)
    corner = room([(x, 1.0) for x in np.arange(-3.0, 1.01, 0.1)] + [(1.0, y) for y in np.arange(-3.0, 1.0, 0.1)],
                  unknown_where=lambda x, y: y > 1.0 or x > 1.0)
    normal = wall_normal(corner, (0.9, 0.9))
    assert normal[0] < -0.5 and normal[1] < -0.5, "out of the corner along its bisector"


def test_nothing_to_hang_on_too_little_floor_or_floor_all_round_gives_no_direction():
    assert wall_normal(room(), (0.0, 0.0)) is None, "free floor all round: a plant on a stand mid-room"
    assert wall_normal(wall_along_x(), (1.0, 0.5)) is None, "0.45 m off the wall is not mounted on it"
    pillar = room([(0.0, 0.0)])
    assert wall_normal(pillar, (0.1, 0.0)) is None, "a single cell with floor all round it"
    closet = room([(x, y) for x in np.arange(-1.0, 1.01, 0.1) for y in np.arange(-1.0, 1.01, 0.1)
                   if abs(x) > 0.85 or abs(y) > 0.85], unknown_where=lambda x, y: abs(x) > 1.0 or abs(y) > 1.0)
    # Inside a 1.7 m closet the free cells surround the memory nearly evenly, but a memory against one
    # wall still faces the floor; one with fewer than MIN_CELLS free cells around it has no side at all.
    assert wall_normal(closet, (0.0, 0.8)) is not None
    cupboard = room([(x, y) for x in np.arange(-0.3, 0.31, 0.1) for y in np.arange(-0.3, 0.31, 0.1)
                     if abs(x) > 0.15 or abs(y) > 0.15], unknown_where=lambda x, y: abs(x) > 0.3 or abs(y) > 0.3)
    assert wall_normal(cupboard, (0.0, 0.1)) is None
    assert MIN_CELLS >= 4


def test_only_the_cells_within_the_radius_count():
    # A corridor 1.0 m wide: the picture's wall at y = 0.45 and the facing wall at y = -0.55, the memory at
    # the picture. Within 0.8 m the floor is on the memory's -y side: the direction is into the corridor.
    world = room([(x, 0.45) for x in np.arange(-3.0, 3.0, 0.1)] + [(x, -0.55) for x in np.arange(-3.0, 3.0, 0.1)],
                 unknown_where=lambda x, y: y > 0.45 or y < -0.55)
    normal = wall_normal(world, (1.0, 0.4), radius_m=0.8)
    assert normal is not None and normal[1] == pytest.approx(-1.0, abs=0.05)
    assert wall_normal(world, (1.0, 0.4), radius_m=0.1) is None, "too few cells to read"
