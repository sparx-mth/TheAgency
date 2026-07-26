"""Tests for cross-frame occupancy change detection (discovery trigger)."""
import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.replanning.map_change import (
    count_new_known_in_corridor, known_mask, newly_known_mask)
from sparx_agency.core.planning.replanning.path_raster import corridor_mask

BEV = OccupancyValues(free=0, occupied=100, unknown=-1)


def _grid(data):
    h, w = data.shape
    return OccupancyGrid2D(
        data.astype(np.int16),
        OccupancyGrid2DParams(resolution=0.1, origin_x=0.0, origin_y=0.0,
                              frame_id="world"),
        values=BEV)


def test_known_mask_free_and_occupied_are_known():
    data = np.array([[0, 100], [-1, -1]], np.int16)
    km = known_mask(_grid(data))
    assert km.tolist() == [[True, True], [False, False]]


def test_newly_known_only_unknown_to_observed():
    prev = np.array([[True, False], [False, False]])
    now = _grid(np.array([[0, 0], [-1, 100]], np.int16))  # (0,0) already known; (0,1),(1,1) new
    nk = newly_known_mask(prev, now)
    # (0,0) was known before -> not new; (0,1) unknown->free -> new; (1,1) unknown->occ -> new
    assert nk.tolist() == [[False, True], [False, True]]


def test_count_new_known_in_corridor_route_relevance():
    """Only newly-known cells INSIDE the route corridor are counted."""
    h = w = 30
    prev = np.zeros((h, w), bool)               # nothing known at commit
    data = np.full((h, w), -1, np.int16)
    data[:] = 0                                 # now everything observed (free)
    g = _grid(data)
    # A horizontal route through row 15; corridor radius 2 -> rows 13..17.
    path = [Pose2D(0.05, 1.5), Pose2D(2.9, 1.5)]
    corridor = corridor_mask(path, g, radius_cells=2)
    n = count_new_known_in_corridor(prev, g, corridor)
    assert n == int(corridor.sum())             # all corridor cells are newly known
    # An off-corridor reveal is not counted: only reveal rows 0..2 (outside corridor).
    prev2 = np.zeros((h, w), bool)
    data2 = np.full((h, w), -1, np.int16)
    data2[0:3, :] = 0                           # only far rows observed
    n2 = count_new_known_in_corridor(prev2, _grid(data2), corridor)
    assert n2 == 0


def test_count_new_known_shape_mismatch_raises():
    g = _grid(np.zeros((5, 5), np.int16))
    with pytest.raises(ValueError):
        count_new_known_in_corridor(np.zeros((4, 4), bool), g, np.zeros((5, 5), bool))
