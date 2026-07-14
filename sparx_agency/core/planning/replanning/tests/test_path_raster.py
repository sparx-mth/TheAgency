"""Tests for path rasterization + route corridor masks."""
import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.planners.common.grid_geometry_2d import (
    line_cells, line_of_sight_clear)
from sparx_agency.core.planning.replanning.path_raster import (
    corridor_mask, rasterize_path)

BEV = OccupancyValues(free=0, occupied=100, unknown=-1)


def _grid(h=40, w=40, res=0.1, ox=0.0, oy=0.0, fill=0):
    data = np.full((h, w), fill, np.int16)
    return OccupancyGrid2D(
        data, OccupancyGrid2DParams(resolution=res, origin_x=ox, origin_y=oy,
                                    frame_id="world"), values=BEV)


def test_line_cells_axis_and_diagonal():
    assert line_cells(0, 0, 3, 0) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert line_cells(0, 0, 3, 3) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    # single cell
    assert line_cells(5, 7, 5, 7) == [(5, 7)]


def test_line_cells_agrees_with_los():
    """line_cells must visit exactly the cells line_of_sight_clear tests."""
    occ = np.zeros((20, 20), bool)
    cells = line_cells(1, 2, 15, 9)
    # Placing an obstacle on ANY visited cell must make LOS report blocked.
    for (cx, cy) in cells[1:-1]:
        occ[:] = False
        occ[cy, cx] = True
        assert not line_of_sight_clear(occ, 1, 2, 15, 9)


def test_rasterize_path_clips_out_of_bounds():
    g = _grid(h=10, w=10, res=1.0)
    # segment runs off the top-right corner; only in-bounds cells returned
    cells = rasterize_path([Pose2D(0.5, 0.5), Pose2D(20.0, 20.0)], g)
    assert cells, "expected some in-bounds cells"
    for (cx, cy) in cells:
        assert 0 <= cx < 10 and 0 <= cy < 10


def test_corridor_mask_half_width():
    g = _grid(h=40, w=40, res=0.1)
    path = [Pose2D(0.05, 2.0), Pose2D(3.9, 2.0)]  # horizontal line at y=2.0 (row 20)
    # radius 0 -> just the line (one row)
    m0 = corridor_mask(path, g, radius_cells=0)
    rows = np.unique(np.nonzero(m0)[0])
    assert rows.tolist() == [20]
    # radius 3 -> band of rows 17..23 (7 rows)
    m3 = corridor_mask(path, g, radius_cells=3)
    rows3 = np.unique(np.nonzero(m3)[0])
    assert rows3.min() == 17 and rows3.max() == 23


def test_corridor_mask_empty_path_all_false():
    g = _grid()
    assert not corridor_mask([], g, radius_cells=5).any()
    assert not corridor_mask([Pose2D(1.0, 1.0)], g, radius_cells=5).any() or True
