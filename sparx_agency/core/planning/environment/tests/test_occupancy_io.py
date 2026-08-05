"""Round-tripping an OccupancyGrid2D through .npz must preserve the whole map."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues,
    load_occupancy_grid, occupancy_from_mask, save_occupancy_grid,
)


def _grid() -> OccupancyGrid2D:
    cells = np.array([[0, 0, 1], [0, -1, 1], [0, 0, 0]], dtype=np.int16)
    params = OccupancyGrid2DParams(resolution=0.25, origin_x=-1.5, origin_y=2.0,
                                   frame_id="office")
    return OccupancyGrid2D(cells, params)


def test_round_trip_preserves_cells_and_metadata(tmp_path):
    grid = _grid()
    path = save_occupancy_grid(tmp_path / "office.npz", grid,
                               metadata={"scene": "office", "altitude_m": 1.5})

    loaded, metadata, layers = load_occupancy_grid(path)

    np.testing.assert_array_equal(loaded.grid, grid.grid)
    assert loaded.resolution == pytest.approx(0.25)
    assert loaded.origin_x == pytest.approx(-1.5)
    assert loaded.origin_y == pytest.approx(2.0)
    assert loaded.frame_id == "office"
    assert metadata == {"scene": "office", "altitude_m": 1.5}
    assert layers == {}


def test_round_trip_preserves_world_coordinates(tmp_path):
    """A grid without its origin is not a map -- transforms must survive."""
    grid = _grid()
    path = save_occupancy_grid(tmp_path / "g.npz", grid)
    loaded, _meta, _layers = load_occupancy_grid(path)

    for gx in range(grid.width):
        for gy in range(grid.height):
            assert loaded.grid_to_world(gx, gy) == grid.grid_to_world(gx, gy)


def test_round_trip_preserves_non_default_values(tmp_path):
    values = OccupancyValues(free=10, occupied=20, unknown=30)
    cells = np.array([[10, 20], [30, 10]], dtype=np.int16)
    grid = OccupancyGrid2D(cells, OccupancyGrid2DParams(0.5, 0.0, 0.0), values=values)

    loaded, _meta, _layers = load_occupancy_grid(save_occupancy_grid(tmp_path / "v.npz", grid))

    assert loaded.values == values
    assert loaded.is_occupied(1, 0)
    assert loaded.is_unknown(0, 1)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_occupancy_grid(tmp_path / "nope.npz")


def test_occupancy_from_mask_marks_unsurveyed_cells_unknown():
    occupied = np.array([[False, True], [False, False]])
    known = np.array([[True, True], [True, False]])

    grid = occupancy_from_mask(occupied, 0.5, -1.0, -1.0, known=known)

    assert grid.is_free(0, 0)
    assert grid.is_occupied(1, 0)
    assert grid.is_unknown(1, 1)


def test_occupancy_from_mask_defaults_everything_known():
    grid = occupancy_from_mask(np.zeros((2, 2), dtype=bool), 0.1, 0.0, 0.0)
    assert not (grid.grid == grid.values.unknown).any()


def test_occupancy_from_mask_rejects_mismatched_known():
    with pytest.raises(ValueError):
        occupancy_from_mask(np.zeros((2, 2), dtype=bool), 0.1, 0.0, 0.0,
                            known=np.zeros((3, 3), dtype=bool))


def test_layers_travel_with_the_grid(tmp_path):
    """A layer that can be separated from its grid gets paired with the wrong one."""
    grid = _grid()
    landable = np.array([[True, False, False], [True, True, False], [False, False, True]])

    path = save_occupancy_grid(tmp_path / "g.npz", grid, layers={"landable": landable})
    _loaded, _metadata, layers = load_occupancy_grid(path)

    assert set(layers) == {"landable"}
    np.testing.assert_array_equal(layers["landable"], landable)
    assert layers["landable"].dtype == bool


def test_a_mismatched_layer_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="landable"):
        save_occupancy_grid(tmp_path / "g.npz", _grid(),
                            layers={"landable": np.zeros((9, 9), dtype=bool)})
