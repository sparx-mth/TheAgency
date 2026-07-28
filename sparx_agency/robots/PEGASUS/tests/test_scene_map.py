"""Map storage: a map must be findable by the altitude it was surveyed at.

Silently loading a map made at a different altitude would plan routes through
whatever is at head height while flying at knee height, so the altitude is part
of the identity rather than a footnote in the metadata.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment import occupancy_from_mask
from sparx_agency.robots.PEGASUS.adapters import scene_map


def _grid():
    occupied = np.zeros((8, 10), dtype=bool)
    occupied[0, :] = True
    return occupancy_from_mask(occupied, 0.25, -1.0, -2.0, frame_id="world")


def test_altitude_is_part_of_the_filename():
    a = scene_map.map_path("office", 1.5)
    b = scene_map.map_path("office", 2.0)
    assert a != b
    assert "150cm" in a.name and "200cm" in b.name


def test_near_identical_altitudes_map_to_one_file():
    assert scene_map.map_path("office", 1.5) == scene_map.map_path("office", 1.500001)


def test_round_trip_through_the_canonical_location(tmp_path):
    grid = _grid()
    scene_map.save_scene_map("office", 1.5, grid, {"resolution_m": 0.25}, tmp_path)

    loaded, metadata, layers = scene_map.load_scene_map("office", 1.5, tmp_path)

    np.testing.assert_array_equal(loaded.grid, grid.grid)
    assert loaded.frame_id == "world"
    assert metadata["scene"] == "office"
    assert metadata["altitude_m"] == pytest.approx(1.5)
    assert metadata["resolution_m"] == 0.25


def test_a_missing_map_names_the_command_that_makes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="survey_scene.py"):
        scene_map.load_scene_map("warehouse", 1.5, tmp_path)


def test_a_map_at_another_altitude_is_not_silently_substituted(tmp_path):
    scene_map.save_scene_map("office", 1.5, _grid(), {}, tmp_path)
    with pytest.raises(FileNotFoundError):
        scene_map.load_scene_map("office", 2.0, tmp_path)


def test_available_maps_lists_what_is_on_disk(tmp_path):
    scene_map.save_scene_map("office", 1.5, _grid(), {}, tmp_path)
    scene_map.save_scene_map("simple_room", 2.25, _grid(), {}, tmp_path)

    found = {(scene, altitude) for scene, altitude, _ in scene_map.available_maps(tmp_path)}
    assert found == {("office", 1.5), ("simple_room", 2.25)}


def test_available_maps_is_empty_when_there_is_no_directory(tmp_path):
    assert scene_map.available_maps(tmp_path / "nothing") == []


def test_the_landable_layer_round_trips(tmp_path):
    """Goals are drawn from this layer; losing it would put one on a desk."""
    grid = _grid()
    landable = np.zeros(grid.grid.shape, dtype=bool)
    landable[3:6, 2:5] = True

    scene_map.save_scene_map("office", 1.5, grid, {}, tmp_path,
                             layers={scene_map.LANDABLE_LAYER: landable})
    _loaded, _metadata, layers = scene_map.load_scene_map("office", 1.5, tmp_path)

    np.testing.assert_array_equal(layers[scene_map.LANDABLE_LAYER], landable)


def test_a_map_without_the_landable_layer_still_loads(tmp_path):
    """Maps surveyed before landability was measured must not become unreadable."""
    scene_map.save_scene_map("office", 1.5, _grid(), {}, tmp_path)
    _loaded, _metadata, layers = scene_map.load_scene_map("office", 1.5, tmp_path)
    assert scene_map.LANDABLE_LAYER not in layers
