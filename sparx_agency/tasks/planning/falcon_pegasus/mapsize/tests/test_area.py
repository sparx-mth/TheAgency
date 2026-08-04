"""Deriving the three boxes, and reproducing the numbers we already fly."""
from __future__ import annotations

from dataclasses import replace

import pytest

from sparx_agency.tasks.planning.falcon_pegasus.mapsize.area import Box, ExplorationArea

# The whole-office area, as the six run files describe it today.
OFFICE = ExplorationArea(
    building=(-23.0, -33.2, 5.1, 38.7),
    flight_band=(1.0, 2.2),
    vertical_extent=(-0.2, 2.4),
    resolution=0.10,
    margin=2.0,
)
OFFICE_CRUISE_M = 1.4


def test_box_is_the_footprint_over_the_flight_band():
    box = OFFICE.box
    assert (box.min_x, box.min_y, box.min_z) == pytest.approx((-23.0, -33.2, 1.0))
    assert (box.max_x, box.max_y, box.max_z) == pytest.approx((5.1, 38.7, 2.2))


def test_map_matches_the_hand_written_run_files():
    """The migration must not move a single wall."""
    grid = OFFICE.map
    assert (grid.min_x, grid.min_y, grid.min_z) == pytest.approx((-25.0, -35.2, -0.2))
    assert (grid.max_x, grid.max_y, grid.max_z) == pytest.approx((7.1, 40.7, 2.4))


def test_vbox_matches_the_hand_written_run_files():
    """With a slab height, the drawn box is a thin cut at cruise altitude."""
    vbox = replace(OFFICE, visualisation_slab_at=OFFICE_CRUISE_M).vbox
    assert (vbox.min_x, vbox.min_y, vbox.min_z) == pytest.approx((-23.5, -33.7, 1.3))
    assert (vbox.max_x, vbox.max_y, vbox.max_z) == pytest.approx((5.6, 39.2, 1.5))


def test_vbox_defaults_to_the_whole_allocated_grid():
    """What most falcon/ environments already did by hand."""
    assert OFFICE.vbox == OFFICE.map


def test_an_explicit_visualisation_box_wins():
    area = replace(OFFICE, visualisation=(-8.0, -8.0, -2.0, 8.0, 8.0, 1.7))
    assert area.vbox == Box(-8.0, -8.0, -2.0, 8.0, 8.0, 1.7)


def test_an_explicit_box_beats_a_slab_height():
    area = replace(
        OFFICE,
        visualisation=(-8.0, -8.0, -2.0, 8.0, 8.0, 1.7),
        visualisation_slab_at=1.4,
    )
    assert area.vbox.min_z == pytest.approx(-2.0)


def test_a_scalar_margin_is_symmetric():
    assert OFFICE.margins == (2.0, 2.0)


def test_an_asymmetric_margin_is_kept_per_side():
    """small_house.yaml really is -10.5..10.0 around a -8..8 box."""
    area = replace(OFFICE, building=(-8.0, -8.0, 8.0, 8.0), margin=(2.5, 2.0))
    assert area.margins == (2.5, 2.0)
    assert area.map.min_x == pytest.approx(-10.5)
    assert area.map.max_x == pytest.approx(10.0)


def test_from_dict_reads_an_asymmetric_margin():
    area = ExplorationArea.from_dict(
        {
            "building": [-8.0, -8.0, 8.0, 8.0],
            "flight_band": [0.0, 2.2],
            "vertical_extent": [-1.0, 3.5],
            "resolution": 0.1,
            "margin": [2.5, 2.0],
        }
    )
    assert area.margins == (2.5, 2.0)


def test_from_dict_rejects_a_three_sided_margin():
    with pytest.raises(ValueError, match="one number, or two"):
        ExplorationArea.from_dict(
            {
                "building": [0.0, 0.0, 10.0, 10.0],
                "flight_band": [1.0, 2.0],
                "vertical_extent": [0.0, 3.0],
                "resolution": 0.2,
                "margin": [1.0, 2.0, 3.0],
            }
        )


def test_office_exploration_volume():
    """2424 m3 — the number the run files' own comment quotes."""
    assert OFFICE.box.volume == pytest.approx(2424.0, rel=1e-3)


def test_grid_shape_rounds_up_like_falcon():
    """FALCON uses ceil, so a partial voxel still costs a whole one."""
    box = Box(0.0, 0.0, 0.0, 1.05, 2.0, 0.3)
    assert box.grid_shape(0.1) == (11, 20, 3)


def test_grid_shape_rejects_a_useless_resolution():
    with pytest.raises(ValueError, match="resolution must be positive"):
        Box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).grid_shape(0.0)


def test_contains_allows_touching_faces():
    outer = Box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    assert outer.contains(Box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0))
    assert not outer.contains(Box(-0.1, 0.0, 0.0, 10.0, 10.0, 10.0))


def test_to_falcon_emits_the_keys_the_planner_reads():
    keys = set(Box(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).to_falcon("box"))
    assert keys == {
        "box_min_x", "box_min_y", "box_min_z",
        "box_max_x", "box_max_y", "box_max_z",
    }


def test_from_dict_reads_an_area_block():
    area = ExplorationArea.from_dict(
        {
            "building": [-23.0, -33.2, 5.1, 38.7],
            "flight_band": [1.0, 2.2],
            "vertical_extent": [-0.2, 2.4],
            "resolution": 0.10,
            "margin": 2.0,
        }
    )
    assert area == OFFICE


def test_from_dict_defaults_the_margin():
    area = ExplorationArea.from_dict(
        {
            "building": [0.0, 0.0, 10.0, 10.0],
            "flight_band": [1.0, 2.0],
            "vertical_extent": [0.0, 3.0],
            "resolution": 0.2,
        }
    )
    assert area.margin == 2.0


def test_from_dict_names_the_missing_key():
    with pytest.raises(ValueError, match="flight_band"):
        ExplorationArea.from_dict(
            {"building": [0, 0, 1, 1], "vertical_extent": [0, 1], "resolution": 0.1}
        )


def test_from_dict_rejects_a_footprint_of_the_wrong_length():
    with pytest.raises(ValueError, match="must be 4 numbers"):
        ExplorationArea.from_dict(
            {
                "building": [0.0, 0.0, 10.0],
                "flight_band": [1.0, 2.0],
                "vertical_extent": [0.0, 3.0],
                "resolution": 0.2,
            }
        )
