"""Expanding a run file, and failing on the host instead of inside the container."""
from __future__ import annotations

import copy

import pytest
import yaml

from sparx_agency.tasks.planning.falcon_pegasus.mapsize.expand import (
    expand_run,
    load_run,
    write_expanded,
)

RUN = {
    "run": {
        "name": "whole_office",
        "scene": "office",
        "cruise_altitude_m": 1.4,
    },
    "map_config": {
        "map_file": "",
        "map_dimension": 2,
        "init_x": -8.93,
        "area": {
            "building": [-23.0, -27.2, 5.1, 38.7],
            "flight_band": [1.0, 2.2],
            "vertical_extent": [-0.2, 2.4],
            "resolution": 0.10,
            "margin": [2.0, 8.0, 2.0, 2.0],
        },
        "scale": 1.0,
    },
}


def a_run(**area_overrides) -> dict:
    """A copy of RUN with its area block adjusted."""
    config = copy.deepcopy(RUN)
    config["map_config"]["area"].update(area_overrides)
    return config


def test_expansion_emits_every_key_falcon_reads():
    map_size = expand_run(a_run()).config["map_config"]["map_size"]
    expected = {
        "{}_{}_{}".format(prefix, bound, axis)
        for prefix in ("map", "box", "vbox")
        for bound in ("min", "max")
        for axis in ("x", "y", "z")
    }
    assert expected <= set(map_size)
    assert map_size["resolution"] == 0.10


def test_expansion_reproduces_the_current_office_numbers():
    map_size = expand_run(a_run()).config["map_config"]["map_size"]
    assert map_size["map_min_x"] == pytest.approx(-25.0)
    assert map_size["map_max_y"] == pytest.approx(40.7)
    assert map_size["box_min_z"] == pytest.approx(1.0)
    assert map_size["vbox_max_z"] == pytest.approx(1.5)


def test_the_area_block_is_replaced_and_the_rest_survives():
    map_config = expand_run(a_run()).config["map_config"]
    assert "area" not in map_config
    assert map_config["init_x"] == -8.93
    assert map_config["map_dimension"] == 2


def test_cost_is_reported_alongside():
    expanded = expand_run(a_run())
    assert expanded.cost.shape == (321, 759, 26)
    assert expanded.cost.megabytes > 200.0


def test_a_legacy_run_file_says_what_to_do():
    config = copy.deepcopy(RUN)
    config["map_config"].pop("area")
    config["map_config"]["map_size"] = {"map_min_x": -25.0}
    with pytest.raises(ValueError, match="long-hand"):
        expand_run(config)


def test_missing_cruise_altitude_is_caught():
    config = a_run()
    config["run"].pop("cruise_altitude_m")
    with pytest.raises(ValueError, match="cruise_altitude_m"):
        expand_run(config)


def test_an_inside_out_footprint_is_caught():
    with pytest.raises(ValueError, match="inside out on x"):
        expand_run(a_run(building=[5.1, -33.2, -23.0, 38.7]))


def test_a_margin_smaller_than_the_drawn_box_is_caught():
    with pytest.raises(ValueError, match="below the 0.5 m"):
        expand_run(a_run(margin=0.2))


def test_a_vertical_extent_that_misses_the_drawn_slab_is_caught():
    """Cruise is 1.4 m; allocating only up to 1.2 m would draw outside the grid."""
    with pytest.raises(ValueError, match="does not contain the drawn box"):
        expand_run(a_run(vertical_extent=[-0.2, 1.2], flight_band=[0.5, 1.1]))


def test_a_useless_resolution_is_caught():
    with pytest.raises(ValueError, match="resolution must be positive"):
        expand_run(a_run(resolution=0.0))


def test_an_explicit_resolution_differing_from_the_volume_rule_is_flagged():
    """The office box is 2222 m3, so FALCON's own rule would pick 10 cm."""
    expanded = expand_run(a_run(resolution=0.2))
    assert any("volume rule" in note for note in expanded.warnings)
    assert not expand_run(a_run(resolution=0.1)).warnings


def test_the_resolution_note_says_which_way_the_memory_goes():
    """Coarser than the rule saves; finer than it costs. Both stated as a factor."""
    coarser = expand_run(a_run(resolution=0.2)).warnings[0]
    assert "8x less memory" in coarser

    # A box over 4000 m3 makes the rule pick 20 cm, so 10 cm is now the finer choice.
    finer = expand_run(
        a_run(building=[-40.0, -40.0, 40.0, 40.0], resolution=0.1)
    ).warnings[0]
    assert "8x more memory" in finer


def test_a_grid_over_two_gigabytes_is_flagged():
    expanded = expand_run(
        a_run(building=[-250.0, -250.0, 250.0, 250.0], vertical_extent=[-1.0, 20.0])
    )
    assert any("GB" in note for note in expanded.warnings)


def test_round_trip_through_a_file(tmp_path):
    source = tmp_path / "run.yaml"
    source.write_text(yaml.safe_dump(RUN), encoding="utf-8")

    expanded = expand_run(load_run(source))
    out = write_expanded(expanded, tmp_path / "nested" / "expanded.yaml")

    reloaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert reloaded["map_config"]["map_size"]["map_min_x"] == pytest.approx(-25.0)
    assert reloaded["run"]["cruise_altitude_m"] == 1.4
    assert "do not edit" in out.read_text(encoding="utf-8")


def test_load_run_demands_both_blocks(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"run": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="map_config"):
        load_run(path)
