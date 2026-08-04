"""Every environment describes a mappable area, and none of them is empty.

These fly the real aircraft, so a map file that only fails once the container is
up is a wasted flight preparation. Everything here runs on the host in
milliseconds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sparx_agency.tasks.planning.falcon_pegasus.mapsize import expand_map, load_map

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"
MAP_FILES = sorted(MAPS_DIR.glob("*.yaml"))


def test_there_are_map_files_to_check():
    """A glob matching nothing would make everything below pass."""
    assert MAP_FILES


@pytest.mark.parametrize("path", MAP_FILES, ids=lambda p: p.name)
def test_map_file_is_not_empty(path):
    """`small_warehouse.yaml` sat here at zero bytes from the commit that added it."""
    assert path.stat().st_size > 0, "{} is an empty file".format(path.name)


@pytest.mark.parametrize("path", MAP_FILES, ids=lambda p: p.name)
def test_map_file_expands_to_a_valid_map_config(path):
    """The area must derive three nested boxes FALCON will accept."""
    expanded = expand_map(load_map(path))
    map_size = expanded.config["map_config"]["map_size"]

    for prefix in ("map", "box", "vbox"):
        for axis in ("x", "y", "z"):
            low = map_size["{}_min_{}".format(prefix, axis)]
            high = map_size["{}_max_{}".format(prefix, axis)]
            assert low < high, "{} {} is inside out".format(prefix, axis)

    assert map_size["resolution"] > 0.0
    assert expanded.cost.voxels > 0


@pytest.mark.parametrize("path", MAP_FILES, ids=lambda p: p.name)
def test_every_environment_states_its_resolution(path):
    """The whole point: no environment inherits a grid from its box volume."""
    area = load_map(path)["map_config"]["area"]
    assert "resolution" in area


@pytest.mark.parametrize("path", MAP_FILES, ids=lambda p: p.name)
def test_map_file_does_not_quietly_allocate_a_fortune(path):
    """A guard on today's environments, not a general limit."""
    expanded = expand_map(load_map(path))
    assert expanded.cost.megabytes < 1024.0, (
        "{} would allocate {:.0f} MB of voxel grid".format(
            path.name, expanded.cost.megabytes
        )
    )
