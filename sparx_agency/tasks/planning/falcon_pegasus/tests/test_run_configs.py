"""Every launch file parses, and every run file describes a flyable area.

Both failures this guards against cost a container start to discover, and one of
them reports itself against the wrong thing entirely: a double hyphen anywhere in
an XML comment makes roslaunch say "Invalid roslaunch XML syntax" with a line
number, and nothing about comments.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

from sparx_agency.tasks.planning.falcon_pegasus.mapsize import expand_run, load_run

PACKAGE = Path(__file__).resolve().parent.parent
LAUNCH_FILES = sorted((PACKAGE / "adapter" / "launch").glob("*.launch"))
RUN_FILES = sorted((PACKAGE / "runs").glob("*.yaml"))
# The six numbered office runs, which share one exploration box. The warehouse
# runs are a different building and are deliberately not in this list.
OFFICE_RUN_FILES = sorted((PACKAGE / "runs").glob("[1-6]_*.yaml"))


def test_there_are_launch_files_and_run_files_to_check():
    """A glob that silently matches nothing would make everything below pass."""
    assert LAUNCH_FILES
    assert RUN_FILES
    assert len(OFFICE_RUN_FILES) == 6


@pytest.mark.parametrize("path", LAUNCH_FILES, ids=lambda p: p.name)
def test_launch_file_is_well_formed_xml(path):
    """XML forbids `--` inside a comment, and roslaunch will not tell you that."""
    try:
        ElementTree.parse(path)
    except ElementTree.ParseError as exc:
        pytest.fail(
            "{} is not well-formed XML: {}. If the line is inside a comment, "
            "look for a double hyphen.".format(path.name, exc)
        )


@pytest.mark.parametrize("path", RUN_FILES, ids=lambda p: p.name)
def test_run_file_expands_to_a_valid_map_config(path):
    """The area must derive three nested boxes FALCON will accept."""
    expanded = expand_run(load_run(path))
    map_size = expanded.config["map_config"]["map_size"]

    for prefix in ("map", "box", "vbox"):
        for axis in ("x", "y", "z"):
            low = map_size["{}_min_{}".format(prefix, axis)]
            high = map_size["{}_max_{}".format(prefix, axis)]
            assert low < high, "{} {} is inside out".format(prefix, axis)

    assert map_size["resolution"] > 0.0
    assert expanded.cost.voxels > 0


@pytest.mark.parametrize("path", RUN_FILES, ids=lambda p: p.name)
def test_run_file_does_not_quietly_allocate_a_fortune(path):
    """A guard on the office runs, not a general limit — see mapsize/README.md."""
    expanded = expand_run(load_run(path))
    assert expanded.cost.megabytes < 1024.0, (
        "{} would allocate {:.0f} MB of voxel grid".format(
            path.name, expanded.cost.megabytes
        )
    )


def test_the_six_office_runs_still_share_one_exploration_box():
    """They differ only in where the aircraft starts and how long it flies.

    Scoped to the six numbered office runs on purpose: the warehouse runs are a
    different building and must NOT share this box, so globbing every run file
    here would turn a correct warehouse config into a failure.
    """
    boxes = {
        tuple(
            expand_run(load_run(path)).config["map_config"]["map_size"][key]
            for key in sorted(
                k for k in ("box_min_x", "box_min_y", "box_max_x", "box_max_y")
            )
        )
        for path in OFFICE_RUN_FILES
    }
    assert len(boxes) == 1
