"""Turn a run file's ``area`` block into the ``map_size`` block FALCON reads.

The expansion happens on the host, before the container starts, for two reasons.
It fails in a hundred milliseconds with a sentence rather than after a docker
pull and a roslaunch with a glog ``CHECK`` and a stack trace; and it can print
what the run is about to cost while there is still time to change it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple

import yaml

from sparx_agency.tasks.planning.falcon_pegasus.mapsize.area import (
    VBOX_MARGIN_M,
    Box,
    ExplorationArea,
)
from sparx_agency.tasks.planning.falcon_pegasus.mapsize.memory import (
    GridCost,
    grid_cost,
    implicit_resolution,
)

# Warn above this. Not a limit — a Jetson Orin NX has 16 GB shared with the GPU,
# and the rest of the stack wants a good deal of it.
LARGE_GRID_WARNING_BYTES = 2.0 * 1024 ** 3


@dataclass(frozen=True)
class ExpandedRun:
    """A run file with its ``map_size`` filled in, and what it will cost.

    Attributes:
        config: The whole run config, ready to write out for ``rosparam load``.
        area: The area it was derived from.
        cost: What allocating the ``map`` box will take.
        warnings: Things worth saying that are not errors.
    """

    config: dict
    area: ExplorationArea
    cost: GridCost
    warnings: Tuple[str, ...]


def _load(path: Path, required: tuple) -> dict:
    """Read a config file and insist on the blocks it must carry.

    Args:
        path: The file.
        required: Top-level keys that must be present.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If it is empty, not a mapping, or missing a block.
    """
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("{}: the file is empty".format(path))

    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("{}: expected a YAML mapping".format(path))

    for key in required:
        if key not in config:
            raise ValueError("{}: missing the `{}` block".format(path, key))
    return config


def load_run(path: Path) -> dict:
    """Read a ``falcon_pegasus/runs/*.yaml``.

    One file describes both the aircraft and FALCON's map, on purpose, so the
    two cannot disagree about where the aircraft starts.

    Args:
        path: Path to the run file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If ``run`` or ``map_config`` is missing.
    """
    return _load(path, ("run", "map_config"))


def load_map(path: Path) -> dict:
    """Read a ``falcon/maps/*.yaml``.

    These carry only ``map_config`` — there is no aircraft block, because the
    aircraft is a real one and nothing here spawns it.

    Args:
        path: Path to the map file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If ``map_config`` is missing or the file is empty.
    """
    return _load(path, ("map_config",))


def _check_boxes(area: ExplorationArea, vbox: Box) -> None:
    """Fail on a geometry FALCON would abort on, with a message that says why.

    Mirrors the ``CHECK_LT``/``CHECK_LE`` block in ``map_server.cpp``, which
    aborts the node with a stack trace and no explanation.

    Args:
        area: The parsed area.
        vbox: The derived visualisation box.

    Raises:
        ValueError: If any box is inside out or containment is broken.
    """
    if area.resolution <= 0.0:
        raise ValueError(
            "map_config.area.resolution must be positive, got {}".format(area.resolution)
        )
    if area.visualisation_slab_at is not None and min(area.margins) < VBOX_MARGIN_M:
        # Only the slab rule pushes the drawn box outside the exploration box.
        raise ValueError(
            "map_config.area.margin is {} m, below the {} m the drawn slab needs. "
            "The allocated grid must reach at least as far as what is drawn."
            .format(area.margin, VBOX_MARGIN_M)
        )

    for name, box in (("box", area.box), ("map", area.map), ("vbox", vbox)):
        for axis, low, high in (
            ("x", box.min_x, box.max_x),
            ("y", box.min_y, box.max_y),
            ("z", box.min_z, box.max_z),
        ):
            if low >= high:
                raise ValueError(
                    "{} is inside out on {}: min {} is not below max {}".format(
                        name, axis, low, high
                    )
                )

    if not area.map.contains(area.box):
        raise ValueError(
            "the allocated grid does not contain the exploration box — check "
            "margin ({} m) and vertical_extent {}".format(
                area.margin, area.vertical_extent
            )
        )
    if not area.map.contains(vbox):
        raise ValueError(
            "the allocated grid does not contain the drawn box. vertical_extent "
            "{} must span the {:.2f}..{:.2f} m slab at cruise height.".format(
                area.vertical_extent, vbox.min_z, vbox.max_z
            )
        )


def _warnings(area: ExplorationArea, cost: GridCost) -> List[str]:
    """Things worth saying out loud that are not reasons to stop."""
    notes: List[str] = []

    would_be = implicit_resolution(area.box.volume)
    if abs(would_be - area.resolution) > 1e-9:
        ratio = (would_be / area.resolution) ** 3
        direction = (
            "{:.0f}x less memory".format(1.0 / ratio)
            if ratio < 1.0
            else "{:.0f}x more memory".format(ratio)
        )
        notes.append(
            "resolution is set to {:.2f} m; FALCON's own volume rule would have "
            "picked {:.2f} m for a {:.0f} m3 exploration box. The explicit value "
            "wins — that is the point — and it costs {}.".format(
                area.resolution, would_be, area.box.volume, direction
            )
        )

    if cost.total_bytes > LARGE_GRID_WARNING_BYTES:
        notes.append(
            "the voxel grid alone is {:.1f} GB, before ROS, the bridge and the "
            "recorder. A Jetson Orin NX has 16 GB shared with its GPU."
            .format(cost.total_bytes / 1024 ** 3)
        )

    return notes


def _read_area(map_config: dict) -> ExplorationArea:
    """Parse the ``area`` block, with a useful message when it is not there.

    Args:
        map_config: The file's ``map_config`` mapping.

    Returns:
        The parsed area.

    Raises:
        ValueError: If the block is missing or malformed.
    """
    if "area" in map_config:
        return ExplorationArea.from_dict(map_config["area"])
    if "map_size" in map_config:
        raise ValueError(
            "this file still uses the long-hand `map_size` block. Convert it to "
            "`area` (building, flight_band, vertical_extent, resolution) -- see "
            "mapsize/README.md."
        )
    raise ValueError("map_config has neither an `area` block nor a `map_size` one")


def _expand(config: dict, area: ExplorationArea) -> ExpandedRun:
    """Derive the three boxes and cost them.

    Args:
        config: The whole parsed file.
        area: Its exploration area.

    Returns:
        The expanded config.

    Raises:
        ValueError: If the geometry is invalid.
    """
    vbox = area.vbox
    _check_boxes(area, vbox)
    cost = grid_cost(area.map, area.resolution)
    map_config = dict(config["map_config"])

    map_size = {}
    map_size.update(area.map.to_falcon("map"))
    map_size.update(area.box.to_falcon("box"))
    map_size.update(vbox.to_falcon("vbox"))
    # Read by the patched map_server: an explicit resolution overrides the
    # volume rule. An unpatched FALCON ignores this key and falls back to it.
    map_size["resolution"] = area.resolution

    map_config.pop("area", None)
    map_config["map_size"] = map_size

    expanded = dict(config)
    expanded["map_config"] = map_config

    return ExpandedRun(
        config=expanded,
        area=area,
        cost=cost,
        warnings=tuple(_warnings(area, cost)),
    )


def expand_run(config: dict) -> ExpandedRun:
    """Expand a ``falcon_pegasus`` run file.

    The drawn box is a thin slab at the aircraft's cruise height, because that
    stack's recorder wants one horizontal cut of the map per frame.

    Args:
        config: A parsed run file, with both ``run`` and ``map_config``.

    Returns:
        The expanded run.

    Raises:
        ValueError: If the area is missing, malformed, or geometrically invalid.
    """
    area = _read_area(config["map_config"])

    if "cruise_altitude_m" not in config["run"]:
        raise ValueError(
            "run.cruise_altitude_m is required: the drawn box is a thin slab "
            "centred on it"
        )

    if area.visualisation is None:
        area = replace(
            area, visualisation_slab_at=float(config["run"]["cruise_altitude_m"])
        )

    return _expand(config, area)


def expand_map(config: dict) -> ExpandedRun:
    """Expand a ``falcon`` map file.

    No aircraft block and no slab: unless the file names a ``visualisation`` box
    of its own, what gets drawn is the whole allocated grid, which is what most
    of these environments already did by hand.

    Args:
        config: A parsed map file, with ``map_config``.

    Returns:
        The expanded config.

    Raises:
        ValueError: If the area is missing, malformed, or geometrically invalid.
    """
    return _expand(config, _read_area(config["map_config"]))


def load_any(path: Path) -> dict:
    """Read either kind of config file without expanding it.

    Args:
        path: A ``falcon_pegasus`` run file or a ``falcon`` map file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If it is empty, not a mapping, or has no ``map_config``.
    """
    return _load(path, ("map_config",))


def expand_any(config: dict) -> ExpandedRun:
    """Expand whichever kind of config this is.

    Args:
        config: A parsed run file or map file.

    Returns:
        The expanded config.

    Raises:
        ValueError: If the area is missing, malformed, or invalid.
    """
    return expand_run(config) if "run" in config else expand_map(config)


def load_and_expand(path: Path) -> ExpandedRun:
    """Read either kind of config file and expand it.

    A ``falcon_pegasus`` run file carries a ``run`` block alongside its map, and
    its recorder wants a thin slab at cruise height. A ``falcon`` map file
    carries only ``map_config``, and draws the whole allocated grid. Both shell
    launchers call this, so neither has to know which it is holding.

    Args:
        path: A run file or a map file.

    Returns:
        The expanded config.

    Raises:
        ValueError: If the file is empty, malformed, or geometrically invalid.
    """
    return expand_any(load_any(path))


def write_expanded(expanded: ExpandedRun, path: Path) -> Path:
    """Write the expanded config where ``rosparam load`` can read it.

    Args:
        expanded: The result of :func:`expand_run`.
        path: Destination file.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated from the run file's `area` block -- do not edit.\n"
        "# Change map_config.area in runs/<run>.yaml instead.\n"
    )
    path.write_text(
        header + yaml.safe_dump(expanded.config, sort_keys=False), encoding="utf-8"
    )
    return path
