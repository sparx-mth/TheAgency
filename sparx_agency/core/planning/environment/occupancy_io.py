"""Persist an :class:`OccupancyGrid2D` to a single ``.npz`` file and read it back.

A surveyed map of a building is expensive to produce and never changes, so it
belongs on disk next to the environment it describes rather than being rebuilt
on every run. That is the whole reason this exists: the simulated data-collection
pipeline surveys an Isaac Sim scene once (minutes of raycasting) and then boots
hundreds of flights against the cached result in milliseconds.

One file holds the cells *and* the metadata needed to interpret them --
resolution, origin and the free/occupied/unknown encoding -- because a grid
without its origin is not a map, and keeping them in separate files invites the
two drifting apart.

``metadata`` is a free-form JSON dict for whatever produced the map (which
scene, at what altitude, with what robot radius). It is carried through
untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .occupancy_grid2d import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues

SUFFIX = ".npz"


LAYER_PREFIX = "layer_"


def save_occupancy_grid(
    path, grid: OccupancyGrid2D, metadata: Optional[Dict] = None,
    layers: Optional[Dict] = None,
) -> Path:
    """Write ``grid``, its metadata and any extra layers to a compressed ``.npz``.

    Args:
        path: Destination file. Parent directories are created.
        grid: The grid to store.
        metadata: Optional JSON-serialisable provenance dict, returned verbatim
            by :func:`load_occupancy_grid`.
        layers: Optional extra per-cell boolean masks, co-registered with
            ``grid`` -- anything the survey measured that is not occupancy. They
            travel in the same file for the same reason the origin does: a layer
            that can be separated from its grid will eventually be paired with
            the wrong one.

    Returns:
        The path written.

    Raises:
        ValueError: If a layer's shape does not match the grid's.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = grid.params
    values = grid.values

    arrays = {
        "grid": grid.grid.astype(np.int16, copy=False),
        "resolution": np.float64(params.resolution),
        "origin": np.array([params.origin_x, params.origin_y], dtype=np.float64),
        "frame_id": np.array(params.frame_id),
        "values": np.array([values.free, values.occupied, values.unknown], dtype=np.int16),
        "metadata": np.array(json.dumps(metadata or {})),
    }
    for name, mask in (layers or {}).items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != grid.grid.shape:
            raise ValueError(
                f"layer {name!r} has shape {mask.shape}, grid has {grid.grid.shape}"
            )
        arrays[LAYER_PREFIX + name] = mask
    np.savez_compressed(str(path), **arrays)
    return path


def load_occupancy_grid(path) -> Tuple[OccupancyGrid2D, Dict, Dict]:
    """Read back a grid written by :func:`save_occupancy_grid`.

    Args:
        path: The ``.npz`` file to read.

    Returns:
        ``(grid, metadata, layers)``. ``layers`` is empty for a file written
        without any.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no occupancy grid at {path}")

    with np.load(str(path), allow_pickle=False) as data:
        cells = data["grid"]
        origin = data["origin"]
        free, occupied, unknown = (int(v) for v in data["values"])
        params = OccupancyGrid2DParams(
            resolution=float(data["resolution"]),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            frame_id=str(data["frame_id"]),
        )
        metadata = json.loads(str(data["metadata"]))
        layers = {name[len(LAYER_PREFIX):]: np.asarray(data[name], dtype=bool)
                  for name in data.files if name.startswith(LAYER_PREFIX)}

    values = OccupancyValues(free=free, occupied=occupied, unknown=unknown)
    return OccupancyGrid2D(cells, params, values=values), metadata, layers


def occupancy_from_mask(
    occupied: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    frame_id: str = "map",
    known: Optional[np.ndarray] = None,
) -> OccupancyGrid2D:
    """Build an :class:`OccupancyGrid2D` from a boolean obstacle mask.

    The usual shape of a survey result: one boolean array saying where geometry
    is, optionally a second saying which cells were surveyed at all (everything
    outside the building's footprint is UNKNOWN, not free -- treating unsurveyed
    space as free would let a planner route the robot straight out of the map).

    Args:
        occupied: ``(H, W)`` boolean, True where an obstacle is. Indexed
            ``[gy, gx]``, matching :class:`OccupancyGrid2D`.
        resolution: Metres per cell.
        origin_x: World x of cell ``(0, 0)``'s lower corner.
        origin_y: World y of cell ``(0, 0)``'s lower corner.
        frame_id: Coordinate frame name.
        known: Optional ``(H, W)`` boolean, True where the cell was surveyed.
            Cells outside it become UNKNOWN. Defaults to everything known.

    Returns:
        The grid, using the default FREE=0 / OCCUPIED=1 / UNKNOWN=-1 encoding.
    """
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 2:
        raise ValueError(f"occupied must be 2D, got shape {occupied.shape}")

    values = OccupancyValues()
    cells = np.where(occupied, values.occupied, values.free).astype(np.int16)
    if known is not None:
        known = np.asarray(known, dtype=bool)
        if known.shape != occupied.shape:
            raise ValueError(
                f"known {known.shape} does not match occupied {occupied.shape}"
            )
        cells[~known] = values.unknown

    params = OccupancyGrid2DParams(
        resolution=float(resolution), origin_x=float(origin_x),
        origin_y=float(origin_y), frame_id=frame_id,
    )
    return OccupancyGrid2D(cells, params, values=values)
