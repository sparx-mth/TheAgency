"""Write an occupancy grid as a nav2 ``map_server`` PGM plus YAML.

The format ROS tooling already reads, so the map drops straight into
``map_server``, RViz or any of this repo's viewers without a converter.

The one thing that bites everybody: **a PGM's first row is the top of the
image, which is maximum y**, while every grid in this codebase indexes row 0 at
minimum y. The flip happens here and nowhere else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

OCCUPIED_PIXEL = 0
FREE_PIXEL = 254
UNKNOWN_PIXEL = 205
OCCUPIED_THRESH = 0.65
# nav2 decodes a pixel as occ = 1 - pixel/255 and calls it free when
# occ < free_thresh. The unknown grey is 205, i.e. occ = 0.196, so the usual
# 0.25 makes every unknown cell read back as *free* -- a map_server that
# invents open space where nothing was surveyed. 0.196 is the largest
# threshold that still keeps the grey on the unknown side of the comparison.
FREE_THRESH = 0.196
MODE = "trinary"


def to_pgm_image(occupied: np.ndarray, known: np.ndarray = None) -> np.ndarray:
    """Convert a boolean occupancy grid to nav2 greyscale pixels.

    Args:
        occupied: ``(H, W)`` boolean, True where geometry is, indexed with row
            0 at minimum y.
        known: Optional ``(H, W)`` boolean, True where the cell was surveyed.
            Cells outside it become the unknown grey. Omit it for a
            ground-truth map, where every cell is known by construction.

    Returns:
        ``(H, W)`` uint8 image in PGM row order -- row 0 is **maximum** y.
    """
    occupied = np.asarray(occupied, dtype=bool)
    image = np.where(occupied, OCCUPIED_PIXEL, FREE_PIXEL).astype(np.uint8)
    if known is not None:
        image[~np.asarray(known, dtype=bool)] = UNKNOWN_PIXEL
    return np.flipud(image)


def write_nav2_map(
    output_dir,
    name: str,
    occupied: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    known: np.ndarray = None,
) -> Tuple[Path, Path]:
    """Write ``<name>.pgm`` and ``<name>.yaml`` in nav2 ``map_server`` format.

    Args:
        output_dir: Destination directory, created if missing.
        name: Basename, without a suffix.
        occupied: ``(H, W)`` boolean occupancy, row 0 at minimum y.
        resolution: Metres per cell.
        origin_x: World x of the grid's lower-left corner.
        origin_y: World y of the grid's lower-left corner.
        known: Optional surveyed mask; see :func:`to_pgm_image`.

    Returns:
        ``(pgm_path, yaml_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = to_pgm_image(occupied, known=known)

    pgm_path = output_dir / (name + ".pgm")
    with pgm_path.open("wb") as handle:
        handle.write(b"P5\n")
        handle.write(b"# ground truth, generated from world geometry\n")
        handle.write(("%d %d\n255\n" % (image.shape[1], image.shape[0])).encode())
        handle.write(image.tobytes())

    yaml_path = output_dir / (name + ".yaml")
    document = {
        "image": pgm_path.name,
        "mode": MODE,
        "resolution": float(resolution),
        "origin": [float(origin_x), float(origin_y), 0.0],
        "negate": 0,
        "occupied_thresh": OCCUPIED_THRESH,
        "free_thresh": FREE_THRESH,
    }
    with yaml_path.open("w") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False)
    return pgm_path, yaml_path


def read_pgm(path) -> np.ndarray:
    """Read back a binary P5 PGM, for tests and round-trip checks.

    Args:
        path: The ``.pgm`` file.

    Returns:
        ``(H, W)`` uint8 image in PGM row order.

    Raises:
        ValueError: If the file is not a binary P5 PGM of 8-bit samples.
    """
    data = Path(path).read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError("%s is not a binary P5 PGM" % (path,))

    fields = []
    cursor = 2
    while len(fields) < 3:
        while cursor < len(data) and data[cursor:cursor + 1].isspace():
            cursor += 1
        if data[cursor:cursor + 1] == b"#":
            cursor = data.index(b"\n", cursor) + 1
            continue
        start = cursor
        while cursor < len(data) and not data[cursor:cursor + 1].isspace():
            cursor += 1
        fields.append(int(data[start:cursor]))
    if fields[2] != 255:
        raise ValueError("only 8-bit PGMs are supported, got maxval %d" % (fields[2],))

    width, height = fields[0], fields[1]
    pixels = data[cursor + 1:cursor + 1 + width * height]
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)
