"""Geometry of the boolean raster a slab of 3D geometry is drawn into.

Rasterising a mesh needs five numbers -- cell size, the world coordinate of the
grid's lower-left corner, and the grid's extent -- and every stage of the
pipeline needs all five. Passing them as one frozen value keeps the origin and
the resolution together, because a grid separated from its origin is not a map.

The convention here is plain grid indexing: ``grid[row, col]`` with ``row 0``
at **minimum** y and ``col 0`` at minimum x. That is deliberately *not* the
top-down convention used by nav2 PGM images; whatever writes an image flips.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """A rectangular, axis-aligned raster in world metres.

    Attributes:
        resolution: Metres per cell. Cells are square.
        origin_x: World x of the lower-left corner of cell ``(0, 0)``.
        origin_y: World y of the lower-left corner of cell ``(0, 0)``.
        width: Number of columns (x).
        height: Number of rows (y).
    """

    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.resolution > 0.0:
            raise ValueError("resolution must be positive, got %r" % (self.resolution,))
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "grid must have positive extent, got %dx%d" % (self.width, self.height)
            )

    @property
    def shape(self) -> Tuple[int, int]:
        """``(height, width)`` -- the shape of the boolean array."""
        return int(self.height), int(self.width)

    def empty(self) -> np.ndarray:
        """Return a fresh all-False grid of this shape."""
        return np.zeros(self.shape, dtype=bool)

    def to_cell_coords(self, points_xy: np.ndarray) -> np.ndarray:
        """Convert world XY to continuous cell coordinates.

        Cell ``(col, row)`` spans ``[col, col + 1) x [row, row + 1)`` in the
        returned coordinates, so ``floor()`` of the result is the cell index.

        Args:
            points_xy: ``(..., 2)`` or ``(..., 3)`` array of world points. Only
                the first two components are read.

        Returns:
            ``(..., 2)`` float64 array of continuous cell coordinates.
        """
        points = np.asarray(points_xy, dtype=np.float64)
        if points.shape[-1] < 2:
            raise ValueError("points must have at least 2 components")
        out = np.empty(points.shape[:-1] + (2,), dtype=np.float64)
        out[..., 0] = (points[..., 0] - float(self.origin_x)) / float(self.resolution)
        out[..., 1] = (points[..., 1] - float(self.origin_y)) / float(self.resolution)
        return out
