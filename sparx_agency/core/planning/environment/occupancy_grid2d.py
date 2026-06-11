"""
OccupancyGrid2D for exploration (supports UNKNOWN).

This class is intentionally separate from Costmap2D:
- Costmap2D is binary occupancy (free/occupied) and may include clearance fields.
- OccupancyGrid2D models SLAM-style maps with UNKNOWN cells (unobserved).

Coordinates:
- World coordinates are meters in the grid frame.
- Grid coordinates are integer indices (gx, gy), with numpy indexing grid[gy, gx].
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class OccupancyGrid2DParams:
    """Grid metadata for coordinate transforms."""
    resolution: float  # meters per cell
    origin_x: float    # world x-coordinate of cell (0,0)
    origin_y: float    # world y-coordinate of cell (0,0)
    frame_id: str = "map"


@dataclass(frozen=True)
class OccupancyValues:
    """
    Semantic values used inside the grid.
    Keep numeric to allow wrapping existing numpy maps.
    """
    free: int = 0
    occupied: int = 1
    unknown: int = -1


class OccupancyGrid2D:
    """
    Discrete occupancy grid with UNKNOWN support.

    grid: np.ndarray of shape (H, W), integer-like values
          expected to include at least free/occupied/unknown.
    """

    def __init__(
        self,
        grid: np.ndarray,
        params: OccupancyGrid2DParams,
        *,
        values: OccupancyValues = OccupancyValues(),
    ) -> None:
        if grid.ndim != 2:
            raise ValueError(f"grid must be 2D, got shape {grid.shape}")

        self._grid = grid.astype(np.int16, copy=False)
        self._params = params
        self._values = values
        self._height, self._width = self._grid.shape

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def params(self) -> OccupancyGrid2DParams:
        return self._params

    @property
    def values(self) -> OccupancyValues:
        return self._values

    @property
    def width(self) -> int:
        return int(self._width)

    @property
    def height(self) -> int:
        return int(self._height)

    @property
    def resolution(self) -> float:
        return float(self._params.resolution)

    @property
    def origin_x(self) -> float:
        return float(self._params.origin_x)

    @property
    def origin_y(self) -> float:
        return float(self._params.origin_y)

    @property
    def frame_id(self) -> str:
        return str(self._params.frame_id)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self._width and 0 <= gy < self._height

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(floor((x - self._params.origin_x) / self._params.resolution))
        gy = int(floor((y - self._params.origin_y) / self._params.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = (gx + 0.5) * self._params.resolution + self._params.origin_x
        y = (gy + 0.5) * self._params.resolution + self._params.origin_y
        return x, y

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def value_at(self, gx: int, gy: int) -> Optional[int]:
        if not self.in_bounds(gx, gy):
            return None
        return int(self._grid[gy, gx])

    def is_free(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return False
        return int(self._grid[gy, gx]) == self._values.free

    def is_unknown(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return False
        return int(self._grid[gy, gx]) == self._values.unknown

    def is_occupied(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return True
        return int(self._grid[gy, gx]) == self._values.occupied
