"""
2D occupancy grid for path planning.

Provides grid-world coordinate transforms, occupancy queries, and optional
clearance (distance-to-obstacle) lookups.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CostmapParams:
    """Grid metadata for coordinate transforms."""
    resolution: float  # meters per cell
    origin_x: float    # world x-coordinate of cell (0,0)
    origin_y: float    # world y-coordinate of cell (0,0)
    frame_id: str = "map"


class Costmap2D:
    """
    Binary occupancy grid with optional clearance field.

    Attributes:
        occupancy: Binary grid (H, W), where 0=free, 1=occupied.
        clearance: Optional distance-to-obstacle field (H, W) in meters.
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        params: CostmapParams,
        *,
        clearance: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            occupancy: 2D array of shape (height, width). Non-zero = occupied.
            params: Grid metadata (resolution, origin, frame).
            clearance: Optional distance field in meters, same shape as occupancy.
        """
        if occupancy.ndim != 2:
            raise ValueError(f"occupancy must be 2D, got shape {occupancy.shape}")

        # Normalize to binary uint8
        occ = (occupancy != 0).astype(np.uint8)

        self._occupancy = occ
        self._params = params
        self._height, self._width = occ.shape

        self._clearance: Optional[np.ndarray] = None
        if clearance is not None:
            if clearance.shape != occ.shape:
                raise ValueError(
                    f"clearance shape {clearance.shape} != occupancy shape {occ.shape}"
                )
            self._clearance = clearance.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def occupancy(self) -> np.ndarray:
        return self._occupancy

    @property
    def params(self) -> CostmapParams:
        return self._params

    @property
    def clearance(self) -> Optional[np.ndarray]:
        return self._clearance

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def resolution(self) -> float:
        return self._params.resolution

    @property
    def origin_x(self) -> float:
        return self._params.origin_x

    @property
    def origin_y(self) -> float:
        return self._params.origin_y

    @property
    def frame_id(self) -> str:
        return self._params.frame_id

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def in_bounds(self, gx: int, gy: int) -> bool:
        """Check if grid cell (gx, gy) is within map bounds."""
        return 0 <= gx < self._width and 0 <= gy < self._height

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices."""
        gx = int(floor((x - self._params.origin_x) / self._params.resolution))
        gy = int(floor((y - self._params.origin_y) / self._params.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = (gx + 0.5) * self._params.resolution + self._params.origin_x
        y = (gy + 0.5) * self._params.resolution + self._params.origin_y
        return x, y

    # ------------------------------------------------------------------
    # Occupancy queries
    # ------------------------------------------------------------------

    def is_free(self, gx: int, gy: int) -> bool:
        """Check if cell is free. Out-of-bounds cells are considered occupied."""
        if not self.in_bounds(gx, gy):
            return False
        return self._occupancy[gy, gx] == 0

    def is_occupied(self, gx: int, gy: int) -> bool:
        """Check if cell is occupied. Out-of-bounds cells are considered occupied."""
        if not self.in_bounds(gx, gy):
            return True
        return self._occupancy[gy, gx] != 0

    # ------------------------------------------------------------------
    # Clearance queries
    # ------------------------------------------------------------------

    def clearance_at(self, gx: int, gy: int) -> float:
        """
        Get clearance (meters) at grid cell.

        Returns 0.0 if no clearance map or cell is out-of-bounds.
        """
        if self._clearance is None or not self.in_bounds(gx, gy):
            return 0.0
        return float(self._clearance[gy, gx])

    def world_clearance(self, x: float, y: float) -> float:
        """Get clearance (meters) at world position."""
        gx, gy = self.world_to_grid(x, y)
        return self.clearance_at(gx, gy)