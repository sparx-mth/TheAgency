"""
2D costmap / occupancy grid utilities.

This class represents a binary occupancy grid (free/occupied) with:
- world <-> grid transforms
- obstacle dilation via safety margin (optional, caller can precompute)
- optional clearance map (distance-to-obstacle in grid units or meters)

Notes:
- This module intentionally does NOT load YAML/PGM. Loading belongs in adapters (e.g., ros/adapters/map_loader.py).
- Grid convention: (gx, gy) are integer indices: gx in [0,width), gy in [0,height)
- World convention: (x,y) in meters in the same frame as origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CostmapParams:
    """Metadata needed for world<->grid conversion."""
    resolution: float               # meters per pixel
    origin_x: float                 # world meters of grid (0,0) corner
    origin_y: float
    frame_id: str = "map"


class Costmap2D:
    """
    Binary occupancy grid and optional clearance.

    Attributes:
        occupancy: uint8 numpy array of shape (H, W).
            Convention: 0 = free, 1 = occupied.
        clearance: optional float32 numpy array of shape (H, W).
            Meaning: distance to nearest obstacle (meters or pixels; see clearance_unit).
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        params: CostmapParams,
        *,
        clearance: Optional[np.ndarray] = None,
        clearance_unit: str = "meters",   # "meters" or "cells"
    ) -> None:
        if occupancy.ndim != 2:
            raise ValueError(f"occupancy must be 2D (H,W), got shape={occupancy.shape}")

        occ = occupancy
        if occ.dtype != np.uint8:
            occ = occ.astype(np.uint8, copy=False)

        if not np.isin(occ, [0, 1]).all():
            # allow any nonzero treated as occupied; normalize to 0/1
            occ = (occ != 0).astype(np.uint8)

        self.occupancy: np.ndarray = occ
        self.params: CostmapParams = params

        self._height: int = int(occ.shape[0])
        self._width: int = int(occ.shape[1])

        self.clearance: Optional[np.ndarray] = None
        self.clearance_unit: str = clearance_unit

        if clearance is not None:
            if clearance.shape != occ.shape:
                raise ValueError(
                    f"clearance shape must match occupancy shape. "
                    f"occ={occ.shape}, clearance={clearance.shape}"
                )
            self.clearance = clearance.astype(np.float32, copy=False)

            if clearance_unit not in ("meters", "cells"):
                raise ValueError("clearance_unit must be 'meters' or 'cells'")

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def resolution(self) -> float:
        return float(self.params.resolution)

    @property
    def frame_id(self) -> str:
        return self.params.frame_id

    # ------------------------------------------------------------------
    # Bounds / transforms
    # ------------------------------------------------------------------

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self._width and 0 <= gy < self._height

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world meters -> grid indices.

        Uses floor by default (stable). You can change to round if your pipeline prefers.
        """
        gx = int(floor((x - self.params.origin_x) / self.params.resolution))
        gy = int(floor((y - self.params.origin_y) / self.params.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """
        Convert grid indices -> world meters (cell center).
        """
        x = (gx + 0.5) * self.params.resolution + self.params.origin_x
        y = (gy + 0.5) * self.params.resolution + self.params.origin_y
        return float(x), float(y)

    # ------------------------------------------------------------------
    # Occupancy / clearance queries
    # ------------------------------------------------------------------

    def is_free(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return False
        return self.occupancy[gy, gx] == 0

    def is_occupied(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return True
        return self.occupancy[gy, gx] != 0

    def clearance_at(self, gx: int, gy: int) -> float:
        """
        Return clearance at cell (gx,gy).

        If no clearance map is available, returns 0.0.
        """
        if self.clearance is None:
            return 0.0
        if not self.in_bounds(gx, gy):
            return 0.0
        val = float(self.clearance[gy, gx])
        if self.clearance_unit == "cells":
            return val * self.params.resolution
        return val

    def world_clearance(self, x: float, y: float) -> float:
        gx, gy = self.world_to_grid(x, y)
        return self.clearance_at(gx, gy)
