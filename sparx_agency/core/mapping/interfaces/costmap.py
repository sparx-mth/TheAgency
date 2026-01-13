from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class GridSpec:
    resolution_m: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    frame_id: str = "map"


class Costmap(ABC):
    """
    2D costmap API.

    Convention:
      - grid stores values in [0..100], or -1 for unknown
      - frame_id is the map/world frame for the grid
    """

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def update_from_points_xy(
    #     self,
    #     points_xy: np.ndarray,      # (N,2) in map/world frame
    #     stamp_sec: Optional[float] = None,
    # ) -> None:
    #     raise NotImplementedError

    @abstractmethod
    def update_from_cloud(
            self,
            cloud_xyz: np.ndarray,  # (N,3) in map/world frame
            sensor_origin: np.ndarray,  # (3,) [x, y, z] in map/world frame
    ) -> None:
        """Process 3D data to update the 2D grid."""
        pass

    @abstractmethod
    def get_grid(self) -> Tuple[GridSpec, np.ndarray]:
        """Returns (spec, grid_int8) where grid is shape (H,W) int8."""
        raise NotImplementedError
