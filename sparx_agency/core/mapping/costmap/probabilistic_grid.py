# probabilistic_grid.py

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.core.mapping.interfaces.costmap import Costmap, GridSpec


class ProbabilisticGridCostmap(Costmap):
    """
    A simple accumulated 2D log-odds grid in a fixed square area centered at (0,0) in the map frame.
    Conventions:
      - cloud_xyz is in WORLD/MAP coordinates: x forward/east, y left/north, z up
      - grid_data is (H, W) with indexing [gy, gx]
    """

    def __init__(self, cfg: Optional[ProbabilisticGridConfig] = None):
        self._last_indices = None
        self.cfg = cfg or ProbabilisticGridConfig()

        self.width = int(np.ceil(self.cfg.size_m / self.cfg.resolution_m))
        self.height = int(np.ceil(self.cfg.size_m / self.cfg.resolution_m))

        # Internal state: (H, W)
        self._lo = np.zeros((self.height, self.width), dtype=np.float32)
        self._seen_mask = np.zeros((self.height, self.width), dtype=bool)

        # Grid origin (bottom-left in world coords if you follow OccupancyGrid convention)
        self.origin_x = -0.5 * self.cfg.size_m
        self.origin_y = -0.5 * self.cfg.size_m

    def reset(self) -> None:
        self._lo.fill(0.0)
        self._seen_mask.fill(False)
        self._last_indices = None

    def update_from_cloud(self, cloud_xyz: np.ndarray, sensor_origin: np.ndarray) -> None:
        if cloud_xyz is None or cloud_xyz.shape[0] == 0:
            return

        res = float(self.cfg.resolution_m)

        # World -> grid indices
        gx = ((cloud_xyz[:, 0] - self.origin_x) / res).astype(np.int32)  # x -> col
        gy = ((cloud_xyz[:, 1] - self.origin_y) / res).astype(np.int32)  # y -> row
        gz = cloud_xyz[:, 2]

        in_bounds = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)

        # Height band to decide if this point is an obstacle
        is_obstacle = (gz > self.cfg.min_height_obstacle) & (gz < self.cfg.max_height_obstacle)

        obs_pts_mask = in_bounds & is_obstacle

        # Count obstacle points per cell (H,W) indexed as [gy, gx]
        counts = np.zeros((self.height, self.width), dtype=np.int32)
        np.add.at(counts, (gy[obs_pts_mask], gx[obs_pts_mask]), 1)

        confirmed_obs = counts >= self.cfg.points_to_occupied  # (H,W) bool

        # Points in cells that are NOT confirmed occupied contribute as free evidence
        # (avoid over-freeing true walls)
        thin_points = in_bounds & (~confirmed_obs[gy, gx])

        # Log-odds update masks
        obs_mask = in_bounds & is_obstacle
        free_mask = in_bounds & (~is_obstacle)

        # Apply updates
        self._lo[confirmed_obs] += self.cfg.lo_occ
        self._lo[gy[thin_points], gx[thin_points]] += self.cfg.lo_free
        self._lo[gy[obs_mask], gx[obs_mask]] += self.cfg.lo_occ
        self._lo[gy[free_mask], gx[free_mask]] += self.cfg.lo_free

        self._seen_mask[gy[in_bounds], gx[in_bounds]] = True
        self._lo = np.clip(self._lo, self.cfg.lo_min, self.cfg.lo_max)

        if self.cfg.debug:
            # Save confirmed occupied indices for visualization
            self._last_indices = np.where(confirmed_obs)  # (gy_idx, gx_idx)

    def get_grid(self) -> Tuple[GridSpec, np.ndarray]:
        """
        Returns:
          spec: GridSpec
          grid_data: int8 (H,W) in OccupancyGrid-like values:
            -1 unknown, 20 free-ish, 100 occupied, 120 highlight current confirmed_obs (debug)
        """
        grid_data = np.full((self.height, self.width), -1, dtype=np.int8)

        # History layers
        mask_free = self._seen_mask & (self._lo <= 0.0)
        grid_data[mask_free] = 20

        mask_wall = self._seen_mask & (self._lo > 0.5)
        grid_data[mask_wall] = 100

        # Highlight last confirmed occupied
        if self._last_indices is not None:
            gy_idx, gx_idx = self._last_indices
            grid_data[gy_idx, gx_idx] = 120

        spec = GridSpec(
            self.cfg.resolution_m,
            self.width,
            self.height,
            self.origin_x,
            self.origin_y,
            self.cfg.frame_id,
        )
        return spec, grid_data