from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from .probabilistic_grid_config import ProbabilisticGridConfig, bresenham
from sparx_agency.core.mapping.interfaces.costmap import Costmap, GridSpec


class ProbabilisticGridCostmap(Costmap):
    def __init__(self, cfg: Optional[ProbabilisticGridConfig] = None):
        self._last_indices = None
        self.cfg = cfg or ProbabilisticGridConfig()
        self.width = int(np.ceil(self.cfg.size_m / self.cfg.resolution_m))
        self.height = int(np.ceil(self.cfg.size_m / self.cfg.resolution_m))

        # Internal state
        self._lo = np.zeros((self.height, self.width), dtype=np.float32)
        # Persistent mask to track which cells have been observed at least once
        self._seen_mask = np.zeros((self.height, self.width), dtype=bool)

        self.origin_x = -0.5 * self.cfg.size_m
        self.origin_y = -0.5 * self.cfg.size_m


    def reset(self) -> None:
        self._lo.fill(0.0)
        self._seen_mask.fill(False)

    def update_from_cloud(self, cloud_xyz: np.ndarray, sensor_origin: np.ndarray):
        res = self.cfg.resolution_m
        # Use WORLD coordinates (cloud_xyz) for gx/gy
        gx = ((cloud_xyz[:, 1] - self.origin_y) / res).astype(np.int32)  # Map World Y to Grid X
        gy = ((cloud_xyz[:, 0] - self.origin_x) / res).astype(np.int32)  # Map World X to Grid Y
        gz = cloud_xyz[:, 2]

        in_bounds = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        is_obstacle = (gz > self.cfg.min_height_obstacle) & (gz < self.cfg.max_height_obstacle)
        obs_pts_mask = in_bounds & is_obstacle
        counts = np.zeros((self.height, self.width), dtype=np.int32)
        # for EVERY point. If 50 points fall in cell (10, 10),
        # counts[10, 10] will equal 50.
        np.add.at(counts, (gx[obs_pts_mask], gy[obs_pts_mask]), 1)
        confirmed_obs = (counts >= self.cfg.points_to_occupied)
        thin_points = in_bounds & (~confirmed_obs[gy, gx])

        # Log-odds update
        obs_mask = in_bounds & is_obstacle
        free_mask = in_bounds & (~is_obstacle)

        self._lo[confirmed_obs] += self.cfg.lo_occ
        self._lo[gx[thin_points], gy[thin_points]] += self.cfg.lo_free
        self._lo[gx[obs_mask], gy[obs_mask]] += self.cfg.lo_occ
        self._lo[gx[free_mask], gy[free_mask]] += self.cfg.lo_free
        self._seen_mask[gx[in_bounds], gy[in_bounds]] = True
        self._lo = np.clip(self._lo, self.cfg.lo_min, self.cfg.lo_max)
        if self.cfg.debug:
            self._last_indices = np.where(confirmed_obs)

    def get_grid(self) -> Tuple[GridSpec, np.ndarray]:
        # 1. Start with Gray (Unknown -1)
        grid_data = np.full((self.width, self.height), -1, dtype=np.int8)

        # 2. Fill in the history
        mask_free = self._seen_mask & (self._lo <= 0)
        grid_data[mask_free] = 20  # Light Gray

        mask_wall = self._seen_mask & (self._lo > 0.5)
        grid_data[mask_wall] = 100  # Black

        # 3. Highlight the CURRENT frame in a special value
        if hasattr(self, '_last_indices') and self._last_indices is not None:
            lx, ly = self._last_indices
            # Values > 100 show up as colored (Red/Purple) in 'Costmap' mode
            grid_data[lx, ly] = 120

        spec = GridSpec(self.cfg.resolution_m, self.width, self.height,
                        self.origin_x, self.origin_y, self.cfg.frame_id)
        return spec, grid_data