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
        gx = ((-cloud_xyz[:, 0] - self.origin_x) / res).astype(np.int32)
        gy = ((cloud_xyz[:, 1] - self.origin_y) / res).astype(np.int32)
        gz = cloud_xyz[:, 2]

        in_bounds = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        # GATE: 0.4 to 2.5m catches walls but ignores the floor/odom-drift
        is_obstacle = (gz > 1.1) & (gz < 2.5)

        # Track current indices for the "Red" highlight
        valid_idx = in_bounds
        # self._last_indices = (gx[valid_idx], gy[valid_idx])

        # Log-odds update
        obs_mask = in_bounds & is_obstacle
        free_mask = in_bounds & (~is_obstacle)
        self._lo[gx[obs_mask], gy[obs_mask]] += self.cfg.lo_occ
        self._lo[gx[free_mask], gy[free_mask]] += self.cfg.lo_free
        self._seen_mask[gx[in_bounds], gy[in_bounds]] = True
        self._lo = np.clip(self._lo, self.cfg.lo_min, self.cfg.lo_max)

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