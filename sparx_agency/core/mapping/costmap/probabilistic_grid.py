from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from .probabilistic_grid_config import ProbabilisticGridConfig, bresenham
from sparx_agency.core.mapping.interfaces.costmap import Costmap, GridSpec


class ProbabilisticGridCostmap(Costmap):
    def __init__(self, cfg: Optional[ProbabilisticGridConfig] = None):
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
        # Explicitly cast to int32 immediately
        gx = ((cloud_xyz[:, 0] - self.origin_x) / res).astype(np.int32)
        gy = ((cloud_xyz[:, 1] - self.origin_y) / res).astype(np.int32)
        gz = cloud_xyz[:, 2]  # Height stays as float for filtering

        # 1. Bounds check
        in_bounds = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)

        # 2. Height Gate: Only points between 0.3m and 2.0m are obstacles.
        # This deletes the floor 'Triangle' shadow
        is_obstacle = (gz > 0.3) & (gz < 2.0)

        # 3. Mark everything in-bounds as 'Seen'
        # We use integers here to avoid the IndexError
        self._seen_mask[gy[in_bounds], gx[in_bounds]] = True

        # 4. Update Occupancy ONLY for obstacle points
        # Re-cast to int32 here as a safety 'Sanity Check' against the IndexError
        valid_gx = gx[in_bounds & is_obstacle].astype(np.int32)
        valid_gy = gy[in_bounds & is_obstacle].astype(np.int32)

        if valid_gx.size > 0:
            self._lo[valid_gy, valid_gx] = np.clip(
                self._lo[valid_gy, valid_gx] + self.cfg.lo_occ,
                self.cfg.lo_min, self.cfg.lo_max
            )
    def _clear_rays(self, origin, txs, tys):
        res = self.cfg.resolution_m
        sx = int((origin[0] - self.origin_x) / res)
        sy = int((origin[1] - self.origin_y) / res)

        for tx, ty in zip(txs, tys):
            for cx, cy in bresenham(sx, sy, tx, ty):
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    self._seen_mask[cy, cx] = True
                    if (cx, cy) != (tx, ty):
                        # Subtract log-odds for free space
                        self._lo[cy, cx] = np.clip(self._lo[cy, cx] + self.cfg.lo_free, self.cfg.lo_min,
                                                   self.cfg.lo_max)

    def get_grid(self) -> Tuple[GridSpec, np.ndarray]:
        # Sigmoid conversion: lo=0 -> 50%, lo>>0 -> 100%, lo<<0 -> 0%
        prob = 100.0 / (1.0 + np.exp(-self._lo))
        grid_data = prob.astype(np.int8)

        # CRITICAL: -1 is ONLY for areas never touched by the camera
        grid_data[~self._seen_mask] = -1

        # For areas we HAVE seen, if probability is low, force to 0 (Free)
        # so it doesn't look like "Unknown" -1 in RViz
        mask_free = self._seen_mask & (self._lo < -0.2)
        grid_data[mask_free] = 0

        spec = GridSpec(
            resolution_m=self.cfg.resolution_m,
            width=self.width,
            height=self.height,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            frame_id=self.cfg.frame_id
        )
        return spec, grid_data