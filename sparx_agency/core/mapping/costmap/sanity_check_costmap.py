from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from sparx_agency.core.mapping.interfaces.costmap import Costmap, GridSpec


class SanityCheckCostmap(Costmap):
    def __init__(self, size_m=100.0, res=0.3):
        self.res = res
        self.width = int(size_m / res)
        self.height = int(size_m / res)
        self.origin_x = -size_m / 2
        self.origin_y = -size_m / 2
        self._grid = np.full((self.height, self.width), -1, dtype=np.int8)

    def update_from_cloud(self, cloud_xyz: np.ndarray, sensor_origin: np.ndarray):
        if cloud_xyz.size == 0: return
        # 1. RESET EVERY FRAME (No Memory)
        self._grid.fill(0)
        try:
            # 2. THE STRICT GATE (1.1m to 2.5m)
            # We use world Z (cloud_xyz[:, 2])
            mask = (cloud_xyz[:, 2] > 1.1) & (cloud_xyz[:, 2] < 2.5)
            obs_pts = cloud_xyz[mask]

            if obs_pts.size == 0: return

            # 3. COORDINATE MAPPING
            # If the cloud is pointing along Blue (Z) but should be Red (X),
            # it means the transform failed. We map them to the grid here:
            gx = ((obs_pts[:, 0] - self.origin_x) / self.res).astype(np.int32)
            gy = ((obs_pts[:, 1] - self.origin_y) / self.res).astype(np.int32)

            # 4. BOUNDS CHECK & DRAW
            valid = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)

            # Draw obstacles as Black (100)
            self._grid[gy[valid], gx[valid]] = 100

            # OPTIONAL: Mark the drone's current position as a different color (e.g. 50)
            # to see if the grid is centered correctly
            self.mark_drone_pos(sensor_origin)
        except Exception as e:
            print(f"Error in update from cloud sanity: {e}")

    def mark_drone_pos(self, t):
        dgx = int((t[0] - self.origin_x) / self.res)
        dgy = int((t[1] - self.origin_y) / self.res)
        if 0 <= dgx < self.width and 0 <= dgy < self.height:
            self._grid[dgy, dgx] = 50  # Gray dot for the drone

    def get_grid(self) -> Tuple[GridSpec, np.ndarray]:
        # Ensure frame_id matches what RViz expects
        spec = GridSpec(self.res, self.width, self.height, self.origin_x, self.origin_y, "simple_drone/odom")
        return spec, self._grid

    def reset(self):
        self._grid.fill(-1)