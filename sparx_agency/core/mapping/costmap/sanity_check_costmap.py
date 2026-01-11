from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from sparx_agency.core.mapping.interfaces.costmap import Costmap, GridSpec


class SanityCheckCostmap(Costmap):

    def __init__(self, size_m=50.0, res=0.2):
        self.res = res
        self.width = int(size_m / res)
        self.height = int(size_m / res)
        self.origin_x = -size_m / 2
        self.origin_y = -size_m / 2
        # Start with all Unknown (-1)
        self._grid = np.full((self.height, self.width), -1, dtype=np.int8)

    def update_from_cloud(self, cloud_xyz: np.ndarray, sensor_origin: np.ndarray):
        """
        cloud_xyz: (N, 3) in WORLD (odom) coordinates.
        """
        if cloud_xyz.size == 0: return

        # 1. MOMENTARY CLEAR: Clear the grid every frame for debugging
        self._grid.fill(0)
        # 2. THE SANITY GATE:
        # Only points between 0.2m and 2.0m height are obstacles.
        # This removes the floor 'Triangle' because floor Z is ~0.0.
        # Inside your occupancy grid update:
        mask = (cloud_xyz[:, 2] > 0.3) & (cloud_xyz[:, 2] < 2.0)
        # This ignores the floor (0.0m) and ceiling
        obs_pts = cloud_xyz[mask]

        # 3. Project only these true obstacles to the 2D grid
        gx = ((obs_pts[:, 0] - self.origin_x) / self.res).astype(np.int32)
        gy = ((obs_pts[:, 1] - self.origin_y) / self.res).astype(np.int32)

        valid = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        self._grid[gy[valid], gx[valid]] = 100

    def get_grid(self) -> Tuple[GridSpec, np.ndarray]:
        # Ensure frame_id matches what RViz expects
        spec = GridSpec(self.res, self.width, self.height, self.origin_x, self.origin_y, "simple_drone/odom")
        return spec, self._grid

    def reset(self):
        self._grid.fill(-1)