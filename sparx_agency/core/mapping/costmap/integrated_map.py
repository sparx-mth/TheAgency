import numpy as np
from typing import Set, Tuple
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import bresenham, ProbabilisticGridConfig


class IntegratedMap:
    def __init__(self, cfg: ProbabilisticGridConfig):
        self.cfg = cfg
        self.res = cfg.resolution_m
        self.width = int(np.ceil(cfg.size_m / cfg.resolution_m))
        self.height = int(np.ceil(cfg.size_m / cfg.resolution_m))

        # Internal state (matching your naming)
        self._lo = np.zeros((self.height, self.width), dtype=np.float32)
        self._seen_mask = np.zeros((self.height, self.width), dtype=bool)
        self.voxels_3d: Set[Tuple[int, int, int]] = set()

        self.origin_x = -0.5 * cfg.size_m
        self.origin_y = -0.5 * cfg.size_m

    import numpy as np
    from typing import Set, Tuple
    from sparx_agency.core.mapping.costmap.probabilistic_grid_config import bresenham

    class IntegratedMap:
        def __init__(self, cfg):
            self.cfg = cfg
            self.res = cfg.resolution_m
            self.width = int(np.ceil(cfg.size_m / cfg.resolution_m))
            self.height = int(np.ceil(cfg.size_m / cfg.resolution_m))

            # Internal state matching your ProbabilisticGridCostmap
            self._lo = np.zeros((self.height, self.width), dtype=np.float32)
            self._seen_mask = np.zeros((self.height, self.width), dtype=bool)
            self.voxels_3d: Set[Tuple[int, int, int]] = set()

            self.origin_x = -0.5 * cfg.size_m
            self.origin_y = -0.5 * cfg.size_m

    def update(self, pts_w: np.ndarray, sensor_o: np.ndarray, accumulate: bool):
        if not accumulate:
            self.voxels_3d.clear()
            self._lo.fill(0.0)
            self._seen_mask.fill(False)

        # 1. Map World XZ to Grid XY
        # World X (Left/Right) -> Grid X (gx)
        # World Z (Forward/Depth) -> Grid Y (gy)
        gx = ((pts_w[:, 0] - self.origin_x) / self.res).astype(np.int32)
        gy = ((pts_w[:, 2] - self.origin_y) / self.res).astype(np.int32)

        # World Y is the height (used for obstacle filtering)
        gz = pts_w[:, 1]

        in_bounds = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)

        # 2. Voxel height filtering
        is_obstacle = (gz > self.cfg.min_height_obstacle) & (gz < self.cfg.max_height_obstacle)
        valid_indices = np.where(in_bounds & is_obstacle)[0]

        for i in valid_indices:
            self.voxels_3d.add((int(gy[i]), int(gx[i]), int(gz[i] / self.res)))

        # 3. Raycast from sensor origin (mapping X and Z)
        s_gx = int((sensor_o[0] - self.origin_x) / self.res)
        s_gy = int((sensor_o[2] - self.origin_y) / self.res)

        unique_obs = np.unique(np.stack([gx[in_bounds & is_obstacle], gy[in_bounds & is_obstacle]], axis=1), axis=0)

        for egx, egy in unique_obs:
            line = list(bresenham(s_gx, s_gy, int(egx), int(egy)))
            for cx, cy in line[:-1]:
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    self._lo[cy, cx] += self.cfg.lo_free

            lx, ly = line[-1]
            if 0 <= lx < self.width and 0 <= ly < self.height:
                self._lo[ly, lx] += self.cfg.lo_occ
                self._seen_mask[ly, lx] = True

        self._lo = np.clip(self._lo, self.cfg.lo_min, self.cfg.lo_max)

    def get_viz_data(self):
        """Public accessor for visualization to avoid attribute errors."""
        return self._lo, self._seen_mask, self.voxels_3d