import numpy as np
from typing import Set, Tuple
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import (
    fast_process_endpoints,
    ProbabilisticGridConfig,
)


class IntegratedMap:
    def __init__(self, cfg: ProbabilisticGridConfig):
        self.cfg = cfg
        self.res = cfg.resolution_m
        self.width  = int(np.ceil(cfg.size_m / cfg.resolution_m))
        self.height = int(np.ceil(cfg.size_m / cfg.resolution_m))

        self._lo        = np.zeros((self.height, self.width), dtype=np.float32)
        self._seen_mask = np.zeros((self.height, self.width), dtype=bool)

        self.origin_x = -0.5 * cfg.size_m
        self.origin_y = -0.5 * cfg.size_m

    def update(self, pts_w: np.ndarray, sensor_o: np.ndarray, accumulate: bool):
        if not accumulate:
            self._lo.fill(0.0)
            self._seen_mask.fill(False)

        # Body/world frame: X=forward, Y=left, Z=up.
        # Project onto the XY ground plane; filter by Z height.
        gx = ((pts_w[:, 0] - self.origin_x) / self.res).astype(np.int32)  # col
        gy = ((pts_w[:, 1] - self.origin_y) / self.res).astype(np.int32)  # row
        gz =   pts_w[:, 2]                                                  # height

        in_bounds   = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        is_obstacle = (gz > self.cfg.min_height_obstacle) & (gz < self.cfg.max_height_obstacle)
        valid       = in_bounds & is_obstacle

        if not np.any(valid):
            return

        # Sensor grid position (row, col) — clamp to map so raycasting stays in bounds
        s_row = int(np.clip((sensor_o[1] - self.origin_y) / self.res, 0, self.height - 1))
        s_col = int(np.clip((sensor_o[0] - self.origin_x) / self.res, 0, self.width  - 1))

        # Unique obstacle cells in (col, row) order, then split to row/col arrays for kernel
        unique_obs = np.unique(
            np.stack([gx[valid], gy[valid]], axis=1), axis=0
        )
        ep_rows = unique_obs[:, 1].astype(np.int32)  # gy → row
        ep_cols = unique_obs[:, 0].astype(np.int32)  # gx → col

        fast_process_endpoints(
            self._lo, self._seen_mask,
            np.int32(s_row), np.int32(s_col),
            ep_rows, ep_cols,
            self.cfg.lo_free, self.cfg.lo_occ,
            self.cfg.lo_min,  self.cfg.lo_max,
        )

    def get_viz_data(self):
        return self._lo, self._seen_mask, set()
