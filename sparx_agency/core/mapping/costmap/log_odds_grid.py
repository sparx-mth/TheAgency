from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from sparx_agency.core.mapping.interfaces.costmap import Costmap, GridSpec


@dataclass
class LogOddsGridConfig:
    resolution_m: float = 0.3
    size_m: float = 40.0
    frame_id: str = "map"

    # log-odds update parameters
    l_occ: float = 0.85
    l_free: float = -0.40   # currently unused (ray-tracing not implemented)
    l_min: float = -4.0
    l_max: float = 4.0

    # evidence control
    points_per_cell_cap: int = 50     # cap per update (per cell)
    points_to_strong_hit: int = 10    # maps to stronger update

    unknown_value: int = -1


class LogOddsGridCostmap(Costmap):
    """
    Log-odds occupancy grid:
      - marks occupied based on point evidence
      - does NOT raytrace free space (by design, for “1-hour” robustness)
    """

    def __init__(self, cfg: LogOddsGridConfig):
        self.cfg = cfg
        self.width = int(round(cfg.size_m / cfg.resolution_m))
        self.height = int(round(cfg.size_m / cfg.resolution_m))
        self.origin_x = -0.5 * cfg.size_m
        self.origin_y = -0.5 * cfg.size_m

        self._L = np.zeros((self.height, self.width), dtype=np.float32)  # log-odds
        self._seen = np.zeros((self.height, self.width), dtype=np.bool_) # was ever updated?
        self._last_stamp_sec: Optional[float] = None

    def reset(self) -> None:
        self._L.fill(0.0)
        self._seen.fill(False)
        self._last_stamp_sec = None

    def _world_to_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx = ((x - self.origin_x) / self.cfg.resolution_m).astype(np.int32)
        gy = ((y - self.origin_y) / self.cfg.resolution_m).astype(np.int32)
        return gx, gy

    def update_from_points_xy(self, points_xy: np.ndarray, stamp_sec: Optional[float] = None) -> None:
        if points_xy.size == 0:
            return
        self._last_stamp_sec = stamp_sec

        x = points_xy[:, 0]
        y = points_xy[:, 1]
        gx, gy = self._world_to_grid(x, y)

        m = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        gx = gx[m]
        gy = gy[m]
        if gx.size == 0:
            return

        idx = gy * self.width + gx
        binc = np.bincount(idx, minlength=self.width * self.height).astype(np.int32)
        binc = np.minimum(binc, self.cfg.points_per_cell_cap)
        binc = binc.reshape((self.height, self.width))

        hit_mask = binc > 0
        if not np.any(hit_mask):
            return

        # Stronger hit if more points in that cell for this update
        strength = np.clip(binc.astype(np.float32) / float(self.cfg.points_to_strong_hit), 0.2, 1.0)
        dL = self.cfg.l_occ * strength

        self._L[hit_mask] = np.clip(self._L[hit_mask] + dL[hit_mask], self.cfg.l_min, self.cfg.l_max)
        self._seen[hit_mask] = True

    def get_grid(self) -> Tuple[GridSpec, np.ndarray]:
        spec = GridSpec(
            resolution_m=self.cfg.resolution_m,
            width=self.width,
            height=self.height,
            origin_x=float(self.origin_x),
            origin_y=float(self.origin_y),
            frame_id=self.cfg.frame_id,
        )

        grid = np.full((self.height, self.width), self.cfg.unknown_value, dtype=np.int8)

        if np.any(self._seen):
            # Convert log-odds -> probability
            L = self._L
            p = 1.0 / (1.0 + np.exp(-L))
            grid[self._seen] = np.clip(p[self._seen] * 100.0, 0.0, 100.0).astype(np.int8)

        return spec, grid
