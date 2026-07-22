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
    """Log-odds occupancy grid with optional ray-cast free-space marking."""

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

    def apply_free_mask(self, mask: np.ndarray) -> None:
        """Decrement log-odds for every True cell in mask (ray-cast free space)."""
        self._L[mask] = np.clip(self._L[mask] + self.cfg.l_free, self.cfg.l_min, self.cfg.l_max)
        self._seen[mask] = True

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

    def get_prob(self, unknown: float = np.nan) -> np.ndarray:
        """
        Returns p_occ in [0,1] with unknown as NaN.
        """
        p = 1.0 / (1.0 + np.exp(-self._L))
        out = p.astype(np.float32, copy=False)
        if unknown is not None:
            out = out.copy()
            out[~self._seen] = np.float32(unknown)
        return out

    def extract_window(self, center_x: float, center_y: float, size_m: float) -> tuple[GridSpec, np.ndarray]:
        """
        Extract a square window around (center_x, center_y), returns (spec, p_occ_window).
        """
        half = 0.5 * float(size_m)

        x0 = center_x - half
        y0 = center_y - half
        x1 = center_x + half
        y1 = center_y + half

        gx0, gy0 = self._world_to_grid(np.array([x0]), np.array([y0]))
        gx1, gy1 = self._world_to_grid(np.array([x1]), np.array([y1]))

        gx0 = int(np.clip(gx0[0], 0, self.width))
        gy0 = int(np.clip(gy0[0], 0, self.height))
        gx1 = int(np.clip(gx1[0], 0, self.width))
        gy1 = int(np.clip(gy1[0], 0, self.height))

        # ensure ordering
        if gx1 < gx0: gx0, gx1 = gx1, gx0
        if gy1 < gy0: gy0, gy1 = gy1, gy0

        p = self.get_prob()
        win = p[gy0:gy1, gx0:gx1].copy()

        spec = GridSpec(
            resolution_m=self.cfg.resolution_m,
            width=win.shape[1],
            height=win.shape[0],
            origin_x=self.origin_x + gx0 * self.cfg.resolution_m,
            origin_y=self.origin_y + gy0 * self.cfg.resolution_m,
            frame_id=self.cfg.frame_id,
        )
        return spec, win

    def update_from_cloud(self, cloud_xyz: np.ndarray, sensor_origin: np.ndarray) -> None:
        """
        Satisfies Costmap ABC.

        This is a hit-only integration (no raytrace):
          - uses XY of points
          - classifies occupied by Z band (configurable)
        """
        if cloud_xyz is None or cloud_xyz.shape[0] == 0:
            return

        pts = np.asarray(cloud_xyz, dtype=np.float32).reshape((-1, 3))
        if pts.shape[0] == 0:
            return

        # Optional: filter invalid
        m = np.isfinite(pts).all(axis=1)
        pts = pts[m]
        if pts.shape[0] == 0:
            return

        # Decide occupied based on height band (you can tune these in your cfg)
        # If your config already has names, use them. Otherwise add defaults.
        z = pts[:, 2]
        z_min = getattr(self.cfg, "min_height_obstacle", -0.2)
        z_max = getattr(self.cfg, "max_height_obstacle", 1.5)
        is_occ = (z >= z_min) & (z <= z_max)

        # If your class already has update_from_points_xy(x, y, is_occ, stamp_sec)
        # use that. Otherwise implement update via your existing logic.
        if hasattr(self, "update_from_points_xy"):
            # signature could vary; try the common one
            try:
                self.update_from_points_xy(pts[:, 0], pts[:, 1], is_occ)
            except TypeError:
                # fallback if your method wants (N,2)
                xy = pts[:, :2]
                self.update_from_points_xy(xy)
        else:
            # Minimal fallback: if you store _lo/_seen like the other costmaps,
            # you should implement the same internal update here.
            raise NotImplementedError(
                "LogOddsGridCostmap has no update_from_points_xy; add it or implement update here.")
