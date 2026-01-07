from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from sparx_agency.core.mapping.interfaces.costmap import Costmap, GridSpec
from .probabilistic_grid_config import ProbabilisticGridConfig, bresenham


class ProbabilisticGridCostmap(Costmap):
    def __init__(self, cfg: Optional[ProbabilisticGridConfig] = None):
        self.cfg = cfg or ProbabilisticGridConfig()

        self.width = int(np.ceil(self.cfg.size_m / self.cfg.resolution_m))
        self.height = int(np.ceil(self.cfg.size_m / self.cfg.resolution_m))

        self.origin_x = -0.5 * self.cfg.size_m
        self.origin_y = -0.5 * self.cfg.size_m

        self._lo = np.zeros((self.height, self.width), dtype=np.float32)
        self._counts = np.zeros((self.height, self.width), dtype=np.uint16)

        self._last_stamp_sec: Optional[float] = None

    def reset(self) -> None:
        self._lo.fill(0.0)
        self._counts.fill(0)
        self._last_stamp_sec = None

    def update_from_points_xy(self, points_xy: np.ndarray, stamp_sec: Optional[float] = None) -> None:
        """
        Option A: count points per cell.
        If count >= points_to_occupied -> occupancy evidence increases.
        """
        if points_xy is None or points_xy.size == 0:
            return

        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points_xy must be Nx2")

        # clear per-frame counts
        self._counts.fill(0)

        cx = np.floor((pts[:, 0] - self.origin_x) / self.cfg.resolution_m).astype(np.int32)
        cy = np.floor((pts[:, 1] - self.origin_y) / self.cfg.resolution_m).astype(np.int32)
        m = (cx >= 0) & (cx < self.width) & (cy >= 0) & (cy < self.height)
        cx, cy = cx[m], cy[m]
        if cx.size == 0:
            return

        np.add.at(self._counts, (cy, cx), 1)
        if self.cfg.max_points_cap > 0:
            np.minimum(self._counts, self.cfg.max_points_cap, out=self._counts)

        occ = self._counts >= int(self.cfg.points_to_occupied)
        self._lo[occ] = np.clip(self._lo[occ] + self.cfg.lo_occ, self.cfg.lo_min, self.cfg.lo_max)

        self._last_stamp_sec = stamp_sec

    def get_grid(self) -> Tuple[GridSpec, np.ndarray]:
        """
        Returns (spec, grid_int8) where grid is shape (H,W) int8.
        Values: 0..100, or -1 for unknown.
        """
        spec = GridSpec(
            resolution_m=float(self.cfg.resolution_m),
            width=int(self.width),
            height=int(self.height),
            origin_x=float(self.origin_x),
            origin_y=float(self.origin_y),
            frame_id=self.cfg.frame_id,
        )

        # log-odds -> probability
        p = 1.0 / (1.0 + np.exp(-self._lo))
        grid = (p * 100.0).astype(np.int8)

        # unknown where log-odds still ~0 (no evidence)
        unknown = np.abs(self._lo) < 1e-3
        grid[unknown] = np.int8(self.cfg.unknown_value)

        return spec, grid

    def update_from_cloud_base_raycast(
            self,
            cloud_base: np.ndarray,  # Nx3 in BASE coords (x fwd, y left, z up)
            robot_xy_yaw: tuple[float, float, float],
            stamp_sec: Optional[float] = None,
    ) -> None:
        """
        Option B:
          - raycast from robot cell to each point endpoint
          - cells along ray => free evidence
          - endpoint cell => occupied evidence
        """
        if cloud_base is None or cloud_base.size == 0:
            return

        rx, ry, yaw = robot_xy_yaw
        pts = np.asarray(cloud_base, dtype=np.float32)

        # Filter by z and range in BASE
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        rng = np.sqrt(x * x + y * y)
        m = (
                (z >= self.cfg.z_min_m) & (z <= self.cfg.z_max_m) &
                (rng >= 0.0) & (rng <= self.cfg.max_range_m)
        )
        pts = pts[m]
        if pts.shape[0] == 0:
            return

        # Rolling window: shift grid to keep robot near center (optional)
        if self.cfg.rolling_window:
            new_origin_x = float(rx) - 0.5 * float(self.cfg.size_m)
            # center on robot
            new_origin_y = float(ry) - 0.5 * float(self.cfg.size_m)
            self._shift_to_new_origin(new_origin_x, new_origin_y)

        # Robot cell
        r_cx = int(np.floor((rx - self.origin_x) / self.cfg.resolution_m))
        r_cy = int(np.floor((ry - self.origin_y) / self.cfg.resolution_m))
        if not (0 <= r_cx < self.width and 0 <= r_cy < self.height):
            return

        cyaw = float(np.cos(yaw))
        syaw = float(np.sin(yaw))

        # For speed, sample points (optional): keep every k-th point if needed
        # pts = pts[::2]

        for i in range(pts.shape[0]):
            bx, by = float(pts[i, 0]), float(pts[i, 1])

            # base -> map (2D) using robot pose (x,y,yaw)
            mx = rx + cyaw * bx - syaw * by
            my = ry + syaw * bx + cyaw * by

            e_cx = int(np.floor((mx - self.origin_x) / self.cfg.resolution_m))
            e_cy = int(np.floor((my - self.origin_y) / self.cfg.resolution_m))

            if not (0 <= e_cx < self.width and 0 <= e_cy < self.height):
                continue

            # Trace cells; mark free except final endpoint
            cells = bresenham(r_cx, r_cy, e_cx, e_cy)
            last = None
            for c in cells:
                last = c
                # free update (we will overwrite endpoint later)
                xg, yg = c
                self._lo[yg, xg] = np.clip(self._lo[yg, xg] + self.cfg.lo_free, self.cfg.lo_min, self.cfg.lo_max)

            if last is not None:
                lx, ly = last
                self._lo[ly, lx] = np.clip(self._lo[ly, lx] + self.cfg.lo_occ, self.cfg.lo_min, self.cfg.lo_max)

        self._last_stamp_sec = stamp_sec

    def _shift_to_new_origin(self, new_origin_x: float, new_origin_y: float) -> None:
        """
        Keep grid aligned with world while moving origin.
        We approximate by integer-cell rolling of the log-odds grid.
        """
        res = float(self.cfg.resolution_m)
        dx_cells = int(np.round((self.origin_x - new_origin_x) / res))
        dy_cells = int(np.round((self.origin_y - new_origin_y) / res))

        if dx_cells == 0 and dy_cells == 0:
            self.origin_x = new_origin_x
            self.origin_y = new_origin_y
            return

        self._lo = np.roll(self._lo, shift=(dy_cells, dx_cells), axis=(0, 1))

        # clear newly uncovered regions
        if dy_cells > 0:
            self._lo[:dy_cells, :] = 0.0
        elif dy_cells < 0:
            self._lo[dy_cells:, :] = 0.0

        if dx_cells > 0:
            self._lo[:, :dx_cells] = 0.0
        elif dx_cells < 0:
            self._lo[:, dx_cells:] = 0.0

        self.origin_x = new_origin_x
        self.origin_y = new_origin_y