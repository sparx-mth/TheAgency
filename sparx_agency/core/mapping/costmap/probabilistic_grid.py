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

    def update_from_cloud_cam_raycast(
            self,
            cloud_cam: np.ndarray,  # Nx3 in optical cam convention
            pose_xy_yaw: tuple[float, float, float],
            stamp_sec: Optional[float] = None,
    ) -> None:
        """
        Option B:
          - Convert optical cloud to ground-plane points relative to robot
          - Ray-cast from robot cell to endpoint cell:
                traversed cells -> free evidence
                endpoint cell   -> occupied evidence
        pose_xy_yaw is robot base in map frame.
        """
        if cloud_cam is None or cloud_cam.size == 0:
            return

        x0_map, y0_map, yaw = pose_xy_yaw

        pts = np.asarray(cloud_cam, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("cloud_cam must be Nx3")

        # Optical -> base-ish convention.
        # optical: x right, y down, z forward
        # base ground: X forward, Y left, Z up
        if self.cfg.cloud_is_optical:
            X = pts[:, 2]  # forward
            Y = -pts[:, 0]  # left
            Z = -pts[:, 1]  # up
        else:
            X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

        # Filter by height and range
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        if self.cfg.z_min_m is not None:
            m &= (Z >= float(self.cfg.z_min_m))
        if self.cfg.z_max_m is not None:
            m &= (Z <= float(self.cfg.z_max_m))

        # Use forward range (X) + lateral (Y) to compute planar range
        r = np.sqrt(X * X + Y * Y)
        m &= (r > 0.05) & (r <= float(self.cfg.max_range_m))
        X, Y = X[m], Y[m]
        if X.size == 0:
            return

        # Rotate robot-relative points into map frame using yaw, then translate
        c = math.cos(yaw)
        s = math.sin(yaw)
        px_map = x0_map + (c * X - s * Y)
        py_map = y0_map + (s * X + c * Y)

        # Convert robot origin to grid cell
        gx0 = int(np.floor((x0_map - self.origin_x) / self.cfg.resolution_m))
        gy0 = int(np.floor((y0_map - self.origin_y) / self.cfg.resolution_m))
        if gx0 < 0 or gx0 >= self.width or gy0 < 0 or gy0 >= self.height:
            # robot outside grid
            return

        # For performance: sample endpoints (optional)
        # You can thin points here if needed:
        # px_map = px_map[::3]; py_map = py_map[::3]

        # Ray-cast for each endpoint
        for x1, y1 in zip(px_map, py_map):
            gx1 = int(np.floor((x1 - self.origin_x) / self.cfg.resolution_m))
            gy1 = int(np.floor((y1 - self.origin_y) / self.cfg.resolution_m))
            if gx1 < 0 or gx1 >= self.width or gy1 < 0 or gy1 >= self.height:
                continue

            cells = list(bresenham(gx0, gy0, gx1, gy1))
            if len(cells) == 0:
                continue

            # Free along the ray excluding endpoint
            for gx, gy in cells[:-1]:
                self._lo[gy, gx] = np.clip(self._lo[gy, gx] + self.cfg.lo_free,
                                           self.cfg.lo_min, self.cfg.lo_max)

            # Occupied at endpoint
            ex, ey = cells[-1]
            self._lo[ey, ex] = np.clip(self._lo[ey, ex] + self.cfg.lo_occ,
                                       self.cfg.lo_min, self.cfg.lo_max)

        self._last_stamp_sec = stamp_sec
