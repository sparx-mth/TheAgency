"""Backproject a depth image into a 2D log-odds occupancy grid."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridCostmap


def _raytrace_free(
    grid: LogOddsGridCostmap,
    cam_world_xy: np.ndarray,   # (2,) camera XY in world frame
    occupied_xy: np.ndarray,    # (N, 2) occupied world XY
    stride: int = 4,            # use every Nth occupied point for rays
) -> None:
    """Draw Bresenham rays from camera to each occupied point, marking cells free."""
    cam_gx, cam_gy = grid._world_to_grid(
        np.array([cam_world_xy[0]]), np.array([cam_world_xy[1]])
    )
    cx_g, cy_g = int(cam_gx[0]), int(cam_gy[0])

    # Skip ray-casting if camera is outside the grid
    if not (0 <= cx_g < grid.width and 0 <= cy_g < grid.height):
        return

    sub_xy = occupied_xy[::stride]
    end_gx, end_gy = grid._world_to_grid(sub_xy[:, 0], sub_xy[:, 1])

    free_mask = np.zeros((grid.height, grid.width), dtype=np.uint8)
    for i in range(len(end_gx)):
        ex, ey = int(end_gx[i]), int(end_gy[i])
        if not (0 <= ex < grid.width and 0 <= ey < grid.height):
            continue
        cv2.line(free_mask, (cx_g, cy_g), (ex, ey), 1, 1)
        free_mask[ey, ex] = 0   # endpoint is occupied, not free

    grid.apply_free_mask(free_mask.astype(bool))


def texture_confidence_mask(
    bgr: np.ndarray,
    target_h: int,
    target_w: int,
    thresh: float = 8.0,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Return a boolean mask (target_h, target_w) — True where texture is sufficient.

    Uses Laplacian gradient magnitude on grayscale RGB.  Low-texture regions
    (white walls, featureless surfaces) are masked out so their unreliable
    DA3 depth doesn't poison the occupancy grid.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    mag = np.abs(lap)
    if blur_ksize > 1:
        mag = cv2.blur(mag, (blur_ksize, blur_ksize))
    mask = (mag > thresh).astype(np.uint8)
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def update_grid_from_depth(
    grid: LogOddsGridCostmap,
    depth_m: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    z_min_world: float = 0.0,
    z_max_world: float = 3.0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 5.0,
    downsample: int = 4,
    stamp_sec: Optional[float] = None,
    raytrace: bool = True,
    raytrace_stride: int = 4,
    confidence_mask: Optional[np.ndarray] = None,
) -> int:
    """
    Backproject depth pixels to world frame, filter by height, update grid XY.

    Camera convention: Z=forward, X=right, Y=down (OpenCV).
    World convention: Z=up (must match the frame of world_T_cam).

    Args:
        raytrace: if True, mark ray cells as free before stamping occupied.
        raytrace_stride: subsample occupied points for ray-casting (performance).

    Returns:
        Number of occupied grid points stamped.
    """
    h, w = depth_m.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    us = np.arange(0, w, downsample, dtype=np.float32)
    vs = np.arange(0, h, downsample, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel().astype(np.int32), vv.ravel().astype(np.int32)

    z = depth_m[vv, uu]
    valid = np.isfinite(z) & (z >= depth_min_m) & (z <= depth_max_m)
    if confidence_mask is not None:
        valid &= confidence_mask[vv, uu]
    z, uu, vv = z[valid], uu[valid], vv[valid]
    if z.size == 0:
        return 0

    X = (uu - cx) * z / fx
    Y = (vv - cy) * z / fy
    pts_cam = np.stack([X, Y, z, np.ones_like(z)])   # (4, N)
    pts_world = (world_T_cam @ pts_cam).T             # (N, 4)

    height_ok = (pts_world[:, 2] >= z_min_world) & (pts_world[:, 2] <= z_max_world)
    xy = pts_world[height_ok, :2]
    if xy.shape[0] == 0:
        return 0

    if raytrace:
        cam_world_xy = world_T_cam[:2, 3]
        _raytrace_free(grid, cam_world_xy, xy, stride=raytrace_stride)

    grid.update_from_points_xy(xy, stamp_sec=stamp_sec)
    return int(xy.shape[0])