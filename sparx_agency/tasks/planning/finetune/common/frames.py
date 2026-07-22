"""Per-frame geometry: a single (depth, intrinsics) frame -> a local body-frame
occupancy grid.

This is the missing glue the fine-tuning label pipeline needs. The repo's mapping
stack builds occupancy on a persistent BEV grid (``PotentialMapper``, egocentric
400x400, numba raycasting). For offline label generation we instead want a small,
self-contained, single-frame local map so we can turn *one* RGB-D observation into
a potential-field / ESDF that shapes a target trajectory.

Frame conventions
-----------------
* **Camera optical** (OpenCV): ``x=right, y=down, z=forward`` -- the frame a
  pinhole back-projection produces.
* **Body FLU** (REP-103, what NavDP / FlowNav trajectories live in):
  ``x=forward, y=left, z=up``. All grids and labels downstream use this frame.

The camera is assumed rigidly mounted at ``camera_height_m`` above the ground and
pitched down by ``pitch_deg`` (nose-down positive). Both default to the XTEND
cruise configuration (1.0 m, 0 deg) but **pitch must be measured on hardware** --
it is not encoded anywhere in the live stack (see the fine-tune README).

ROS-free, numpy-only (cv2/scipy pulled in by the core layers we call). Lives under
``tasks/`` so it may use modern Python; it only *imports* the Python-3.8 core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.environment.occupancy_grid2d import (
    OccupancyGrid2D,
    OccupancyGrid2DParams,
    OccupancyValues,
)

# Semantic cell values shared by the whole fine-tune stack. ``occupied=100`` /
# ``free=0`` / ``unknown=-1`` matches the ROS OccupancyGrid convention the
# correctors expect (``occ_thresh`` is applied on ``value/100``).
OCC_VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


@dataclass(frozen=True)
class LocalMapConfig:
    """Geometry of the single-frame local occupancy map (body FLU frame).

    Attributes:
        resolution_m: Grid cell size in meters.
        forward_extent_m: How far ahead (``+x``) the grid reaches from the robot.
        half_width_m: Lateral half-extent; the grid spans ``[-half, +half]`` in
            ``+y`` (left).
        z_band_m: Absolute height band ``(low, high)`` in meters used only as the
            **fallback** when ``remove_ground_plane`` is off or no ground plane is
            found. Sensitive to ``camera_height_m`` / ``pitch_deg`` being right.
        depth_range_m: Valid depth interval; pixels outside are dropped.
        camera_height_m: Camera height above the ground plane (meters). Only used
            by the absolute-band fallback; the plane fit is height-invariant.
        pitch_deg: Camera pitch, nose-down positive (degrees). Only the absolute
            band depends on it; the plane fit is pitch-invariant.
        stride: Pixel subsampling stride for back-projection (speed lever).
        remove_ground_plane: Fit the ground plane per frame and mark obstacles by
            height *above it* (robust to unknown camera height/pitch, e.g. a low
            drone at takeoff). Falls back to ``z_band_m`` if no ground is visible.
        obstacle_band_m: Height band **above the fitted ground** counted as an
            obstacle (floor and ceiling excluded). Used when the plane fit succeeds.
        ground_fit_thresh_m: Inlier distance (m) for the robust ground-plane fit.
    """

    resolution_m: float = 0.10
    forward_extent_m: float = 8.0
    half_width_m: float = 5.0
    z_band_m: Tuple[float, float] = (0.10, 2.0)
    depth_range_m: Tuple[float, float] = (0.2, 15.0)
    camera_height_m: float = 1.0
    pitch_deg: float = 0.0
    stride: int = 2
    remove_ground_plane: bool = True
    obstacle_band_m: Tuple[float, float] = (0.15, 2.0)
    ground_fit_thresh_m: float = 0.12


def depth_to_body_cloud(
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    config: LocalMapConfig,
) -> np.ndarray:
    """Back-project a metric depth image to a ground-referenced body-FLU cloud.

    Args:
        depth_m: ``(H, W)`` float32 depth in meters; non-positive / non-finite
            pixels are treated as invalid.
        intrinsics: Pinhole intrinsics matching ``depth_m``'s resolution.
        config: Local-map geometry (uses ``depth_range_m``, ``camera_height_m``,
            ``pitch_deg``, ``stride``).

    Returns:
        ``(N, 3)`` float32 array of ``[forward, left, height_above_ground]`` for
        the valid pixels, in meters.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be 2D, got shape {depth.shape}")

    s = max(int(config.stride), 1)
    depth = depth[::s, ::s]
    h, w = depth.shape

    # Intrinsics are for the full-resolution image; rescale for the stride and for
    # any resolution mismatch between the calibration and the actual depth frame.
    sx = w / float(intrinsics.width)
    sy = h / float(intrinsics.height)
    fx = intrinsics.fx * sx
    fy = intrinsics.fy * sy
    cx = intrinsics.cx * sx
    cy = intrinsics.cy * sy

    us = np.arange(w, dtype=np.float32)
    vs = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    z = depth

    valid = (
        np.isfinite(z)
        & (z >= config.depth_range_m[0])
        & (z <= config.depth_range_m[1])
    )
    uu, vv, z = uu[valid], vv[valid], z[valid]

    # Optical frame (x=right, y=down, z=forward).
    x_opt = (uu - cx) * z / fx
    y_opt = (vv - cy) * z / fy
    z_opt = z

    # Camera-FLU (before extrinsics): forward=z, left=-x, up=-y.
    fwd0 = z_opt
    left0 = -x_opt
    up0 = -y_opt

    # Apply the fixed camera pitch (rotation about the +left/y axis). Nose-down
    # positive: a forward ray tilts below the horizon.
    theta = np.deg2rad(config.pitch_deg)
    c, sn = np.cos(theta), np.sin(theta)
    fwd = c * fwd0 + sn * up0
    left = left0
    up = -sn * fwd0 + c * up0

    # Height above the ground plane: camera sits at camera_height_m.
    height = up + config.camera_height_m

    return np.stack([fwd, left, height], axis=1).astype(np.float32)


def _ground_plane_residual(fwd, left, height, thresh):
    """Height of each point above a robustly-fit ground plane, or ``None``.

    Fits ``height ~= a*fwd + b*left + c`` by trimmed least-squares (seeded from the
    lower half of the cloud, then re-fit on inliers). The plane absorbs a wrong
    camera height (its offset ``c``) and pitch/roll (its tilt ``a, b``), so the
    floor maps to ~zero residual regardless of those being right. Returns ``None``
    when there is no dominant, roughly-horizontal ground surface (e.g. a wall-only
    scene) so the caller can fall back to the absolute band.
    """
    if fwd.size < 50:
        return None
    a_mat = np.stack([fwd, left, np.ones_like(fwd)], axis=1)
    inl = height < np.median(height)                 # seed: the lower half by height
    coef = None
    for _ in range(3):                               # trimmed-LS refinement
        if int(inl.sum()) < 30:
            return None
        coef, *_ = np.linalg.lstsq(a_mat[inl], height[inl], rcond=None)
        inl = np.abs(height - a_mat @ coef) < thresh
    # a genuine ground plane is dominant and roughly horizontal (small tilt)
    if inl.mean() < 0.10 or np.hypot(coef[0], coef[1]) > 1.0:
        return None
    return height - a_mat @ coef


def cloud_to_occupancy_grid(
    cloud_body: np.ndarray,
    config: LocalMapConfig,
) -> OccupancyGrid2D:
    """Rasterize a body-FLU cloud into a single-frame occupancy grid.

    Points whose ground-relative height falls inside ``config.z_band_m`` are
    stamped occupied. No ray-tracing is done, so unobserved cells stay ``free``
    (the correctors/ESDF treat unknown as free anyway, and a single frame has
    occlusion that a persistent map would resolve).

    Args:
        cloud_body: ``(N, 3)`` ``[forward, left, height]`` cloud from
            :func:`depth_to_body_cloud`.
        config: Local-map geometry.

    Returns:
        An :class:`OccupancyGrid2D` in the body FLU frame (``x=forward``,
        ``y=left``), origin at the robot, indexed ``grid[gy, gx]``.
    """
    res = config.resolution_m
    n_fwd = int(round(config.forward_extent_m / res))
    n_left = int(round(2.0 * config.half_width_m / res))
    grid = np.full((n_left, n_fwd), OCC_VALUES.free, dtype=np.int16)

    params = OccupancyGrid2DParams(
        resolution=res,
        origin_x=0.0,               # robot at forward=0
        origin_y=-config.half_width_m,  # left spans [-half, +half]
        frame_id="body",
    )

    if cloud_body.size:
        fwd, left, height = cloud_body[:, 0], cloud_body[:, 1], cloud_body[:, 2]
        keep = None
        if config.remove_ground_plane:
            resid = _ground_plane_residual(fwd, left, height, config.ground_fit_thresh_m)
            if resid is not None:                    # obstacles = height above ground
                lo, hi = config.obstacle_band_m
                keep = (resid >= lo) & (resid <= hi)
        if keep is None:                             # fallback: absolute height band
            z_lo, z_hi = config.z_band_m
            keep = (height >= z_lo) & (height <= z_hi)
        gx = np.floor((fwd[keep] - params.origin_x) / res).astype(np.int64)
        gy = np.floor((left[keep] - params.origin_y) / res).astype(np.int64)
        inb = (gx >= 0) & (gx < n_fwd) & (gy >= 0) & (gy < n_left)
        grid[gy[inb], gx[inb]] = OCC_VALUES.occupied

    return OccupancyGrid2D(grid, params, values=OCC_VALUES)


def occupancy_binary(grid: OccupancyGrid2D) -> np.ndarray:
    """Return an ``(H, W)`` uint8 obstacle mask (1 = occupied) for SDF/ESDF."""
    return (grid.grid == grid.values.occupied).astype(np.uint8)


def occupancy_probability(grid: OccupancyGrid2D) -> np.ndarray:
    """Return an ``(H, W)`` float32 occupancy probability in ``[0, 1]``.

    ``occupied -> 1.0``, everything else ``-> 0.0`` (unknown is treated as free,
    matching :class:`EsdfLayer` / the correctors' default).
    """
    return (grid.grid == grid.values.occupied).astype(np.float32)
