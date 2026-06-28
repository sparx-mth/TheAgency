from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")

try:
    from numba import njit
except Exception as e:
    print(f"[costmap] Numba unavailable, using pure Python fallback: {e}")

    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def update_ray_logodds(L, r0, c0, r1, c1, lo_free, lo_occ):
    """Update log-odds grid L along a ray from (r0,c0) to (r1,c1)."""
    last_r = None
    last_c = None

    for r, c in bresenham(r0, c0, r1, c1):
        last_r, last_c = r, c
        L[r, c] += lo_free  # mark as free for now

    # overwrite endpoint to occupied (undo free + add occ)
    if last_r is not None:
        L[last_r, last_c] += (lo_occ - lo_free)

def bresenham(x0: int, y0: int, x1: int, y1: int):
    """Integer grid traversal from (x0,y0) to (x1,y1) inclusive."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

# @njit
# def fast_update_ray_logodds(L, r0, c0, r1, c1, lo_free, lo_occ, lo_min, lo_max):
#     """Numba-accelerated raycasting (Bresenham logic)."""
#     dx = abs(r1 - r0)
#     dy = abs(c1 - c0)
#     sr = 1 if r0 < r1 else -1
#     sc = 1 if c0 < c1 else -1
#     err = dx - dy
#
#     curr_r, curr_c = r0, c0
#
#     while True:
#         # Update current cell (Free)
#         L[curr_r, curr_c] = max(lo_min, L[curr_r, curr_c] + lo_free)
#
#         if curr_r == r1 and curr_c == c1:
#             break
#
#         e2 = 2 * err
#         if e2 > -dy:
#             err -= dy
#             curr_r += sr
#         if e2 < dx:
#             err += dx
#             curr_c += sc
#
#     # Overwrite endpoint with occupancy
#     L[r1, c1] = min(lo_max, L[r1, c1] + (lo_occ - lo_free))


@njit
def fast_process_endpoints(L, seen, r0, c0, gz, gl, lo_free, lo_occ, lo_min, lo_max):
    """
    Numba kernel to handle all raycasting in one pass.
    Updates both Log-Odds (L) and the 'seen' mask.
    """
    n_pts = gz.shape[0]
    H, W = L.shape

    for i in range(n_pts):
        r1, c1 = gz[i], gl[i]

        # Inline Bresenham for maximum speed
        dx = abs(r1 - r0)
        dy = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dx - dy

        curr_r, curr_c = r0, c0
        while True:
            # Mark visibility and update free-space evidence
            if 0 <= curr_r < H and 0 <= curr_c < W:
                seen[curr_r, curr_c] = True
                L[curr_r, curr_c] = max(lo_min, L[curr_r, curr_c] + lo_free)

            if curr_r == r1 and curr_c == c1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_r += sr
            if e2 < dx:
                err += dx
                curr_c += sc

        # Correct the endpoint (was marked free, now mark as occupied)
        if 0 <= r1 < H and 0 <= c1 < W:
            L[r1, c1] = min(lo_max, L[r1, c1] - lo_free + lo_occ)

@dataclass
class ProbabilisticGridConfig:
    # Map geometry
    resolution_m: float = 0.30
    size_m: float = 100.0                 # 12m x 12m
    frame_id: str = "map"

    # Rolling window (keeps robot near center, but retains history by shifting grid)
    rolling_window: bool = True

    # Evidence model (log-odds)
    lo_occ: float = 1.2                 # how strongly an endpoint increases occupancy
    lo_free: float = -0.80               # how strongly a traversed cell decreases occupancy
    lo_min: float = -4.0
    lo_max: float =  4.0

    # Unknown value for nav_msgs/OccupancyGrid
    unknown_value: int = -1
    # evidence model
    points_to_occupied: int = 10      # threshold in a cell to consider occupied
    max_points_cap: int = 50         # cap per-cell counter
    min_height_obstacle: float = 0.3
    max_height_obstacle: float = 3.5

    # Filtering / sensing limits
    max_range_m: float = 10.0            # forward 5-10m as you asked
    z_min_m: float = -2.0                # below base (meters)
    z_max_m: float =  2.0                # above base (meters)

    # Frame convention: your cloud from pinhole is usually "optical":
    #   x right, y down, z forward
    # We'll convert to base-style (x forward, y left, z up) before yaw rotation.
    cloud_is_optical: bool = True

    debug: bool = False