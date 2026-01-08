from __future__ import annotations

from dataclasses import dataclass

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
    points_to_occupied: int = 5      # threshold in a cell to consider occupied
    max_points_cap: int = 50         # cap per-cell counter

    # Filtering / sensing limits
    max_range_m: float = 10.0            # forward 5-10m as you asked
    z_min_m: float = -2.0                # below base (meters)
    z_max_m: float =  2.0                # above base (meters)

    # Frame convention: your cloud from pinhole is usually "optical":
    #   x right, y down, z forward
    # We'll convert to base-style (x forward, y left, z up) before yaw rotation.
    cloud_is_optical: bool = True