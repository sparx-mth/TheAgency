# core/mapping/costmap/distance_field.py
"""
Distance-to-obstacle (clearance) field generation for 2D occupancy grids.

Output convention:
- clearance[y,x] is in METERS
- occupied cells get clearance 0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError as e:  # pragma: no cover
    distance_transform_edt = None  # type: ignore
    _SCIPY_IMPORT_ERROR = str(e)
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True, slots=True)
class DistanceFieldParams:
    """
    Parameters for clearance (distance field) computation.

    Attributes:
        allow_no_scipy: If True and scipy is unavailable, returns zeros.
        dtype: Output dtype (float32 recommended).
    """
    allow_no_scipy: bool = False
    dtype: type = np.float32


def compute_clearance_field(
    occupancy: np.ndarray,
    *,
    resolution: float,
    params: DistanceFieldParams = DistanceFieldParams(),
) -> np.ndarray:
    """
    Compute clearance field (distance to nearest occupied cell) in meters.

    Args:
        occupancy: (H,W) uint8/bool/int. Nonzero treated as occupied.
        resolution: meters per cell.
        params: DistanceFieldParams.

    Returns:
        clearance: np.ndarray (H,W) float, in meters.
    """
    if occupancy.ndim != 2:
        raise ValueError(f"occupancy must be 2D, got shape {occupancy.shape}")
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    occ = (occupancy != 0)
    free = ~occ

    if distance_transform_edt is None:
        if params.allow_no_scipy:
            return np.zeros_like(occupancy, dtype=params.dtype)
        raise ImportError(
            "scipy is required for distance field (scipy.ndimage.distance_transform_edt). "
            f"Import error: {_SCIPY_IMPORT_ERROR}"
        )

    # distance_transform_edt computes distance to the nearest zero element.
    # We want: for each free cell -> distance to nearest occupied cell.
    # So we pass (occupied == 0) ? Actually: use `free` as input where free=True.
    # The EDT returns distance to nearest False (0). So invert:
    # If we pass `free` (True for free), then False occurs at occupied -> distance to occupied.
    dist_cells = distance_transform_edt(free)

    clearance_m = (dist_cells.astype(np.float32, copy=False) * float(resolution)).astype(params.dtype, copy=False)

    # Ensure occupied cells have exactly 0 clearance (numerical hygiene)
    clearance_m[occ] = 0.0
    return clearance_m
