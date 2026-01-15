# core/mapping/costmap/inflation.py
"""
Obstacle inflation (safety margin) for 2D occupancy grids.

Input/Output convention:
- occupancy == 0 -> free
- occupancy == 1 -> occupied
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal, Optional

import numpy as np

try:
    from scipy.ndimage import binary_dilation
except ImportError as e:  # pragma: no cover
    binary_dilation = None  # type: ignore
    _SCIPY_IMPORT_ERROR = str(e)
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True, slots=True)
class InflationParams:
    """
    Parameters for obstacle inflation.

    Attributes:
        radius_m: Inflation radius in meters.
        kernel: Structuring element shape.
        allow_no_scipy: If True and scipy is unavailable, inflation becomes a no-op.
    """
    radius_m: float
    kernel: Literal["disk", "square"] = "disk"
    allow_no_scipy: bool = False

    def __post_init__(self) -> None:
        if self.radius_m < 0:
            raise ValueError(f"radius_m must be >= 0, got {self.radius_m}")


def _make_kernel(radius_cells: int, shape: str) -> np.ndarray:
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=bool)

    if shape == "square":
        k = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
        return k

    # disk (approx circle)
    yy, xx = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    mask = (xx * xx + yy * yy) <= (radius_cells * radius_cells)
    return mask.astype(bool)


def inflate_occupancy(
    occupancy: np.ndarray,
    *,
    resolution: float,
    params: InflationParams,
) -> np.ndarray:
    """
    Inflate occupied cells by `params.radius_m`.

    Args:
        occupancy: (H,W) uint8/bool/int. Nonzero treated as occupied.
        resolution: meters per cell.
        params: InflationParams.

    Returns:
        Inflated occupancy as np.uint8 (H,W), values in {0,1}.
    """
    if occupancy.ndim != 2:
        raise ValueError(f"occupancy must be 2D, got shape {occupancy.shape}")
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    occ = (occupancy != 0)

    if params.radius_m <= 0:
        return occ.astype(np.uint8)

    radius_cells = int(ceil(params.radius_m / resolution))

    if binary_dilation is None:
        if params.allow_no_scipy:
            # No-op fallback
            return occ.astype(np.uint8)
        raise ImportError(
            "scipy is required for inflation (scipy.ndimage.binary_dilation). "
            f"Import error: {_SCIPY_IMPORT_ERROR}"
        )

    kernel = _make_kernel(radius_cells, params.kernel)
    inflated = binary_dilation(occ, structure=kernel)
    return inflated.astype(np.uint8)
