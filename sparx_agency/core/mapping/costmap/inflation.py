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
from typing import Literal

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt, maximum_filter
except ImportError as e:  # pragma: no cover
    distance_transform_edt = None  # type: ignore
    maximum_filter = None  # type: ignore
    _SCIPY_IMPORT_ERROR = str(e)
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class InflationParams:
    """
    Parameters for obstacle inflation.

    Attributes:
        radius_m: Inflation radius in meters.
        kernel: Structuring element shape.
        allow_no_scipy: If True and scipy is unavailable, inflation becomes a no-op.
    """
    radius_m: float
    kernel: Literal["disk", "square"] = "square"
    allow_no_scipy: bool = False

    def __post_init__(self) -> None:
        if self.radius_m < 0:
            raise ValueError(f"radius_m must be >= 0, got {self.radius_m}")


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

    if maximum_filter is None:
        if params.allow_no_scipy:
            return occ.astype(np.uint8)
        raise ImportError(
            "scipy is required for inflation (scipy.ndimage). "
            f"Import error: {_SCIPY_IMPORT_ERROR}"
        )

    if params.kernel == "square":
        # maximum_filter is separable â†' exact axis-aligned rectangular
        # dilation.  Corners stay sharp, no staircase artifacts.
        size = 2 * radius_cells + 1
        inflated = maximum_filter(occ.astype(np.uint8), size=size) > 0
    else:
        # EDT thresholding â†' exact Euclidean circles (rounded corners).
        dist_to_occ = distance_transform_edt(~occ)
        inflated = dist_to_occ <= radius_cells
    return inflated.astype(np.uint8)
