# core/mapping/costmap/sdf.py
"""
Signed Distance Field for 2D occupancy grids.

Uses scipy.ndimage.distance_transform_edt (C-backed, fast) instead of
skfmm.distance used in MORE — no extra dependency, runs well on edge.

Convention:
    sdf > 0  →  free space  (distance to nearest obstacle)
    sdf = 0  →  obstacle boundary
    sdf < 0  →  inside obstacle (distance to nearest free cell, negated)

All distances are in **meters**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True, slots=True)
class SDFParams:
    """
    Parameters for signed-distance-field computation.

    Attributes:
        clamp_m:  If set, clamp absolute SDF values to this range.
                  Mirrors MORE's `distance_scale` truncation.
        normalize: If True *and* clamp_m is set, rescale clamped SDF
                   to [-1, 1].  Useful as a boundary-cost input
                   (equivalent to MORE's ``1 − sdf/f``).
    """
    clamp_m: Optional[float] = None
    normalize: bool = False

    def __post_init__(self) -> None:
        if self.clamp_m is not None and self.clamp_m <= 0:
            raise ValueError(f"clamp_m must be > 0, got {self.clamp_m}")


def compute_sdf(
    occupancy: np.ndarray,
    resolution: float,
    params: SDFParams = SDFParams(),
) -> np.ndarray:
    """
    Compute a signed distance field from a binary occupancy grid.

    Args:
        occupancy: (H, W) array.  Nonzero → occupied.
        resolution: Meters per cell.
        params: Optional clamping / normalization settings.

    Returns:
        sdf: (H, W) float32 array in meters (or normalized to [-1, 1]).
    """
    if occupancy.ndim != 2:
        raise ValueError(f"occupancy must be 2D, got shape {occupancy.shape}")
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    occ = occupancy != 0
    dist_free = distance_transform_edt(~occ)  # 0 at occupied
    dist_occ = distance_transform_edt(occ)    # 0 at free

    sdf_cells = dist_free - dist_occ
    sdf_m = (sdf_cells * resolution).astype(np.float32)

    if params.clamp_m is not None:
        np.clip(sdf_m, -params.clamp_m, params.clamp_m, out=sdf_m)
        if params.normalize:
            sdf_m /= params.clamp_m

    return sdf_m


def boundary_cost_field(
    occupancy: np.ndarray,
    resolution: float,
    distance_scale_m: float = 3.0,
) -> np.ndarray:
    """
    Produce a [0, 1] cost field analogous to MORE's ``compute_sdf``:
        cost = 1  on / inside obstacles,
        cost → 0  far from obstacles.

    Args:
        occupancy: (H, W) binary grid.
        resolution: Meters per cell.
        distance_scale_m: Max influence distance (MORE's ``distance_scale``).

    Returns:
        cost: (H, W) float32 in [0, 1].
    """
    sdf = compute_sdf(
        occupancy, resolution,
        SDFParams(clamp_m=distance_scale_m, normalize=True),
    )
    cost = 1.0 - np.clip(sdf, 0.0, 1.0)
    return cost