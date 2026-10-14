"""Carry a coarse occupancy map into a finer grid as a sparse prior.

Each known coarse cell is written into exactly one fine cell; the rest stay
unknown, so the finer run still has frontiers to refine.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.environment.occupancy_grid2d import (
    OccupancyGrid2D,
    OccupancyGrid2DParams,
)

REPRESENTATIVES = ("center", "random")
_EDGE_TOLERANCE_DECIMALS = 9


def _fine_span(coarse_index, ratio):
    """Fine cells whose centres fall inside each coarse cell: [lo, hi)."""
    edges = np.round(np.stack([coarse_index, coarse_index + 1]) * ratio - 0.5, _EDGE_TOLERANCE_DECIMALS)
    lo, hi = np.ceil(edges).astype(np.int64)
    return lo, hi


def _representative_index(coarse_index, ratio, representative, rng):
    lo, hi = _fine_span(coarse_index, ratio)
    if representative == "random":
        return lo + np.floor(rng.random(len(lo)) * (hi - lo)).astype(np.int64)
    center = np.floor(np.round((coarse_index + 0.5) * ratio, _EDGE_TOLERANCE_DECIMALS)).astype(np.int64)
    return np.clip(center, lo, hi - 1)


def sparsify_to_fine(coarse: OccupancyGrid2D, fine_resolution: float, representative: str = "center",
                     seed: int = 0) -> OccupancyGrid2D:
    """Fine grid over the coarse extent with one seeded cell per known coarse cell."""
    if representative not in REPRESENTATIVES:
        raise ValueError("representative must be one of %s, got %r" % (REPRESENTATIVES, representative))
    ratio = coarse.resolution / fine_resolution
    if ratio < 1.0:
        raise ValueError("fine_resolution %.3f is coarser than the source %.3f" % (fine_resolution, coarse.resolution))
    shape = (int(np.ceil(round(coarse.height * ratio, _EDGE_TOLERANCE_DECIMALS))),
             int(np.ceil(round(coarse.width * ratio, _EDGE_TOLERANCE_DECIMALS))))
    fine = np.full(shape, coarse.values.unknown, dtype=np.int16)
    ys, xs = np.nonzero(coarse.grid != coarse.values.unknown)
    rng = np.random.default_rng(seed)
    fy = _representative_index(ys, ratio, representative, rng)
    fx = _representative_index(xs, ratio, representative, rng)
    fine[fy, fx] = coarse.grid[ys, xs]
    params = OccupancyGrid2DParams(fine_resolution, coarse.origin_x, coarse.origin_y, coarse.frame_id)
    return OccupancyGrid2D(fine, params, values=coarse.values)
