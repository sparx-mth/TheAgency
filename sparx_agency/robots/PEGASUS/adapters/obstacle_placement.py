"""Choose realistic new obstacle positions inside a scene's surveyed free space.

Pure numpy against an already-surveyed :class:`OccupancyGrid2D` -- no Isaac
import, so it is unit-tested on a laptop. The Isaac-side prim duplication that
actually places geometry at these coordinates is :mod:`scene_augment`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D


@dataclass(frozen=True)
class Placement:
    """One chosen obstacle position, world frame."""
    x: float
    y: float
    rotation_deg: float


def sample_placements(
    grid: OccupancyGrid2D,
    landable: np.ndarray,
    count: int,
    min_spacing_m: float,
    keepout: Sequence[Tuple[float, float, float]] = (),
    rng: Optional[np.random.Generator] = None,
) -> List[Placement]:
    """Pick up to ``count`` non-overlapping spots on a scene's free, landable floor.

    Args:
        grid: The scene's surveyed 2D occupancy map.
        landable: Boolean mask, same shape as ``grid.grid``, of cells clear to
            the floor (see ``scene_map.LANDABLE_LAYER``). Landable rather than
            merely flyable, so a new obstacle never floats in mid-air over a
            desk it would otherwise hide.
        count: How many positions to return.
        min_spacing_m: Minimum centre-to-centre distance between any two
            returned placements, so duplicates read as separate objects
            instead of a stacked pile.
        keepout: ``(x, y, radius_m)`` circles no placement may fall inside --
            typically the scene's spawn point, so a new obstacle never blocks
            takeoff.
        rng: Source of randomness. A fresh default generator if omitted --
            pass one explicitly for a reproducible layout.

    Returns:
        Up to ``count`` placements with a random yaw, in the order they were
        accepted. Fewer than ``count`` only when the free, landable area is too
        small or too fragmented to fit them all at the requested spacing.

    Raises:
        ValueError: If ``count`` is not positive, ``landable``'s shape does not
            match the grid, or no free landable cell exists at all.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if landable.shape != grid.grid.shape:
        raise ValueError(
            f"landable mask shape {landable.shape} does not match grid shape "
            f"{grid.grid.shape}"
        )

    free = grid.grid == grid.values.free
    candidate_gy, candidate_gx = np.nonzero(free & landable)
    if candidate_gy.size == 0:
        raise ValueError("no free, landable cell in this scene's surveyed map")

    rng = rng if rng is not None else np.random.default_rng()
    order = rng.permutation(candidate_gy.size)

    accepted: List[Placement] = []
    accepted_xy = np.empty((0, 2), dtype=np.float64)
    for idx in order:
        if len(accepted) >= count:
            break
        x, y = grid.grid_to_world(int(candidate_gx[idx]), int(candidate_gy[idx]))
        if any((x - kx) ** 2 + (y - ky) ** 2 < kr ** 2 for kx, ky, kr in keepout):
            continue
        if accepted_xy.shape[0] > 0:
            distances = np.hypot(accepted_xy[:, 0] - x, accepted_xy[:, 1] - y)
            if np.any(distances < min_spacing_m):
                continue
        accepted.append(Placement(x=x, y=y, rotation_deg=float(rng.uniform(0.0, 360.0))))
        accepted_xy = np.vstack([accepted_xy, [[x, y]]])

    return accepted
