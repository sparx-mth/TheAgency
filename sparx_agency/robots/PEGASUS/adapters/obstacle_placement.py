"""Choose realistic new obstacle positions inside a scene's surveyed free space.

Pure numpy (+ scipy for connected-components) against an already-surveyed
:class:`OccupancyGrid2D` -- no Isaac import, so it is unit-tested on a laptop.
The Isaac-side prim duplication that actually places geometry at these
coordinates is :mod:`scene_augment`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D


@dataclass(frozen=True)
class Placement:
    """One chosen obstacle position, world frame."""
    x: float
    y: float
    rotation_deg: float


def _disk_offsets(radius_cells: int) -> np.ndarray:
    """``(dy, dx)`` offsets of a filled disk of the given radius, in cells."""
    r = max(int(radius_cells), 0)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    mask = xx * xx + yy * yy <= r * r
    return np.stack([yy[mask], xx[mask]], axis=1)


def _reachable_count(free: np.ndarray, seed_gx: int, seed_gy: int) -> int:
    """Size of the connected component of ``free`` containing ``(seed_gx, seed_gy)``.

    0 if the seed cell itself is not free.
    """
    if not free[seed_gy, seed_gx]:
        return 0
    labels, _ = ndimage.label(free, structure=np.ones((3, 3), dtype=bool))
    return int((labels == labels[seed_gy, seed_gx]).sum())


def sample_placements(
    grid: OccupancyGrid2D,
    landable: np.ndarray,
    count: int,
    min_spacing_m: float,
    keepout: Sequence[Tuple[float, float, float]] = (),
    obstacle_radius_m: float = 0.3,
    max_reachable_loss_fraction: float = 0.02,
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
            takeoff. The first entry, if any, also anchors the connectivity
            check below (it is a point guaranteed to be on the flyable side of
            every corridor).
        obstacle_radius_m: Footprint radius used only to check a candidate spot
            does not sever a passage -- see below. Should be at least half the
            longest axis of the largest object actually being duplicated, so a
            placement that fits a small object is not wrongly assumed safe for
            a big one.
        max_reachable_loss_fraction: How much of the *currently* reachable area
            (from ``keepout[0]``, updated as placements are accepted) a single
            new placement may cut off before it is rejected. This is what
            stops an obstacle from sitting in the one cell wide corridor that
            is the only way to a side room: `min_spacing_m` alone only keeps
            obstacles apart from each other, not off the sole path between two
            areas -- a scene surveyed and confirmed correct still had one
            duplicate that fully sealed a hallway, because nothing had ever
            checked reachability, only spacing.
        rng: Source of randomness. A fresh default generator if omitted --
            pass one explicitly for a reproducible layout.

    Returns:
        Up to ``count`` placements with a random yaw, in the order they were
        accepted. Fewer than ``count`` only when the free, landable area is too
        small or too fragmented to fit them all at the requested spacing
        without blocking a passage.

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

    # Anchor the connectivity check on the spawn point when one is given (it
    # is guaranteed flyable and on the "main" side of the map); otherwise fall
    # back to an arbitrary free cell -- reachability is still checked relative
    # to *something*, just not necessarily the area a flight would start in.
    if keepout:
        seed_gx, seed_gy = grid.world_to_grid(keepout[0][0], keepout[0][1])
    else:
        seed_gx, seed_gy = int(candidate_gx[0]), int(candidate_gy[0])

    radius_cells = max(int(round(obstacle_radius_m / grid.resolution)), 0)
    disk_offsets = _disk_offsets(radius_cells)
    height, width = grid.grid.shape

    working_free = free.copy()
    baseline_reachable = _reachable_count(working_free, seed_gx, seed_gy)

    accepted: List[Placement] = []
    accepted_xy = np.empty((0, 2), dtype=np.float64)
    for idx in order:
        if len(accepted) >= count:
            break
        gx, gy = int(candidate_gx[idx]), int(candidate_gy[idx])
        if not working_free[gy, gx]:
            continue  # already consumed by an earlier placement's footprint
        x, y = grid.grid_to_world(gx, gy)
        if any((x - kx) ** 2 + (y - ky) ** 2 < kr ** 2 for kx, ky, kr in keepout):
            continue
        if accepted_xy.shape[0] > 0:
            distances = np.hypot(accepted_xy[:, 0] - x, accepted_xy[:, 1] - y)
            if np.any(distances < min_spacing_m):
                continue

        footprint_gy = gy + disk_offsets[:, 0]
        footprint_gx = gx + disk_offsets[:, 1]
        in_bounds = ((footprint_gy >= 0) & (footprint_gy < height)
                    & (footprint_gx >= 0) & (footprint_gx < width))
        footprint_gy, footprint_gx = footprint_gy[in_bounds], footprint_gx[in_bounds]

        # The footprint's own cells stop being free regardless of whether this
        # placement fragments anything -- that is not a sign of blocking a
        # passage, just the obstacle occupying its own space. Subtracting it
        # out of the expectation isolates the thing actually being checked:
        # whether cells *outside* the footprint became unreachable too.
        footprint_was_free = int(working_free[footprint_gy, footprint_gx].sum())
        trial_free = working_free.copy()
        trial_free[footprint_gy, footprint_gx] = False
        reachable_after = _reachable_count(trial_free, seed_gx, seed_gy)
        expected_after = baseline_reachable - footprint_was_free
        if reachable_after < expected_after * (1.0 - max_reachable_loss_fraction):
            continue  # would cut off part of the map -- likely the only way through

        working_free = trial_free
        baseline_reachable = reachable_after
        accepted.append(Placement(x=x, y=y, rotation_deg=float(rng.uniform(0.0, 360.0))))
        accepted_xy = np.vstack([accepted_xy, [[x, y]]])

    return accepted
