"""Cross-frame occupancy-map change detection for replan triggering.

ROS-free and 3.8-compatible. The replanning policy snapshots which cells were
*known* (FREE or OCCUPIED, i.e. observed -- not UNKNOWN) at the moment it last
committed a route. On a later frame it compares that snapshot with the live grid
to answer: "how many cells that I had never observed are now observed, AND lie in
the corridor of the route I am flying?". A large count means a real chunk of new,
route-relevant information arrived (e.g. the drone turned 90 deg and revealed a
whole region the path crosses) -- worth re-optimising. A small count is noise and
must NOT trigger a replan (that is the oscillation the policy exists to prevent).
Off-corridor reveals can still open a shorter route, so the caller may ALSO sum
:func:`newly_known_mask` map-wide against a higher floor rather than ignore them.

Both masks must share the grid's shape; the caller guards grid compatibility
(origin/resolution/shape) before diffing -- an index-wise diff is only meaningful
on a world-fixed grid.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D


def known_mask(grid: OccupancyGrid2D) -> np.ndarray:
    """Boolean ``(H, W)`` mask of observed cells (``value != UNKNOWN``).

    FREE and OCCUPIED both count as known; only UNKNOWN (unobserved) is False.
    """
    return grid.grid != grid.values.unknown


def newly_known_mask(prev_known: np.ndarray, grid: OccupancyGrid2D) -> np.ndarray:
    """Cells that were UNKNOWN at the snapshot and are observed now.

    Args:
        prev_known: ``known_mask`` captured at the last commit (same shape).
        grid: Live grid.

    Returns:
        Boolean ``(H, W)`` mask, ``True`` where a cell newly became known.
    """
    return (~prev_known) & known_mask(grid)


def changed_mask(prev_values: np.ndarray, grid: OccupancyGrid2D) -> np.ndarray:
    """Cells whose occupancy VALUE changed since the snapshot.

    A superset of :func:`newly_known_mask`: it also catches FREE<->OCCUPIED flips
    in already-observed space (e.g. new obstacles appearing in mapped free area),
    which a known-mask diff is blind to.

    Args:
        prev_values: Raw grid values (``grid.grid.copy()``) captured at the last
            commit (same shape).
        grid: Live grid.

    Returns:
        Boolean ``(H, W)`` mask, ``True`` where a cell's value changed.
    """
    return prev_values != grid.grid


def count_changed_in_corridor(
    prev_values: np.ndarray, grid: OccupancyGrid2D, corridor: np.ndarray
) -> int:
    """Count changed cells (vs ``prev_values``) that fall inside ``corridor``.

    Args:
        prev_values: Raw grid values captured at the last commit.
        grid: Live grid (same shape as ``prev_values`` / ``corridor``).
        corridor: Boolean route-corridor mask from :func:`path_raster.corridor_mask`.

    Returns:
        Number of route-relevant changed cells.

    Raises:
        ValueError: If the three arrays do not share a shape (the caller must have
            let the grid change lattice without resnapshotting).
    """
    if prev_values.shape != grid.grid.shape or corridor.shape != grid.grid.shape:
        raise ValueError(
            "shape mismatch: prev_values=%r grid=%r corridor=%r"
            % (prev_values.shape, grid.grid.shape, corridor.shape))
    return int((changed_mask(prev_values, grid) & corridor).sum())


def count_new_known_in_corridor(
    prev_known: np.ndarray, grid: OccupancyGrid2D, corridor: np.ndarray
) -> int:
    """Count newly-known cells (vs ``prev_known``) that fall inside ``corridor``.

    Args:
        prev_known: ``known_mask`` at the last commit.
        grid: Live grid (same shape as ``prev_known`` / ``corridor``).
        corridor: Boolean route-corridor mask from :func:`path_raster.corridor_mask`.

    Returns:
        Number of route-relevant newly-observed cells.

    Raises:
        ValueError: If the three arrays do not share a shape (would make the
            index-wise diff meaningless -- the caller must have let the grid
            change size/origin without resnapshotting).
    """
    if prev_known.shape != grid.grid.shape or corridor.shape != grid.grid.shape:
        raise ValueError(
            "shape mismatch: prev_known=%r grid=%r corridor=%r"
            % (prev_known.shape, grid.grid.shape, corridor.shape))
    return int((newly_known_mask(prev_known, grid) & corridor).sum())
