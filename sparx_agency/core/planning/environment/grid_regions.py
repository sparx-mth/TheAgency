"""Connected regions of a boolean grid, and the one that is a building's floor.

Two questions keep coming up against a surveyed map and they are the same
computation: *which cells can be reached from here* (mission sampling draws its
start/goal pairs from one connected block, so an unreachable goal is
structurally impossible) and *which cells are the inside of the building*
(exploration coverage has to divide by something, and free space outside the
walls is not part of the answer).

This lives beside the map types rather than beside either consumer, because
those consumers are in different packages and a second copy of a flood fill is
exactly the drift this tree has paid for before. It is also the reason it is
*here* and not under ``planning/planners/common/`` with the other grid
primitives: that package's ``__init__`` imports OMPL, and a process whose whole
job is to close an MP4 file cleanly should not be holding a library that
corrupts the heap at shutdown.

Pure numpy, no scipy, no OpenCV, Python 3.8 syntax -- the same constraints as
everything else under ``core/``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def flood_region(mask, gy, gx, connectivity=8):
    # type: (np.ndarray, int, int, int) -> np.ndarray
    """The set of cells reachable from ``(gy, gx)`` without leaving ``mask``.

    A scanline fill: each pop claims a whole horizontal run at once and only
    seeds the runs above and below it, so the work is proportional to the number
    of cells rather than to the region's diameter. That distinction is not
    academic -- the repeated-dilation form this replaced costs one pass over the
    whole array per cell of diameter, which on a 25 x 59 m building at 5 cm is
    over a thousand passes.

    Args:
        mask: ``(H, W)`` boolean, True where a cell belongs to the region.
            Indexed ``[gy, gx]``, matching :class:`OccupancyGrid2D`.
        gy: Seed row.
        gx: Seed column.
        connectivity: 4 (edges only) or 8 (edges and corners).

    Returns:
        ``(H, W)`` boolean. Empty if the seed is out of bounds or not in
        ``mask``.

    Raises:
        ValueError: ``mask`` is not 2D, or ``connectivity`` is not 4 or 8.
    """
    grid = np.asarray(mask, dtype=bool)
    if grid.ndim != 2:
        raise ValueError("mask must be 2D, got shape %r" % (grid.shape,))
    diagonal = _diagonal(connectivity)
    height, width = grid.shape
    out = np.zeros((height, width), dtype=bool)
    if not (0 <= gy < height and 0 <= gx < width) or not grid[gy, gx]:
        return out

    stack = [(int(gy), int(gx))]  # type: List[Tuple[int, int]]
    while stack:
        y, x = stack.pop()
        if out[y, x] or not grid[y, x]:
            continue
        row, claimed = grid[y], out[y]
        left = x
        while left > 0 and row[left - 1] and not claimed[left - 1]:
            left -= 1
        right = x
        while right + 1 < width and row[right + 1] and not claimed[right + 1]:
            right += 1
        claimed[left:right + 1] = True
        # With 8-connectivity a run also touches the cell diagonally off each
        # end, so the rows above and below are scanned one cell wider.
        lo = max(0, left - 1) if diagonal else left
        hi = min(width - 1, right + 1) if diagonal else right
        for neighbour_y in (y - 1, y + 1):
            if not 0 <= neighbour_y < height:
                continue
            span = grid[neighbour_y, lo:hi + 1] & ~out[neighbour_y, lo:hi + 1]
            found = np.nonzero(span)[0]
            if not found.size:
                continue
            # One seed per contiguous run in the span; the fill spreads from
            # there. Pushing every cell would work and would be slower.
            starts = np.concatenate(([0], np.nonzero(np.diff(found) > 1)[0] + 1))
            for start in starts:
                stack.append((neighbour_y, lo + int(found[start])))
    return out


def connected_regions(mask, connectivity=8):
    # type: (np.ndarray, int) -> List[np.ndarray]
    """Split ``mask`` into its connected components, largest first.

    Args:
        mask: ``(H, W)`` boolean array of cells that belong together.
        connectivity: 4 or 8. 8 is the default because it is what the mission
            sampler has always used; 4 is the right choice for anything asking
            whether a robot could *pass*, since two obstacles touching at a
            corner leave no gap.

    Returns:
        A list of boolean masks, one per component, sorted by descending cell
        count. Empty if nothing is set.
    """
    remaining = np.asarray(mask, dtype=bool).copy()
    regions = []  # type: List[np.ndarray]
    while remaining.any():
        gy, gx = np.argwhere(remaining)[0]
        region = flood_region(remaining, int(gy), int(gx), connectivity)
        regions.append(region)
        remaining &= ~region
    regions.sort(key=lambda r: int(r.sum()), reverse=True)
    return regions


def largest_enclosed_region(mask, connectivity=4):
    # type: (np.ndarray, int) -> Optional[np.ndarray]
    """The biggest component that does not touch the edge of the grid.

    For a surveyed building this is its floor. A ground-truth map computed from
    world geometry has free space on *both* sides of the outer wall, and the
    outside is a component that runs off the edge of the map -- so "the largest
    free component" is not enough on its own, and neither is a seed pose, which
    silently answers a different question if the seed lands in a cupboard.
    Excluding the components that reach the border keeps the building and drops
    the open world around it; taking the largest of what is left drops the
    watertight interiors of closed furniture, which read as free and are
    reachable by nothing.

    Args:
        mask: ``(H, W)`` boolean, True for free space.
        connectivity: 4 or 8. 4 by default: a diagonal pinch between two
            obstacles is not a way through.

    Returns:
        ``(H, W)`` boolean, or ``None`` when every component reaches the border
        -- an open field, or a map cropped inside the walls.
    """
    for region in connected_regions(mask, connectivity):
        if not _touches_border(region):
            return region
    return None


def _touches_border(region):
    # type: (np.ndarray) -> bool
    return bool(region[0].any() or region[-1].any()
                or region[:, 0].any() or region[:, -1].any())


def _diagonal(connectivity):
    # type: (int) -> bool
    if int(connectivity) == 4:
        return False
    if int(connectivity) == 8:
        return True
    raise ValueError("connectivity must be 4 or 8, got %r" % (connectivity,))
