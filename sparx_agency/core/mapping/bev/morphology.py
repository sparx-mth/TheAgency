"""
Small pure-numpy spatial operators used by the BEV projector: 3D neighbour
counting, 2D border-clamped shift, directional/count wall completion, a
dependency-free 4-connected dilation, and small-component (speck) removal.

Kept numpy-only (no scipy) so the whole `bev` package imports cleanly inside
FALCON's container. core/mapping/costmap/inflation.py is the richer,
scipy-based inflation used elsewhere in the stack.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _shift_add(acc: np.ndarray, src: np.ndarray, dz: int, dy: int, dx: int) -> None:
    """acc += src shifted by (dz,dy,dx), border-clamped (no wrap)."""
    Z, Y, X = src.shape

    def rng(d, n):
        return max(0, -d), n - max(0, d), max(0, d), n - max(0, -d)

    zs0, zs1, zd0, zd1 = rng(dz, Z)
    ys0, ys1, yd0, yd1 = rng(dy, Y)
    xs0, xs1, xd0, xd1 = rng(dx, X)
    acc[zd0:zd1, yd0:yd1, xd0:xd1] += src[zs0:zs1, ys0:ys1, xs0:xs1]


def count_neighbors_3d(occ: np.ndarray, conn: int) -> np.ndarray:
    """Per-voxel count of occupied neighbours (uint8). conn in {6,18,26}."""
    lim = 1 if conn == 6 else (2 if conn == 18 else 3)
    acc = np.zeros(occ.shape, np.uint8)
    occ8 = occ.view(np.uint8)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                l1 = abs(dz) + abs(dy) + abs(dx)
                if 0 < l1 <= lim:
                    _shift_add(acc, occ8, dz, dy, dx)
    return acc


def shift2(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """2D border-clamped shift (no wrap)."""
    out = np.zeros_like(m)
    Y, X = m.shape
    ys0, ys1, yd0, yd1 = (max(0, -dy), Y - max(0, dy), max(0, dy), Y - max(0, -dy))
    xs0, xs1, xd0, xd1 = (max(0, -dx), X - max(0, dx), max(0, dx), X - max(0, -dx))
    out[yd0:yd1, xd0:xd1] = m[ys0:ys1, xs0:xs1]
    return out


def dilate4(mask: np.ndarray, iters: int) -> np.ndarray:
    """4-connected binary dilation by `iters` steps (pure numpy)."""
    m = mask
    for _ in range(iters):
        o = m.copy()
        o[1:, :] |= m[:-1, :]
        o[:-1, :] |= m[1:, :]
        o[:, 1:] |= m[:, :-1]
        o[:, :-1] |= m[:, 1:]
        m = o
    return m


def bridge_fill(occ: np.ndarray, blocked: np.ndarray, *, mode: str,
                n_neighbors: int, iters: int):
    """
    Fill UNKNOWN gaps in walls. Returns (occ_filled, n_filled).

    "directional": fill a cell only if occupied cells bracket it on two
        opposite sides (L&R, U&D, or a diagonal). Closes a one-cell hole in a
        wall line but cannot flood an open room (open cells lack opposite
        support).
    "count": fill a cell with >= n_neighbors occupied 8-neighbours.
    Never fills a `blocked` cell (observed-free or a protected opening).
    """
    if mode == "off" or iters <= 0:
        return occ, 0
    occ = occ.copy()
    n_filled = 0
    for _ in range(iters):
        if mode == "count":
            cnt = (shift2(occ, 0, -1).astype(np.uint8) + shift2(occ, 0, 1)
                   + shift2(occ, -1, 0) + shift2(occ, 1, 0)
                   + shift2(occ, -1, -1) + shift2(occ, 1, 1)
                   + shift2(occ, -1, 1) + shift2(occ, 1, -1))
            cand = cnt >= n_neighbors
        else:  # directional
            cand = ((shift2(occ, 0, -1) & shift2(occ, 0, 1))
                    | (shift2(occ, -1, 0) & shift2(occ, 1, 0))
                    | (shift2(occ, -1, -1) & shift2(occ, 1, 1))
                    | (shift2(occ, -1, 1) & shift2(occ, 1, -1)))
        cand &= ~occ & ~blocked
        if not cand.any():
            break
        n_filled += int(cand.sum())
        occ |= cand
    return occ, n_filled


def _label_components(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Label 4/8-connected occupied components (pure numpy, no scipy).

    Iterative max-propagation: every occupied cell repeatedly takes the largest
    label in its neighbourhood until stable, so each component collapses to a
    single id. connectivity 8 treats diagonal neighbours as connected (an
    L-corner stays ONE component); 4 uses only the orthogonal neighbours.
    Returns an int64 (H, W) label map, 0 on background.
    """
    H, W = mask.shape
    ids = np.arange(1, H * W + 1, dtype=np.int64).reshape(H, W)
    lab = np.where(mask, ids, 0)
    while True:
        nb = lab.copy()
        nb[1:, :] = np.maximum(nb[1:, :], lab[:-1, :])
        nb[:-1, :] = np.maximum(nb[:-1, :], lab[1:, :])
        nb[:, 1:] = np.maximum(nb[:, 1:], lab[:, :-1])
        nb[:, :-1] = np.maximum(nb[:, :-1], lab[:, 1:])
        if connectivity == 8:
            nb[1:, 1:] = np.maximum(nb[1:, 1:], lab[:-1, :-1])
            nb[:-1, :-1] = np.maximum(nb[:-1, :-1], lab[1:, 1:])
            nb[1:, :-1] = np.maximum(nb[1:, :-1], lab[:-1, 1:])
            nb[:-1, 1:] = np.maximum(nb[:-1, 1:], lab[1:, :-1])
        nb[~mask] = 0
        if np.array_equal(nb, lab):
            return lab
        lab = nb


def remove_small_components(mask: np.ndarray, min_size: int,
                           connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Drop occupied connected components smaller than ``min_size`` CELLS (area).

    A raw-area gate: a component survives on total cell count alone. Kept as a
    general tool, but note it treats a compact 2x2 clump (4 cells) as an obstacle
    while dropping a straight 3-cell wall segment -- for "is it a wall?" prefer
    ``remove_non_wall_components`` (a linear-run test). <=1 is a no-op.

    Args:
        mask: (H, W) bool occupied mask. Not mutated.
        min_size: Minimum cells for a component to survive.
        connectivity: 4 or 8.
    Returns:
        (filtered_mask, n_removed_cells).
    """
    if min_size <= 1 or not mask.any():
        return mask, 0
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8, got %r" % connectivity)
    lab = _label_components(mask, connectivity)
    uniq, counts = np.unique(lab[mask], return_counts=True)
    small = uniq[counts < min_size]
    if small.size == 0:
        return mask, 0
    remove = np.isin(lab, small) & mask
    return (mask & ~remove), int(remove.sum())


def remove_non_wall_components(mask: np.ndarray, min_run: int,
                              connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Drop components with no straight run of ``min_run`` consecutive cells.

    A real wall in a BEV is a LINE: ``min_run`` cells in a row along one of the
    four directions (horizontal, vertical, or either diagonal). A noise clump --
    even a compact 2x2 (4 cells) or an L-tromino (3 cells) -- has no such run and
    is culled, while a 3-in-a-row segment or a long diagonal survives. This is a
    shape-aware "is it wall-like?" test, stricter than raw area: it keeps thin
    linear walls and drops blobby specks of equal or greater cell count. It is
    purely spatial, so a stuck phantom the drone can never re-observe free is
    removed regardless.

    ``connectivity`` only groups cells into components (8 keeps an L-corner
    whole so its two arms are judged together); the run test always allows all
    four line directions. A run is detected by ANDing ``min_run`` shifted copies
    of the mask, so a cell survives iff it starts (with its neighbours) a full
    straight segment. <=1 is a no-op.

    Args:
        mask: (H, W) bool occupied mask. Not mutated.
        min_run: Minimum consecutive collinear cells for a component to survive.
        connectivity: 4 or 8.
    Returns:
        (filtered_mask, n_removed_cells).
    """
    if min_run <= 1 or not mask.any():
        return mask, 0
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8, got %r" % connectivity)
    # Cells that START a straight run of length min_run in some direction:
    # AND the mask with itself shifted 1..min_run-1 steps along that direction.
    run_start = np.zeros_like(mask)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        acc = mask.copy()
        for k in range(1, min_run):
            acc &= shift2(mask, -k * dy, -k * dx)
        run_start |= acc
    if not run_start.any():
        return (mask & False), int(mask.sum())
    lab = _label_components(mask, connectivity)
    keep = np.unique(lab[run_start])
    remove = mask & ~np.isin(lab, keep)
    return (mask & ~remove), int(remove.sum())