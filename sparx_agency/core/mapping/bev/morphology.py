"""
Small pure-numpy spatial operators used by the BEV projector: 3D neighbour
counting, 2D border-clamped shift, directional/count wall completion, and a
dependency-free 4-connected dilation.

Kept numpy-only (no scipy) so the whole `bev` package imports cleanly inside
FALCON's container. core/mapping/costmap/inflation.py is the richer,
scipy-based inflation used elsewhere in the stack.
"""
from __future__ import annotations

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