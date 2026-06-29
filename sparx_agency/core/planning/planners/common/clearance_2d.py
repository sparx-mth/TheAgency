"""Euclidean clearance (distance-to-obstacle) field for 2D grids — scipy-free.

This is the single primitive behind the weighted A* planner's wall-avoidance
and corridor-centering behaviour: it turns a boolean obstacle mask into a float
field giving, for every cell, the Euclidean distance to the nearest obstacle (in
meters). The medial axis of a corridor is exactly the locus of *maximum*
clearance, so a cost that decreases with clearance pulls the planned route to the
middle of the corridor.

Why a hand-rolled transform instead of ``scipy.ndimage.distance_transform_edt``:
the FALCON ROS1/Noetic adapter imports ``core`` under Python 3.8 with **numpy
1.17 and no scipy** (see ``tasks/planning/falcon/run_falcon.sh``). A scipy import
on the planner path would crash that node. So this module uses only numpy and
stays Python-3.8 compatible. (``core.mapping.costmap.distance_field`` keeps the
scipy version for the host-only mapping stack.)

Algorithm: the exact separable squared Euclidean distance transform (Saito &
Toriwaki / Felzenszwalb–Huttenlocher), run as two 1-D minimisations — first down
columns, then across rows. Both passes are **bounded** to the clearance band we
actually care about (``max_clearance_m``), which keeps the work to a handful of
vectorised ``np.minimum`` calls and the result exact for every true distance up
to that cap (larger distances are clamped — the cost layer treats them all as
"open" anyway). Off-grid cells are treated as free (no phantom wall at the map
border), matching the rest of the planner.
"""
from __future__ import annotations

from math import ceil

import numpy as np


def clearance_field(
    occupied: np.ndarray, resolution: float, max_clearance_m: float
) -> np.ndarray:
    """Distance from every cell to the nearest obstacle, in meters.

    Args:
        occupied: ``(H, W)`` boolean mask (True = obstacle).
        resolution: Meters per cell (> 0).
        max_clearance_m: Distances are computed exactly up to this value and
            clamped above it. Pick the largest clearance the cost layer cares
            about (lethal radius + soft band); a smaller cap is faster.

    Returns:
        ``(H, W)`` float64 array of clearances in meters. Obstacle cells are
        ``0.0``; cells farther than ``max_clearance_m`` from any obstacle are
        clamped to a value ``>= max_clearance_m``.
    """
    if occupied.ndim != 2:
        raise ValueError(f"occupied must be 2D, got shape {occupied.shape}")
    if resolution <= 0.0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    h, w = occupied.shape
    occ = occupied.astype(bool, copy=False)

    # Work in squared cell units to stay exact and avoid sqrt until the end.
    cap_cells = int(ceil(max(max_clearance_m, 0.0) / resolution)) + 1
    big = float((cap_cells + 1) ** 2)  # squared-distance sentinel ("no obstacle in band")
    seed = np.where(occ, 0.0, big)     # 0 at obstacles, "infinite" elsewhere

    # Pass 1 — exact bounded 1-D squared EDT down each column.
    # g[y,x] = min over |s|<=cap of seed[y+s, x] + s^2 (vertical sq-distance).
    g = seed.copy()
    for s in range(1, min(cap_cells, h - 1) + 1):
        s2 = float(s * s)
        np.minimum(g[s:, :], seed[:-s, :] + s2, out=g[s:, :])   # obstacle above
        np.minimum(g[:-s, :], seed[s:, :] + s2, out=g[:-s, :])  # obstacle below

    # Pass 2 — bounded 1-D squared EDT across each row of the column result.
    # d[y,x] = min over |s|<=cap of g[y, x+s] + s^2 = exact squared EDT.
    d = g.copy()
    for s in range(1, min(cap_cells, w - 1) + 1):
        s2 = float(s * s)
        np.minimum(d[:, s:], g[:, :-s] + s2, out=d[:, s:])      # obstacle left
        np.minimum(d[:, :-s], g[:, s:] + s2, out=d[:, :-s])     # obstacle right

    np.minimum(d, big, out=d)
    clearance = np.sqrt(d) * float(resolution)
    clearance[occ] = 0.0  # numerical hygiene: obstacles are exactly zero
    return clearance
