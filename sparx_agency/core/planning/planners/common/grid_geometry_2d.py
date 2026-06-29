"""Grid-space geometry helpers for 2D planning.

These are small, reusable primitives that operate on boolean occupancy /
float cost arrays in grid (cell) coordinates. They are deliberately free of any
world-frame or ROS concepts so they can back any grid planner:

- :func:`dilate_mask` — binary obstacle inflation (4-connected, N iterations).
- :func:`line_of_sight_clear` — Bresenham visibility test between two cells.
- :func:`los_smooth_cells` — greedy any-angle string-pulling post-pass.
- :func:`simplify_path_cells` — Douglas–Peucker reduction that keeps the path
  shape (and its clearance) — the centring-friendly alternative to string-pulling.
- :func:`snap_to_free_cell` — nearest finite-cost cell within a radius.

All functions index arrays as ``arr[y, x]`` (numpy row-major, matching the
``OccupancyGrid2D`` convention) and take/return cells as ``(x, y)``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Inflate a boolean mask by ``iterations`` cells (4-connected).

    Equivalent to an N-step binary dilation with a plus-shaped structuring
    element. Used to grow obstacles by the robot radius before planning.

    Args:
        mask: ``(H, W)`` boolean array (True = set/obstacle).
        iterations: Number of dilation passes (<= 0 returns a copy).

    Returns:
        A new boolean array of the same shape.
    """
    m = mask.copy()
    for _ in range(max(0, int(iterations))):
        o = m.copy()
        o[1:, :] |= m[:-1, :]
        o[:-1, :] |= m[1:, :]
        o[:, 1:] |= m[:, :-1]
        o[:, :-1] |= m[:, 1:]
        m = o
    return m


def line_of_sight_clear(
    occ: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> bool:
    """Bresenham line test: True if no cell on the segment is occupied.

    Args:
        occ: ``(H, W)`` boolean occupancy (True = blocked). Cells are assumed
            in-bounds; callers restrict endpoints to the grid.
        x0, y0: Start cell.
        x1, y1: End cell.

    Returns:
        True iff every traversed cell (inclusive of both endpoints) is free.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if occ[y, x]:
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def los_smooth_cells(cells: Sequence[Cell], occ: np.ndarray) -> List[Cell]:
    """Greedy line-of-sight smoothing (any-angle string-pulling).

    From each kept cell, jump to the farthest later cell still in clear line of
    sight. This removes A*'s grid staircase, leaving only the necessary corners
    where the obstacle layout actually forces a turn.

    Note: string-pulling makes the path *taut*, so it pulls toward the inside of
    corners — undesirable when the route is meant to stay centred in a corridor.
    For that case prefer :func:`simplify_path_cells`, which keeps the A* shape
    (and thus its clearance) and only drops redundant near-collinear points.

    Args:
        cells: Ordered cell path (e.g. an A* result).
        occ: ``(H, W)`` boolean occupancy (True = blocked).

    Returns:
        A reduced list of corner cells; endpoints are preserved.
    """
    if len(cells) <= 2:
        return list(cells)
    out: List[Cell] = [cells[0]]
    i = 0
    n = len(cells)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if line_of_sight_clear(
                occ, cells[i][0], cells[i][1], cells[j][0], cells[j][1]
            ):
                break
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _max_perp_offset(cells: Sequence[Cell], a: int, b: int) -> Tuple[float, int]:
    """Farthest interior cell from the chord ``cells[a]``→``cells[b]``.

    Returns ``(distance_in_cells, index)``; index is ``-1`` when there is no
    interior point. Distance is the perpendicular distance to the chord (or the
    endpoint distance when the chord has zero length).
    """
    (x0, y0), (x1, y1) = cells[a], cells[b]
    dx, dy = float(x1 - x0), float(y1 - y0)
    seg = (dx * dx + dy * dy) ** 0.5
    best_d, best_i = -1.0, -1
    for k in range(a + 1, b):
        px, py = cells[k]
        if seg == 0.0:
            d = ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
        else:
            d = abs(dx * (y0 - py) - dy * (x0 - px)) / seg
        if d > best_d:
            best_d, best_i = d, k
    return best_d, best_i


def simplify_path_cells(
    cells: Sequence[Cell], lethal: np.ndarray, epsilon_cells: float
) -> List[Cell]:
    """Douglas–Peucker path simplification that preserves the A* shape.

    Unlike :func:`los_smooth_cells`, this never moves the route off the original
    polyline by more than ``epsilon_cells``: it only deletes points that are
    near-collinear with their kept neighbours. A staircase along a straight run
    collapses to its endpoints, while genuine corners (a corridor bend, a detour
    around an obstacle) are kept — so a centred A* route stays centred and keeps
    its clearance, just with far fewer waypoints. A segment is additionally never
    collapsed if the resulting chord would cross a lethal cell (safety belt;
    with a sub-robot ``epsilon`` this effectively never fires).

    Args:
        cells: Ordered cell path (consecutive cells adjacent), e.g. an A* result.
        lethal: ``(H, W)`` boolean collision mask (True = blocked).
        epsilon_cells: Max perpendicular deviation, in cells, allowed when
            dropping a point.

    Returns:
        A reduced list of cells; endpoints are preserved.
    """
    n = len(cells)
    if n <= 2:
        return list(cells)
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        dmax, idx = _max_perp_offset(cells, a, b)
        chord_clear = line_of_sight_clear(
            lethal, cells[a][0], cells[a][1], cells[b][0], cells[b][1]
        )
        if idx >= 0 and (dmax > epsilon_cells or not chord_clear):
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return [cells[i] for i in range(n) if keep[i]]


def snap_to_free_cell(
    cost: np.ndarray, x: int, y: int, max_radius: int
) -> Optional[Cell]:
    """Find the nearest cell with finite cost within a Chebyshev radius.

    Searches outward in expanding square rings (radius 1..max_radius) and
    returns the first traversable cell found. Used to relocate a goal that
    landed on an occupied/inflated cell onto the closest free cell.

    Args:
        cost: ``(H, W)`` float cost array (``inf`` = blocked).
        x, y: Centre cell.
        max_radius: Maximum ring radius in cells.

    Returns:
        The nearest free cell ``(x, y)``, or ``None`` if none within radius.
    """
    h, w = cost.shape
    for r in range(1, int(max_radius) + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and np.isfinite(cost[ny, nx]):
                    return nx, ny
    return None
