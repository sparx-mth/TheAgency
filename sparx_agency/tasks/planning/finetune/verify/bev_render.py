"""Matplotlib drawing helpers for the verification tool (no interaction state).

Every function takes an ``Axes`` and the per-frame data and renders one panel:
the colour image with the clicked goal, the depth image, the bird's-eye field
with both trajectories, and the original-vs-corrected comparison. All BEV panels
are in the body FLU frame: ``x = forward``, ``y = left``, meters, robot at origin.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def path_xy(path) -> np.ndarray:
    """``Path2D`` -> ``(N, 2)`` ``[fwd, left]`` float32 array."""
    return np.array([(p.x, p.y) for p in path.points], dtype=np.float32)


def _grid_extent(occ) -> list:
    """imshow extent ``[x0, x1, y0, y1]`` (forward, left meters) for an OccupancyGrid2D."""
    h, w = occ.grid.shape                       # (n_left, n_fwd)
    res = occ.resolution
    return [occ.origin_x, occ.origin_x + w * res, occ.origin_y, occ.origin_y + h * res]


def draw_image(ax, rgb: np.ndarray, uv: Optional[Tuple[int, int]], title: str) -> None:
    """Show an RGB image and mark the clicked pixel with a red cross."""
    ax.clear()
    ax.imshow(rgb)
    if uv is not None:
        ax.plot(uv[0], uv[1], "x", color="red", ms=14, mew=3)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_depth(ax, depth_m: np.ndarray, uv: Optional[Tuple[int, int]]) -> None:
    """Show a depth image (turbo colormap over valid range) and the clicked pixel."""
    ax.clear()
    d = depth_m.copy()
    d[~np.isfinite(d)] = 0.0
    vmax = float(np.percentile(d[d > 0], 98)) if np.any(d > 0) else 1.0
    ax.imshow(np.ma.masked_where(d <= 0, d), cmap="turbo", vmin=0.0, vmax=vmax)
    if uv is not None:
        ax.plot(uv[0], uv[1], "x", color="white", ms=14, mew=3)
    ax.set_title("depth (click a pixel to set the goal)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_bev(ax, target, goal_xy: Tuple[float, float], field_mode: str = "esdf",
             clearance: float = 0.35):
    """Draw the BEV field + occupancy + NavDP and corrected trajectories.

    Returns the ``imshow`` handle and a colorbar label so the caller can manage a
    shared colorbar.
    """
    ax.clear()
    occ = target.occupancy
    ext = _grid_extent(occ)
    sdf = target.sdf_m
    if field_mode == "repulsion":
        field = np.maximum(clearance - sdf, 0.0)
        im = ax.imshow(field, extent=ext, origin="lower", cmap="hot",
                       vmin=0.0, vmax=max(clearance, 1e-3), aspect="equal", zorder=0)
        label = "repulsion  max(clearance - ESDF, 0)  [m]"
    else:
        im = ax.imshow(np.clip(sdf, -1.0, 2.0), extent=ext, origin="lower",
                       cmap="RdYlBu", vmin=-1.0, vmax=2.0, aspect="equal", zorder=0)
        label = "signed ESDF  (<0 inside wall, >0 free)  [m]"

    occ_mask = occ.grid == occ.values.occupied
    ys, xs = np.where(occ_mask)
    ax.scatter(xs * occ.resolution + occ.origin_x, ys * occ.resolution + occ.origin_y,
               s=3, c="black", alpha=0.45, zorder=1, linewidths=0)

    seed = path_xy(target.seed_path)
    corr = path_xy(target.corrected_path)
    ax.plot(seed[:, 0], seed[:, 1], "o-", color="darkorange", ms=3, lw=2,
            label="NavDP", zorder=3)
    ax.plot(corr[:, 0], corr[:, 1], "o-", color="lime", ms=3, lw=2,
            label="corrected", zorder=4)
    ax.plot(0.0, 0.0, "^", color="cyan", ms=13, zorder=5, label="robot")
    ax.plot(goal_xy[0], goal_xy[1], "*", color="magenta", ms=20, zorder=6, label="goal")

    fwd_max = max(2.5, float(seed[:, 0].max()), float(corr[:, 0].max()), goal_xy[0]) + 0.7
    lat = max(1.5, float(np.abs(seed[:, 1]).max()), float(np.abs(corr[:, 1]).max()),
              abs(goal_xy[1])) + 0.7
    ax.set_xlim(ext[0], min(ext[1], fwd_max))
    ax.set_ylim(max(ext[2], -lat), min(ext[3], lat))
    ax.set_xlabel("forward [m]")
    ax.set_ylabel("left [m]")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    return im, label


def draw_comparison(ax, target, goal_xy: Tuple[float, float]) -> None:
    """Overlay NavDP vs corrected with per-waypoint shift vectors + stats title."""
    ax.clear()
    seed = path_xy(target.seed_path)
    corr = path_xy(target.corrected_path)
    n = min(len(seed), len(corr))
    for i in range(n):
        ax.plot([seed[i, 0], corr[i, 0]], [seed[i, 1], corr[i, 1]],
                "-", color="gray", lw=0.6, zorder=1)
    ax.plot(seed[:, 0], seed[:, 1], "o-", color="darkorange", ms=3, lw=2,
            label="NavDP (original)", zorder=2)
    ax.plot(corr[:, 0], corr[:, 1], "o-", color="lime", ms=3, lw=2,
            label="corrected (target)", zorder=3)
    ax.plot(goal_xy[0], goal_xy[1], "*", color="magenta", ms=16, zorder=4)

    shift = np.linalg.norm(corr[:n] - seed[:n], axis=1)
    ax.set_title("moved %d/%d wp   max shift %.2f m   mean %.2f m"
                 % (int(target.num_moved), n, float(shift.max()), float(shift.mean())),
                 fontsize=9)
    ax.set_xlabel("forward [m]")
    ax.set_ylabel("left [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
