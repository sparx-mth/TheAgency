"""Matplotlib output: 2D occupancy grid + drone trajectory + object markers."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")   # headless — safe on Jetson without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridCostmap
from sparx_agency.demos.Demo_No4_XTEND_MapRoom.room_mapper.object_placer import ObjectMarker

_LABEL_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
]


_OCC_THRESHOLD = 50   # cells below this (p < 0.5) are treated as free


def _largest_free_region(free: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest connected free-space region.

    Discards floating noise fragments outside the main room.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        free.astype(np.uint8), connectivity=4
    )
    if n <= 1:
        return free
    # component 0 is background; find largest non-background component
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest)


def _wall_surface_mask(occ: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Return True for occupied cells that border at least one FREE (white interior) cell.

    Excludes the outer wall face (bordering grey unknown) — keeping only the
    single inner surface line facing the room interior.
    """
    pad = np.pad(free, 1, mode='constant', constant_values=False)
    has_free_neighbor = (
        pad[0:-2, 0:-2] | pad[0:-2, 1:-1] | pad[0:-2, 2:] |
        pad[1:-1, 0:-2] |                    pad[1:-1, 2:] |
        pad[2:,   0:-2] | pad[2:,   1:-1] | pad[2:,   2:]
    )
    return occ & has_free_neighbor


def _build_rgb_image(grid_int8: np.ndarray, convex_walls: bool = True) -> np.ndarray:
    rgb = np.full((*grid_int8.shape, 3), 180, dtype=np.uint8)   # unknown = mid-grey
    occ_full = grid_int8 >= _OCC_THRESHOLD
    free_raw = (grid_int8 >= 0) & ~occ_full
    # Keep only the largest connected free region — removes floating noise fragments
    free = _largest_free_region(free_raw) if np.any(free_raw) else free_raw

    if convex_walls and np.any(free):
        # Compute convex hull of observed free space and draw its boundary as the wall.
        # DA3 smears corners diagonally — the convex hull of the interior is the correct
        # shape for a rectangular room. We skip the original occupied cells entirely to
        # avoid painting their noise inside the hull.
        contours, _ = cv2.findContours(
            free.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            hull = cv2.convexHull(max(contours, key=cv2.contourArea))
            interior = np.zeros(grid_int8.shape, dtype=np.uint8)
            cv2.fillPoly(interior, [hull], 1)
            rgb[interior.astype(bool)] = 245           # white free space
            boundary = np.zeros(grid_int8.shape, dtype=np.uint8)
            cv2.drawContours(boundary, [hull], 0, 1, thickness=1)
            rgb[boundary.astype(bool)] = [30, 30, 30]  # clean dark wall
        else:
            rgb[free] = 245
    else:
        rgb[free] = 245
        occ = _wall_surface_mask(occ_full, free)
        vals = grid_int8[occ].astype(np.float32) / 100.0
        dark = (255 * (1.0 - vals)).astype(np.uint8)
        rgb[occ] = dark[:, None]

    return rgb


def _build_fig(
    grid: LogOddsGridCostmap,
    trajectory_world: List[Tuple[float, float]],
    objects: List[ObjectMarker],
    tag_fixes: Optional[List[Tuple[float, float]]],
    title: str,
    convex_walls: bool = True,
) -> plt.Figure:
    spec, grid_int8 = grid.get_grid()
    x0, y0 = spec.origin_x, spec.origin_y
    x1 = x0 + spec.width  * spec.resolution_m
    y1 = y0 + spec.height * spec.resolution_m

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(_build_rgb_image(grid_int8, convex_walls=convex_walls), origin="lower",
              extent=[x0, x1, y0, y1], aspect="equal")

    legend_handles = []
    if trajectory_world:
        xs, ys = zip(*trajectory_world)
        ax.plot(xs, ys, "b-", linewidth=1.5, zorder=3)
        ax.plot(xs[0], ys[0], "go", markersize=9, zorder=4)
        ax.plot(xs[-1], ys[-1], "r^", markersize=9, zorder=4)
        legend_handles.append(mpatches.Patch(color="blue", label="trajectory"))

    if tag_fixes:
        fx_x, fx_y = zip(*tag_fixes)
        ax.plot(fx_x, fx_y, "y*", markersize=14, zorder=5)
        legend_handles.append(mpatches.Patch(color="yellow", label="AprilTag fix"))

    label_set = sorted({o.label for o in objects})
    color_map = {lbl: _LABEL_COLORS[i % len(_LABEL_COLORS)] for i, lbl in enumerate(label_set)}

    # Stagger annotations that land in the same grid bucket (0.3 m cells)
    bucket_count: dict = {}
    for obj in objects:
        c = color_map[obj.label]
        if obj.suspicious:
            ax.plot(obj.world_x, obj.world_y, "X", color="red", markersize=14,
                    markeredgecolor="darkred", markeredgewidth=1.5, zorder=7)
        else:
            ms = max(7, int(11 * obj.tag_confidence))
            ax.plot(obj.world_x, obj.world_y, "D", color=c, markersize=ms,
                    markeredgecolor="k", markeredgewidth=0.8, zorder=6)
        bucket = (round(obj.world_x / 0.3), round(obj.world_y / 0.3))
        slot = bucket_count.get(bucket, 0)
        bucket_count[bucket] = slot + 1
        xoff = 7 + (slot % 2) * 35
        yoff = 4 + slot * 16
        tag_str = f" t{obj.tag_ids}" if obj.tag_ids else ""
        label_color = "red" if obj.suspicious else c
        prefix = "[!] " if obj.suspicious else ""
        ax.annotate(
            f"{prefix}{obj.label}{tag_str}",
            (obj.world_x, obj.world_y),
            textcoords="offset points", xytext=(xoff, yoff),
            fontsize=7, color=label_color, fontweight="bold" if obj.suspicious else "normal",
            arrowprops=dict(arrowstyle="-", color=label_color, lw=0.6) if slot > 0 else None,
        )
    for lbl in label_set:
        legend_handles.append(mpatches.Patch(color=color_map[lbl], label=lbl))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def save_map_png(
    grid: LogOddsGridCostmap,
    trajectory_world: List[Tuple[float, float]],
    objects: List[ObjectMarker],
    output_path: str,
    title: str = "Room Map",
    tag_fixes: Optional[List[Tuple[float, float]]] = None,
    convex_walls: bool = True,
) -> None:
    fig = _build_fig(grid, trajectory_world, objects, tag_fixes, title,
                     convex_walls=convex_walls)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Saved: {output_path}")


def render_map_rgb(
    grid: LogOddsGridCostmap,
    trajectory_world: List[Tuple[float, float]],
    objects: List[ObjectMarker],
    tag_fixes: Optional[List[Tuple[float, float]]] = None,
    title: str = "Room Map",
    convex_walls: bool = True,
) -> np.ndarray:
    """Render map to HxWx3 uint8 RGB array (for live cv2 preview)."""
    fig = _build_fig(grid, trajectory_world, objects, tag_fixes, title,
                     convex_walls=convex_walls)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img