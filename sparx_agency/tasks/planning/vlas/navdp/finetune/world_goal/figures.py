"""Seeing the difference: the same situation, flown by both models, over the map.

A table of paired deltas says whether something improved. It does not say
*what* changed, and for a navigation policy that is usually the more useful
question -- "it stopped cutting the inside of left-hand corners" is actionable,
"min_clear_m +0.07, p=0.001" is not.

Each panel is one held-out sample drawn on the surveyed map: the drone with its
heading, the direction of the goal, and the trajectory each arm produced from
that identical frame. Panels are chosen by how hard the expert route turns,
because a straight corridor looks the same whatever the policy does.

matplotlib only. Runs headless (Agg).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.polyline import to_world
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import Scene

ARM_STYLE = {
    "expert": {"color": "#4caf50", "linewidth": 2.4, "linestyle": "--", "zorder": 3},
    "baseline": {"color": "#eb6834", "linewidth": 2.0, "zorder": 4},
    "trained": {"color": "#2a78d6", "linewidth": 2.0, "zorder": 5},
}
WINDOW_M = 8.0


def _crop(scene: Scene, pose: Sequence[float], window_m: float):
    """Occupancy crop around a pose, plus the extent for ``imshow``."""
    resolution = scene.resolution
    cx, cy = scene.grid.world_to_grid(float(pose[0]), float(pose[1]))
    half = int(window_m / resolution)
    x0, x1 = max(0, cx - half), min(scene.grid.width, cx + half)
    y0, y1 = max(0, cy - half), min(scene.grid.height, cy + half)
    patch = scene.grid.grid[y0:y1, x0:x1]
    image = np.zeros(patch.shape, dtype=np.float32)
    image[patch == scene.grid.values.occupied] = 1.0
    image[patch == scene.grid.values.unknown] = 0.55
    extent = (x0 * resolution + scene.grid.origin_x,
              x1 * resolution + scene.grid.origin_x,
              y0 * resolution + scene.grid.origin_y,
              y1 * resolution + scene.grid.origin_y)
    return image, extent


def _pick(turns: np.ndarray, count: int) -> np.ndarray:
    """Indices spread across the turn range, weighted toward the interesting end."""
    if turns.size <= count:
        return np.arange(turns.size)
    order = np.argsort(turns)
    sharp = order[-max(1, count // 2):]
    rest = order[:-sharp.size]
    spread = rest[np.linspace(0, rest.size - 1, count - sharp.size).astype(int)]
    return np.concatenate([spread, sharp])


def route_panels(out_path, dataset, scenes: Sequence[Scene],
                 arms: Dict[str, List[np.ndarray]], count: int = 12,
                 window_m: float = WINDOW_M) -> Path:
    """Draw ``count`` held-out samples, every arm on the same axes.

    Args:
        out_path: PNG to write.
        dataset: The evaluated :class:`~.dataset.WorldGoalDataset` (for labels).
        scenes: Loaded scenes, indexed by the sample's scene id.
        arms: ``{arm name: [body-frame (T, 2) trajectory per sample]}``, all
            index-aligned with each other and with the dataset.
        count: Panels to draw.
        window_m: Half-width of the map crop, metres.

    Returns:
        The path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    any_arm = next(iter(arms.values()))
    available = len(any_arm)
    turns = dataset.samples["turn_deg"][:available].astype(np.float64)
    chosen = _pick(turns, min(count, available))

    columns = 4
    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.8 * rows))
    flat = np.atleast_1d(axes).ravel()

    for axis, index in zip(flat, chosen):
        pose = dataset.samples["pose"][index].astype(np.float64)
        goal = dataset.samples["goal_world"][index].astype(np.float64)
        scene = scenes[int(dataset.samples["scene"][index])]
        image, extent = _crop(scene, pose, window_m)
        axis.imshow(image, cmap="Greys", origin="lower", extent=extent,
                    vmin=0.0, vmax=1.0, interpolation="nearest")

        for name, paths in arms.items():
            world = to_world(np.asarray(paths[index], dtype=np.float64), pose)
            axis.plot(world[:, 0], world[:, 1], label=name, **ARM_STYLE[name])

        axis.plot([pose[0]], [pose[1]], "o", color="black", markersize=5, zorder=6)
        heading = 1.2 * np.array([np.cos(pose[2]), np.sin(pose[2])])
        axis.arrow(pose[0], pose[1], heading[0], heading[1], width=0.06,
                   color="black", zorder=6, length_includes_head=True)

        direction = goal - pose[:2]
        distance = float(np.hypot(*direction))
        if distance <= window_m * 0.9:
            axis.plot([goal[0]], [goal[1]], "*", color="#d81b60", markersize=13, zorder=7)
        else:
            tip = pose[:2] + direction / distance * (window_m * 0.85)
            axis.plot([tip[0]], [tip[1]], ">", color="#d81b60", markersize=9, zorder=7)

        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal")
        axis.set_title(f"turn {turns[index]:.0f}deg   goal {distance:.1f} m",
                       fontsize=8)
        axis.tick_params(labelsize=6)

    for axis in flat[len(chosen):]:
        axis.axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(arms), fontsize=9)
    figure.suptitle("Held-out test samples: the same frame, flown by each model",
                    fontsize=12)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))

    out_path = Path(out_path)
    figure.savefig(out_path, dpi=130, bbox_inches="tight")
    figure.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    figure.clf()
    return out_path


def coverage_map(out_path, dataset, scene: Scene,
                 split_boxes: Optional[Dict[str, Sequence]] = None) -> Path:
    """Where a split's samples actually are, over the building.

    Worth looking at once: it is the quickest way to see that a split plan drawn
    on paper matches where the aircraft has really flown.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occupied = (scene.grid.grid == scene.grid.values.occupied).astype(np.float32)
    extent = (scene.bounds[0], scene.bounds[2], scene.bounds[1], scene.bounds[3])
    figure, axis = plt.subplots(figsize=(5.5, 11))
    axis.imshow(occupied, cmap="Greys", origin="lower", extent=extent, vmin=0, vmax=1.4)
    poses = dataset.samples["pose"]
    axis.scatter(poses[:, 0], poses[:, 1], s=1.5, alpha=0.25, color="#2a78d6",
                 label=f"{dataset.split} anchors")
    axis.set_aspect("equal")
    axis.legend(fontsize=8, loc="upper right")
    axis.set_title(f"{scene.name}: {dataset.split} split coverage", fontsize=10)
    figure.tight_layout()
    out_path = Path(out_path)
    figure.savefig(out_path, dpi=130, bbox_inches="tight")
    figure.clf()
    return out_path
