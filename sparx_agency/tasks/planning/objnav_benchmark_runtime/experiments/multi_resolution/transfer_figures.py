"""Side-by-side view of the coarse map, the sparse prior it became, and the fine result."""
from __future__ import annotations

import numpy as np


def _extent(world):
    left, bottom = world.origin_x, world.origin_y
    return (left, left + world.width * world.resolution, bottom, bottom + world.height * world.resolution)


def _known_box(world, padding_m=2.0):
    ys, xs = np.nonzero(world.grid >= 0)
    if not len(xs):
        return _extent(world)
    return (world.origin_x + xs.min() * world.resolution - padding_m, world.origin_x + (xs.max() + 1) * world.resolution + padding_m,
            world.origin_y + ys.min() * world.resolution - padding_m, world.origin_y + (ys.max() + 1) * world.resolution + padding_m)


def write_transfer_overview(path, panels):
    """panels: (title, world) pairs, drawn on the fine map's known extent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    box = _known_box(panels[-1][1])
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), dpi=130)
    for ax, (title, world) in zip(np.atleast_1d(axes), panels):
        image = np.where(world.grid < 0, 0, np.where(world.grid == 0, 1, 2))
        ax.imshow(image, origin="lower", extent=_extent(world), interpolation="nearest",
                  cmap=ListedColormap(("#aeb7bd", "#f5f3ed", "#252b30")), vmin=0, vmax=2)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("%s (%.2f m cells)" % (title, world.resolution))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
