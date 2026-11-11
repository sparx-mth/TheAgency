"""How much unknown floor lies behind a room's frontiers: one figure per room, no double counting.

The exits' ``unseen_m2`` (``opening_nodes``) cast the camera's cone from
each exit's threshold into the unknown and summed the gains, so two
exits onto the same unmapped room counted it twice, and a frontier that
was not an exit -- the far edge of a room the depth range cut short --
counted for nothing. Here every frontier cluster of the room casts the
same cone from its seed cell toward the unknown
(:func:`~opening_nodes.unknown_heading`), the cones are UNITED, and the
room is told once how much unknown floor its frontiers open onto
(:func:`unknown_behind`). With walls not seen yet that unknown is mostly
the rest of the room itself; with every wall seen it lies beyond the
room's exits. Rendered on the oracle's room line as ``frontier=2 (~30m2
unknown behind them)``.

:func:`room_frontier_seeds` is the offline stand-in for the live frontier
inventory, so a recording can be replayed through the same measure.

Python 3.8 syntax; numpy and scipy.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, label as connected_components

from sparx_agency.core.planning.exploration.view_gain import UnknownView
from sparx_agency.core.planning.object_search.rooms.opening_nodes import unknown_heading


def unknown_behind(view: UnknownView, world, cells: Iterable[Tuple[int, int]]) -> Tuple[float, np.ndarray]:
    """``(m2, mask)``: the unknown floor the camera's cone reaches from the frontier ``cells``, united.

    Each cell casts the cone from its centre toward the unknown
    (:func:`~opening_nodes.unknown_heading`); a cell with no readable
    heading casts nothing. The mask is the union over the cells.
    """
    mask = np.zeros(world.grid.shape, dtype=bool)
    for cell in cells:
        gx, gy = int(cell[0]), int(cell[1])
        if not world.in_bounds(gx, gy):
            continue
        heading = unknown_heading(world, (gx, gy))
        if heading is None:
            continue
        mask |= view.unknown_ahead(world.grid_to_world(gx, gy), heading)
    return float(mask.sum()) * view.cell_area_m2, mask


def rooms_unknown_behind(view: UnknownView, world, seeds_by_pid: Dict[int, Sequence[Tuple[int, int]]]) -> Dict[int, float]:
    """``{pid: m2}`` -- :func:`unknown_behind` for every room's frontier seeds."""
    return {int(pid): unknown_behind(view, world, cells)[0] for pid, cells in seeds_by_pid.items()}


def room_frontier_seeds(world, labels: np.ndarray, resolved: Optional[np.ndarray] = None,
                        min_cluster_cells: int = 4) -> Dict[int, List[Tuple[int, int]]]:
    """``{pid: [(gx, gy), ...]}`` -- one seed cell per frontier cluster of every room, read off the map alone.

    A frontier cell is a labelled free cell with an unknown 4-neighbour
    (settled unknown excepted); clusters are 8-connected components of
    at least ``min_cluster_cells`` cells, and the seed is the cluster cell
    nearest its centroid. The offline stand-in for the live inventory's
    goals (``accessible_frontiers``), which also asks for a route.
    """
    grid = world.grid
    unknown = grid == world.values.unknown
    if resolved is not None:
        unknown = unknown & ~np.asarray(resolved, dtype=bool)
    free = (grid == world.values.free) & (labels > 0)
    beside = binary_dilation(unknown, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    frontier = free & beside
    out = {}  # type: Dict[int, List[Tuple[int, int]]]
    if not frontier.any():
        return out
    comp, n = connected_components(frontier, structure=np.ones((3, 3), dtype=bool))
    for k in range(1, n + 1):
        ys, xs = np.nonzero(comp == k)
        if len(xs) < int(min_cluster_cells):
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        nearest = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        pid = int(labels[ys[nearest], xs[nearest]]) - 1
        out.setdefault(pid, []).append((int(xs[nearest]), int(ys[nearest])))
    return out


def coarse_unknown_words(m2: Optional[float], count: int) -> str:
    """The ``frontier=`` value: the count, and in brackets the unknown floor behind the clusters.

    ``2 (~30m2 unknown behind them)``, ``1 (under 1m2 unknown behind it)``;
    the count alone when nothing was measured or there is no frontier.
    """
    count = max(0, int(count))
    if count == 0 or m2 is None or not math.isfinite(float(m2)):
        return "%d" % count
    pronoun = "it" if count == 1 else "them"
    value = float(m2)
    if value < 0.5:
        return "%d (under 1m2 unknown behind %s)" % (count, pronoun)
    return "%d (~%dm2 unknown behind %s)" % (count, int(round(value)), pronoun)
