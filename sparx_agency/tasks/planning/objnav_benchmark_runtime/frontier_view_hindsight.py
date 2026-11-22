"""What the chosen frames showed, measured afterwards: the space behind a node that was mapped later, and what waiting would have bought.

Two questions about a selection that cannot be answered when it is made,
and can be from the rest of the recording. Both are asked of every scoring
of a replay (``frontier_view_replay.ScoringEvent``) and written onto it.

**In hindsight** (:func:`add_hindsight`). The quantity a selector is after
is how much of the space behind the boundary the chosen frames show. The
unknown within :data:`BEHIND_M` of the node (grown from its cells through
unknown only, so not across a wall) is taken at the scoring action; the
part of it the episode went on to map is looked up in the FINAL map; and
every frame logged up to the scoring action is asked whether it had a
clear line to each such cell, on the final map, inside its picture. The
colour image has no range limit, so this is what the frame could show
whether or not its depth resolved it. Three shares of that later-mapped
space come out:

* ``shown`` -- by the frames the variant chose;
* ``available`` -- by any frame logged so far (no selector can beat it);
* ``best_frame`` -- by the single best frame for this measure;

and ``current_frame``, by the frame of the scoring action alone.
``shown / available`` is how much of what there was to show the selector
found. ``available`` is the sobering one: where it is next to nothing, the
space behind the boundary was in no picture at all -- which is why it was
still unknown -- and whatever is asked about the node is asked about what
stands in front of it.

The sight line here is the direct one, without the selector's slack round a
corner: this measures what a frame COULD show, and a cell's slack would
credit every frame with the first strip behind the jamb it looks along.

**What waiting would buy** (:func:`add_patience`). A node is scored the
first action it has a usable view, which is nearly always the action it
appears, on the frame that made it. The same node on the LAST action it
was present is offered every frame logged until then; its views there,
beside its views at scoring, say what the score-once rule gives up by
asking at once.

Python 3.9 syntax, numpy and scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from sparx_agency.core.planning.object_search.frontier_views.selection import select_views, storey_frames
from sparx_agency.core.planning.object_search.frontier_views.settings import ViewSettings
from sparx_agency.core.planning.object_search.frontier_views.sight import SEEN, SightScene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_extent_replay import VALUES

#: How far behind a node the unknown is taken to be "the space behind it", metres, grown through unknown only.
BEHIND_M = 2.0
#: How far a frame is asked about that space in hindsight: the colour image's reach, not the depth's.
HINDSIGHT_RANGE_M = 12.0
#: Fewest later-mapped cells behind a node for its hindsight shares to be counted (a fifth of a square metre).
MIN_REVEALED_CELLS = 20


def behind_cells(grid, settled, node, reach_cells: int):
    """``(rows, cols)`` of the unsettled unknown within ``reach_cells`` of ``node``, reached through unknown alone."""
    height, width = grid.shape
    pad = int(reach_cells) + 2
    y0, y1 = max(0, int(node.rows.min()) - pad), min(height, int(node.rows.max()) + pad + 1)
    x0, x1 = max(0, int(node.cols.min()) - pad), min(width, int(node.cols.max()) + pad + 1)
    dark = (grid[y0:y1, x0:x1] == VALUES.unknown) & ~np.asarray(settled[y0:y1, x0:x1], dtype=bool)
    seed = np.zeros(dark.shape, dtype=bool)
    seed[node.rows - y0, node.cols - x0] = True
    grown = binary_dilation(seed, structure=np.ones((3, 3), dtype=bool), iterations=int(reach_cells), mask=dark | seed)
    rows, cols = np.nonzero(grown & dark)
    return rows + y0, cols + x0


def add_hindsight(replayed, reach_m: float = BEHIND_M, range_m: float = HINDSIGHT_RANGE_M) -> None:
    """Fill ``event.hindsight`` of every scoring of every variant; see the module docstring.

    ``{"behind_cells", "revealed_cells", "cells"}`` always -- the unknown
    behind the node at scoring, how much of it was mapped later, and those
    cells -- and ``shown`` / ``available`` / ``best_frame`` /
    ``current_frame`` when at least :data:`MIN_REVEALED_CELLS` were.
    """
    settings = ViewSettings(max_range_m=range_m, corner_slack_cells=0)
    reach_cells = int(round(reach_m / replayed.maps.resolution))
    finals, measured = {}, {}
    for variant in replayed.variants:
        for event in replayed.events[variant.name]:
            key = (event.step, event.node.node_id)
            if key not in measured:
                if event.floor_id not in finals:
                    finals[event.floor_id] = replayed.world(replayed.steps[-1]["step"], event.floor_id)[0]
                measured[key] = _behind(replayed, event, finals[event.floor_id], settings, reach_cells)
            behind, cells, offered, seen = measured[key]
            event.hindsight = {"behind_cells": behind, "revealed_cells": int(cells[0].size), "cells": cells}
            if seen is None:
                continue
            index = {record.name: i for i, record in enumerate(offered)}
            chosen = [index[name] for name in event.views.names if name in index]
            current = [i for i, record in enumerate(offered) if record.step == event.step]
            event.hindsight.update(
                shown=float(seen[chosen].any(axis=0).mean()) if chosen else 0.0,
                available=float(seen.any(axis=0).mean()),
                best_frame=float(seen.mean(axis=1).max()),
                current_frame=float(seen[current].any(axis=0).mean()) if current else 0.0)


def _behind(replayed, event, final, settings: ViewSettings, reach_cells: int):
    """``(behind, cells, offered, seen)`` of one scoring: the unknown behind its node, the part mapped later, and which frames saw it.

    ``final`` is the storey's map after the episode's last action. ``seen``
    is ``(frames, cells)`` bool over the frames offered at the scoring
    action and the later-mapped cells, read on ``final``; None when too
    little was mapped later to measure.
    """
    world, settled = replayed.world(event.step, event.floor_id)
    rows, cols = behind_cells(world.grid, settled, event.node, reach_cells)
    mapped = final.grid[rows, cols] != VALUES.unknown
    cells = (rows[mapped], cols[mapped])
    offered = storey_frames(replayed.frames.until(event.step), event.pose.z, settings.storey_tolerance_m)
    seen = None
    if int(mapped.sum()) >= MIN_REVEALED_CELLS:
        scene = SightScene(final, cells[0], cells[1], replayed.frames.camera, settings)
        seen = np.stack([scene.sight(record.pose).stage == SEEN for record in offered])
    return int(rows.size), cells, offered, seen


def add_patience(replayed) -> None:
    """Fill ``event.later`` of every scoring: the same node's views on the last action it was present.

    ``{"step", "usable", "score", "range_m", "through_fraction"}``, chosen
    by the scoring's own variant from the frames that variant is offered
    then.
    """
    for variant in replayed.variants:
        for event in replayed.events[variant.name]:
            step, floor_id, node, pose = replayed.last_nodes[event.node.node_id]
            views = event.views
            if step != event.step:
                world, settled = replayed.world(step, floor_id)
                views = select_views(world, [node], replayed.offered(variant, step), replayed.frames.camera,
                                     variant.settings, settled, reference_z=pose.z)[node.node_id]
            event.later = {"step": step, "usable": views.usable, "score": views.score,
                           "range_m": views.views[0].range_m if views.usable else None,
                           "through_fraction": views.through_fraction}
