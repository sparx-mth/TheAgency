# core/mapping/topology/room_stats.py
"""Per-room statistics over an occupancy grid and a room label image.

Numpy helpers ported from the flown SJTU ``semantic_mapper_node.py``,
unbound from the ROS node into free functions: door discovery, door-to-
room linking, frontier cluster counting, room-at-cell majority-vote
lookup, and the golden-ratio room color.

One of them is not from the flown node: :func:`door_room_pairs`, which
vets ``link_doors``' proximity answer against
:func:`~sparx_agency.core.mapping.topology.room_adjacency.room_adjacency`.
It lives beside ``link_doors`` because it is what a caller reaches for
in the same breath, while the adjacency scan it consumes is shared with
the segmenter and lives in its own module.

Conventions (shared with ``room_segmentation.py``):
    * ``grid``: (H, W) int8 occupancy — ``UNKNOWN`` (-1) unknown,
      0..``FREE_MAX`` free, >= ``OCC_MIN`` occupied; indexed ``[y, x]``.
    * ``room_lbl``: (H, W) int label image, 0 = no room, > 0 = a room.
      Labels are returned exactly as they appear in the array; whether
      they are fresh labels or pid+1 is the caller's convention.
    * Cells are ``(cx, cy)`` = (col, row) pairs.
"""

from __future__ import annotations

import colorsys
from collections import Counter
from typing import (Dict, Iterable, List, Optional, Sequence, Set,
                    Tuple)

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as cc_label

from sparx_agency.core.mapping.topology.room_segmentation import FREE_MAX, UNKNOWN


def discover_doors(
    grid: np.ndarray,
    door_cells: Sequence[Tuple[int, int]],
    discover_radius_cells: int,
) -> np.ndarray:
    """Report which doors the map has seen at all.

    A door is discovered when any grid cell within a +-radius square
    patch around its cell is non-unknown (free or occupied both count).
    Out-of-bounds doors are never discovered. The function is
    stateless: callers wanting the flown once-discovered-stays-
    discovered behaviour should OR the result into their own flags.

    Args:
        grid: (H, W) int8 occupancy grid.
        door_cells: Door positions as ``(cx, cy)`` cell pairs.
        discover_radius_cells: Half-size of the square search patch.

    Returns:
        (len(door_cells),) bool — True where the door is discovered.
    """
    H, W = grid.shape
    r = int(discover_radius_cells)
    out = np.zeros(len(door_cells), dtype=bool)
    for i, (dcx, dcy) in enumerate(door_cells):
        if not (0 <= dcx < W and 0 <= dcy < H):
            continue
        y0, y1 = max(0, dcy - r), min(H, dcy + r + 1)
        x0, x1 = max(0, dcx - r), min(W, dcx + r + 1)
        patch = grid[y0:y1, x0:x1]
        if (patch != UNKNOWN).any():
            out[i] = True
    return out


def link_doors(
    room_lbl: np.ndarray,
    door_cells: Sequence[Tuple[int, int]],
    cut_radius_cells: int,
    match_radius_cells: int,
) -> List[List[int]]:
    """Associate each door with the rooms it opens into.

    A room is linked to a door when its cells intersect the annulus
    between the cut radius (exclusive) and the match radius (inclusive)
    around the door cell — i.e. just outside the disk that
    ``compute_rooms`` punched through the skeleton. Out-of-bounds doors
    link to nothing. Callers should pass only discovered doors (the
    flown node skipped undiscovered ones).

    Args:
        room_lbl: (H, W) int room label image (0 = no room).
        door_cells: Door positions as ``(cx, cy)`` cell pairs.
        cut_radius_cells: Inner (exclusive) annulus radius — the door
            cut radius used during segmentation.
        match_radius_cells: Outer (inclusive) annulus radius.

    Returns:
        One sorted list of room labels per door, in input order.
    """
    H, W = room_lbl.shape
    r_in = int(cut_radius_cells)
    r_out = int(match_radius_cells)
    links = []  # type: List[List[int]]
    for dcx, dcy in door_cells:
        if not (0 <= dcx < W and 0 <= dcy < H):
            links.append([])
            continue
        y0, y1 = max(0, dcy - r_out), min(H, dcy + r_out + 1)
        x0, x1 = max(0, dcx - r_out), min(W, dcx + r_out + 1)
        ys = np.arange(y0, y1) - dcy
        xs = np.arange(x0, x1) - dcx
        dd = ys[:, None] ** 2 + xs[None, :] ** 2
        annulus = (dd > r_in * r_in) & (dd <= r_out * r_out)
        touched = np.unique(room_lbl[y0:y1, x0:x1][annulus])
        links.append(sorted(int(v) for v in touched if v > 0))
    return links


def door_room_pairs(
    links: Iterable[Sequence[int]],
    adjacency: Set[Tuple[int, int]],
) -> List[List[Tuple[int, int]]]:
    """Vet each door's rooms down to the pairs that genuinely touch.

    :func:`link_doors` answers PROXIMITY — whichever rooms have cells
    in the annulus around a door — which near a corner picks up a room
    on the far side of a wall, and then every pair of rooms in that
    annulus becomes a scene-graph edge. Vetting the candidates against
    :func:`room_adjacency` here turns those candidates into the edges
    the topology actually has.

    A separate function rather than a change inside ``link_doors``:
    the annulus is flown behaviour its callers and tests rely on, and
    the adjacency set is one scan shared by everything in a tick that
    needs it, so it is passed in rather than recomputed per door.

    A door whose candidate rooms do not touch contributes NO pair. Its
    rooms are not connected through it, and inventing an edge between
    them is the defect this exists to remove.

    Args:
        links: One room-label list per door, as :func:`link_doors`
            returns them.
        adjacency: ``(low, high)`` label pairs that genuinely touch, as
            :func:`room_adjacency` returns them.

    Returns:
        One list of ``(low, high)`` label pairs per door, in input
        order, sorted and possibly empty. Every pair is in
        ``adjacency``, so a consumer may draw each pair as an edge
        without checking anything itself.
    """
    out = []  # type: List[List[Tuple[int, int]]]
    for rooms in links:
        uniq = sorted({int(r) for r in rooms})
        out.append([(a, b) for i, a in enumerate(uniq) for b in uniq[i + 1:]
                    if (a, b) in adjacency])
    return out


def count_frontier_clusters(
    grid: np.ndarray,
    room_lbl: np.ndarray,
    min_cluster_cells: int,
) -> Dict[int, int]:
    """Count frontier clusters per room.

    A frontier cell is a free cell 4-adjacent to an unknown cell
    (diagonal contacts produce noisier clusters on room corners, so
    they do not count). Frontier cells are clustered by 8-connectivity;
    clusters below the size floor are dropped as noise; each surviving
    cluster is assigned to one room by majority vote over the room
    labels at its cells (label-0 cells abstain).

    Args:
        grid: (H, W) int8 occupancy grid.
        room_lbl: (H, W) int room label image (0 = no room).
        min_cluster_cells: Clusters smaller than this are dropped.

    Returns:
        ``{room_label: cluster_count}`` with an entry (possibly 0) for
        every label present in ``room_lbl``.
    """
    labels = [int(v) for v in np.unique(room_lbl) if v > 0]
    counts = {lbl: 0 for lbl in labels}

    free = (grid >= 0) & (grid <= FREE_MAX)
    unk = grid == UNKNOWN
    struct4 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=bool)
    unk_dil = binary_dilation(unk, structure=struct4)
    frontier = free & unk_dil
    if not frontier.any():
        return counts

    fc_lbl, n = cc_label(frontier, structure=np.ones((3, 3), np.uint8))
    if n == 0:
        return counts

    flat_lbl = fc_lbl.ravel()
    flat_room = room_lbl.ravel()
    for k in range(1, n + 1):
        sel = flat_lbl == k
        size = int(sel.sum())
        if size < min_cluster_cells:
            continue
        rooms_here = flat_room[sel]
        rooms_here = rooms_here[rooms_here > 0]
        if rooms_here.size == 0:
            continue
        winner = int(Counter(rooms_here.tolist()).most_common(1)[0][0])
        if winner in counts:
            counts[winner] += 1
    return counts


def room_at_cell(
    room_lbl: np.ndarray,
    cx: int,
    cy: int,
    snap_cells: int = 3,
) -> Optional[int]:
    """Return the room label at a cell, snapping to a nearby room.

    Objects projected from the camera often land ON a wall cell, in a
    doorway cut disk, or a couple of cells outside the healed free
    mask. Those should still be credited to the adjacent room, not
    dropped. If the exact cell is not in any room, the non-zero labels
    within a +-``snap_cells`` window vote and the majority wins.

    At 0.15 m resolution, ``snap_cells=3`` is a 0.9 m x 0.9 m window —
    conservative enough that an object in a corridor is not snapped
    across a wall into a room (walls are at least one cut-disk radius
    wide). Pass ``snap_cells=0`` for strict exact-cell lookup.

    Args:
        room_lbl: (H, W) int room label image (0 = no room).
        cx: Cell column.
        cy: Cell row.
        snap_cells: Half-size of the majority-vote window.

    Returns:
        The room label, or None when the cell is out of bounds or no
        room is within the window.
    """
    H, W = room_lbl.shape
    if not (0 <= cx < W and 0 <= cy < H):
        return None
    v = int(room_lbl[cy, cx])
    if v > 0:
        return v
    if snap_cells > 0:
        r = int(snap_cells)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        patch = room_lbl[y0:y1, x0:x1]
        vals = patch[patch > 0]
        if vals.size > 0:
            winner = Counter(vals.tolist()).most_common(1)[0][0]
            return int(winner)
    return None


def room_color(i: int) -> Tuple[float, float, float]:
    """Golden-ratio room color: distinct, stable hues per room id.

    ``hue = (i * 0.6180339887) % 1``, then HSV with s=0.85, v=0.95 —
    the exact constants flown in the SJTU scene-graph markers.
    Deliberately separate from the task-level visualization helpers
    (which use s=0.8, v=0.9): this is the flown room palette, kept
    bit-identical to the source.

    Args:
        i: Room id (any non-negative int; pids work directly).

    Returns:
        ``(r, g, b)`` floats in [0, 1].
    """
    h = (i * 0.6180339887) % 1.0
    return colorsys.hsv_to_rgb(h, 0.85, 0.95)
