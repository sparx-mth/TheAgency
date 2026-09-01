# core/mapping/topology/room_adjacency.py
"""Which labelled regions genuinely touch, over a room label image.

One border scan, two consumers that must not disagree:

* :mod:`~sparx_agency.core.mapping.topology.room_merge` contracts a
  region adjacency graph to repair watershed over-segmentation;
* the scene graph draws a room-to-room edge for every pair of rooms
  that are connected.

Before this module, the scene graph's edges came from ``link_doors``
alone, which is PROXIMITY: whichever rooms have cells in the annulus
around a door, then every pair of them. A door near a corner picks up a
room on the far side of a wall, and an edge appears between two rooms
with a wall between them — the defect an operator reported as "how is
R11 connected to R16?".

The rule here is the operator's own: two rooms are connected only if a
path leads from one to the other without crossing a wall or a third
room, i.e. only if their regions are directly adjacent in the label
image. Wall cells are not free and carry label 0, so rooms either side
of a wall never touch; the doorway cells that both segmenters carve and
then hand back to their nearest room DO touch, so rooms either side of
a doorway are adjacent, which is exactly the wanted asymmetry.

Measured on the captured hospital BEV
(``tests/fixtures/live_bev_hospital.npz``, 413x200 @ 0.15 m, 29 rooms,
35 doors): the doors propose 61 distinct room pairs, of which 57 are
genuinely adjacent — 4 edges crossed a wall. The scan itself costs
0.20 ms median on that grid.

Dependencies: numpy only. Host-owned path, like the rest of
``core/mapping/topology``.
"""

from __future__ import annotations

from typing import Iterator, Set, Tuple

import numpy as np

# Neighbour shifts as ``(this cell, the next cell)`` VIEWS of one grid:
# 4-connectivity is the two orthogonal steps, 8-connectivity adds the two
# diagonals. Views rather than rolls, so nothing is copied and nothing
# wraps around the grid edge.
_BORDER_SHIFTS = {
    4: ((np.s_[:, :-1], np.s_[:, 1:]),
        (np.s_[:-1, :], np.s_[1:, :])),
    8: ((np.s_[:, :-1], np.s_[:, 1:]),
        (np.s_[:-1, :], np.s_[1:, :]),
        (np.s_[:-1, :-1], np.s_[1:, 1:]),
        (np.s_[:-1, 1:], np.s_[1:, :-1])),
}


def iter_label_borders(
    room_lbl: np.ndarray,
    connectivity: int = 4,
) -> Iterator[Tuple[Tuple, Tuple, np.ndarray]]:
    """Yield a label image's shared borders, one neighbour shift at a time.

    The single border scan in this package: :func:`room_adjacency` uses
    it, and so does the region adjacency graph ``room_merge``
    contracts, so the segmenter's idea of "these two basins touch" and
    the scene graph's idea of "these two rooms are connected" cannot
    drift apart. It yields the raw samples rather than a reduction
    because the two callers reduce them differently — one to a pair
    set, the other to a saddle clearance and a barrier flag per pair.

    Args:
        room_lbl: (H, W) int label image, 0 = no region.
        connectivity: 4 (orthogonal steps only) or 8 (diagonals too).

    Yields:
        ``(here, there, border)`` per shift that has one: two index
        expressions selecting the two sides of the shift, and a bool
        mask shaped like those views, True where the sides carry
        different non-zero labels.

    Raises:
        ValueError: If ``connectivity`` is neither 4 nor 8.
    """
    if connectivity not in _BORDER_SHIFTS:
        raise ValueError("connectivity must be 4 or 8, got %r"
                         % (connectivity,))
    for here, there in _BORDER_SHIFTS[connectivity]:
        a, b = room_lbl[here], room_lbl[there]
        border = (a != b) & (a > 0) & (b > 0)
        if border.any():
            yield here, there, border


def room_adjacency(room_lbl: np.ndarray,
                   connectivity: int = 4) -> Set[Tuple[int, int]]:
    """The room pairs whose regions genuinely touch.

    This is the room-to-room edge rule: two rooms are connected only
    when cells of one sit directly against cells of the other, so a
    path between them crosses no wall and no third room. Proximity to a
    shared door is NOT adjacency.

    Args:
        room_lbl: (H, W) int room label image, 0 = no room.
        connectivity: 4 (the default, and what the edge rule means) or
            8. Under 4 two rooms meeting only corner to corner are not
            adjacent — a diagonal contact is a pixel artefact of the
            watershed, not a doorway.

    Returns:
        Set of ``(low, high)`` label pairs, each once. Label 0 never
        appears and no pair names one room twice.
    """
    out = set()  # type: Set[Tuple[int, int]]
    for here, there, border in iter_label_borders(room_lbl, connectivity):
        a = room_lbl[here][border].astype(np.int64)
        b = room_lbl[there][border].astype(np.int64)
        out.update(zip(np.minimum(a, b).tolist(),
                       np.maximum(a, b).tolist()))
    return out
