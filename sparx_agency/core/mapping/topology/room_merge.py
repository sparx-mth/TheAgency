# core/mapping/topology/room_merge.py
"""Repair watershed over-segmentation by basin dynamics.

The clearance watershed in
:mod:`~sparx_agency.core.mapping.topology.room_watershed` cannot
under-segment — every local maximum of the clearance field seeds its own
basin — but it can OVER-segment: one physical room with two wide spots
separated by furniture, or bent into an L, grows two peaks and comes out
as two adjacent rooms. Operators see this immediately ("R10 and R11 are
the same room, and it is not even a big one").

Why the obvious fix is wrong, measured
--------------------------------------
The tempting repair is to merge any two basins whose shared border is
wide. It does not work. On the captured hospital BEV
(``tests/fixtures/live_bev_hospital.npz``, 413x200 @ 0.15 m) the maximum
clearance along a shared border — the SADDLE — separates the two cases
only weakly:

    spurious splits   R30-R31 3.79 m, R7-R8 1.80 m, R31-R34 1.80 m
    real doorways     0.00 - 0.45 m
    all 67 pairs      p10 0.15  p25 0.45  p50 0.75  p75 1.35  p90 1.50

so the merge threshold has to sit around 1.2 m — and a corridor touches
many rooms at 1.2-1.5 m, so merging every wide-saddle pair CASCADES the
whole floor into one region through the corridors: 15 rooms with the
largest covering 82%, which is precisely the collapse the watershed was
written to fix.

What works: dynamics, not width
-------------------------------
Borrowed from the morphological notion of a maximum's *dynamics* (its
prominence): for two adjacent basins A and B with saddle clearance ``s``
and peak clearances ``peak_A``, ``peak_B``::

    depth(A, B) = min(peak_A - s, peak_B - s)

which is how much clearance is LOST descending from the shallower of the
two peaks to the saddle between them. Repeatedly merge the pair with the
smallest depth while that depth is below ``min_dynamics_m``.

A real room keeps its identity because its peak towers over the doorway
it is joined through (3.0 m peak against a 0.5 m saddle -> depth 2.5 m,
never merged). A spurious split collapses because neither lobe is much
deeper than the neck between them (2.0 m peak against a 1.8 m saddle ->
depth 0.2 m, merged). The corridor cascade does not happen: a corridor
is *narrow*, so its own peak sits barely above the saddles it shares,
and it merges at most with its immediate lobes rather than chaining.

Measured on the same captured BEV, 43 basins with the 35 listed hospital
doors carved, largest room as a share of the segmented area:

    merge_dynamics_m   rooms   largest      merge_dynamics_m   rooms  largest
      0.00 (off)         43      10.7%        0.60               29    12.2%
      0.10               42      10.7%        0.75               28    12.2%
      0.20               39      10.7%        0.90               28    12.2%
      0.30               36      12.2%        1.00               27    12.2%
      0.40               33      12.2%        1.25               26    12.4%
      0.50 (default)     29      12.2%        2.00               26    12.4%

Ground truth for this building is 20 rooms + 7 corridors = 27 regions.
The result is flat from 0.50 to 2.00 m and the largest room never moves
past 12.4%, so the knob is a plateau rather than a knife edge. Without
the door list the same sweep is much sharper (35 -> 29 -> 22 -> 9 rooms
at 0.20 / 0.30 / 0.50 m), which is the measurement behind the next
paragraph: the doors are what stop geometry alone from over-merging.

Doors are a hard barrier
------------------------
The user's model is "a room is a closed area bounded by doors", so two
basins the door carve separated are NEVER merged, whatever their
dynamics. The carve is gone by the time this module runs — the watershed
hands its cells back to the nearest basin — so the barrier is recovered
from the carve mask itself: a border whose cells lie inside a carved
door disk is a door, and its graph edge is marked unmergeable for good.

Cost
----
One pass over the label image builds the whole region adjacency graph;
the contraction then runs on the graph alone (tens of nodes) and the
image is touched again only once, to apply the final label remap. On the
413x200 fixture (43 basins) the stage measures 1.17 ms median / 1.74 ms
p90 — 0.50 ms of it the border scan and 0.18 ms the peaks — against a
~46 ms segmentation and the mapper's 500 ms tick.

Dependencies: numpy, plus ``room_adjacency.iter_label_borders`` for the
border scan it shares with the scene graph's room-to-room edges.
Host-owned path, like the rest of ``core/mapping/topology``.
"""

from __future__ import annotations

import heapq
from typing import Dict, Optional, Set, Tuple

import numpy as np

from sparx_agency.core.mapping.topology.room_adjacency import (
    iter_label_borders,
)


def merge_basins_by_dynamics(
    labels: np.ndarray,
    dt: np.ndarray,
    min_dynamics_m: float,
    barrier_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Merge adjacent basins whose dynamics fall below a threshold.

    Args:
        labels: (H, W) int label image, 0 = no basin, 1..N = basins.
            Labels need not be contiguous; the caller compacts them.
        dt: (H, W) float clearance field in metres, the same field the
            watershed flooded.
        min_dynamics_m: Merge while the smallest ``min(peak_A - s,
            peak_B - s)`` over adjacent pairs is below this. Zero or
            negative DISABLES merging and returns ``labels`` unchanged
            — an explicit off switch rather than a no-op threshold,
            because ``depth`` can be slightly negative (the saddle is
            the maximum over BOTH sides of the border, so it may exceed
            the shallower basin's own peak) and a literal ``< 0.0``
            would still merge such a pair.
        barrier_mask: (H, W) bool, True where a border between two
            basins is a hard boundary — the door carve. A pair sharing
            any barrier cell is never merged. ``None`` means pure
            geometry.

    Returns:
        (H, W) int32 label image with merged basins carrying a single
        label (the smallest of the merged labels). Labels are NOT
        compacted; every surviving label is one of the inputs.

    Raises:
        ValueError: If ``labels`` and ``dt`` disagree in shape, or if
            ``barrier_mask`` is given and does not match either.
    """
    if labels.shape != dt.shape:
        raise ValueError("labels %r and dt %r must have the same shape"
                         % (labels.shape, dt.shape))
    if barrier_mask is not None and barrier_mask.shape != labels.shape:
        raise ValueError("barrier_mask %r does not match labels %r"
                         % (barrier_mask.shape, labels.shape))
    if not min_dynamics_m > 0.0:
        return labels.astype(np.int32, copy=False)

    n = int(labels.max()) if labels.size else 0
    if n < 2:
        return labels.astype(np.int32, copy=False)

    pairs, saddles, barred = _basin_borders(labels, dt, barrier_mask, n)
    if not len(pairs):
        return labels.astype(np.int32, copy=False)

    graph = _BasinGraph(pairs, saddles, barred, _basin_peaks(labels, dt, n))
    _contract(graph, float(min_dynamics_m))
    return graph.remap()[labels]


def _basin_borders(
    labels: np.ndarray,
    dt: np.ndarray,
    barrier_mask: Optional[np.ndarray],
    n_labels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the region adjacency graph in ONE pass over the image.

    Every 4-adjacent pair of cells carrying different non-zero labels is
    a border sample, found by the same
    :func:`~sparx_agency.core.mapping.topology.room_adjacency.iter_label_borders`
    scan that answers ``room_adjacency`` for the scene graph — one
    implementation, so a pair of basins this merges and a pair of rooms
    that gets a graph edge are adjacent in exactly the same sense.
    Samples are grouped by the (low, high) label pair and reduced to the
    MAXIMUM clearance along that border (the saddle) and to whether any
    sample sits on a barrier cell.

    Args:
        labels: (H, W) int label image.
        dt: (H, W) float clearance field in metres.
        barrier_mask: (H, W) bool hard-boundary mask, or ``None``.
        n_labels: Maximum label value, used to pack a pair into one int.

    Returns:
        Tuple of ``(pairs, saddles, barred)``: an (E, 2) int64 array of
        ``(low, high)`` label pairs, an (E,) float array of saddle
        clearances in metres, and an (E,) bool array of barrier flags.
    """
    lows, highs, clears, bars = [], [], [], []
    for here, there, border in iter_label_borders(labels, connectivity=4):
        ai = labels[here][border].astype(np.int64)
        bi = labels[there][border].astype(np.int64)
        lows.append(np.minimum(ai, bi))
        highs.append(np.maximum(ai, bi))
        clears.append(np.maximum(dt[here][border], dt[there][border]))
        bars.append(np.zeros(border.sum(), bool) if barrier_mask is None
                    else barrier_mask[here][border]
                    | barrier_mask[there][border])
    if not lows:
        return (np.zeros((0, 2), np.int64), np.zeros(0, float),
                np.zeros(0, bool))

    stride = n_labels + 1
    codes = np.concatenate(lows) * stride + np.concatenate(highs)
    order = np.argsort(codes, kind="stable")
    ordered = codes[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    saddles = np.maximum.reduceat(np.concatenate(clears)[order], starts)
    barred = np.logical_or.reduceat(np.concatenate(bars)[order], starts)
    keys = ordered[starts]
    return (np.stack([keys // stride, keys % stride], axis=1),
            saddles, barred)


def _basin_peaks(labels: np.ndarray, dt: np.ndarray,
                 n_labels: int) -> np.ndarray:
    """Maximum clearance inside each basin, indexed by label.

    Args:
        labels: (H, W) int label image.
        dt: (H, W) float clearance field in metres.
        n_labels: Maximum label value.

    Returns:
        (n_labels + 1,) float array; entry 0 is meaningless.
    """
    peaks = np.zeros(n_labels + 1, float)
    np.maximum.at(peaks, labels.ravel(), dt.ravel())
    return peaks


def _contract(graph: "_BasinGraph", min_dynamics_m: float) -> None:
    """Merge the shallowest pair until nothing is shallower than the bar.

    A lazy heap: entries are pushed with the depth they had when pushed
    and re-validated on pop, so the graph is never rescanned and the
    image is never touched. Popping a depth at or above the bar ends the
    run, which is sound because every live edge always has an entry
    holding its CURRENT depth — the only edges whose depth can change
    are those incident to the root that just absorbed a neighbour, and
    those are all re-pushed.

    Args:
        graph: The basin graph, contracted in place.
        min_dynamics_m: Merge strictly below this depth, in metres.
    """
    heap = [(graph.depth(a, b), a, b) for a, b in graph.edges()]
    heapq.heapify(heap)
    while heap:
        depth, a, b = heapq.heappop(heap)
        if depth >= min_dynamics_m:
            return
        ra, rb = graph.find(a), graph.find(b)
        if ra == rb or not graph.adjacent(ra, rb):
            continue
        current = graph.depth(ra, rb)
        if current != depth:
            heapq.heappush(heap, (current, ra, rb))
            continue
        if graph.barred(ra, rb):
            continue
        root = graph.union(ra, rb)
        for neighbour in graph.neighbours(root):
            heapq.heappush(heap, (graph.depth(root, neighbour), root,
                                  neighbour))


class _BasinGraph:
    """Region adjacency graph of the basins, contracted by union-find.

    Nodes are basin labels; an edge carries the saddle clearance along
    the shared border and a barrier flag. Contracting an edge takes the
    maximum peak of the two basins and, for every shared neighbour, the
    maximum of the two saddles — so the graph alone answers every later
    query and the label image is read exactly once, before this is
    built, and written exactly once, after it is done.
    """

    def __init__(self, pairs: np.ndarray, saddles: np.ndarray,
                 barred: np.ndarray, peaks: np.ndarray) -> None:
        """Build the graph from the border reduction.

        Args:
            pairs: (E, 2) int array of ``(low, high)`` label pairs.
            saddles: (E,) float saddle clearances in metres.
            barred: (E,) bool, True where the border is a hard boundary.
            peaks: (n_labels + 1,) float peak clearance per label.
        """
        self._n = int(peaks.size) - 1
        self._adj = {}      # type: Dict[int, Dict[int, float]]
        self._barred = {}   # type: Dict[int, Set[int]]
        for (a, b), saddle, is_barred in zip(pairs, saddles, barred):
            a, b, saddle = int(a), int(b), float(saddle)
            self._adj.setdefault(a, {})[b] = saddle
            self._adj.setdefault(b, {})[a] = saddle
            if is_barred:
                self._barred.setdefault(a, set()).add(b)
                self._barred.setdefault(b, set()).add(a)
        self._peak = {r: float(peaks[r]) for r in self._adj}
        self._parent = {r: r for r in self._adj}

    def edges(self) -> Tuple[Tuple[int, int], ...]:
        """Every edge once, as ``(low, high)`` label pairs."""
        return tuple((a, b) for a, nbs in self._adj.items()
                     for b in nbs if a < b)

    def neighbours(self, root: int) -> Tuple[int, ...]:
        """The labels currently adjacent to ``root``."""
        return tuple(self._adj[root])

    def adjacent(self, a: int, b: int) -> bool:
        """Whether ``a`` and ``b`` still share a border."""
        return b in self._adj.get(a, {})

    def barred(self, a: int, b: int) -> bool:
        """Whether a hard boundary forbids merging ``a`` with ``b``."""
        return b in self._barred.get(a, ())

    def depth(self, a: int, b: int) -> float:
        """Dynamics of the shallower of two adjacent basins, in metres."""
        saddle = self._adj[a][b]
        return min(self._peak[a] - saddle, self._peak[b] - saddle)

    def find(self, label: int) -> int:
        """The surviving root of ``label``, with path compression."""
        while self._parent[label] != label:
            self._parent[label] = self._parent[self._parent[label]]
            label = self._parent[label]
        return label

    def union(self, a: int, b: int) -> int:
        """Contract the edge ``(a, b)``; the smaller label survives.

        Args:
            a: One root label.
            b: The other root label; must be adjacent to ``a``.

        Returns:
            The surviving root label.
        """
        keep, drop = (a, b) if a < b else (b, a)
        self._parent[drop] = keep
        self._peak[keep] = max(self._peak[keep], self._peak[drop])
        del self._adj[keep][drop]
        for neighbour, saddle in self._adj.pop(drop).items():
            self._adj[neighbour].pop(drop, None)
            if neighbour == keep:
                continue
            widest = max(self._adj[keep].get(neighbour, saddle), saddle)
            self._adj[keep][neighbour] = widest
            self._adj[neighbour][keep] = widest
        self._inherit_barriers(keep, drop)
        return keep

    def _inherit_barriers(self, keep: int, drop: int) -> None:
        """Move ``drop``'s hard boundaries onto ``keep``.

        A door that separated the absorbed basin from a neighbour still
        separates the merged one, so the flag is unioned rather than
        dropped — conservative, in the direction the operator's "a room
        is bounded by doors" rule asks for.

        Args:
            keep: The surviving root label.
            drop: The absorbed root label.
        """
        for neighbour in self._barred.pop(drop, ()):
            self._barred[neighbour].discard(drop)
            if neighbour == keep:
                continue
            self._barred.setdefault(keep, set()).add(neighbour)
            self._barred.setdefault(neighbour, set()).add(keep)

    def remap(self) -> np.ndarray:
        """Old label -> surviving label, as a lookup table.

        Returns:
            (n_labels + 1,) int32 array to index the label image with,
            applying every merge in one vectorised pass.
        """
        table = np.arange(self._n + 1, dtype=np.int32)
        for label in self._parent:
            table[label] = self.find(label)
        return table
