# core/mapping/topology/room_registry.py
"""Persistent room identities across segmentation ticks (IoU and containment matching).

:func:`~sparx_agency.core.mapping.topology.room_segmentation.compute_rooms`
relabels rooms 1..N fresh every tick, so label 2 this tick need not be
label 2 the next. ``RoomRegistry`` matches each fresh room mask against
the previous tick's masks by greedy best-IoU 1:1 assignment and hands
out persistent ids (pids). Pids increase monotonically and are never
reused, so a vanished room's identity stays retired.

Ported from the flown SJTU ``semantic_mapper_node.py``; the IoU matching
math is unchanged. Since 2026-10-05 a pair that fails the IoU threshold
still matches when one mask mostly **contains** the other
(:attr:`RoomRegistry.containment_threshold` of the smaller mask's cells
lie in the larger): a room seen through its door is a sliver that grows
tenfold as the agent walks in, and IoU alone retired its number at every
growth spurt -- the Hanson ObjectNav recording renumbered one bedroom
R11 -> R14 -> R16 while standing in it, and with the number went the
record of having stood in it ("entered=no" to the oracle). IoU matches
are consumed first; containment matches only fill what IoU left, so a
split's larger half keeps the number and the smaller gets a new one, and
a merge's survivor is the old room with the larger overlap.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from sparx_agency.core.mapping.topology.room_segmentation import RoomStats


@dataclass(frozen=True)
class TrackedRoom:
    """One room with a persistent identity.

    Attributes:
        id: Persistent room id (pid), stable across ticks while the
            room keeps matching by IoU.
        mask: (H, W) bool membership mask from the latest tick.
        n_cells: Number of cells in the mask.
        centroid: World-frame ``(wx, wy)`` centroid, produced by the
            ``cell_to_world`` callable passed to :meth:`RoomRegistry.update`.
    """

    id: int
    mask: np.ndarray
    n_cells: int
    centroid: Tuple[float, float]


class RoomRegistry:
    """Greedy best-IoU 1:1 matcher of fresh rooms to the previous tick, with containment as the fallback.

    Attributes:
        iou_threshold: Minimum IoU for a fresh room to inherit a
            previous room's pid. The flown default parameter was 0.15
            (tolerant to mask drift while exploring); the class default
            mirrors the source's constructor default of 0.25.
        containment_threshold: A pair below the IoU threshold still
            matches when at least this share of the SMALLER mask's cells
            lie inside the larger one; 1.0 or more disables the fallback
            (the historical IoU-only matcher).
        rooms: ``OrderedDict[int, TrackedRoom]`` — the current rooms
            keyed by pid, replaced wholesale on every update.
    """

    def __init__(self, iou_threshold: float = 0.25, first_pid: int = 0, containment_threshold: float = 0.6) -> None:
        """Initialize an empty registry.

        Args:
            iou_threshold: Minimum IoU to keep a pid across ticks.
            first_pid: The first pid this registry hands out. A building
                keeps one registry per storey; starting each new storey's
                registry after the highest pid any storey has used keeps a
                room number unique across the building, so "R0" names one
                room in the recording, not one per floor.
            containment_threshold: See the class attribute.
        """
        self.iou_threshold = float(iou_threshold)
        if not 0.0 < float(containment_threshold):
            raise ValueError("containment_threshold must be positive, got %r" % (containment_threshold,))
        self.containment_threshold = float(containment_threshold)
        self.rooms = OrderedDict()  # type: "OrderedDict[int, TrackedRoom]"
        if int(first_pid) < 0:
            raise ValueError("first_pid must be non-negative, got %r" % (first_pid,))
        self._next = int(first_pid)

    @property
    def next_pid(self) -> int:
        """The pid the next unmatched room will receive; every pid below it is used or retired."""
        return self._next

    def update(
        self,
        stats: List[RoomStats],
        cell_to_world: Callable[[float, float], Tuple[float, float]],
    ) -> "OrderedDict[int, TrackedRoom]":
        """Match fresh rooms to the previous tick and assign pids.

        Every (fresh, previous) pair with any mask overlap and IoU at
        or above the threshold becomes a candidate; candidates are
        consumed greedily in descending IoU order, each fresh room and
        each pid used at most once. Pairs under the IoU threshold whose
        smaller mask lies at least ``containment_threshold`` inside the
        larger are candidates of a second tier, consumed after every IoU
        candidate, in descending containment. Unmatched fresh rooms get
        new, never-reused pids.

        Args:
            stats: Fresh rooms from ``compute_rooms`` (label order).
            cell_to_world: ``(cx, cy) -> (wx, wy)`` converter used to
                express each room centroid in world coordinates.

        Returns:
            The new ``rooms`` mapping (also stored on the registry),
            keyed by pid in ``stats`` order.
        """
        pairs = []
        for i, s in enumerate(stats):
            for pid, prev in self.rooms.items():
                if prev.mask.shape != s.mask.shape:
                    continue
                inter = int(np.logical_and(s.mask, prev.mask).sum())
                if inter == 0:
                    continue
                prev_cells = int(prev.mask.sum())
                union = s.n_cells + prev_cells - inter
                iou = inter / max(1, union)
                if iou >= self.iou_threshold:
                    pairs.append((1, iou, i, pid))
                    continue
                containment = inter / max(1, min(s.n_cells, prev_cells))
                if containment >= self.containment_threshold:
                    pairs.append((0, containment, i, pid))

        pairs.sort(reverse=True)
        i2id, used = {}, set()
        for _, _, i, pid in pairs:
            if i in i2id or pid in used:
                continue
            i2id[i] = pid
            used.add(pid)
        for i in range(len(stats)):
            if i not in i2id:
                i2id[i] = self._next
                self._next += 1

        new = OrderedDict()  # type: "OrderedDict[int, TrackedRoom]"
        for i, s in enumerate(stats):
            pid = i2id[i]
            wx, wy = cell_to_world(*s.centroid_cells)
            new[pid] = TrackedRoom(id=pid, mask=s.mask,
                                   n_cells=s.n_cells, centroid=(wx, wy))
        self.rooms = new
        return self.rooms
