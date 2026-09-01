# core/mapping/topology/room_registry.py
"""Persistent room identities across segmentation ticks (IoU matching).

:func:`~sparx_agency.core.mapping.topology.room_segmentation.compute_rooms`
relabels rooms 1..N fresh every tick, so label 2 this tick need not be
label 2 the next. ``RoomRegistry`` matches each fresh room mask against
the previous tick's masks by greedy best-IoU 1:1 assignment and hands
out persistent ids (pids). Pids increase monotonically and are never
reused, so a vanished room's identity stays retired.

Ported from the flown SJTU ``semantic_mapper_node.py``; the matching
math is unchanged.
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
    """Greedy best-IoU 1:1 matcher of fresh rooms to the previous tick.

    Attributes:
        iou_threshold: Minimum IoU for a fresh room to inherit a
            previous room's pid. The flown default parameter was 0.15
            (tolerant to mask drift while exploring); the class default
            mirrors the source's constructor default of 0.25.
        rooms: ``OrderedDict[int, TrackedRoom]`` — the current rooms
            keyed by pid, replaced wholesale on every update.
    """

    def __init__(self, iou_threshold: float = 0.25) -> None:
        """Initialize an empty registry.

        Args:
            iou_threshold: Minimum IoU to keep a pid across ticks.
        """
        self.iou_threshold = float(iou_threshold)
        self.rooms = OrderedDict()  # type: "OrderedDict[int, TrackedRoom]"
        self._next = 0

    def update(
        self,
        stats: List[RoomStats],
        cell_to_world: Callable[[float, float], Tuple[float, float]],
    ) -> "OrderedDict[int, TrackedRoom]":
        """Match fresh rooms to the previous tick and assign pids.

        Every (fresh, previous) pair with any mask overlap and IoU at
        or above the threshold becomes a candidate; candidates are
        consumed greedily in descending IoU order, each fresh room and
        each pid used at most once. Unmatched fresh rooms get new,
        never-reused pids.

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
                union = s.n_cells + int(prev.mask.sum()) - inter
                iou = inter / max(1, union)
                if iou >= self.iou_threshold:
                    pairs.append((iou, i, pid))

        pairs.sort(reverse=True)
        i2id, used = {}, set()
        for _, i, pid in pairs:
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
