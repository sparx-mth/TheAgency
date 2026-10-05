# core/mapping/topology/tests/test_room_registry.py
"""Persistence tests for the IoU-based room identity registry."""

from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_registry import RoomRegistry
from sparx_agency.core.mapping.topology.room_segmentation import RoomStats

SHAPE = (40, 40)


def make_stats(fresh_id, y0, y1, x0, x1):
    """RoomStats for an axis-aligned rectangle room."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0:y1, x0:x1] = True
    ys, xs = np.where(mask)
    return RoomStats(id=fresh_id, mask=mask, n_cells=int(mask.sum()),
                     centroid_cells=(float(xs.mean()), float(ys.mean())))


def identity_c2w(cx, cy):
    return (cx, cy)


def test_same_room_keeps_pid_across_ticks():
    reg = RoomRegistry(iou_threshold=0.25)
    rooms1 = reg.update([make_stats(1, 5, 20, 5, 20)], identity_c2w)
    assert list(rooms1.keys()) == [0]
    rooms2 = reg.update([make_stats(1, 5, 20, 5, 20)], identity_c2w)
    assert list(rooms2.keys()) == [0]
    assert rooms2[0].n_cells == 15 * 15


def test_grown_room_rematches_by_iou():
    reg = RoomRegistry(iou_threshold=0.25)
    reg.update([make_stats(1, 5, 20, 5, 20)], identity_c2w)
    # The room grows as the drone explores: same corner, larger extent.
    rooms = reg.update([make_stats(1, 5, 25, 5, 25)], identity_c2w)
    assert list(rooms.keys()) == [0]
    assert rooms[0].n_cells == 20 * 20


def test_moved_room_below_iou_gets_new_pid():
    reg = RoomRegistry(iou_threshold=0.25)
    reg.update([make_stats(1, 0, 10, 0, 10)], identity_c2w)
    # Disjoint mask: no overlap, cannot match.
    rooms = reg.update([make_stats(1, 20, 30, 20, 30)], identity_c2w)
    assert list(rooms.keys()) == [1]


def test_vanished_room_pid_is_not_reused():
    reg = RoomRegistry(iou_threshold=0.25)
    reg.update([make_stats(1, 0, 10, 0, 10)], identity_c2w)  # pid 0
    reg.update([], identity_c2w)                             # room vanishes
    assert reg.rooms == {}
    rooms = reg.update([make_stats(1, 0, 10, 0, 10)], identity_c2w)
    # Same mask as pid 0 had, but pid 0 is retired: a fresh pid is issued.
    assert list(rooms.keys()) == [1]


def test_two_rooms_keep_pids_when_stats_order_swaps():
    reg = RoomRegistry(iou_threshold=0.25)
    a = make_stats(1, 0, 15, 0, 15)
    b = make_stats(2, 20, 35, 20, 35)
    reg.update([a, b], identity_c2w)  # a -> pid 0, b -> pid 1
    rooms = reg.update([b, a], identity_c2w)
    assert rooms[0].mask[5, 5] and not rooms[0].mask[25, 25]
    assert rooms[1].mask[25, 25] and not rooms[1].mask[5, 5]


def test_greedy_matching_prefers_higher_iou():
    reg = RoomRegistry(iou_threshold=0.1)
    reg.update([make_stats(1, 0, 20, 0, 20)], identity_c2w)  # pid 0
    # Two fresh rooms both overlap pid 0; the bigger-overlap one wins it.
    big_overlap = make_stats(1, 0, 20, 0, 15)
    small_overlap = make_stats(2, 0, 20, 15, 30)
    rooms = reg.update([big_overlap, small_overlap], identity_c2w)
    assert rooms[0].n_cells == big_overlap.n_cells
    assert 1 in rooms and rooms[1].n_cells == small_overlap.n_cells


def test_centroid_is_converted_to_world():
    reg = RoomRegistry()
    rooms = reg.update([make_stats(1, 0, 10, 0, 10)],
                       lambda cx, cy: (cx * 0.1, cy * 0.1))
    wx, wy = rooms[0].centroid
    assert abs(wx - 0.45) < 1e-9
    assert abs(wy - 0.45) < 1e-9


def test_shape_change_never_matches():
    reg = RoomRegistry(iou_threshold=0.1)
    reg.update([make_stats(1, 0, 10, 0, 10)], identity_c2w)  # pid 0
    other = np.zeros((10, 10), dtype=bool)
    other[:, :] = True
    stats = RoomStats(id=1, mask=other, n_cells=100,
                      centroid_cells=(4.5, 4.5))
    rooms = reg.update([stats], identity_c2w)
    assert list(rooms.keys()) == [1]
def test_a_registry_can_start_after_another_storeys_pids_so_room_numbers_are_unique_across_a_building():
    """One registry per storey; the second storey's starts after the first's highest pid, so "R0"
    names one room in the recording and the hallway upstairs is not a second R0."""
    upstairs = RoomRegistry(iou_threshold=0.25)
    upstairs.update([make_stats(1, 0, 10, 0, 10), make_stats(2, 20, 30, 20, 30)], identity_c2w)   # pids 0, 1
    assert upstairs.next_pid == 2
    downstairs = RoomRegistry(iou_threshold=0.25, first_pid=upstairs.next_pid)
    rooms = downstairs.update([make_stats(1, 0, 10, 0, 10)], identity_c2w)
    assert list(rooms.keys()) == [2], "the same mask on another storey is another room with its own number"
    assert downstairs.next_pid == 3
    rooms = downstairs.update([make_stats(1, 0, 10, 0, 10), make_stats(2, 20, 30, 20, 30)], identity_c2w)
    assert list(rooms.keys()) == [2, 3]
    with pytest.raises(ValueError):
        RoomRegistry(first_pid=-1)


# -- containment (2026-10-05) -------------------------------------------------------
def test_a_room_that_grows_tenfold_keeps_its_pid_by_containment():
    """A room seen through its door is a sliver; walking in grows it past any IoU threshold
    (the Hanson ObjectNav recording renumbered one bedroom three times while standing in it)."""
    reg = RoomRegistry(iou_threshold=0.25)
    reg.update([make_stats(1, 10, 12, 10, 20)], identity_c2w)            # 2 x 10 cells: the sliver
    rooms = reg.update([make_stats(1, 5, 30, 5, 30)], identity_c2w)      # 25 x 25: the room (IoU 0.03)
    assert list(rooms.keys()) == [0]
    strict = RoomRegistry(iou_threshold=0.25, containment_threshold=1.5)   # the historical matcher
    strict.update([make_stats(1, 10, 12, 10, 20)], identity_c2w)
    assert list(strict.update([make_stats(1, 5, 30, 5, 30)], identity_c2w).keys()) == [1]


def test_containment_matches_fill_only_what_iou_left_so_splits_and_merges_keep_the_larger_overlap():
    reg = RoomRegistry(iou_threshold=0.25)
    reg.update([make_stats(1, 0, 20, 0, 30)], identity_c2w)               # pid 0: 20 x 30
    # A split: the larger half keeps the pid (IoU 0.67), the smaller half is new, though it is fully
    # contained in the old room.
    rooms = reg.update([make_stats(1, 0, 20, 0, 20), make_stats(2, 0, 20, 20, 30)], identity_c2w)
    assert list(rooms.keys()) == [0, 1]
    # A merge: the survivor is the old room with the larger overlap; the other pid retires.
    rooms = reg.update([make_stats(1, 0, 20, 0, 30)], identity_c2w)
    assert list(rooms.keys()) == [0]
    # A tiny room inside a big fresh one does not steal the big one's pid from its IoU match.
    reg = RoomRegistry(iou_threshold=0.25)
    reg.update([make_stats(1, 0, 20, 0, 20), make_stats(2, 25, 27, 25, 30)], identity_c2w)   # pids 0 (big), 1 (tiny)
    rooms = reg.update([make_stats(1, 0, 30, 0, 32)], identity_c2w)                            # one fresh room holding both
    assert list(rooms.keys()) == [0]
    # Two fresh pieces each holding half of a vanished sliver: half is under the containment bar.
    reg = RoomRegistry(iou_threshold=0.25, containment_threshold=0.6)
    reg.update([make_stats(1, 10, 12, 10, 20)], identity_c2w)
    rooms = reg.update([make_stats(1, 0, 30, 0, 15), make_stats(2, 0, 30, 15, 30)], identity_c2w)
    assert list(rooms.keys()) == [1, 2]


def test_containment_threshold_is_validated():
    with pytest.raises(ValueError):
        RoomRegistry(containment_threshold=0.0)
