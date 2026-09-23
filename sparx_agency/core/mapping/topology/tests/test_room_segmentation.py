# core/mapping/topology/tests/test_room_segmentation.py
"""Synthetic-grid tests for the grid-based room segmentation pipeline.

Recreates the intent of the old stack's ``test_voronoi.py`` scenarios:
two rooms joined by a corridor read as ONE room before any door cut and
TWO after cutting at the corridor door; per-cell labels are consistent;
a far-away door cuts nothing.
"""

from __future__ import annotations

import numpy as np

from sparx_agency.core.mapping.topology.room_segmentation import (
    RoomSegmentationParams,
    _ridge_fallback,
    compute_rooms,
)

# Door in the middle of the corridor: (cx, cy) = (col, row).
CORRIDOR_DOOR = (40, 17)


def two_rooms_and_corridor():
    """Two 30x30 rooms joined by a 4-cell-tall corridor."""
    free = np.zeros((60, 80), dtype=bool)
    free[5:35, 5:35] = True    # left room
    free[5:35, 45:75] = True   # right room
    free[15:19, 35:45] = True  # corridor
    return free


def test_no_doors_yields_one_room():
    free = two_rooms_and_corridor()
    room_lbl, skeleton, stats = compute_rooms(
        free, [], RoomSegmentationParams(door_cut_cells=4, min_room_cells=40))
    assert len(stats) == 1
    assert stats[0].id == 1
    assert skeleton.any()
    # Both sides carry the same label.
    assert room_lbl[20, 20] == 1
    assert room_lbl[20, 60] == 1


def test_door_cut_splits_into_two_rooms():
    free = two_rooms_and_corridor()
    room_lbl, skeleton, stats = compute_rooms(
        free, [CORRIDOR_DOOR],
        RoomSegmentationParams(door_cut_cells=4, min_room_cells=40))
    assert len(stats) == 2

    left = int(room_lbl[20, 20])
    right = int(room_lbl[20, 60])
    assert left > 0 and right > 0
    assert left != right

    # The cut disk removed the skeleton around the door.
    dcx, dcy = CORRIDOR_DOOR
    assert not skeleton[dcy - 1:dcy + 2, dcx - 1:dcx + 2].any()


def test_per_cell_labels_are_consistent():
    """Each room interior is uniformly labeled with its own label."""
    free = two_rooms_and_corridor()
    room_lbl, _, stats = compute_rooms(
        free, [CORRIDOR_DOOR],
        RoomSegmentationParams(door_cut_cells=4, min_room_cells=40))
    assert len(stats) == 2

    left = int(room_lbl[20, 20])
    right = int(room_lbl[20, 60])
    # Deep-interior blocks, far from the healed boundary and the cut.
    assert (room_lbl[10:30, 10:30] == left).all()
    assert (room_lbl[10:30, 50:70] == right).all()
    # Stats masks tile exactly the labeled cells.
    for s in stats:
        assert s.n_cells == int(s.mask.sum())
        assert (room_lbl[s.mask] == s.id).all()


def test_far_door_cuts_nothing():
    free = two_rooms_and_corridor()
    _, _, stats = compute_rooms(
        free, [(0, 0)],
        RoomSegmentationParams(door_cut_cells=4, min_room_cells=40))
    assert len(stats) == 1


def test_out_of_bounds_door_is_ignored():
    free = two_rooms_and_corridor()
    _, _, stats = compute_rooms(
        free, [(-5, 10), (1000, 1000)],
        RoomSegmentationParams(door_cut_cells=4, min_room_cells=40))
    assert len(stats) == 1


def test_empty_free_mask_yields_no_rooms():
    free = np.zeros((20, 20), dtype=bool)
    room_lbl, skeleton, stats = compute_rooms(free, [])
    assert stats == []
    assert not room_lbl.any()
    assert not skeleton.any()


def test_min_room_cells_drops_small_rooms():
    free = two_rooms_and_corridor()
    # A threshold above each half's size kills everything.
    _, _, stats = compute_rooms(
        free, [CORRIDOR_DOOR],
        RoomSegmentationParams(door_cut_cells=4, min_room_cells=10_000))
    assert stats == []


def test_centroids_land_inside_their_rooms():
    free = two_rooms_and_corridor()
    _, _, stats = compute_rooms(
        free, [CORRIDOR_DOOR],
        RoomSegmentationParams(door_cut_cells=4, min_room_cells=40))
    for s in stats:
        cx, cy = s.centroid_cells
        assert s.mask[int(round(cy)), int(round(cx))]


def test_ridge_fallback_produces_interior_skeleton():
    free = np.zeros((30, 40), dtype=bool)
    free[5:25, 5:35] = True
    sk = _ridge_fallback(free)
    assert sk.any()
    # Skeleton lies strictly inside the free mask.
    assert not sk[~free].any()
    # The mask center (a DT ridge) is on the skeleton.
    assert sk[15, 20]
