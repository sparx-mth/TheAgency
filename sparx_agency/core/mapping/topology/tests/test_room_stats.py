# core/mapping/topology/tests/test_room_stats.py
"""Tests for the free-function room statistics helpers.

The ``room_adjacency`` / ``door_room_pairs`` group at the end covers the
room-to-room edge rule: ``link_doors`` answers proximity, and only the
pairs whose regions genuinely touch may become graph edges.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_adjacency import room_adjacency
from sparx_agency.core.mapping.topology.room_segmentation import (
    FREE_MAX,
    UNKNOWN,
)
from sparx_agency.core.mapping.topology.room_stats import (
    count_frontier_clusters,
    discover_doors,
    door_room_pairs,
    link_doors,
    room_at_cell,
    room_color,
)
from sparx_agency.core.mapping.topology.room_watershed import (
    WatershedRoomParams,
    segment_rooms_watershed,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / \
    "live_bev_hospital.npz"


def unknown_grid(h=30, w=40):
    return np.full((h, w), UNKNOWN, dtype=np.int8)


# ── discover_doors ────────────────────────────────────────────────────


def test_door_in_fully_unknown_grid_is_not_discovered():
    grid = unknown_grid()
    assert not discover_doors(grid, [(20, 15)], 2).any()


def test_door_near_seen_cells_is_discovered():
    grid = unknown_grid()
    grid[15, 22] = 0  # one free cell inside the +-2 patch of (20, 15)
    disc = discover_doors(grid, [(20, 15), (5, 5)], 2)
    assert disc.tolist() == [True, False]


def test_occupied_cell_also_discovers_a_door():
    grid = unknown_grid()
    grid[15, 20] = 100
    assert discover_doors(grid, [(20, 15)], 1).tolist() == [True]


def test_out_of_bounds_door_is_never_discovered():
    grid = np.zeros((30, 40), dtype=np.int8)  # all free
    disc = discover_doors(grid, [(-1, 5), (39, 30), (5, 5)], 2)
    assert disc.tolist() == [False, False, True]


# ── link_doors ────────────────────────────────────────────────────────


def door_between_two_rooms():
    """Two labeled rooms with a gap; the door sits in the gap at (20, 10)."""
    room_lbl = np.zeros((20, 40), dtype=np.int32)
    room_lbl[:, :18] = 1
    room_lbl[:, 23:] = 2
    return room_lbl, (20, 10)


def test_door_annulus_links_both_rooms():
    room_lbl, door = door_between_two_rooms()
    links = link_doors(room_lbl, [door], cut_radius_cells=2,
                       match_radius_cells=6)
    assert links == [[1, 2]]


def test_annulus_excludes_the_cut_disk():
    # Room cells only INSIDE the cut radius must not link.
    room_lbl = np.zeros((20, 40), dtype=np.int32)
    room_lbl[10, 20] = 1  # exactly at the door cell, dd=0 <= r_in^2
    links = link_doors(room_lbl, [(20, 10)], cut_radius_cells=2,
                       match_radius_cells=6)
    assert links == [[]]


def test_far_door_links_nothing():
    room_lbl, _ = door_between_two_rooms()
    links = link_doors(room_lbl, [(20, 10), (-3, 2)], cut_radius_cells=1,
                       match_radius_cells=2)
    # Match radius 2 does not reach either room (3 cells away) from
    # column 20, and the out-of-bounds door links nothing.
    assert links == [[], []]


# ── count_frontier_clusters ───────────────────────────────────────────


def test_frontier_edge_counts_one_cluster_for_the_room():
    grid = np.full((20, 20), UNKNOWN, dtype=np.int8)
    grid[5:15, 5:15] = 0            # free room amid unknown
    room_lbl = np.zeros((20, 20), dtype=np.int32)
    room_lbl[5:15, 5:15] = 1
    counts = count_frontier_clusters(grid, room_lbl, min_cluster_cells=4)
    # The whole free boundary touches unknown: one 8-connected ring.
    assert counts == {1: 1}


def test_walled_room_has_zero_frontiers():
    grid = np.full((20, 20), 100, dtype=np.int8)  # occupied everywhere
    grid[5:15, 5:15] = 0
    room_lbl = np.zeros((20, 20), dtype=np.int32)
    room_lbl[5:15, 5:15] = 1
    counts = count_frontier_clusters(grid, room_lbl, min_cluster_cells=1)
    assert counts == {1: 0}


def test_small_frontier_clusters_are_dropped():
    grid = np.full((10, 10), 100, dtype=np.int8)
    grid[4:6, 4:6] = 0        # tiny free pocket
    grid[3, 4] = UNKNOWN      # makes (4,4) and (4,5)... only row-4 frontier
    room_lbl = np.zeros((10, 10), dtype=np.int32)
    room_lbl[4:6, 4:6] = 1
    small = count_frontier_clusters(grid, room_lbl, min_cluster_cells=4)
    assert small == {1: 0}
    kept = count_frontier_clusters(grid, room_lbl, min_cluster_cells=1)
    assert kept == {1: 1}


def test_frontier_cluster_majority_vote_between_rooms():
    grid = np.full((10, 30), UNKNOWN, dtype=np.int8)
    grid[5, 5:14] = 0   # a free strip under unknown: all 9 cells frontier
    room_lbl = np.zeros((10, 30), dtype=np.int32)
    room_lbl[5, 5:11] = 1   # 6 cells of room 1
    room_lbl[5, 11:14] = 2  # 3 cells of room 2
    counts = count_frontier_clusters(grid, room_lbl, min_cluster_cells=2)
    assert counts == {1: 1, 2: 0}


def test_frontier_counts_cover_all_labels():
    grid = np.full((10, 10), 100, dtype=np.int8)
    room_lbl = np.zeros((10, 10), dtype=np.int32)
    room_lbl[1, 1] = 3
    room_lbl[8, 8] = 7
    counts = count_frontier_clusters(grid, room_lbl, min_cluster_cells=1)
    assert counts == {3: 0, 7: 0}


# ── room_at_cell ──────────────────────────────────────────────────────


def test_exact_cell_lookup():
    room_lbl = np.zeros((10, 10), dtype=np.int32)
    room_lbl[4, 6] = 2
    assert room_at_cell(room_lbl, 6, 4) == 2


def test_wall_cell_snaps_to_adjacent_room():
    room_lbl = np.zeros((10, 10), dtype=np.int32)
    room_lbl[3:7, 3:7] = 5
    # (8, 5) is 2 cells outside the room: within the default +-3 window.
    assert room_at_cell(room_lbl, 8, 5) == 5


def test_snap_majority_wins():
    room_lbl = np.zeros((10, 10), dtype=np.int32)
    room_lbl[5, 4] = 1
    room_lbl[5, 6] = 2
    room_lbl[6, 6] = 2
    assert room_at_cell(room_lbl, 5, 5, snap_cells=1) == 2


def test_no_room_within_window_returns_none():
    room_lbl = np.zeros((10, 10), dtype=np.int32)
    room_lbl[0, 0] = 1
    assert room_at_cell(room_lbl, 9, 9, snap_cells=3) is None


def test_strict_lookup_with_zero_snap():
    room_lbl = np.zeros((10, 10), dtype=np.int32)
    room_lbl[5, 5] = 1
    assert room_at_cell(room_lbl, 6, 5, snap_cells=0) is None


def test_out_of_bounds_cell_returns_none():
    room_lbl = np.ones((10, 10), dtype=np.int32)
    assert room_at_cell(room_lbl, -1, 5) is None
    assert room_at_cell(room_lbl, 5, 10) is None


# ── room_color ────────────────────────────────────────────────────────


def test_room_color_matches_flown_formula():
    for i in range(10):
        expected = colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0, 0.85, 0.95)
        assert room_color(i) == expected


def test_room_colors_are_valid_and_distinct():
    colors = [room_color(i) for i in range(8)]
    for c in colors:
        assert all(0.0 <= v <= 1.0 for v in c)
    assert len(set(colors)) == len(colors)


# ── room_adjacency ────────────────────────────────────────────────────


def test_rooms_sharing_a_border_are_adjacent():
    room_lbl = np.zeros((10, 10), dtype=np.int32)
    room_lbl[:, :5] = 1
    room_lbl[:, 5:] = 2
    assert room_adjacency(room_lbl) == {(1, 2)}


def test_rooms_separated_by_one_wall_cell_are_not_adjacent():
    """One column of label 0 between them is a wall, and walls disconnect."""
    room_lbl = np.zeros((10, 11), dtype=np.int32)
    room_lbl[:, :5] = 1
    room_lbl[:, 6:] = 2
    assert room_adjacency(room_lbl) == set()


def test_a_diagonal_touch_is_not_adjacency_under_connectivity_4():
    """Corner-to-corner contact is a watershed pixel artefact, not a door."""
    room_lbl = np.zeros((4, 4), dtype=np.int32)
    room_lbl[0, 0] = 1
    room_lbl[1, 1] = 2
    assert room_adjacency(room_lbl) == set()
    assert room_adjacency(room_lbl, connectivity=8) == {(1, 2)}


def test_label_zero_is_never_a_room():
    """Every room borders the background; none of that is an edge."""
    room_lbl = np.zeros((8, 8), dtype=np.int32)
    room_lbl[1:3, 1:3] = 4
    assert room_adjacency(room_lbl) == set()


def test_adjacency_pairs_are_sorted_and_unique():
    room_lbl = np.zeros((6, 9), dtype=np.int32)
    room_lbl[:, :3] = 7
    room_lbl[:, 3:6] = 2
    room_lbl[:, 6:] = 5
    assert room_adjacency(room_lbl) == {(2, 7), (2, 5)}


def test_an_unsupported_connectivity_raises():
    with pytest.raises(ValueError):
        room_adjacency(np.zeros((4, 4), dtype=np.int32), connectivity=6)


# ── door_room_pairs ───────────────────────────────────────────────────


def test_a_door_between_adjacent_rooms_keeps_its_pair():
    assert door_room_pairs([[1, 2]], {(1, 2)}) == [[(1, 2)]]


def test_a_door_whose_rooms_do_not_touch_contributes_no_pair():
    """The reported defect: proximity is not connectivity."""
    assert door_room_pairs([[1, 2]], set()) == [[]]


def test_only_the_touching_pairs_of_a_crowded_door_survive():
    """Three regions meet at a door and only two of the pairs are real.

    The room list is not a clique, which is why the vetted PAIRS are the
    answer and re-pairing the room list downstream is not.
    """
    assert door_room_pairs([[3, 5, 9]], {(3, 5), (5, 9)}) == \
        [[(3, 5), (5, 9)]]


def test_pairs_are_per_door_and_in_input_order():
    links = [[1, 2], [], [2, 3], [1, 3]]
    adjacency = {(1, 2), (2, 3)}
    assert door_room_pairs(links, adjacency) == \
        [[(1, 2)], [], [(2, 3)], []]


def test_duplicate_and_unsorted_room_labels_do_not_duplicate_an_edge():
    assert door_room_pairs([[2, 1, 2]], {(1, 2)}) == [[(1, 2)]]


# ── the false-edge regression ─────────────────────────────────────────


def rooms_a_door_sees_across_a_wall():
    """Three rooms; the door's annulus reaches one of them through a wall.

    Room 1 and room 2 meet along a real doorway at column 25. Room 3 is
    below room 1 behind a solid wall and beside room 2 behind another,
    touching neither — but it has cells inside the door's match annulus,
    which is the geometry that used to link R11 to R16 with a wall
    between them.

    Returns:
        ``(room_lbl, door_cell)``.
    """
    room_lbl = np.zeros((45, 60), dtype=np.int32)
    room_lbl[5:19, 5:25] = 1        # room 1, left and above
    room_lbl[5:31, 27:51] = 2       # room 2, right of the wall
    room_lbl[14:18, 25] = 1         # the doorway: the two rooms meet here
    room_lbl[14:18, 26] = 2
    room_lbl[24:36, 5:25] = 3       # room 3, behind a 5-cell wall
    return room_lbl, (25, 15)


def test_a_door_never_links_a_room_across_a_wall():
    room_lbl, door = rooms_a_door_sees_across_a_wall()

    proximity = link_doors(room_lbl, [door], cut_radius_cells=2,
                           match_radius_cells=12)
    assert proximity == [[1, 2, 3]], \
        "the annulus is supposed to see the room across the wall"

    adjacency = room_adjacency(room_lbl)
    assert adjacency == {(1, 2)}
    assert door_room_pairs(proximity, adjacency) == [[(1, 2)]], \
        "an edge survived to a room with a wall in the way"


@pytest.mark.skipif(not FIXTURE.is_file(),
                    reason="captured BEV fixture %s is missing" % (FIXTURE,))
def test_the_real_bev_loses_its_edges_through_walls():
    """The regression on the captured hospital BEV.

    MEASURED with the flown parameters (0.15 m cells, 1.60 m door cut,
    0.90 m match radius, so an 11..13 cell annulus): the 35 doors
    propose 61 distinct room pairs across 29 rooms, of which 4 join
    rooms that never touch. Those 4 are the edges an operator sees
    crossing a wall.
    """
    with np.load(str(FIXTURE)) as data:
        grid = data["grid"]
        res = float(data["res"])
        doors = [(int(cx), int(cy)) for cx, cy in data["door_cells"]]
    free = (grid >= 0) & (grid <= FREE_MAX)
    room_lbl, _, stats = segment_rooms_watershed(
        free, res, WatershedRoomParams(), doors)
    assert stats, "the captured BEV produced no rooms at all"

    cut = max(1, int(round(1.60 / res)))
    match = max(cut + 2, int(round(0.90 / res)))
    links = link_doors(room_lbl, doors, cut, match)
    adjacency = room_adjacency(room_lbl)
    vetted = door_room_pairs(links, adjacency)

    proposed = {(a, b) for rooms in links
                for i, a in enumerate(sorted(rooms))
                for b in sorted(rooms)[i + 1:]}
    kept = {pair for pairs in vetted for pair in pairs}
    assert kept <= adjacency
    assert kept < proposed, (
        "the adjacency constraint removed nothing; measured 4 of 61 "
        "proposed pairs crossing a wall, got %d of %d"
        % (len(proposed - kept), len(proposed)))
    assert len(proposed - kept) < 0.25 * len(proposed), (
        "measured 4 of 61 proposed pairs removed, got %d of %d -- that is "
        "not pruning, that is a different segmentation"
        % (len(proposed - kept), len(proposed)))
