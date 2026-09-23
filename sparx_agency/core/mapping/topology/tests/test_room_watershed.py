# core/mapping/topology/tests/test_room_watershed.py
"""Tests for the clearance-watershed room segmenter.

Two halves. The synthetic half pins the contract the node depends on:
geometry alone splits what the skeleton cut cannot, a listed door
separates whatever the geometry says, no free cell is orphaned by the
carve, runts are dropped, and ids/centroids match ``compute_rooms``.

A third group pins the over-segmentation repair (``merge_dynamics_m``):
two lobes joined by a wide neck become one room, a narrow doorway
survives, a listed door is never merged across whatever the geometry
says, and 0.0 reproduces the raw watershed cell for cell.

The regression half replays the REAL captured FALCON BEV that motivated
the module (413x200 @ 0.15 m, ``fixtures/live_bev_hospital.npz``) and
asserts the property that was failing: the largest room must stay a
minority of the segmented area. The flown door-cut pipeline reaches 86%
on this same grid, which is what "collapsed into one room" means. The
fixture also carries ``door_cells`` — the 35 entries of
``robots/SJTU/maps/hospital_doors.yaml`` resolved into this grid through
its own ``ox``/``oy``/``res`` — so the regression can replay the
configuration that actually flies rather than a door-free one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed

from sparx_agency.core.mapping.topology import room_watershed
from sparx_agency.core.mapping.topology.room_segmentation import (
    RoomSegmentationParams,
    compute_rooms,
    door_disk_mask,
    heal_free_mask,
)
from sparx_agency.core.mapping.topology.room_merge import (
    merge_basins_by_dynamics,
)
from sparx_agency.core.mapping.topology.room_watershed import (
    WatershedRoomParams,
    segment_rooms_watershed,
)

RES = 0.15  # metres per cell, the flown BEV resolution
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE = FIXTURES / "live_bev_hospital.npz"


def two_chambers_and_neck():
    """Two 40x40 chambers joined by a 6-cell-wide neck.

    Wide enough that the medial axis stays one component through the
    neck (so ``compute_rooms`` with no doors reads one room), narrow
    enough that the clearance field has a saddle there.
    """
    free = np.zeros((60, 110), dtype=bool)
    free[10:50, 5:45] = True     # left chamber
    free[10:50, 65:105] = True   # right chamber
    free[27:33, 45:65] = True    # neck
    return free


def one_narrow_hall():
    """A single 20x30-cell hall — one clearance basin, no saddle.

    Sized so a ``door_cut_m`` disk (1.60 m = 11 cells at 0.15 m) spans
    its 20-cell width and severs it, while both halves stay above
    ``min_room_cells``.
    """
    free = np.zeros((60, 60), dtype=bool)
    free[10:30, 5:35] = True
    return free


def chamber_corridor_closet():
    """A big chamber, a thin corridor, and a 10x10 closet on the end.

    The corridor is 4 cells wide (0.3 m clearance, below
    ``min_clearance_m``) so it seeds nothing; the closet does seed, and
    its basin comes to 178 cells — a runt or a room depending on the
    floor under test.
    """
    free = np.zeros((70, 120), dtype=bool)
    free[10:50, 5:45] = True    # 1600-cell chamber
    free[28:32, 45:85] = True   # corridor
    free[25:35, 85:95] = True   # closet
    return free


def two_lobes_and_a_wide_neck():
    """One room the watershed splits in two: two lobes, a wide neck.

    Two 20x20 lobes joined by a 14-cell (2.1 m) neck — furniture in the
    middle of a small room, or an L-bend. Each lobe seeds its own
    clearance peak at 1.50 m and the saddle in the neck is 1.05 m, so
    the DYNAMICS are 0.45 m: below the 0.50 m default, above nothing
    else. Compare ``two_chambers_and_neck``, whose 3.00 m peaks over a
    0.45 m saddle give 2.55 m of dynamics and never merge.
    """
    return _lobes(neck=14)


def two_lobes_and_a_narrow_neck():
    """The same room with a 12-cell (1.8 m) neck: dynamics 0.60 m.

    Two cells narrower than :func:`two_lobes_and_a_wide_neck` and the
    pair is already above the default threshold, which is what "the
    knob discriminates" means at this scale.
    """
    return _lobes(neck=12)


def _lobes(neck, lobe=20, gap=14, pad=8):
    """Two square lobes joined by a neck of ``neck`` cells.

    Args:
        neck: Neck width in cells; the only knob the tests vary.
        lobe: Side of each square lobe, in cells.
        gap: Length of the neck, in cells.
        pad: Free border around the shape, in cells.

    Returns:
        (H, W) bool free mask.
    """
    free = np.zeros((lobe + 2 * pad, 2 * lobe + gap + 2 * pad), dtype=bool)
    free[pad:pad + lobe, pad:pad + lobe] = True
    free[pad:pad + lobe, pad + lobe + gap:pad + 2 * lobe + gap] = True
    top = pad + (lobe - neck) // 2
    free[top:top + neck, pad + lobe:pad + lobe + gap] = True
    return free


# Dead centre of the neck built by :func:`_lobes` with its defaults.
WIDE_NECK_DOOR = (35, 18)


def unmerged_reference(free, resolution, params, doors):
    """The watershed pipeline with the merge stage physically absent.

    Runs ``segment_rooms_watershed``'s own stages by hand, stopping
    short of :func:`merge_basins_by_dynamics`, so "0.0 disables merging"
    can be checked against something other than itself.

    Args:
        free: (H, W) bool free mask.
        resolution: Metres per cell.
        params: The same knobs the segmenter is called with.
        doors: Door cells as ``(cx, cy)`` pairs.

    Returns:
        The ``(room_lbl, stats)`` pair the segmenter would return.
    """
    healed = heal_free_mask(free)
    clearance = distance_transform_edt(healed) * float(resolution)
    carve = door_disk_mask(healed.shape, doors,
                           int(round(params.door_cut_m / resolution)))
    flood = healed & ~carve
    markers, _ = room_watershed._clearance_markers(
        clearance, flood, params, resolution)
    labels = room_watershed._reclaim_carved(
        watershed(-clearance, markers, mask=flood), healed)
    return room_watershed._collect_rooms(labels, params.min_room_cells)


def load_live_bev():
    """The captured hospital BEV as ``(free_mask, resolution, doors)``.

    ``doors`` is the flown 35-entry hospital door list already resolved
    into this grid's cells, so the regression replays the configuration
    the node actually runs rather than a door-free one.
    """
    with np.load(str(FIXTURE)) as data:
        grid = data["grid"]
        res = float(data["res"])
        doors = [(int(cx), int(cy)) for cx, cy in data["door_cells"]]
    # nav_msgs semantics: -1 unknown, 0..49 free, >=50 occupied.
    return (grid >= 0) & (grid <= 49), res, doors


def test_geometry_splits_what_the_skeleton_cut_cannot():
    """The neck reads as two rooms; the flown segmenter sees one."""
    free = two_chambers_and_neck()

    _, _, doors_stats = compute_rooms(
        free, [], RoomSegmentationParams(min_room_cells=150))
    assert len(doors_stats) == 1

    room_lbl, _, stats = segment_rooms_watershed(free, RES)
    assert len(stats) == 2
    left = int(room_lbl[30, 20])
    right = int(room_lbl[30, 90])
    assert left > 0 and right > 0
    assert left != right


def test_a_listed_door_forces_a_split_geometry_would_not_make():
    """One hall is one room until a door cell cuts it in two.

    Nothing in the clearance field suggests a boundary here — the split
    exists only because the door is listed.
    """
    free = one_narrow_hall()
    _, _, plain = segment_rooms_watershed(free, RES)
    assert len(plain) == 1

    # A door on the hall's mid-line, (cx, cy) = (col, row).
    room_lbl, _, split = segment_rooms_watershed(
        free, RES, WatershedRoomParams(), [(20, 20)])
    assert len(split) == 2
    assert room_lbl[20, 10] != room_lbl[20, 30]


def test_the_door_carve_orphans_no_free_cell():
    """Every healed free cell carries a label after the carve reclaim."""
    free = two_chambers_and_neck()
    door = (55, 30)  # dead centre of the neck
    room_lbl, _, stats = segment_rooms_watershed(
        free, RES, WatershedRoomParams(), [door])
    assert len(stats) == 2

    healed = heal_free_mask(free)
    assert not (healed & (room_lbl == 0)).any()
    # Including the carved cells themselves, which the reclaim hands back.
    assert room_lbl[door[1], door[0]] > 0


def test_rooms_under_the_floor_are_dropped():
    """The same closet is a room under one floor and a runt under another.

    Merging is pinned OFF so this measures the size floor alone. It is
    also the honest cost of the merge stage: the closet's peak clearance
    is 0.75 m over a 0.30 m corridor saddle, i.e. 0.45 m of dynamics, so
    at the 0.50 m default it is absorbed into the chamber instead of
    surviving as a runt — which the next test pins.
    """
    free = chamber_corridor_closet()
    off = dict(merge_dynamics_m=0.0)

    kept_lbl, _, kept = segment_rooms_watershed(
        free, RES, WatershedRoomParams(min_room_cells=150, **off))
    assert len(kept) == 2
    assert kept_lbl[30, 90] > 0

    floor = 250
    dropped_lbl, _, dropped = segment_rooms_watershed(
        free, RES, WatershedRoomParams(min_room_cells=floor, **off))
    assert len(dropped) == 1
    assert all(s.n_cells >= floor for s in dropped)
    assert not any(s.mask[30, 90] for s in dropped)
    assert dropped_lbl[30, 90] == 0


def test_the_merge_runs_before_the_size_floor():
    """An absorbed lobe keeps its cells instead of being orphaned.

    With merging on, the closet joins the chamber rather than being
    dropped as a runt, so its 178 cells stay in a room. Ordering the
    merge after the floor would have thrown them away.
    """
    free = chamber_corridor_closet()
    lbl, _, stats = segment_rooms_watershed(
        free, RES, WatershedRoomParams(min_room_cells=250))
    assert len(stats) == 1
    assert lbl[30, 90] == stats[0].id
    assert stats[0].n_cells == 1856


def test_stats_ids_are_compact_and_centroids_are_cx_cy():
    """Same id/mask/n_cells/centroid semantics as ``compute_rooms``."""
    free = two_chambers_and_neck()
    room_lbl, _, stats = segment_rooms_watershed(free, RES)

    assert [s.id for s in stats] == list(range(1, len(stats) + 1))
    for s in stats:
        assert s.n_cells == int(s.mask.sum())
        assert np.array_equal(room_lbl == s.id, s.mask)
        cx, cy = s.centroid_cells
        ys, xs = np.where(s.mask)
        assert cx == pytest.approx(xs.mean())
        assert cy == pytest.approx(ys.mean())
        # (cx, cy) is (col, row): the chambers are wider apart in x.
        assert 0 <= cx < free.shape[1]
        assert 0 <= cy < free.shape[0]


def test_the_skeleton_is_non_empty_and_lies_in_the_healed_mask():
    """The RViz 'open space' spine is returned even though rooms ignore it."""
    free = two_chambers_and_neck()
    _, skeleton, stats = segment_rooms_watershed(free, RES)

    assert skeleton.dtype == bool
    assert skeleton.any()
    healed = heal_free_mask(free)
    assert not (skeleton & ~healed).any()
    # Restricted to the surviving rooms, as compute_rooms restricts its own.
    rooms = np.zeros_like(healed)
    for s in stats:
        rooms |= s.mask
    assert not (skeleton & ~rooms).any()


def test_empty_and_all_free_masks_do_not_raise():
    """Degenerate inputs return empties rather than exploding mid-tick."""
    empty = np.zeros((40, 40), dtype=bool)
    room_lbl, skeleton, stats = segment_rooms_watershed(empty, RES)
    assert stats == []
    assert not room_lbl.any()
    assert not skeleton.any()


def test_a_non_positive_resolution_raises():
    with pytest.raises(ValueError):
        segment_rooms_watershed(one_narrow_hall(), 0.0)


def test_segmentation_is_deterministic():
    """Same input twice -> identical labels, masks and centroids."""
    free = two_chambers_and_neck()
    doors = [(55, 30)]
    a_lbl, a_sk, a_stats = segment_rooms_watershed(
        free, RES, WatershedRoomParams(), doors)
    b_lbl, b_sk, b_stats = segment_rooms_watershed(
        free, RES, WatershedRoomParams(), doors)

    assert np.array_equal(a_lbl, b_lbl)
    assert np.array_equal(a_sk, b_sk)
    assert [s.centroid_cells for s in a_stats] == \
        [s.centroid_cells for s in b_stats]


def _shape(stats):
    """``(room count, largest room as a share of the segmented area)``."""
    total = sum(s.n_cells for s in stats) or 1
    return len(stats), max((s.n_cells for s in stats), default=0) / total


@pytest.mark.skipif(not FIXTURE.is_file(),
                    reason="captured BEV fixture %s is missing" % (FIXTURE,))
def test_the_real_bev_does_not_collapse_into_one_room():
    """The regression: on the captured hospital BEV the flown door-cut
    pipeline gives one room covering most of the floor, and this one
    must not — with the merge stage on or off, doors carved or not."""
    free, res, doors = load_live_bev()
    for cut in (doors, ()):
        for merge_m in (0.0, WatershedRoomParams().merge_dynamics_m):
            _, _, stats = segment_rooms_watershed(
                free, res, WatershedRoomParams(merge_dynamics_m=merge_m),
                cut)
            assert stats, "the captured BEV produced no rooms at all"
            count, largest = _shape(stats)
            assert largest < 0.40, (
                "%d doors, merge %.2f m: largest room is %.0f%% of the "
                "segmented area across %d rooms"
                % (len(cut), merge_m, 100 * largest, count))


@pytest.mark.skipif(not FIXTURE.is_file(),
                    reason="captured BEV fixture %s is missing" % (FIXTURE,))
def test_the_real_bev_merges_into_a_sane_room_count():
    """The over-segmentation repair, measured on the real BEV.

    MEASURED with the fixture's 35 doors carved: 43 rooms / largest
    10.7% with the merge off, 29 rooms / largest 12.2% at the 0.50 m
    default, against a ground truth of 20 rooms + 7 corridors = 27
    regions. The bands below are those numbers with room to breathe;
    the point is that merging removes a third of the rooms WITHOUT the
    largest one growing, which is what separates this from the
    saddle-width merge that cascades the floor into one region.
    """
    free, res, doors = load_live_bev()

    _, _, raw = segment_rooms_watershed(
        free, res, WatershedRoomParams(merge_dynamics_m=0.0), doors)
    raw_count, raw_largest = _shape(raw)
    assert raw_count == 43, "measured 43 unmerged rooms, got %d" % raw_count

    _, _, merged = segment_rooms_watershed(free, res, WatershedRoomParams(),
                                           doors)
    count, largest = _shape(merged)
    assert 24 <= count <= 34, (
        "measured 29 rooms at the 0.50 m default (ground truth 27), got %d"
        % count)
    assert count < raw_count, (
        "the merge stage removed nothing: %d rooms either way" % count)
    assert largest < 0.40, (
        "measured a largest room of 12.2%%, got %.1f%% across %d rooms"
        % (100 * largest, count))
    assert largest < raw_largest + 0.05, (
        "merging inflated the largest room from %.1f%% to %.1f%% -- that is "
        "the cascade this algorithm exists to avoid"
        % (100 * raw_largest, 100 * largest))


@pytest.mark.skipif(not FIXTURE.is_file(),
                    reason="captured BEV fixture %s is missing" % (FIXTURE,))
def test_the_real_bev_leaves_no_explored_cell_unassigned():
    """No healed free cell of a real map is orphaned."""
    free, res, doors = load_live_bev()
    room_lbl, _, _ = segment_rooms_watershed(free, res, WatershedRoomParams(),
                                             doors)
    healed = heal_free_mask(free)
    orphans = int((healed & (room_lbl == 0)).sum())
    # Only sub-floor rooms may be dropped; nothing else may be unlabelled.
    assert orphans < 0.02 * int(healed.sum()), \
        "%d of %d healed free cells have no room" % (orphans, healed.sum())


# ── the over-segmentation repair ─────────────────────────────────────

def test_two_wide_lobes_merge_into_one_room():
    """The reported defect: one small room reported as R10 and R11.

    Peaks 1.50 m either side of a 1.05 m saddle -> 0.45 m of dynamics,
    under the 0.50 m default, so the watershed's two basins come back as
    one room covering both lobes.
    """
    free = two_lobes_and_a_wide_neck()

    _, _, split = segment_rooms_watershed(
        free, RES, WatershedRoomParams(merge_dynamics_m=0.0))
    assert len(split) == 2, "the watershed is supposed to over-segment here"

    room_lbl, _, merged = segment_rooms_watershed(free, RES)
    assert len(merged) == 1
    assert merged[0].n_cells == sum(s.n_cells for s in split)
    assert room_lbl[18, 15] == room_lbl[18, 55] > 0


def test_a_narrow_doorway_survives_the_merge():
    """Two rooms joined by a real doorway stay two rooms.

    Both scales: the 12-cell neck (0.60 m of dynamics, just above the
    bar) and the 40x40 chambers of ``two_chambers_and_neck`` (2.55 m,
    nowhere near it). Merging must not touch either.
    """
    narrow_lbl, _, narrow = segment_rooms_watershed(
        two_lobes_and_a_narrow_neck(), RES)
    assert len(narrow) == 2
    assert narrow_lbl[18, 15] != narrow_lbl[18, 55]

    _, _, chambers = segment_rooms_watershed(two_chambers_and_neck(), RES)
    assert len(chambers) == 2


def test_a_door_separated_pair_never_merges():
    """A listed door outranks the geometry, however small the dynamics.

    The same 0.45 m-dynamics pair that merges above stays split once a
    door sits in the neck — and stays split at a threshold ten times the
    default, because the edge is marked unmergeable rather than merely
    expensive.
    """
    free = two_lobes_and_a_wide_neck()
    for threshold in (WatershedRoomParams().merge_dynamics_m, 5.0):
        room_lbl, _, stats = segment_rooms_watershed(
            free, RES, WatershedRoomParams(merge_dynamics_m=threshold),
            [WIDE_NECK_DOOR])
        assert len(stats) == 2, \
            "merge_dynamics_m=%.2f merged across a listed door" % threshold
        assert room_lbl[18, 15] != room_lbl[18, 55]


def test_merging_off_reproduces_the_raw_watershed_exactly():
    """``merge_dynamics_m=0.0`` is an off switch, not a small threshold.

    Compared cell for cell against the pipeline stages run by hand, so
    the merge stage is provably the only difference between the two.
    """
    for free, doors in ((two_lobes_and_a_wide_neck(), []),
                        (two_lobes_and_a_wide_neck(), [WIDE_NECK_DOOR]),
                        (two_chambers_and_neck(), [(55, 30)]),
                        (chamber_corridor_closet(), [])):
        params = WatershedRoomParams(merge_dynamics_m=0.0)
        room_lbl, _, stats = segment_rooms_watershed(free, RES, params, doors)
        want_lbl, want_stats = unmerged_reference(free, RES, params, doors)
        assert np.array_equal(room_lbl, want_lbl)
        assert [(s.id, s.n_cells, s.centroid_cells) for s in stats] == \
            [(s.id, s.n_cells, s.centroid_cells) for s in want_stats]


def test_the_merge_stage_alone_is_a_no_op_when_disabled():
    """The core call returns its input labels for any non-positive bar."""
    labels = np.array([[1, 1, 2], [1, 0, 2], [1, 2, 2]], dtype=np.int32)
    dt = np.full(labels.shape, 1.0)
    for threshold in (0.0, -1.0):
        assert np.array_equal(
            merge_basins_by_dynamics(labels, dt, threshold), labels)


def test_the_merge_stage_rejects_mismatched_grids():
    """A shape mismatch raises rather than broadcasting into nonsense."""
    labels = np.ones((4, 4), dtype=np.int32)
    with pytest.raises(ValueError):
        merge_basins_by_dynamics(labels, np.ones((4, 5)), 0.5)
    with pytest.raises(ValueError):
        merge_basins_by_dynamics(labels, np.ones((4, 4)), 0.5,
                                 np.zeros((5, 4), bool))


def test_the_merged_segmentation_is_deterministic():
    """Same input twice -> identical labels, through the merge stage too."""
    free = two_lobes_and_a_wide_neck()
    runs = [segment_rooms_watershed(free, RES, WatershedRoomParams(),
                                    [WIDE_NECK_DOOR]) for _ in range(2)]
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])
    assert [s.centroid_cells for s in runs[0][2]] == \
        [s.centroid_cells for s in runs[1][2]]
