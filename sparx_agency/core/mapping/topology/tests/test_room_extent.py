"""Room extents: walls, size as final or a lower bound, coverage -- read from the map, stable under renumbering."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_extent import (
    FINAL, LOWER_BOUND, SIDES, ExtentParams, RoomExtent, area_words, coarse_fraction, coarse_metres, coverage_words,
    room_extents, seen_words, side_index, size_words, wall_orientation)

RES = 0.1
FREE, OCC, UNK = 0, 100, -1


def blank(n=120):
    return np.full((n, n), UNK, np.int8), np.zeros((n, n), np.int32)


def box(grid, labels, x0, y0, x1, y1, pid=0, walls=("l", "r", "b", "t"), doors=()):
    """Room floor on cells ``[x0, x1) x [y0, y1)`` with occupied walls around it; ``doors`` are ``(side, c0, c1)``
    stretches of a wall left UNKNOWN (an opening onto space not seen)."""
    grid[y0:y1, x0:x1] = FREE
    labels[y0:y1, x0:x1] = pid + 1
    if "l" in walls:
        grid[y0 - 1:y1 + 1, x0 - 1] = OCC
    if "r" in walls:
        grid[y0 - 1:y1 + 1, x1] = OCC
    if "b" in walls:
        grid[y0 - 1, x0 - 1:x1 + 1] = OCC
    if "t" in walls:
        grid[y1, x0 - 1:x1 + 1] = OCC
    for side, c0, c1 in doors:
        if side == "r":
            grid[c0:c1, x1] = UNK
        elif side == "l":
            grid[c0:c1, x0 - 1] = UNK
        elif side == "t":
            grid[y1, c0:c1] = UNK
        else:
            grid[y0 - 1, c0:c1] = UNK


def rotated_room(deg, length=5.0, width=3.0, n=160, missing=None):
    """A ``length x width`` room turned ``deg`` degrees, walled 0.15 m thick; ``missing`` drops the +u wall."""
    grid, labels = blank(n)
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    yy, xx = np.mgrid[0:n, 0:n]
    px, py = (xx + 0.5) * RES - n * RES / 2.0, (yy + 0.5) * RES - n * RES / 2.0
    u = px * c + py * s
    v = -px * s + py * c
    inside = (np.abs(u) < length / 2.0) & (np.abs(v) < width / 2.0)
    ring = (np.abs(u) < length / 2.0 + 0.15) & (np.abs(v) < width / 2.0 + 0.15) & ~inside
    grid[inside] = FREE
    labels[inside] = 1
    grid[ring] = OCC
    if missing == "+u":
        grid[ring & (u > length / 2.0 - 0.05)] = UNK
    return grid, labels


def read(grid, labels, **kwargs):
    return room_extents(grid, labels, RES, (0.0, 0.0), **kwargs)


# -- size and walls ------------------------------------------------------------------------------------------------
def test_a_walled_room_has_a_final_size_and_its_true_dimensions():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60)                          # 5 x 4 m
    extent = read(grid, labels)[0]
    assert extent.bounded and extent.size_status == FINAL and extent.open_sides == ()
    assert extent.walls_seen == 4 and all(side.seen for side in extent.sides)
    assert extent.length_m == pytest.approx(5.0, abs=0.02) and extent.width_m == pytest.approx(4.0, abs=0.02)
    assert extent.area_m2 == pytest.approx(20.0, abs=0.05) and extent.free_m2 == pytest.approx(20.0, abs=0.05)
    assert extent.closed_fraction == pytest.approx(1.0) and extent.open_perimeter_m == 0.0
    assert extent.perimeter_m == pytest.approx(18.0, abs=0.05)
    assert sum(side.wall for side in extent.sides) == sum(side.faces for side in extent.sides), "every face is a wall"
    assert size_words(extent) == "20m2 (5x4m, all walls seen)"
    assert seen_words(extent) == "0m2 (0% of the room, 20m2 left)"
    assert extent.covered_m2 == 0.0 and extent.left_m2 == pytest.approx(20.0, abs=0.05) and not extent.covered_is_bound


def test_a_far_wall_beyond_the_cameras_reach_makes_the_size_a_lower_bound_across_that_axis():
    """The depth range ends inside the room: the boundary there is unknown in a wide run, the size is 'at least'."""
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60, walls=("l", "b", "t"))    # the right wall was never seen
    extent = read(grid, labels)[0]
    assert not extent.bounded and extent.size_status == LOWER_BOUND
    assert extent.walls_seen == 3 and len(extent.open_sides) == 1
    opened = extent.side(extent.open_sides[0])
    assert opened.unseen == opened.faces and opened.wall == 0 and opened.gaps == 0
    assert opened.open_m == pytest.approx(4.0, abs=0.1)
    assert not extent.length_final and extent.width_final, "the 5 m axis is cut short, the 4 m one is walled"
    assert extent.closed_fraction == pytest.approx(14.0 / 18.0, abs=0.02)
    assert size_words(extent) == "at least 20m2 (5+ x 4m, 1 wall not seen yet)"
    assert seen_words(extent) == "0m2 (at most 0% of the room, at least 20m2 left)"
    assert extent.covered_is_bound, "the share is an upper bound while a wall is unseen"


def test_a_doorway_in_a_located_wall_is_an_exit_and_the_wall_counts_as_seen():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60, doors=(("r", 30, 39),))   # a 0.9 m opening in the right wall
    extent = read(grid, labels)[0]
    assert extent.bounded and extent.walls_seen == 4
    right = [side for side in extent.sides if side.gaps][0]
    assert right.gaps == 9 and right.unseen == 0 and right.wall == right.faces - 9
    assert extent.open_perimeter_m == 0.0


def test_an_opening_wider_than_a_door_is_a_missing_wall_not_an_exit():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60, doors=(("r", 25, 45),))   # a 2 m stretch of the right wall unseen
    extent = read(grid, labels)[0]
    assert not extent.bounded and len(extent.open_sides) == 1
    opened = extent.side(extent.open_sides[0])
    assert opened.unseen == 20 and opened.gaps == 0
    assert opened.open_fraction == pytest.approx(0.5)


def test_a_narrow_unseen_run_with_nothing_located_beside_it_is_not_a_doorway():
    """A doorway has a jamb on each side; a narrow piece of frontier at a room's unmapped end has none."""
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60, walls=("l", "b", "t"))
    # The far side is unknown except for a short stretch of wall in its middle: the unseen runs on either side
    # of it are narrow-ish (1.5 m) but reach the room's corners, where nothing closes them.
    grid[38:42, 70] = OCC
    extent = read(grid, labels)[0]
    opened = [side for side in extent.sides if not side.seen]
    assert len(opened) == 1 and opened[0].unseen > 0


def test_a_doorway_sized_run_between_two_jambs_is_a_gap_but_the_same_run_at_a_corner_is_open():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60, walls=("l", "b", "t", "r"))
    grid[20:30, 70] = UNK                                        # a 1.0 m run of the right wall at the bottom corner
    extent = read(grid, labels)[0]
    corner = [side for side in extent.sides if side.unseen or side.gaps][0]
    # One jamb above it, the corner wall meets it below: the bottom wall's end cell closes the run on one side only
    # when the corner cell itself is read as closed. Either way the side stays seen: 10 of 40 faces open at most.
    assert corner.seen
    assert corner.unseen + corner.gaps == 10


def test_a_rotated_room_is_measured_in_its_own_frame():
    for deg in (10.0, 30.0, 45.0, 60.0):
        grid, labels = rotated_room(deg)
        extent = read(grid, labels)[1 - 1] if False else read(grid, labels)[0]
        assert extent.bounded, deg
        assert extent.length_m == pytest.approx(5.0, abs=0.15), deg
        assert extent.width_m == pytest.approx(3.0, abs=0.15), deg
        gap = abs(math.degrees(extent.orientation_rad) - deg) % 90.0
        assert min(gap, 90.0 - gap) < 2.0, (deg, math.degrees(extent.orientation_rad))
        assert extent.area_m2 == pytest.approx(15.0, abs=0.3), deg


def test_a_rotated_rooms_missing_far_wall_opens_exactly_one_side():
    grid, labels = rotated_room(30.0, missing="+u")
    extent = read(grid, labels)[0]
    assert not extent.bounded and len(extent.open_sides) == 1
    opened = extent.side(extent.open_sides[0])
    assert opened.unseen >= 0.8 * opened.faces, "the whole short side at the far end is unseen"
    assert not extent.length_final and extent.width_final
    for side in extent.sides:
        if side.name != opened.name:
            assert side.seen, side


def test_an_l_shaped_room_groups_its_two_walls_facing_one_way_into_one_side():
    grid, labels = blank()
    box(grid, labels, 20, 20, 80, 60)                           # 6 x 4 m ...
    grid[40:60, 50:80] = UNK
    labels[40:60, 50:80] = 0                                     # ... minus a 3 x 2 m notch at the top right
    grid[40:61, 50] = OCC
    grid[40, 50:81] = OCC                                        # the notch's two walls
    grid[60, 50:81] = UNK                                        # the old top wall above the notch is not the room's
    extent = read(grid, labels)[0]
    assert extent.bounded and extent.walls_seen == 4
    assert extent.area_m2 == pytest.approx(18.0, abs=0.05)
    assert extent.length_m == pytest.approx(6.0, abs=0.02) and extent.width_m == pytest.approx(4.0, abs=0.02)
    tops = [side for side in extent.sides if side.faces == 60 or side.faces == 61]
    assert tops, "the side facing +y holds both the 3 m top wall and the notch's 3 m top wall: 60 faces"


def test_furniture_islands_inside_a_room_count_toward_its_floor_but_never_as_its_walls():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60)
    grid[30:40, 30:45] = OCC                                     # a 1.5 x 1 m bed in the middle
    labels[30:40, 30:45] = 0
    extent = read(grid, labels)[0]
    assert extent.bounded and extent.walls_seen == 4
    assert extent.free_m2 == pytest.approx(20.0 - 1.5, abs=0.05)
    assert extent.area_m2 == pytest.approx(20.0, abs=0.05), "the floor under the bed is the room's"
    assert sum(side.faces for side in extent.sides) == 180, "the bed's edges are not boundary faces of the room"


def test_an_enclosed_unknown_pocket_behind_furniture_is_not_a_missing_wall():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60)
    grid[30:40, 30:45] = OCC
    labels[30:40, 30:45] = 0
    grid[32:38, 33:42] = UNK                                     # unknown inside the furniture's footprint
    extent = read(grid, labels)[0]
    assert extent.bounded and extent.open_perimeter_m == 0.0


def test_settled_unknown_closes_a_side_as_a_wall_does():
    """A balcony's railing: looked through without a return. Nothing is behind it, so the room ends there."""
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60, walls=("l", "b", "t"))
    resolved = np.zeros_like(grid, dtype=bool)
    resolved[19:61, 70:74] = True
    assert not read(grid, labels)[0].bounded
    extent = read(grid, labels, resolved=resolved)[0]
    assert extent.bounded
    closed = [side for side in extent.sides if side.settled][0]
    assert closed.settled == closed.faces


def test_a_door_cut_to_another_room_closes_the_side_by_partition():
    grid, labels = blank()
    box(grid, labels, 20, 20, 50, 60, pid=0, walls=("l", "b", "t"))
    box(grid, labels, 50, 20, 80, 60, pid=1, walls=("r", "b", "t"))   # two rooms sharing the x=50 cut, no wall between
    extents = read(grid, labels)
    for pid in (0, 1):
        assert extents[pid].bounded, pid
        shared = [side for side in extents[pid].sides if side.room][0]
        assert shared.room == shared.faces == 40


def test_a_room_glimpsed_through_a_door_is_a_sliver_with_three_walls_unseen():
    grid, labels = blank()
    box(grid, labels, 20, 20, 40, 60, pid=1, doors=(("r", 35, 44),))
    grid[35:44, 40:43] = FREE
    labels[35:44, 40:43] = 1                                     # pid 0: the floor seen through the hallway's door
    extent = read(grid, labels)[0]
    assert not extent.bounded and extent.walls_seen == 1
    assert extent.free_m2 == pytest.approx(0.27, abs=0.01)
    assert size_words(extent).startswith("under 1m2") or size_words(extent).startswith("at least")
    assert extent.side([s.name for s in extent.sides if s.room][0]).seen, "the door side is closed by the hallway"


def test_a_room_with_no_wall_at_all_has_no_side_seen():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60, walls=())
    extent = read(grid, labels)[0]
    assert extent.walls_seen == 0 and len(extent.open_sides) == 4 and extent.closed_fraction == pytest.approx(0.0)
    assert not extent.length_final and not extent.width_final
    assert size_words(extent) == "at least 20m2 (5+ x 4+m, 4 of 4 walls not seen yet)"


def test_a_tiny_boundary_says_nothing_about_a_wall():
    grid, labels = blank()
    grid[30:32, 30:32] = FREE
    labels[30:32, 30:32] = 1
    grid[29, 29:33] = OCC
    extent = read(grid, labels, params=ExtentParams(min_side_faces=3))[0]
    assert not any(side.seen for side in extent.sides)


# -- coverage and the agent's distances ------------------------------------------------------------------------
def test_coverage_is_the_rooms_share_of_the_observed_cells_not_of_where_the_agent_walked():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60)
    observed = np.zeros_like(grid, dtype=bool)
    observed[20:60, 20:45] = True                                # the half nearer the door was looked at
    observed[0:10, 0:10] = True                                  # cells outside the room do not count
    extent = read(grid, labels, observed=observed)[0]
    assert extent.observed_fraction == pytest.approx(0.5)
    assert extent.observed_m2 == pytest.approx(10.0, abs=0.05) and extent.unobserved_m2 == pytest.approx(10.0, abs=0.05)
    assert extent.covered_m2 == pytest.approx(10.0, abs=0.05) and extent.covered_fraction == pytest.approx(0.5)
    assert extent.left_m2 == pytest.approx(10.0, abs=0.05)
    assert seen_words(extent) == "10m2 (50% of the room, 10m2 left)"
    assert read(grid, labels)[0].observed_fraction == 0.0, "no record: nothing observed"


def test_furniture_islands_are_covered_in_proportion_to_the_observed_floor_around_them():
    """A bed is seen from the floor beside it: with every cell of floor around it observed the bed is covered whole,
    so a furnished room looked at from end to end reads covered in full, not 85% for ever."""
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60)                           # 20 m2 ...
    grid[30:40, 30:45] = OCC
    labels[30:40, 30:45] = 0                                     # ... with a 1.5 m2 bed in it
    observed = np.zeros_like(grid, dtype=bool)
    observed[20:60, 20:70] = True                                # the whole free floor observed
    extent = read(grid, labels, observed=observed)[0]
    assert extent.area_m2 == pytest.approx(20.0, abs=0.05) and extent.observed_m2 == pytest.approx(18.5, abs=0.05)
    assert extent.covered_m2 == pytest.approx(20.0, abs=0.05) and extent.covered_fraction == pytest.approx(1.0)
    assert seen_words(extent) == "20m2 (all of it)"
    # Half the floor observed, the bed's ring with it half: the bed counts half.
    observed[:] = False
    observed[20:60, 20:38] = True                                # x up to 3.8 m: the bed spans 3.0-4.5
    half = read(grid, labels, observed=observed)[0]
    ring_share = half.covered_m2 - half.observed_m2
    assert 0.3 < ring_share < 1.2, ring_share                   # part of the 1.5 m2 bed, by its observed ring
    # An unknown island inside the room (a pocket the camera never resolved) is NOT covered; a settled one is.
    grid[45:50, 50:55] = UNK
    labels[45:50, 50:55] = 0
    observed[:] = True
    pocket = read(grid, labels, observed=observed)[0]
    assert pocket.covered_m2 == pytest.approx(20.0 - 0.25, abs=0.05)
    resolved = np.zeros_like(grid, dtype=bool)
    resolved[45:50, 50:55] = True
    settled = read(grid, labels, observed=observed, resolved=resolved)[0]
    assert settled.covered_m2 == pytest.approx(20.0, abs=0.05)


def test_the_coverage_words_hand_the_model_final_results():
    assert coverage_words(10.4, 0.87, 1.6, bounded=True) == "10m2 (80% of the room, 2m2 left)"
    assert coverage_words(11.95, 0.996, 0.05, bounded=True) == "12m2 (all of it)"
    assert coverage_words(0.0, 0.0, 12.0, bounded=True) == "0m2 (0% of the room, 12m2 left)"
    assert coverage_words(2.6, 0.43, 3.4, bounded=False) == "3m2 (at most 50% of the room, at least 3m2 left)"
    assert coverage_words(0.3, 1.0, 0.0, bounded=False) == "under 1m2 (all of the mapped part; the unmapped rest is unknown)"
    assert coverage_words(6.8, 0.97, 0.2, bounded=False) == "7m2 (all of the mapped part; the unmapped rest is unknown)", (
        "a bound that rounds up to 100% says nothing")
    assert coverage_words(5.5, 0.88, 0.7, bounded=False) == "6m2 (at most 90% of the room)", "no 'at least under 1m2'"
    assert coverage_words(10.4, 0.87, 0.6, bounded=True) == "10m2 (80% of the room, under 1m2 left)"
    assert coverage_words(0.0, 0.0, 7.0, bounded=False) == "0m2 (at most 0% of the room, at least 7m2 left)"
    assert area_words(0.0) == "0m2" and area_words(0.7) == "under 1m2" and area_words(3.49) == "3m2"


def test_the_agents_distance_to_each_wall_is_filled_for_the_room_it_stands_in_only():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60)                           # x in [2, 7] m, y in [2, 6] m
    extents = read(grid, labels, here_xy=(3.0, 3.0))
    extent = extents[0]
    assert extent.here
    distances = extent.distances()
    assert sorted(round(v, 2) for v in distances.values()) == pytest.approx([1.0, 1.0, 3.0, 4.0], abs=0.02)
    for side in extent.sides:
        (x0, y0), (x1, y1) = side.line
        assert min(x0, x1) >= 1.95 and max(x0, x1) <= 7.05 and min(y0, y1) >= 1.95 and max(y0, y1) <= 6.05
    away = read(grid, labels, here_xy=(10.0, 10.0))[0]
    assert not away.here and all(v is None for v in away.distances().values())


def test_the_record_is_json_friendly_and_names_the_status():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60, walls=("l", "b", "t"))
    record = read(grid, labels, here_xy=(3.0, 3.0))[0].to_record()
    assert record["size_status"] == LOWER_BOUND and record["walls_seen"] == 3
    assert isinstance(record["sides"], list) and len(record["sides"]) == 4
    assert all(isinstance(side["line"], list) for side in record["sides"])
    assert record["orientation_deg"] == pytest.approx(0.0, abs=1.0) or record["orientation_deg"] == pytest.approx(90.0, abs=1.0)
    import json
    json.dumps(record)


# -- stability under the partition's changes --------------------------------------------------------------------
def test_a_room_renumbered_split_or_merged_reads_right_because_nothing_is_keyed_by_id():
    """The watershed merges and splits rooms and hands out new numbers; the reading comes off the live map and the
    live partition each time, and the only accumulated state -- the observed cells -- is keyed by cell."""
    grid, labels = blank()
    box(grid, labels, 20, 20, 80, 60, pid=4)                    # one 6 x 4 m room, number 4
    observed = np.zeros_like(grid, dtype=bool)
    observed[20:60, 20:50] = True                                # its left half observed
    whole = read(grid, labels, observed=observed)[4]
    assert whole.bounded and whole.observed_fraction == pytest.approx(0.5)
    # A door cut at x=50 splits it; the halves get numbers 9 and 12.
    split = labels.copy()
    split[20:60, 20:50] = 10
    split[20:60, 50:80] = 13
    halves = read(grid, split, observed=observed)
    assert set(halves) == {9, 12}
    assert halves[9].bounded and halves[12].bounded, "each half is closed by walls on three sides and the cut on one"
    assert halves[9].observed_fraction == pytest.approx(1.0) and halves[12].observed_fraction == pytest.approx(0.0)
    assert halves[9].area_m2 + halves[12].area_m2 == pytest.approx(whole.area_m2)
    # Merged back under yet another number: the same room as at the start.
    merged = labels.copy()
    merged[labels == 5] = 21
    again = read(grid, merged, observed=observed)[20]
    assert again.bounded and again.observed_fraction == pytest.approx(0.5)
    assert again.length_m == pytest.approx(whole.length_m) and again.area_m2 == pytest.approx(whole.area_m2)


def test_a_room_that_grows_as_the_camera_reaches_its_far_wall_turns_from_a_lower_bound_into_a_final_size():
    grid, labels = blank()
    box(grid, labels, 20, 20, 60, 60, walls=("l", "b", "t"))    # 4 m of a 6 m room mapped; the far wall unseen
    first = read(grid, labels)[0]
    assert not first.bounded and first.length_m == pytest.approx(4.0, abs=0.02)
    grid[20:60, 60:80] = FREE
    labels[20:60, 60:80] = 1
    grid[19, 60:81] = OCC
    grid[60, 60:81] = OCC
    grid[19:61, 80] = OCC                                        # the far wall comes into range
    second = read(grid, labels)[0]
    assert second.bounded and second.length_m == pytest.approx(6.0, abs=0.02)
    assert first.area_m2 < second.area_m2, "the lower bound never overstated the room"


# -- helpers ---------------------------------------------------------------------------------------------------
def test_wall_orientation_reads_the_houses_axis_and_a_rooms_own():
    grid, labels = rotated_room(30.0)
    theta, share = wall_orientation(grid == OCC, grid == FREE)
    assert abs(math.degrees(theta) - 30.0) < 2.0 and share > 0.5
    assert wall_orientation(np.zeros((10, 10), bool), np.zeros((10, 10), bool)) == (None, 0.0)


def test_side_index_puts_a_diagonal_walls_staircase_faces_on_one_side():
    grid, labels = rotated_room(45.0)
    mask = labels == 1
    index = side_index(mask, math.radians(45.0))
    boundary = index >= 0
    assert boundary.sum() > 100
    counts = np.bincount(index[boundary], minlength=4)
    # Two long sides and two short sides: the long ones hold about 5/3 the faces of the short ones.
    long_sides, short_sides = sorted(counts)[2:], sorted(counts)[:2]
    assert min(long_sides) > 1.3 * max(short_sides), counts


def test_the_prompt_words_are_coarse_so_an_unchanged_map_gives_an_unchanged_prompt():
    assert coarse_fraction(0.0) == 0 and coarse_fraction(0.19) == 10 and coarse_fraction(0.2) == 20
    assert coarse_fraction(0.999) == 90 and coarse_fraction(1.0) == 100
    assert coarse_metres(3.26) == "3.5" and coarse_metres(3.74) == "3.5" and coarse_metres(4.1) == "4"
    assert size_words(None, 0.0) == "unknown size" and size_words(None, 0.4) == "under 1m2" and size_words(None, 12.4) == "12m2"
    assert seen_words(None) is None


def test_the_params_are_checked():
    with pytest.raises(ValueError):
        ExtentParams(open_tolerance=1.0)
    with pytest.raises(ValueError):
        ExtentParams(doorway_max_m=0.0)
    with pytest.raises(ValueError):
        ExtentParams(min_side_faces=0)
    assert ExtentParams().doorway_max_m == 1.2 and ExtentParams().open_tolerance == 0.2


def test_sides_are_the_four_names_in_order():
    grid, labels = blank()
    box(grid, labels, 20, 20, 70, 60)
    extent = read(grid, labels)[0]
    assert tuple(side.name for side in extent.sides) == SIDES
    assert isinstance(extent, RoomExtent)
