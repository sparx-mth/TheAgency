"""Tests for the room-label grid contract between mapper and viz.

The viz tints a room by reading a cell value out of
``/scene_graph/room_labels_grid`` and resolving it through the
``grid_pid_map`` of the ``/scene_graph`` JSON. Both halves come from one
``{pid: grid value}`` mapping, and everything that can silently break the
tint lives in that mapping: a value that moves under a room that is still
there (flicker), a value shared by two rooms (wrong colour), a value in the
grid with no entry in the map (the viz falls back to tinting by the raw
value, i.e. by whichever room happens to have that pid).

Pure functions only — no rclpy, so this runs in the plain ``.venv``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.tasks.mapping.scene_graph.ros2.payloads import (
    MAX_ROOM_VALUE,
    MIN_ROOM_VALUE,
    NO_ROOM_VALUE,
    assign_room_grid_values,
    grid_pid_map,
    room_value_grid,
    scene_graph_payload,
)


def _mask(shape, box):
    """A bool mask of ``shape`` with ``box`` = ``(y0, y1, x0, x1)`` set."""
    out = np.zeros(shape, dtype=bool)
    y0, y1, x0, x1 = box
    out[y0:y1, x0:x1] = True
    return out


class TestGridValueAssignment:
    def test_first_tick_hands_out_the_low_values_in_order(self):
        assert assign_room_grid_values([4, 9, 2]) == {4: 1, 9: 2, 2: 3}

    def test_a_persisting_room_keeps_its_value_across_ticks(self):
        first = assign_room_grid_values([0, 1, 2])
        # Same rooms, a different iteration order, plus a newcomer.
        second = assign_room_grid_values([2, 0, 1, 7], first)
        for pid in (0, 1, 2):
            assert second[pid] == first[pid], "room %d re-tinted" % pid
        assert second[7] not in first.values()

    def test_values_are_stable_over_many_ticks_of_churn(self):
        values = assign_room_grid_values([0])
        anchor = values[0]
        for tick in range(1, 40):
            # One long-lived room, one that appears and vanishes each tick.
            pids = [0] + ([100 + tick] if tick % 2 else [])
            values = assign_room_grid_values(pids, values)
            assert values[0] == anchor

    def test_a_value_is_recycled_only_after_its_room_is_gone(self):
        first = assign_room_grid_values([0, 1, 2])
        gone = assign_room_grid_values([0, 2], first)      # room 1 vanishes
        assert 1 not in gone
        assert gone[0] == first[0] and gone[2] == first[2]
        # Its value is free now, so the next new room takes it.
        after = assign_room_grid_values([0, 2, 5], gone)
        assert after[5] == first[1]

    def test_no_two_live_rooms_ever_share_a_value(self):
        values = {}
        for tick in range(60):
            pids = [tick, tick + 1, tick + 2]              # a sliding window
            values = assign_room_grid_values(pids, values)
            assert len(set(values.values())) == len(values)

    def test_every_value_is_inside_the_band(self):
        values = assign_room_grid_values(range(MAX_ROOM_VALUE))
        assert len(values) == MAX_ROOM_VALUE
        assert min(values.values()) == MIN_ROOM_VALUE
        assert max(values.values()) == MAX_ROOM_VALUE

    def test_surplus_rooms_get_no_value_rather_than_a_colliding_one(self):
        values = assign_room_grid_values(range(MAX_ROOM_VALUE + 25))
        assert len(values) == MAX_ROOM_VALUE
        assert len(set(values.values())) == MAX_ROOM_VALUE
        assert all(MIN_ROOM_VALUE <= v <= MAX_ROOM_VALUE
                   for v in values.values())

    def test_an_out_of_band_or_duplicated_previous_value_is_reassigned(self):
        values = assign_room_grid_values([1, 2, 3],
                                         {1: 0, 2: 999, 3: MAX_ROOM_VALUE + 1})
        assert len(set(values.values())) == 3
        assert all(MIN_ROOM_VALUE <= v <= MAX_ROOM_VALUE
                   for v in values.values())
        values = assign_room_grid_values([1, 2], {1: 7, 2: 7})
        assert values[1] == 7 and values[2] != 7

    def test_duplicate_pids_are_collapsed(self):
        assert assign_room_grid_values([3, 3, 3]) == {3: 1}

    def test_no_rooms_is_an_empty_mapping(self):
        assert assign_room_grid_values([]) == {}
        assert assign_room_grid_values([], {4: 2}) == {}


class TestGridPidMap:
    def test_the_map_goes_grid_value_to_pid_keyed_by_string(self):
        # The direction the viz reads: pid_map[str(cell value)] -> pid.
        assert grid_pid_map({7: 1, 130: 2}) == {"1": 7, "2": 130}

    def test_the_map_survives_a_json_round_trip_unchanged(self):
        wire = grid_pid_map({0: 1, 42: 2})
        assert json.loads(json.dumps(wire)) == wire

    def test_the_payload_carries_it_under_grid_pid_map(self):
        payload = scene_graph_payload(
            stamp=1.0, resolution=0.15, origin_xy=(-5.0, -5.0), rooms=[],
            doors=[], drone_xy=None, drone_room_id=None,
            grid_values={11: 1, 12: 2})
        assert payload["grid_pid_map"] == {"1": 11, "2": 12}
        assert json.loads(json.dumps(payload))["grid_pid_map"] == \
            payload["grid_pid_map"]

    def test_the_key_exists_even_with_no_rooms(self):
        payload = scene_graph_payload(
            stamp=0.0, resolution=0.15, origin_xy=(0.0, 0.0), rooms=[],
            doors=[], drone_xy=None, drone_room_id=None)
        assert payload["grid_pid_map"] == {}


class TestRoomValueGrid:
    def test_cells_carry_the_room_value_and_zero_elsewhere(self):
        shape = (6, 8)
        masks = {5: _mask(shape, (0, 3, 0, 4)), 9: _mask(shape, (3, 6, 4, 8))}
        values = assign_room_grid_values(masks.keys())
        grid = room_value_grid(shape, masks, values)
        assert grid.shape == shape and grid.dtype == np.int8
        assert grid[0, 0] == values[5]
        assert grid[5, 7] == values[9]
        assert grid[0, 7] == NO_ROOM_VALUE
        assert set(np.unique(grid)) == {NO_ROOM_VALUE, values[5], values[9]}

    def test_a_big_pid_does_not_overflow_the_int8_field(self):
        # The whole reason for the indirection: pid 200 written straight into
        # an int8 cell would read back as -56.
        shape = (2, 2)
        masks = {200: np.ones(shape, dtype=bool)}
        grid = room_value_grid(shape, masks,
                               assign_room_grid_values(masks.keys()))
        assert grid.min() >= MIN_ROOM_VALUE and grid.max() <= MAX_ROOM_VALUE

    def test_every_value_on_the_grid_is_resolvable_through_the_map(self):
        shape = (10, 10)
        masks = {3: _mask(shape, (0, 5, 0, 5)),
                 8: _mask(shape, (5, 10, 5, 10))}
        values = assign_room_grid_values(masks.keys())
        grid = room_value_grid(shape, masks, values)
        pid_map = grid_pid_map(values)
        for value in np.unique(grid):
            if value == NO_ROOM_VALUE:
                continue
            # No fallback to the raw value anywhere — that is the wrong colour.
            assert str(int(value)) in pid_map
        assert {pid_map[str(int(v))] for v in np.unique(grid)
                if v != NO_ROOM_VALUE} == {3, 8}

    def test_a_room_absent_from_the_map_is_absent_from_the_grid(self):
        shape = (4, 4)
        masks = {1: _mask(shape, (0, 2, 0, 4)), 2: _mask(shape, (2, 4, 0, 4))}
        # Room 2 got no value (as a surplus room would): it must not inherit
        # room 1's value, it must simply not be painted.
        values = {1: assign_room_grid_values([1])[1]}
        grid = room_value_grid(shape, masks, values)
        assert set(np.unique(grid)) == {NO_ROOM_VALUE, values[1]}
        assert np.all(grid[2:, :] == NO_ROOM_VALUE)

    def test_a_value_without_a_mask_raises(self):
        with pytest.raises(KeyError):
            room_value_grid((3, 3), {}, {4: 1})
