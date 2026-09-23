"""Tests for the scene-graph latched-topic payload assembly.

``ros2/payloads.py`` exists for one reason: a numpy scalar that reaches
``json.dumps`` raises ``TypeError`` at publish time, in a timer callback, in
flight — and ``np.int64``/``np.float32`` are exactly what the segmentation and
landmark code hands the nodes. So every test here feeds numpy in and asserts
plain Python comes out, and that the dict then survives a real
``json.dumps``/``json.loads`` round trip unchanged.

The second job is the topic contract itself: the key names and nesting of
``/scene_graph`` and ``/perception/objects``, which the room classifier, the
LLM oracle, the target watcher and the dashboard all parse by name.

The third is the ``room_labels_grid`` indirection — an ``int8`` grid cannot
hold an unbounded pid, so the grid carries a small per-room value and the
payload carries the ``grid_pid_map`` that resolves it. The grid and the map are
built from one mapping per tick, and the test that matters is that they never
disagree: every non-zero cell published must be resolvable to a pid.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.core.mapping.objects.landmarks import ObjectLandmark
from sparx_agency.tasks.mapping.scene_graph.ros2.payloads import (
    MAX_ROOM_VALUE,
    MIN_ROOM_VALUE,
    NO_ROOM_VALUE,
    assign_room_grid_values,
    door_entry,
    grid_pid_map,
    object_entry,
    objects_payload,
    room_entry,
    room_value_grid,
    scene_graph_payload,
)

SCENE_GRAPH_KEYS = {"stamp", "resolution", "origin", "rooms", "doors", "drone",
                    "grid_pid_map"}

PLAIN_TYPES = (bool, int, float, str, type(None))


def assert_plain(value, path="payload"):
    """Assert every leaf is a plain builtin, not a numpy scalar.

    ``isinstance`` would not do: ``np.float64`` subclasses ``float`` while
    ``np.float32`` does not, so only an exact type check proves the coercion
    actually happened everywhere.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str, "%s: key %r is %s" % (path, key,
                                                           type(key).__name__)
            assert_plain(item, "%s[%r]" % (path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            assert_plain(item, "%s[%d]" % (path, i))
    else:
        assert type(value) in PLAIN_TYPES, (
            "%s is %s, not a plain builtin" % (path, type(value).__name__))


def numpy_landmark(landmark_id=3, class_name="hospital bed"):
    """A landmark carrying exactly the numpy scalars the mapper produces."""
    return ObjectLandmark(id=np.int64(landmark_id), class_name=class_name,
                          xy=(np.float32(1.5), np.float32(-2.25)),
                          count=np.int32(4))


def numpy_room(room_id=2):
    return room_entry(
        room_id=np.int64(room_id),
        centroid_xy=np.array([1.25, -0.5], dtype=np.float32),
        n_cells=np.int64(420),
        time_in_room_s=np.float32(12.5),
        frontier_clusters=np.int32(3),
        color_rgb=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        objects=[object_entry(numpy_landmark())],
        door_indices=np.array([0, 1], dtype=np.int64))


def numpy_door(index=0):
    return door_entry(index=np.int64(index),
                      xy=(np.float32(3.0), np.float32(4.0)),
                      discovered=np.bool_(True),
                      room_ids=np.array([1, 2], dtype=np.int64),
                      room_pairs=np.array([[1, 2]], dtype=np.int64))


class TestNumpyIsTheReasonThisModuleExists:
    """The failure mode being prevented, stated as a test."""

    @pytest.mark.parametrize("value", [np.int64(3), np.int32(3),
                                       np.float32(1.5), np.bool_(True)])
    def test_raw_numpy_scalars_break_json_dumps(self, value):
        with pytest.raises(TypeError):
            json.dumps(value)

    def test_the_same_scalars_survive_once_routed_through_the_payloads(self):
        payload = objects_payload(np.float64(12.5), [numpy_landmark()])
        json.dumps(payload)          # must not raise
        assert_plain(payload)


class TestObjectsPayload:
    """``/perception/objects``: ``{"stamp", "objects": [{id, class, xy, count}]}``."""

    def test_contract_keys_and_nesting(self):
        payload = objects_payload(1.0, [numpy_landmark()])
        assert set(payload.keys()) == {"stamp", "objects"}
        assert isinstance(payload["objects"], list)
        assert set(payload["objects"][0].keys()) == {"id", "class", "xy",
                                                     "count"}
        assert len(payload["objects"][0]["xy"]) == 2

    def test_values_are_plain_and_correct(self):
        entry = object_entry(numpy_landmark(landmark_id=7))
        assert entry == {"id": 7, "class": "hospital bed",
                         "xy": [1.5, -2.25], "count": 4}
        assert_plain(entry)

    def test_round_trips_through_json_exactly(self):
        payload = objects_payload(np.float64(9.75),
                                  [numpy_landmark(0), numpy_landmark(1)])
        assert json.loads(json.dumps(payload)) == payload

    def test_an_empty_landmark_set_is_a_valid_payload(self):
        payload = objects_payload(0.0, [])
        assert payload == {"stamp": 0.0, "objects": []}
        assert json.loads(json.dumps(payload)) == payload

    def test_a_generator_of_landmarks_is_accepted(self):
        payload = objects_payload(1.0, (numpy_landmark(i) for i in range(3)))
        assert [o["id"] for o in payload["objects"]] == [0, 1, 2]


class TestSceneGraphPayload:
    """``/scene_graph``: the room/door/drone structure every consumer parses."""

    def test_top_level_contract_keys(self):
        payload = scene_graph_payload(
            stamp=np.float64(10.0), resolution=np.float32(0.15),
            origin_xy=np.array([-5.0, -7.5], dtype=np.float32),
            rooms=[numpy_room()], doors=[numpy_door()],
            drone_xy=np.array([0.5, 0.25], dtype=np.float32),
            drone_room_id=np.int64(2))
        assert set(payload.keys()) == SCENE_GRAPH_KEYS
        assert len(payload["origin"]) == 2
        assert set(payload["drone"].keys()) == {"xy", "room_id"}
        assert payload["drone"]["xy"] == [0.5, 0.25]
        assert payload["drone"]["room_id"] == 2

    def test_room_contract_keys_and_nesting(self):
        room = numpy_room(room_id=4)
        assert set(room.keys()) == {"id", "centroid", "cells",
                                    "time_in_room_s", "frontier_clusters",
                                    "color", "objects", "doors"}
        assert room["id"] == 4
        assert len(room["centroid"]) == 2
        assert len(room["color"]) == 3
        assert room["doors"] == [0, 1]
        # Objects nest as whole object entries, not ids.
        assert set(room["objects"][0].keys()) == {"id", "class", "xy", "count"}

    def test_door_contract_keys_and_types(self):
        door = numpy_door(index=5)
        assert set(door.keys()) == {"index", "xy", "discovered", "rooms",
                                    "room_pairs"}
        assert door["index"] == 5
        assert door["discovered"] is True
        assert door["rooms"] == [1, 2]
        # The edges, as plain lists of two plain ints.
        assert door["room_pairs"] == [[1, 2]]

    def test_a_door_that_connects_nothing_carries_no_pair(self):
        """Proximity to a door is not a connection; the default is empty."""
        door = door_entry(index=0, xy=(0.0, 0.0), discovered=True,
                          room_ids=[])
        assert door["rooms"] == [] and door["room_pairs"] == []

    def test_every_value_is_plain_after_a_numpy_only_build(self):
        payload = scene_graph_payload(
            stamp=np.float64(10.0), resolution=np.float32(0.15),
            origin_xy=np.array([-5.0, -7.5], dtype=np.float32),
            rooms=[numpy_room(0), numpy_room(1)],
            doors=[numpy_door(0), numpy_door(1)],
            drone_xy=np.array([0.5, 0.25], dtype=np.float32),
            drone_room_id=np.int64(2),
            grid_values={np.int64(0): np.int64(1), np.int64(1): np.int64(2)})
        json.dumps(payload)          # must not raise
        assert_plain(payload)

    def test_round_trips_through_json_exactly(self):
        payload = scene_graph_payload(
            stamp=np.float64(10.0), resolution=np.float32(0.15),
            origin_xy=np.array([-5.0, -7.5], dtype=np.float32),
            rooms=[numpy_room(0)], doors=[numpy_door(0)],
            drone_xy=(0.5, 0.25), drone_room_id=np.int64(2),
            grid_values={0: 1, 4: 2})
        assert json.loads(json.dumps(payload)) == payload

    def test_grid_pid_map_is_emitted_inverted_with_string_keys(self):
        payload = scene_graph_payload(1.0, 0.15, (0.0, 0.0), [], [], None,
                                      None, grid_values={7: 1, 9: 2})
        assert payload["grid_pid_map"] == {"1": 7, "2": 9}

    def test_grid_pid_map_defaults_to_empty_when_no_grid_is_published(self):
        payload = scene_graph_payload(1.0, 0.15, (0.0, 0.0), [], [], None, None)
        assert payload["grid_pid_map"] == {}
        assert set(payload.keys()) == SCENE_GRAPH_KEYS

    def test_no_odometry_yet_gives_explicit_nulls(self):
        """Before the first odom: ``drone`` exists, both fields are null."""
        payload = scene_graph_payload(1.0, 0.15, (0.0, 0.0), [], [],
                                      drone_xy=None, drone_room_id=None)
        assert payload["drone"] == {"xy": None, "room_id": None}
        assert json.loads(json.dumps(payload)) == payload

    def test_a_located_drone_outside_any_room_keeps_its_position(self):
        payload = scene_graph_payload(1.0, 0.15, (0.0, 0.0), [], [],
                                      drone_xy=(1.0, 2.0), drone_room_id=None)
        assert payload["drone"] == {"xy": [1.0, 2.0], "room_id": None}

    def test_rooms_and_doors_are_copied_not_aliased(self):
        """The node mutates its per-tick lists; the published dict must not follow."""
        rooms, doors = [numpy_room(0)], [numpy_door(0)]
        payload = scene_graph_payload(1.0, 0.15, (0.0, 0.0), rooms, doors,
                                      None, None)
        rooms.append(numpy_room(1))
        doors.append(numpy_door(1))
        assert len(payload["rooms"]) == 1
        assert len(payload["doors"]) == 1

    def test_an_empty_scene_graph_is_a_valid_payload(self):
        payload = scene_graph_payload(np.float64(0.0), np.float32(0.15),
                                      (0.0, 0.0), [], [], None, None)
        assert payload["rooms"] == [] and payload["doors"] == []
        assert json.loads(json.dumps(payload)) == payload


class TestRoomGridValues:
    """``pid -> int8 grid value``: stable while a room lives, recycled after."""

    def test_first_tick_assigns_the_lowest_values_in_order(self):
        assert assign_room_grid_values([4, 9, 2]) == {4: 1, 9: 2, 2: 3}

    def test_a_surviving_room_keeps_its_value(self):
        first = assign_room_grid_values([4, 9, 2])
        second = assign_room_grid_values([9, 2, 4], first)
        assert second == first, "a persisting room must not be re-coloured"

    def test_a_new_room_takes_a_free_value_without_disturbing_the_others(self):
        first = assign_room_grid_values([4, 9])
        second = assign_room_grid_values([4, 9, 11], first)
        assert second[4] == first[4] and second[9] == first[9]
        assert second[11] == 3

    def test_a_dead_rooms_value_is_recycled(self):
        first = assign_room_grid_values([4, 9])          # 4->1, 9->2
        second = assign_room_grid_values([9, 12], first)  # 4 is gone
        assert second[9] == 2
        assert second[12] == 1

    def test_duplicate_pids_are_ignored(self):
        assert assign_room_grid_values([4, 4, 9]) == {4: 1, 9: 2}

    def test_values_stay_inside_the_int8_safe_band(self):
        values = assign_room_grid_values(range(200))
        assert min(values.values()) >= MIN_ROOM_VALUE
        assert max(values.values()) <= MAX_ROOM_VALUE
        assert len(set(values.values())) == len(values)

    def test_more_rooms_than_values_drops_the_surplus_instead_of_colliding(self):
        values = assign_room_grid_values(range(200))
        assert len(values) == MAX_ROOM_VALUE - MIN_ROOM_VALUE + 1

    @pytest.mark.parametrize("stale", [0, -1, 101, 900])
    def test_an_out_of_band_carried_value_is_reassigned(self, stale):
        values = assign_room_grid_values([7], {7: stale})
        assert MIN_ROOM_VALUE <= values[7] <= MAX_ROOM_VALUE

    def test_two_rooms_can_never_share_a_value(self):
        values = assign_room_grid_values([4, 9], {4: 5, 9: 5})
        assert len(set(values.values())) == 2

    def test_grid_pid_map_keys_are_json_stable_strings(self):
        mapping = grid_pid_map({7: 1, 9: 2})
        assert mapping == {"1": 7, "2": 9}
        assert json.loads(json.dumps(mapping)) == mapping
        assert_plain(mapping)

    def test_grid_pid_map_coerces_numpy_pids_and_values(self):
        mapping = grid_pid_map({np.int64(7): np.int8(3)})
        assert mapping == {"3": 7}
        assert_plain(mapping)


class TestRoomValueGrid:
    """The ``/scene_graph/room_labels_grid`` image itself."""

    @staticmethod
    def masks():
        a = np.zeros((6, 8), dtype=bool)
        a[0:3, 0:4] = True
        b = np.zeros((6, 8), dtype=bool)
        b[3:6, 4:8] = True
        return {4: a, 9: b}

    def test_grid_is_int8_of_the_requested_shape(self):
        grid = room_value_grid((6, 8), self.masks(), {4: 1, 9: 2})
        assert grid.shape == (6, 8)
        assert grid.dtype == np.int8

    def test_cells_carry_their_rooms_value_and_nothing_else(self):
        grid = room_value_grid((6, 8), self.masks(), {4: 1, 9: 2})
        assert grid[0, 0] == 1
        assert grid[5, 7] == 2
        assert grid[0, 7] == NO_ROOM_VALUE
        assert set(np.unique(grid).tolist()) == {NO_ROOM_VALUE, 1, 2}

    def test_no_rooms_gives_an_empty_grid(self):
        grid = room_value_grid((6, 8), {}, {})
        assert np.all(grid == NO_ROOM_VALUE)

    def test_the_top_of_the_band_survives_int8(self):
        """Value 100 must come back as 100, not a wrapped negative."""
        grid = room_value_grid((6, 8), self.masks(),
                               {4: MAX_ROOM_VALUE, 9: MIN_ROOM_VALUE})
        assert grid[0, 0] == MAX_ROOM_VALUE

    def test_a_value_without_a_mask_raises(self):
        """Otherwise the published grid and grid_pid_map silently disagree."""
        with pytest.raises(KeyError):
            room_value_grid((6, 8), self.masks(), {4: 1, 9: 2, 13: 3})

    def test_grid_and_pid_map_agree_on_every_published_cell(self):
        """The invariant the pair exists to keep: no cell resolves to nothing."""
        values = assign_room_grid_values([4, 9])
        grid = room_value_grid((6, 8), self.masks(), values)
        mapping = grid_pid_map(values)
        for value in np.unique(grid):
            if int(value) == NO_ROOM_VALUE:
                continue
            assert str(int(value)) in mapping
        assert set(mapping.values()) == {4, 9}
