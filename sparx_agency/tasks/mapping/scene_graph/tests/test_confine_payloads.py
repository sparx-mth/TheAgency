"""Tests for the room-confinement geometry.

One property matters more than all the others and has its own test: **the
aircraft must always be inside the fence**. A keep-in box the drone is outside
of is not a fence, it is a trap -- the planner finds no legal position
anywhere, retires every frontier, and the aircraft freezes until the lease
lapses. That was measured in flight before this module existed, and
:func:`test_the_fence_always_contains_the_aircraft` is that flight.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.tasks.mapping.scene_graph.ros2.confine_payloads import (
    Z_MAX, Z_MIN, box3, confine_payload, door_seals, release_payload,
    room_bbox)
from sparx_agency.tasks.mapping.scene_graph.tests.test_payloads import (
    assert_plain)

RES = 0.5
ORIGIN = (-10.0, -20.0)


def mask_at(rows, cols, shape=(20, 30)):
    """A rectangular room mask over ``rows`` x ``cols`` cell ranges."""
    m = np.zeros(shape, dtype=bool)
    m[rows[0]:rows[1], cols[0]:cols[1]] = True
    return m


def inside(box, x, y):
    """Whether ``(x, y)`` is inside a flat 6-value box."""
    return box[0] <= x <= box[3] and box[1] <= y <= box[4]


# -- the box --------------------------------------------------------------
def test_room_bbox_covers_every_cell_of_the_mask():
    m = mask_at((4, 9), (6, 13))
    xmin, ymin, xmax, ymax = room_bbox(m, RES, ORIGIN, margin_m=0.0)
    ys, xs = np.nonzero(m)
    for gx, gy in zip(xs, ys):
        cx = ORIGIN[0] + (gx + 0.5) * RES
        cy = ORIGIN[1] + (gy + 0.5) * RES
        assert xmin <= cx <= xmax and ymin <= cy <= ymax


def test_the_box_is_grown_never_shrunk():
    """A box tight to the mask fences the aircraft out of its own room."""
    m = mask_at((4, 9), (6, 13))
    tight = room_bbox(m, RES, ORIGIN, margin_m=0.0)
    grown = room_bbox(m, RES, ORIGIN, margin_m=0.5)
    assert grown[0] == pytest.approx(tight[0] - 0.5)
    assert grown[1] == pytest.approx(tight[1] - 0.5)
    assert grown[2] == pytest.approx(tight[2] + 0.5)
    assert grown[3] == pytest.approx(tight[3] + 0.5)


def test_an_empty_mask_produces_no_box():
    """A renumbered room: send no fence rather than a fence around nothing."""
    assert room_bbox(np.zeros((10, 10), dtype=bool), RES, ORIGIN) is None


def test_box3_spans_every_altitude():
    box = box3(1.0, 2.0, 3.0, 4.0)
    assert box == [1.0, 2.0, Z_MIN, 3.0, 4.0, Z_MAX]
    assert len(box) == 6


# -- the door seals -------------------------------------------------------
DOORS = [
    {"index": 0, "xy": [1.0, 2.0], "rooms": [7, 8]},
    {"index": 1, "xy": [5.0, 6.0], "rooms": [8, 9]},
    {"index": 2, "xy": [9.0, 9.0], "rooms": [7]},
]


def test_only_this_rooms_doors_are_sealed():
    """Sealing a far door fences the route the aircraft needs on the way out."""
    seals = door_seals(DOORS, room_id=7, half_m=1.0)
    assert len(seals) == 2
    assert any(inside(s, 1.0, 2.0) for s in seals)
    assert any(inside(s, 9.0, 9.0) for s in seals)
    assert not any(inside(s, 5.0, 6.0) for s in seals)


def test_a_seal_covers_the_whole_doorway_width():
    seal = door_seals(DOORS, room_id=7, half_m=0.9)[0]
    # The hospital's doorways are 0.93 m; the seal must cover the jambs.
    assert (seal[3] - seal[0]) >= 0.93
    assert (seal[4] - seal[1]) >= 0.93


def test_malformed_doors_are_skipped_not_raised_on():
    doors = [{"xy": "nope", "rooms": [7]}, {"rooms": [7]},
             {"xy": [1.0, 2.0], "rooms": "seven"},
             {"xy": [3.0, 4.0], "rooms": [7]}]
    seals = door_seals(doors, room_id=7)
    assert len(seals) == 1
    assert inside(seals[0], 3.0, 4.0)


def test_a_room_with_no_doors_gets_no_seals():
    assert door_seals(DOORS, room_id=99) == []


# -- the whole request ----------------------------------------------------
def test_confine_payload_shape_and_plainness():
    m = mask_at((4, 9), (6, 13))
    payload = confine_payload(7, m, RES, ORIGIN, DOORS, (0.0, -12.0), 8.0)
    assert payload is not None
    assert_plain(payload)
    assert payload["room_id"] == 7
    assert payload["lease_s"] == pytest.approx(8.0)
    assert all(len(b) == 6 for b in payload["keep_in"])
    assert all(len(b) == 6 for b in payload["keep_out"])
    assert len(payload["keep_out"]) == 2


def test_the_fence_always_contains_the_aircraft():
    """The measured failure: a fence the drone is outside of freezes it.

    The aircraft may legitimately be outside the room's own box -- standing in
    the doorway on arrival, or pushed out by the follower's overshoot -- and
    the planner cannot tell "fenced out" from "no legal position exists".
    """
    m = mask_at((4, 9), (6, 13))
    far = (9.0, 5.0)                       # nowhere near the room
    payload = confine_payload(7, m, RES, ORIGIN, DOORS, far, 8.0)
    assert any(inside(b, far[0], far[1]) for b in payload["keep_in"]), (
        "the aircraft is outside every keep-in box; this is a trap, not a fence")


def test_the_room_itself_is_still_inside_the_fence():
    m = mask_at((4, 9), (6, 13))
    payload = confine_payload(7, m, RES, ORIGIN, DOORS, (9.0, 5.0), 8.0)
    ys, xs = np.nonzero(m)
    cx = ORIGIN[0] + (float(xs.mean()) + 0.5) * RES
    cy = ORIGIN[1] + (float(ys.mean()) + 0.5) * RES
    assert any(inside(b, cx, cy) for b in payload["keep_in"])


def test_no_pose_still_produces_a_usable_fence():
    m = mask_at((4, 9), (6, 13))
    payload = confine_payload(7, m, RES, ORIGIN, DOORS, None, 8.0)
    assert len(payload["keep_in"]) == 1


def test_an_empty_mask_produces_no_request():
    payload = confine_payload(7, np.zeros((10, 10), dtype=bool), RES, ORIGIN,
                              DOORS, (0.0, 0.0), 8.0)
    assert payload is None


def test_release_is_recognisable_as_a_release():
    payload = release_payload(7)
    assert_plain(payload)
    assert payload["lease_s"] == 0.0
    assert payload["keep_in"] == []
    assert payload["keep_out"] == []
