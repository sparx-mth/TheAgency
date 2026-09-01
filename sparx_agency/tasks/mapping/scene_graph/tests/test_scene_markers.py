"""Tests for the RViz MarkerArray view of the scene graph.

``scene_markers`` is pure geometry over plain data, so the whole picture is
checkable message by message — no node, no rclpy, no display. Needs
``visualization_msgs``, i.e. a sourced ROS 2:

    source /opt/ros/jazzy/setup.bash
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \\
        sparx_agency/tasks/mapping/scene_graph/tests/test_scene_markers.py

What is worth pinning down, because each of these is invisible in review and
obvious on screen:

* the array **clears itself** — a leading ``DELETEALL`` and globally unique
  ids, or a retired pid's markers survive a BEV reshape and read as a room
  that is still there;
* the **namespace set** is exactly the flown one, since RViz toggles by
  namespace and a renamed one silently disappears from the display;
* ``room_fill`` draws every cell of a small room and **caps** a large one,
  which is the difference between a 20 k-point message per room and a
  drawable one;
* the ``room_room_edges`` dog-leg is ``centroid -> door -> door -> centroid``
  at one height — a straight centroid-to-centroid line would claim a
  connection that does not exist through the wall between them;
* ``room_polygons`` **closes** and lands on the cells it encloses — an
  outline half a cell off, or left open where the room runs off the edge of
  the BEV, reads as a room with a missing wall;
* ``room_object_edges`` follows the node's own ``objects_by_room`` grouping,
  so the drawn parent-room link and the ``/scene_graph`` JSON can never
  disagree — and an object in no room draws no edge rather than a line to
  the nearest centroid;
* the **z-stack** 3.0 / 3.7 / 4.5 of the three text lines over a centroid,
  which is the only thing keeping them from overprinting each other.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

visualization_msgs = pytest.importorskip("visualization_msgs")

from visualization_msgs.msg import Marker  # noqa: E402

from sparx_agency.core.mapping.objects import landmarks  # noqa: E402
from sparx_agency.core.mapping.topology import room_color  # noqa: E402
from sparx_agency.tasks.mapping.scene_graph.ros2 import (  # noqa: E402
    scene_markers as sm)

RESOLUTION = 0.25
ORIGIN = (-2.0, -3.0)
GRID_SHAPE = (12, 16)  # (H, W): row 0 is minimum y, as on the BEV

OBJECTS = [{"class": "chair", "xy": [0.4, -1.2], "count": 3},
           {"class": "chair", "xy": [0.9, -1.7], "count": 1},
           {"class": "bed", "xy": [2.2, 1.1], "count": 2}]


def _state(**overrides) -> sm.SceneMarkerState:
    """Two rooms, a spine through both, three doors (two of them found)."""
    mask_a = np.zeros(GRID_SHAPE, bool)
    mask_a[1:5, 1:6] = True                     # 20 cells
    mask_b = np.zeros(GRID_SHAPE, bool)
    mask_b[7:11, 9:15] = True                   # 24 cells
    skeleton = np.zeros(GRID_SHAPE, bool)
    skeleton[3, 1:6] = True                     # 5 cells in room 0
    skeleton[9, 9:15] = True                    # 6 cells in room 1
    state = sm.SceneMarkerState(
        resolution=RESOLUTION,
        origin_xy=ORIGIN,
        room_masks={0: mask_a, 3: mask_b},
        room_centroids={0: (0.5, -1.5), 3: (2.5, 1.5)},
        skeleton=skeleton,
        dwell_s={0: 42.4},
        frontier_counts={0: 3},
        doors=[
            # index 0 links both rooms; 1 is a found dead end; 2 is unopened.
            {"index": 0, "xy": [1.5, 0.0], "discovered": True,
             "rooms": [0, 3], "room_pairs": [[0, 3]]},
            {"index": 1, "xy": [-1.0, -2.0], "discovered": True,
             "rooms": [0], "room_pairs": []},
            {"index": 2, "xy": [3.0, -2.5], "discovered": False,
             "rooms": [], "room_pairs": []},
        ],
        door_cut_m=0.60,
        room_types={0: {"label": "office", "confidence": 0.82}},
        room_probs={0: {"prob": 0.7}, 3: {"prob": 0.3}},
        objects=OBJECTS,
        # The node's own tick assignment, as shipped in rooms[].objects.
        objects_by_room={0: [OBJECTS[0], OBJECTS[1]], 3: [OBJECTS[2]]},
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _array(state=None, **kwargs):
    """The full MarkerArray for a state, with a seeded subsampler."""
    return sm.scene_marker_array(state if state is not None else _state(),
                                 kwargs.pop("frame_id", "world"),
                                 rng=np.random.default_rng(0), **kwargs)


def _by_ns(array, ns):
    return [m for m in array.markers if m.ns == ns]


def _one(array, ns):
    markers = _by_ns(array, ns)
    assert len(markers) == 1, "expected exactly one %r marker, got %d" % (
        ns, len(markers))
    return markers[0]


# ── the array clears itself ────────────────────────────────────────────
def test_deleteall_leads_the_array():
    array = _array()
    assert array.markers[0].action == Marker.DELETEALL
    assert all(m.action == Marker.ADD for m in array.markers[1:])


def test_marker_ids_are_globally_unique():
    ids = [m.id for m in _array().markers]
    assert len(set(ids)) == len(ids)
    # The counter starts past the DELETEALL's id, and never restarts per
    # namespace: two namespaces sharing an id is legal in RViz but makes the
    # array impossible to reason about when one is deleted.
    assert min(m.id for m in _array().markers[1:]) == 1


def test_every_marker_carries_the_frame_and_orientation():
    for marker in _array(frame_id="map").markers[1:]:
        assert marker.header.frame_id == "map"
        assert marker.pose.orientation.w == 1.0
        assert marker.lifetime.sec == 0 and marker.lifetime.nanosec == 0


def test_empty_frame_id_is_refused():
    with pytest.raises(ValueError):
        sm.MarkerFactory("")


# ── the namespace set ──────────────────────────────────────────────────
def test_namespace_set_is_exactly_the_intended_one():
    counts = Counter(m.ns for m in _array().markers)
    assert counts == Counter({
        "": 1,                        # the DELETEALL
        "room_fill": 2,
        "room_polygons": 2,           # one contour each, both rooms solid
        "skeleton": 2,
        "rooms": 2,
        "room_labels": 2,
        "room_type_labels": 1,        # only room 0 is classified
        "room_probabilities": 2,
        "doors_discovered": 2,
        "doors_pending": 1,
        "door_labels": 2,
        "door_labels_pending": 1,
        "cut_disks": 2,               # discovered doors only
        "room_room_edges": 1,         # only door 0 links a pair
        "room_object_edges": 3,       # every object has a parent room here
        "objects:chair": 1,           # one CUBE_LIST per class
        "objects:bed": 1,
        "object_labels": 3,           # one per object
    })


def test_cut_disks_can_be_switched_off():
    array = _array(_state(draw_cut_disks=False))
    assert _by_ns(array, "cut_disks") == []
    assert len(_by_ns(array, "doors_discovered")) == 2


def test_missing_skeleton_drops_only_the_spine():
    array = _array(_state(skeleton=None))
    assert _by_ns(array, "skeleton") == []
    assert len(_by_ns(array, "room_fill")) == 2


def test_skeleton_from_another_tick_raises():
    stale = np.zeros((4, 4), bool)
    with pytest.raises(ValueError):
        _array(_state(skeleton=stale))


# ── room fill and the point cap ────────────────────────────────────────
def test_room_fill_draws_every_cell_below_the_cap():
    state = _state()
    array = _array(state)
    counts = sorted(len(m.points) for m in _by_ns(array, "room_fill"))
    assert counts == sorted(int(m.sum()) for m in state.room_masks.values())


def test_room_fill_cell_centres_land_in_world():
    """Row 0 is minimum y and there is no flip: cell (cx, cy) -> world."""
    mask = np.zeros(GRID_SHAPE, bool)
    mask[2, 3] = True
    array = _array(_state(room_masks={0: mask}, skeleton=None,
                          room_centroids={}, room_types={}, room_probs={},
                          objects=[], doors=[]))
    point = _one(array, "room_fill").points[0]
    assert point.x == pytest.approx(ORIGIN[0] + 3.5 * RESOLUTION)
    assert point.y == pytest.approx(ORIGIN[1] + 2.5 * RESOLUTION)
    assert point.z == pytest.approx(sm.Z_FILL)


def test_room_fill_is_capped_above_the_limit():
    big = np.ones((200, 200), bool)          # 40 000 cells
    array = _array(_state(room_masks={0: big}, skeleton=None))
    points = _one(array, "room_fill").points
    assert len(points) == sm.FILL_CAP
    # Subsampled WITHOUT replacement: no cell is drawn twice.
    assert len({(p.x, p.y) for p in points}) == sm.FILL_CAP


def test_skeleton_is_capped_and_room_coloured():
    big = np.ones((100, 100), bool)
    array = _array(_state(room_masks={0: big}, skeleton=big.copy()))
    marker = _one(array, "skeleton")
    assert len(marker.points) == sm.SKELETON_CAP
    r, g, b = room_color(0)
    assert (marker.color.r, marker.color.g, marker.color.b) == \
        pytest.approx((r, g, b))
    assert marker.color.a == 1.0     # opaque, unlike the fill under it


def test_room_fill_and_skeleton_share_the_room_hue():
    array = _array()
    fills = {m.color.r for m in _by_ns(array, "room_fill")}
    spines = {m.color.r for m in _by_ns(array, "skeleton")}
    assert fills == spines
    assert all(m.color.a == pytest.approx(sm.FILL_ALPHA)
               for m in _by_ns(array, "room_fill"))


# ── the dog-leg edge ───────────────────────────────────────────────────
def test_room_room_edge_is_a_dog_leg_through_the_door():
    edge = _one(_array(), "room_room_edges")
    assert edge.type == Marker.LINE_LIST
    assert edge.scale.x == pytest.approx(sm.EDGE_WIDTH_M)
    assert (edge.color.r, edge.color.g, edge.color.b) == \
        pytest.approx(sm.GOLD)
    assert len(edge.points) == 4
    assert [(p.x, p.y) for p in edge.points] == [
        (0.5, -1.5), (1.5, 0.0), (1.5, 0.0), (2.5, 1.5)]
    assert all(p.z == pytest.approx(sm.Z_EDGE) for p in edge.points)


def test_undiscovered_doors_draw_no_edge():
    doors = [{"index": 0, "xy": [1.5, 0.0], "discovered": False,
              "rooms": [0, 3], "room_pairs": [[0, 3]]}]
    assert _by_ns(_array(_state(doors=doors)), "room_room_edges") == []


def test_only_the_listed_pairs_are_drawn_never_pairs_of_the_room_list():
    """The rooms at a door are not all connected to each other.

    A door where rooms 0-3 touch and 3-7 touch, but 0 and 7 have a wall
    between them, must draw two edges and not the three that pairing up
    its room list would give. That third edge is the defect: it crosses
    a wall.
    """
    state = _state()
    state.room_centroids[7] = (4.0, -2.0)
    state.doors = [{"index": 0, "xy": [1.5, 0.0], "discovered": True,
                    "rooms": [0, 3, 7], "room_pairs": [[0, 3], [3, 7]]}]
    edges = _by_ns(_array(state), "room_room_edges")
    assert len(edges) == 2
    far_ends = {(round(e.points[0].x, 2), round(e.points[0].y, 2)) for e in
                edges}
    assert far_ends == {(0.5, -1.5), (2.5, 1.5)}


def test_a_door_with_rooms_but_no_pairs_draws_no_edge():
    """Vetted away by adjacency upstream: nothing to draw here."""
    doors = [{"index": 0, "xy": [1.5, 0.0], "discovered": True,
              "rooms": [0, 3], "room_pairs": []}]
    assert _by_ns(_array(_state(doors=doors)), "room_room_edges") == []


def test_a_pair_naming_an_unknown_or_repeated_room_draws_no_edge():
    """Pids restart on a BEV reshape; a stale pair must not paint."""
    for pairs in ([[0, 99]], [[3, 3]], [[0]]):
        doors = [{"index": 0, "xy": [1.5, 0.0], "discovered": True,
                  "rooms": [0, 3], "room_pairs": pairs}]
        assert _by_ns(_array(_state(doors=doors)), "room_room_edges") == []


# ── doors ──────────────────────────────────────────────────────────────
def test_discovered_and_pending_doors_differ_only_in_ns_colour_alpha():
    array = _array()
    found = _by_ns(array, "doors_discovered")[0]
    pending = _one(array, "doors_pending")
    for marker in (found, pending):
        assert marker.type == Marker.CYLINDER
        assert (marker.scale.x, marker.scale.y, marker.scale.z) == \
            pytest.approx((0.5, 0.5, sm.DOOR_HEIGHT_M))
        # Standing ON the floor, not sunk half into it.
        assert marker.pose.position.z == pytest.approx(sm.DOOR_HEIGHT_M / 2)
    assert (found.color.r, found.color.g, found.color.b) == \
        pytest.approx(sm.DOOR_OPEN_COLOR)
    assert found.color.a == pytest.approx(0.95)
    assert (pending.color.r, pending.color.g, pending.color.b) == \
        pytest.approx(sm.DOOR_PENDING_COLOR)
    assert pending.color.a == pytest.approx(0.35)


def test_door_labels_name_the_door_index():
    array = _array()
    assert sorted(m.text for m in _by_ns(array, "door_labels")) == ["d0", "d1"]
    pending = _one(array, "door_labels_pending")
    assert pending.text == "d2"
    assert pending.color.a == pytest.approx(0.55)
    for marker in _by_ns(array, "door_labels") + [pending]:
        assert marker.type == Marker.TEXT_VIEW_FACING
        assert marker.pose.position.z == pytest.approx(1.75)
        assert marker.scale.z == pytest.approx(sm.DOOR_TEXT_HEIGHT_M)


def test_cut_disk_diameter_is_twice_the_cut_radius():
    disk = _by_ns(_array(_state(door_cut_m=0.6)), "cut_disks")[0]
    assert (disk.scale.x, disk.scale.y) == pytest.approx((1.2, 1.2))
    assert disk.scale.z == pytest.approx(0.03)
    assert disk.pose.position.z == pytest.approx(sm.Z_CUT_DISK)


# ── the text stack over a centroid ─────────────────────────────────────
def test_text_z_stack_is_3_0_then_3_7_then_4_5():
    array = _array()
    heights = {ns: {m.pose.position.z for m in _by_ns(array, ns)}
               for ns in ("room_labels", "room_type_labels",
                          "room_probabilities")}
    assert heights == {"room_labels": {sm.Z_ROOM_LABEL},
                       "room_type_labels": {sm.Z_ROOM_TYPE},
                       "room_probabilities": {sm.Z_ROOM_PROB}}
    assert (sm.Z_ROOM_LABEL, sm.Z_ROOM_TYPE, sm.Z_ROOM_PROB) == (3.0, 3.7, 4.5)


def test_room_label_text_is_ascii_dwell_and_frontiers():
    texts = {m.text for m in _by_ns(_array(), "room_labels")}
    assert texts == {"R0  t=42s  F=3", "R3  t=0s  F=0"}
    assert all(t.isascii() for t in texts)


def test_room_type_and_probability_text():
    array = _array()
    assert _one(array, "room_type_labels").text == "office (0.82)"
    assert sorted(m.text for m in _by_ns(array, "room_probabilities")) == \
        ["P=0.30", "P=0.70"]


def test_probability_colour_ramps_blue_to_red_by_value():
    array = _array()
    hot, cold = sorted(_by_ns(array, "room_probabilities"),
                       key=lambda m: -m.color.r)
    assert hot.text == "P=0.70"
    # The top room saturates the ramp; the ramp is normalised by the max.
    assert (hot.color.r, hot.color.g, hot.color.b) == \
        pytest.approx(sm.probability_color(0.7, 0.7))
    assert hot.color.r > cold.color.r          # redder
    assert hot.color.b < cold.color.b          # less blue


def test_text_for_a_room_that_vanished_is_dropped():
    """A label whose pid is gone has no centroid to sit on."""
    array = _array(_state(room_types={9: {"label": "ward",
                                          "confidence": 0.5}},
                          room_probs={9: {"prob": 1.0}}))
    assert _by_ns(array, "room_type_labels") == []
    assert _by_ns(array, "room_probabilities") == []


def test_room_centroid_sphere():
    spheres = _by_ns(_array(), "rooms")
    assert {m.type for m in spheres} == {Marker.SPHERE}
    assert {m.pose.position.z for m in spheres} == {sm.Z_CENTROID}
    assert all(m.scale.x == pytest.approx(0.6) for m in spheres)


# ── objects ────────────────────────────────────────────────────────────
def test_objects_are_one_cube_list_per_class_in_the_class_colour():
    array = _array()
    chairs = _one(array, "objects:chair")
    assert chairs.type == Marker.CUBE_LIST
    assert (chairs.color.r, chairs.color.g, chairs.color.b) == \
        pytest.approx(landmarks.class_color("chair"))
    # Two chairs, each drawn as a 2x2-cell patch.
    assert len(chairs.points) == 2 * sm.OBJECT_PATCH_CELLS ** 2
    assert all(p.z == pytest.approx(sm.Z_OBJECT) for p in chairs.points)


def test_object_labels_carry_the_class_and_count():
    texts = sorted(m.text for m in _by_ns(_array(), "object_labels"))
    assert texts == ["bed x2", "chair x1", "chair x3"]
    assert all(m.pose.position.z == pytest.approx(sm.Z_OBJECT_LABEL)
               for m in _by_ns(_array(), "object_labels"))


# ── room outlines ──────────────────────────────────────────────────────
def _bbox(marker):
    xs = [p.x for p in marker.points]
    ys = [p.y for p in marker.points]
    return (min(xs), min(ys), max(xs), max(ys))


def test_room_polygon_traces_the_mask_boundary_closed_and_room_coloured():
    """A 4x5-cell room outlines the rectangle those cells occupy.

    The 0.5 contour of the mask runs half a cell outside the True cells, so
    in world terms the outline encloses exactly the filled rectangle: cells
    (1..4, 1..5) span x -1.75..-0.5 and y -2.75..-1.75 at this origin and
    resolution.
    """
    array = _array()
    polys = _by_ns(array, "room_polygons")
    assert len(polys) == 2
    room0 = polys[0]                     # rooms are drawn in sorted pid order
    assert room0.type == Marker.LINE_STRIP
    assert room0.scale.x == pytest.approx(sm.ROOM_OUTLINE_WIDTH_M)
    r, g, b = room_color(0)
    assert (room0.color.r, room0.color.g, room0.color.b) == \
        pytest.approx((r, g, b))
    assert room0.color.a == 1.0          # opaque over its own translucent fill
    assert _bbox(room0) == pytest.approx((-1.75, -2.75, -0.5, -1.75))
    assert all(p.z == pytest.approx(sm.Z_OUTLINE) for p in room0.points)
    assert sm.Z_FILL < sm.Z_OUTLINE < sm.Z_OBJECT

    # Closed: the strip ends where it started, so there is no seam.
    first, last = room0.points[0], room0.points[-1]
    assert (last.x, last.y, last.z) == pytest.approx((first.x, first.y,
                                                      first.z))
    # ... and the closing point is a copy, not the same Point object aliased
    # into the list twice.
    assert last is not first


def test_room_polygons_are_one_strip_per_contour_so_holes_are_drawn():
    holed = np.zeros(GRID_SHAPE, bool)
    holed[1:9, 1:9] = True
    holed[4, 4] = False                  # an unmapped island inside the room
    array = _array(_state(room_masks={0: holed}, skeleton=None))
    polys = _by_ns(array, "room_polygons")
    assert len(polys) == 2               # outer boundary + the hole
    assert all(m.ns == "room_polygons" for m in polys)
    # Both are the room's colour: the hole belongs to the same room.
    for marker in polys:
        assert (marker.color.r, marker.color.g, marker.color.b) == \
            pytest.approx(room_color(0))


def test_room_polygons_degrade_to_nothing_without_skimage(monkeypatch,
                                                          caplog):
    """No skimage costs the outlines and nothing else, loudly and once."""
    monkeypatch.setattr(sm, "find_contours", None)
    monkeypatch.setattr(sm, "_WARNED_NO_SKIMAGE", [])
    with caplog.at_level("ERROR", logger=sm.__name__):
        array = _array()
        second = _array()
    assert _by_ns(array, "room_polygons") == []
    assert len(_by_ns(array, "room_fill")) == 2      # the rest still drawn
    assert len(_by_ns(second, "room_polygons")) == 0
    errors = [r for r in caplog.records if "room_polygons" in r.getMessage()]
    assert len(errors) == 1              # once, not once per marker period


# ── object -> parent room edges ────────────────────────────────────────
def test_object_edge_runs_from_the_object_up_to_its_room_centroid():
    edges = _by_ns(_array(), "room_object_edges")
    assert len(edges) == 3
    edge = edges[0]
    assert edge.type == Marker.LINE_LIST
    assert len(edge.points) == 2         # exactly one segment per object
    assert edge.scale.x == pytest.approx(sm.OBJECT_EDGE_WIDTH_M)
    assert (edge.color.r, edge.color.g, edge.color.b) == \
        pytest.approx(sm.OBJECT_EDGE_COLOR)
    assert edge.color.a == pytest.approx(sm.OBJECT_EDGE_ALPHA)
    start, end = edge.points
    assert (start.x, start.y, start.z) == pytest.approx(
        (0.4, -1.2, sm.Z_OBJECT))
    assert (end.x, end.y, end.z) == pytest.approx((0.5, -1.5, sm.Z_CENTROID))


def test_every_object_edge_ends_on_its_own_rooms_centroid():
    ends = [(e.points[1].x, e.points[1].y)
            for e in _by_ns(_array(), "room_object_edges")]
    assert sorted(ends) == [(0.5, -1.5), (0.5, -1.5), (2.5, 1.5)]


def test_object_outside_every_room_draws_no_edge():
    """The bed is on the map but in no room: cube and label, no edge."""
    state = _state(objects_by_room={0: [OBJECTS[0], OBJECTS[1]]})
    array = _array(state)
    edges = _by_ns(array, "room_object_edges")
    assert len(edges) == 2
    assert all(e.points[1].y == pytest.approx(-1.5) for e in edges)
    # The object itself is untouched by the missing link.
    assert len(_by_ns(array, "object_labels")) == 3
    assert _by_ns(array, "objects:bed") != []


def test_object_edge_for_a_room_that_vanished_is_dropped():
    """A grouping whose pid has no centroid has nothing to point at."""
    state = _state(objects_by_room={9: [OBJECTS[0]]})
    assert _by_ns(_array(state), "room_object_edges") == []


# ── degenerate input ───────────────────────────────────────────────────
def test_empty_scene_still_publishes_a_clearing_array():
    array = _array(sm.SceneMarkerState(resolution=RESOLUTION,
                                       origin_xy=ORIGIN))
    assert len(array.markers) == 1
    assert array.markers[0].action == Marker.DELETEALL
