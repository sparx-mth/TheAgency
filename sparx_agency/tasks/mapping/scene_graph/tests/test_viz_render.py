"""Tests for the scene-graph mission-dashboard renderer.

``viz_render`` is pure functions over parsed-JSON dicts and numpy grids, so the
whole picture is checkable by sampling the returned canvas — no ROS, no
display, no torch. The four things worth pinning down:

* the **startup case** — nothing has arrived yet, every state key absent or
  None, and the dashboard must still produce a frame instead of raising;
* the **live case** — a full snapshot paints room tints, the BEV, doors,
  objects, the trail and the drone, and the room colors land as the stable
  ``room_color`` palette converted to BGR (an RGB/BGR swap here is invisible in
  code review and obvious in a pixel sample);
* the **probability bars** scale with probability, which is the one number an
  operator reads off the side panel mid-flight;
* the **target banner** appears only once the target has actually been seen;
* the **room graph** — gold room -> door -> room edges only for *discovered*
  doors, green object -> room links (capped per room), crisp per-room outlines,
  room nodes in the room palette, door state, and the legend — which is the
  half of the old RViz view the dashboard was missing.

Colors are compared with a small tolerance where they alpha-blend; the graph
marks are drawn opaque, so those are matched exactly. Gold in particular can be
counted exactly: neither ``room_color`` (v=0.95, so no channel reaches 255) nor
``class_color`` (s=0.80, so no channel reaches 0) can ever produce
``EDGE_BGR``, which makes "is there an edge here?" a clean pixel question. The
green object link *is* blended, so its tests use states with no BEV, no room
grid and no backdrop: flat ``PANEL_BG`` underneath turns the blend back into
one exact color (see :func:`link_bgr`).
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.mapping.objects.landmarks import class_color
from sparx_agency.core.mapping.topology import room_color
from sparx_agency.tasks.mapping.scene_graph import viz_canvas as vc
from sparx_agency.tasks.mapping.scene_graph import viz_graph_overlay as ovl
from sparx_agency.tasks.mapping.scene_graph import viz_render as viz

GRID_RES = 0.25
GRID_ORIGIN = (-5.0, -5.0)
PANEL_SIZE = (400, 300)          # (w, h), small enough to keep tests fast
SIDE_SIZE = (840, 900)


# ── expectation helpers (deliberately independent of the module) ────────

def room_bgr(pid):
    """The room palette color as OpenCV BGR — recomputed, not imported.

    Written out here on purpose: if ``viz_render`` ever forgets the RGB->BGR
    flip, importing its private converter would hide the bug, while this
    independent expectation catches it.
    """
    r, g, b = room_color(int(pid))
    return np.array([int(b * 255), int(g * 255), int(r * 255)], dtype=np.uint8)


def object_bgr(class_name):
    """The object-class palette color as OpenCV BGR (same reasoning)."""
    r, g, b = class_color(str(class_name))
    return np.array([int(b * 255), int(g * 255), int(r * 255)], dtype=np.uint8)


def blend(base, color, alpha):
    """Alpha-blend one flat color over another, as uint8."""
    return (np.asarray(base, dtype=np.float32) * (1.0 - alpha)
            + np.asarray(color, dtype=np.float32) * alpha).astype(np.uint8)


def count_color(img, color, tol=0):
    """Pixels of ``img`` within ``tol`` counts of ``color`` on every channel."""
    diff = np.abs(img.astype(np.int16) - np.asarray(color, dtype=np.int16))
    return int(np.sum(np.all(diff <= tol, axis=2)))


def px(img, x, y):
    """One pixel as a plain int triple (BGR)."""
    return tuple(int(v) for v in img[y, x])


def window(img, x, y, radius):
    """The square of side ``2*radius+1`` centred on a pixel, clipped."""
    y0, y1 = max(0, y - radius), min(img.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(img.shape[1], x + radius + 1)
    return img[y0:y1, x0:x1]


def gold_near(panel, extent, size, xy, radius=3):
    """Gold edge pixels within ``radius`` of a world point."""
    x, y = viz.world_to_px(extent, size, float(xy[0]), float(xy[1]))
    return count_color(window(panel, x, y, radius), vc.EDGE_BGR)


def link_bgr():
    """The object -> room link as it lands on a bare panel.

    The link is blended, not opaque, so its pixel value is only fixed once the
    background is: every state in :class:`TestRoomObjectLinks` deliberately
    carries no BEV, no room grid and no backdrop, which leaves flat
    ``PANEL_BG`` under every link and makes this an exact color.
    """
    return blend(vc.PANEL_BG, vc.OBJ_EDGE_BGR, vc.OBJ_EDGE_ALPHA)


def link_near(panel, extent, size, xy, radius=2):
    """Object-link pixels within ``radius`` of a world point."""
    x, y = viz.world_to_px(extent, size, float(xy[0]), float(xy[1]))
    return count_color(window(panel, x, y, radius), link_bgr(), tol=1)


def midpoint(a, b):
    """The world point halfway along a segment (the affine maps it to the
    halfway pixel, so it is always *on* the drawn line)."""
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


# ── synthetic state ─────────────────────────────────────────────────────

def room_grid():
    """A pid-label grid: two rooms (labels 1 and 2) either side of a wall."""
    grid = np.zeros((40, 40), dtype=np.int8)
    grid[5:35, 4:18] = 1
    grid[5:35, 22:36] = 2
    return {"grid": grid, "resolution": GRID_RES, "origin": GRID_ORIGIN}


def bev_grid():
    """A BEV: unknown border, free interior, one occupied wall down the middle."""
    grid = np.full((40, 40), -1, dtype=np.int8)
    grid[2:38, 2:38] = 0
    grid[2:38, 19:21] = 100
    return {"grid": grid, "resolution": GRID_RES, "origin": GRID_ORIGIN}


def full_state():
    """Everything the node can ever hold, in the shapes of the topic contract."""
    objects = [
        {"id": 0, "class": "chair", "xy": [-3.0, 2.5], "count": 3},
        {"id": 1, "class": "hospital bed", "xy": [2.0, 2.0], "count": 5},
        {"id": 2, "class": "iv stand", "xy": [3.0, -2.0], "count": 2},
    ]
    return {
        "bev": bev_grid(),
        "room_grid": room_grid(),
        "scene_graph": {
            "stamp": 10.0, "resolution": GRID_RES, "origin": list(GRID_ORIGIN),
            # label 1 -> pid 0, label 2 -> pid 1 (the viz reads this map)
            "grid_pid_map": {"1": 0, "2": 1},
            "rooms": [
                {"id": 0, "centroid": [-2.4, 0.0], "cells": 420,
                 "time_in_room_s": 12.0, "frontier_clusters": 2,
                 "color": list(room_color(0)), "objects": [objects[0]],
                 "doors": [0]},
                {"id": 1, "centroid": [2.4, 0.0], "cells": 410,
                 "time_in_room_s": 3.0, "frontier_clusters": 0,
                 "color": list(room_color(1)), "objects": objects[1:],
                 "doors": [0, 1]},
            ],
            "doors": [
                {"index": 0, "xy": [0.0, -1.0], "discovered": True,
                 "rooms": [0, 1], "room_pairs": [[0, 1]]},
                {"index": 1, "xy": [4.0, -4.0], "discovered": False,
                 "rooms": [], "room_pairs": []},
            ],
            "drone": {"xy": [-1.0, -2.0], "room_id": 0},
        },
        "objects": {"stamp": 10.0, "objects": objects},
        "room_labels": {
            "0": {"label": "ward", "confidence": 0.8, "reasoning": "beds"},
            "1": {"label": "corridor", "confidence": 0.6, "reasoning": "long"},
        },
        "oracle": {"stamp": 10.0, "target": "wheelchair", "model": "gpt-x",
                   "source": "llm",
                   "rooms": [
                       {"id": 0, "label": "ward", "prob": 0.9, "reason": "beds",
                        "time_in_room_s": 12.0, "frontier_clusters": 2},
                       {"id": 1, "label": "corridor", "prob": 0.1,
                        "reason": "transit", "time_in_room_s": 3.0,
                        "frontier_clusters": 0},
                   ]},
        "target_seen": False,
        "target_info": None,
        "pose": (-1.0, -2.0, 0.7),
        "trail": [(-4.0, -4.0), (-3.0, -3.5), (-2.0, -3.0), (-1.0, -2.0)],
        "sim_time": 42.0,
        "footer": ["bev age 0.5s", "oracle tick 2.0s ago", "room grid up"],
    }


def oracle_state(entries):
    """A side-panel state carrying only oracle rows."""
    return {"oracle": {"target": "wheelchair", "source": "llm",
                       "rooms": list(entries)}}


# Two rooms three metres either side of the origin. Only room centroids feed
# ``compute_extent`` in a graph-only state, so these two points alone fix the
# zoom at (-4.5, -3.375) .. (4.5, 3.375) on a 400x300 panel — every world
# coordinate the link tests name is chosen against that rectangle.
LINK_CENTROIDS = {0: (-3.0, 0.0), 1: (3.0, 0.0)}


def obj_entry(oid, xy, class_name="chair"):
    """One object in the wire shape ``payloads.object_entry`` produces."""
    return {"id": int(oid), "class": class_name,
            "xy": [float(xy[0]), float(xy[1])], "count": 1}


def link_state(objects_by_room, doors=()):
    """A graph-only state: the two rooms above, owning the objects given.

    No BEV, no room grid, no trail and no pose, so the panel is flat
    ``PANEL_BG`` everywhere the graph itself does not paint — which is what
    makes :func:`link_bgr` an exact expectation rather than a range.
    """
    rooms = [{"id": pid, "centroid": list(xy), "cells": 100,
              "time_in_room_s": 0.0, "frontier_clusters": 0,
              "objects": list(objects_by_room.get(pid, [])), "doors": []}
             for pid, xy in sorted(LINK_CENTROIDS.items())]
    return {"scene_graph": {"rooms": rooms, "doors": list(doors)}}


class TestWorldToPx:
    """The world->pixel affine: y flips, and everything clamps into the panel."""

    EXTENT = (-5.0, -5.0, 5.0, 5.0)

    def test_max_y_is_row_zero_and_min_y_is_the_last_row(self):
        assert viz.world_to_px(self.EXTENT, (100, 100), -5.0, 5.0) == (0, 0)
        assert viz.world_to_px(self.EXTENT, (100, 100), 4.99, -4.99) == (99, 99)

    def test_centre_maps_to_centre(self):
        assert viz.world_to_px(self.EXTENT, (100, 100), 0.0, 0.0) == (50, 50)

    def test_points_outside_the_extent_clamp_into_the_panel(self):
        assert viz.world_to_px(self.EXTENT, (100, 100), -500.0, 500.0) == (0, 0)
        assert viz.world_to_px(self.EXTENT, (100, 100), 500.0, -500.0) == (99, 99)

    def test_a_degenerate_extent_does_not_divide_by_zero(self):
        assert viz.world_to_px((1.0, 1.0, 1.0, 1.0), (64, 48), 1.0, 1.0) \
            == (0, 0)


class TestEmptyState:
    """Startup: nothing has arrived on any topic yet."""

    def test_render_scene_with_a_completely_empty_state(self):
        canvas = viz.render_scene({})
        assert canvas.shape == (900, 1600, 3)
        assert canvas.dtype == np.uint8

    def test_render_scene_honours_a_custom_size(self):
        canvas = viz.render_scene({}, size=(640, 480), map_panel_w=320)
        assert canvas.shape == (480, 640, 3)
        assert canvas.dtype == np.uint8

    def test_render_scene_with_every_contract_key_present_but_empty(self):
        """Exactly the node's snapshot before the first message arrives."""
        state = {"bev": None, "room_grid": None, "scene_graph": None,
                 "room_labels": None, "oracle": None, "objects": None,
                 "target_seen": False, "target_info": None, "pose": None,
                 "trail": [], "sim_time": 0.0, "footer": []}
        canvas = viz.render_scene(state, size=(640, 480), map_panel_w=320)
        assert canvas.shape == (480, 640, 3)
        assert canvas.dtype == np.uint8

    def test_empty_map_panel_is_the_flat_background(self):
        panel = viz.render_map_panel({}, None, PANEL_SIZE)
        assert panel.shape == (PANEL_SIZE[1], PANEL_SIZE[0], 3)
        assert count_color(panel, vc.PANEL_BG) == panel.shape[0] * panel.shape[1]

    def test_empty_side_panel_still_draws_its_waiting_text(self):
        panel = viz.render_side_panel({}, SIDE_SIZE)
        assert panel.shape == (SIDE_SIZE[1], SIDE_SIZE[0], 3)
        # Title, subtitle, divider and the "waiting for rooms..." line: the
        # panel is never a single flat color even with no state at all.
        assert len(np.unique(panel.reshape(-1, 3), axis=0)) > 1

    def test_a_one_point_trail_does_not_raise(self):
        viz.render_map_panel({"trail": [(0.0, 0.0)]}, None, PANEL_SIZE)

    def test_a_scene_graph_with_no_rooms_or_doors_renders(self):
        state = {"scene_graph": {"rooms": [], "doors": [], "drone": None}}
        viz.render_scene(state, size=(640, 480), map_panel_w=320)


class TestFullState:
    """A live snapshot: every layer must actually reach the canvas."""

    @pytest.fixture
    def panel(self):
        return viz.render_map_panel(full_state(), None, PANEL_SIZE)

    def test_render_scene_shape_and_dtype(self):
        canvas = viz.render_scene(full_state(), size=(1000, 620),
                                  map_panel_w=520)
        assert canvas.shape == (620, 1000, 3)
        assert canvas.dtype == np.uint8

    def test_room_tints_land_as_the_room_palette_in_bgr(self, panel):
        """Sample inside each room: the tint is that room's color, blended."""
        free = blend(vc.PANEL_BG, vc.BEV_FREE_BGR, vc.BEV_FREE_ALPHA)
        for pid, world_xy in ((0, (-3.0, 0.0)), (1, (3.0, 0.0))):
            expected = blend(free, room_bgr(pid), vc.ROOM_TINT_ALPHA)
            x, y = viz.world_to_px(viz.compute_extent(full_state(), None,
                                                      PANEL_SIZE),
                                   PANEL_SIZE, world_xy[0], world_xy[1])
            got = np.array(px(panel, x, y), dtype=np.int16)
            assert np.all(np.abs(got - expected.astype(np.int16)) <= 2), (
                "room %d sampled %s, expected ~%s" % (pid, got, expected))
            assert count_color(panel, expected, tol=2) > 500

    def test_the_two_rooms_are_visibly_different_colors(self, panel):
        extent = viz.compute_extent(full_state(), None, PANEL_SIZE)
        left = px(panel, *viz.world_to_px(extent, PANEL_SIZE, -3.0, 0.0))
        right = px(panel, *viz.world_to_px(extent, PANEL_SIZE, 3.0, 0.0))
        assert sum(abs(a - b) for a, b in zip(left, right)) > 40

    def test_occupied_bev_cells_are_painted_opaque(self, panel):
        assert count_color(panel, vc.BEV_OCC_BGR) > 100

    def test_the_trail_and_the_drone_are_painted(self, panel):
        assert count_color(panel, vc.TRAIL_BGR) > 20
        assert count_color(panel, vc.DRONE_BGR) > 20

    def test_a_discovered_door_is_filled_and_an_undiscovered_one_is_hollow(
            self, panel):
        filled = count_color(panel, vc.DOOR_BGR)
        hollow = count_color(panel, vc.DOOR_UNDISCOVERED_BGR)
        assert filled > 0 and hollow > 0
        assert filled > hollow, (
            "the discovered door should paint more pixels than the outline of "
            "the undiscovered one (%d vs %d)" % (filled, hollow))

    def test_every_object_is_drawn_in_its_class_color(self, panel):
        for name in ("chair", "hospital bed", "iv stand"):
            assert count_color(panel, object_bgr(name)) > 5, (
                "no %s dot on the canvas" % name)

    def test_the_full_scene_is_not_a_flat_image(self):
        canvas = viz.render_scene(full_state(), size=(1000, 620),
                                  map_panel_w=520)
        assert len(np.unique(canvas.reshape(-1, 3), axis=0)) > 50

    def test_room_tint_falls_back_to_the_label_value_without_a_pid_map(self):
        """No ``grid_pid_map`` -> the grid label IS the pid (documented)."""
        state = {"room_grid": room_grid(), "scene_graph": {"rooms": []}}
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        for label in (1, 2):
            expected = blend(vc.PANEL_BG, room_bgr(label), vc.ROOM_TINT_ALPHA)
            assert count_color(panel, expected, tol=2) > 500


class TestProbabilityBars:
    """The side panel's one quantitative element."""

    @staticmethod
    def bar_pixels(prob, room_id=3):
        """Room-colored pixels in the side panel (swatch + probability bar)."""
        panel = viz.render_side_panel(
            oracle_state([{"id": room_id, "label": "ward", "prob": prob}]),
            SIDE_SIZE)
        return count_color(panel, room_bgr(room_id))

    def test_a_likely_room_paints_more_bar_than_an_unlikely_one(self):
        assert self.bar_pixels(0.9) > self.bar_pixels(0.1)

    def test_bar_length_is_proportional_to_probability(self):
        """Nine times the probability, nine times the bar (swatch removed)."""
        empty = self.bar_pixels(0.0)
        low = self.bar_pixels(0.1) - empty
        high = self.bar_pixels(0.9) - empty
        assert low > 0
        assert 8.0 < high / float(low) < 10.0, (
            "bar scaling is not linear in probability: %d vs %d" % (high, low))

    def test_probability_is_clamped_to_one(self):
        assert self.bar_pixels(5.0) == self.bar_pixels(1.0)

    def test_negative_probability_does_not_paint_a_backwards_bar(self):
        assert self.bar_pixels(-1.0) <= self.bar_pixels(0.0)

    def test_rows_are_ordered_by_descending_probability(self):
        """The likelier room's swatch sits above the unlikelier one's."""
        panel = viz.render_side_panel(
            oracle_state([{"id": 3, "label": "hall", "prob": 0.1},
                          {"id": 7, "label": "ward", "prob": 0.9}]),
            SIDE_SIZE)
        rows = {}
        for pid in (3, 7):
            hits = np.nonzero(np.all(panel == room_bgr(pid), axis=2))[0]
            assert hits.size > 0, "room %d has no row" % pid
            rows[pid] = int(hits.min())
        assert rows[7] < rows[3], "p=0.9 room must be drawn above p=0.1"

    def test_a_room_the_oracle_has_not_scored_yet_still_gets_a_row(self):
        state = {"scene_graph": {"rooms": [
            {"id": 5, "centroid": [0.0, 0.0], "cells": 30,
             "time_in_room_s": 1.0, "frontier_clusters": 1, "objects": []}]}}
        panel = viz.render_side_panel(state, SIDE_SIZE)
        assert count_color(panel, room_bgr(5)) > 0

    def test_a_scored_room_absent_from_the_scene_graph_still_gets_a_row(self):
        panel = viz.render_side_panel(
            oracle_state([{"id": 11, "label": "lab", "prob": 0.4}]), SIDE_SIZE)
        assert count_color(panel, room_bgr(11)) > 0

    def test_many_rooms_do_not_overflow_the_panel(self):
        rows = [{"id": i, "label": "r%d" % i, "prob": 1.0 - 0.01 * i}
                for i in range(40)]
        panel = viz.render_side_panel(oracle_state(rows), SIDE_SIZE)
        assert panel.shape == (SIDE_SIZE[1], SIDE_SIZE[0], 3)


class TestTargetBanner:
    """The banner is the mission's terminal state — it must not appear early."""

    BANNER_BGR = (0, 0, 120)

    def test_no_banner_before_the_target_is_seen(self):
        panel = viz.render_map_panel(full_state(), None, PANEL_SIZE)
        assert count_color(panel[:34], self.BANNER_BGR) == 0

    def test_banner_and_ring_are_painted_once_the_target_is_seen(self):
        state = full_state()
        state["target_seen"] = True
        state["target_info"] = {"target": "wheelchair",
                                "matched_class": "hospital bed",
                                "xy": [2.0, 2.0], "conf": 0.87}
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel[:34], self.BANNER_BGR) > 1000
        assert count_color(panel, vc.TARGET_BGR) > 20

    def test_banner_survives_a_target_info_without_a_position(self):
        state = {"target_seen": True, "target_info": {"target": "wheelchair"}}
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel[:34], self.BANNER_BGR) > 1000

    def test_banner_survives_a_missing_target_info(self):
        panel = viz.render_map_panel({"target_seen": True}, None, PANEL_SIZE)
        assert count_color(panel[:34], self.BANNER_BGR) > 1000


class TestRoomGraphEdges:
    """The headline of the old RViz view: gold room -> door -> room links.

    The dog-leg matters, not just the presence of gold: a straight
    centroid-to-centroid line would cut through the wall the door is in, so the
    edge is checked at both leg midpoints *and* at the door itself.
    """

    EXTENT = None  # filled per test from the same state the panel was drawn from

    @staticmethod
    def panel_and_extent(state, size=PANEL_SIZE):
        return (viz.render_map_panel(state, None, size),
                viz.compute_extent(state, None, size))

    def test_a_discovered_door_links_its_two_rooms_in_gold(self):
        panel, extent = self.panel_and_extent(full_state())
        # room 0 (-2.4, 0) -> door (0, -1) -> room 1 (2.4, 0)
        assert gold_near(panel, extent, PANEL_SIZE, (-1.2, -0.5)) > 0, \
            "no gold on the leg from room 0 to the door"
        assert gold_near(panel, extent, PANEL_SIZE, (1.2, -0.5)) > 0, \
            "no gold on the leg from the door to room 1"

    def test_the_edge_passes_through_the_door_not_between_the_centroids(self):
        """Gold at the door, and none on the straight line the door is off."""
        panel, extent = self.panel_and_extent(full_state())
        assert gold_near(panel, extent, PANEL_SIZE, (0.0, -1.0), radius=12) > 0
        # The straight centroid-to-centroid line would run along y = 0 here;
        # the dog-leg is a full door offset below it.
        assert gold_near(panel, extent, PANEL_SIZE, (-1.2, 0.0)) == 0

    def test_an_undiscovered_door_draws_no_edge(self):
        """Not been through it -> the mapper has not proved the rooms connect."""
        state = full_state()
        state["scene_graph"]["doors"][0]["discovered"] = False
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, vc.EDGE_BGR) == 0

    def test_a_door_linking_an_unknown_room_draws_no_edge(self):
        """Room pids restart on a BEV reshape; a stale link must not paint."""
        state = full_state()
        state["scene_graph"]["doors"][0]["room_pairs"] = [[0, 99]]
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, vc.EDGE_BGR) == 0

    def test_a_door_naming_one_room_twice_draws_no_self_edge(self):
        state = full_state()
        state["scene_graph"]["doors"][0]["room_pairs"] = [[1, 1]]
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, vc.EDGE_BGR) == 0

    def test_a_door_without_a_room_pairs_key_does_not_raise(self):
        state = full_state()
        del state["scene_graph"]["doors"][0]["room_pairs"]
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, vc.EDGE_BGR) == 0

    def test_rooms_near_a_door_are_not_an_edge_without_a_vetted_pair(self):
        """The false-edge fix, at the drawing end.

        The mapper vets a door's rooms against which regions genuinely
        touch and ships the survivors as pairs. A door that still names
        two rooms but carries no pair is a door with a wall behind it,
        and the panel must stay clean.
        """
        state = full_state()
        state["scene_graph"]["doors"][0]["room_pairs"] = []
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, vc.EDGE_BGR) == 0

    def test_the_edge_is_drawn_under_the_object_dots(self):
        """Objects and their labels stay readable over the graph."""
        state = full_state()
        state["objects"]["objects"] = [
            {"id": 9, "class": "chair", "xy": [-1.2, -0.5], "count": 1}]
        panel, extent = self.panel_and_extent(state)
        x, y = viz.world_to_px(extent, PANEL_SIZE, -1.2, -0.5)
        assert np.array_equal(np.array(px(panel, x, y)), object_bgr("chair"))


class TestRoomNodes:
    """Rooms read as graph nodes, not just tinted blobs."""

    def test_a_node_disc_sits_at_each_centroid_in_the_room_color(self):
        state = full_state()
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        extent = viz.compute_extent(state, None, PANEL_SIZE)
        for pid, centroid in ((0, (-2.4, 0.0)), (1, (2.4, 0.0))):
            x, y = viz.world_to_px(extent, PANEL_SIZE, *centroid)
            assert np.array_equal(np.array(px(panel, x, y)), room_bgr(pid)), \
                "room %d node is not the room color" % pid

    def test_the_node_has_a_white_ring_around_the_disc(self):
        state = full_state()
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        extent = viz.compute_extent(state, None, PANEL_SIZE)
        x, y = viz.world_to_px(extent, PANEL_SIZE, -2.4, 0.0)
        assert count_color(window(panel, x, y, 8), vc.NODE_RING_BGR) > 8

    def test_the_chip_is_the_old_marker_text_plus_what_the_llm_knows(self):
        room = {"id": 3, "centroid": [0.0, 0.0], "time_in_room_s": 12.4,
                "frontier_clusters": 2}
        labels = {"3": {"label": "ward"}}
        assert ovl._room_chip_text(room, 3, labels, {3: 0.87}) == \
            "R3 ward  t=12s F=2  p=0.87"

    def test_the_chip_survives_no_label_and_no_probability(self):
        room = {"id": 5, "centroid": [0.0, 0.0], "time_in_room_s": 0.0,
                "frontier_clusters": 0}
        assert ovl._room_chip_text(room, 5, None, None) == "R5  t=0s F=0"

    def test_the_chip_reports_dwell_and_frontiers_without_the_oracle(self):
        room = {"id": 2, "centroid": [0.0, 0.0], "time_in_room_s": 61.6,
                "frontier_clusters": 4}
        assert ovl._room_chip_text(room, 2, {}, {}) == "R2  t=62s F=4"


class TestDoorState:
    """Seen vs pending doors, and the zoom-dependent ``d{index}`` labels."""

    @staticmethod
    def door_state(discovered, xy=(0.0, 0.0), index=7, span=2.0):
        """One door, and a trail that fixes the zoom (doors do not set it)."""
        return {"trail": [(-span, -span), (span, span)],
                "scene_graph": {"rooms": [],
                                "doors": [{"index": index, "xy": list(xy),
                                           "discovered": discovered,
                                           "rooms": []}]}}

    def test_a_discovered_door_paints_more_than_a_pending_outline(self):
        seen = viz.render_map_panel(self.door_state(True), None, PANEL_SIZE)
        pending = viz.render_map_panel(self.door_state(False), None, PANEL_SIZE)
        assert count_color(seen, vc.DOOR_BGR) > \
            count_color(pending, vc.DOOR_UNDISCOVERED_BGR)

    def test_pixels_per_metre_is_the_panel_width_over_the_world_width(self):
        assert vc.pixels_per_metre((-5.0, -5.0, 5.0, 5.0), (400, 300)) == 40.0
        assert vc.pixels_per_metre((0.0, 0.0, 0.0, 0.0), (400, 300)) > 0.0

    def test_door_labels_are_dropped_when_the_view_is_too_zoomed_out(self):
        """A hospital-wide view would otherwise be a field of overlapping text."""
        close = self.door_state(True)
        far = self.door_state(True, span=400.0)   # a hospital-wide zoom
        close_ppm = vc.pixels_per_metre(
            viz.compute_extent(close, None, PANEL_SIZE), PANEL_SIZE)
        far_ppm = vc.pixels_per_metre(
            viz.compute_extent(far, None, PANEL_SIZE), PANEL_SIZE)
        assert close_ppm > ovl.DOOR_LABEL_MIN_PPM > far_ppm
        # tol: OpenCV's hairline glyph stroke lands a couple of counts under
        # the requested color even with antialiasing off, so an exact match
        # would read "no label" for a label that is plainly there.
        labelled = count_color(
            viz.render_map_panel(close, None, PANEL_SIZE), vc.DOOR_LABEL_BGR,
            tol=4)
        unlabelled = count_color(
            viz.render_map_panel(far, None, PANEL_SIZE), vc.DOOR_LABEL_BGR,
            tol=4)
        assert labelled > 0 and unlabelled == 0

    def test_a_door_without_an_index_still_draws_its_glyph(self):
        state = self.door_state(True)
        del state["scene_graph"]["doors"][0]["index"]
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, vc.DOOR_BGR) > 0


class TestRoomObjectLinks:
    """The containment half of the graph: every object tied to its room.

    ``/scene_graph`` already says which room owns which object (each room
    carries its own ``objects`` list), so these tests are about the *picture*:
    a line that actually reaches from the dot to the node, nothing drawn for a
    room that owns nothing, the per-room cap honoured, and the link staying
    underneath the gold door topology it must not compete with.
    """

    @staticmethod
    def panel_and_extent(state, size=PANEL_SIZE):
        return (viz.render_map_panel(state, None, size),
                viz.compute_extent(state, None, size))

    def test_each_object_is_linked_to_the_centroid_of_its_room(self):
        """Two objects in room 0 -> green on both segments, at both ends."""
        first, second = (-3.5, 2.0), (-1.8, -2.0)
        state = link_state({0: [obj_entry(0, first),
                                obj_entry(1, second, "hospital bed")]})
        panel, extent = self.panel_and_extent(state)
        node = LINK_CENTROIDS[0]
        for where in (first, second):
            assert link_near(panel, extent, PANEL_SIZE,
                             midpoint(where, node)) > 0, \
                "no link on the segment from %s to the room node" % (where,)
            # A quarter of the way along too: a stub at the midpoint would
            # pass the check above without the line ever reaching the object.
            assert link_near(panel, extent, PANEL_SIZE,
                             midpoint(where, midpoint(where, node))) > 0, \
                "the link from %s does not reach the object" % (where,)

    def test_a_room_that_owns_nothing_draws_no_links(self):
        panel = viz.render_map_panel(link_state({}), None, PANEL_SIZE)
        assert count_color(panel, link_bgr(), tol=1) == 0

    def test_an_object_links_only_to_the_room_that_claims_it(self):
        """Room 0's object gets no line to room 1's node."""
        where = (-3.5, 2.0)
        state = link_state({0: [obj_entry(0, where)]})
        panel, extent = self.panel_and_extent(state)
        assert link_near(panel, extent, PANEL_SIZE,
                         midpoint(where, LINK_CENTROIDS[0])) > 0
        assert link_near(panel, extent, PANEL_SIZE,
                         midpoint(where, LINK_CENTROIDS[1])) == 0

    def test_crossing_links_are_the_same_green_as_a_lone_one(self):
        """One blend for the whole layer — a crossing must not darken.

        Room 0 owns an object over on room 1's side and vice versa, so the two
        links provably cross, at (0, 0.6).
        """
        state = link_state({0: [obj_entry(0, (2.0, 1.0))],
                            1: [obj_entry(1, (-2.0, 1.0))]})
        panel, extent = self.panel_and_extent(state)
        assert link_near(panel, extent, PANEL_SIZE, (0.0, 0.6),
                         radius=1) > 0, \
            "the two links do not actually cross where this test assumes"
        double = blend(link_bgr(), vc.OBJ_EDGE_BGR, vc.OBJ_EDGE_ALPHA)
        assert count_color(panel, double, tol=1) == 0, \
            "a crossing blended twice into a second, darker green"

    def test_the_link_is_drawn_under_the_gold_door_edges(self):
        """Gold is the skeleton; the containment link is a soft underlay."""
        state = link_state({0: [obj_entry(0, LINK_CENTROIDS[1])]},
                           doors=[{"index": 0, "xy": [0.0, 0.0],
                                   "discovered": True, "rooms": [0, 1],
                                   "room_pairs": [[0, 1]]}])
        panel, extent = self.panel_and_extent(state)
        # The link and the gold leg run along exactly the same pixel row here.
        assert gold_near(panel, extent, PANEL_SIZE, (1.5, 0.0)) > 0
        assert link_near(panel, extent, PANEL_SIZE, (1.5, 0.0), radius=1) == 0

    def test_the_link_is_drawn_under_the_object_dots(self):
        """The dot and its class color stay readable on top of the link."""
        where = (-3.5, 2.0)
        state = link_state({0: [obj_entry(0, where)]})
        state["objects"] = {"objects": [obj_entry(0, where)]}
        panel, extent = self.panel_and_extent(state)
        x, y = viz.world_to_px(extent, PANEL_SIZE, where[0], where[1])
        assert np.array_equal(np.array(px(panel, x, y)), object_bgr("chair"))

    def test_links_are_capped_per_room_and_the_surplus_is_reported(self):
        """Past the cap a ward's links stop being a graph and become a fan."""
        cap = ovl.MAX_OBJECT_LINKS_PER_ROOM
        node = LINK_CENTROIDS[0]
        kept = [obj_entry(i, (node[0] + 0.5 * np.cos(i * 0.31),
                              node[1] + 0.5 * np.sin(i * 0.31)))
                for i in range(cap)]
        far = [(3.2 + 0.05 * k, -3.0) for k in range(10)]
        surplus = [obj_entry(cap + k, far[k]) for k in range(10)]

        at_cap = viz.render_map_panel(link_state({0: kept}), None, PANEL_SIZE)
        over = link_state({0: kept + surplus})
        panel, extent = self.panel_and_extent(over)

        assert count_color(at_cap, link_bgr(), tol=1) > 0
        assert count_color(panel, link_bgr(), tol=1) == \
            count_color(at_cap, link_bgr(), tol=1), \
            "objects past the cap still painted links"
        for where in far:
            assert link_near(panel, extent, PANEL_SIZE, where, radius=4) == 0

    def test_the_dropped_count_is_returned_for_the_legend(self):
        cap = ovl.MAX_OBJECT_LINKS_PER_ROOM
        objects = [obj_entry(i, (-3.0, 0.5)) for i in range(cap + 7)]
        assert self.dropped({0: objects}) == 7
        assert self.dropped({0: objects[:2]}) == 0

    @staticmethod
    def dropped(objects_by_room):
        """Links the layer refused to draw, straight from the layer itself."""
        panel = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), vc.PANEL_BG,
                        dtype=np.uint8)
        return ovl.draw_room_object_edges(
            panel, link_state(objects_by_room)["scene_graph"],
            (-4.5, -3.375, 4.5, 3.375))

    def test_no_graph_at_all_draws_nothing_and_drops_nothing(self):
        panel = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), vc.PANEL_BG,
                        dtype=np.uint8)
        assert ovl.draw_room_object_edges(panel, None,
                                          (-1.0, -1.0, 1.0, 1.0)) == 0
        assert count_color(panel, vc.PANEL_BG) == \
            PANEL_SIZE[0] * PANEL_SIZE[1]

    def test_an_object_without_a_position_is_skipped_not_fatal(self):
        state = link_state({0: [{"id": 0, "class": "chair", "count": 1},
                                obj_entry(1, (-3.5, 2.0))]})
        panel, extent = self.panel_and_extent(state)
        assert link_near(panel, extent, PANEL_SIZE,
                         midpoint((-3.5, 2.0), LINK_CENTROIDS[0])) > 0

    def test_a_room_without_a_centroid_links_nothing(self):
        """There is no node to link *to*. (The panel as a whole still raises
        on such a payload — ``compute_extent`` needs every centroid, and a
        room without one is a mapper bug worth hearing about — but the link
        layer itself must not be the thing that raises.)"""
        panel = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), vc.PANEL_BG,
                        dtype=np.uint8)
        graph = link_state({0: [obj_entry(0, (-3.5, 2.0))]})["scene_graph"]
        del graph["rooms"][0]["centroid"]
        assert ovl.draw_room_object_edges(panel, graph,
                                          (-4.5, -3.375, 4.5, 3.375)) == 0
        assert count_color(panel, link_bgr(), tol=1) == 0

    def test_a_room_without_an_objects_key_is_not_fatal(self):
        state = link_state({0: [obj_entry(0, (-3.5, 2.0))]})
        del state["scene_graph"]["rooms"][0]["objects"]
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, link_bgr(), tol=1) == 0

    def test_the_live_snapshot_links_every_object_it_maps(self):
        """The full state, not a fixture built for this test."""
        state = full_state()
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        extent = viz.compute_extent(state, None, PANEL_SIZE)
        for room in state["scene_graph"]["rooms"]:
            for obj in room["objects"]:
                mid = midpoint(obj["xy"], room["centroid"])
                x, y = viz.world_to_px(extent, PANEL_SIZE, mid[0], mid[1])
                # Blended over the room tint here, so the exact color is not
                # PANEL_BG's: what is pinned is that the pixel moved toward
                # green relative to the tint it would otherwise show.
                b, g, r = (int(v) for v in panel[y, x])
                assert g > b and g > r, (
                    "no link between object %s and room %d (pixel %s)"
                    % (obj["class"], room["id"], (b, g, r)))


class TestRoomOutlines:
    """Room extents, crisp: the old RViz ``room_polygons`` on the dashboard.

    The tint says roughly where a room is; the outline says exactly where it
    ends, which is what makes two rooms sharing a wall read as two rooms.
    """

    @staticmethod
    def grid_only(pid_map=None):
        """The room grid with no rooms in the graph — so the only pixels that
        can carry a pure room color are outline pixels (nodes would too)."""
        graph = {"rooms": [], "doors": []}
        if pid_map is not None:
            graph["grid_pid_map"] = pid_map
        return {"room_grid": room_grid(), "scene_graph": graph}

    def test_each_room_is_outlined_in_its_own_color(self):
        """No pid map -> the grid value is the pid, same as the tint."""
        panel = viz.render_map_panel(self.grid_only(), None, PANEL_SIZE)
        for label in (1, 2):
            assert count_color(panel, room_bgr(label)) > 0, \
                "room %d has no outline" % label

    def test_the_outline_resolves_through_the_grid_pid_map(self):
        """Grid values are not pids; the outline must follow the same map the
        tint does, or a room would be outlined in a foreign room's color."""
        panel = viz.render_map_panel(self.grid_only({"1": 5, "2": 6}), None,
                                     PANEL_SIZE)
        for pid, label in ((5, 1), (6, 2)):
            assert count_color(panel, room_bgr(pid)) > 0, \
                "grid value %d is not outlined as pid %d" % (label, pid)
            assert count_color(panel, room_bgr(label)) == 0, \
                "grid value %d was outlined as if it were the pid" % label

    def test_the_outline_is_a_boundary_not_a_fill(self):
        """It rings the room: far fewer pixels than the tint, and the middle
        of the room still shows the blended tint underneath."""
        panel = viz.render_map_panel(self.grid_only(), None, PANEL_SIZE)
        extent = viz.compute_extent(self.grid_only(), None, PANEL_SIZE)
        for label, inside in ((1, (-2.5, 0.0)), (2, (2.5, 0.0))):
            tint = blend(vc.PANEL_BG, room_bgr(label), vc.ROOM_TINT_ALPHA)
            outline = count_color(panel, room_bgr(label))
            assert 0 < outline < count_color(panel, tint, tol=2) / 5.0
            x, y = viz.world_to_px(extent, PANEL_SIZE, inside[0], inside[1])
            assert np.all(np.abs(np.array(px(panel, x, y), dtype=np.int16)
                                 - tint.astype(np.int16)) <= 2), \
                "the room interior was painted over by the outline"

    def test_the_outline_sits_on_the_edge_of_the_room(self):
        """Its bounding box is the room's, and it is hollow in the middle."""
        panel = viz.render_map_panel(self.grid_only(), None, PANEL_SIZE)
        tint = blend(vc.PANEL_BG, room_bgr(1), vc.ROOM_TINT_ALPHA)
        rows, cols = np.nonzero(np.all(panel == room_bgr(1), axis=2))
        t_rows, t_cols = np.nonzero(
            np.all(np.abs(panel.astype(np.int16) - tint.astype(np.int16))
                   <= 2, axis=2))
        assert abs(int(rows.min()) - int(t_rows.min())) <= 1
        assert abs(int(rows.max()) - int(t_rows.max())) <= 1
        assert abs(int(cols.min()) - int(t_cols.min())) <= 1
        assert abs(int(cols.max()) - int(t_cols.max())) <= 1

    def test_two_touching_rooms_each_keep_their_own_side(self):
        """A shared boundary is one pixel of each color, not one winner."""
        grid = np.zeros((40, 40), dtype=np.int8)
        grid[5:35, 4:20] = 1
        grid[5:35, 20:36] = 2
        state = {"room_grid": {"grid": grid, "resolution": GRID_RES,
                               "origin": GRID_ORIGIN},
                 "scene_graph": {"rooms": [], "doors": []}}
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, room_bgr(1)) > 0
        assert count_color(panel, room_bgr(2)) > 0

    def test_a_state_with_rooms_but_no_room_grid_still_renders(self):
        """The grid is a separate latched topic and can simply be late."""
        state = full_state()
        del state["room_grid"]
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        extent = viz.compute_extent(state, None, PANEL_SIZE)
        # No outlines without the grid, but the rest of the graph is intact.
        assert gold_near(panel, extent, PANEL_SIZE, (-1.2, -0.5)) > 0
        assert count_color(panel, vc.DOOR_BGR) > 0

    def test_an_empty_room_grid_paints_no_outline(self):
        state = {"room_grid": {"grid": np.zeros((20, 20), dtype=np.int8),
                               "resolution": GRID_RES, "origin": GRID_ORIGIN},
                 "scene_graph": {"rooms": [], "doors": []}}
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, vc.PANEL_BG) == \
            PANEL_SIZE[0] * PANEL_SIZE[1]


class TestLegend:
    """The key to the vocabulary — and it must not clutter a tiny panel."""

    LEGEND_SIZE = (800, 600)

    @staticmethod
    def pose_only():
        """Live content, but nothing that could paint the legend's colors."""
        return {"pose": (0.0, 0.0, 0.0)}

    def test_the_legend_paints_its_whole_vocabulary(self):
        panel = viz.render_map_panel(self.pose_only(), None, self.LEGEND_SIZE)
        # No doors, no edges and no trail in this state, so every one of these
        # can only have come from the legend swatches.
        assert count_color(panel, vc.EDGE_BGR) > 0, "no gold edge swatch"
        assert count_color(panel, vc.DOOR_BGR) > 0, "no door swatch"
        assert count_color(panel, vc.DOOR_UNDISCOVERED_BGR) > 0, \
            "no pending-door swatch"
        assert count_color(panel, vc.TRAIL_BGR) > 0, "no trail swatch"
        assert count_color(panel, room_bgr(0)) > 0, "no room swatch"

    def test_the_legend_sits_in_the_bottom_left(self):
        panel = viz.render_map_panel(self.pose_only(), None, self.LEGEND_SIZE)
        rows, cols = np.nonzero(np.all(panel == vc.EDGE_BGR, axis=2))
        assert rows.min() > self.LEGEND_SIZE[1] // 2
        assert cols.max() < self.LEGEND_SIZE[0] // 2

    def test_no_legend_on_a_panel_too_small_to_read_it(self):
        panel = viz.render_map_panel(self.pose_only(), None, PANEL_SIZE)
        assert count_color(panel, vc.EDGE_BGR) == 0
        assert count_color(panel, vc.TRAIL_BGR) == 0

    def test_no_legend_before_any_live_layer_has_arrived(self):
        """A startup frame explains nothing, so it stays clean."""
        panel = viz.render_map_panel({}, None, self.LEGEND_SIZE)
        assert count_color(panel, vc.PANEL_BG) == \
            self.LEGEND_SIZE[0] * self.LEGEND_SIZE[1]

    def test_the_legend_names_the_object_link_and_the_room_outline(self):
        """Both new elements need a caption or they are unexplained noise."""
        captions = [caption for _, caption in ovl._LEGEND_ROWS]
        assert any("object in room" in c for c in captions)
        assert any("outline" in c for c in captions)
        panel = viz.render_map_panel(self.pose_only(), None, self.LEGEND_SIZE)
        # Opaque in the swatch (the blended map color would vanish on the
        # legend plate), so this can only be the object-link swatch.
        assert count_color(panel, vc.OBJ_EDGE_BGR) > 0, "no object-link swatch"

    def test_the_legend_says_so_when_links_have_been_capped(self):
        """A thinned picture must never pass for a complete one."""
        size = self.LEGEND_SIZE

        def plate_top(dropped):
            panel = np.full((size[1], size[0], 3), vc.PANEL_BG, dtype=np.uint8)
            ovl.draw_legend(panel, dropped)
            rows = np.nonzero(np.all(panel == vc.LEGEND_BORDER_BGR,
                                     axis=2))[0]
            return int(rows.min())

        assert plate_top(12) < plate_top(0), \
            "the capped-links footnote did not grow the legend plate"

    def test_the_capped_footnote_appears_on_a_rendered_frame(self):
        state = full_state()
        crowd = [{"id": 100 + i, "class": "chair",
                  "xy": [-2.4 + 0.01 * i, 0.2], "count": 1}
                 for i in range(ovl.MAX_OBJECT_LINKS_PER_ROOM + 5)]
        state["scene_graph"]["rooms"][0]["objects"] = crowd
        plain = viz.render_map_panel(full_state(), None, self.LEGEND_SIZE)
        capped = viz.render_map_panel(state, None, self.LEGEND_SIZE)

        def top(panel):
            rows = np.nonzero(np.all(panel == vc.LEGEND_BORDER_BGR,
                                     axis=2))[0]
            return int(rows.min())

        assert top(capped) < top(plain)

    def test_the_legend_does_not_hide_the_room_graph(self):
        """Edges still reach the canvas on a legend-sized panel."""
        state = full_state()
        panel = viz.render_map_panel(state, None, self.LEGEND_SIZE)
        extent = viz.compute_extent(state, None, self.LEGEND_SIZE)
        assert gold_near(panel, extent, self.LEGEND_SIZE, (-1.2, -0.5)) > 0


class TestOlderPayloads:
    """The mapper's payload predates the graph view; nothing may hard-require it."""

    def test_a_scene_graph_with_no_doors_key_at_all_renders(self):
        state = full_state()
        del state["scene_graph"]["doors"]
        panel = viz.render_map_panel(state, None, PANEL_SIZE)
        assert count_color(panel, vc.EDGE_BGR) == 0
        assert count_color(panel, room_bgr(0)) > 0   # nodes and tints survive

    def test_rooms_without_dwell_or_frontier_fields_render(self):
        state = full_state()
        for room in state["scene_graph"]["rooms"]:
            room.pop("time_in_room_s")
            room.pop("frontier_clusters")
        viz.render_scene(state, size=(1000, 620), map_panel_w=520)

    def test_the_whole_dashboard_still_renders_with_the_graph_view_on(self):
        canvas = viz.render_scene(full_state())
        assert canvas.shape == (900, 1600, 3)
        assert count_color(canvas, vc.EDGE_BGR) > 0

    def test_rooms_carrying_no_objects_list_render(self):
        """``rooms[].objects`` is what the link layer reads; it may be absent
        on a payload older than the object mapper."""
        state = full_state()
        for room in state["scene_graph"]["rooms"]:
            room.pop("objects")
        canvas = viz.render_scene(state, size=(1000, 620), map_panel_w=520)
        assert canvas.shape == (620, 1000, 3)

    def test_a_scene_graph_with_no_rooms_key_at_all_renders(self):
        state = full_state()
        del state["scene_graph"]["rooms"]
        viz.render_scene(state, size=(1000, 620), map_panel_w=520)
