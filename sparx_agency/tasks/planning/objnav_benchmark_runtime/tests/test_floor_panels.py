"""Minimal map geometry, shared adaptive scale and display-only persistence."""
from types import SimpleNamespace

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import FRONTIER_COLOR, FloorPanels, UNKNOWN_GRAY


def panels_with_rooms():
    settings = SimpleNamespace(map_resolution_m=0.1, map_size_m=80.0)
    panels = FloorPanels(2, settings, SimpleNamespace(x=0.0, y=0.0))
    for floor, slot in enumerate(panels.slots):
        slot["floor_id"] = floor
        panels.bindings[floor] = floor
        slot["grid"][370:430, 300:500] = 0
        slot["grid"][370:430, 300] = 100
        slot["room_labels"][370:430, 301:400] = 1
        slot["room_labels"][370:430, 400:500] = 2
        slot["rooms"] = 2
    return panels


def test_viewport_fits_small_houses_and_grows_beyond_the_old_32_metre_crop():
    panels = panels_with_rooms()
    panels.render(0, size=(640, 720))
    before = panels._bounds
    assert panels.span_m < 32
    assert before[0] <= 300 and before[2] >= 500
    panels.slots[1]["grid"][380:420, 650:720] = 0
    panels.render(1, size=(640, 720))
    after = panels._bounds
    assert panels.span_m > 40
    assert after[0] <= before[0] and after[2] >= 720
    assert panels.metadata()[0]["viewport_cells"] == panels.metadata()[1]["viewport_cells"]
    panels.slots[1]["grid"][380:420, 650:720] = -1
    panels.render(0)
    assert panels._bounds == after, "viewport does not pump when observations disappear"


def test_objects_probabilities_and_visit_order_do_not_clutter_the_map_and_frontiers_are_small_diamonds():
    """The verbose search facts stay off the map; the accessible frontier goals -- the "blue diamonds" the search
    column counts -- are drawn as small markers (since 2026-10-22: before, the column named markers that were not
    there, and a recording could not show which way the inventory pointed)."""
    panels = panels_with_rooms()
    plain = panels.render(0, size=(640, 720))
    panels.slots[0]["objects"] = [{"xy": [i / 10.0, 0.0], "class": "chair", "id": i} for i in range(100)]
    search = {"rooms": [{"id": 0, "centroid": [-5.0, 0.0], "label": "very long room name", "prob": 0.99}],
              "order": [0], "accessible_frontiers": []}
    np.testing.assert_array_equal(plain, panels.render(0, size=(640, 720), search=search))
    search["accessible_frontiers"] = [[0.0, 0.0]]
    marked = panels.render(0, size=(640, 720), search=search)
    changed = np.any(marked != plain, axis=-1)
    assert changed.any() and changed.sum() <= 60, "one small diamond, not a cloud"
    ys, xs = np.nonzero(changed)
    assert ys.max() - ys.min() <= 8 and xs.max() - xs.min() <= 8
    assert np.any(np.all(marked[changed] == FRONTIER_COLOR, axis=-1)), "drawn in the frontier colour (antialiased edges aside)"


def test_boundaries_are_drawn_on_both_floors_and_saved_as_numeric_partitions(tmp_path):
    panels = panels_with_rooms()
    original = [s["room_labels"].copy() for s in panels.slots]
    image = panels.render(0, size=(640, 720))
    for part in (image[:360], image[360:]):
        outline_error = np.max(np.abs(part.astype(np.int16) - (150, 155, 160)), axis=-1)
        assert np.any(outline_error <= 16), "antialiased room outlines on each stacked floor"
    panels.save(tmp_path)
    with np.load(tmp_path / "room_partitions.npz", allow_pickle=False) as saved:
        for i, labels in enumerate(original):
            np.testing.assert_array_equal(labels, saved["slot_%d" % i])
            np.testing.assert_array_equal(labels, panels.slots[i]["room_labels"])


def test_unknown_slots_do_not_acquire_room_geometry_from_other_floors():
    panels = panels_with_rooms()
    panels.slots[1] = panels._unknown()
    before = panels.slots[1]["room_labels"].copy()
    image = panels.render(0, size=(640, 720))
    assert np.any(np.all(image[360:] == UNKNOWN_GRAY, axis=-1))
    np.testing.assert_array_equal(panels.slots[1]["room_labels"], before)
    assert panels.metadata()[1]["unknown"]


def test_capturing_room_boundaries_makes_an_independent_snapshot():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation
    p, ep = setup_policy("bed")
    obs = observation(ep, 0, depth=3)
    p.plan(obs)
    panels = FloorPanels(2, p.settings, obs.pose)
    panels.capture(p, obs)
    saved = panels.slots[0]["room_labels"].copy()
    assert saved.any() and not np.shares_memory(saved, p.graph.labels)
    p.graph.labels.fill(0)
    np.testing.assert_array_equal(panels.slots[0]["room_labels"], saved)


# -- 2026-10-22: the map journal -- every action's map, as deltas, so a review stands at any step from the log -----
def test_the_map_journal_replays_the_grid_the_settled_mask_and_the_partition_of_any_step():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import (
        JOURNAL_LAYERS, journal_to_arrays, replay_journal)
    panels = panels_with_rooms()
    slot = panels.slots[0]
    shape = panels.shape
    # The fixture pre-draws a floor; in an episode the slot starts all-unknown and the first capture journals the
    # first map as action 0's delta. Replay the fixture's floor the same way.
    grid0, labels0 = slot["grid"].copy(), slot["room_labels"].copy()
    slot["grid"].fill(-1)
    slot["room_labels"].fill(0)
    for layer in JOURNAL_LAYERS:
        slot["journal"][layer] = []
    FloorPanels._journal(slot, "grid", 0, slot["grid"], grid0)
    FloorPanels._journal(slot, "room_labels", 0, slot["room_labels"], labels0)
    # Action 3 maps a strip of floor and a wall; action 7 maps more floor, settles a pocket and renumbers a room.
    grid3 = grid0.copy()
    grid3[300:310, 100:160] = 0
    grid3[310, 100:160] = 100
    FloorPanels._journal(slot, "grid", 3, slot["grid"], grid3)
    FloorPanels._journal(slot, "settled", 3, slot["settled"], np.zeros(shape, bool))
    FloorPanels._journal(slot, "room_labels", 3, slot["room_labels"], slot["room_labels"].copy())
    grid7 = grid3.copy()
    grid7[320:330, 100:120] = 0
    settled7 = np.zeros(shape, bool)
    settled7[340:345, 100:105] = True
    labels7 = slot["room_labels"].copy()
    labels7[labels7 == 2] = 9
    FloorPanels._journal(slot, "grid", 7, slot["grid"], grid7)
    FloorPanels._journal(slot, "settled", 7, slot["settled"], settled7)
    FloorPanels._journal(slot, "room_labels", 7, slot["room_labels"], labels7)
    assert [d[0] for d in slot["journal"]["grid"]] == [0, 3, 7] and [d[0] for d in slot["journal"]["settled"]] == [7]
    assert [d[0] for d in slot["journal"]["room_labels"]] == [0, 7], "an unchanged layer writes nothing"
    assert len(slot["journal"]["grid"][1][1]) == 10 * 60 + 60, "the delta is the changed cells alone"
    arrays = panels.journal_arrays()
    assert set(arrays) >= {"slot_0_grid_steps", "slot_0_grid_offsets", "slot_0_grid_index", "slot_0_grid_values",
                           "slot_1_settled_steps", "origin", "resolution_m", "shape", "floor_ids"}
    assert arrays["slot_0_grid_steps"].tolist() == [0, 3, 7] and arrays["slot_0_grid_offsets"].tolist()[-1] == len(arrays["slot_0_grid_index"])
    for step, expected in ((0, grid0), (2, grid0), (3, grid3), (5, grid3), (7, grid7), (400, grid7)):
        np.testing.assert_array_equal(replay_journal(arrays, 0, "grid", step, shape, -1), expected)
    assert not replay_journal(arrays, 0, "settled", 6, shape, 0).any()
    np.testing.assert_array_equal(replay_journal(arrays, 0, "settled", 7, shape, 0).astype(bool), settled7)
    np.testing.assert_array_equal(replay_journal(arrays, 0, "room_labels", 7, shape, 0), labels7)
    assert (replay_journal(arrays, 0, "room_labels", 6, shape, 0) == 2).any(), "before the renumbering"
    assert journal_to_arrays("x", [], np.int8)["x_offsets"].tolist() == [0]
    # The arrays round-trip through npz as numbers only.
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    buffer.seek(0)
    loaded = dict(np.load(buffer, allow_pickle=False))
    np.testing.assert_array_equal(replay_journal(loaded, 0, "grid", 7, shape, -1), grid7)
