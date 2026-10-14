"""Minimal map geometry, shared adaptive scale and display-only persistence."""
from types import SimpleNamespace

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import FloorPanels, UNKNOWN_GRAY


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


def test_objects_probabilities_frontiers_and_visit_order_do_not_clutter_the_map():
    panels = panels_with_rooms()
    plain = panels.render(0, size=(640, 720))
    panels.slots[0]["objects"] = [{"xy": [i / 10.0, 0.0], "class": "chair", "id": i} for i in range(100)]
    search = {"rooms": [{"id": 0, "centroid": [-5.0, 0.0], "label": "very long room name", "prob": 0.99}],
              "order": [0], "accessible_frontiers": [[float(i), 0.0] for i in range(20)]}
    np.testing.assert_array_equal(plain, panels.render(0, size=(640, 720), search=search))


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
