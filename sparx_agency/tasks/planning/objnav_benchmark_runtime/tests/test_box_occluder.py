"""The box pipeline's occluder test: a surface in front of the detected object, read from the depth alone.

Synthetic frames shaped on Collierville/000018 (2026-10-15): the television's box held 79 % television
at 4.1 m and 20 % of a seat back's rail at 1.3 m in its lower middle, continuing under the box; the box
pipeline read the rail. The counter-examples are the furniture the rule must leave alone.
"""
from dataclasses import replace

import numpy as np
import pytest

from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.core.mapping.objects.box_occluder import depth_band_m, occluder_in_front
from sparx_agency.core.planning.object_search.perception.projection import observed_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy

#: The television's box of the recording (113 x 68 px, centred), the television at 4.1 m, the rail at 1.33 m.
BOX = (226, 214, 339, 282)
TV_M, RAIL_M = 4.1, 1.33


def frame(episode, depth):
    return replace(observation(episode, 0), depth_m=depth.astype(np.float32))


def television_behind_a_rail(episode, rail_top_row=244, rail_cols=(257, 307), rail_bottom=None):
    """A wall and television at 4.1 m filling the frame; a rail at 1.33 m from ``rail_top_row`` down to
    ``rail_bottom`` (the frame's bottom by default: it continues under the box), over ``rail_cols``."""
    k = episode.camera.intrinsics
    depth = np.full((k.height, k.width), TV_M)
    bottom = k.height if rail_bottom is None else rail_bottom
    depth[rail_top_row:bottom, rail_cols[0]:rail_cols[1]] = RAIL_M
    return depth


def test_the_rail_in_front_of_the_television_is_set_aside_and_the_television_is_read():
    _, ep = setup_policy()
    obs = frame(ep, television_behind_a_rail(ep))
    occluder = occluder_in_front(obs.depth_m, BOX, RAIL_M, ep.camera.min_depth_m, ep.camera.max_depth_m)
    assert occluder is not None and occluder.dtype == bool and occluder.shape == obs.depth_m.shape
    x1, y1, x2, y2 = BOX
    assert occluder[:y1].sum() == 0 and occluder[y2:].sum() == 0, "only pixels inside the box are returned"
    assert occluder[244:282, 257:307].all() and not occluder[214:244, 226:339].any()
    rows = []
    [(label, xyz)] = list(observed_objects(obs, [DetectionWire("tv", 0.65, BOX)], 0.5, rows))
    row = rows[0]
    assert row["status"] == "projected"
    assert row["occluder_depth_m"] == pytest.approx(RAIL_M, abs=0.02) and row["occluder_px"] == 38 * 50
    assert row["range_near_m"] == pytest.approx(TV_M, abs=0.1) and xyz[0] == pytest.approx(TV_M, abs=0.1)
    assert row["height_range_m"][1] - row["height_range_m"][0] > 0.3, "the television's own extent, not a 7 cm band"


def test_without_the_test_the_rail_would_have_been_the_range():
    """The regression itself, measured: the 30th percentile of the box's centre lands on the rail."""
    from sparx_agency.core.mapping.objects.geometry import robust_bbox_depth
    _, ep = setup_policy()
    depth = television_behind_a_rail(ep).astype(np.float32)
    assert robust_bbox_depth(depth, BOX, min_depth_m=0.5, max_depth_m=5.0) == pytest.approx(RAIL_M, abs=0.02)


def test_a_near_surface_that_reaches_the_top_of_the_box_is_the_object():
    """A chair in front of a wall: the chair fills its box to the top edge; the wall behind it spans the
    box and covers as much of it, and is not read as the chair."""
    _, ep = setup_policy()
    x1, y1, x2, y2 = BOX
    depth = television_behind_a_rail(ep, rail_top_row=y1 + 2, rail_cols=(260, 300))      # a 40 px post from the top down
    assert occluder_in_front(depth, BOX, RAIL_M, 0.5, 5.0) is None
    rows = []
    list(observed_objects(frame(ep, depth), [DetectionWire("chair", 0.9, BOX)], 0.5, rows))
    assert rows[0]["status"] == "projected" and "occluder_depth_m" not in rows[0]
    assert rows[0]["range_near_m"] == pytest.approx(RAIL_M, abs=0.05)


def test_a_far_surface_that_does_not_span_the_box_is_part_of_the_object():
    """A bed from its foot, a couch from the front: the far part (headboard, backrest) spans the width and
    the upper part of the box only; the near part is the object's near edge and its range stands."""
    _, ep = setup_policy()
    k = ep.camera.intrinsics
    x1, y1, x2, y2 = BOX
    depth = np.full((k.height, k.width), 1.0)
    depth[y1:y1 + (y2 - y1) // 2, x1:x2] = 2.5           # the far part: the upper half of the box only
    depth[y1 + (y2 - y1) // 2:, :] = 1.0                 # the near part runs on below the box, like a mattress edge or a seat
    assert occluder_in_front(depth, BOX, 1.0, 0.5, 5.0) is None


def test_two_parts_of_one_object_less_than_the_gap_apart_are_one_object():
    """A toilet's bowl at 1.0 m and its tank at 1.3 m: no second surface in the sense of the test."""
    _, ep = setup_policy()
    depth = television_behind_a_rail(ep)
    depth[depth > 2.0] = 1.3
    assert occluder_in_front(depth, BOX, RAIL_M, 0.5, 5.0) is None


def test_a_near_surface_covering_more_of_the_box_than_the_far_one_is_the_object():
    _, ep = setup_policy()
    depth = television_behind_a_rail(ep, rail_top_row=232, rail_cols=(226, 339))      # the lower three quarters, full width
    assert occluder_in_front(depth, BOX, RAIL_M, 0.5, 5.0) is None


def test_too_little_depth_is_no_verdict_and_a_bad_image_raises():
    _, ep = setup_policy()
    k = ep.camera.intrinsics
    assert occluder_in_front(np.full((k.height, k.width), np.nan), BOX, RAIL_M, 0.5, 5.0) is None
    assert occluder_in_front(np.full((k.height, k.width), TV_M), BOX, TV_M, 0.5, 5.0) is None, "one surface"
    assert occluder_in_front(np.full((k.height, k.width), TV_M), (0, 0, 2, 2), TV_M, 0.5, 5.0) is None, "a degenerate box"
    with pytest.raises(ValueError):
        occluder_in_front(np.zeros(10), BOX, RAIL_M, 0.5, 5.0)


def test_the_band_is_the_pipelines_own():
    assert depth_band_m(1.0) == pytest.approx(0.12) and depth_band_m(4.0) == pytest.approx(0.24)


def test_the_mask_pipeline_is_not_touched():
    """With a segmentation mask the object's pixels are known; the box test never runs."""
    _, ep = setup_policy()
    obs = frame(ep, television_behind_a_rail(ep))
    detection = DetectionWire("tv", 0.65, BOX)
    mask = np.zeros(obs.depth_m.shape, dtype=bool)
    x1, y1, x2, y2 = BOX
    mask[y1:y2, x1:x2] = obs.depth_m[y1:y2, x1:x2] > 2.0
    rows = []
    assert list(observed_objects(obs, [detection], 0.5, rows, masks={detection: mask}))
    assert rows[0]["segmented"] is True and "occluder_depth_m" not in rows[0]
    assert rows[0]["range_near_m"] == pytest.approx(TV_M, abs=0.1)

