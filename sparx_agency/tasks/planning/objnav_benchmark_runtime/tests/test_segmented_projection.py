"""The masked depth read: an occluder inside the target's box no longer ends the episode.

The failure this guards against (2026-10-12): the agent STOPped on a sofa with
a chair standing in front of it. The box pipeline reads the target's range as
a low percentile of the depth inside the box, and the chair's pixels were
inside the box and nearer, so the sofa was "reached" a metre early. With the
``segmentation`` setting on, the target's boxes go to a box-prompted Segment
Anything service and the read is taken on the object's own pixels.

Fake services throughout: the detector returns the same box every frame, the
segmenter paints the box's interior minus the occluder, exactly what MobileSAM
did on the synthetic frame recorded in ``core/mapping/segmentation/README.md``.
"""
from __future__ import annotations

import numpy as np
import pytest
import requests

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_fallback import SEGMENTER
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import observed_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeDetector, FakeLLM, setup_policy

#: The fixture detector's box (``test_method.FakeDetector``) and the occluder standing in its centre.
BOX = (280, 200, 360, 280)
OCCLUDER = (300, 215, 340, 280)          # rows 215-280, cols 300-340: the whole shrunken box and more
TARGET_M, OCCLUDER_M = 3.0, 0.7


def scene(episode, step, occluder=True):
    """The chair at 3 m filling the frame, and (by default) something at 0.7 m in front of its box."""
    k = episode.camera.intrinsics
    depth = np.full((k.height, k.width), TARGET_M, np.float32)
    if occluder:
        x1, y1, x2, y2 = OCCLUDER
        depth[y1:y2, x1:x2] = OCCLUDER_M
    return ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8), depth,
                             AgentPose(0, 0, 0, 0), episode.camera, "chair", step)


def object_mask(shape, box=BOX, occluder=OCCLUDER):
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    x1, y1, x2, y2 = occluder
    mask[y1:y2, x1:x2] = False
    return mask


class FakeSegmenter:
    """Paints each prompted box minus the occluder; records every call."""

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure
        self.last_inference_ms = 290.0

    def health(self):
        return {"model": "mobile_sam.pt"}

    def segment(self, rgb, boxes):
        self.calls.append([tuple(b) for b in boxes])
        if self.failure is not None:
            raise self.failure
        return [object_mask(rgb.shape[:2], box=tuple(int(v) for v in box)) for box in boxes]


def transport_failure():
    error = ObjNavInternalError("Segmenter service failed: connection refused")
    error.__cause__ = requests.ConnectionError("refused")
    return error


# ── the projection ───────────────────────────────────────────────────────────

def test_the_box_reads_the_occluder_and_the_mask_reads_the_object():
    _, ep = setup_policy()
    obs = scene(ep, 0)
    detection = DetectionWire("chair", 0.9, BOX)
    rows = []
    assert list(observed_objects(obs, [detection], 0.5, rows))                     # the box pipeline projects ...
    assert rows[0]["status"] == "projected" and rows[0]["range_near_m"] == pytest.approx(OCCLUDER_M, abs=0.05)
    assert "segmented" not in rows[0]
    rows = []
    masks = {detection: object_mask(obs.depth_m.shape)}
    assert list(observed_objects(obs, [detection], 0.5, rows, masks=masks))       # ... and so does the mask pipeline,
    row = rows[0]
    assert row["status"] == "projected" and row["segmented"] is True
    assert row["range_near_m"] == pytest.approx(TARGET_M, abs=0.1)                # on the object's own range
    assert row["xyz"][0] == pytest.approx(TARGET_M, abs=0.1)
    assert row["mask_pixels"] == 80 * 80 - 40 * 65 and 0.5 < row["mask_fraction"] < 0.7
    assert row["radius_m"] == pytest.approx(0.5 * 80 * TARGET_M / ep.camera.intrinsics.fx, abs=0.01)


def test_a_target_box_due_a_mask_it_did_not_get_is_not_read_the_old_way():
    _, ep = setup_policy()
    obs = scene(ep, 0)
    detection = DetectionWire("chair", 0.9, BOX)
    rows = []
    assert list(observed_objects(obs, [detection], 0.5, rows, masks={detection: None})) == []
    assert rows[0]["status"] == "segmentation_unavailable" and rows[0]["segmented"] is False
    # A box of another class is not in the mapping: the box pipeline, as always.
    table = DetectionWire("table", 0.9, BOX)
    rows = []
    assert list(observed_objects(obs, [table], 0.5, rows, masks={detection: None}))
    assert rows[0]["status"] == "projected" and "segmented" not in rows[0]


def test_an_empty_mask_is_no_measurement_and_a_wrong_shape_raises():
    _, ep = setup_policy()
    obs = scene(ep, 0)
    detection = DetectionWire("chair", 0.9, BOX)
    rows = []
    assert list(observed_objects(obs, [detection], 0.5, rows, masks={detection: np.zeros(obs.depth_m.shape, bool)})) == []
    assert rows[0]["status"] == "empty_mask" and rows[0]["mask_pixels"] == 0
    with pytest.raises(ValueError, match="mask shape"):
        list(observed_objects(obs, [detection], 0.5, [], masks={detection: np.ones((4, 4), bool)}))


def test_a_deep_object_seen_whole_passes_on_the_mask_path_where_the_coherence_gate_would_refuse_it():
    """A bed from its foot: the mask spans 2.0-4.0 m, far more than the +/-6 % band; the box's central quarter
    is the gate's business, the mask's near-surface component carries the read."""
    _, ep = setup_policy()
    k = ep.camera.intrinsics
    depth = np.full((k.height, k.width), 5.0, np.float32)
    x1, y1, x2, y2 = 100, 150, 540, 470
    depth[y1:y2, x1:x2] = np.linspace(4.0, 2.0, y2 - y1, dtype=np.float32)[:, None]      # nearer toward the bottom
    obs = ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8), depth, AgentPose(0, 0, 0, 0), ep.camera, "bed", 0)
    detection = DetectionWire("bed", 0.9, (x1, y1, x2, y2))
    mask = np.zeros(depth.shape, bool)
    mask[y1:y2, x1:x2] = True
    rows = []
    assert list(observed_objects(obs, [detection], 0.5, rows, masks={detection: mask}))
    assert rows[0]["status"] == "projected" and rows[0]["supported_fraction"] < 0.35
    assert 2.0 <= rows[0]["range_near_m"] <= 2.8


# ── the policy ───────────────────────────────────────────────────────────────

def test_without_segmentation_the_closing_stops_on_the_occluder_and_with_it_the_approach_goes_on():
    """The regression itself: two frames of the chair's box at depth 0.7 (the occluder's) STOP the
    box pipeline inside the terminal range; the mask pipeline reads 3 m, locks the same two frames and
    keeps closing. The context check is off on both sides: at 3 m the synthetic chair's measured top
    is a doubtful height for its class, which is that check's business, not this test's."""
    closing = {"context_check": False}
    p, ep = setup_policy(target_closing=closing)
    p.plan(scene(ep, 0))
    assert p.plan(scene(ep, 1)).stop and p.closing.locked
    assert p.closing.observed_near_m == pytest.approx(OCCLUDER_M, abs=0.05)
    assert p.settings.segmentation is False and p.perception.masks is None

    seg = FakeSegmenter()
    q, ep = setup_policy(segmentation=True, segmenter=seg, target_closing=closing)
    first = q.plan(scene(ep, 0))
    assert not first.stop and q.closing.active
    second = q.plan(scene(ep, 1))
    assert not second.stop and q.closing.locked
    assert q.closing.observed_near_m == pytest.approx(TARGET_M, abs=0.1)
    assert second.waypoints and second.waypoints[-1][0] > 0.3                  # closing in on the object, not stopping
    assert q.perception.counts["segmented_frames"] == 2 and q.perception.counts["segmented_boxes"] == 2
    assert seg.calls == [[BOX], [BOX]]
    assert q.perception.diagnostics()["segmenter_evidence"]["boxes"] == 1
    assert q.configuration()["segmentation"]["enabled"] is True
    assert q.configuration()["perception_fusion"].endswith("/target-masked")


def test_only_the_targets_boxes_are_prompted_and_only_once_per_frame():
    seg = FakeSegmenter()
    p, ep = setup_policy(segmentation=True, segmenter=seg)
    chair = DetectionWire("chair", 0.9, BOX)
    table = DetectionWire("table", 0.8, (20, 300, 200, 470))
    weak = DetectionWire("chair", 0.2, (400, 300, 480, 400))          # under both floors: not a prompt
    p.detector.detect = lambda rgb: [table, chair, weak]
    obs = scene(ep, 0)
    p.perception.fuse(obs)
    p.perception.fuse(obs)
    assert seg.calls == [[BOX]]
    # A re-sighting box between the closing's track_confidence and detection_confidence is still the
    # closing's to read, so it is prompted too: no target box reaches the box pipeline by the threshold gap.
    q, ep2 = setup_policy(segmentation=True, segmenter=FakeSegmenter(), detection_confidence=0.5,
                          target_closing={"track_confidence": 0.3})
    q.detector.detect = lambda rgb: [DetectionWire("chair", 0.4, BOX)]
    q.perception.observe(scene(ep2, 0))
    assert q.segmenter.calls == [[BOX]]
    rows = {(row["label"], row["confidence"]): row for row in p.perception.projections}
    assert rows[("chair", 0.9)]["segmented"] is True and rows[("chair", 0.9)]["status"] == "fused"
    assert "segmented" not in rows[("table", 0.8)] and rows[("table", 0.8)]["status"] == "fused"
    assert rows[("chair", 0.2)]["status"] == "low_confidence" and "segmented" not in rows[("chair", 0.2)]


def test_a_segmenter_transport_failure_arms_the_backoff_and_leaves_the_target_unread():
    seg = FakeSegmenter(failure=transport_failure())
    p, ep = setup_policy(segmentation=True, segmenter=seg)
    assert not p.plan(scene(ep, 0)).stop
    assert not p.closing.active                                 # no projection, no takeover
    rows = {row["label"]: row for row in p.perception.projections}
    assert rows["chair"]["status"] == "segmentation_unavailable"
    assert p.perception.counts["segmenter_failures"] == 1
    assert p.fallback.service_unavailable(SEGMENTER, 1) and p.fallback.stats[SEGMENTER + "_failures"] == 1
    assert "failed" in p.perception.diagnostics()["segmenter_evidence"]
    assert not p.plan(scene(ep, 1)).stop
    assert p.perception.counts["segmenter_skipped"] == 1 and len(seg.calls) == 1


def test_segmenter_identity_drift_ends_the_run():
    seg = FakeSegmenter(failure=ObjNavInternalError("Segmenter was reconfigured during inference"))
    p, ep = setup_policy(segmentation=True, segmenter=seg)
    with pytest.raises(ObjNavInternalError, match="reconfigured"):
        p.plan(scene(ep, 0))


def test_the_flag_is_strict_and_needs_a_segmenter():
    with pytest.raises(ValueError, match="segmentation must be boolean"):
        RPTSettings(segmentation="yes")
    with pytest.raises(ValueError, match="no segmenter"):
        RPTSearchPolicy(FakeDetector(), FakeLLM(), RPTSettings(segmentation=True))
    assert RPTSettings().segmentation is False
    off = RPTSearchPolicy(FakeDetector(), FakeLLM(), RPTSettings())
    assert off.configuration()["segmentation"] == {"enabled": False, "depth": "robust_bbox_depth over the shrunken box"}


def test_the_recording_tints_the_targets_mask_and_draws_nothing_for_a_box_without_one():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import method_snapshot, render_dashboard
    seg = FakeSegmenter()
    p, ep = setup_policy(segmentation=True, segmenter=seg, target_closing={"context_check": False})
    obs = scene(ep, 0)
    command = p.plan(obs)
    frame = render_dashboard(p, obs, [(0, 0)], {"action": "MOVE_FORWARD", "info": command.info}, ep.episode_id, method_snapshot(p))
    assert frame.shape == (900, 1600, 3)
    camera = frame[40:520, :640]                                 # the camera panel, 640 x 480 as the frame
    x1, y1, x2, y2 = OCCLUDER
    bx1, by1, bx2, by2 = BOX
    on_object = camera[by1 + 4, bx1 + 4]                         # inside the box, outside the occluder: magenta tint
    on_occluder = camera[(y1 + y2) // 2, (x1 + x2) // 2]        # the occluder: no tint
    assert on_object[0] > 80 and on_object[2] > 80 and on_object[1] < 40, on_object
    assert tuple(on_occluder) == (0, 0, 0), on_occluder
    p.perception.masks = {DetectionWire("chair", 0.9, BOX): None}
    frame = render_dashboard(p, obs, [(0, 0)], {"action": "MOVE_FORWARD", "info": command.info}, ep.episode_id, method_snapshot(p))
    assert tuple(frame[40 + by1 + 4, bx1 + 4]) == (0, 0, 0)
