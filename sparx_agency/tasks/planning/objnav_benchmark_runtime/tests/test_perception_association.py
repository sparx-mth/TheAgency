"""Landmark association in the ObjectNav wiring: classes never merge, same-class instances are told apart by size.

The first frame of the Allensville couch run (``1e3ec4018b4a``) held a cabinet
with two vases and a cup on it; only the cabinet reached the map. The vases
projected with valid depth and were dropped as ``same_frame_association`` --
the cabinet's 0.70 m centroid radius claimed them -- and the cup fell under the
0.35 confidence floor. The toilet run (``b536f3c4fe05``) voted a toilet and the
bathtub 0.5 m beside it into one landmark, which flipped to ``bathtub`` and had
the toilet's lock released as "contradicted by the map". Since 2026-10-07 the
runtime's map takes no cross-class association at all (``class_votes=False``):
a cup and the table it stands on are two instances in memory whatever their
proximity or footprint overlap, and two observations of one class are one
instance only within 0.35 m or at a footprint-disc IoU of 0.25.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import clipped_box
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception_cycle import contradicted_by_map
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy


def frame(episode, step, boxes, depth=1.0):
    """An observation whose detector returns ``boxes`` over a flat depth of ``depth`` metres."""
    k = episode.camera.intrinsics
    return ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8),
                             np.full((k.height, k.width), depth, np.float32),
                             AgentPose(0, 0, 0, 0), episode.camera, "chair", step)


def fused(policy):
    return [(row["label"], row["status"], row.get("landmark_id")) for row in policy.perception.projections
            if row["label"] not in ("door frame",)]


def test_a_vase_on_a_cabinet_is_its_own_landmark_not_a_same_frame_duplicate_of_the_cabinet():
    """Allensville couch run, frame 0: ``cabinet 0.65`` fused, ``vase 0.48`` and ``vase 0.40`` dropped as
    ``same_frame_association``. The small disc sits inside the big one (IoU 0.1) within 0.70 m of its
    centroid and 0.23 m above it -- inside the height tolerance. Two objects, two landmarks."""
    p, ep = setup_policy()
    cabinet = DetectionWire("cabinet", .65, (0, 317, 369, 479))         # 0.48 m footprint at 1 m
    vase = DetectionWire("vase", .48, (244, 269, 286, 332))              # a 0.15 m footprint (the clamp)
    p.detector.detect = lambda rgb: [cabinet, vase]
    p.perception.fuse(frame(ep, 0, None))
    rows = {row["label"]: row for row in p.perception.projections}
    assert rows["cabinet"]["status"] == "fused" and rows["vase"]["status"] == "fused", rows
    assert rows["cabinet"]["landmark_id"] != rows["vase"]["landmark_id"]
    assert sorted(lm.class_name for lm in p.landmarks.all_landmarks()) == ["cabinet", "vase"]
    assert p.perception.counts.get("same_frame_association", 0) == 0
    # The second frame confirms both; the room sees a vase, not only a cabinet.
    p.perception.fuse(frame(ep, 1, None))
    assert sorted(lm.class_name for lm in p.landmarks.confirmed()) == ["cabinet", "vase"]


def test_a_second_box_of_the_same_object_in_one_frame_is_still_a_same_frame_duplicate_but_another_class_is_not():
    """Two ``cabinet`` boxes of one cabinet feed its landmark once (the rule was never about them); a
    ``desk`` box over the same pixels is another class -- its own instance, never merged, since 2026-10-07."""
    p, ep = setup_policy()
    # Two cabinet boxes with a box IoU under the per-frame alias filter's 0.5, landing 0.33 m apart on
    # one footprint: the second is the same object's second box.
    p.detector.detect = lambda rgb: [DetectionWire("cabinet", .65, (0, 317, 369, 479)),
                                     DetectionWire("cabinet", .45, (200, 330, 420, 479))]
    p.perception.fuse(frame(ep, 0, None))
    assert [row["status"] for row in p.perception.projections] == ["fused", "same_frame_association"]
    assert len(p.landmarks) == 1
    q, ep2 = setup_policy()
    q.detector.detect = lambda rgb: [DetectionWire("cabinet", .65, (0, 317, 369, 479)),
                                     DetectionWire("desk", .45, (20, 330, 340, 479))]
    q.perception.fuse(frame(ep2, 0, None))
    statuses = {row["label"]: row["status"] for row in q.perception.projections}
    assert statuses == {"cabinet": "fused", "desk": "fused"}
    assert sorted(lm.class_name for lm in q.landmarks.all_landmarks()) == ["cabinet", "desk"]


def test_the_runtime_map_never_merges_classes_and_tells_same_class_instances_apart_by_size():
    """The wiring (``floor_context.new``): no class voting, a 0.35 m centroid floor, footprint IoU 0.25."""
    p, ep = setup_policy()
    assert not p.landmarks.class_votes
    assert p.settings.landmark_dedupe_radius_m == pytest.approx(0.35) and p.settings.landmark_footprint_iou == pytest.approx(0.25)
    # A cup on a table: the same spot, two instances in memory.
    table = p.landmarks.observe("dining table", (2.0, 0.0), frame_id=0, radius_m=1.0)
    cup = p.landmarks.observe("cup", (2.0, 0.0), frame_id=0, radius_m=0.15)
    assert cup is not table and len(p.landmarks) == 2
    # A sofa box on a confirmed bed is a sofa landmark, and the map contradicts nothing.
    for frame in range(1, 6):
        bed = p.landmarks.observe("bed", (5.0, 0.0), frame_id=frame, radius_m=0.9)
    sofa = p.landmarks.observe("sofa", (5.1, 0.0), frame_id=6, radius_m=0.9)
    assert sofa is not bed and bed.class_name == "bed" and bed.votes == {} and sofa.count == 1
    p._object_geometry[bed.id] = (5.0, 0.0, 0.5)
    assert contradicted_by_map(p, "sofa", (5.1, 0.0, 0.5), 0.9) is None
    assert p.landmarks.confirmed() == [bed] or sofa not in p.landmarks.confirmed()
    # Two dining chairs 0.6 m apart are two chairs (the 0.70 m radius merged them); the same chair re-seen is one.
    left = p.landmarks.observe("chair", (8.0, 0.0), frame_id=0, radius_m=0.25)
    assert p.landmarks.observe("chair", (8.6, 0.0), frame_id=1, radius_m=0.25) is not left
    assert p.landmarks.observe("chair", (8.2, 0.1), frame_id=2, radius_m=0.25) is left
    for bad in ({"landmark_dedupe_radius_m": 0.0}, {"landmark_footprint_iou": 0.0}, {"landmark_footprint_iou": 1.5}):
        with pytest.raises(ValueError):
            RPTSettings(**bad)


def test_a_toilet_beside_a_bathtub_does_not_vote_the_toilet_into_a_bathtub():
    """The library's VOTING mode (no longer the runtime's, but kept for its other users): Allensville toilet
    run, actions 0-8, the toilet landmark (4 votes) took 5 ``bathtub`` votes from the fixture 0.35-0.55 m
    beside it, flipped, and the lock was released as contradicted by the map. Across classes the footprints
    must agree now; the runtime goes further and never merges classes at all (the test above)."""
    lmap = ObjectLandmarkMap(nearest_match=True, class_votes=True)
    for step in range(4):
        toilet = lmap.observe("toilet", (-0.29, -3.62), frame_id=step, radius_m=0.22)
    for step in range(4, 9):
        other = lmap.observe("bathtub", (-0.42, -3.10), frame_id=step, radius_m=0.20)
    assert other is not toilet and len(lmap) == 2
    assert toilet.class_name == "toilet" and toilet.votes == {"toilet": 4}
    assert other.class_name == "bathtub" and lmap.relabels == []
    # The same two observations, same class: the centroid rule still merges them (one object seen twice).
    same = ObjectLandmarkMap(nearest_match=True, class_votes=True)
    first = same.observe("toilet", (-0.29, -3.62), frame_id=0, radius_m=0.22)
    assert same.observe("toilet", (-0.42, -3.10), frame_id=1, radius_m=0.20) is first
    # Comparable footprints of two classes on one spot are the vote the model was written for.
    vote = ObjectLandmarkMap(nearest_match=True, class_votes=True)
    bed = vote.observe("bed", (0.0, 0.0), frame_id=0, radius_m=1.0)
    assert vote.observe("sofa", (0.3, 0.0), frame_id=1, radius_m=0.9) is bed and bed.votes == {"bed": 1, "sofa": 1}
    # An unmeasured footprint keeps the centroid rule across classes, as before.
    plain = ObjectLandmarkMap(nearest_match=True, class_votes=True)
    a = plain.observe("bed", (0.0, 0.0), frame_id=0)
    assert plain.observe("sofa", (0.3, 0.0), frame_id=1, radius_m=0.9) is a
    # The historical rule, by the flag.
    old = ObjectLandmarkMap(nearest_match=True, class_votes=True, cross_class_footprint_only=False)
    t = old.observe("toilet", (-0.29, -3.62), frame_id=0, radius_m=0.22)
    assert old.observe("bathtub", (-0.42, -3.10), frame_id=1, radius_m=0.20) is t


def test_a_box_cut_only_by_the_bottom_edge_is_the_floor_cutoff_not_a_sliver():
    """The Allensville toilet at 1.2-1.7 m: 101-218 px wide, 160-190 px tall, y2 = 477-480. Refused
    twenty-three times as clipped, never two frames in a row."""
    _, ep = setup_policy()
    k = ep.camera.intrinsics
    assert not clipped_box((187, 290, 300, 477), k, 8), "the toilet at 1.7 m"
    assert not clipped_box((252, 320, 470, 479), k, 8), "the toilet at 1.2 m"
    assert clipped_box((0, 304, 69, 480), k, 8), "cut by the LEFT edge too: a lateral sliver"
    assert clipped_box((457, 338, 640, 479), k, 8), "cut by the right edge too"
    assert clipped_box((300, 400, 360, 480), k, 8), "60 px wide: a sliver whatever edge it touches"
    assert clipped_box((300, 100, 400, 480), k, 8) is False or (400 - 300) < 0.3 * k.width, "tall and wide enough: not narrow"
    assert clipped_box((300, 200, 400, 480), k, 8), "its top is in the upper half: not the floor cutoff"
    assert clipped_box((0, 269, 84, 479), k, 8), "the Ranchester bed sliver, as before"


def test_the_detection_floor_is_the_takeover_tracking_threshold():
    """``cup 0.34`` on the Allensville table was a ``low_confidence`` row at the 0.35 floor; a weak box
    costs nothing on its own (two votes confirm a landmark), so the floor is 0.30."""
    p, ep = setup_policy()
    assert p.settings.detection_confidence == pytest.approx(0.30) == p.settings.target_closing.track_confidence
    p.detector.detect = lambda rgb: [DetectionWire("cup", .34, (200, 200, 260, 260)),
                                     DetectionWire("bottle", .28, (400, 200, 440, 260))]
    p.perception.fuse(frame(ep, 0, None))
    statuses = {row["label"]: row["status"] for row in p.perception.projections}
    assert statuses == {"cup": "fused", "bottle": "low_confidence"}
    assert p.landmarks.confirmed() == [], "one weak sighting confirms nothing"

