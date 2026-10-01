"""The perfect stair detector: a bounding box for a staircase only when the staircase is in the frame."""
from __future__ import annotations

import json
import math

import numpy as np

from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_ground_truth import Connector, GroundTruthStairs
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_sightings import (
    DEPTH_SLACK_M, MIN_VISIBLE_POINTS, GroundTruthStairDetector, StairSighting, StairSightings)

CAMERA = PROTOCOL.camera()


def flight(x0=1.2, x1=3.0, rise=0.9, offset=(0.0, 0.0)):
    """A straight flight along +x from ``x0`` to ``x1``, three treads wide, climbing ``rise``; ENU."""
    xs = np.linspace(x0, x1, 10)
    points = []
    for x in xs:
        z = rise * (x - x0) / (x1 - x0)
        for y in (-0.3, 0.0, 0.3):
            points.append((x + offset[0], y + offset[1], z))
    polyline = ((x0 + offset[0], offset[1], 0.0), (x1 + offset[0], offset[1], rise))
    return Connector(0, polyline[0], polyline[-1], 0.0, 2.7, polyline, 2.0, tuple(points))


def frame(depth, pose=AgentPose(0.0, 0.0, 0.0, 0.0), step=0):
    k = CAMERA.intrinsics
    image = np.full((k.height, k.width), depth, np.float32) if np.isscalar(depth) else depth
    return ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8), image, pose, CAMERA, "toilet", step)


def test_a_staircase_in_front_of_the_camera_is_a_box_on_the_frame():
    detector = GroundTruthStairDetector(GroundTruthStairs([flight()], [0.0, 2.7], True))
    [sighting] = detector.detect(frame(3.5))
    assert sighting.connector_id == 0 and sighting.visible >= MIN_VISIBLE_POINTS
    x1, y1, x2, y2 = sighting.bbox
    k = CAMERA.intrinsics
    assert 0 <= x1 < x2 < k.width and 0 <= y1 < y2 < k.height, "a box inside the image"
    assert x1 < k.cx < x2, "the flight is dead ahead: the box straddles the image centre"
    assert 1.0 < sighting.distance_m < 1.6, "the nearest visible tread is a little over a metre away"
    detection = sighting.as_detection()
    assert detection["label"] == "stairs" and detection["confidence"] == 1.0 and detection["source"] == "ground_truth"
    json.dumps(detection)


def test_a_staircase_behind_a_nearer_wall_is_not_seen():
    detector = GroundTruthStairDetector(GroundTruthStairs([flight()], [0.0, 2.7], True))
    assert detector.detect(frame(0.7)) == [], "the depth image reads a wall 0.7 m away in every pixel"
    assert detector.detect(frame(3.5)), "the same geometry, no wall: seen"


def test_a_partly_hidden_staircase_is_seen_by_the_treads_that_show():
    detector = GroundTruthStairDetector(GroundTruthStairs([flight()], [0.0, 2.7], True))
    k = CAMERA.intrinsics
    depth = np.full((k.height, k.width), 3.5, np.float32)
    depth[:, : k.width // 2] = 0.6                            # a wall covers the left half of the view
    [sighting] = detector.detect(frame(depth))
    full = detector.detect(frame(3.5))[0]
    assert MIN_VISIBLE_POINTS <= sighting.visible < full.visible
    assert sighting.bbox[0] >= k.width // 2 - 8, "the box is the visible half"


def test_a_staircase_behind_the_agent_or_beyond_the_depth_range_is_not_seen():
    detector = GroundTruthStairDetector(GroundTruthStairs([flight()], [0.0, 2.7], True))
    assert detector.detect(frame(3.5, pose=AgentPose(0.0, 0.0, 0.0, math.pi))) == [], "facing away"
    far = GroundTruthStairDetector(GroundTruthStairs([flight(offset=(6.0, 0.0))], [0.0, 2.7], True))
    assert far.detect(frame(4.9)) == [], "7 m away: past the sensor's 5 m"
    beside = GroundTruthStairDetector(GroundTruthStairs([flight(offset=(0.0, 3.0))], [0.0, 2.7], True))
    assert beside.detect(frame(3.5)) == [], "3 m to the left of a 79-degree camera: outside the frame"


def test_the_depth_slack_absorbs_the_navmesh_sitting_a_few_centimetres_above_the_treads():
    detector = GroundTruthStairDetector(GroundTruthStairs([flight()], [0.0, 2.7], True))
    k = CAMERA.intrinsics
    # A depth image that reads exactly the treads, a little nearer than the navmesh sample: still a sighting.
    points = np.asarray(flight().surface, dtype=float)
    from sparx_agency.core.planning.objnav.camera_geometry import project_to_image
    pixels, z = project_to_image(points, CAMERA, AgentPose(0.0, 0.0, 0.0, 0.0))
    depth = np.full((k.height, k.width), 3.5, np.float32)
    for (u, v), d in zip(pixels, z):
        if 0 <= int(round(u)) < k.width and 0 <= int(round(v)) < k.height:
            depth[int(round(v)), int(round(u))] = d - DEPTH_SLACK_M + 0.01
    assert detector.detect(frame(depth))
    for (u, v), d in zip(pixels, z):
        if 0 <= int(round(u)) < k.width and 0 <= int(round(v)) < k.height:
            depth[int(round(v)), int(round(u))] = d - DEPTH_SLACK_M - 0.05
    assert detector.detect(frame(depth)) == [], "nearer than the slack: something stands in front of the treads"


def test_a_connector_without_a_surface_sample_is_detected_by_its_polyline():
    bare = Connector(3, (1.2, 0.0, 0.0), (3.0, 0.0, 0.9), 0.0, 2.7, ((1.2, 0.0, 0.0), (3.0, 0.0, 0.9)), 2.0)
    detector = GroundTruthStairDetector(GroundTruthStairs([bare], [0.0, 2.7], True))
    [sighting] = detector.detect(frame(3.5))
    assert sighting.connector_id == 3 and sighting.visible >= MIN_VISIBLE_POINTS


def test_sightings_remember_every_staircase_seen_and_report_the_first_sighting_once():
    detector = GroundTruthStairDetector(GroundTruthStairs([flight()], [0.0, 2.7], True))
    memory = StairSightings()
    assert not memory.seen(0) and memory.footprint(0) == []
    first = memory.observe(detector.detect(frame(3.5, step=4)))
    assert [s.connector_id for s in first] == [0] and memory.seen(0)
    again = memory.observe(detector.detect(frame(3.5, step=5)))
    assert again == [], "the second frame is not a first sighting"
    assert memory.observe([]) == [] and memory.latest == []
    record = memory.records[0]
    assert record["first_step"] == 4 and record["last_step"] == 5 and record["sightings"] == 2
    footprint = memory.footprint(0)
    assert len(footprint) >= MIN_VISIBLE_POINTS
    assert all(math.dist(a[:2], b[:2]) > 0.05 or a is b for a in footprint for b in footprint if a != b), "deduplicated"
    diagnostics = memory.diagnostics()
    json.dumps(diagnostics)
    assert diagnostics["seen"] == [0] and diagnostics["records"][0]["footprint_points"] == len(footprint)


def test_mark_seen_stands_in_for_a_frame_and_a_sighting_is_json():
    memory = StairSightings()
    assert memory.mark_seen(7, 12, [(1.0, 2.0, 0.0), (1.5, 2.0, 0.3)]) is True
    assert memory.mark_seen(7, 13) is False, "the second marking is not a first sighting"
    assert memory.seen(7) and len(memory.footprint(7)) == 2
    json.dumps(StairSighting(7, 12, (1, 2, 3, 4), 6, 8, 1.5, ((1.0, 2.0, 0.0),)).as_detection())


def test_an_island_split_connector_is_never_offered_as_a_way_between_storeys():
    walkable = flight()
    split = Connector(1, (5.0, 0.0, 0.0), (7.0, 0.0, 2.7), 0.0, 2.7, ((5.0, 0.0, 0.0), (7.0, 0.0, 2.7)), 3.0,
                      traversable=False)
    stairs = GroundTruthStairs([walkable, split], [0.0, 2.7], True)
    assert [c.id for c in stairs.touching(0.0, 0.3)] == [0]
    assert stairs.nearest(6.0, 0.0, 0.0, 0.3, within_m=0.75) is None, "standing on the split flight explains nothing"
    assert stairs.nearest(2.0, 0.0, 0.0, 0.3, within_m=0.75) is walkable
    assert stairs.by_id(1) is split and stairs.by_id(9) is None
    assert stairs.diagnostics()["connectors"][1]["traversable"] is False
    read = GroundTruthStairs.from_metadata({"floor_levels": [{"height_m": 0.0}, {"height_m": 2.7}], "stair_connectors": [
        {"id": 0, "bottom_xyz": [5.0, 0.0, 0.0], "top_xyz": [7.0, 0.0, 2.7], "bottom_z": 0.0, "top_z": 2.7,
         "polyline_xyz": [[5.0, 0.0, 0.0], [7.0, 0.0, 2.7]], "length_m": 3.0, "traversable": False,
         "surface_xyz": [[5.5, 0.0, 0.5]], "bottom_exit_xyz": [4.0, 0.0, 0.0], "top_exit_xyz": None}]})
    assert read.connectors[0].traversable is False and read.connectors[0].surface == ((5.5, 0.0, 0.5),)
    assert read.connectors[0].far_exit(2.7, 0.3) == (4.0, 0.0, 0.0) and read.connectors[0].far_exit(0.0, 0.3) is None


