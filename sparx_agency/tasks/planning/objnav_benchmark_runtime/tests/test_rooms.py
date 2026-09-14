"""Door and revisable-room regressions on synthetic observed geometry."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_classifier import RoomTypeClassifier
from sparx_agency.core.mapping.topology.room_watershed import WatershedRoomParams
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.labels import fake_label_mapper
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.fixtures import camera as synthetic_camera
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DoorSettings, ObservedDoors, doorway_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RevisableRoomLabels, RoomLabelSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.scene_graph import ObservedSceneGraph
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM


class ScriptedClassifier:
    def __init__(self):
        self.calls = 0

    def chat_json(self, system, user):
        self.calls += 1
        return {"label": "living_room" if self.calls == 1 else "bedroom",
                "confidence": 0.8, "reasoning": "current observed objects"}


def test_room_labels_wait_then_revise_counts_and_refresh():
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client)
    assert tracker.update({1: ["chair"]}, 0)[1].label == "unknown"
    assert client.calls == 0
    assert tracker.update({1: ["chair", "chair", "bed"]}, 10)[1].label == "living_room"
    assert tracker.metadata[1]["provisional"]
    # Same kinds, different counts: the original frozenset cache missed this.
    assert tracker.update({1: ["chair", "bed", "bed"]}, 20)[1].label == "bedroom"
    assert client.calls == 2
    tracker.update({1: ["chair", "bed", "bed"]}, 30)
    assert client.calls == 2
    tracker.update({1: ["chair", "bed", "bed"]}, 70)
    assert client.calls == 3 and not tracker.metadata[1]["provisional"]
    assert any(row["previous"] == "living_room" and row["label"] == "bedroom" for row in tracker.history)
    assert tracker.update({1: ["chair", "bed", "bed"]}, 80, partition_changed=True)[1].label == "unknown"


def test_core_classifier_preserves_default_cache_but_allows_explicit_refresh():
    client = ScriptedClassifier()
    classifier = RoomTypeClassifier(client)
    classifier.classify(["chair"])
    classifier.classify(["chair", "chair"])
    assert client.calls == 1
    assert classifier.classify(["chair"], refresh=True).label == "bedroom"


def observation(step, y=0.0):
    camera = synthetic_camera()
    depth = np.full((480, 640), 4.5, np.float32)
    depth[80:440, 240:270] = 2.0
    depth[80:440, 370:400] = 2.0
    depth[80:140, 240:400] = 2.0
    return ObjNavObservation(np.zeros((480, 640, 3), np.uint8), depth,
                             AgentPose(0, y, 0, 0), camera, "chair", step)


def test_open_door_uses_frame_not_far_wall_depth():
    detection = DetectionWire("doorway", 0.8, (240, 80, 400, 440))
    result = doorway_position(observation(0), detection, DoorSettings())
    assert result is not None
    xy, width = result
    assert xy[0] == pytest.approx(2.0)  # open centre's 4.5 m is the far wall
    assert 0.7 < width < 1.0


def test_door_confirmation_requires_distinct_frames_and_viewpoints():
    tracker = ObservedDoors()
    door = DetectionWire("doorway", 0.8, (240, 80, 400, 440))
    alias = replace(door, cls="door frame")
    tracker.update(observation(0), [door, alias])
    assert len(tracker.landmarks) == 1 and not tracker.confirmed()
    tracker.update(observation(1), [door, alias])
    tracker.update(observation(2), [door, alias])
    assert not tracker.confirmed()  # three identical viewpoints are not enough
    tracker.update(observation(3, y=0.25), [door, alias])
    assert len(tracker.confirmed()) == 1 and tracker.revision == 1
    assert tracker.confirmed()[0].count == 4


def test_missing_frame_depth_or_wide_box_cannot_create_a_door():
    obs = observation(0)
    obs.depth_m[:] = np.nan
    assert doorway_position(obs, DetectionWire("door", 0.9, (240, 80, 400, 440)), DoorSettings()) is None
    assert doorway_position(observation(0), DetectionWire("door", 0.9, (0, 0, 640, 480)), DoorSettings()) is None


def test_confirmed_door_splits_rooms_and_survives_merging():
    cells = np.full((100, 160), 100, np.int8)
    cells[20:80, 10:75] = 0
    cells[20:80, 85:150] = 0
    cells[44:56, 75:85] = 0
    world = OccupancyGrid2D(cells, OccupancyGrid2DParams(0.1, 0, 0),
                            values=OccupancyValues(free=0, occupied=100, unknown=-1))
    # Aggressive merging proves that the door, not just tuning, preserves the split.
    settings = WatershedRoomParams(min_room_separation_m=1, min_clearance_m=0.3,
                                    min_room_cells=40, door_cut_m=0.75, merge_dynamics_m=10)
    graph = ObservedSceneGraph(FakeLLM(), segmentation=settings)
    target = fake_label_mapper().target_labels("chair")
    graph.update(world, [], target)
    assert len(graph.registry.rooms) == 1
    door = SimpleNamespace(id=0, xy=(8.05, 5.05), count=3)
    graph.update(world, [], target, doors=[door], step=10)
    assert len(graph.registry.rooms) == 2
    assert len(graph.doors) == 1 and graph.doors[0]["room_pairs"]
    assert graph.partition_revision == 1


def test_low_score_door_needs_geometry_and_multiple_views_not_just_confidence():
    tracker = ObservedDoors()
    door = DetectionWire("open doorway", 0.06, (240, 80, 400, 440))
    tracker.update(observation(0), [door])
    assert not tracker.confirmed()
    tracker.update(observation(1, y=0.25), [door])
    tracker.update(observation(2, y=0.25), [door])
    assert len(tracker.confirmed()) == 1


def test_strong_conflicting_object_prevents_a_false_door_boundary():
    tracker = ObservedDoors()
    door = DetectionWire("door", 0.06, (240, 80, 400, 440))
    cabinet = DetectionWire("cabinet", 0.9, door.xyxy)
    for step in range(4):
        tracker.update(observation(step, y=0.1 * step), [door, cabinet])
    assert not tracker.confirmed() and tracker.depth_accepted == 0


