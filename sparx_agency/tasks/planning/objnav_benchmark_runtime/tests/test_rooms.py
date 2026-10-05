"""Door and revisable-room regressions, independent of any Gibson scene labels."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_classifier import RoomTypeClassifier
from sparx_agency.core.mapping.topology.room_watershed import WatershedRoomParams
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DoorSettings, ObservedDoors, doorway_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RevisableRoomLabels, RoomLabelSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.scene_graph import ObservedSceneGraph
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM


class ScriptedClassifier:
    def __init__(self, bedroom_for=()):
        self.calls = 0
        self.bedroom_for = tuple(bedroom_for)

    def chat_json(self, system, user):
        self.calls += 1
        if any(name in user for name in self.bedroom_for):
            return {"label": "bedroom", "confidence": 0.9, "reasoning": "a signature object"}
        return {"label": "living_room" if self.calls == 1 else "bedroom",
                "confidence": 0.8, "reasoning": "current observed objects"}


def test_room_labels_wait_then_revise_counts_and_refresh():
    """The historical three-object gate, as an explicit setting."""
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client, RoomLabelSettings(min_objects=3, min_classes=2, min_evidence_updates=2))
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


def test_one_confirmed_object_is_a_weak_clue_and_two_classes_a_strong_one():
    """A bed seen from the doorway names the room; a second kind of object makes the name strong."""
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client)
    labels = tracker.update({1: ["bed"]}, 0)
    assert labels[1].label == "living_room" and client.calls == 1, "one object: the model is asked at once"
    assert tracker.metadata[1]["strength"] == "weak" and tracker.metadata[1]["provisional"]
    labels = tracker.update({1: ["bed", "wardrobe"]}, 5)
    assert labels[1].label == "bedroom" and client.calls == 2, "the label is revised on the new evidence"
    assert tracker.metadata[1]["strength"] == "strong", "two classes, confident model"
    assert tracker.history[-1]["previous"] == "living_room" and tracker.history[-1]["label"] == "bedroom"


def test_a_strong_label_needs_a_distinctive_kind_and_one_signature_object_is_enough():
    """Ranchester couch, 2026-10-05: a cabinet and a potted plant made an upstairs room a strong
    'living_room' at 0.95 and put a living room on the storey summary the node oracle reads."""
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client)
    tracker.update({1: ["cabinet", "potted plant"]}, 0)
    assert tracker.metadata[1]["strength"] == "weak", "two generic kinds describe no room"
    tracker.update({1: ["cabinet", "potted plant", "sofa"]}, 5)
    assert tracker.metadata[1]["strength"] == "strong", "a sofa is distinctive"
    counted = RevisableRoomLabels(ScriptedClassifier(), RoomLabelSettings(distinctive_required=False))
    counted.update({1: ["cabinet", "potted plant"]}, 0)
    assert counted.metadata[1]["strength"] == "strong", "the former rule, one knob away"
    # One signature object (a bed) with the model agreeing is strong on its own; a cabinet is not.
    signed = RevisableRoomLabels(ScriptedClassifier(bedroom_for=("bed",)))
    signed.update({1: ["bed"]}, 0)
    assert signed.metadata[1]["strength"] == "strong" and signed.metadata[1]["signature"]
    signed.update({2: ["cabinet"]}, 0)
    assert signed.metadata[2]["strength"] == "weak" and not signed.metadata[2]["signature"]
    with pytest.raises(ValueError):
        RoomLabelSettings(distinctive_required="yes")


def test_pending_evidence_is_classified_on_request_for_one_room_only():
    """The background refresh records evidence without asking; the loop asks for the room it is in."""
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client)
    tracker.update({1: ["bed"], 2: ["sink"]}, 0, allow_query=False)
    assert client.calls == 0 and tracker.pending(1) and tracker.pending(2)
    assert tracker.labels == {}
    label = tracker.query_room(1, 3)
    assert label is not None and label.label == "living_room" and client.calls == 1
    assert not tracker.pending(1) and tracker.pending(2), "room 2 was not asked about"
    assert tracker.query_room(1, 4).label == "living_room" and client.calls == 1, "unchanged evidence: no call"
    assert tracker.query_room(7, 4) is None, "a room with no evidence has no label"
    tracker.update({1: ["bed", "lamp"], 2: ["sink"]}, 6, allow_query=False)
    assert tracker.pending(1), "new evidence in room 1 is pending again"


def test_a_clue_is_a_new_kind_of_object_and_a_judged_kind_set_is_not_bought_twice():
    """Counts and re-partitions must not turn the mid-visit relabel into a call per action."""
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client)
    tracker.update({1: ["bed"]}, 0, allow_query=False)
    tracker.query_room(1, 0)
    assert client.calls == 1
    tracker.update({1: ["bed", "bed", "bed"]}, 1, allow_query=False)
    assert not tracker.pending(1), "a second and third bed are not a clue"
    tracker.update({1: ["bed", "bed", "bed"]}, 2, allow_query=False, partition_changed=True)
    assert tracker.pending(1), "a re-partition forgets what the room was told"
    tracker.query_room(1, 2)
    assert client.calls == 1, "... but the verdict for {bed} is reused, not bought again"
    assert tracker.labels[1].label == "living_room"
    tracker.update({1: ["bed", "bed", "bed"]}, 3)                # a loop point: count-sensitive, re-asked
    assert client.calls == 2 and tracker.labels[1].label == "bedroom"
    tracker.update({1: ["bed", "bed", "bed"]}, 4, partition_changed=True)
    assert client.calls == 2, "the classifier's own cache answers the identical signature at a loop point too"


@pytest.mark.parametrize("kwargs", [{"min_objects": 0}, {"strong_classes": 0}, {"strong_confidence": 1.5}])
def test_room_label_settings_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        RoomLabelSettings(**kwargs)


def test_core_classifier_preserves_default_cache_but_allows_explicit_refresh():
    client = ScriptedClassifier()
    classifier = RoomTypeClassifier(client)
    classifier.classify(["chair"])
    classifier.classify(["chair", "chair"])
    assert client.calls == 1
    assert classifier.classify(["chair"], refresh=True).label == "bedroom"


def observation(step, y=0.0):
    camera = PROTOCOL.camera()
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
    target = gibson_label_mapper().target_labels("chair")
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




def test_furniture_standing_on_occupied_cells_belongs_to_the_room_around_it():
    """Hanson 2026-10-04: the bed and the desks of the bedroom were ``room: null`` -- an object occupies
    its own cells, which the watershed never labels -- and the room was named by its one chair."""
    cells = np.full((100, 160), 100, np.int8)
    cells[20:80, 10:75] = 0
    cells[20:80, 85:150] = 0
    cells[44:56, 75:85] = 0
    cells[30:52, 25:45] = 100                                 # the bed: a 2 m x 2.2 m block inside room A
    world = OccupancyGrid2D(cells, OccupancyGrid2DParams(0.1, 0, 0),
                            values=OccupancyValues(free=0, occupied=100, unknown=-1))
    settings = WatershedRoomParams(min_room_separation_m=1, min_clearance_m=0.3,
                                    min_room_cells=40, door_cut_m=0.75, merge_dynamics_m=10)
    graph = ObservedSceneGraph(FakeLLM(), segmentation=settings)
    target = gibson_label_mapper().target_labels("chair")
    door = SimpleNamespace(id=0, xy=(8.05, 5.05), count=3)
    bed = SimpleNamespace(id=0, class_name="bed", xy=(3.5, 4.1), count=5, radius_m=1.0)        # the block's centre
    chair = SimpleNamespace(id=1, class_name="chair", xy=(12.0, 5.0), count=3, radius_m=0.25)  # on room B's floor
    far = SimpleNamespace(id=2, class_name="vase", xy=(15.9, 9.9), count=3, radius_m=0.1)       # in the outer wall, 1 m from any floor
    graph.update(world, [bed, chair, far], target, doors=[door], step=10, reason=False)
    assert len(graph.registry.rooms) == 2
    a = graph.room_at(world, (2.0, 7.0))
    b = graph.room_at(world, (12.0, 5.0))
    assert a is not None and b is not None and a != b
    assert graph.room_at(world, bed.xy) is None, "the bed's own cells are not room floor"
    assert graph.object_room(world, bed) == a, "... but the bed stands in room A"
    assert graph.objects_in(a) == ["bed"] and graph.objects_in(b) == ["chair"]
    assert graph.object_room(world, far) is None, "an object a metre from any floor is nobody's"
    assert graph.room_near(world, bed.xy, 1.6) == a and graph.room_near(world, (12.0, 5.0), 0.1) == b
