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
from sparx_agency.core.planning.object_search.perception.doors import DoorSettings, ObservedDoors, doorway_position
from sparx_agency.core.mapping.topology.revisable_room_labels import RevisableRoomLabels, RoomLabelSettings
from sparx_agency.core.planning.object_search.rooms.scene_graph import ObservedSceneGraph
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM


class ScriptedClassifier:
    def __init__(self, bedroom_for=()):
        self.calls = 0
        self.bedroom_for = tuple(bedroom_for)

    def chat_json(self, system, user):
        self.calls += 1
        if any(name in user for name in self.bedroom_for):
            return {"label": "bedroom", "confidence": 0.9, "reasoning": "scripted bedroom"}
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


def test_no_object_kind_counts_for_more_than_another_in_a_labels_strength():
    """One kind of object, whichever it is, gives a weak label even when the model is confident; two kinds, whichever, a strong one."""
    for kind in ("bed", "cabinet", "sofa", "toilet"):
        tracker = RevisableRoomLabels(ScriptedClassifier(bedroom_for=(kind,)))
        tracker.update({1: [kind]}, 0)
        assert tracker.metadata[1]["strength"] == "weak", kind
    tracker = RevisableRoomLabels(ScriptedClassifier())
    tracker.update({1: ["cabinet", "potted plant"]}, 0)
    assert tracker.metadata[1]["strength"] == "strong"


def test_every_object_kind_is_evidence_so_a_cabinet_alone_is_classified():
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client)
    labels = tracker.update({1: ["cabinet"], 2: ["cabinet", "potted plant", "book"]}, 0)
    assert labels[1].label != "unknown" and labels[2].label != "unknown" and client.calls == 2
    assert not tracker.pending(1)


def test_the_removed_label_strength_settings_are_rejected():
    for knob in ("signature_objects", "distinctive_required", "generic_evidence_gate"):
        with pytest.raises(TypeError):
            RoomLabelSettings(**{knob: False})


def test_an_empty_passage_is_classified_on_its_shape_alone_weakly_and_once_per_geometry():
    """A hallway has no furniture to name it (2026-10-21): a long narrow room with doorways and nothing in it is
    asked about on its geometry, the label is WEAK and provisional (it never rules a room out), the geometry
    bucket is part of the cache key, and a square empty room with one door is not asked at all."""
    class Shaped:
        def __init__(self):
            self.calls = []

        def chat_json(self, system, user):
            self.calls.append(user)
            if "long and narrow" in user and "- sofa" not in user:
                return {"label": "corridor", "confidence": 0.75, "reasoning": "a long empty passage with doors"}
            return {"label": "living_room", "confidence": 0.8, "reasoning": "objects"}

    client = Shaped()
    tracker = RevisableRoomLabels(client)
    corridor = {"area_m2": 7.0, "elongation": 4.0, "doors": 3}
    square = {"area_m2": 12.0, "elongation": 1.1, "doors": 1}
    labels = tracker.update({1: [], 2: [], 3: ["cabinet"]}, 0, facts={1: corridor, 2: square, 3: corridor})
    assert labels[1].label == "hallway" and labels[2].label == "unknown", "the passage is named, the square room waits"
    assert labels[3].label == "hallway", "a cabinet is classified on its objects with the geometry beside them"
    assert len(client.calls) == 2 and "doorways opening off it: 3" in client.calls[0]
    assert tracker.metadata[1]["strength"] == "weak" and tracker.metadata[1]["provisional"] and tracker.metadata[1]["shape_only"]
    assert tracker.metadata[1]["trigger"] == "shape"
    assert tracker.update({1: []}, 1, facts={1: corridor})[1].label == "hallway" and len(client.calls) == 2, "cached"
    # A sofa appears: the objects take over, and the geometry still travels with them.
    labels = tracker.update({1: ["sofa"]}, 2, facts={1: corridor})
    assert labels[1].label == "living_room" and len(client.calls) == 3 and "Room geometry:" in client.calls[-1]
    assert not tracker.metadata[1]["shape_only"]
    # Off by setting: an empty room is never asked about.
    quiet = RevisableRoomLabels(Shaped(), RoomLabelSettings(shape_only_rooms=False))
    assert quiet.update({1: []}, 0, facts={1: corridor})[1].label == "unknown"
    assert RoomLabelSettings().shape_candidate({"area_m2": 1.0, "elongation": 5.0, "doors": 3}) is False, "a sliver is no room"
    with pytest.raises(ValueError):
        RoomLabelSettings(corridor_elongation=0.0)


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
