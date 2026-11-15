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
            return {"label": "bedroom", "confidence": 0.9, "reasoning": "a bed was seen"}
        return {"label": "living_room" if self.calls == 1 else "bedroom",
                "confidence": 0.8, "reasoning": "current observed objects"}


def test_room_labels_wait_then_revise_counts_and_refresh():
    """The historical three-object gate, as an explicit setting -- and, since 2026-11-09, a verdict that differs
    without a new KIND of evidence (the same kinds under new counts, the periodic refresh) is held until it repeats:
    the scripted model answers living_room once and bedroom ever after, which is the churn of a 3B labeller."""
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client, RoomLabelSettings(min_objects=3, min_classes=2, min_evidence_updates=2))
    assert tracker.update({1: ["chair"]}, 0)[1].label == "unknown"
    assert client.calls == 0
    assert tracker.update({1: ["chair", "chair", "bed"]}, 10)[1].label == "living_room"
    assert tracker.metadata[1]["provisional"]
    # Same kinds, different counts: re-asked (the frozenset cache would have missed it), the new verdict HELD.
    assert tracker.update({1: ["chair", "bed", "bed"]}, 20)[1].label == "living_room"
    assert client.calls == 2 and tracker.held == 1 and tracker.metadata[1]["trigger"] == "held"
    assert tracker.history[-1]["held"] is True and tracker.history[-1]["label"] == "bedroom"
    tracker.update({1: ["chair", "bed", "bed"]}, 30)
    assert client.calls == 2
    # The periodic refresh repeats the verdict: adopted, and the label is no longer provisional.
    assert tracker.update({1: ["chair", "bed", "bed"]}, 70)[1].label == "bedroom"
    assert client.calls == 3 and not tracker.metadata[1]["provisional"]
    assert any(row["previous"] == "living_room" and row["label"] == "bedroom" and not row.get("held")
               for row in tracker.history)
    assert tracker.update({1: ["chair", "bed", "bed"]}, 80, partition_changed=True)[1].label == "unknown"


def test_a_new_kind_of_evidence_relabels_at_once_and_the_hysteresis_can_be_switched_off():
    """A wardrobe beside the bed is new evidence: the label follows it on that update; with ``relabel_hysteresis``
    off, a counts-only re-ask relabels at once as before 2026-11-09."""
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client)
    assert tracker.update({1: ["bed"]}, 0)[1].label == "living_room"
    assert tracker.update({1: ["bed", "wardrobe"]}, 5)[1].label == "bedroom" and tracker.held == 0
    client = ScriptedClassifier()
    legacy = RevisableRoomLabels(client, RoomLabelSettings(relabel_hysteresis=False))
    assert legacy.update({1: ["bed"]}, 0)[1].label == "living_room"
    assert legacy.update({1: ["bed", "bed"]}, 5)[1].label == "bedroom" and legacy.held == 0
    with pytest.raises(ValueError):
        RoomLabelSettings(relabel_hysteresis=1)


def test_a_kind_lost_to_the_partition_holds_the_label_for_a_refresh_and_a_kind_added_relabels_at_once():
    """EU6Fwq7SyZv/000083 re-run (2026-11-09): a 10 m2 dining room (cabinet, chair, counter, sink, table) read
    bathroom for the 40 actions the partition credited its table to the neighbour, then dining room again. A room's
    objects do not leave it: a verdict on a strict subset of the kinds its label was adopted on is held -- through
    repeats, through the partition changes the watershed makes on most actions -- for ``refresh_steps`` actions;
    a room that stays smaller that long was split for real and takes the verdict. A kind added relabels at once."""
    client = ScriptedClassifier()
    settings = RoomLabelSettings()
    tracker = RevisableRoomLabels(client, settings)
    assert tracker.update({1: ["chair", "lamp"]}, 0)[1].label == "living_room"
    # The lamp stands across the moved boundary (a partition change, as the watershed reports one): held.
    assert tracker.update({1: ["chair"]}, 10, partition_changed=True)[1].label == "living_room"
    assert tracker.held == 1 and tracker.history[-1]["held"] and tracker.history[-1]["lost"] == ["lamp"]
    assert tracker.update({1: ["chair", "chair"]}, 20, partition_changed=True)[1].label == "living_room", \
        "a repeat on lost kinds within the refresh is no adoption"
    assert tracker.held == 1, "one held entry per verdict, not per re-ask"
    # The lamp is credited back: the kinds the label was adopted on, the label stands, nothing pending.
    assert tracker.update({1: ["chair", "lamp"]}, 30, partition_changed=True)[1].label == "living_room"
    assert not tracker._pending
    # Lost again and staying lost past a refresh: the room was split for real, the verdict is taken.
    assert tracker.update({1: ["chair"]}, 40, partition_changed=True)[1].label == "living_room" and tracker.held == 2
    assert tracker.update({1: ["chair"]}, 40 + settings.refresh_steps)[1].label == "bedroom"
    assert tracker.history[-1]["trigger"] == "periodic_refresh" and not tracker.history[-1].get("held")
    # A kind the label never saw is new evidence: adopted at once.
    tracker = RevisableRoomLabels(ScriptedClassifier(), settings)
    assert tracker.update({1: ["chair", "lamp"]}, 0)[1].label == "living_room"
    assert tracker.update({1: ["chair", "lamp", "bed"]}, 10)[1].label == "bedroom"
    assert tracker.history[-1]["trigger"] == "evidence_changed" and not tracker.history[-1].get("held")


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


def test_a_strong_label_is_two_kinds_of_object_and_a_confident_model_whichever_kinds():
    """Until 2026-11-15 two hand-written tables shaped this: a "generic" object (a cabinet, a plant) did not count
    toward a strong label, and one "signature" object (a bed, a toilet, an oven) made a label strong on its own
    (``room_priors.py``). Both went: strength counts the evidence, and what the objects say is the classifier's.
    Nothing withholds a room on a strong label any more, so a wrong strong label costs what the oracle makes of it."""
    tracker = RevisableRoomLabels(ScriptedClassifier())
    tracker.update({1: ["cabinet", "potted plant"]}, 0)
    assert tracker.metadata[1]["strength"] == "strong", "two kinds, the model confident"
    tracker.update({1: ["cabinet", "potted plant", "sofa"]}, 5)
    assert tracker.metadata[1]["strength"] == "strong"
    signed = RevisableRoomLabels(ScriptedClassifier(bedroom_for=("bed",)))
    signed.update({1: ["bed"]}, 0)
    assert signed.metadata[1]["label"] == "bedroom" and signed.metadata[1]["strength"] == "weak", "one kind is a clue"
    assert "signature" not in signed.metadata[1] and "conflicting_signatures" not in signed.metadata[1]
    signed.update({2: ["bed", "bed", "oven", "refrigerator", "cabinet"]}, 0)       # Darden/000000's merged region
    assert signed.metadata[2]["label"] == "bedroom" and signed.metadata[2]["strength"] == "strong"
    for removed in ("signature_objects", "distinctive_required", "generic_evidence_gate"):
        with pytest.raises(TypeError):
            RoomLabelSettings(**{removed: True})


def test_a_room_of_any_objects_is_classified_by_the_model_not_gated_by_a_table():
    """Ranchester couch, 2026-10-05, action 34: one cabinet made a 'kitchen' at 0.9 on the upper storey. A table
    of "generic" objects then kept such rooms ``unknown`` without a call; it went with ``room_priors.py`` on
    2026-11-15. Every object is evidence the model is asked about; a label resting on one kind is WEAK and is
    shown to the oracle as ``type=...?``."""
    client = ScriptedClassifier()
    tracker = RevisableRoomLabels(client)
    labels = tracker.update({1: ["cabinet"], 2: ["cabinet", "potted plant", "book"]}, 0)
    assert labels[1].label != "unknown" and labels[2].label != "unknown"
    assert client.calls == 2, "one call per distinct evidence"
    assert tracker.metadata[1]["strength"] == "weak" and tracker.metadata[1]["trigger"] != "evidence_gate"
    assert tracker.metadata[2]["strength"] == "strong"


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
    tracker = RevisableRoomLabels(client, RoomLabelSettings(shape_rule=False))      # the model's verdict, as before 2026-11-10
    corridor = {"area_m2": 7.0, "elongation": 4.0, "doors": 3}
    square = {"area_m2": 12.0, "elongation": 1.1, "doors": 1}
    labels = tracker.update({1: [], 2: [], 3: ["cabinet"]}, 0, facts={1: corridor, 2: square, 3: corridor})
    assert labels[1].label == "hallway" and labels[2].label == "unknown", "the passage is named, the square room waits"
    assert labels[3].label == "hallway", "generic furniture alone does not stop the shape question"
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


def test_an_empty_passage_is_a_hallway_by_rule_without_a_call_and_an_object_relabels_it_at_once():
    """Since 2026-11-10 (``shape_rule``): the room with nothing in it is not a question for the model -- the
    calls on zero objects were 64 % hallway and otherwise noise, and two runs answered one prompt differently."""
    class Counting:
        def __init__(self):
            self.calls = []

        def chat_json(self, system, user):
            self.calls.append(user)
            return {"label": "living_room", "confidence": 0.8, "reasoning": "objects"}

    client = Counting()
    tracker = RevisableRoomLabels(client, RoomLabelSettings(shape_rule=True))   # off by default since 2026-11-11
    corridor = {"area_m2": 7.0, "elongation": 4.0, "doors": 3}
    square = {"area_m2": 12.0, "elongation": 1.1, "doors": 1}
    labels = tracker.update({1: [], 2: []}, 0, facts={1: corridor, 2: square})
    assert labels[1].label == "hallway" and labels[2].label == "unknown" and client.calls == []
    assert tracker.shape_ruled == 1 and tracker.queries == 0
    assert tracker.metadata[1]["strength"] == "weak" and tracker.metadata[1]["shape_only"] and tracker.metadata[1]["trigger"] == "shape"
    assert labels[1].confidence == RoomLabelSettings().shape_rule_confidence < RoomLabelSettings().strong_confidence
    assert tracker.update({1: ["sofa"]}, 2, facts={1: corridor})[1].label == "living_room" and len(client.calls) == 1, \
        "an object in the room is the model's question again, at once"
    by_default = RevisableRoomLabels(Counting())
    assert not by_default.settings.shape_rule and by_default.update({1: []}, 0, facts={1: corridor})[1].label == "living_room", \
        "the rule is off by default (2026-11-11): the empty passage is the model's question, as before"
    with pytest.raises(ValueError, match="shape_rule_confidence"):
        RoomLabelSettings(shape_rule_confidence=0.9)


def test_a_dining_room_needs_chairs_at_its_table():
    """Since 2026-11-10 (``dining_min_chairs``): under the benchmark's label table every desk, side table and coffee
    table is a ``table``, and the 3B model made dining rooms of them; the verdict stands only on three chairs or more."""
    class Dining:
        def __init__(self):
            self.calls = 0

        def chat_json(self, system, user):
            self.calls += 1
            return {"label": "dining_room", "confidence": 0.85, "reasoning": "a table"}

    client = Dining()
    tracker = RevisableRoomLabels(client, RoomLabelSettings(dining_min_chairs=3))   # off by default since 2026-11-11
    assert tracker.update({1: ["table"]}, 0)[1].label == "unknown" and tracker.dining_downgrades == 1
    assert "3 chairs" in tracker.labels[1].reasoning and tracker.labels[1].confidence == 0.85
    assert tracker.update({2: ["table", "chair"]}, 1)[2].label == "unknown", "a desk and its chair"
    assert tracker.update({3: ["table", "chair", "chair", "chair", "chair"]}, 2)[3].label == "dining_room"
    assert tracker.update({4: ["table", "armchair", "chair", "chair"]}, 3)[4].label == "dining_room", "an alias counts as a chair"
    assert tracker.dining_downgrades == 2 and client.calls == 4
    legacy = RevisableRoomLabels(Dining())
    assert legacy.settings.dining_min_chairs == 0 and legacy.update({1: ["table"]}, 0)[1].label == "dining_room", \
        "the rule is off by default (2026-11-11): the model's word"
    with pytest.raises(ValueError, match="dining_min_chairs"):
        RoomLabelSettings(dining_min_chairs=-1)


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
    tracker.update({1: ["bed", "bed", "bed"]}, 3)                # a loop point: count-sensitive, re-asked ...
    assert client.calls == 2 and tracker.labels[1].label == "living_room", "... and the differing verdict held"
    assert tracker.held == 1
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
