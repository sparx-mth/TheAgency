"""Evidence-gated, revisable room labels around the existing core classifier.

One object is a clue. A person at a doorway who sees a bed calls the room a
bedroom without stepping in, and is right often enough that the search
should act on it; when the next object says otherwise the label changes
(a sink read as a kitchen becomes a bathroom once a toilet shows). So the
gate is ONE confirmed landmark by default, and every label carries its
STRENGTH: ``weak`` from a single class of evidence or a hesitant model,
``strong`` from two or more distinct classes the model is confident about
-- or from one **signature object** (:data:`room_priors.SIGNATURE_OBJECTS`:
a bed, a toilet, an oven) whose room type the model agrees with, since
2026-10-05: a bed seen through a door IS a bedroom, and the Hanson
recording walked in to scan one for a toilet because one kind of object
was by rule a weak label. The consumers decide what a weak clue may do --
the room-search loop lets one end a room's visit only when the model is
confident in it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from sparx_agency.core.mapping.topology.room_classifier import RoomLabel, RoomTypeClassifier, facts_bucket
from sparx_agency.core.mapping.topology.room_priors import generic_object, signature_type

WEAK = "weak"
STRONG = "strong"


@dataclass(frozen=True)
class RoomLabelSettings:
    """When a room may be labelled, and what makes a label strong.

    Attributes:
        min_objects: Confirmed landmarks in the room before the model is
            asked. One: a single object is a clue worth acting on.
        min_classes: Distinct classes among them. One, for the same reason.
        min_evidence_updates: Scene-graph updates that must have carried the
            room's evidence. One: the landmark itself is already confirmed
            across frames before it counts.
        refresh_steps: Actions after which an unchanged room is re-asked.
        strong_classes: Distinct classes at or above which a label is
            ``strong`` -- with the model's confidence at or above
            ``strong_confidence``.
        strong_confidence: See ``strong_classes``.
        signature_objects: One object of a kind that names its room type on
            its own (:data:`room_priors.SIGNATURE_OBJECTS`) makes the label
            ``strong`` when the model's label is that type and its
            confidence clears ``strong_confidence``. ``False`` restores the
            two-kinds rule alone.
        distinctive_required: The ``strong_classes`` kinds must include at
            least one that is not generic (:data:`room_priors.GENERIC_OBJECTS`:
            a cabinet, a plant, a book ...), since 2026-10-05 -- a cabinet
            and a potted plant made an upstairs room a strong "living_room".
            ``False`` counts every kind.
        generic_evidence_gate: A room whose only evidence is generic objects
            is not classified at all -- it stays ``unknown`` and the model
            is not asked (since 2026-10-05: one cabinet made a "kitchen" at
            0.9 on an upper storey, which the storey summary then showed the
            node oracle as a kitchen found). The objects are still shown to
            the oracle on the room's line. ``False`` asks the model.
        shape_only_rooms: Ask about a room that has NO classifiable objects
            on its geometry alone when it reads as a passage (since
            2026-10-21): long and narrow (``corridor_elongation`` or more)
            or with ``corridor_doors`` or more doorways off it, and at least
            ``shape_min_area_m2`` of floor. A hallway, a foyer, a landing
            have no furniture to name them, and ``unknown`` left the oracle
            to value every corridor as a possible bedroom. One call per
            geometry bucket (:func:`room_classifier.facts_bucket`); the
            label is WEAK and never rules a room out. ``False`` keeps the
            object gate alone.
        corridor_elongation: See ``shape_only_rooms``.
        corridor_doors: See ``shape_only_rooms``.
        shape_min_area_m2: See ``shape_only_rooms``.
    """

    min_objects: int = 1
    min_classes: int = 1
    min_evidence_updates: int = 1
    refresh_steps: int = 50
    strong_classes: int = 2
    strong_confidence: float = 0.65
    signature_objects: bool = True
    distinctive_required: bool = True
    generic_evidence_gate: bool = True
    shape_only_rooms: bool = True
    corridor_elongation: float = 2.5
    corridor_doors: int = 2
    shape_min_area_m2: float = 2.0

    def __post_init__(self):
        for name in ("min_objects", "min_classes", "min_evidence_updates", "refresh_steps", "strong_classes",
                     "corridor_doors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if not 0.0 <= float(self.strong_confidence) <= 1.0:
            raise ValueError("strong_confidence must lie in [0, 1]")
        for name in ("signature_objects", "distinctive_required", "generic_evidence_gate", "shape_only_rooms"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("%s must be a bool" % name)
        for name in ("corridor_elongation", "shape_min_area_m2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("%s must be positive" % name)

    def shape_candidate(self, facts):
        """Whether a room with no classifiable objects is asked about on this geometry."""
        if not self.shape_only_rooms or not facts:
            return False
        area = facts.get("area_m2")
        if area is None or float(area) < self.shape_min_area_m2:
            return False
        elongation = facts.get("elongation")
        doors = facts.get("doors")
        return ((elongation is not None and float(elongation) >= self.corridor_elongation)
                or (doors is not None and int(doors) >= self.corridor_doors))


class RevisableRoomLabels:
    """Reconsider labels on new evidence, periodic refresh and room splits/merges.

    Counts refer to confirmed unique landmarks, not repeated video frames.
    Door geometry is not a room-type feature. Labels stay provisional until
    repeated classification agrees, but can always be revised afterwards.

    Attributes:
        labels: ``{pid: RoomLabel}`` for the live rooms.
        metadata: ``{pid: dict}`` -- the label plus ``strength``,
            ``provisional``, the evidence counts and what triggered it.
        history: Every label change, for the recording.
    """

    def __init__(self, client, settings=None):
        self.settings = settings or RoomLabelSettings()
        s = self.settings
        self.classifier = RoomTypeClassifier(client, min_objects=s.min_objects,
                                             min_classes=s.min_classes, count_sensitive=True)
        self.labels = {}
        self.metadata = {}
        self.history = []
        self._evidence_updates = {}
        self._signatures = {}
        self._query_kinds = {}
        self._kind_cache = {}
        self._last_query = {}
        self._agreement = {}
        self.queries = 0
        self._evidence_steps = {}
        self._objects = {}
        self._facts = {}

    def update(self, objects, step, partition_changed=False, allow_query=True, facts=None):
        """Record every room's evidence and, when allowed, (re-)classify the ones whose evidence changed.

        ``facts`` is ``{pid: {"area_m2", "elongation", "doors"}}``, the
        room's geometry: shown to the model beside the objects, part of the
        cache key, and -- for a room with nothing in it -- the one reason it
        may be asked about at all (``RoomLabelSettings.shape_only_rooms``).
        """
        if partition_changed:
            # Pids survive one side of a split: retaining their old cached
            # label would attach the whole-room verdict to a different region.
            self._signatures.clear()
            self._query_kinds.clear()
            self._evidence_updates.clear()
            self._last_query.clear()
            self._agreement.clear()
            self._evidence_steps.clear()
        live = set(objects)
        self._objects = {pid: list(classes) for pid, classes in objects.items()}
        self._facts = {pid: dict(item) for pid, item in (facts or {}).items() if pid in live}
        self.labels = {pid: label for pid, label in self.labels.items() if pid in live}
        self.metadata = {pid: item for pid, item in self.metadata.items() if pid in live}
        for pid, classes in objects.items():
            self._update_room(pid, classes, step, allow_query)
        return dict(self.labels)

    def pending(self, pid):
        """Has a NEW KIND of object appeared in this room since the model last saw it?

        True for a room with a class of evidence the background refresh has
        recorded but the model has not been asked about -- the clue the loop
        acts on mid-visit. A second chair is not a clue: counts change the
        cached signature and are re-asked at the next loop point, where one
        round covers every room; a kind never seen in the room is asked about
        the action it lands, for that room alone.
        """
        classes = self._objects.get(pid)
        if not classes:
            return False
        kinds = frozenset(name for name, _ in self._signature(classes))
        return kinds != self._query_kinds.get(pid) and self._ready(pid, classes)

    def query_room(self, pid, step):
        """Classify ONE room now if a new kind of object appeared in it since the last query.

        The clue-driven relabel: the background refresh records evidence
        without asking the model (``allow_query=False``); the loop asks for
        the room it stands in or heads to, and only for it, so a new object
        costs one bounded call for the room it matters to rather than a
        round over every room. A set of kinds the model has already judged
        on this floor is not bought again -- the scene graph's re-partitions
        hand the same evidence back under new counts -- so the calls are
        bounded by the distinct kind-sets seen, not by the actions.

        Returns:
            The room's label after the call (or the held one), or None when
            the room has no evidence at all.
        """
        classes = self._objects.get(pid)
        if classes is None:
            return None
        if self.pending(pid):
            self._update_room(pid, classes, step, allow_query=True, clue=True)
        return self.labels.get(pid)

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _signature(classes):
        # Prompt aliases must not count as extra kinds of semantic evidence.
        aliases = {"couch": "sofa", "tv": "television"}
        return tuple(sorted(Counter(aliases.get(name, name) for name in classes).items()))

    def _objects_ready(self, pid, classes):
        s = self.settings
        if s.generic_evidence_gate and classes and all(generic_object(name) for name in classes):
            return False
        return (len(classes) >= s.min_objects and len(set(classes)) >= s.min_classes
                and self._evidence_updates.get(pid, 0) >= s.min_evidence_updates)

    def _ready(self, pid, classes):
        """Whether the room may be classified: on its objects, or -- with none that count -- on its shape."""
        return self._objects_ready(pid, classes) or self._shape_only(pid, classes)

    def _shape_only(self, pid, classes):
        """A room with no classifiable objects whose geometry reads as a passage (``shape_only_rooms``)."""
        if self._objects_ready(pid, classes):
            return False
        s = self.settings
        if classes and not (s.generic_evidence_gate and all(generic_object(name) for name in classes)):
            return False                                   # objects below the thresholds: wait for them
        return (self._evidence_updates.get(pid, 0) >= s.min_evidence_updates
                and s.shape_candidate(self._facts.get(pid)))

    def _update_room(self, pid, classes, step, allow_query=True, clue=False):
        signature = self._signature(classes)
        kinds = frozenset(name for name, _ in signature)
        classes = [name for name, count in signature for _ in range(count)]
        if self._evidence_steps.get(pid) != step:
            self._evidence_updates[pid] = self._evidence_updates.get(pid, 0) + 1
            self._evidence_steps[pid] = step
        s = self.settings
        facts = self._facts.get(pid)
        shape_only = self._shape_only(pid, classes)
        ready = self._objects_ready(pid, classes) or shape_only
        # The geometry bucket is part of the evidence: a room whose shape or doorways changed is re-asked.
        keyed = signature + ((facts_bucket(facts),) if facts else ())
        previous = self.labels.get(pid)
        elapsed = step - self._last_query.get(pid, -s.refresh_steps)
        changed = self._signatures.get(pid) != keyed
        if not ready:
            result = RoomLabel("unknown", 0.0, "insufficient stable confirmed-object evidence")
            reason = "evidence_gate"
            self._agreement[pid] = 0
        elif not allow_query:
            return  # preserve accumulated evidence; classify at a loop point or on the loop's request
        elif changed or elapsed >= s.refresh_steps:
            # Changed evidence the model has already judged (a re-partition handed
            # the same objects back) reuses its verdict -- the label is a function
            # of the evidence, not of the region; a clue re-asks only for a set of
            # kinds never judged; the periodic refresh alone buys a fresh verdict
            # for evidence the model has seen, once per ``refresh_steps``.
            result = None
            if changed:
                result = self.classifier.cached(classes, facts)
                if result is None and clue and not shape_only:
                    result = self._kind_cache.get(kinds)
            if result is None:
                result = self.classifier.classify(classes, refresh=True, facts=facts, shape_only=shape_only)
                self.queries += 1
                self._last_query[pid] = step
                if not shape_only:
                    self._kind_cache[kinds] = result
            self._signatures[pid] = keyed
            self._query_kinds[pid] = kinds
            self._agreement[pid] = (self._agreement.get(pid, 0) + 1
                                    if previous and previous.label == result.label else 1)
            reason = "shape" if shape_only else "evidence_changed" if changed else "periodic_refresh"
        else:
            return
        self.labels[pid] = result
        if previous is None or previous.label != result.label:
            self.history.append({"step": int(step), "room_id": int(pid),
                                 "previous": previous.label if previous else None,
                                 "label": result.label, "trigger": reason,
                                 "objects": dict(signature)})
        kinds = set(classes)
        confident = result.confidence >= s.strong_confidence and result.label != "unknown"
        signed = s.signature_objects and any(signature_type(name) == result.label for name in kinds)
        # Signature objects of two room types in one region (a bed beside an oven and a refrigerator,
        # Darden/000000 of 2026-10-21): the map merged two rooms or the detector erred, and whichever type the
        # model picked is a guess -- WEAK, so the type prior never rules the region out on it.
        conflicting = len({signature_type(name) for name in kinds if signature_type(name)}) > 1
        distinctive = not s.distinctive_required or any(not generic_object(name) for name in kinds)
        agreeing = len(kinds) >= s.strong_classes and distinctive
        strength = STRONG if confident and (agreeing or signed) and not shape_only and not conflicting else WEAK
        self.metadata[pid] = dict(asdict(result), strength=strength, signature=bool(confident and signed and not conflicting),
                                  provisional=(result.label == "unknown" or result.confidence < s.strong_confidence
                                               or self._agreement.get(pid, 0) < 2 or shape_only or conflicting),
                                  conflicting_signatures=bool(conflicting),
                                  evidence_objects=len(classes), evidence_classes=len(set(classes)),
                                  shape_only=bool(shape_only), updated_step=int(step), trigger=reason)


