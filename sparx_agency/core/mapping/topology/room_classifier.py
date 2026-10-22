"""Room-type classification from observed object classes, via an LLM.

Defaults cache by the class set for existing callers. Count-sensitive cache
keys, minimum class diversity and explicit refresh are available for online
room revisiting. Transport/parse errors propagate without caching a failure.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.mapping.topology.llm_client import LLMClient

#: The residential taxonomy (since 2026-10-21, the user's layout list: hallway / corridor, foyer /
#: entryway, dining room, office / study, laundry / utility, balcony / patio, closet / pantry /
#: storage, garage) ahead of the clinical labels the classifier's other users keep.
HOME_LABEL_SET = [
    "kitchen", "bedroom", "bathroom", "living_room", "dining_room",
    "office", "hallway", "foyer", "storage_closet", "laundry_room",
    "balcony", "garage",
]
DEFAULT_LABEL_SET = HOME_LABEL_SET + [
    "lobby", "waiting_area", "patient_room", "exam_room",
    "reception", "unknown",
]

#: What a model writes in prose for a label of the set: "corridor" is the hallway, "study" the
#: office, "patio" the balcony. Applied after lower-casing and underscoring the reply.
LABEL_ALIASES = {
    "corridor": "hallway", "hall": "hallway", "passage": "hallway", "landing": "hallway",
    "entryway": "foyer", "entry": "foyer", "entrance": "foyer", "entrance_hall": "foyer", "vestibule": "foyer",
    "mudroom": "foyer", "entry_hall": "foyer",
    "study": "office", "home_office": "office", "library": "office",
    "utility_room": "laundry_room", "utility": "laundry_room", "laundry": "laundry_room", "boiler_room": "laundry_room",
    "closet": "storage_closet", "pantry": "storage_closet", "storage": "storage_closet", "storage_room": "storage_closet",
    "walk_in_closet": "storage_closet", "wardrobe": "storage_closet", "cupboard": "storage_closet",
    "patio": "balcony", "terrace": "balcony", "porch": "balcony", "deck": "balcony", "veranda": "balcony",
    "lounge": "living_room", "family_room": "living_room", "sitting_room": "living_room", "den": "living_room",
    "living": "living_room", "livingroom": "living_room",
    "dining": "dining_room", "dining_area": "dining_room", "diningroom": "dining_room",
    "toilet": "bathroom", "restroom": "bathroom", "wc": "bathroom", "washroom": "bathroom", "powder_room": "bathroom",
    "ensuite": "bathroom", "en_suite": "bathroom", "shower_room": "bathroom",
    "kitchenette": "kitchen", "master_bedroom": "bedroom", "guest_room": "bedroom", "kids_room": "bedroom",
    "nursery": "bedroom", "bed_room": "bedroom",
}

SYSTEM_PROMPT_TEMPLATE = """You are a scene-understanding assistant that
classifies the rooms of a HOME from the objects observed inside them and,
when given, the room's shape.

You will be given a list of objects observed in one room, and perhaps its
floor area, its shape and how many doorways open off it. Based ONLY on that
and common sense about residential layouts, output one room label.
Observations are incomplete and may contain false detections: a single
detection of one kind of object is a clue, not a proof, and a detector
confuses a dining table or a sofa with a bed, a cabinet with a refrigerator.
Prefer distinctive room-function evidence over generic furniture. A home has
rooms with few objects of their own: a long narrow space with several doorways
and nothing in it is a hallway or corridor; a small space at the entrance is a
foyer; a table with chairs and no kitchen fittings is a dining room; a desk is
an office or study; a washing machine or dryer is a laundry or utility room;
shelves or a tiny windowless space is a closet, pantry or storage; an outdoor
floor with garden furniture is a balcony or patio. Use unknown when evidence is
insufficient or contradictory. Do not invent unobserved objects. Reassess the
current evidence rather than assuming an earlier room label was correct.

Choose the label from this set (do not invent new ones):
{label_set}

Reply with a JSON object of the form:
{{"label": "<one label from the set>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<one short sentence>"}}"""

USER_PROMPT_TEMPLATE = """Room observed objects:
{obj_list}
{facts}
Classify this room."""


def format_room_facts(facts: Optional[Dict[str, object]]) -> str:
    """The room's geometry for the prompt: area, shape and doorways, each only when known."""
    if not facts:
        return ""
    lines = []
    area = facts.get("area_m2")
    if area is not None and math.isfinite(float(area)):
        lines.append("- floor area about %d m2" % max(1, int(round(float(area)))))
    elongation = facts.get("elongation")
    if elongation is not None and math.isfinite(float(elongation)):
        ratio = float(elongation)
        shape = ("long and narrow, %d:1" % int(round(ratio)) if ratio >= 2.5
                 else "elongated, about 2:1" if ratio >= 1.7 else "roughly square")
        lines.append("- shape: %s" % shape)
    doors = facts.get("doors")
    if doors is not None:
        n = int(doors)
        lines.append("- doorways opening off it: %d" % n)
    return "Room geometry:\n" + "\n".join(lines) + "\n" if lines else ""


def facts_bucket(facts: Optional[Dict[str, object]]) -> Tuple:
    """A coarse key of the geometry, for the cache: small / medium / large, corridor or not, 0 / 1 / 2+ doorways."""
    if not facts:
        return ()
    area = facts.get("area_m2")
    size = None if area is None or not math.isfinite(float(area)) else ("S" if float(area) < 6 else "M" if float(area) < 15 else "L")
    elongation = facts.get("elongation")
    corridor = None if elongation is None or not math.isfinite(float(elongation)) else float(elongation) >= 2.5
    doors = facts.get("doors")
    exits = None if doors is None else min(2, int(doors))
    return (size, corridor, exits)


def format_object_list(classes: Sequence[str]) -> str:
    """Collapse duplicate class names into sorted '- name xN' count lines."""
    if not classes:
        return "(no objects observed yet)"
    counts = Counter(classes)
    return "\n".join("- %s x%d" % (name, n) for name, n in sorted(counts.items()))


@dataclass(frozen=True)
class RoomLabel:
    """One room-type verdict: label, self-reported confidence and short rationale."""

    label: str
    confidence: float
    reasoning: str


class RoomTypeClassifier:
    """Objects -> LLM -> room type, with optional evidence-aware cache keys.

    Args:
        client: Existing LLMClient or equivalent chat_json interface.
        label_set: Accepted room labels; unrecognized replies become unknown.
        min_objects: Minimum observed objects before querying the model.
        min_classes: Minimum distinct observed classes before querying.
        count_sensitive: Include class multiplicities in the cache signature.
            False by default, preserving the historical class-set cache.
    """

    def __init__(self, client: LLMClient,
                 label_set: Sequence[str] = DEFAULT_LABEL_SET,
                 min_objects: int = 1, min_classes: int = 1,
                 count_sensitive: bool = False):
        self._client = client
        self._label_set = [str(s) for s in label_set]
        self._min_objects = int(min_objects)
        self._min_classes = int(min_classes)
        self._count_sensitive = bool(count_sensitive)
        if self._min_objects < 1 or self._min_classes < 1:
            raise ValueError("Room evidence thresholds must be positive")
        self._sig_cache: Dict[Tuple, RoomLabel] = {}

    @property
    def label_set(self) -> List[str]:
        """Candidate labels offered to the model (a copy)."""
        return list(self._label_set)

    @property
    def cache_size(self) -> int:
        """Number of distinct successfully classified evidence signatures."""
        return len(self._sig_cache)

    def _signature(self, norm: List[str], facts: Optional[Dict[str, object]] = None) -> Tuple:
        """The cache key: the objects -- and, for a room with none, the geometry bucket.

        A room with objects is keyed by them alone: its geometry is shown to
        the model but does not buy a second verdict when the room grows
        from small to medium or gains a doorway (the first run with the
        bucket in every key asked the model 80-100 times per long episode,
        2026-10-22). An empty room has nothing else to be keyed by.
        """
        objects = (tuple(sorted(Counter(norm).items())) if self._count_sensitive
                   else tuple(sorted(set(norm))))
        if objects:
            return objects
        bucket = facts_bucket(facts)
        return objects if not bucket else objects + (bucket,)

    def cached(self, classes: Sequence[str], facts: Optional[Dict[str, object]] = None) -> Optional[RoomLabel]:
        """The verdict already held for exactly this evidence (the geometry bucket for an empty room), or None.

        For a caller that must decide whether asking is worth a call: the
        room-search loop re-labels a room the moment a new kind of object
        appears in it, and the scene graph's re-partitions hand it the same
        evidence again and again -- a verdict the model has given for that
        evidence is reused rather than bought twice.
        """
        norm = [str(c).strip().lower() for c in classes if str(c).strip()]
        return self._sig_cache.get(self._signature(norm, facts))

    def classify(self, classes: Sequence[str], refresh: bool = False,
                 facts: Optional[Dict[str, object]] = None, shape_only: bool = False) -> RoomLabel:
        """Classify current evidence; refresh bypasses a previous cached answer.

        ``facts`` is the room's geometry (``area_m2``, ``elongation``,
        ``doors``), shown to the model and -- for a room without objects --
        bucketed coarsely into the cache key (:func:`facts_bucket`).
        ``shape_only`` asks about a room with
        fewer objects than the evidence thresholds on its geometry alone --
        a hallway has no furniture to name it (since 2026-10-21); the caller
        bounds those calls. Failure leaves an existing cache entry untouched.
        Caller controls the refresh cadence; this ROS-free core reads no
        clock or episode state.
        """
        norm = [str(c).strip().lower() for c in classes if str(c).strip()]
        if (len(norm) < self._min_objects or len(set(norm)) < self._min_classes) and not (shape_only and facts):
            return RoomLabel("unknown", 0.0, "no objects observed yet" if not norm
                             else "insufficient observed object evidence")
        signature = self._signature(norm, facts)
        cached = self._sig_cache.get(signature)
        if cached is not None and not refresh:
            return cached
        result = self._classify(norm, facts)
        self._sig_cache[signature] = result
        return result

    def _classify(self, classes: List[str], facts: Optional[Dict[str, object]] = None) -> RoomLabel:
        system = SYSTEM_PROMPT_TEMPLATE.format(label_set=", ".join(self._label_set))
        user = USER_PROMPT_TEMPLATE.format(obj_list=format_object_list(classes), facts=format_room_facts(facts))
        reply = self._client.chat_json(system, user)
        # "living room" / "Living-Room" are the set's ``living_room``, "corridor" its ``hallway``: a model
        # that writes the type in prose or by a synonym must not read as ``unknown``.
        label = "_".join(str(reply.get("label", "unknown")).strip().lower().replace("-", " ").replace("/", " ").split())
        label = LABEL_ALIASES.get(label, label)
        if label not in self._label_set:
            label = "unknown"
        try:
            confidence = float(reply.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0
        return RoomLabel(label, confidence, str(reply.get("reasoning", ""))[:200])
