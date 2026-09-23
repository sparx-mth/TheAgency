"""Room-type classification from observed object classes, via an LLM.

The prompt for a room is a function of its OBSERVED OBJECT CLASSES only.
:class:`RoomTypeClassifier` caches the LLM answer by ``frozenset`` of
class names — as long as a room's object *set* doesn't change (counts
may), no new LLM call is made. A room with fewer than ``min_objects``
observed objects is labelled ``"unknown"`` without calling the LLM.

ROS-free port of the LLM plane of the SJTU ``room_classifier_node.py``;
the ROS wiring (topics, markers, tick loop) stays in the task layer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence

from sparx_agency.core.mapping.topology.llm_client import LLMClient

# Default candidate labels (15). Override per deployment if the world
# contains room types this set doesn't name.
DEFAULT_LABEL_SET = [
    "kitchen", "bedroom", "bathroom", "living_room", "dining_room",
    "office", "hallway", "storage_closet", "laundry_room",
    "lobby", "waiting_area", "patient_room", "exam_room",
    "reception", "unknown",
]

SYSTEM_PROMPT_TEMPLATE = """You are a scene-understanding assistant that \
classifies indoor rooms from the objects observed inside them.

You will be given a list of objects observed in one room. Based ONLY on \
those objects and common sense about where they occur, output a single \
best room label.

Choose the label from this set (do not invent new ones):
{label_set}

Reply with a JSON object of the form:
{{"label": "<one label from the set>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<one short sentence>"}}"""


USER_PROMPT_TEMPLATE = """Room observed objects:
{obj_list}

Classify this room."""


def format_object_list(classes: Sequence[str]) -> str:
    """Collapse duplicate class names into ``- name xN`` count lines."""
    if not classes:
        return "(no objects observed yet)"
    c = Counter(classes)
    return "\n".join(f"- {name} x{n}" for name, n in sorted(c.items()))


@dataclass(frozen=True)
class RoomLabel:
    """One room-type verdict.

    Attributes:
        label: A label from the classifier's label set (out-of-set
            replies are coerced to ``"unknown"``).
        confidence: The model's self-reported confidence in [0, 1]
            (0.0 when unparseable or when no LLM call was made).
        reasoning: One short sentence from the model (<= 200 chars).
    """

    label: str
    confidence: float
    reasoning: str


class RoomTypeClassifier:
    """Objects-in-room -> LLM -> room type label, with a signature cache.

    Args:
        client: The :class:`LLMClient` to query.
        label_set: Candidate labels offered to the model. Replies
            outside this set are coerced to ``"unknown"``.
        min_objects: Rooms with fewer observed objects are labelled
            ``"unknown"`` without an LLM call.

    LLM transport/parse errors propagate as exceptions (``RuntimeError``
    / ``ValueError`` from the client) — the caller decides whether to
    keep a stale label; nothing is silently cached on failure.
    """

    def __init__(self, client: LLMClient,
                 label_set: Sequence[str] = DEFAULT_LABEL_SET,
                 min_objects: int = 1):
        self._client = client
        self._label_set = [str(s) for s in label_set]
        self._min_objects = int(min_objects)
        # Cache: frozenset(class names) -> RoomLabel. The signature is
        # *which* classes were seen, so count changes never re-call.
        self._sig_cache: Dict[FrozenSet[str], RoomLabel] = {}

    @property
    def label_set(self) -> List[str]:
        """The candidate labels offered to the model (a copy)."""
        return list(self._label_set)

    @property
    def cache_size(self) -> int:
        """Number of distinct object-set signatures answered so far."""
        return len(self._sig_cache)

    def classify(self, classes: Sequence[str]) -> RoomLabel:
        """Classify one room from its observed object class names.

        Class names are lower-cased/stripped and empties dropped before
        gating and signature computation, mirroring the ROS node's
        normalization of scene-graph entries.
        """
        norm = [str(c).strip().lower() for c in classes
                if str(c).strip()]
        if len(norm) < self._min_objects:
            # Don't call the LLM for an (almost) empty room.
            return RoomLabel(label="unknown", confidence=0.0,
                             reasoning="no objects observed yet")
        sig = frozenset(norm)
        cached = self._sig_cache.get(sig)
        if cached is not None:
            return cached
        result = self._classify(norm)
        self._sig_cache[sig] = result
        return result

    # -- LLM call ------------------------------------------------------
    def _classify(self, classes: List[str]) -> RoomLabel:
        system = SYSTEM_PROMPT_TEMPLATE.format(
            label_set=", ".join(self._label_set))
        user = USER_PROMPT_TEMPLATE.format(
            obj_list=format_object_list(classes))
        reply = self._client.chat_json(system, user)

        label = str(reply.get("label", "unknown")).strip().lower()
        if label not in self._label_set:
            # Coerce unknown-to-us labels into 'unknown' so downstream
            # stays in-set.
            label = "unknown"
        try:
            conf = float(reply.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        reasoning = str(reply.get("reasoning", ""))[:200]
        return RoomLabel(label=label, confidence=conf, reasoning=reasoning)
