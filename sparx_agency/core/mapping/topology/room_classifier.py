"""Room-type classification from observed object classes, via an LLM.

Defaults cache by the class set for existing callers. Count-sensitive cache
keys, minimum class diversity and explicit refresh are available for online
room revisiting. Transport/parse errors propagate without caching a failure.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

from sparx_agency.core.mapping.topology.llm_client import LLMClient

DEFAULT_LABEL_SET = [
    "kitchen", "bedroom", "bathroom", "living_room", "dining_room",
    "office", "hallway", "storage_closet", "laundry_room",
    "lobby", "waiting_area", "patient_room", "exam_room",
    "reception", "unknown",
]

SYSTEM_PROMPT_TEMPLATE = """You are a scene-understanding assistant that
classifies indoor rooms from the objects observed inside them.

You will be given a list of objects observed in one room. Based ONLY on
those objects and common sense about where they occur, output one room label.
Observations are incomplete and may contain false detections. Prefer distinctive
room-function evidence over generic furniture. Use unknown when evidence is
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

Classify this room."""


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

    def classify(self, classes: Sequence[str], refresh: bool = False) -> RoomLabel:
        """Classify current evidence; refresh bypasses a previous cached answer.

        Failure leaves an existing cache entry untouched. Caller controls the
        refresh cadence; this ROS-free core reads no clock or episode state.
        """
        norm = [str(c).strip().lower() for c in classes if str(c).strip()]
        if len(norm) < self._min_objects or len(set(norm)) < self._min_classes:
            return RoomLabel("unknown", 0.0, "no objects observed yet" if not norm
                             else "insufficient observed object evidence")
        signature = (tuple(sorted(Counter(norm).items())) if self._count_sensitive
                     else tuple(sorted(set(norm))))
        cached = self._sig_cache.get(signature)
        if cached is not None and not refresh:
            return cached
        result = self._classify(norm)
        self._sig_cache[signature] = result
        return result

    def _classify(self, classes: List[str]) -> RoomLabel:
        system = SYSTEM_PROMPT_TEMPLATE.format(label_set=", ".join(self._label_set))
        user = USER_PROMPT_TEMPLATE.format(obj_list=format_object_list(classes))
        reply = self._client.chat_json(system, user)
        label = str(reply.get("label", "unknown")).strip().lower()
        if label not in self._label_set:
            label = "unknown"
        try:
            confidence = float(reply.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0
        return RoomLabel(label, confidence, str(reply.get("reasoning", ""))[:200])
