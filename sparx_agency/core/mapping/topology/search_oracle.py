"""LLM search oracle: target + rooms (label, tau_r, F_r) -> room probs.

Asks an LLM: "Given the target object and, for each room, its type, how
long we've searched there, and how many frontier clusters remain, what
is the probability the target is in each room?" Raw LLM outputs often
don't sum to 1 — each raw value is clamped to [0, 1], rooms the model
missed get 0, and the vector is sum-normalized afterwards. If the LLM
returns nothing usable (transport error, missing ``rooms`` list, or an
all-zero vector), the oracle falls back to a uniform distribution
tagged ``source='uniform_fallback'`` so downstream is never starved.

ROS-free port of the LLM plane of the SJTU ``llm_oracle_node.py``; the
ROS wiring (topics, markers, tick loop) stays in the task layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from sparx_agency.core.mapping.topology.llm_client import LLMClient

SYSTEM_PROMPT = """You are a commonsense search-planning oracle for an indoor drone.

You are told a target object and a list of rooms. For each room you are \
given:
  - its type (e.g. "kitchen", "bedroom", "bathroom"),
  - how many seconds the drone has already searched there,
  - how many unexplored "frontier clusters" still remain in that room \
(each cluster is a separate region of free-but-unscanned space).

Your job is to output, for EACH room, the probability that the target is \
currently somewhere in that room. Use your commonsense about which rooms \
typically contain which objects. Also account for the given effort:
  - A lot of prior search time with no success should LOWER the probability.
  - More remaining frontier clusters means more unscanned area, so the \
object is more likely to still be there if it belongs in that room.
  - A room that has not been searched at all but matches the object \
semantically should get a high probability.

The numbers do NOT have to sum to 1 — just give your best per-room estimate \
between 0 and 1. The caller will normalise.

Reply with a JSON object of the form:
{"rooms": [
   {"id": <room_id>, "probability": <float in [0,1]>, "reason": "<one short sentence>"},
   ...
 ]}

Include EVERY room you were given, with its original id. Do not invent rooms."""


USER_PROMPT_TEMPLATE = """Target object: {target}

Rooms:
{rooms_block}

Return probabilities for all {n_rooms} rooms as described."""


@dataclass(frozen=True)
class OracleRoom:
    """One room as presented to the oracle.

    Attributes:
        id: Scene-graph room id (persistent id).
        label: Room-type label (e.g. from ``RoomTypeClassifier``).
        searched_s: Seconds already searched in this room (tau_r).
        frontier_clusters: Unexplored frontier clusters remaining (F_r).
        observed_classes: Object class names observed in the room
            (duplicates fine; deduplicated + sorted for the prompt).
    """

    id: int
    label: str
    searched_s: float = 0.0
    frontier_clusters: int = 0
    observed_classes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OracleResult:
    """Normalized per-room probabilities for one oracle query.

    Attributes:
        probs: ``{room_id: probability}`` summing to 1.0.
        source: ``'llm'`` for a usable model reply, or
            ``'uniform_fallback'`` when the model failed and a uniform
            distribution was substituted.
        reasons: ``{room_id: one-sentence reason}`` ("" for rooms the
            model didn't score, and for all rooms on fallback).
        raw_reply: The parsed model reply for debugging (None when the
            call itself failed).
    """

    probs: Dict[int, float]
    source: str
    reasons: Dict[int, str]
    raw_reply: Optional[Dict[str, Any]]


def format_rooms_block(rooms: Sequence[OracleRoom]) -> str:
    """Render the per-room lines of the oracle's user prompt."""
    lines = []
    for i, r in enumerate(rooms, start=1):
        cls_names = sorted({str(c) for c in r.observed_classes
                            if str(c).strip()})
        obj_blurb = (f" observed: {', '.join(cls_names)}" if cls_names
                     else " observed: (none)")
        lines.append(
            f"{i}. Room id={r.id}  type={r.label}  "
            f"searched={float(r.searched_s):.0f}s  "
            f"remaining_frontier_clusters={int(r.frontier_clusters)}."
            f"{obj_blurb}")
    return "\n".join(lines)


class SearchOracle:
    """Per-room target-probability oracle over an :class:`LLMClient`."""

    def __init__(self, client: LLMClient):
        self._client = client

    def probabilities(self, target: str,
                      rooms: Sequence[OracleRoom]) -> OracleResult:
        """Return a normalized probability per room for ``target``.

        Never raises on LLM trouble — a transport error, a malformed
        reply, or an all-zero vector all degrade to the uniform
        fallback. An empty ``rooms`` sequence is a caller bug and
        raises ``ValueError`` (there is no distribution over nothing).
        """
        if not rooms:
            raise ValueError("SearchOracle needs at least one room")
        user = USER_PROMPT_TEMPLATE.format(
            target=target,
            rooms_block=format_rooms_block(rooms),
            n_rooms=len(rooms),
        )
        try:
            reply = self._client.chat_json(SYSTEM_PROMPT, user)
        except Exception:
            return self._uniform(rooms, raw_reply=None)
        result = self._parse(reply, rooms)
        if result is None:
            # Unusable reply — keep it in the fallback for debugging
            # (the ROS node discarded it; deliberate small deviation).
            return self._uniform(rooms, raw_reply=reply)
        return result

    # -- Internals -----------------------------------------------------
    @staticmethod
    def _uniform(rooms: Sequence[OracleRoom],
                 raw_reply: Optional[Dict[str, Any]]) -> OracleResult:
        u = 1.0 / len(rooms)
        return OracleResult(
            probs={r.id: u for r in rooms},
            source="uniform_fallback",
            reasons={r.id: "" for r in rooms},
            raw_reply=raw_reply,
        )

    @staticmethod
    def _parse(reply: Any,
               rooms: Sequence[OracleRoom]) -> Optional[OracleResult]:
        raw_entries = reply.get("rooms") if isinstance(reply, dict) else None
        if not isinstance(raw_entries, list) or not raw_entries:
            return None

        # Build id -> raw prob. Missing rooms get 0 so normalization
        # distributes mass only over the ones the LLM scored.
        got: Dict[int, float] = {}
        reasons: Dict[int, str] = {}
        for e in raw_entries:
            try:
                rid = int(e.get("id"))
                val = float(e.get("probability", 0.0))
            except (AttributeError, TypeError, ValueError):
                continue
            got[rid] = max(0.0, min(1.0, val))
            reasons[rid] = str(e.get("reason", ""))[:200]

        room_ids = [r.id for r in rooms]
        raw_vec = [got.get(rid, 0.0) for rid in room_ids]
        total = sum(raw_vec)
        if total <= 1e-9:
            # All zeros -> uniform fallback rather than a degenerate
            # distribution.
            return None

        return OracleResult(
            probs={rid: v / total for rid, v in zip(room_ids, raw_vec)},
            source="llm",
            reasons={rid: reasons.get(rid, "") for rid in room_ids},
            raw_reply=reply,
        )
