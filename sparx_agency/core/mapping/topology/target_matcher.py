"""Target-vs-detected-class matching for the scene-graph search stack.

The YOLO class name and the natural-language target rarely match
character-for-character ("car keys" vs "key", "toilet seat" vs
"toilet"). :class:`TargetMatcher` implements the match ladder from the
SJTU ``target_watcher_node.py``:

1. Exact lowercase equality short-circuits to True and NEVER asks the
   LLM — it's wasteful, and small local models were observed answering
   False for identical strings (llama3.2:3b on target='toilet' vs
   class='toilet').
2. A per-instance ``(target, class)`` cache answers repeats.
3. Otherwise the LLM is asked for a boolean verdict.
4. If the LLM is off or errors, :func:`fallback_match` (token overlap)
   answers offline.

NOTE: the offline rung is the *same* rule the localization-free
visual-servo stack acquires on, so there is one copy of it, in
:func:`sparx_agency.core.common.label_match.label_matches`. Only the
ladder above it (exact -> cache -> LLM) is specific to this stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from sparx_agency.core.common.label_match import label_matches
from sparx_agency.core.mapping.topology.llm_client import LLMClient, coerce_bool

MATCH_SYSTEM = """You decide whether a detected object should count as a hit \
for a robot's search target.

Inputs:
  TARGET: the object the user asked the robot to find.
  CLASS:  a word or short phrase from the object detector's vocabulary.

Answer TRUE if a real-world object detected as CLASS would reasonably be \
accepted as the TARGET. Answer FALSE otherwise.

Guidelines:
- Same word, same meaning -> TRUE. (target="toilet", class="toilet" -> TRUE.)
- CLASS is a more specific kind of TARGET -> TRUE. \
(target="keys", class="car key" -> TRUE.)
- CLASS and TARGET are synonyms or near-synonyms -> TRUE. \
(target="couch", class="sofa" -> TRUE; target="mug", class="cup" -> TRUE.)
- TARGET names a specific thing and CLASS is the usual label for its \
main object -> TRUE. (target="toilet seat", class="toilet" -> TRUE; \
target="car keys", class="key" -> TRUE.)
- CLASS is a broader category, a different object, or only an accessory \
of TARGET -> FALSE. (target="apple", class="fruit" -> FALSE; \
target="car keys", class="car" -> FALSE; \
target="laptop", class="monitor" -> FALSE.)
- CLASS is a person or animal and TARGET isn't -> FALSE.

Reply ONLY with a JSON object:
{"match": <true|false>, "reason": "<one short sentence>"}"""


MATCH_USER_TEMPLATE = """TARGET: {target!r}
CLASS:  {cname!r}

Match?"""


def fallback_match(target: str, cname: str) -> bool:
    """Last-resort offline match: token overlap plus substring.

    Known looseness, kept on purpose (ported verbatim): any shared
    token matches, so ``target='car keys'`` vs ``class='car'`` returns
    True here even though the LLM prompt's guidelines explicitly call
    that pair FALSE (the class is only an accessory of the target).
    Offline we prefer a false positive (the drone stops at a car) over
    never firing at all.
    """
    # Delegated, not re-implemented: the visual-servo acquisition gate
    # acquires on exactly this rule, and core/common is the narrowest
    # ring that covers both a mapping and a planning consumer without
    # either package importing the other.
    return label_matches(target, cname)


@dataclass(frozen=True)
class MatchResult:
    """Verdict for one (target, class) pair.

    Attributes:
        match: True when the detected class counts as the target.
        reason: One short sentence — the LLM's reason, or a tag naming
            the ladder rung that answered (<= 160 chars).
    """

    match: bool
    reason: str


class TargetMatcher:
    """The match ladder: exact -> cache -> LLM -> offline fallback.

    Args:
        client: Optional :class:`LLMClient`. With ``None`` the matcher
            works fully offline via :func:`fallback_match`.
        use_llm: Set False to force offline matching even when a
            client was passed.
    """

    def __init__(self, client: Optional[LLMClient] = None,
                 use_llm: bool = True):
        self._client = client if use_llm else None
        self._cache: Dict[Tuple[str, str], MatchResult] = {}

    @property
    def cache_size(self) -> int:
        """Number of distinct (target, class) pairs answered so far."""
        return len(self._cache)

    def matches(self, target: str, cname: str) -> MatchResult:
        """Decide whether a detection of ``cname`` counts as ``target``."""
        t = str(target).strip().lower()
        c = str(cname).strip().lower()
        if not t or not c:
            # The ROS node filtered empty class names before matching;
            # guard here so ""=="" can never read as a hit.
            return MatchResult(match=False, reason="empty target or class")

        # Exact match is an exact match. Never ask the LLM about this.
        if t == c:
            result = MatchResult(match=True, reason="exact match")
            self._cache[(t, c)] = result
            return result

        key = (t, c)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._client is not None:
            result = self._ask_llm(target, cname)
            if result is not None:
                self._cache[key] = result
                return result
            # LLM error -> fall through to the offline fallback.

        verdict = fallback_match(target, cname)
        result = MatchResult(match=verdict,
                             reason="offline token-overlap fallback")
        self._cache[key] = result
        return result

    # -- LLM rung ------------------------------------------------------
    def _ask_llm(self, target: str, cname: str) -> Optional[MatchResult]:
        user = MATCH_USER_TEMPLATE.format(target=target, cname=cname)
        try:
            reply = self._client.chat_json(MATCH_SYSTEM, user)
        except Exception:
            return None
        # coerce_bool, never bool(): this model answers with the word
        # quoted, and bool("false") is True. See coerce_bool's docstring.
        verdict = coerce_bool(reply.get("match"), default=False)
        reason = str(reply.get("reason", ""))[:160]
        return MatchResult(match=verdict, reason=reason)
