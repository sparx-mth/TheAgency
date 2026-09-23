"""LLM search oracle: target + rooms -> a usable P(target is in this room).

**The split this module is built on.** The language model is asked ONE
question, the one it is genuinely good at: *how typical is it for this kind of
object to be in this kind of room?* Everything arithmetic -- how big the room
is, how long we have already searched it, how much unscanned space is left --
is applied in code afterwards. A 3B model cannot compute ``exp(-tau/T)``, and,
worse, when it is shown the effort numbers it double-counts them into the
semantic judgement: the room it has just been told was searched for 300 s comes
back with a low score for the wrong reason, and the caller can no longer tell
semantics from bookkeeping.

**The failure this replaces.** On a hospital flight the old prompt asked
directly for probabilities and got ``1.0 / 0.0 / 0.0`` over three equally
unsearched rooms. The parse clamped nothing useful, sum-normalisation preserved
the one-hot exactly, and the search policy's ``min_prob`` filter then dropped
every room but one -- so the draw had a single candidate and the ranking was
not a ranking at all. Three independent defences now stop that:

* the prompt asks for an integer AFFINITY 0-100 against named anchor bands, and
  forbids both 100 and 0 ("you have not looked inside");
* :func:`affinity_weights` clamps into ``[score_floor, score_ceiling]`` and
  shrinks toward the tick's own mean, so an extreme reply keeps its ORDERING
  while losing its false certainty;
* a room the model omits scores the mean rather than zero -- an unmentioned
  room is one the model forgot, not one it ruled out.

Under that scheme the same ``100/0/0`` reply over three equal rooms becomes
``0.756 / 0.122 / 0.122``: the model's ordering survives, and the policy gets a
real draw instead of a foregone conclusion.

**Mass is deliberately reserved for elsewhere.** ``p_elsewhere`` holds the
probability that the target is in none of the rooms mapped so far. Normalising
over only-mapped-rooms asserts that two rooms into a flight the target is
certainly in one of those two, which is both false and exactly the belief that
stops a search flying toward unmapped space. The returned ``probs`` still sum
to 1 over the rooms (callers depend on that), but ``OracleResult`` carries
``p_present`` beside them so a planner that wants the honest number has it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.mapping.topology.llm_client import LLMClient

SYSTEM_PROMPT = """You rate rooms for a search robot. You have NOT looked \
inside them. You are guessing from the room type, the few objects the robot \
happened to see, and the room's size.

For EACH room give an integer AFFINITY score 0-100: how typical it is for the \
TARGET object to be in a room like that.

Bands (use them):
  85 = the object's usual home (fridge -> kitchen)
  65 = often found there
  50 = no idea, could be anywhere
  30 = unusual but possible
  10 = would be surprising
   3 = essentially impossible (fridge -> lift shaft)

Rules:
 1. Score EVERY room id you are given. Never invent an id.
 2. NEVER use 100 and NEVER use 0. You have not looked inside.
 3. At most ONE room may score above 85.
 4. Scores are independent. They do NOT sum to 100 and they are NOT
    probabilities.
 5. type=unknown means "not identified yet", NOT "empty". Score it 45-60:
    nearer 60 when it is large, nearer 45 when it is small.
 6. Judge TYPE, SEEN objects and SIZE only. Say nothing about how long the
    robot has searched - the caller handles that.
 7. Write "why" FIRST, then "score". Max 8 words, lower case.

Reply with ONLY this JSON, keys in this order:
{"rooms":[{"id":<int>,"why":"<max 8 words>","score":<int 0-100>}]}

Example - TARGET wheelchair:
{"rooms":[{"id":1,"why":"wards park wheelchairs beside beds","score":85},
{"id":2,"why":"corridor, through-route, sometimes parked","score":45},
{"id":3,"why":"unidentified large room, could be anything","score":55},
{"id":4,"why":"tiny cupboard, too small","score":12}]}"""


USER_PROMPT_TEMPLATE = """TARGET: {target}

ROOMS ({n_rooms}):
{rooms_block}

Score all {n_rooms} rooms. JSON only."""


@dataclass(frozen=True)
class OracleScoring:
    """How a raw affinity score becomes a probability. Every knob.

    Attributes:
        score_floor: Scores clamp up to this. A model that writes 0 means
            "surprising", not "impossible" -- it has not looked inside.
        score_ceiling: Scores clamp down to this, for the same reason in the
            other direction. Together with ``w_llm`` this is what stops a
            one-hot reply becoming a one-hot distribution.
        w_llm: How far the clamped scores are kept from the tick's own mean.
            1.0 trusts the model completely; 0.0 ignores it. 0.7 keeps the
            ORDERING intact while removing the false certainty -- an extreme
            reply still ranks its rooms in the same order, it just no longer
            claims the others are impossible.
        area_ref_m2: Room area treated as "normal". Bigger rooms are more
            likely to contain a given object simply by having more places to
            put it; the term is a gentle power, not a proportionality.
        area_exponent: Exponent on ``area / area_ref_m2``. Size must be a
            TIEBREAKER, never a driver. Measured with 0.5: asked for a
            toilet, the model correctly scored the 6 m2 bathroom 85 against
            a 48 m2 ward at 30, and the size term still put the ward first
            (0.327 vs 0.254) -- a 2.8x size ratio beating a 2.2x affinity
            ratio. 0.25 makes that ratio 1.7 and the semantics win, while a
            room four times the size is still ~1.4x preferred at equal
            affinity.
        area_clamp: Hard bounds on the size factor. Even at a low exponent an
            enormous room would otherwise dominate every other term; a
            multiplier outside this range is size deciding the search on its
            own, which it must never do.
        effort_half_life_s: Seconds of searching a reference-sized room before
            its probability halves. Scaled by room area, so a big room is not
            written off in the time it takes to clear a cupboard.
        effort_floor: The discount never falls below this WHILE unscanned
            space remains. A room searched for a long time with frontier left
            is a room we have not finished, not a room we have cleared.
        exhausted_factor: Applied instead once a room reports no frontier
            clusters at all. Small, but not zero: the detector can miss, and a
            room we can never revisit is a room the search can never correct.
        presence_scale: Caps how much of ``p_present`` any single room may
            claim in the noisy-OR. An affinity is "how typical", not a
            calibrated presence probability, so a room scoring the ceiling
            contributes at most this. Affects ``p_present`` only -- never the
            per-room ``probs``, which are normalised and so invariant to it.
    """

    score_floor: float = 3.0
    score_ceiling: float = 90.0
    w_llm: float = 0.7
    area_ref_m2: float = 20.0
    area_exponent: float = 0.25
    area_clamp: Tuple[float, float] = (0.6, 1.6)
    effort_half_life_s: float = 90.0
    effort_floor: float = 0.25
    exhausted_factor: float = 0.15
    presence_scale: float = 0.5


@dataclass(frozen=True)
class OracleRoom:
    """One room as presented to the oracle.

    Attributes:
        id: Scene-graph room id (persistent id).
        label: Room-type label (e.g. from ``RoomTypeClassifier``).
            ``"unknown"`` is the COMMON early case, not an error: the
            classifier needs objects before it can name a room, so most of a
            flight's first minute is unknown rooms. The prompt has an explicit
            rule for it.
        searched_s: Seconds already searched in this room (tau_r). Applied in
            code, NOT shown to the model.
        frontier_clusters: Unexplored frontier clusters remaining (F_r).
            Applied in code, NOT shown to the model.
        observed_classes: Object class names observed in the room
            (duplicates fine; deduplicated, lower-cased and sorted for the
            prompt, then capped -- a 3B given forty class names answers about
            the list rather than the room).
        area_m2: Room floor area in square metres, or 0 when unknown. Shown to
            the model AND used in code: it is the one geometric fact that
            changes both the semantic judgement ("too small to park a
            wheelchair") and the arithmetic.
    """

    id: int
    label: str
    searched_s: float = 0.0
    frontier_clusters: int = 0
    observed_classes: Tuple[str, ...] = field(default_factory=tuple)
    area_m2: float = 0.0


@dataclass(frozen=True)
class OracleResult:
    """Per-room probabilities for one oracle query.

    Attributes:
        probs: ``{room_id: probability}`` summing to 1.0 over the rooms given.
        source: ``'llm'`` for a usable model reply, or ``'uniform_fallback'``
            when the model failed and a uniform distribution was substituted.
        reasons: ``{room_id: one-sentence reason}``.
        raw_reply: The parsed model reply for debugging (None when the call
            itself failed).
        scores: ``{room_id: raw affinity 0-100}`` as the model wrote it,
            before any clamp -- so a degenerate tick is greppable in a flight
            log rather than inferred afterwards.
        p_present: Probability the target is in ANY mapped room. Below 1
            whenever unscanned space remains, which is what keeps a search
            willing to fly somewhere it has not mapped.
        spread: Max minus min of ``probs``. A tick where this is ~0 is a tick
            where the oracle said nothing useful.
    """

    probs: Dict[int, float]
    source: str
    reasons: Dict[int, str]
    raw_reply: Optional[Dict[str, Any]]
    scores: Dict[int, float] = field(default_factory=dict)
    p_present: float = 1.0
    spread: float = 0.0


MAX_CLASSES_IN_PROMPT = 6
"""Observed classes shown per room. A 3B given forty answers about the list."""


def format_rooms_block(rooms: Sequence[OracleRoom]) -> str:
    """Render the per-room lines of the oracle's user prompt.

    Deliberately carries NO ordinal prefix: a leading ``1.`` invites a small
    model to answer with the ordinal instead of the id. And deliberately no
    effort numbers -- see the module docstring.
    """
    lines = []
    for r in rooms:
        names = sorted({str(c).strip().lower() for c in r.observed_classes
                        if str(c).strip()})
        seen = ", ".join(names[:MAX_CLASSES_IN_PROMPT]) if names else "nothing yet"
        size = ("%dm2" % int(round(float(r.area_m2)))
                if float(r.area_m2) > 0.0 else "unknown size")
        lines.append("id=%d  type=%s  size=%s  seen: %s"
                     % (int(r.id), str(r.label), size, seen))
    return "\n".join(lines)


def affinity_weights(scores: Dict[int, float],
                     room_ids: Sequence[int],
                     scoring: OracleScoring) -> Dict[int, float]:
    """Clamp raw affinities and shrink them toward the tick's own mean.

    This is the step that turns a one-hot reply into a ranking. Clamping alone
    would not do it -- ``90/3/3`` is still nearly one-hot after
    normalisation -- so the clamped scores are then pulled toward their own
    mean by ``1 - w_llm``. The ORDER is untouched (the map is affine and
    increasing), only the confidence is.

    A room the model omitted scores the MEAN, not zero: an unmentioned room is
    one the model forgot, not one it ruled out, and zero-filling it is how the
    old parse turned a lazy reply into a permanent exclusion.

    Args:
        scores: ``{room_id: raw score}`` for the rooms the model scored.
        room_ids: Every room, in output order.
        scoring: The knobs.

    Returns:
        ``{room_id: positive weight}``, not normalised.
    """
    lo, hi = float(scoring.score_floor), float(scoring.score_ceiling)
    clamped = {}  # type: Dict[int, float]
    for rid in room_ids:
        raw = scores.get(int(rid))
        if raw is None:
            continue
        clamped[int(rid)] = max(lo, min(hi, float(raw)))
    if not clamped:
        mean = 0.5 * (lo + hi)
    else:
        mean = sum(clamped.values()) / float(len(clamped))
    w = float(scoring.w_llm)
    out = {}  # type: Dict[int, float]
    for rid in room_ids:
        value = clamped.get(int(rid), mean)
        out[int(rid)] = max(1e-6, w * value + (1.0 - w) * mean)
    return out


def size_factor(area_m2: float, scoring: OracleScoring) -> float:
    """Gentle preference for bigger rooms, clamped so it never decides alone.

    ``(area / ref) ** exponent``, bounded by ``scoring.area_clamp``. A bigger
    room really is more likely to hold a given object -- there are more places
    to put it -- but a search that flies to the largest room regardless of
    what the object IS has thrown away the whole point of asking a language
    model. See :attr:`OracleScoring.area_exponent` for the measurement that
    set these numbers.
    """
    area = float(area_m2)
    if area <= 0.0:
        return 1.0
    raw = (area / float(scoring.area_ref_m2)) ** float(scoring.area_exponent)
    lo, hi = scoring.area_clamp
    return float(max(float(lo), min(float(hi), raw)))


def effort_factor(searched_s: float, frontier_clusters: int, area_m2: float,
                  scoring: OracleScoring) -> float:
    """Discount a room by how much of it we have already searched.

    Exponential decay in ``searched_s``, with the half-life scaled by room
    area so a ward is not written off in the time it takes to clear a
    cupboard. Floored while frontier remains -- a room searched for a long
    time with unscanned space left is unfinished, not cleared -- and dropped
    hard once the room reports no frontier at all.
    """
    area = max(1.0, float(area_m2)) if float(area_m2) > 0.0 else float(scoring.area_ref_m2)
    scale = max(1.0, area / float(scoring.area_ref_m2))
    half_life = float(scoring.effort_half_life_s) * scale
    decay = math.exp(-max(0.0, float(searched_s)) / max(1e-6, half_life))
    if int(frontier_clusters) <= 0:
        return float(scoring.exhausted_factor) * decay
    return max(float(scoring.effort_floor), decay)


class SearchOracle:
    """Per-room target-probability oracle over an :class:`LLMClient`."""

    def __init__(self, client: LLMClient,
                 scoring: OracleScoring = OracleScoring()):
        self._client = client
        self.scoring = scoring

    def probabilities(self, target: str,
                      rooms: Sequence[OracleRoom]) -> OracleResult:
        """Return a normalized probability per room for ``target``.

        Never raises on LLM trouble -- a transport error, a malformed reply,
        or an all-unusable score set all degrade to the uniform fallback. An
        empty ``rooms`` sequence is a caller bug and raises ``ValueError``
        (there is no distribution over nothing).
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
        result = self._score(reply, rooms)
        if result is None:
            return self._uniform(rooms, raw_reply=reply)
        return result

    # -- Internals -----------------------------------------------------
    def _uniform(self, rooms: Sequence[OracleRoom],
                 raw_reply: Optional[Dict[str, Any]]) -> OracleResult:
        u = 1.0 / len(rooms)
        return OracleResult(
            probs={r.id: u for r in rooms},
            source="uniform_fallback",
            reasons={r.id: "" for r in rooms},
            raw_reply=raw_reply,
            scores={},
            p_present=1.0,
            spread=0.0,
        )

    def _score(self, reply: Any,
               rooms: Sequence[OracleRoom]) -> Optional[OracleResult]:
        """Parse, clamp, shrink, apply size and effort, normalise."""
        raw_entries = reply.get("rooms") if isinstance(reply, dict) else None
        if not isinstance(raw_entries, list) or not raw_entries:
            return None

        known = {int(r.id) for r in rooms}
        scores = {}  # type: Dict[int, float]
        reasons = {}  # type: Dict[int, str]
        for entry in raw_entries:
            try:
                rid = int(entry.get("id"))
                # 'score' is the new field; 'probability' is accepted so a
                # model that answers in the old shape is still usable rather
                # than dropping the whole tick to uniform. A probability in
                # [0, 1] is scaled to the 0-100 band it belongs in.
                if "score" in entry:
                    value = float(entry.get("score"))
                else:
                    value = float(entry.get("probability")) * 100.0
            except (AttributeError, TypeError, ValueError):
                continue
            if rid not in known:
                continue            # invented id: dropped, never scored
            scores[rid] = value
            reasons[rid] = str(entry.get("why", entry.get("reason", "")))[:200]
        if not scores:
            return None

        room_ids = [int(r.id) for r in rooms]
        weights = affinity_weights(scores, room_ids, self.scoring)

        by_id = {int(r.id): r for r in rooms}
        raw_vec = []
        for rid in room_ids:
            room = by_id[rid]
            raw_vec.append(
                weights[rid]
                * size_factor(room.area_m2, self.scoring)
                * effort_factor(room.searched_s, room.frontier_clusters,
                                room.area_m2, self.scoring))
        total = sum(raw_vec)
        if total <= 1e-9:
            return None

        probs = {rid: v / total for rid, v in zip(room_ids, raw_vec)}
        # P(the target is in ANY room we have mapped), as a noisy-OR over the
        # per-room affinities: 1 - prod(1 - a_i). A SUM would exceed 1 as soon
        # as a few rooms were mapped and would then say nothing at all, which
        # is exactly the "certainly in one of these two" error this number
        # exists to avoid. The residual, 1 - p_present, is the mass belonging
        # to space not yet segmented, and it is what keeps a search willing to
        # fly somewhere new.
        #
        # Heuristic, and labelled as one: an affinity is "how typical", not a
        # calibrated presence probability, so ``presence_scale`` caps how much
        # any single room may claim. Ordering and monotonicity are what this
        # is used for; the absolute value is not calibrated against anything.
        miss = 1.0
        for rid in room_ids:
            room = by_id[rid]
            a = ((weights[rid] / max(1e-9, self.scoring.score_ceiling))
                 * effort_factor(room.searched_s, room.frontier_clusters,
                                 room.area_m2, self.scoring)
                 * float(self.scoring.presence_scale))
            miss *= max(0.0, 1.0 - min(1.0, a))
        p_present = float(min(1.0, 1.0 - miss))
        values = list(probs.values())
        return OracleResult(
            probs=probs,
            source="llm",
            reasons={rid: reasons.get(rid, "") for rid in room_ids},
            raw_reply=reply,
            scores=dict(scores),
            p_present=p_present,
            spread=float(max(values) - min(values)) if values else 0.0,
        )
