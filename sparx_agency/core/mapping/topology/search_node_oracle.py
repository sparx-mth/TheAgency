"""LLM node oracle: target + the nodes a search can go to next -> P(going there finds it).

The successor of :mod:`search_oracle` for the room-search loop. That module
was written around a 3B model: it asked one narrow question (how typical is
this object in a room of this type) and applied frontiers, search time and
room size in code, because the small model double-counted them into its
semantic judgement. This one is written for a capable model -- the runtime
routes it to ``LLMConfig.reasoning_model`` -- and asks the WHOLE question at
once, over every node the search could go to next:

* a **room** the map has segmented on the storey the robot stands on --
  identified by type from the objects seen in it, or still ``unknown`` --
  with its size, the frontier (unexplored boundary) left inside it, how long
  it was searched and how long ago, and the objects confirmed in it;
* a **staircase** to another storey -- up or down; whether that storey was
  visited, what was found and how much was searched there; whether the
  robot arrived on this storey by it a moment ago.

The reply gives independent per-node search-success estimates, not mutually
exclusive target-location shares and not probabilities limited to a burst.
The room-search loop hands these probabilities without renormalisation to
RPT*, whose objective (expected time-to-find) weighs them against the travel
cost the planner computes -- for a staircase that cost includes the climb --
which is why the model is told, twice, not to reason about distance.

What stays in code is the contract, never the judgement: parse and clamp,
drop invented ids, give an omitted node a small share rather than zero,
derive joint failure under RPT*'s independence approximation, refuse a reply with no
usable node (the caller repairs the schema once, then backs off), and reuse
the last reply when the prompt is byte-identical -- the effort numbers are
shown in coarse steps so an unchanged map produces an unchanged prompt.

One more thing stays in code, since 2026-10-05: the **uncertainty floor**
(``unexplored_floor``). A room never entered and not yet identified, or an
opening nothing has been glimpsed through, is a place the search knows
NOTHING about; the least it owes such a place is a look, so its probability
is never read below the floor whatever the model wrote. The Hanson
recording of 2026-10-04 had a 3B model value two never-entered rooms and
six doorways at 0-1% ("small unknown room, no toilet fits" -- the words of
the prompt's own example, applied to a toilet) against 60% for the stairs,
and the agent left the storey at action 27 with every door on it unlooked
into. The floor is the ``entered``/``identified`` flags made binding: the
model still ranks the unexplored places against each other and against the
stairs, but it cannot write any of them off.

The floor has one condition, since the Ranchester couch search of the same
day: the model's own STEP 2. Asked for it as a structured verdict
(``home_here``: ``found`` / ``missing`` / ``elsewhere``), a small model
answers reliably -- "living rooms are downstairs" at every loop point -- and
then writes the worked example's 25 on every upstairs gap all the same, so
thirteen peeks outranked the staircase the agent stood beside. With
``home_here="elsewhere"`` an unexplored place is read at
``unexplored_elsewhere`` (0.10) exactly, rule 2b's "less" applied by the
code: a look into it comes after the storey the target lives on, and the
stairs at 0.6 come first. ``found`` and ``missing`` keep the 0.25 floor.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOM = "room"
STAIRS = "stairs"
OPENING = "opening"
KINDS = (ROOM, STAIRS, OPENING)

#: Largest node probability handed on: RPT*'s heuristic divides by ``1 - p``.
P_CEILING = 1.0 - 1e-6
#: Share, in percent, given to a listed node the model did not score.
OMITTED_PERCENT = 1.0
#: Object classes shown per room; a longer list is answered about the list, not the room.
MAX_CLASSES_IN_PROMPT = 8
#: Default probability floor for an unexplored node (a never-entered, unidentified room;
#: an opening with nothing glimpsed through it). See the module docstring.
UNEXPLORED_FLOOR = 0.25
#: What an unexplored node is read at when the model's STEP 2 says the target's home type does
#: not belong on this storey (``home_here="elsewhere"``) -- rule 2b's "less", applied by the code
#: because a small model does not apply it: the Ranchester couch search of 2026-10-05 had the 3B
#: model write "living rooms are downstairs" at every loop point and 25 on every upstairs gap all
#: the same (the worked example's number), so thirteen peeks outranked the stairs it stood beside.
UNEXPLORED_ELSEWHERE = 0.10
#: The least a node holding a confirmed HOME OBJECT of the target (``SearchNode.home``: a bathtub
#: or a sink for a toilet) is read at, whatever the model wrote: the object co-occurrence prior
#: the prompt states in prose, as arithmetic. The Allensville toilet run of 2026-10-05 valued the
#: strong bathroom holding the bathtub at 0 for the toilet (the room had been finished by sight
#: from the spawn point) and peeked twenty openings before it.
HOME_FLOOR = 0.60
#: The three answers to "is a room of the home type on this storey?".
HOME_FOUND, HOME_MISSING, HOME_ELSEWHERE = "found", "missing", "elsewhere"
HOME_HERE = (HOME_FOUND, HOME_MISSING, HOME_ELSEWHERE)

SYSTEM_PROMPT = """You are the reasoning module of a robot searching ONE building for an \
instance of ONE target category. The robot has partly mapped the building into NODES.
ROOM nodes are regions the robot segmented on the storey it stands on. A room's \
type was inferred from the objects seen inside it; type=unknown means NOT \
IDENTIFIED YET -- never "empty"; a type ending in "?" was inferred from a single \
kind of object and may be wrong (a sink alone reads as a kitchen until a toilet \
shows). Each room line gives: size; frontier = the number of unexplored openings \
or boundaries still accessible inside it (0 is NOT proof of object absence); searched = the time the robot \
has spent inside it; entered=no if the robot has never stood in it (it was only \
seen from outside), else ago = the time since it was last inside; seen = the \
objects confirmed in it; here=yes if the robot stands in it now.
STAIRS nodes are staircases to another storey (up or down). Taking one means \
leaving this storey and searching the other one. The line says whether that \
storey was visited before and, if so, what was found and how much was searched \
there, and whether the robot arrived on this storey by these stairs a moment ago.
OPENING nodes are doorways or gaps at the edge of the mapped floor that lead to \
space NOT seen yet -- a room the robot has not entered, on THIS storey. Going \
there means a quick look in from the threshold, far cheaper than searching a \
room. The line says which mapped room it opens from, whether a door frame was \
seen, and the objects glimpsed through it so far. Value an opening by the room \
likely BEHIND it: glimpsed objects name that room's type (a toilet -> a \
bathroom; a bed -> a bedroom); nothing glimpsed -> an unknown room on this \
storey, judged by STEP 2 -- if the home type is still MISSING on a storey where \
it belongs, an opening with nothing glimpsed is where it would be found; if the \
home type does not belong on this storey, an opening is worth little. An opening \
whose glimpsed objects name the HOME type itself (a sink or a shower for a toilet; \
a sofa or a television for a couch; a stove for a frying pan) is the home room \
seen from its door: the best node there is, 60 or more, even though the target \
itself has not been glimpsed yet.
For EACH node give the probability, in percent, that going there NEXT finds the \
target: the target is there and the robot would see it by going and looking. \
Estimate EACH node independently: these are search-success probabilities, NOT \
shares of a categorical location distribution. Several rooms may contain an \
instance of the target category. Do NOT make the scores sum to 100. There is no \
fixed action or time horizon. The planner uses independent terminal probabilities \
as an approximation and computes the chance of all listed searches failing itself.
Work in this order, and write steps 1 and 2 into the reply before any number:
STEP 1 -- "home": the room type(s) where this object normally lives, most likely \
first (bed -> bedroom; frying pan -> kitchen; toilet -> bathroom; sofa -> living \
room; washing machine -> laundry room or kitchen). Small objects that travel \
(cup, book, phone) have a home too, only a weaker one.
STEP 2 -- "storey": read THIS storey from the types found on it AND from its rank \
in the building, given in the THIS STOREY line (the lowest of N known storeys, the \
highest, or how many lie below and above it). The rooms decide: a storey where a \
kitchen, dining or living room has been FOUND is the ground floor whatever its \
rank. Where none has, the rank says what to expect: of two storeys the lowest is \
the ground floor and the highest the bedroom floor; of three, the lowest is usually \
a basement (laundry, storage, a garage), the middle the ground floor and the highest \
the bedroom floor; the highest storey of a house is never the ground floor. Then say \
whether a room of the home type has been FOUND on this storey or is still MISSING, \
and, if missing, whether the home type belongs on this storey at all. Where rooms \
live in a house: kitchen, dining and living room downstairs; bedrooms upstairs; \
the bathroom BESIDE the bedrooms, so upstairs too (a ground floor has at most a \
small toilet room); an office or study on either. Write the verdict as "home_here": \
"found" (a room of the home type is on this storey), "missing" (none found yet, but \
the home type belongs on this storey) or "elsewhere" (the home type does not belong \
on this storey; it lives on another one). Write about THIS building: the example \
below shows the format and the style of reasoning, not the answer.
STEP 3 -- the numbers, by these rules:
 1. HOME FOUND HERE with frontier left -- or an OPENING whose glimpsed objects name \
the home type (a sink for a toilet; a sofa for a couch) -- has a high search-success \
probability, 60 or more.
 2. HOME MISSING HERE: independently consider (a) UNKNOWN rooms whose SIZE fits the home \
type -- bathrooms ARE small, so a 3-8 m2 unknown room beside bedrooms is most \
likely the bathroom (never "too small to be one"); a 10-20 m2 room is bedroom \
sized; a 20 m2+ room on a ground floor is the living room -- and (b) the STAIRS \
toward the storey where the home type usually is. Judge each by how well the sizes \
fit and how much of each unknown room is unseen. When the home type is not expected \
on this storey, the stairs can have a high probability without suppressing other nodes.
 2b. UNEXPLORED places: an UNKNOWN room with entered=no, and an OPENING with nothing \
glimpsed, are places the robot knows NOTHING about. It owes each of them at least a \
look before "not here" means anything, so each gets at least 25 when home_here is \
"found" or "missing" -- more when the home type is still missing on a storey where it \
belongs -- and about 10 when home_here is "elsewhere": a look into them comes after the \
storey the target lives on. Never write one off for its size.
 2c. An object of the home type's kind seen in a room of ANOTHER type -- a sink or a \
shower in a "living room", a bed in a "kitchen", a stove in a "bedroom" -- means the \
map merged two rooms into one: value that room like an unknown room of the home \
type (30 or more), not by its label.
 3. A room of ANOTHER identified type, entered or seen with objects of only that type, \
gets 0-3, even if the object could conceivably be there. Do not inflate a \
probability merely to avoid choosing.
 4. A room the target cannot be in by type (frying pan in a bathroom, toilet in a \
bedroom) gets 0-1, whatever else its line says.
 5. frontier=0 means no accessible unexplored boundary, not full semantic coverage. \
Lower the probability with search evidence, but allow missed or occluded objects.
 6. Searched long and recently -> low, not zero. Searched briefly, or long ago, \
with frontier left -> stays promising. Never entered, right type -> the best bet; \
never entered, type unknown -> rule 2b.
 7. A hallway or corridor rarely holds the target, but one with frontier LEADS to \
unseen rooms: value it by what it may lead to.
 8. STAIRS to a visited storey: if a room of the home type was FOUND there and is \
barely searched, that storey can be highly promising -- a known kitchen beats an \
unknown room up here that merely might be one. Searched thoroughly with little \
frontier left -> low.
 9. arrived_by=yes: the robot has just come from there; unless it left that storey \
almost unsearched, do not send it straight back.
 10. Changing storeys costs many actions, but that is NOT your concern: the planner \
adds travel and climbing cost itself. Judge only where the target is.
Always:
 11. NEVER use distance, travel time or here=yes as a reason. The planner accounts \
for travel; the room the robot stands in is judged exactly like the others.
 12. Score EVERY node id given, and only those ids.
 13. Integers 0-100 per node, independently; no normalisation across nodes.
 14. Per node write "why" first (at most 12 words, lower case, naming the room's \
likely type and what decided it), then "p".
Reply with ONLY this JSON, keys in this order:
{"home":"<step 1, max 10 words>","storey":"<step 2, max 20 words>","home_here":"found|missing|elsewhere",
 "nodes":[{"id":<int>,"why":"<why>","p":<int>}, ...]}
Example of the FORMAT and the style of reasoning -- a different building every \
time, so never copy its numbers or its words. TARGET television; THIS STOREY: rooms \
found: bedroom; 2 unknown; searched 1min; 3 rooms with frontier left; 2 storeys \
known to the building; this storey is the highest of 2 known storeys, 1 above the lowest:
id=4  ROOM  type=bedroom  size=13m2  frontier=0  searched=1min  ago=3min  seen: bed, lamp
id=9  ROOM  type=unknown  size=6m2  frontier=1  searched=0s  entered=no  seen: nothing yet
id=11  ROOM  type=unknown  size=22m2  frontier=2  searched=0s  entered=no  seen: nothing yet
id=100003  STAIRS down  to a storey NOT visited yet
id=200001  OPENING  doorway off room 11 (type=unknown)  to space NOT seen yet  glimpsed through it: toilet
id=200002  OPENING  gap off room 11 (type=unknown)  to space NOT seen yet  glimpsed through it: nothing yet
{"home":"living room, sometimes a bedroom",
 "storey":"upper floor (the highest of 2; bedroom found); no living room here; living rooms are downstairs",
 "home_here":"elsewhere",
 "nodes":[{"id":4,"why":"bedroom, fully seen, no television","p":2},
{"id":9,"why":"small unexplored room, never entered, a bathroom perhaps","p":10},
{"id":11,"why":"large unexplored upstairs room, maybe a lounge","p":20},
{"id":100003,"why":"unvisited ground floor holds the living room","p":70},
{"id":200001,"why":"toilet glimpsed: a bathroom, no television","p":2},
{"id":200002,"why":"unseen upstairs room, nothing known of it yet","p":10}]}
Had the TARGET been a toilet in the same building, the bathroom belongs up here \
(home_here "missing"): id=200001 with the toilet glimpsed would be 90, id=9 \
(small, beside the bedroom) 45, id=200002 35, id=11 15, and the stairs 15."""
USER_PROMPT_TEMPLATE = """TARGET: {target}

THIS STOREY: {storey}
OTHER STOREYS: {others}

NODES ({n}):
{nodes}

Score all {n} nodes independently. JSON only."""


@dataclass(frozen=True)
class SearchNode:
    """One place the search could go to next, as shown to the model.

    Attributes:
        id: The loop's node id -- a room pid, or a staircase id offset far
            above every pid. The model echoes it; nothing else about it matters.
        kind: :data:`ROOM`, :data:`STAIRS` or :data:`OPENING`.
        label: Room type (``unknown`` while unidentified), ``stairs up`` /
            ``stairs down``, or ``doorway`` / ``gap`` for an opening.
        tentative: The room type rests on a single kind of object (a weak
            label); rendered as ``type=kitchen?`` so the model can weigh it.
        area_m2: Room floor area; 0 when unknown or for stairs.
        frontier_clusters: Unexplored boundary clusters left in the room.
        searched_s: Seconds the robot has spent inside the room.
        last_inside_ago_s: Seconds since it was last inside; None = never.
        here: The robot stands in this room now.
        objects: Object classes confirmed in the room; for an opening, the
            classes glimpsed through it.
        direction: ``+1`` up / ``-1`` down for stairs, 0 for rooms.
        destination_visited: Stairs: the other storey has been stood on.
        destination: Stairs: one-line summary of that storey -- rooms found,
            time searched, frontier left -- or None when unvisited.
        arrived_by: Stairs: the robot came onto this storey by them.
        arrived_ago_s: Stairs: seconds since that arrival, when ``arrived_by``.
        via: Opening: the mapped room it opens from, in prompt words --
            ``room 6 (type=bathroom?)`` -- or None when unknown.
        home: The room holds a confirmed HOME OBJECT of the target (a
            bathtub or a sink for a toilet, a television for a sofa):
            the one room type the target lives in is standing in it, so
            its probability is never read below the oracle's
            ``home_floor`` (:func:`floor_unexplored`), whatever the model
            wrote. The loop sets it from ``room_priors.HOME_OBJECTS``.
    """

    id: int
    kind: str
    label: str
    tentative: bool = False
    area_m2: float = 0.0
    frontier_clusters: int = 0
    searched_s: float = 0.0
    last_inside_ago_s: Optional[float] = None
    here: bool = False
    objects: Tuple[str, ...] = field(default_factory=tuple)
    direction: int = 0
    destination_visited: bool = False
    destination: Optional[str] = None
    arrived_by: bool = False
    arrived_ago_s: Optional[float] = None
    via: Optional[str] = None
    home: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("SearchNode.kind must be one of %s, got %r" % (KINDS, self.kind))

    @property
    def unexplored(self) -> bool:
        """Whether the search knows nothing about this place yet, so the floor applies.

        A room never entered whose type is still ``unknown``; an opening
        nothing has been glimpsed through. A staircase never is -- the
        other storey is valued by what this one turned out to be -- and
        neither is an identified room or an opening with a glimpse, which
        the model values on the evidence.
        """
        if self.kind == ROOM:
            return self.last_inside_ago_s is None and (self.label or "unknown").strip().lower() == "unknown"
        if self.kind == OPENING:
            return not any(str(c).strip() for c in self.objects)
        return False


@dataclass(frozen=True)
class SearchContext:
    """What the model is told about the building around the nodes.

    Attributes:
        storey: One line on the storey the robot stands on: rooms found by
            type, time searched, rooms with frontier left.
        others: One line per other storey known, or the storeys the stairs
            lead to; empty when the building has shown no other storey.
    """

    storey: str = "the only storey mapped so far"
    others: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NodeOracleResult:
    """The model's distribution, made safe for the planner.

    Attributes:
        probs: Independent per-node search-success estimates in ``[0, P_CEILING]``.
        elsewhere: Legacy field name: product of node failure probabilities,
            NOT a categorical estimate of an undiscovered room.
        p_present: One minus ``elsewhere`` under the independence approximation.
        source: ``'llm'`` for a usable reply, ``'uniform_fallback'`` when the
            model failed and a flat distribution was substituted. The runtime
            refuses to fly on the latter.
        reasons: ``{node_id: why}`` in the model's words.
        raw_reply: The parsed reply, for the recording.
        spread: Max minus min of ``probs``; ~0 means the model said nothing.
        reused: The prompt was byte-identical to the last one answered, so
            the kept reply was re-read instead of spending a call.
        omitted: Node ids the model did not score (given :data:`OMITTED_PERCENT`).
        floored: Node ids read at the uncertainty floor because the model
            valued an unexplored place below it (see the module docstring).
        capped: Node ids read DOWN to :data:`UNEXPLORED_ELSEWHERE` because the
            model's own STEP 2 said the home type lives on another storey
            (``home_here="elsewhere"``) and it valued an unexplored place on
            this one above that all the same.
        home_here: The model's structured STEP 2 verdict -- ``found``,
            ``missing`` or ``elsewhere`` -- or None when it gave none.
        reading: The model's steps 1 and 2 in its own words -- ``home`` (where
            the target normally lives), ``storey`` (what this storey is and
            whether the home type was found or is missing here) and
            ``home_here``. Written before the numbers so a mid-size model
            applies its own reading; kept so the recording shows the
            judgement behind the distribution.
    """

    probs: Dict[int, float]
    elsewhere: float
    p_present: float
    source: str
    reasons: Dict[int, str]
    raw_reply: Optional[Dict[str, Any]]
    spread: float = 0.0
    reused: bool = False
    omitted: Tuple[int, ...] = ()
    reading: Dict[str, str] = field(default_factory=dict)
    probability_model: str = "independent_search_success"
    #: Node ids whose probability was raised to the uncertainty floor (``unexplored_floor``).
    floored: Tuple[int, ...] = ()
    #: Node ids whose probability was lowered to ``unexplored_elsewhere`` (see ``home_here``).
    capped: Tuple[int, ...] = ()
    home_here: Optional[str] = None


# -- the prompt -------------------------------------------------------------
def coarse_seconds(seconds: Optional[float]) -> str:
    """Seconds in the steps the prompt uses, so an unchanged map gives an unchanged prompt.

    Under a minute in 10 s steps, then whole minutes: the model does not need
    the exact second, and a prompt that changed every action would spend a
    call on every loop point over a map that did not move.
    """
    if seconds is None:
        return "never"
    value = max(0.0, float(seconds))
    if value < 60.0:
        return "%ds" % (int(round(value / 10.0)) * 10)
    return "%dmin" % int(round(value / 60.0))


def format_node(node: SearchNode) -> str:
    """One prompt line for ``node``."""
    if node.kind == STAIRS:
        parts = ["id=%d" % node.id, "STAIRS %s" % ("up" if node.direction > 0 else "down")]
        if node.destination_visited:
            parts.append("to a storey ALREADY visited (%s)" % (node.destination or "no detail"))
        else:
            parts.append("to a storey NOT visited yet")
        if node.arrived_by:
            parts.append("the robot came %s these stairs %s ago (arrived_by=yes)" % (
                "down" if node.direction > 0 else "up", coarse_seconds(node.arrived_ago_s)))
        return "  ".join(parts)
    if node.kind == OPENING:
        names = sorted({str(c).strip().lower() for c in node.objects if str(c).strip()})
        glimpsed = ", ".join(names[:MAX_CLASSES_IN_PROMPT]) if names else "nothing yet"
        origin = "off %s" % node.via if node.via else "off the mapped floor"
        return "  ".join(["id=%d" % node.id, "OPENING", "%s %s" % (node.label or "gap", origin),
                          "to space NOT seen yet", "glimpsed through it: %s" % glimpsed])
    names = sorted({str(c).strip().lower() for c in node.objects if str(c).strip()})
    seen = ", ".join(names[:MAX_CLASSES_IN_PROMPT]) if names else "nothing yet"
    size = "%dm2" % int(round(float(node.area_m2))) if float(node.area_m2) > 0.0 else "unknown size"
    label = node.label + ("?" if node.tentative and node.label != "unknown" else "")
    parts = ["id=%d" % node.id, "ROOM", "type=%s" % label, "size=%s" % size,
             "frontier=%d" % max(0, int(node.frontier_clusters)),
             "searched=%s" % coarse_seconds(node.searched_s),
             "entered=no" if node.last_inside_ago_s is None else "ago=%s" % coarse_seconds(node.last_inside_ago_s)]
    if node.here:
        parts.append("here=yes")
    parts.append("seen: %s" % seen)
    return "  ".join(parts)


def format_prompt(target: str, nodes: Sequence[SearchNode], context: Optional[SearchContext] = None) -> str:
    """Everything the model is shown for this query."""
    context = context or SearchContext()
    return USER_PROMPT_TEMPLATE.format(
        target=target, storey=context.storey,
        others="; ".join(context.others) if context.others else "none known",
        n=len(nodes), nodes="\n".join(format_node(n) for n in nodes))


# -- the parse --------------------------------------------------------------
def _percent(entry: Dict[str, Any]) -> Optional[float]:
    """The node's share in percent, from ``p`` or one of the field names a model drifts to."""
    for key in ("p", "probability", "percent", "score"):
        if key in entry:
            try:
                value = float(entry[key])
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            if key == "probability" and value <= 1.0:
                value *= 100.0                       # a fraction in a field named probability
            return max(0.0, min(100.0, value))
    return None


def parse_reply(reply: Any, nodes: Sequence[SearchNode]) -> Optional[Tuple[Dict[int, float], float, Dict[int, str], Tuple[int, ...]]]:
    """The scores, elsewhere, reasons and omitted ids in ``reply``, or None when unusable.

    Invented ids are dropped. A listed node the model skipped is given
    :data:`OMITTED_PERCENT` -- forgotten, not ruled out. A reply that scores
    no listed node at all is unusable. An explicit all-zero belief is valid:
    the runtime can explore geometrically rather than invent a semantic prior.
    """
    rows = reply.get("nodes") if isinstance(reply, dict) else None
    if not isinstance(rows, list):
        return None
    known = {int(n.id) for n in nodes}
    scores = {}  # type: Dict[int, float]
    reasons = {}  # type: Dict[int, str]
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            nid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if nid not in known:
            continue
        value = _percent(row)
        if value is None:
            continue
        scores[nid] = value
        reasons[nid] = str(row.get("why", row.get("reason", "")))[:160]
    if not scores:
        return None
    try:
        elsewhere = float(reply.get("elsewhere", 0.0))
    except (TypeError, ValueError):
        elsewhere = 0.0
    if not math.isfinite(elsewhere):
        elsewhere = 0.0
    elsewhere = max(0.0, min(100.0, elsewhere))
    omitted = tuple(sorted(nid for nid in known if nid not in scores))
    for nid in omitted:
        scores[nid] = OMITTED_PERCENT
        reasons[nid] = "(not scored by the model)"
    return scores, elsewhere, reasons, omitted


_HOME_HERE_WORDS = {
    HOME_FOUND: ("found", "present", "here", "yes", "on this storey", "on this floor"),
    HOME_MISSING: ("missing", "not found", "none found", "not yet", "not yet found", "nothing found", "no room",
                   "unseen", "expected", "no"),
    HOME_ELSEWHERE: ("elsewhere", "not here", "another storey", "other storey", "another floor", "other floor",
                     "downstairs", "upstairs", "does not belong", "other level", "another level"),
}


def parse_home_here(reply: Any) -> Optional[str]:
    """The model's ``home_here`` verdict as one of :data:`HOME_HERE`, or None when absent or unreadable.

    The three words are what the prompt asks for; a few phrasings a model
    drifts to are accepted, the longest match first so "not found" is
    ``missing`` and not ``found``. A negated "found" ("none found", "no
    room ... found") is ``missing``; a bare "no" is ``missing`` too -- the
    conservative verdict, which keeps the uncertainty floor -- never
    ``elsewhere``, which lowers every unexplored place on the storey.
    """
    if not isinstance(reply, dict):
        return None
    value = reply.get("home_here")
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower()
    for verdict in HOME_HERE:
        if text == verdict:
            return verdict
    best = None
    for verdict, words in _HOME_HERE_WORDS.items():
        for word in words:
            if word in text and (best is None or len(word) > len(best[1])):
                best = (verdict, word)
    if best is None:
        return None
    if best[0] == HOME_FOUND and any(negation in text for negation in ("not ", "no ", "none", "never", "n't")):
        return HOME_MISSING
    return best[0]


def floor_unexplored(scores: Dict[int, float], nodes: Sequence[SearchNode], unexplored_floor: float,
                     home_here: Optional[str] = None, unexplored_elsewhere: float = UNEXPLORED_ELSEWHERE,
                     home_floor: float = 0.0,
                     ) -> Tuple[Dict[int, float], Tuple[int, ...], Tuple[int, ...]]:
    """Apply rule 2b's arithmetic to the unexplored nodes; return the new scores, the ids raised and the ids lowered.

    The floor and the elsewhere value are fractions in ``[0, 1)``; the
    scores are percentages. With ``home_here`` ``found`` or ``missing`` (or
    unknown) every unexplored node is raised to ``unexplored_floor`` and a
    node the model scored above it is left alone, so the model's ranking
    among the unexplored places survives wherever it valued them seriously.
    With ``home_here="elsewhere"`` -- the model's own reading that the
    target's home type lives on another storey -- every unexplored node is
    read at ``unexplored_elsewhere`` exactly: there is nothing the model can
    know about an unexplored place on this storey beyond that reading, and
    the 3B model writes the worked example's 25 on each of them regardless.
    0 disables the respective value.

    ``home_floor`` (since 2026-10-07) is the least a node with
    :attr:`SearchNode.home` is read at, whatever ``home_here`` says: a
    confirmed bathtub on this storey IS the toilet's home type, found,
    whatever the model wrote about the storey -- the Allensville toilet run
    valued the strong bathroom holding the bathtub at 0 for the toilet.
    """
    out = dict(scores)
    floored, capped = [], []
    if home_here == HOME_ELSEWHERE and unexplored_elsewhere > 0.0:
        percent = 100.0 * float(unexplored_elsewhere)
        for node in nodes:
            nid = int(node.id)
            if nid in out and node.unexplored and abs(out[nid] - percent) > 1e-9:
                (floored if out[nid] < percent else capped).append(nid)
                out[nid] = percent
        homed = []  # type: List[int]
        _floor_home(out, nodes, home_floor, homed)
        capped = [nid for nid in capped if nid not in homed]
        return out, tuple(sorted(set(floored + homed))), tuple(sorted(capped))
    if unexplored_floor > 0.0:
        percent = 100.0 * float(unexplored_floor)
        for node in nodes:
            nid = int(node.id)
            if nid in out and node.unexplored and out[nid] < percent - 1e-9:
                out[nid] = percent
                floored.append(nid)
    _floor_home(out, nodes, home_floor, floored)
    return out, tuple(sorted(set(floored))), ()


def _floor_home(out: Dict[int, float], nodes: Sequence[SearchNode], home_floor: float, floored: List[int]) -> None:
    """Raise every ``home`` node below ``home_floor`` to it, in place, noting the ids in ``floored``."""
    if not home_floor > 0.0:
        return
    percent = 100.0 * float(home_floor)
    for node in nodes:
        nid = int(node.id)
        if nid in out and node.home and out[nid] < percent - 1e-9:
            out[nid] = percent
            floored.append(nid)


def normalise(scores: Dict[int, float], elsewhere: float) -> Tuple[Dict[int, float], float]:
    """Convert each percentage independently; retain the old function signature.

    Old-model ``elsewhere`` is ignored: mixing categorical mass with RPT*'s
    multiplicative survival objective was inconsistent. No visit horizon is
    imposed. Dependencies between real rooms remain a modelling approximation.
    """
    probs = {nid: min(P_CEILING, max(0.0, value / 100.0)) for nid, value in scores.items()}
    failure = 1.0
    for value in probs.values():
        failure *= 1.0 - value
    return probs, failure


# -- the oracle -------------------------------------------------------------
def _accepts_reasoning(client) -> bool:
    """Does ``client.chat_json`` take the ``reasoning`` keyword (or ``**kwargs``)?

    Read from the signature once per call rather than by catching a
    ``TypeError`` from the call itself, which would also catch one raised
    inside the model request and silently repeat it on the default model.
    """
    try:
        parameters = inspect.signature(client.chat_json).parameters
    except (TypeError, ValueError):
        return True
    return "reasoning" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values())


class SearchNodeOracle:
    """Per-node find-probabilities over an LLM client, asked only when the prompt changed.

    Args:
        client: An :class:`~sparx_agency.core.mapping.topology.llm_client.LLMClient`
            or anything with ``chat_json(system, user, ...)``. When the client
            accepts ``reasoning=True`` the call is routed to the config's
            reasoning model; a plainer client is called without it.
        unexplored_floor: The least probability an unexplored node
            (:attr:`SearchNode.unexplored`) is read at, whatever the model
            wrote, when its STEP 2 says the home type is found or missing on
            this storey; 0 disables the floor. See the module docstring.
        unexplored_elsewhere: What an unexplored node is read at when the
            model's STEP 2 says the home type lives on another storey
            (``home_here="elsewhere"``); 0 leaves the model's numbers.
        home_floor: The least a node holding a confirmed home object of
            the target (:attr:`SearchNode.home`) is read at, whatever the
            model wrote and whatever its STEP 2 says; 0 disables.

    Attributes:
        reuses: Queries answered from the kept reply.
        queries: Calls actually made.
        unexplored_floor: As given.
        unexplored_elsewhere: As given.
        home_floor: As given.
    """

    def __init__(self, client, unexplored_floor: float = UNEXPLORED_FLOOR,
                 unexplored_elsewhere: float = UNEXPLORED_ELSEWHERE, home_floor: float = HOME_FLOOR) -> None:
        for name, value in (("unexplored_floor", unexplored_floor), ("unexplored_elsewhere", unexplored_elsewhere),
                            ("home_floor", home_floor)):
            if not (0.0 <= float(value) < 1.0) or not math.isfinite(float(value)):
                raise ValueError("%s must lie in [0, 1), got %r" % (name, value))
        self._client = client
        self._last_prompt = None  # type: Optional[str]
        self._last_reply = None   # type: Optional[Dict[str, Any]]
        self.unexplored_floor = float(unexplored_floor)
        self.unexplored_elsewhere = float(unexplored_elsewhere)
        self.home_floor = float(home_floor)
        self.reuses = 0
        self.queries = 0

    def prompt(self, target: str, nodes: Sequence[SearchNode], context: Optional[SearchContext] = None) -> str:
        """The user prompt for ``nodes`` -- everything the model is shown."""
        return format_prompt(target, nodes, context)

    def remember(self, prompt: str, reply: Dict[str, Any]) -> None:
        """Keep a usable reply as the answer to ``prompt``."""
        self._last_prompt, self._last_reply = prompt, reply

    def ask(self, user: str) -> Dict[str, Any]:
        """One call to the model, on the reasoning route when the client has one."""
        self.queries += 1
        if _accepts_reasoning(self._client):
            return self._client.chat_json(SYSTEM_PROMPT, user, reasoning=True)
        return self._client.chat_json(SYSTEM_PROMPT, user)

    def probabilities(self, target: str, nodes: Sequence[SearchNode],
                      context: Optional[SearchContext] = None) -> NodeOracleResult:
        """Return P(going there next finds ``target``) per node.

        Never raises on model trouble: a transport error or an unusable reply
        degrades to ``source='uniform_fallback'`` for the caller to refuse.
        An empty ``nodes`` sequence is a caller bug and raises ``ValueError``.
        """
        if not nodes:
            raise ValueError("SearchNodeOracle needs at least one node")
        user = self.prompt(target, nodes, context)
        if user == self._last_prompt and self._last_reply is not None:
            kept = self.score(self._last_reply, nodes, self.unexplored_floor, self.unexplored_elsewhere, self.home_floor)
            if kept is not None:
                self.reuses += 1
                return replace(kept, reused=True)
        try:
            reply = self.ask(user)
        except Exception:
            return self.uniform(nodes, raw_reply=None)
        result = self.score(reply, nodes, self.unexplored_floor, self.unexplored_elsewhere, self.home_floor)
        if result is None:
            return self.uniform(nodes, raw_reply=reply if isinstance(reply, dict) else None)
        self.remember(user, reply)
        return result

    # -- internals ------------------------------------------------------------
    @staticmethod
    def uniform(nodes: Sequence[SearchNode], raw_reply: Optional[Dict[str, Any]]) -> NodeOracleResult:
        share = 1.0 / (len(nodes) + 1)
        failure = (1.0 - share) ** len(nodes)
        return NodeOracleResult(probs={n.id: share for n in nodes}, elsewhere=failure, p_present=1.0 - failure,
                                source="uniform_fallback", reasons={n.id: "" for n in nodes},
                                raw_reply=raw_reply, spread=0.0)

    @staticmethod
    def score(reply: Any, nodes: Sequence[SearchNode], unexplored_floor: float = 0.0,
              unexplored_elsewhere: float = 0.0, home_floor: float = 0.0) -> Optional[NodeOracleResult]:
        """Parse, fill the omitted, apply rule 2b to the unexplored (floor, or the elsewhere value) and the home floor, normalise, clamp."""
        parsed = parse_reply(reply, nodes)
        if parsed is None:
            return None
        scores, elsewhere, reasons, omitted = parsed
        home_here = parse_home_here(reply)
        scores, floored, capped = floor_unexplored(scores, nodes, unexplored_floor, home_here, unexplored_elsewhere,
                                                   home_floor=home_floor)
        probs, elsewhere = normalise(scores, elsewhere)
        values = list(probs.values())
        reading = {key: str(reply[key])[:200] for key in ("home", "storey", "home_here")
                   if isinstance(reply, dict) and isinstance(reply.get(key), str) and reply[key].strip()}
        return NodeOracleResult(probs=probs, elsewhere=elsewhere, p_present=1.0 - elsewhere,
                                source="llm", reasons=reasons, raw_reply=reply if isinstance(reply, dict) else None,
                                spread=float(max(values) - min(values)) if values else 0.0, omitted=omitted,
                                reading=reading, floored=floored, capped=capped, home_here=home_here)










