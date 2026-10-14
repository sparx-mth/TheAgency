"""LLM node oracle: target + the nodes a search can go to next -> P(going there finds it).

The successor of :mod:`search_oracle` for the room-search loop. That module
was written around a 3B model: it asked one narrow question (how typical is
this object in a room of this type) and applied frontiers, search time and
room size in code, because the small model double-counted them into its
semantic judgement. This one is written for a capable model (the runtime
routes it to ``LLMConfig.reasoning_model``, a 14B instruct model) and asks
the WHOLE question at once, over every node the search could go to next:

* a **room** the map has segmented on the storey the robot stands on --
  identified by type from the objects seen in it, or still ``unknown`` --
  with its size, the frontier (unexplored boundary) left inside it, its
  **status** (never entered; entered but not looked around; scanned, and how
  long ago), how long it was searched and the objects confirmed in it;
* an **opening** -- a doorway or gap at the edge of the mapped floor leading
  to space not seen yet, valued by the room likely behind it;
* a **staircase** to another storey, when the search is allowed to change
  storeys (the standard single-storey benchmarks never offer one).

**LLM-first (since 2026-10-07).** The model is given the big picture and
asked to apply the knowledge a person has of homes: where the target
normally lives (a toilet in a bathroom -- a separate one or an en-suite off a
bedroom; a television in a living room, then a bedroom, rarely a kitchen,
never a bathroom); what the rooms found so far say about the rooms not yet
identified (a home has one kitchen and one living room, so once the living
room is found an unidentified room is unlikely to be another, and the sofa's
odds there drop; while no bathroom has been seen the small unidentified rooms
and the unlooked doorways are where the toilet must be, so exploration comes
first); and the **budget** -- early in the episode the target is most likely
somewhere not yet looked at, late in it the chance that the detector missed
it in a room already scanned rises while the chance that an unseen room holds
it falls, so a second look at the scanned rooms of the right type becomes
worth its actions. The model writes that judgement into the reply (``home``,
``house``, ``stage``) before any number, and a structured ``pass`` verdict --
``first`` while an unexplored place is still worth looking at before any
scanned room is re-checked, ``second`` when the house has been covered -- which
the loop reads to decide whether scanned rooms are offered as nodes at all.

The reply gives independent per-node search-success estimates, not mutually
exclusive target-location shares. The room-search loop hands these
probabilities without renormalisation to RPT*, whose objective (expected
time-to-find) weighs them against the travel cost the planner computes --
which is why the model is told, twice, not to reason about distance.

What stays in code is the contract, never the judgement: parse and clamp,
drop invented ids, give an omitted node a small share rather than zero,
derive joint failure under RPT*'s independence approximation, refuse a reply
with no usable node (the caller repairs the schema once, then backs off), and
reuse the last reply when the prompt is byte-identical -- the effort and
budget numbers are shown in coarse steps so an unchanged map produces an
unchanged prompt.

Three arithmetic floors remain available as knobs and are OFF by default
(``unexplored_floor``, ``unexplored_elsewhere``, ``home_floor``). They were
written for a 3B model that wrote off never-entered rooms at 0-1% and copied
the worked example's numbers (the Hanson and Ranchester recordings of
2026-10-04/05, see the git history of this file); a 14B model given the
prompt below applies the same rules itself, and a floor that overrides it
would be exactly the default value the search is meant to stop relying on.
The multi-storey supplement's ``home_here`` verdict is still parsed, so a
development run that lifts the stair ban can switch the floors back on.

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
#: an opening with nothing glimpsed through it). OFF: the model is asked to apply the
#: rule itself (see the module docstring); ``SearchNodeOracle(unexplored_floor=0.25)``
#: restores the 3B-era arithmetic.
UNEXPLORED_FLOOR = 0.0
#: What an unexplored node is read at when the multi-storey supplement's ``home_here``
#: verdict says the target's home type lives on another storey. OFF by default.
UNEXPLORED_ELSEWHERE = 0.0
#: The least a node holding a confirmed HOME OBJECT of the target (``SearchNode.home``) is
#: read at. OFF by default: the room's ``seen:`` line shows the bathtub to the model.
HOME_FLOOR = 0.0
#: The three answers to the multi-storey supplement's "is a room of the home type on this storey?".
HOME_FOUND, HOME_MISSING, HOME_ELSEWHERE = "found", "missing", "elsewhere"
HOME_HERE = (HOME_FOUND, HOME_MISSING, HOME_ELSEWHERE)
#: The two answers to "has the house been covered?": ``first`` while an unexplored place is
#: still worth a look before any scanned room is re-checked, ``second`` once it is not.
PASS_FIRST, PASS_SECOND = "first", "second"
PASSES = (PASS_FIRST, PASS_SECOND)
#: Actions are shown to the model in steps of this many, so the prompt does not change
#: on every action over an unchanged map.
BUDGET_STEP = 25

SYSTEM_PROMPT = """You are the reasoning module of a robot searching ONE home for an instance of \
ONE target object category, within a fixed budget of actions. The robot has partly mapped \
the home into NODES. For EACH node you estimate the probability, in percent, that going \
there NEXT finds the target: the target is there and the robot would see it by going and \
looking. You reason like a person who knows how homes are laid out and where things are kept.

NODE LINES.
ROOM: a region the robot segmented. type= was inferred from the objects seen in it \
(unknown = NOT IDENTIFIED YET, never "empty"; a type ending in "?" rests on a single kind \
of object and may be wrong -- a sink alone reads as a kitchen until a toilet shows). \
size= floor area. frontier= unexplored openings or boundaries still accessible inside it \
(0 is NOT proof that nothing is there). status= one of: never_entered (only seen from \
outside, or not at all); entered (the robot stood in it but never completed a look-around); \
scanned(<how>, <ago>) (the robot completed a full look-around in it, or saw most of it from \
one, <ago> ago -- what was visible then was seen ONCE by a detector that misses objects that \
are small, far, half-hidden, behind a door or under furniture). searched= time spent inside. \
seen= objects confirmed in it. here=yes: the robot stands in it now.
OPENING: a doorway or gap at the edge of the mapped floor leading to space NOT seen yet -- a \
room the robot has not entered. Going there is a quick look from the threshold, cheaper than a \
room visit. Value it by the room likely BEHIND it: glimpsed objects name that room (a toilet \
-> a bathroom; a bed -> a bedroom); nothing glimpsed -> an unidentified room, judged by what \
the home still lacks. An opening whose glimpsed objects name the home type of the target \
(a sink or a shower for a toilet; a sofa or a television for a couch) is the home room seen \
from its door: the best node there is, 60 or more.
The HOUSE line is the big picture: every room type found so far (scanned ones included), how \
many rooms are still unidentified, how many openings are not yet looked into, whether any \
unexplored frontier is still reachable, and the budget -- actions used of the total. Use ALL \
of it.

HOW TO JUDGE. Work in this order and write steps 1-3 into the reply before any number.
STEP 1 "home": the room types where this object normally lives, most likely first, and \
where it is rare. Typical homes: a toilet in a bathroom -- a separate one, or an en-suite \
reached through a bedroom; a bed in a bedroom; a sofa or couch in a living room (sometimes a \
family room or an office); a television in a living room, then a bedroom, rarely a kitchen, \
never a bathroom; a chair in a dining room, kitchen, office, bedroom or living room; a potted \
plant in a living room, hallway, kitchen or balcony; a refrigerator, oven or stove in a \
kitchen; a bathtub or shower in a bathroom. Small objects that travel (cup, book, bottle) \
have a home too, only a weaker one.
STEP 2 "house": read the home from the rooms found. A typical home has ONE kitchen, ONE \
living room, ONE dining area, one to four bedrooms, one to three bathrooms (one often \
en-suite off the main bedroom), a hallway, perhaps an office, a laundry or a garage. If a \
type that occurs once has been FOUND, an unidentified room is unlikely to be another of it, \
and the odds of the target's home being among the unidentified rooms drop. If the home type \
is still MISSING, the unidentified rooms and the openings are where it must be: value them \
in proportion to how well their size fits the home type (bathrooms ARE small, 3-8 m2; \
bedrooms 9-20 m2; living rooms 15 m2 and up) and to what else the home still lacks -- \
never write a room off for its size. A home-type object seen in a room of ANOTHER type (a \
sink in a "living room", a bed in a "kitchen") means the map merged two rooms: value that \
room as the home type.
STEP 3 "stage": weigh the budget. EARLY -- much budget left and unexplored places remain: \
the target is most likely somewhere not yet looked at; unexplored rooms and openings first, \
scanned rooms low. LATE -- most of the home seen, few or no unexplored places or frontier \
left, the home type found but the target not seen: the detector may have MISSED the target in a \
scanned room of a fitting type (a toilet behind a half-open door, a television on a wall \
the scan grazed, a chair under a table), and the chance that an unseen room holds it \
falls with every place looked at; raise the scanned rooms of the home type -- a second look \
from a different spot -- and lower the rest. Write "pass": "first" while an unexplored place \
in this home is still worth looking at before any scanned room is re-checked, else "second". \
The first pass over the home comes before the second, unless a scanned home-type room is \
the only candidate left.
STEP 4 the numbers, one per node, independently, integers 0-100:
 - a scanned room of a type the target does not live in: 0-2.
 - a scanned room of the home type with the target not seen: 3-10 early in the budget, \
15-40 late in it, less if it was scanned moments ago.
 - an entered but unscanned room: judge it by type and size like an unexplored one, a little lower.
 - a never-entered room of unknown type, or an opening with nothing glimpsed: never below 10 \
while the home type is missing; by size fit and by what the home lacks once it is found.
 - a never-entered room named as the home type by objects seen through its door, or an \
opening whose glimpse names the home type: 60 or more.
 - a room of the home type with frontier left, entered or not, target not seen: 30 or more.
 - a hallway or corridor rarely holds the target, but one with frontier LEADS to unseen \
rooms: value it by what it may lead to.
 - searched long and recently -> low, not zero; searched briefly or long ago -> stays promising.
 - SPREAD, do not copy: the unexplored places compete for the same missing rooms. A missing \
bathroom is ONE room, so five unlooked doorways share its chance -- about 10-15 each, the \
best-fitting one more, never 35 on every one of them; with many openings and one or two \
missing home-type rooms, most openings get under 15. Several openings often lead to the same \
hallway. The sum over the unexplored places should be about the chance that the target is \
somewhere not yet looked at, not more.
ALWAYS: NEVER use distance, travel time or here=yes as a reason -- the planner charges travel, \
and the room the robot stands in is judged exactly like the others. Several rooms may each \
hold an instance: the numbers are independent search-success probabilities, NOT shares of \
one distribution; do NOT make them sum to 100. Score EVERY node id given, and only those \
ids. Per node write "why" first (at most 12 words, lower case, naming the room's likely type \
and what decided it), then "p".
Reply with ONLY this JSON, keys in this order:
{"home":"<step 1, max 12 words>","house":"<step 2, max 25 words>","stage":"<step 3, max 20 words>",
 "pass":"first|second","nodes":[{"id":<int>,"why":"<why>","p":<int>}, ...]}
EXAMPLE of the FORMAT and the style of reasoning -- a different home every time, so never \
copy its numbers or its words. TARGET toilet; HOUSE: rooms found: kitchen (scanned), \
living_room (scanned), bedroom (never entered), hallway (entered); 1 room unidentified; 1 \
opening not yet looked into; actions used about 150 of 500:
id=2  ROOM  type=kitchen  size=12m2  frontier=0  status=scanned(full rotation, 2min ago)  searched=30s  seen: oven, refrigerator, sink
id=3  ROOM  type=living_room  size=24m2  frontier=1  status=scanned(full rotation, 1min ago)  searched=40s  seen: sofa, television
id=5  ROOM  type=bedroom  size=14m2  frontier=1  status=never_entered  searched=0s  seen: bed
id=6  ROOM  type=unknown  size=5m2  frontier=1  status=never_entered  searched=0s  seen: nothing yet
id=7  ROOM  type=hallway  size=6m2  frontier=1  status=entered  searched=10s  here=yes  seen: nothing yet
id=200001  OPENING  doorway off room 7 (type=hallway)  to space NOT seen yet  glimpsed through it: nothing yet
{"home":"bathroom; often an en-suite off a bedroom",
 "house":"kitchen and living room found and scanned; no bathroom yet; two unidentified rooms, one bathroom-sized; one unlooked doorway",
 "stage":"early, 150 of 500 used, unexplored places remain: first pass",
 "pass":"first",
 "nodes":[{"id":2,"why":"kitchen, scanned, toilets are not in kitchens","p":1},
{"id":3,"why":"living room, scanned, no toilet there","p":1},
{"id":5,"why":"bedroom never entered, an en-suite may open off it","p":25},
{"id":6,"why":"small unidentified room, bathroom sized, bathroom missing","p":55},
{"id":7,"why":"hallway, leads to the unlooked doorway","p":5},
{"id":200001,"why":"unseen room off the hallway, bathroom still missing","p":35}]}
Had the HOUSE line read "every known room identified; 0 openings not yet looked into; no \
unexplored frontier left; actions used about 420 of 500" with a bathroom found and scanned 4 \
minutes ago, the pass would be "second": the scanned bathroom 35, the scanned bedroom 8 (an \
en-suite door the scan may have missed), and the scanned kitchen and living room 1 each."""

#: Appended to :data:`SYSTEM_PROMPT` when a STAIRS node is offered: the multi-storey rules.
STAIRS_SUPPLEMENT = """
THIS SEARCH MAY CHANGE STOREYS. STAIRS nodes are staircases to another storey (up or down). \
Taking one means leaving this storey and searching the other one. The line says whether that \
storey was visited before and, if so, what was found and how much was searched there, and \
whether the robot arrived on this storey by these stairs a moment ago (arrived_by=yes). The \
THIS STOREY line gives this storey's rank in the building (the lowest of N known storeys, the \
highest, or how many lie below and above). Read the storey from the rooms found on it first \
-- a storey with a kitchen, dining or living room is the ground floor whatever its rank -- \
and from the rank where none has: of two storeys the lowest is the ground floor and the \
highest the bedroom floor; of three, the lowest is usually a basement, the middle the ground \
floor and the highest the bedroom floor. Where rooms live in a house: kitchen, dining and \
living room downstairs; bedrooms upstairs; the bathroom beside the bedrooms, so upstairs too \
(a ground floor has at most a small toilet room); an office on either. Add to the reply, \
before "nodes", the key "home_here": "found" (a room of the home type is on this storey), \
"missing" (none found yet, but the home type belongs on this storey) or "elsewhere" (the home \
type lives on another storey). STAIRS to an unvisited storey where the home type belongs can \
be high without suppressing the other nodes; stairs to a visited storey where the home type \
was FOUND and barely searched can be highly promising; arrived_by=yes: do not send the robot \
straight back unless it left that storey almost unsearched. An unexplored place on this \
storey is worth about 10 when home_here is "elsewhere" -- a look into it comes after the \
storey the target lives on. Changing storeys costs many actions, but that is NOT your \
concern: the planner adds travel and climbing cost itself."""

USER_PROMPT_TEMPLATE = """TARGET: {target}

HOUSE: {house}
{storeys}
NODES ({n}):
{nodes}

Score all {n} nodes independently. JSON only."""
STOREY_LINES_TEMPLATE = """THIS STOREY: {storey}
OTHER STOREYS: {others}
"""


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
            bathtub or a sink for a toilet, a television for a sofa). Read
            at the oracle's ``home_floor`` at least when that floor is on;
            the loop sets it from ``room_priors.HOME_OBJECTS``.
        scanned: How the room was finished by the scan ledger, in prompt
            words (``full rotation``, ``seen from another room's scan``,
            ``looked into, no frontier left``, ``fragment``), or None while
            it is not finished. Rendered as ``status=scanned(<how>, <ago>)``.
        scanned_ago_s: Seconds since that verdict, when known.
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
    scanned: Optional[str] = None
    scanned_ago_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("SearchNode.kind must be one of %s, got %r" % (KINDS, self.kind))

    @property
    def unexplored(self) -> bool:
        """Whether the search knows nothing about this place yet, so the (optional) floor applies.

        A room never entered, never scanned, whose type is still ``unknown``;
        an opening nothing has been glimpsed through. A staircase never is --
        the other storey is valued by what this one turned out to be -- and
        neither is an identified room or an opening with a glimpse, which
        the model values on the evidence.
        """
        if self.kind == ROOM:
            return (self.last_inside_ago_s is None and self.scanned is None
                    and (self.label or "unknown").strip().lower() == "unknown")
        if self.kind == OPENING:
            return not any(str(c).strip() for c in self.objects)
        return False

    @property
    def status(self) -> str:
        """The room's ``status=`` word for the prompt: ``never_entered``, ``entered`` or ``scanned(...)``."""
        if self.scanned:
            ago = coarse_seconds(self.scanned_ago_s)
            return "scanned(%s%s)" % (self.scanned, "" if ago == "never" else ", %s ago" % ago)
        return "never_entered" if self.last_inside_ago_s is None else "entered"


@dataclass(frozen=True)
class SearchContext:
    """What the model is told about the home around the nodes.

    Attributes:
        house: The big-picture line: every room type found so far with its
            status, the rooms still unidentified, the openings not yet
            looked into, and the budget (``actions used about 150 of 500``,
            see :func:`budget_line`). Empty reads as ``nothing known yet``.
        storey: Multi-storey only: one line on the storey the robot stands
            on -- rooms found by type, time searched, rooms with frontier
            left, its rank in the building. Shown when a STAIRS node is
            offered or another storey is known.
        others: Multi-storey only: one line per other storey known.
    """

    house: str = ""
    storey: str = ""
    others: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def multi_storey(self) -> bool:
        """Whether another storey is known to the search (the supplement's lines are shown)."""
        return bool(self.others)


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
        reading: The model's steps 1-3 in its own words -- ``home`` (where
            the target normally lives), ``house`` (what the rooms found say
            about the rest), ``stage`` (how the budget weighs), plus the
            multi-storey ``storey`` and ``home_here`` when given. Written
            before the numbers so the model applies its own reading; kept so
            the recording shows the judgement behind the distribution.
        floored: Node ids raised to the uncertainty floor (when it is on).
        capped: Node ids lowered to ``unexplored_elsewhere`` (when it is on).
        home_here: The multi-storey supplement's STEP verdict -- ``found``,
            ``missing`` or ``elsewhere`` -- or None when not given.
        pass_verdict: The model's ``pass`` -- :data:`PASS_FIRST` while an
            unexplored place is still worth looking at before any scanned
            room is re-checked, :data:`PASS_SECOND` once the home has been
            covered -- or None when not given. The loop reads it to decide
            whether scanned rooms are offered as nodes.
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
    floored: Tuple[int, ...] = ()
    capped: Tuple[int, ...] = ()
    home_here: Optional[str] = None
    pass_verdict: Optional[str] = None


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


def budget_line(actions_used: int, budget: int, step: int = BUDGET_STEP) -> str:
    """``actions used about 150 of 500`` -- the budget in :data:`BUDGET_STEP` steps, for an unchanged prompt.

    The used count is rounded DOWN to the step (``about 150`` for 150-174),
    never above what was actually spent; the budget itself is exact.
    """
    used = max(0, int(actions_used))
    total = max(1, int(budget))
    step = max(1, int(step))
    coarse = min(total, (used // step) * step)
    return "actions used about %d of %d" % (coarse, total)


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
             "status=%s" % node.status,
             "searched=%s" % coarse_seconds(node.searched_s)]
    if node.here:
        parts.append("here=yes")
    parts.append("seen: %s" % seen)
    return "  ".join(parts)


def format_prompt(target: str, nodes: Sequence[SearchNode], context: Optional[SearchContext] = None) -> str:
    """Everything the model is shown for this query."""
    context = context or SearchContext()
    storeys = ""
    if context.multi_storey or any(n.kind == STAIRS for n in nodes):
        storeys = STOREY_LINES_TEMPLATE.format(
            storey=context.storey or "the only storey mapped so far",
            others="; ".join(context.others) if context.others else "none known")
    return USER_PROMPT_TEMPLATE.format(
        target=target, house=context.house or "nothing known yet", storeys=storeys,
        n=len(nodes), nodes="\n".join(format_node(n) for n in nodes))


def system_prompt(nodes: Sequence[SearchNode], context: Optional[SearchContext] = None) -> str:
    """The system prompt for this query: the single-storey rules, plus the stairs supplement when a storey change is on offer."""
    if any(n.kind == STAIRS for n in nodes) or (context is not None and context.multi_storey):
        return SYSTEM_PROMPT + STAIRS_SUPPLEMENT
    return SYSTEM_PROMPT


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
    """The multi-storey ``home_here`` verdict as one of :data:`HOME_HERE`, or None when absent or unreadable.

    The three words are what the supplement asks for; a few phrasings a
    model drifts to are accepted, the longest match first so "not found" is
    ``missing`` and not ``found``. A negated "found" ("none found", "no
    room ... found") is ``missing``; a bare "no" is ``missing`` too -- the
    conservative verdict -- never ``elsewhere``, which lowers every
    unexplored place on the storey when that floor is on.
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


_PASS_WORDS = {
    PASS_SECOND: ("second", "2nd", "re-check", "recheck", "revisit", "covered", "again"),
    PASS_FIRST: ("first", "1st", "explore", "unexplored"),
}


def parse_pass(reply: Any) -> Optional[str]:
    """The model's ``pass`` verdict as :data:`PASS_FIRST` or :data:`PASS_SECOND`, or None when absent or unreadable.

    The two words are what the prompt asks for; a few phrasings a model
    drifts to are accepted, ``second`` tested first so "second, not first"
    is read as second. A reply without the key is None: the loop then keeps
    scanned rooms out of the nodes as the first pass does -- the
    conservative reading, which never re-enters a room on a missing word.
    """
    if not isinstance(reply, dict):
        return None
    value = reply.get("pass")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return PASS_SECOND if int(value) >= 2 else PASS_FIRST
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower()
    for verdict in PASSES:
        if text == verdict:
            return verdict
    for verdict in (PASS_SECOND, PASS_FIRST):
        if any(word in text for word in _PASS_WORDS[verdict]):
            return verdict
    return None


def floor_unexplored(scores: Dict[int, float], nodes: Sequence[SearchNode], unexplored_floor: float,
                     home_here: Optional[str] = None, unexplored_elsewhere: float = UNEXPLORED_ELSEWHERE,
                     home_floor: float = 0.0,
                     ) -> Tuple[Dict[int, float], Tuple[int, ...], Tuple[int, ...]]:
    """Apply the optional arithmetic floors to the unexplored and home nodes; return the new scores, the ids raised and the ids lowered.

    The floor and the elsewhere value are fractions in ``[0, 1)``; the
    scores are percentages. With ``home_here`` ``found`` or ``missing`` (or
    unknown) every unexplored node is raised to ``unexplored_floor`` and a
    node the model scored above it is left alone. With
    ``home_here="elsewhere"`` -- the multi-storey reading that the target's
    home type lives on another storey -- every unexplored node is read at
    ``unexplored_elsewhere`` exactly. ``home_floor`` is the least a node
    with :attr:`SearchNode.home` is read at. 0 disables the respective
    value; all three are 0 by default (see the module docstring).
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
            wrote; 0 (the default) leaves the model's numbers.
        unexplored_elsewhere: What an unexplored node is read at when the
            multi-storey supplement's verdict says the home type lives on
            another storey; 0 (the default) leaves the model's numbers.
        home_floor: The least a node holding a confirmed home object of
            the target (:attr:`SearchNode.home`) is read at; 0 (the
            default) leaves the model's numbers.

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
        """The user prompt for ``nodes`` -- everything the model is shown beside the system prompt."""
        return format_prompt(target, nodes, context)

    def remember(self, prompt: str, reply: Dict[str, Any]) -> None:
        """Keep a usable reply as the answer to ``prompt``."""
        self._last_prompt, self._last_reply = prompt, reply

    def ask(self, user: str, system: str = SYSTEM_PROMPT) -> Dict[str, Any]:
        """One call to the model, on the reasoning route when the client has one."""
        self.queries += 1
        if _accepts_reasoning(self._client):
            return self._client.chat_json(system, user, reasoning=True)
        return self._client.chat_json(system, user)

    def probabilities(self, target: str, nodes: Sequence[SearchNode],
                      context: Optional[SearchContext] = None) -> NodeOracleResult:
        """Return P(going there next finds ``target``) per node.

        Never raises on model trouble: a transport error or an unusable reply
        degrades to ``source='uniform_fallback'`` for the caller to refuse.
        An empty ``nodes`` sequence is a caller bug.
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
            reply = self.ask(user, system_prompt(nodes, context))
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
        """Parse, fill the omitted, apply the optional floors, normalise, clamp; read the verdicts."""
        parsed = parse_reply(reply, nodes)
        if parsed is None:
            return None
        scores, elsewhere, reasons, omitted = parsed
        home_here = parse_home_here(reply)
        scores, floored, capped = floor_unexplored(scores, nodes, unexplored_floor, home_here, unexplored_elsewhere,
                                                   home_floor=home_floor)
        probs, elsewhere = normalise(scores, elsewhere)
        values = list(probs.values())
        reading = {key: str(reply[key])[:200] for key in ("home", "house", "stage", "pass", "storey", "home_here")
                   if isinstance(reply, dict) and isinstance(reply.get(key), str) and reply[key].strip()}
        return NodeOracleResult(probs=probs, elsewhere=elsewhere, p_present=1.0 - elsewhere,
                                source="llm", reasons=reasons, raw_reply=reply if isinstance(reply, dict) else None,
                                spread=float(max(values) - min(values)) if values else 0.0, omitted=omitted,
                                reading=reading, floored=floored, capped=capped, home_here=home_here,
                                pass_verdict=parse_pass(reply))
