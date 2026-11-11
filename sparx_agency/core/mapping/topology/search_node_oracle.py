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
  **exits** (the doorways and gaps at its edge leading to space not seen
  yet, how many stand at a seen door frame, and what was glimpsed through
  them -- since 2026-10-21 the room's attributes, not nodes of their own),
  its **status** (never entered; entered but not looked around; scanned, and
  how long ago), how long it was searched and the objects confirmed in it;
* a **staircase** to another storey, when the search is allowed to change
  storeys (the standard single-storey benchmarks never offer one);
* an **opening** -- a doorway or gap at the edge of the mapped floor leading
  to space not seen yet, valued by the room likely behind it -- only under
  the deprecated exits-as-nodes mode (``OpeningSettings.exits_as_nodes``),
  which brings :data:`OPENINGS_SUPPLEMENT` into the system prompt. The
  room-level design values a room by its type AND its exits: a high exit
  count raises a room only where its type fits the target's home (a
  bathroom's unlooked door for a toilet, never a living room's gaps), and
  the room's visit walks its exits itself.

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

**Size and coverage, final results only (since 2026-11-11).** The code
does the arithmetic and the model is handed the results. The room line
carries, in the node's order: the type; the size from the room extents
(``core/mapping/topology/room_extent.py``) -- FINAL (``12m2 (4x3m, all
walls seen)``) only when every wall is located on the depth-derived map, a
LOWER BOUND otherwise (``at least 12m2 (5+ x 3m, 1 wall not seen yet)``,
the ``+`` marking the dimension the camera's reach cut short); the coverage
-- the floor the camera has looked at from within detection range, in m2,
as a share of the WHOLE room and as what is left (``covered=10m2 (80% of
the room, 2m2 left)``), the share an upper bound and the remainder a lower
one when the size is (``covered=3m2 (at most 40% of the room, at least 4m2
left)``); the objects seen; the frontier count with the unknown floor
behind the clusters, united (``frontier=2 (~30m2 unknown behind them)``)
and the exits with their door frames; the status; the time spent. The
rules put COVERAGE FIRST -- the main fact for how much is left, the size
saying what the remainder can hold, the model judging on its own whether
the target fits -- and TIME SECOND: coverage overrides ``searched=`` both
ways. Where the robot stands (``here=yes`` until then) and every distance
are left out of the prompt: the planner charges travel. The agent's
distance to each wall is in the node record for the review alone.

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
#: The ``NodeOracleResult.source`` values the runtime flies on: the model's own numbers, or the flat
#: ablation's (``SearchNodeOracle(flat_p=...)``). ``uniform_fallback`` -- a model that failed -- is refused.
SOURCE_LLM, SOURCE_FLAT = "llm", "flat"
USABLE_SOURCES = (SOURCE_LLM, SOURCE_FLAT)

SYSTEM_PROMPT = """You are the reasoning module of a robot searching ONE home for an instance of \
ONE target object category, within a fixed budget of actions. The robot has partly mapped \
the home into NODES. For EACH node you estimate the probability, in percent, that the target \
is there -- in that room, where the robot would see it by going there and looking. You reason \
like a person who knows how homes are laid out and where things are kept, and how big the \
target is: every number on a node line is a FINAL result computed for you; read it, do not \
recompute it.

NODE LINES.
ROOM: a region the robot segmented. type= was inferred from the objects seen in it \
(unknown = NOT IDENTIFIED YET, never "empty"; a type ending in "?" rests on a single kind \
of object and may be wrong -- a sink alone reads as a kitchen until a toilet shows). \
size= the room's floor area, read from the walls the depth camera has located: "12m2 (4x3m, \
all walls seen)" is FINAL -- the room is that big and no bigger; "at least 12m2 (5+ x 3m, 1 \
wall not seen yet)" is a LOWER BOUND -- a wall is still beyond the camera's reach or hidden, \
the "+" marks the dimension cut short, and the room may be much bigger than the part mapped \
so far (a living room seen from its door reads "at least 6m2" until its far wall is seen). \
covered= how much of the room's floor the camera has looked at from within detection range \
(about 4 m) -- where the detector has had its chance, NOT where the robot walked -- in m2, as \
a share of the WHOLE room, and what is LEFT: "10m2 (80% of the room, 2m2 left)". When the \
size is a lower bound the share is an UPPER bound and the remainder a lower one: "3m2 (at \
most 40% of the room, at least 4m2 left)" -- the rest of the room is neither mapped nor \
covered, and "all of the mapped part; the unmapped rest is unknown" means everything mapped \
was looked at and the room goes on beyond its unseen walls. seen: the objects confirmed in \
it. frontier= its unexplored boundaries still accessible (0 is NOT proof that nothing is \
there), and in brackets about how much unknown floor lies behind them -- what the camera's \
rays from them reach into the unknown: with walls not seen yet that unknown is mostly the \
rest of this room, with all walls seen it lies beyond its exits, in rooms the robot has not \
entered. exits= how many of those boundaries are doorways or gaps at its edge leading to \
space NOT seen yet, and how many stand at a seen door frame (exits=2 (1 doorway)). They are \
NOT nodes of their own: going to the room means walking its exits and looking through each. \
glimpsed through its exits: objects seen beyond those thresholds -- they name the rooms \
BEHIND the exits (a toilet -> a bathroom; a bed -> a bedroom), not this room. status= one \
of: never_entered (only seen from outside, or not at all); entered (the robot stood in it \
but never completed a look-around: what is left of it is its uncovered part and what lies \
behind its frontier= boundaries); scanned(<how>, <ago>) (the robot completed a full \
look-around in it, or saw most of it from one, <ago> ago -- what was visible then was seen \
ONCE by a detector that misses objects that are small, far, half-hidden, behind a door or \
under furniture; a scanned room with exits= left is offered again for what lies BEHIND them, \
not for itself). searched= time the robot has spent inside it. Where the robot stands now and \
how far each room is are NOT shown: the planner charges travel; you judge each room on what \
it is and what is left of it.
The HOUSE line is the big picture: every room type found so far (scanned ones included), how \
many rooms are still unidentified, how many exits (openings) of the whole home are not yet \
looked into, whether any unexplored frontier is still reachable -- or only slivers of it, or frontier that no route \
from here reaches: then the home is NOT covered, the robot is cut off from the rest of it -- and the budget -- actions \
used of the total. Use ALL of it.

HOW MUCH IS LEFT -- COVERAGE FIRST. covered= is the MAIN fact for how much of a room is \
still to be searched; the size says what that remainder can hold. You know how big the target \
is: judge whether it fits in what is LEFT. 2m2 left of a 12m2 bedroom can hide a cup, a book \
or a chair, not a bed or a sofa; "at least 14m2 left" of a room of "at least 20m2" can hold \
anything, and the room may be far bigger. A room with all walls seen and covered 90% or \
more, the target not seen, is as good as scanned -- almost nothing is left to look at. A \
room of "at least" size, or covered under half, still holds most of itself unseen and stays \
promising. frontier= says where the uncovered part is reached from and how much unknown lies \
behind it: frontiers onto a large unknown lead to the rest of the room or to rooms not found \
yet; frontiers onto a few m2 lead nowhere new.
TIME IS SECONDARY. searched= is read only beside covered=, and coverage overrides it: many \
actions in a big room at low coverage take NOTHING from it -- a big room reasonably takes \
many actions while much of it is left -- and a few actions in a room already covered are \
enough to give it up. Never lower a room for its searched= alone, and never keep a covered \
room up because it was searched briefly.

EXITS. What lies behind a room's exits is decided by the HOME, never by the count alone: an \
exit leads to a room of this home, and the rooms the home still lacks say which. Exits RAISE \
a room ONLY when its type fits the target -- a bathroom with an unlooked door (a second \
bathroom or an en-suite beyond it) when the target is a toilet; a bedroom whose exit may be \
an en-suite's; a kitchen's exits for a refrigerator -- or when a glimpse through one of them \
names the home type (a toilet glimpsed through an exit of the hallway: that room is the \
bathroom seen from its door, 60 or more). Exits of a hallway, a corridor or an unidentified \
room are valued by what the home still lacks behind them: while the bathroom is missing, \
they are where it must be. Exits of a room whose type does NOT fit the target do NOT raise \
it, however many: three gaps off a living room, for a toilet, lead to strips of the living \
room or to rooms already found, and the living room stays low. A room with many exits and no \
fit is no more promising than one with none.

HOW TO JUDGE. Work in this order and write steps 1-3 into the reply before any number.
STEP 1 "home": the room types where this object normally lives, most likely first, and \
where it is rare. Typical homes: a toilet in a bathroom -- a separate one, or an en-suite \
reached through a bedroom; a bed in a bedroom; a sofa or couch in a living room (sometimes a \
family room or an office); a television in a living room, then a bedroom, rarely a kitchen, \
never a bathroom; a chair in a dining room, kitchen, office, bedroom or living room; a potted \
plant in a living room, hallway, kitchen or balcony; a refrigerator, oven or stove in a \
kitchen; a bathtub or shower in a bathroom. Small objects that travel (cup, book, bottle) \
have a home too, only a weaker one. When the prompt carries a WHERE IT LIVES line under the \
TARGET, that line IS step 1 -- the room types it names, in its order, with its rare ones rare: \
copy it into "home" and reason from it; never replace it with a guess of your own.
STEP 2 "house": read the home from the rooms found. A typical home has ONE kitchen, ONE \
living room, ONE dining area, one to four bedrooms, one to three bathrooms (one often \
en-suite off the main bedroom), a hallway, perhaps an office, a laundry or a garage. If a \
type that occurs once has been FOUND, an unidentified room is unlikely to be another of it, \
and the odds of the target's home being among the unidentified rooms drop. If the home type \
is still MISSING, the unidentified rooms and the unlooked exits are where it must be: value \
them in proportion to how well their size fits the home type (bathrooms ARE small, 3-8 m2; \
bedrooms 9-20 m2; living rooms 15 m2 and up) and to what else the home still lacks -- \
never write a room off for its size, and never read a size of "at least" as the room's \
size: a room with walls not seen yet may be any type its lower bound allows. A home-type \
object seen in a room of ANOTHER type (a sink in a "living room", a bed in a "kitchen") \
means the map merged two rooms: value that room as the home type.
STEP 3 "stage": weigh the budget. EARLY -- much budget left and unexplored places remain: \
the target is most likely somewhere not yet looked at; unexplored rooms and the rooms whose \
exits lead to them first, scanned rooms low. LATE -- most of the home seen, few or no \
unexplored places or frontier \
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
 - an entered but unscanned room: value only what is LEFT of it -- its covered= remainder \
and the unknown behind its frontier= -- at most HALF of what an unexplored room of the same \
type and size would get; under 10 when all walls are seen and 80% or more is covered. All \
walls seen and 90% or more covered, the target not seen, is as good as scanned: 0-5 whatever \
its type and whatever searched= reads. A room of "at least" size, or covered under half, keeps \
its value whatever searched= reads. Never write "known to contain" the target of a room the \
target was not seen in.
 - a never-entered room of unknown type: never below 15 while the home type is missing; by \
size fit and by what the home lacks once it is found. Its frontier= count measures the MAP, \
not the room: a never-entered room with frontier=0 or 1 is as unexplored as one with five -- \
clearing one boundary of an unentered room opens the whole of it and what lies beyond. \
Never lower a never-entered room for a low frontier count.
 - a never-entered room named as the home type by objects seen through its door, or a room \
whose exits' glimpse names the home type behind them: 60 or more.
 - a never-entered room of the home type with frontier left, target not seen: 30 or more; the \
same room entered: 10-20, more only if much of it is left (covered under half, or walls not \
seen yet).
 - prefer the parts of the home NOT looked at: a room with none of its neighbours entered or \
scanned, or one whose exits lead away from everything entered and scanned, leads to a part \
of the home not seen at all; a gap at the edge of a scanned room usually shows a strip of it.
 - a hallway or corridor rarely holds the target, but one with exits LEADS to unseen rooms: \
value it by what the home still lacks behind them.
 - a scanned room re-offered late: the longer ago its scan, the more a second look from \
another spot is worth; scanned moments ago -> low, not zero.
 - SPREAD, do not copy: the unexplored rooms and the exits of the rooms found compete for the \
same missing rooms. A missing bathroom is ONE room, so five unlooked exits spread over three \
rooms share its chance -- the room whose type and exits fit it best gets most, the others \
about 10-15, never 35 on every room with an exit; with many exits and one or two missing \
home-type rooms, most such rooms get under 15. Several exits often lead to the same hallway. \
The sum over the unexplored places should be about the chance that the target is somewhere \
not yet looked at, not more.
ALWAYS: NEVER use distance or travel time as a reason -- they are not shown, and the planner \
charges travel itself. Several rooms may each hold an instance: the numbers are independent \
probabilities, NOT shares of one distribution; do NOT make them sum to 100. Score EVERY node \
id given, and only those ids. Per node write "why" first (at most 12 words, lower case, \
naming the room's likely type and what decided it), then "p".
Reply with ONLY this JSON, keys in this order:
{"home":"<step 1, max 12 words>","house":"<step 2, max 25 words>","stage":"<step 3, max 20 words>",
 "pass":"first|second","nodes":[{"id":<int>,"why":"<why>","p":<int>}, ...]}
EXAMPLE of the FORMAT and the style of reasoning -- a different home every time, so never \
copy its numbers or its words. TARGET toilet; HOUSE: rooms found: kitchen (scanned), \
living_room (scanned), bedroom (never entered), hallway (entered); 1 room unidentified; 3 \
openings not yet looked into; actions used about 150 of 500:
id=2  ROOM  type=kitchen  size=12m2 (4x3m, all walls seen)  covered=11m2 (90% of the room, 1m2 left)  seen: oven, refrigerator, sink  frontier=0  exits=0  status=scanned(full rotation, 2min ago)  searched=30s
id=3  ROOM  type=living_room  size=24m2 (6x4m, all walls seen)  covered=20m2 (80% of the room, 4m2 left)  seen: sofa, television  frontier=1 (~2m2 unknown behind it)  exits=1  status=scanned(full rotation, 1min ago)  searched=40s
id=5  ROOM  type=bedroom  size=at least 9m2 (3.5+ x 3m, 1 wall not seen yet)  covered=4m2 (at most 50% of the room, at least 5m2 left)  seen: bed  frontier=1 (~12m2 unknown behind it)  exits=1 (1 doorway)  status=never_entered  searched=0s
id=6  ROOM  type=unknown  size=at least 3m2 (2+ x 1.5+m, 2 of 4 walls not seen yet)  covered=1m2 (at most 40% of the room, at least 2m2 left)  seen: nothing yet  frontier=1 (~6m2 unknown behind it)  exits=0  status=never_entered  searched=0s
id=7  ROOM  type=hallway  size=6m2 (4x1.5m, all walls seen)  covered=4m2 (70% of the room, 2m2 left)  seen: nothing yet  frontier=1 (~15m2 unknown behind it)  exits=1 (1 doorway)  status=entered  searched=10s
{"home":"bathroom; often an en-suite off a bedroom",
 "house":"kitchen and living room found and scanned; no bathroom yet; one unidentified room, at least 3m2 with two walls unseen, could be a bathroom; unlooked doorways off the bedroom and the hallway",
 "stage":"early, 150 of 500 used, unexplored places remain: first pass",
 "pass":"first",
 "nodes":[{"id":2,"why":"kitchen, scanned, 1m2 left, toilets are not in kitchens","p":1},
{"id":3,"why":"living room, scanned, 4m2 left; its gap is no bathroom door","p":2},
{"id":5,"why":"bedroom never entered, at least 5m2 unseen, its doorway may be an en-suite's","p":35},
{"id":6,"why":"small unidentified room, mostly unseen, bathroom sized so far, bathroom missing","p":50},
{"id":7,"why":"hallway, 2m2 left of it; its unlooked doorway leads to the missing bathroom","p":25}]}
Had the HOUSE line read "every known room identified; 0 openings not yet looked into; no \
unexplored frontier left; actions used about 420 of 500" with a bathroom found and scanned 4 \
minutes ago, the pass would be "second": the scanned bathroom 35, the scanned bedroom 8 (an \
en-suite door the scan may have missed), and the scanned kitchen and living room 1 each."""

#: Appended to :data:`SYSTEM_PROMPT` when an OPENING node is offered -- the deprecated
#: exits-as-nodes mode (``OpeningSettings.exits_as_nodes``, the 2026-10-04 design): an exit
#: shown as a node of its own is valued by the room likely behind it and visited by a peek.
OPENINGS_SUPPLEMENT = """
OPENING NODES ARE ON OFFER. OPENING: a doorway or gap at the edge of the mapped floor leading \
to space NOT seen yet -- a room the robot has not entered -- offered as a node of its own \
(it is then not counted among its room's exits=). Going there is a quick look from the \
threshold, cheaper than a room visit. Value it by the room likely BEHIND it: glimpsed objects \
name that room (a toilet -> a bathroom; a bed -> a bedroom); nothing glimpsed -> an \
unidentified room, judged by what the home still lacks. An opening whose glimpsed objects \
name the home type of the target (a sink or a shower for a toilet; a sofa or a television for \
a couch) is the home room seen from its door: the best node there is, 60 or more. Openings \
compete with the unexplored rooms for the same missing rooms (SPREAD applies to them too). \
Line format: id=200001  OPENING  doorway off room 7 (type=hallway)  to space NOT seen yet  \
glimpsed through it: nothing yet -- in the example above, with the hallway's doorway offered \
so, the opening would take the hallway's 25 and the hallway itself 5."""

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

USER_PROMPT_TEMPLATE = """TARGET: {target}{home}

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
        exits: Room: the doorways and gaps at its edge leading to space not
            seen yet -- its open GAPs, read off the live map by
            ``opening_nodes.detect_openings`` and credited to the room they
            open from (since 2026-10-21 the room's attribute, not nodes of
            their own: the room's visit walks them). 0 for the other kinds.
        doorways: Room: how many of those exits stand at a confirmed door
            frame. Rendered as ``exits=2 (1 doorway)``.
        glimpsed: Room: object classes seen THROUGH its exits, beyond their
            thresholds -- they name the rooms behind the exits, not this
            one. Rendered as ``glimpsed through its exits: ...``.
        unseen_m2: Room: unknown floor the rays from its exits' thresholds
            reach, summed (since 2026-10-22). For the record since
            2026-11-11: the room line carries ``unknown_behind_m2`` (one
            united figure over all its frontiers) in the ``frontier=``
            brackets instead, so one unmapped room is not counted per exit.
        footprint_m2: Room: the floor area of its footprint -- the mapped
            free cells plus the islands they enclose (the floor under a bed
            or a table) -- from the room extents (since 2026-11-11,
            ``core/mapping/topology/room_extent.py``); 0 when not read, and
            ``area_m2`` is shown instead.
        length_m: Room: its extent along its longer axis, metres (0: unread).
        width_m: Room: its extent along its shorter axis, metres.
        length_final: Room: both walls across the longer axis are located;
            False renders the dimension with a ``+`` (a lower bound).
        width_final: Room: the same across the shorter axis.
        walls_seen: Room: how many of its four sides are located -- a wall,
            a door cut or settled unknown along at most ``open_tolerance``
            of the side's faces open to unseen space in runs wider than a
            doorway. None when not read.
        bounded: Room: all four walls located, so the size is FINAL;
            False renders ``size=at least ...`` with the count of walls not
            seen; None when not read (the size is shown as before).
        covered_fraction: Room: the share of the WHOLE room the camera has
            covered from within detection range (the sight ledger's per-cell
            record, furniture islands credited from the floor around them),
            0-1 -- exact when ``bounded``, an UPPER bound when the size is a
            lower bound. Rendered with ``covered_m2`` and ``left_m2`` as
            ``covered=10m2 (80% of the room, 2m2 left)`` or ``covered=3m2
            (at most 40% of the room, at least 4m2 left)`` (tens; down for
            a share, up for a bound). None when not read: nothing is rendered.
        covered_m2: Room: the covered floor, square metres.
        left_m2: Room: the floor still to be looked at -- exact when
            bounded, a lower bound when not.
        unknown_behind_m2: Room: the unknown floor the camera's cone
            reaches from its frontier clusters, united (since 2026-11-11,
            ``frontier_unknown.unknown_behind``). Rendered in the
            ``frontier=`` brackets: ``frontier=2 (~30m2 unknown behind
            them)``. None when not measured: the count alone is shown.
        open_perimeter_m: Room: metres of its boundary open to unseen space
            in runs wider than a doorway -- where the camera's reach ended
            inside the room. For the record; not rendered.
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
    exits: int = 0
    doorways: int = 0
    glimpsed: Tuple[str, ...] = field(default_factory=tuple)
    unseen_m2: float = 0.0
    footprint_m2: float = 0.0
    length_m: float = 0.0
    width_m: float = 0.0
    length_final: bool = True
    width_final: bool = True
    walls_seen: Optional[int] = None
    bounded: Optional[bool] = None
    covered_fraction: Optional[float] = None
    covered_m2: float = 0.0
    left_m2: float = 0.0
    unknown_behind_m2: Optional[float] = None
    open_perimeter_m: float = 0.0

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
        target_home: Where the target normally lives, in prompt words
            (:func:`~room_priors.target_home_line`: ``lives in: bathroom,
            kitchen (most likely first); rare in: bedroom``); rendered as a
            ``WHERE IT LIVES`` line under the TARGET, which the system prompt
            makes step 1. Empty leaves step 1 to the model (since 2026-11-09).
    """

    house: str = ""
    storey: str = ""
    others: Tuple[str, ...] = field(default_factory=tuple)
    target_home: str = ""

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
        source: ``'llm'`` for a usable reply, ``'flat'`` for the flat-probability
            ablation (every node at ``SearchNodeOracle.flat_p``, no model asked),
            ``'uniform_fallback'`` when the model failed and a flat distribution
            was substituted. The runtime refuses to fly on the latter.
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
    label = node.label + ("?" if node.tentative and node.label != "unknown" else "")
    # The room's facts in the order the node holds them: type, size, coverage, objects, frontier and exits
    # with the unknown behind them, status, time spent. Where the robot stands is not shown: the planner
    # charges travel, and the room it stands in is judged like the others.
    parts = ["id=%d" % node.id, "ROOM", "type=%s" % label, "size=%s" % format_size(node)]
    coverage = format_coverage(node)
    if coverage:
        parts.append("covered=%s" % coverage)
    parts += ["seen: %s" % seen,
              "frontier=%s" % format_frontier(node.frontier_clusters, node.unknown_behind_m2),
              "exits=%s" % format_exits(node.exits, node.doorways),
              "status=%s" % node.status,
              "searched=%s" % coarse_seconds(node.searched_s)]
    glimpsed = sorted({str(c).strip().lower() for c in node.glimpsed if str(c).strip()})
    if glimpsed:
        parts.append("glimpsed through its exits: %s" % ", ".join(glimpsed[:MAX_CLASSES_IN_PROMPT]))
    return "  ".join(parts)


def coarse_metres(value: float, step: float = 0.5) -> str:
    """Metres in ``step`` steps (``3.5``; ``4`` for a whole number), so an unchanged map gives an unchanged prompt."""
    rounded = round(float(value) / step) * step
    text = "%.1f" % rounded
    return text[:-2] if text.endswith(".0") else text


def format_size(node: SearchNode) -> str:
    """The ``size=`` value of a room line.

    With the room extents read (``bounded`` not None): ``12m2 (3.5x3.5m,
    all walls seen)`` when every wall is located, else ``at least 12m2
    (5+ x 3.5m, 1 wall not seen yet)`` -- the ``+`` marks a dimension the
    camera's reach cut short, and the area is what is mapped so far, never
    presented as the room's. Without them: the area alone, as before
    (``12m2``, ``under 1m2``, ``unknown size``).
    """
    area = max(float(node.footprint_m2), float(node.area_m2))
    if node.bounded is None:
        return "unknown size" if area <= 0.0 else "under 1m2" if area < 1.0 else "%dm2" % int(round(area))
    length = coarse_metres(node.length_m) + ("" if node.length_final else "+")
    width = coarse_metres(node.width_m) + ("" if node.width_final else "+")
    if node.bounded:
        area_text = "under 1m2" if area < 1.0 else "%dm2" % int(round(area))
        return "%s (%sx%sm, all walls seen)" % (area_text, length, width)
    missing = 4 - int(node.walls_seen or 0)
    walls = "1 wall not seen yet" if missing == 1 else "%d of 4 walls not seen yet" % missing
    if area < 1.0:
        return "under 1m2 mapped so far (%s x %sm, %s)" % (length, width, walls)
    return "at least %dm2 (%s x %sm, %s)" % (int(round(area)), length, width, walls)


def area_words(value: float) -> str:
    """Square metres for the prompt: ``0m2``, ``under 1m2``, else the nearest whole ``12m2``."""
    value = float(value)
    if value < 0.05:
        return "0m2"
    if value < 1.0:
        return "under 1m2"
    return "%dm2" % int(round(value))


def format_coverage(node: SearchNode) -> Optional[str]:
    """The ``covered=`` value of a room line: the floor the camera has looked at from detection range, as final results.

    A bounded room: ``10m2 (80% of the room, 2m2 left)`` -- the share of
    the whole room rounded DOWN to tens, and what is left -- or ``12m2
    (all of it)``. A room whose size is a lower bound: the share is an
    UPPER bound rounded UP and the remainder a LOWER bound, ``3m2 (at most
    40% of the room, at least 4m2 left)``; with everything mapped covered,
    ``7m2 (all of the mapped part; the unmapped rest is unknown)``. The
    model is handed the numbers it would otherwise derive. None when the
    record was not read.
    """
    if node.covered_fraction is None:
        return None
    covered = area_words(node.covered_m2)
    fraction = max(0.0, min(1.0, float(node.covered_fraction)))
    if node.bounded:
        if fraction >= 0.995:
            return "%s (all of it)" % covered
        percent = int(math.floor(fraction * 10.0 + 1e-9)) * 10
        return "%s (%d%% of the room, %s left)" % (covered, percent, area_words(node.left_m2))
    percent = min(100, int(math.ceil(fraction * 10.0 - 1e-9)) * 10)
    if percent >= 100:
        # "at most 100%" says nothing: what is known is that everything mapped was looked at.
        return "%s (all of the mapped part; the unmapped rest is unknown)" % covered
    if float(node.left_m2) < 1.0:
        return "%s (at most %d%% of the room)" % (covered, percent)      # "at least under 1m2 left" says nothing
    return "%s (at most %d%% of the room, at least %s left)" % (covered, percent, area_words(node.left_m2))


def format_frontier(frontier_clusters: int, unknown_behind_m2: Optional[float] = None) -> str:
    """The ``frontier=`` value of a room line: the count, and in brackets the unknown floor behind the clusters, united.

    ``2 (~30m2 unknown behind them)``, ``1 (under 1m2 unknown behind it)``;
    the count alone when nothing was measured or there is no frontier.
    """
    count = max(0, int(frontier_clusters))
    if count == 0 or unknown_behind_m2 is None or not math.isfinite(float(unknown_behind_m2)):
        return "%d" % count
    pronoun = "it" if count == 1 else "them"
    value = float(unknown_behind_m2)
    if value < 0.5:
        return "%d (under 1m2 unknown behind %s)" % (count, pronoun)
    return "%d (~%dm2 unknown behind %s)" % (count, int(round(value)), pronoun)


def format_exits(exits: int, doorways: int = 0) -> str:
    """The ``exits=`` value of a room line: the count, and in brackets how many stand at a seen door frame.

    ``exits=2 (1 doorway)``; the count alone when no doorway is known. The
    unseen floor behind the exits is no longer rendered here (since
    2026-11-11): the room's ``frontier=`` brackets carry one united figure
    for all its frontiers, the exits among them.
    """
    count = max(0, int(exits))
    doors = max(0, min(count, int(doorways)))
    if doors > 0:
        return "%d (%d doorway%s)" % (count, doors, "" if doors == 1 else "s")
    return "%d" % count


def format_prompt(target: str, nodes: Sequence[SearchNode], context: Optional[SearchContext] = None) -> str:
    """Everything the model is shown for this query."""
    context = context or SearchContext()
    storeys = ""
    if context.multi_storey or any(n.kind == STAIRS for n in nodes):
        storeys = STOREY_LINES_TEMPLATE.format(
            storey=context.storey or "the only storey mapped so far",
            others="; ".join(context.others) if context.others else "none known")
    home = ("\nWHERE IT LIVES: %s" % context.target_home.strip()) if context.target_home.strip() else ""
    return USER_PROMPT_TEMPLATE.format(
        target=target, home=home, house=context.house or "nothing known yet", storeys=storeys,
        n=len(nodes), nodes="\n".join(format_node(n) for n in nodes))


def system_prompt(nodes: Sequence[SearchNode], context: Optional[SearchContext] = None) -> str:
    """The system prompt for this query: the single-storey rules, plus the stairs supplement when a storey change is on offer, plus the openings supplement when an exit is offered as a node of its own (deprecated)."""
    text = SYSTEM_PROMPT
    if any(n.kind == STAIRS for n in nodes) or (context is not None and context.multi_storey):
        text += STAIRS_SUPPLEMENT
    if any(n.kind == OPENING for n in nodes):
        text += OPENINGS_SUPPLEMENT
    return text


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
        flat_p: The ablation that measures what the oracle contributes: when
            given, every node is read at this probability and the model is
            never asked (``source='flat'``), so RPT* orders the nodes by
            travel cost alone -- the nearest first -- and nothing else in the
            search changes (the room labels, the exclusions, the landmarks at
            their fixed probability). None (the default) asks the model.

    Attributes:
        reuses: Queries answered from the kept reply.
        queries: Calls actually made.
        unexplored_floor: As given.
        unexplored_elsewhere: As given.
        home_floor: As given.
        flat_p: As given.
    """

    def __init__(self, client, unexplored_floor: float = UNEXPLORED_FLOOR,
                 unexplored_elsewhere: float = UNEXPLORED_ELSEWHERE, home_floor: float = HOME_FLOOR,
                 flat_p: Optional[float] = None) -> None:
        for name, value in (("unexplored_floor", unexplored_floor), ("unexplored_elsewhere", unexplored_elsewhere),
                            ("home_floor", home_floor)):
            if not (0.0 <= float(value) < 1.0) or not math.isfinite(float(value)):
                raise ValueError("%s must lie in [0, 1), got %r" % (name, value))
        if flat_p is not None and (isinstance(flat_p, bool) or not math.isfinite(float(flat_p))
                                   or not 0.0 < float(flat_p) < 1.0):
            raise ValueError("flat_p must lie in (0, 1) or be None, got %r" % (flat_p,))
        self._client = client
        self._last_prompt = None  # type: Optional[str]
        self._last_reply = None   # type: Optional[Dict[str, Any]]
        self.unexplored_floor = float(unexplored_floor)
        self.unexplored_elsewhere = float(unexplored_elsewhere)
        self.home_floor = float(home_floor)
        self.flat_p = None if flat_p is None else float(flat_p)
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
        if self.flat_p is not None:
            return self.flat(nodes, self.flat_p)
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
    def flat(nodes: Sequence[SearchNode], p: float) -> NodeOracleResult:
        """The flat ablation's result: every node at ``p``, no model asked, no reading, no verdict."""
        share = min(float(p), P_CEILING)
        failure = (1.0 - share) ** len(nodes)
        return NodeOracleResult(probs={n.id: share for n in nodes}, elsewhere=failure, p_present=1.0 - failure,
                                source=SOURCE_FLAT, reasons={n.id: "flat ablation" for n in nodes},
                                raw_reply=None, spread=0.0)

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
                                source=SOURCE_LLM, reasons=reasons, raw_reply=reply if isinstance(reply, dict) else None,
                                spread=float(max(values) - min(values)) if values else 0.0, omitted=omitted,
                                reading=reading, floored=floored, capped=capped, home_here=home_here,
                                pass_verdict=parse_pass(reply))
