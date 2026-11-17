"""LLM node oracle: the target's name + the rooms mapped so far -> P(the target is in each room).

The room-search loop asks a language model one question per loop point, over
every node the search could go to next. This module is the whole of that
question: the prompt, the parse, and the contract the planner relies on. The
runtime routes the call to ``LLMConfig.reasoning_model``.

**The division of labour.** The model gives one thing: for each room, the
probability that the target is in it. Distance, and the weighing of distance
against every probability, is RPT*'s (``room_costs.build_instance`` charges
the travel, the solver minimises the expected time to find). So the prompt
carries no distance, no position and no word on where the robot stands
(``here=yes`` was dropped on 2026-11-16; :attr:`SearchNode.here` is kept for
the record and never rendered), and the model is told that travel is the
planner's.

**The model's own knowledge, and none of ours (since 2026-11-16).** The
system prompt opens by saying whose judgement is wanted -- a person's who
knows how homes are laid out -- and tells the model to use its own reasoning
and its own knowledge of standard homes and buildings, in which rooms
furniture and objects of each kind are found, and to derive every
probability from that. It holds no category knowledge: no room named as the
place of any object, no number or band a probability should fall in, no list
of a benchmark's goal categories, no count of rooms a home has, no worked
example. The only object it ever sees named is the one on the ``TARGET:``
line of the user prompt -- the name of the object the current episode
searches for (``TargetLabels.query``), given in every call. Until that day
the prompt was 14,751 characters of our conclusions: the rooms of each HM3D
goal category, the rooms a typical home has, a band of numbers for every
room status, "coverage first, time second", and a worked example
(``CHANGELOG.md`` 2026-11-16, "the oracle's prompt").

**What the data means, said in words (since 2026-11-17).** Beside what each
field of a line is, the prompt says what the search so far means for a
room, because a small model does not draw it from the fields alone (under
the facts-only wording of 2026-11-16 the 3B scored the rooms it had entered
above the rooms it had never entered, 41 % against 35 % on average over a
hundred recorded queries; under this one, 39 % against 54 %): the more of a
room is mapped and the more actions were spent searching it, the less of it
is left to hold the target -- its floor not mapped yet and what lies behind
its frontiers; a room never entered has not been searched, and is the
likelier the more the other rooms were searched; and a room's number is
about the target being IN the room -- walking through a searched room to
reach another is the planner's affair. No number, no threshold and no
example goes with any of it.

**What a room line holds: four things (since 2026-11-17).** Its type --
``unknown`` while unclassified, and then, only then, the objects seen in it;
``?`` on a tentative type; the floor mapped out of the room's floor, in
square metres and as a share (``mapped=6 of 12 m2 (50%)``: the observed
part of the room's FREE floor, ``room_extent.py`` -- furniture and pockets
the robot cannot stand on are not the room's floor); the frontiers left in
it; and the effort spent inside, ``never`` for a room not entered. Until
that day the line held everything the scene graph stores about the room
except distance: its size and the sides not mapped, its exits and what was
glimpsed through them, a status word, the rooms a door joins it to, its
objects with their counts and a note when the target's kind was detected in
it. All of that is still on the :class:`SearchNode`, for the recording;
which rooms a room is joined to and how far anything is are the planner's,
and are not the model's to read.

**One unit of effort.** With :attr:`SearchContext.action_time_s` given, the
effort on a line is written in actions (``searched=40 actions``), the unit
the HOUSE line's budget is in; without it, in time (``40s``, ``2min``). In
coarse steps, so the line does not change with every action.

**The reply** is ``{"home", "house", "stage", "pass", "nodes": [{"id",
"why", "p"}]}``, unchanged: three short readings in the model's words, kept
for the recording; a ``pass`` verdict (``first`` / ``second``), recorded;
and one integer percentage per node. The percentages are independent
per-node estimates, not shares of one distribution, and are handed to RPT*
without renormalisation.

**Between the reply and RPT* (:meth:`SearchNodeOracle.score`).** The model
need not write exactly what the planner takes; the code makes it so, from
the reply and the nodes alone and with no knowledge of any room or object.
*Format*: the number under ``p`` or a name a model drifts to, as a number
or as text, a percent sign after it; invented ids dropped; a reply with no
usable node refused (the caller repairs the schema once, then backs off --
never a made-up distribution). *Scale*: percentages, also when the reply
was written in fractions of one, clamped into 0-100 and below 1 as
probabilities (RPT*'s heuristic divides by ``1 - p``). *Meaning*: a node the
model left out is forgotten, not ruled out (a small share); **a room never
entered is never zero** (:func:`keep_unentered`: nobody has looked, so
"certain it is not there" is read as not given, and the loop's floor for
such a room then holds), while the zero of a room the robot has been inside
stands and keeps that room out of the order. The last reply is reused when
the prompt is byte-identical. One older floor remains as a knob and is OFF
by default (``unexplored_floor``). What the loop does to the numbers after
this -- by exploration state -- is ``RoomSearchLoop._scaled_values``.

An **opening** -- a doorway or gap at the edge of the mapped floor -- is a
node of its own only under the deprecated exits-as-nodes mode
(``OpeningSettings.exits_as_nodes``), which appends
:data:`OPENINGS_SUPPLEMENT`: the meaning of that line, nothing more. There
is no staircase node and no multi-storey supplement: the search is
single-storey, and both were deleted on 2026-11-16 until the multi-storey
work brings them back.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from sparx_agency.core.mapping.topology.search_node_prompts import (
    DEFAULT_PROMPT, NodeOraclePrompt, PROMPTS, prompt_version)

ROOM = "room"
OPENING = "opening"
KINDS = (ROOM, OPENING)

#: Largest node probability handed on: RPT*'s heuristic divides by ``1 - p``.
P_CEILING = 1.0 - 1e-6
#: Share, in percent, given to a listed node the model did not score.
OMITTED_PERCENT = 1.0
#: Object classes shown per room, the most numerous first; a longer list is answered about the list, not the room.
MAX_CLASSES_IN_PROMPT = 8
#: Default probability floor for an unexplored node (a never-entered, unidentified room;
#: an opening with nothing glimpsed through it). OFF: the model's numbers stand (see the
#: module docstring); ``SearchNodeOracle(unexplored_floor=0.25)`` restores the 3B-era arithmetic.
UNEXPLORED_FLOOR = 0.0
#: The two answers to "what is worth more now?": ``first`` while what is left unexplored is,
#: ``second`` once re-checking the rooms already scanned is.
PASS_FIRST, PASS_SECOND = "first", "second"
PASSES = (PASS_FIRST, PASS_SECOND)
#: The budget is shown to the model in steps of this many actions, so the prompt does not change on every action
#: over an unchanged map (a room's own durations are coarse too: :func:`coarse_duration`).
BUDGET_STEP = 25
#: The ``NodeOracleResult.source`` values the runtime flies on: the model's own numbers, or the flat
#: ablation's (``SearchNodeOracle(flat_p=...)``). ``uniform_fallback`` -- a model that failed -- is refused.
SOURCE_LLM, SOURCE_FLAT = "llm", "flat"
USABLE_SOURCES = (SOURCE_LLM, SOURCE_FLAT)

#: The texts of the DEFAULT prompt version (``search_node_prompts.py`` holds every version and what all of them
#: must hold to), under the names this module has always exported: the paper's listings
#: (``docs/paper/sync_prompts.py``) and the tests read them. A run may use another version
#: (``SearchNodeOracle(prompt=...)``); these three are then not what it sends.
SYSTEM_PROMPT = PROMPTS[DEFAULT_PROMPT].system
OPENINGS_SUPPLEMENT = PROMPTS[DEFAULT_PROMPT].openings_supplement
USER_PROMPT_TEMPLATE = PROMPTS[DEFAULT_PROMPT].user_template


@dataclass(frozen=True)
class SearchNode:
    """One place the search could go to next, as shown to the model.

    Attributes:
        id: The loop's node id -- a room pid, or an opening's id offset far
            above every pid. The model echoes it; nothing else about it matters.
        kind: :data:`ROOM` or :data:`OPENING`.
        label: Room type (``unknown`` while unidentified), or ``doorway`` /
            ``gap`` for an opening.
        tentative: The room type rests on a single kind of object (a weak
            label); rendered as ``type=kitchen?`` so the model can weigh it.
        area_m2: Room: its floor area -- the free cells the partition gives
            it, floor the robot can stand on (the furniture standing in the
            room and the pockets nothing can enter are not counted); 0 when
            unknown. The total of the line's ``mapped=``.
        frontier_clusters: Unexplored boundary clusters left in the room.
        searched_s: Seconds the robot has spent inside the room. Rendered in
            actions when the context gives the action time
            (``searched=40 actions``), else in time (``searched=40s``); a
            room never entered reads ``searched=never``.
        last_inside_ago_s: Seconds since it was last inside; None = never.
            Read for one thing only: whether the room was ever entered
            (``searched=never``). How long ago the robot left a room says
            where the robot is, and is not shown.
        here: The robot stands in this room now. For the record: NEVER
            rendered -- the room the robot stands in is the room at distance
            zero, and distance is the planner's (``here=yes`` was on the
            line until 2026-11-16).
        objects: The confirmed objects in the room, one entry per object, so
            a class repeats as often as it was found. On the line only while
            the room is not classified (``type=unknown (seen: chair x4,
            table)``, most numerous first): a classified room's type stands
            for them. For an opening, the classes glimpsed through it.
        via: Opening: the mapped room it opens from, in prompt words --
            ``room 6 (type=bathroom?)`` -- or None when unknown.
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
            reach, summed (since 2026-10-22) -- how much unexplored space
            the exits open onto. Rendered inside the ``exits=`` brackets as
            ``~30m2 unseen behind them``; 0 adds nothing.
        footprint_m2: Room: the floor area of its footprint -- the mapped
            free cells plus the islands they enclose (the floor under a bed
            or a table) -- from the room extents (since 2026-11-16,
            ``core/mapping/topology/room_extent.py``); 0 when not read, and
            ``area_m2`` is shown instead.
        length_m: Room: its extent along its longer axis, metres (0: unread).
        width_m: Room: its extent along its shorter axis, metres.
        length_final: Room: both sides across the longer axis are located;
            False renders the dimension with a ``+`` (a lower bound).
        width_final: Room: the same across the shorter axis.
        walls_seen: Room: how many of its four sides are located -- a wall,
            a door cut or settled unknown along at most ``open_tolerance``
            of the side's faces open to unseen space in runs wider than a
            doorway. None when not read.
        bounded: Room: all four sides located, so no part of the room lies
            outside the map -- rendered ``all sides mapped``, never as a
            final size: the partition may still join the room to its
            neighbour (see the module docstring). False renders ``size=at
            least ...`` with the count of sides not mapped; None when not
            read (the area alone is shown).
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
        open_perimeter_m: Room: metres of its boundary open to unseen space
            in runs wider than a doorway -- where the camera's reach ended
            inside the room. Rendered in the size's brackets of a room with
            a side not mapped (``open edge 2m``), from half a metre up.
        connected: Room: the types of the rooms a confirmed door or a
            doorway the robot walked or saw through joins it to (the scene
            graph's edges, ``ObservedSceneGraph.doors``), one entry per
            neighbouring room. Rendered as ``connected to: bedroom x2,
            hallway``; nothing when empty.
        target_seen: Room: an object of the target's own kind is among the
            confirmed objects of this room -- the detector found it and the
            search goes on all the same (the closing let it go: no route, a
            failed inspection). For the record since 2026-11-17: not on the
            line (until then True read ``target: detected here, not
            reached``); a confirmed target is a landmark node of the loop's
            own, at a fixed probability, without the model.
        observed_m2: Room: how much of ``area_m2`` -- the free floor, the
            cells the robot can stand on -- the camera has seen from within
            detection range with a clear line of sight (the sight ledger's
            per-cell record, ``RoomExtent.observed_m2``), square metres.
            Unlike ``covered_m2`` it counts no furniture island and no
            settled pocket: floor the robot cannot reach is not part of the
            room. Rendered with ``area_m2`` as ``mapped=6 of 12 m2 (50%)``.
            None when the record was not read: ``mapped=unknown of 12 m2``.

    **What a room line carries (since 2026-11-17): four things** -- ``type=``,
    ``mapped=``, ``frontier=`` and ``searched=`` (:func:`format_node`). Every
    other attribute here is kept for the recording and for the code that
    builds nodes, and is not shown to the model; where an attribute's
    description says "rendered as", that was its place on the line until then.
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
    via: Optional[str] = None
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
    open_perimeter_m: float = 0.0
    connected: Tuple[str, ...] = field(default_factory=tuple)
    target_seen: Optional[bool] = None
    observed_m2: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("SearchNode.kind must be one of %s, got %r" % (KINDS, self.kind))

    @property
    def unexplored(self) -> bool:
        """Whether the search knows nothing about this place yet, so the (optional) floor applies.

        A room never entered, never scanned, whose type is still ``unknown``;
        an opening nothing has been glimpsed through. An identified room and
        an opening with a glimpse are not: the model values them on the
        evidence.
        """
        if self.kind == ROOM:
            return (self.last_inside_ago_s is None and self.scanned is None
                    and (self.label or "unknown").strip().lower() == "unknown")
        if self.kind == OPENING:
            return not any(str(c).strip() for c in self.objects)
        return False

    @property
    def status(self) -> str:
        """The room's ``status=`` word with its durations in time: ``never_entered``, ``entered`` or ``scanned(...)``."""
        return self.status_words()

    def status_words(self, action_time_s: Optional[float] = None) -> str:
        """The room's ``status=`` word for the prompt; the scan's age in actions when ``action_time_s`` is given."""
        if self.scanned:
            if self.scanned_ago_s is None:
                return "scanned(%s)" % self.scanned
            return "scanned(%s, %s ago)" % (self.scanned, coarse_duration(self.scanned_ago_s, action_time_s))
        return "never_entered" if self.last_inside_ago_s is None else "entered"


@dataclass(frozen=True)
class SearchContext:
    """What the model is told about the home around the nodes, and the unit its lines are written in.

    Attributes:
        house: The big-picture line: every room type found so far with its
            status, the rooms still unidentified, the openings not yet
            looked into, and the budget (``actions used about 150 of 500``,
            see :func:`budget_line`). Empty reads as ``nothing known yet``.
        action_time_s: Seconds one action stands for. Given, the durations
            on a node line are written in actions -- the unit of the HOUSE
            line's budget (``searched=40 actions``); None (a rig that does
            not count actions) writes them in time (``searched=40s``).
    """

    house: str = ""
    action_time_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.action_time_s is not None and not (math.isfinite(float(self.action_time_s))
                                                   and float(self.action_time_s) > 0.0):
            raise ValueError("action_time_s must be positive and finite or None, got %r" % (self.action_time_s,))


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
        reading: The model's own words before its numbers -- ``home`` (where
            the target is normally found), ``house`` (what the rooms found
            say about the building), ``stage`` (how far the search has got)
            and ``pass``. Kept so the recording shows the judgement behind
            the distribution; nothing reads them for a decision.
        floored: Node ids whose number the code raised: a room never entered
            that the model put at zero (:func:`keep_unentered`), and -- when
            the optional floor is on -- the unexplored nodes under it.
        pass_verdict: The model's ``pass`` -- :data:`PASS_FIRST` while what
            is left unexplored is worth more than re-checking the rooms
            already scanned, :data:`PASS_SECOND` once it is not -- or None
            when not given. Shown and recorded; the loop's second pass is
            opened by the geometry, not by this word.
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


def coarse_duration(seconds: Optional[float], action_time_s: Optional[float] = None) -> str:
    """A duration for the prompt: in actions when ``action_time_s`` is given, else in time (:func:`coarse_seconds`).

    ``40 actions`` -- the unit the HOUSE line's budget is in, so the model
    can set the effort spent in a room beside the budget left. The steps are
    :func:`coarse_seconds`'s own (tens of seconds under a minute, whole
    minutes above), converted: the line changes exactly when the time
    rendering would, never on every action. None reads ``never``.
    """
    if seconds is None:
        return "never"
    if action_time_s is None:
        return coarse_seconds(seconds)
    value = max(0.0, float(seconds))
    bucket = round(value / 10.0) * 10.0 if value < 60.0 else round(value / 60.0) * 60.0
    return "%d actions" % int(round(bucket / float(action_time_s)))


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


def counted(names: Sequence[str], limit: Optional[int] = None) -> str:
    """``chair x4, table`` -- the distinct names in ``names`` with their counts, the most numerous first.

    Case and surrounding space are folded; a name that occurs once carries
    no count; ties are alphabetical, so the same room reads the same way
    every time. ``limit`` keeps the first that many kinds and says how many
    it left off (``chair x4, table, +3 more``): the list is what the model
    reads the absence of an object from, so a cut must show. Empty for no
    names.
    """
    counts = {}  # type: Dict[str, int]
    for name in names:
        key = str(name).strip().lower()
        if key:
            counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    words = [name if count == 1 else "%s x%d" % (name, count) for name, count in ordered]
    if limit is not None and len(words) > int(limit):
        words = words[:int(limit)] + ["+%d more" % (len(words) - int(limit))]
    return ", ".join(words)


def kinds(names: Sequence[str]) -> Tuple[str, ...]:
    """The distinct names in ``names``, case and surrounding space folded, sorted."""
    return tuple(sorted({str(name).strip().lower() for name in names if str(name).strip()}))


def format_node(node: SearchNode, action_time_s: Optional[float] = None) -> str:
    """One prompt line for ``node``. A room's line holds four things and nothing else (since 2026-11-17).

    * ``type=`` -- the room's classification, ``?`` on a tentative one. A
      room not classified yet reads ``unknown`` and, then only, lists the
      objects seen in it: ``type=unknown (seen: chair x4, table)``.
    * ``mapped=`` -- :func:`format_mapped`: the floor already seen out of
      the room's floor, in square metres and as a share.
    * ``frontier=`` -- the frontier clusters left in the room.
    * ``searched=`` -- the effort already spent inside
      (:func:`coarse_duration`), ``never`` for a room never entered.

    Which rooms a room is joined to and how far anything is are the
    planner's and are not on the line; nor are the exits, the status word or
    the objects of a classified room, which the node still carries for the
    record.
    """
    if node.kind == OPENING:
        glimpsed = counted(kinds(node.objects), MAX_CLASSES_IN_PROMPT) or "nothing yet"
        origin = "off %s" % node.via if node.via else "off the mapped floor"
        return "  ".join(["id=%d" % node.id, "OPENING", "%s %s" % (node.label or "gap", origin),
                          "to space NOT seen yet", "glimpsed through it: %s" % glimpsed])
    label = (node.label or "unknown").strip() or "unknown"
    if label.lower() == "unknown":
        label = "unknown (seen: %s)" % (counted(node.objects, MAX_CLASSES_IN_PROMPT) or "nothing yet")
    elif node.tentative:
        label += "?"
    searched = None if node.last_inside_ago_s is None else node.searched_s
    return "  ".join(["id=%d" % node.id, "ROOM", "type=%s" % label, "mapped=%s" % format_mapped(node),
                      "frontier=%d" % max(0, int(node.frontier_clusters)),
                      "searched=%s" % coarse_duration(searched, action_time_s)])


def format_mapped(node: SearchNode) -> str:
    """The ``mapped=`` value of a room line: ``6 of 12 m2 (50%)`` -- results the model reads and does not compute.

    The second number is the room's floor (:attr:`SearchNode.area_m2`: free
    cells only, floor the robot can stand on), the first how much of it the
    camera has seen from detection range (:attr:`SearchNode.observed_m2`),
    and the share is theirs: a room of 4 m2 and one of 40 m2 can both stand
    at 50 %, which is why the areas are said too. Whole square metres from
    ten up and one decimal under it, where a whole number says too little
    (``1.0 of 1.2 m2``); the share in tens, with ``100%`` kept for a room
    seen whole -- and the mapped part then never reads as the whole floor
    under it (``11 of 12 m2 (90%)``, not ``12 of 12``). Without the
    observation record the first number is ``unknown``; a room of no known
    size reads ``unknown``.
    """
    total = max(0.0, float(node.area_m2))
    if total <= 0.0:
        return "unknown"
    step = 1.0 if total >= 9.95 else 0.1
    places = "%.0f" if step == 1.0 else "%.1f"
    whole = places % max(total, step)                # a sliver of a few cells is not a room "of 0.0 m2"
    if node.observed_m2 is None:
        return "unknown of %s m2" % whole
    seen = max(0.0, min(total, float(node.observed_m2)))
    share = seen / total
    percent = 100 if share >= 0.995 else min(90, int(share * 10.0 + 0.5) * 10)
    part = whole if percent == 100 else places % seen
    if percent < 100 and part == whole:
        part = places % (float(whole) - step)
    return "%s of %s m2 (%d%%)" % (part, whole, percent)


def coarse_metres(value: float, step: float = 0.5) -> str:
    """Metres in ``step`` steps (``3.5``; ``4`` for a whole number), so an unchanged map gives an unchanged prompt."""
    rounded = round(float(value) / step) * step
    text = "%.1f" % rounded
    return text[:-2] if text.endswith(".0") else text


def format_size(node: SearchNode) -> str:
    """The ``size=`` value of a room line, in words that claim what the extents know and no more.

    With the room extents read (``bounded`` not None): ``12m2 (3.5x3.5m,
    all sides mapped)`` when every side is located -- no part of the room
    lies outside the map; NOT "final": the partition may still join it to
    its neighbour (see the module docstring) -- else ``at least 12m2
    (5+ x 3.5m, 1 of 4 sides not mapped yet, open edge 2m)``: the ``+``
    marks a dimension the camera's reach cut short, the open edge is the
    length of boundary still open to unseen space (said from half a metre
    up), and the area is what is mapped so far, never presented as the
    room's. Without the extents: the area alone (``12m2``, ``under 1m2``,
    ``unknown size``).
    """
    area = max(float(node.footprint_m2), float(node.area_m2))
    if node.bounded is None:
        return "unknown size" if area <= 0.0 else "under 1m2" if area < 1.0 else "%dm2" % int(round(area))
    length = coarse_metres(node.length_m) + ("" if node.length_final else "+")
    width = coarse_metres(node.width_m) + ("" if node.width_final else "+")
    if node.bounded:
        area_text = "under 1m2" if area < 1.0 else "%dm2" % int(round(area))
        return "%s (%sx%sm, all sides mapped)" % (area_text, length, width)
    missing = 4 - int(node.walls_seen or 0)
    sides = "%d of 4 sides not mapped yet" % missing
    if float(node.open_perimeter_m) >= 0.5:
        sides += ", open edge %sm" % coarse_metres(node.open_perimeter_m)
    if area < 1.0:
        return "under 1m2 mapped so far (%s x %sm, %s)" % (length, width, sides)
    return "at least %dm2 (%s x %sm, %s)" % (int(round(area)), length, width, sides)


def area_words(value: float) -> str:
    """Square metres for the prompt: ``0m2``, ``under 1m2``, else the nearest whole ``12m2``."""
    value = float(value)
    if value < 0.05:
        return "0m2"
    if value < 1.0:
        return "under 1m2"
    return "%dm2" % int(round(value))


def format_coverage(node: SearchNode) -> Optional[str]:
    """The ``covered=`` value of a room line: the floor the camera has looked at from detection range, as results the model reads and does not compute.

    A room with every side mapped: ``10m2 (80% of the room, 2m2 left)`` --
    the share of the whole room rounded DOWN to tens, and what is left --
    or ``12m2 (all of it)``. A room whose size is a lower bound: the share
    is an UPPER bound rounded UP and the remainder a LOWER bound, ``3m2 (at
    most 40% of the room, at least 4m2 left)``; with everything mapped
    covered, ``7m2 (all of the mapped part; the unmapped rest is
    unknown)``. None when the record was not read.
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


def format_exits(exits: int, doorways: int = 0, unseen_m2: float = 0.0) -> str:
    """The ``exits=`` value of a room line: the count, and in brackets how many stand at a seen door frame and how much unseen floor lies behind them.

    ``exits=2 (1 doorway, ~30m2 unseen behind them)``; a count alone when
    neither detail is known.
    """
    count = max(0, int(exits))
    doors = max(0, min(count, int(doorways)))
    details = []
    if doors > 0:
        details.append("%d doorway%s" % (doors, "" if doors == 1 else "s"))
    if count > 0 and float(unseen_m2) >= 0.5:
        details.append("~%dm2 unseen behind them" % int(round(float(unseen_m2))))
    if not details:
        return "%d" % count
    return "%d (%s)" % (count, ", ".join(details))


def format_prompt(target: str, nodes: Sequence[SearchNode], context: Optional[SearchContext] = None,
                  version: Union[str, NodeOraclePrompt, None] = None) -> str:
    """Everything the model is shown for this query: the target's name, the HOUSE line and one line per node.

    ``version`` names the prompt version whose frame is used (the default
    version for None); the HOUSE line and the node lines are the same under all.
    """
    context = context or SearchContext()
    return prompt_version(version).user_template.format(
        target=target, house=context.house or "nothing known yet", n=len(nodes),
        nodes="\n".join(format_node(n, context.action_time_s) for n in nodes))


def system_prompt(nodes: Sequence[SearchNode], context: Optional[SearchContext] = None,
                  version: Union[str, NodeOraclePrompt, None] = None) -> str:
    """The system prompt of ``version`` (the default for None) for this query, plus its openings supplement when an exit is offered as a node of its own (deprecated)."""
    chosen = prompt_version(version)
    text = chosen.system
    if any(n.kind == OPENING for n in nodes):
        text += chosen.openings_supplement
    return text


# -- the parse --------------------------------------------------------------
def _raw(entry: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    """The number ``entry`` gives for its node and the key it stands under, or None when there is none to read.

    ``p``, or one of the field names a model drifts to; a number written as
    text is read (``"40"``), with a percent sign after it too (``"40%"``).
    """
    for key in ("p", "probability", "percent", "score"):
        if key in entry:
            value = entry[key]
            if isinstance(value, str):
                value = value.strip().rstrip("%").strip()
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return (value, key) if math.isfinite(value) else None
    return None


def _percent(entry: Dict[str, Any], unit_scale: bool = False) -> Optional[float]:
    """The node's share in percent, 0-100, from what :func:`_raw` reads.

    A fraction of one is turned into a percentage when the field is named
    ``probability`` or when the whole reply was written that way
    (``unit_scale``, :func:`written_as_fractions`).
    """
    raw = _raw(entry)
    if raw is None:
        return None
    value, key = raw
    if unit_scale or (key == "probability" and value <= 1.0):
        value *= 100.0
    return max(0.0, min(100.0, value))


def written_as_fractions(rows: Sequence[Any]) -> bool:
    """Whether a reply's numbers are fractions of one (``0.4``) where percentages (``40``) were asked for.

    Read off the reply as a whole, never off one entry: no number above 1,
    and at least one strictly between 0 and 1. Whole numbers alone -- a reply
    of ``0`` and ``1`` -- are percentages as asked: a model that means "one
    percent" is not to be read as certain.
    """
    values = [raw[0] for raw in (_raw(row) for row in rows if isinstance(row, dict)) if raw is not None]
    return bool(values) and max(values) <= 1.0 and any(0.0 < value < 1.0 for value in values)


def parse_reply(reply: Any, nodes: Sequence[SearchNode]) -> Optional[Tuple[Dict[int, float], float, Dict[int, str], Tuple[int, ...]]]:
    """The scores, elsewhere, reasons and omitted ids in ``reply``, or None when unusable.

    The scores are percentages whatever the model wrote them as: a number
    as text, a percent sign, a reply in fractions of one
    (:func:`written_as_fractions`), a value outside 0-100 (clamped).
    Invented ids are dropped. A listed node the model skipped is given
    :data:`OMITTED_PERCENT` -- forgotten, not ruled out. A reply that scores
    no listed node at all is unusable. An explicit all-zero belief is valid:
    the runtime can explore geometrically rather than invent a semantic prior.
    """
    rows = reply.get("nodes") if isinstance(reply, dict) else None
    if not isinstance(rows, list):
        return None
    known = {int(n.id) for n in nodes}
    listed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            nid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if nid in known:
            listed.append((nid, row))
    unit_scale = written_as_fractions([row for _, row in listed])
    scores = {}  # type: Dict[int, float]
    reasons = {}  # type: Dict[int, str]
    for nid, row in listed:
        value = _percent(row, unit_scale)
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


_PASS_WORDS = {
    PASS_SECOND: ("second", "2nd", "re-check", "recheck", "revisit", "covered", "again"),
    PASS_FIRST: ("first", "1st", "explore", "unexplored"),
}


def parse_pass(reply: Any) -> Optional[str]:
    """The model's ``pass`` verdict as :data:`PASS_FIRST` or :data:`PASS_SECOND`, or None when absent or unreadable.

    The two words are what the prompt asks for; a few phrasings a model
    drifts to are accepted, ``second`` tested first so "second, not first"
    is read as second. A reply without the key is None.
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


def keep_unentered(scores: Dict[int, float], nodes: Sequence[SearchNode]) -> Tuple[Dict[int, float], Tuple[int, ...]]:
    """A room never entered is never handed on at zero; return the new scores and the ids raised.

    A zero says "certain the target is not there". Of a room the robot has
    been inside that is the model's verdict on what was looked at, and it
    stands: the room is then no node for the planner (the loop offers
    nothing under ``LoopSettings.min_prob``). Of a room seen from outside
    only it is more than anybody knows, so a number under
    :data:`OMITTED_PERCENT` is read as not given -- small, not nothing --
    and the loop's floor for a room never entered
    (``LoopSettings.unvisited_floor``) then holds for it as for every other
    such room, which is what keeps it in the order. The rule reads one fact,
    whether the robot was ever inside (:attr:`SearchNode.last_inside_ago_s`),
    and nothing about what the room or the target is.
    """
    out = dict(scores)
    raised = []
    for node in nodes:
        nid = int(node.id)
        if node.kind == ROOM and node.last_inside_ago_s is None and nid in out and out[nid] < OMITTED_PERCENT:
            out[nid] = OMITTED_PERCENT
            raised.append(nid)
    return out, tuple(sorted(raised))


def floor_unexplored(scores: Dict[int, float], nodes: Sequence[SearchNode], unexplored_floor: float,
                     ) -> Tuple[Dict[int, float], Tuple[int, ...]]:
    """Apply the optional arithmetic floor to the unexplored nodes; return the new scores and the ids raised.

    The floor is a fraction in ``[0, 1)``; the scores are percentages. Every
    unexplored node (:attr:`SearchNode.unexplored`) the model scored below
    the floor is raised to it; a node scored above it is left alone. 0 (the
    default) disables it: the model's numbers stand.
    """
    out = dict(scores)
    floored = []
    if unexplored_floor > 0.0:
        percent = 100.0 * float(unexplored_floor)
        for node in nodes:
            nid = int(node.id)
            if nid in out and node.unexplored and out[nid] < percent - 1e-9:
                out[nid] = percent
                floored.append(nid)
    return out, tuple(sorted(set(floored)))


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
        flat_p: The ablation that measures what the oracle contributes: when
            given, every node is read at this probability and the model is
            never asked (``source='flat'``), so RPT* orders the nodes by
            travel cost alone -- the nearest first -- and nothing else in the
            search changes (the room labels, the exclusions, the landmarks at
            their fixed probability). None (the default) asks the model.
        prompt: The prompt version the model is asked under: a registered
            name (``search_node_prompts.PROMPTS``), a
            :class:`~search_node_prompts.NodeOraclePrompt` (a wording under
            trial), or None for the default version. Wordings are tuned per
            model; :attr:`version` says which one this oracle sends.

    Attributes:
        reuses: Queries answered from the kept reply.
        queries: Calls actually made.
        unexplored_floor: As given.
        flat_p: As given.
        version: The :class:`~search_node_prompts.NodeOraclePrompt` in use.

    Raises:
        ValueError: A floor or a flat probability out of range, or a prompt
            version no one registered.
    """

    def __init__(self, client, unexplored_floor: float = UNEXPLORED_FLOOR,
                 flat_p: Optional[float] = None, prompt: Union[str, NodeOraclePrompt, None] = None) -> None:
        if not (0.0 <= float(unexplored_floor) < 1.0) or not math.isfinite(float(unexplored_floor)):
            raise ValueError("unexplored_floor must lie in [0, 1), got %r" % (unexplored_floor,))
        if flat_p is not None and (isinstance(flat_p, bool) or not math.isfinite(float(flat_p))
                                   or not 0.0 < float(flat_p) < 1.0):
            raise ValueError("flat_p must lie in (0, 1) or be None, got %r" % (flat_p,))
        self._client = client
        self._last_prompt = None  # type: Optional[str]
        self._last_reply = None   # type: Optional[Dict[str, Any]]
        self.unexplored_floor = float(unexplored_floor)
        self.flat_p = None if flat_p is None else float(flat_p)
        self.version = prompt_version(prompt)
        self.reuses = 0
        self.queries = 0

    def prompt(self, target: str, nodes: Sequence[SearchNode], context: Optional[SearchContext] = None) -> str:
        """The user prompt for ``nodes`` -- everything the model is shown beside the system prompt."""
        return format_prompt(target, nodes, context, self.version)

    def system(self, nodes: Sequence[SearchNode], context: Optional[SearchContext] = None) -> str:
        """The system prompt for ``nodes`` under this oracle's prompt version."""
        return system_prompt(nodes, context, self.version)

    def remember(self, prompt: str, reply: Dict[str, Any]) -> None:
        """Keep a usable reply as the answer to ``prompt``."""
        self._last_prompt, self._last_reply = prompt, reply

    def ask(self, user: str, system: Optional[str] = None) -> Dict[str, Any]:
        """One call to the model, on the reasoning route when the client has one (``system``: this version's, for None)."""
        self.queries += 1
        if system is None:
            system = self.version.system
        if _accepts_reasoning(self._client):
            return self._client.chat_json(system, user, reasoning=True)
        return self._client.chat_json(system, user)

    def probabilities(self, target: str, nodes: Sequence[SearchNode],
                      context: Optional[SearchContext] = None) -> NodeOracleResult:
        """Return P(``target`` is there) per node, in the model's judgement.

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
            kept = self.score(self._last_reply, nodes, self.unexplored_floor)
            if kept is not None:
                self.reuses += 1
                return replace(kept, reused=True)
        try:
            reply = self.ask(user, self.system(nodes, context))
        except Exception:
            return self.uniform(nodes, raw_reply=None)
        result = self.score(reply, nodes, self.unexplored_floor)
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
    def score(reply: Any, nodes: Sequence[SearchNode], unexplored_floor: float = 0.0) -> Optional[NodeOracleResult]:
        """What the planner is handed for ``reply``: the model's numbers, checked for format, scale and meaning.

        The model need not write exactly what RPT* takes; this makes it so,
        from the reply and the nodes alone and with no knowledge of any
        room or object:

        1. **Format** (:func:`parse_reply`): the number under ``p`` or a
           name a model drifts to, as a number or as text; ids nobody
           listed dropped; a reply that scores no listed node is unusable
           (None: the caller repairs the schema once, then backs off).
        2. **Scale** (:func:`parse_reply`, :func:`normalise`): percentages,
           whether written as such or as fractions of one, clamped into
           0-100 and turned into probabilities under :data:`P_CEILING`,
           each on its own -- never renormalised.
        3. **Meaning**: a listed node the model did not score is forgotten,
           not ruled out (:data:`OMITTED_PERCENT`); a room never entered is
           never zero (:func:`keep_unentered`), while the zero of a room
           the robot has been inside stands and keeps it out of the order;
           the optional floor (:func:`floor_unexplored`, off by default).

        The ``pass`` verdict and the three readings are read beside them.
        """
        parsed = parse_reply(reply, nodes)
        if parsed is None:
            return None
        scores, elsewhere, reasons, omitted = parsed
        scores, kept = keep_unentered(scores, nodes)
        scores, floored = floor_unexplored(scores, nodes, unexplored_floor)
        probs, elsewhere = normalise(scores, elsewhere)
        values = list(probs.values())
        reading = {key: str(reply[key])[:200] for key in ("home", "house", "stage", "pass")
                   if isinstance(reply, dict) and isinstance(reply.get(key), str) and reply[key].strip()}
        return NodeOracleResult(probs=probs, elsewhere=elsewhere, p_present=1.0 - elsewhere,
                                source=SOURCE_LLM, reasons=reasons, raw_reply=reply if isinstance(reply, dict) else None,
                                spread=float(max(values) - min(values)) if values else 0.0, omitted=omitted,
                                reading=reading, floored=tuple(sorted(set(kept) | set(floored))),
                                pass_verdict=parse_pass(reply))
