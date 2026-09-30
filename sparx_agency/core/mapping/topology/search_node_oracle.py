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

The reply is a distribution: for each node the probability, in percent, that
going there NEXT finds the target, plus ``elsewhere`` -- the target is in none
of them. The room-search loop hands the node probabilities, unnormalised, to
RPT*, whose objective (expected time-to-find) weighs them against the travel
cost the planner computes -- for a staircase that cost includes the climb --
which is why the model is told, twice, not to reason about distance.

What stays in code is the contract, never the judgement: parse and clamp,
drop invented ids, give an omitted node a small share rather than zero,
rescale so the node shares and ``elsewhere`` make one, refuse a reply with no
usable node (the caller repairs the schema once, then backs off), and reuse
the last reply when the prompt is byte-identical -- the effort numbers are
shown in coarse steps so an unchanged map produces an unchanged prompt.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

ROOM = "room"
STAIRS = "stairs"
KINDS = (ROOM, STAIRS)

#: Largest node probability handed on: RPT*'s heuristic divides by ``1 - p``.
P_CEILING = 1.0 - 1e-6
#: Share, in percent, given to a listed node the model did not score.
OMITTED_PERCENT = 1.0
#: Object classes shown per room; a longer list is answered about the list, not the room.
MAX_CLASSES_IN_PROMPT = 8

SYSTEM_PROMPT = """You are the reasoning module of a robot searching ONE building for ONE \
target object. The robot has partly mapped the building into NODES.
ROOM nodes are regions the robot segmented on the storey it stands on. A room's \
type was inferred from the objects seen inside it; type=unknown means NOT \
IDENTIFIED YET -- never "empty"; a type ending in "?" was inferred from a single \
kind of object and may be wrong (a sink alone reads as a kitchen until a toilet \
shows). Each room line gives: size; frontier = the number of unexplored openings \
or boundaries still inside it (0 = fully observed); searched = the time the robot \
has spent inside it; ago = the time since it was last inside (never = not entered \
yet); seen = the objects confirmed in it; here=yes if the robot stands in it now.
STAIRS nodes are staircases to another storey (up or down). Taking one means \
leaving this storey and searching the other one. The line says whether that \
storey was visited before and, if so, what was found and how much was searched \
there, and whether the robot arrived on this storey by these stairs a moment ago.
For EACH node give the probability, in percent, that going there NEXT finds the \
target: the target is there and the robot would see it by going and looking. \
Also give "elsewhere": the chance the target is in none of the listed nodes (a \
searched room where the detector missed it, or space no node leads to). All node \
values plus elsewhere sum to 100.
Work in this order, and write steps 1 and 2 into the reply before any number:
STEP 1 -- "home": the room type(s) where this object normally lives, most likely \
first (bed -> bedroom; frying pan -> kitchen; toilet -> bathroom; sofa -> living \
room; washing machine -> laundry room or kitchen). Small objects that travel \
(cup, book, phone) have a home too, only a weaker one.
STEP 2 -- "storey": read THIS storey from the types found on it. A storey with a \
kitchen, dining or living room is a GROUND floor; a storey of bedrooms and \
bathrooms is an UPPER floor; laundry, storage and garages are basements. Then say \
whether a room of the home type has been FOUND on this storey or is still MISSING, \
and, if missing, whether the home type belongs on this storey at all. Where rooms \
live in a house: kitchen, dining and living room downstairs; bedrooms upstairs; \
the bathroom BESIDE the bedrooms, so upstairs too (a ground floor has at most a \
small toilet room); an office or study on either.
STEP 3 -- the numbers, by these rules:
 1. HOME FOUND HERE with frontier left: it takes most of the mass.
 2. HOME MISSING HERE: the mass goes to (a) UNKNOWN rooms whose SIZE fits the home \
type -- bathrooms ARE small, so a 3-8 m2 unknown room beside bedrooms is most \
likely the bathroom (never "too small to be one"); a 10-20 m2 room is bedroom \
sized; a 20 m2+ room on a ground floor is the living room -- and (b) the STAIRS \
toward the storey where the home type usually is. Split between \
them by how well the sizes fit and how much of each unknown room is unseen. When \
the home type is not expected on this storey at all, the stairs take MOST of it.
 3. A room of ANOTHER identified type gets 0-3, even if the object could \
conceivably be there. Do not spread mass "just in case" -- elsewhere is for that.
 4. A room the target cannot be in by type (frying pan in a bathroom, toilet in a \
bedroom) gets 0-1, whatever else its line says.
 5. frontier=0 means fully observed: 0-2 unless the object is small enough to hide \
from the views taken.
 6. Searched long and recently -> low, not zero. Searched briefly, or long ago, \
with frontier left -> keeps a share. Never entered, right type -> the best bet.
 7. A hallway or corridor rarely holds the target, but one with frontier LEADS to \
unseen rooms: value it by what it may lead to.
 8. STAIRS to a visited storey: if a room of the home type was FOUND there and is \
barely searched, that storey holds MOST of the mass -- a known kitchen beats an \
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
 13. Integers 0-100; node values plus elsewhere sum to 100. Keep elsewhere for what \
the nodes genuinely cannot cover (5-15 is typical); never park mass there to avoid \
deciding.
 14. Per node write "why" first (at most 12 words, lower case, naming the room's \
likely type and what decided it), then "p".
Reply with ONLY this JSON, keys in this order:
{"home":"<step 1, max 10 words>","storey":"<step 2, max 20 words>",
 "nodes":[{"id":<int>,"why":"<why>","p":<int>}, ...],"elsewhere":<int>}
Example of the FORMAT and the style of reasoning -- a different building every \
time, so never copy its numbers. TARGET television, robot on an upper storey:
id=4  ROOM  type=bedroom  size=13m2  frontier=0  searched=1min  ago=3min  seen: bed, lamp
id=9  ROOM  type=unknown  size=6m2  frontier=1  searched=0s  ago=never  seen: nothing yet
id=11  ROOM  type=unknown  size=22m2  frontier=2  searched=0s  ago=never  seen: nothing yet
id=100003  STAIRS down  to a storey NOT visited yet
{"home":"living room, sometimes a bedroom",
 "storey":"upper floor (bedroom found); no living room here; living rooms are downstairs",
 "nodes":[{"id":4,"why":"bedroom, fully seen, no television","p":2},
{"id":9,"why":"bathroom-sized room, no television fits","p":1},
{"id":11,"why":"large upstairs room, maybe a lounge","p":30},
{"id":100003,"why":"unvisited ground floor holds the living room","p":60}],"elsewhere":7}"""
USER_PROMPT_TEMPLATE = """TARGET: {target}

THIS STOREY: {storey}
OTHER STOREYS: {others}

NODES ({n}):
{nodes}

Score all {n} nodes and elsewhere. JSON only."""


@dataclass(frozen=True)
class SearchNode:
    """One place the search could go to next, as shown to the model.

    Attributes:
        id: The loop's node id -- a room pid, or a staircase id offset far
            above every pid. The model echoes it; nothing else about it matters.
        kind: :data:`ROOM` or :data:`STAIRS`.
        label: Room type (``unknown`` while unidentified) or ``stairs up`` /
            ``stairs down``.
        tentative: The room type rests on a single kind of object (a weak
            label); rendered as ``type=kitchen?`` so the model can weigh it.
        area_m2: Room floor area; 0 when unknown or for stairs.
        frontier_clusters: Unexplored boundary clusters left in the room.
        searched_s: Seconds the robot has spent inside the room.
        last_inside_ago_s: Seconds since it was last inside; None = never.
        here: The robot stands in this room now.
        objects: Object classes confirmed in the room.
        direction: ``+1`` up / ``-1`` down for stairs, 0 for rooms.
        destination_visited: Stairs: the other storey has been stood on.
        destination: Stairs: one-line summary of that storey -- rooms found,
            time searched, frontier left -- or None when unvisited.
        arrived_by: Stairs: the robot came onto this storey by them.
        arrived_ago_s: Stairs: seconds since that arrival, when ``arrived_by``.
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

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("SearchNode.kind must be one of %s, got %r" % (KINDS, self.kind))


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
        probs: ``{node_id: probability}`` in ``[0, P_CEILING]`` -- NOT
            normalised over the nodes; they sum to :attr:`p_present`.
        elsewhere: The share the model kept for "in none of these".
        p_present: ``sum(probs)`` -- the chance the target is reachable
            through some listed node.
        source: ``'llm'`` for a usable reply, ``'uniform_fallback'`` when the
            model failed and a flat distribution was substituted. The runtime
            refuses to fly on the latter.
        reasons: ``{node_id: why}`` in the model's words.
        raw_reply: The parsed reply, for the recording.
        spread: Max minus min of ``probs``; ~0 means the model said nothing.
        reused: The prompt was byte-identical to the last one answered, so
            the kept reply was re-read instead of spending a call.
        omitted: Node ids the model did not score (given :data:`OMITTED_PERCENT`).
        reading: The model's steps 1 and 2 in its own words -- ``home`` (where
            the target normally lives) and ``storey`` (what this storey is and
            whether the home type was found or is missing here). Written
            before the numbers so a mid-size model applies its own reading;
            kept so the recording shows the judgement behind the distribution.
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
    names = sorted({str(c).strip().lower() for c in node.objects if str(c).strip()})
    seen = ", ".join(names[:MAX_CLASSES_IN_PROMPT]) if names else "nothing yet"
    size = "%dm2" % int(round(float(node.area_m2))) if float(node.area_m2) > 0.0 else "unknown size"
    label = node.label + ("?" if node.tentative and node.label != "unknown" else "")
    parts = ["id=%d" % node.id, "ROOM", "type=%s" % label, "size=%s" % size,
             "frontier=%d" % max(0, int(node.frontier_clusters)),
             "searched=%s" % coarse_seconds(node.searched_s),
             "ago=%s" % coarse_seconds(node.last_inside_ago_s)]
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
    no listed node at all is unusable, and so is one whose total mass is
    zero with no ``elsewhere`` to carry it.
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
    if sum(scores.values()) + elsewhere <= 0.0:
        return None
    return scores, elsewhere, reasons, omitted


def normalise(scores: Dict[int, float], elsewhere: float) -> Tuple[Dict[int, float], float]:
    """Node shares and elsewhere as fractions of one, whatever the model's arithmetic did."""
    total = sum(scores.values()) + elsewhere
    scale = 1.0 / total if total > 0.0 else 0.0
    probs = {nid: min(P_CEILING, max(0.0, value * scale)) for nid, value in scores.items()}
    return probs, max(0.0, min(1.0, elsewhere * scale))


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

    Attributes:
        reuses: Queries answered from the kept reply.
        queries: Calls actually made.
    """

    def __init__(self, client) -> None:
        self._client = client
        self._last_prompt = None  # type: Optional[str]
        self._last_reply = None   # type: Optional[Dict[str, Any]]
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
            kept = self.score(self._last_reply, nodes)
            if kept is not None:
                self.reuses += 1
                return replace(kept, reused=True)
        try:
            reply = self.ask(user)
        except Exception:
            return self.uniform(nodes, raw_reply=None)
        result = self.score(reply, nodes)
        if result is None:
            return self.uniform(nodes, raw_reply=reply if isinstance(reply, dict) else None)
        self.remember(user, reply)
        return result

    # -- internals ------------------------------------------------------------
    @staticmethod
    def uniform(nodes: Sequence[SearchNode], raw_reply: Optional[Dict[str, Any]]) -> NodeOracleResult:
        share = 1.0 / (len(nodes) + 1)
        return NodeOracleResult(probs={n.id: share for n in nodes}, elsewhere=share, p_present=1.0 - share,
                                source="uniform_fallback", reasons={n.id: "" for n in nodes},
                                raw_reply=raw_reply, spread=0.0)

    @staticmethod
    def score(reply: Any, nodes: Sequence[SearchNode]) -> Optional[NodeOracleResult]:
        """Parse, fill the omitted, normalise, clamp."""
        parsed = parse_reply(reply, nodes)
        if parsed is None:
            return None
        scores, elsewhere, reasons, omitted = parsed
        probs, elsewhere = normalise(scores, elsewhere)
        values = list(probs.values())
        reading = {key: str(reply[key])[:200] for key in ("home", "storey")
                   if isinstance(reply, dict) and isinstance(reply.get(key), str) and reply[key].strip()}
        return NodeOracleResult(probs=probs, elsewhere=elsewhere, p_present=float(sum(values)),
                                source="llm", reasons=reasons, raw_reply=reply if isinstance(reply, dict) else None,
                                spread=float(max(values) - min(values)) if values else 0.0, omitted=omitted,
                                reading=reading)










