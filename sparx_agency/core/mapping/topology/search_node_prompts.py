"""The node oracle's prompt, by version: what a model is told, under a name the run record carries.

A *version* is the whole of what the model is given beside the facts of one
query: the system prompt, the supplement for the deprecated exits-as-nodes
mode, and the frame of the user prompt. The facts themselves -- the HOUSE
line and one line per node -- are written by ``search_node_oracle.py`` and
are the same under every version.

**Why versions.** A wording is read by one model and may be misread by the
next: what a 14B model takes from one sentence, a 3B model needs spelled out
field by field. So a wording is written against ONE model, probed on it
(``tasks/planning/objnav_benchmark_runtime/tests/probe_node_oracle.py``), and
registered here under a name; a run selects one by that name
(``LoopSettings.oracle_prompt``) and its record carries the name, the model
the wording was probed on, and the digest of the text
(``configuration()["node_oracle"]["prompt"]``) beside the model that answered
(``method.llm`` / ``method.llm_identity`` in the frozen configuration). A
version is never edited after a run has used it: a changed wording is a new
version, and the digest shows it when that rule is broken.

**What every version holds to** (``tests/test_search_node_prompts.py`` runs
over the whole registry, so a new version cannot slip past it):

* The model is told to use its own reasoning and its own knowledge of
  standard homes and buildings, and to derive every probability from that.
* No knowledge of ours about where things are: no object named, no room type
  named as the place of anything, no probability, band or typical value, no
  list of a benchmark's categories, no worked example. The one object a
  model ever sees named is the target of the episode, on the ``TARGET:``
  line of every query.
* No rule of judgement. A version may state FACTS about a field -- what it
  measures, how it was obtained (a detector looks at every camera frame; the
  covered floor is floor that detector has checked) -- and leaves what
  follows from them to the model.
* No distance, no position: travel is the planner's.
* The reply format, unchanged: ``{"home", "house", "stage", "pass",
  "nodes": [{"id", "why", "p"}]}``.

**Adding a version for another model** -- the whole procedure:

1. Copy the nearest :class:`NodeOraclePrompt` below under a new ``name``
   (``facts-<model>-v<N>``) with ``tuned_on`` set to the model's tag, and
   add it to :data:`PROMPTS`.
2. Probe it on that model, and against the version it came from::

       LLM_REASONING_MODEL=<model> .venv/bin/python -m \\
           sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.probe_node_oracle --prompt <name>

3. Reword and repeat until the probe's claims hold; run the unit tests.
4. Select it for a run with ``{"room_search": {"oracle_prompt": "<name>"}}``
   in the policy config (``LoopSettings.oracle_prompt``), and record the
   probe's figures in ``core/mapping/topology/README.md``.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Union

#: The placeholders a user template must carry: the target's name, the HOUSE line, the node count, the node lines.
USER_TEMPLATE_FIELDS = ("{target}", "{house}", "{n}", "{nodes}")
#: The keys of the reply a system prompt must ask for (the oracle's parser reads them): a version whose text
#: names one of them nowhere is refused when it is built.
REPLY_KEYS = ("home", "house", "stage", "pass", "nodes", "id", "why", "p")


@dataclass(frozen=True)
class NodeOraclePrompt:
    """One version of the node oracle's prompt.

    Attributes:
        name: The version's name -- what a run selects it by and what its
            record carries. Never reused for another wording.
        tuned_on: The model tag this wording was written against and probed
            on (``qwen2.5:3b-instruct``). Documentation for the record: the
            code never reads it, and any model may be asked under any version.
        system: The system prompt: the task, whose knowledge to use, what
            each field of a line means, the reply format.
        openings_supplement: Appended to ``system`` when an OPENING node is
            offered (the deprecated exits-as-nodes mode): what that line means.
        user_template: The frame of the user prompt, with the placeholders
            of :data:`USER_TEMPLATE_FIELDS`.

    Raises:
        ValueError: A blank name or text, a user template that lacks a
            placeholder, or a system prompt that does not ask for one of
            the reply's keys.
    """

    name: str
    tuned_on: str
    system: str
    openings_supplement: str
    user_template: str

    def __post_init__(self) -> None:
        for field_name in ("name", "tuned_on", "system", "openings_supplement", "user_template"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("NodeOraclePrompt.%s must be a non-blank string" % field_name)
        missing = [word for word in USER_TEMPLATE_FIELDS if word not in self.user_template]
        if missing:
            raise ValueError("NodeOraclePrompt %r: the user template lacks %s" % (self.name, ", ".join(missing)))
        unasked = [key for key in REPLY_KEYS if '"%s":' % key not in self.system]
        if unasked:
            raise ValueError("NodeOraclePrompt %r: the system prompt does not ask for the reply's %s"
                             % (self.name, ", ".join(unasked)))

    @property
    def sha256(self) -> str:
        """The digest of the three texts: what tells two wordings apart whatever they are called."""
        joined = "\x00".join((self.system, self.openings_supplement, self.user_template))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def describe(self) -> Dict[str, Any]:
        """What a run record says of the prompt it used."""
        return {"version": self.name, "tuned_on": self.tuned_on, "sha256": self.sha256,
                "system_chars": len(self.system)}


#: What an OPENING line means (the deprecated exits-as-nodes mode): the same for every version so far.
OPENINGS_SUPPLEMENT = """
An OPENING line is a doorway or gap at the edge of the mapped floor that leads to space not seen yet, \
offered as a node of its own (it is then not counted in its room's exits=): estimate the probability that \
the target is in the space behind it. "off room N (type=...)": the room it opens from. "glimpsed through \
it:" objects seen beyond it."""

#: Written against ``qwen2.5:3b-instruct`` on 2026-11-16 (the model the laptop runs). A small model does not work
#: out for itself what a field implies, so every fact it would have to infer is SAID, as a fact: what a node
#: covers (the room and the unseen space behind its exits); that a detector looks for the target in every frame
#: and has not found it on the floor a line lists as covered; that the number asked for is about the part of the
#: node not checked yet; that an unidentified room may be any kind; what the two ends of the scale mean. The
#: question is repeated, with the target's name, at the end of the user prompt, and the reply format is on one
#: line and says it ends with the nodes (a 3B closed the object after its three readings otherwise). One thing a
#: line can carry is deliberately NOT in this legend: the note ``target: detected here, not reached``, which says
#: what it means in its own words. Explaining it here made the 3B read its absence from a line as certainty
#: that the target is not there -- every node of a query at 0 -- and that text held 4 of the probe's 19 claims
#: where this one holds 9.
#: What the probe showed, and what else did NOT help, is in ``core/mapping/topology/README.md``.
FACTS_QWEN_3B_V1 = NodeOraclePrompt(
    name="facts-qwen2.5-3b-v1",
    tuned_on="qwen2.5:3b-instruct",
    system="""You are the reasoning module of a robot that searches a building for one object. The TARGET line names it. \
The robot has mapped part of the building and divided the map into rooms. Each line under NODES is one \
place it can search next: a room, together with the unseen space its exits open onto. A detector looks for \
the target in every camera frame; on the floor a line lists as covered it has looked and not found it. For \
every node, estimate the probability, in percent, that the target is in the part of it not checked yet: the \
floor of the room that is left, and the unseen space behind its exits.

Use your own reasoning and your own knowledge of standard homes and buildings: how they are laid out, and \
in which rooms furniture and objects of each kind are normally found. Decide for yourself where this \
target makes sense, read what each line says about its room, and derive every probability from that. No \
rules and no typical values are supplied: the judgement is yours.

Fields of a room line:
type= the room's classification, inferred from the objects seen in it. "unknown": the robot has not \
determined yet what kind of room this is; it may be any kind. A trailing "?": tentative, the evidence is \
thin.
size= floor area and dimensions on the current map. "all sides mapped": no part of the room lies outside \
the map, though the map's division into rooms may still be revised. "at least": a side is not mapped yet \
and the room continues beyond it ("+": that dimension is cut short; "open edge": the length of the \
unmapped side).
covered= floor of the room the detector has already checked: its area and its share of the room. "left": \
floor it has not checked yet.
frontier= boundaries to unexplored space that the robot can reach inside the room.
exits= doorways and gaps on the room's edge that lead to space not seen yet; in brackets, how many are at \
a door frame and how much unseen floor lies behind them. "glimpsed through its exits:" objects detected \
beyond them, outside this room.
status= never_entered (seen from outside only), entered (the robot has been inside), or scanned(<how>, \
<how long ago>) (its look at the room is complete; <how> says in what way).
searched= how long the robot has spent inside the room so far.
connected to: the types of the rooms it is joined to by a door or a doorway.
seen: the kinds of object detected in the room so far ("xN": N of a kind; "+N more": kinds not listed).
The HOUSE line summarises the building so far: room types found and their status, rooms not identified, \
openings (exits of any room) not yet looked into, whether unexplored frontier is reachable, and actions \
used out of the budget.

Judge every node on its own: several may each hold a target, so the numbers are independent and need not \
sum to 100. Leave travel out: a planner weighs your probabilities against the distances. Score every id \
given, and no other.
Reply with ONLY this JSON, keys in this order, ending with one "nodes" entry per node:
{"home":"<where this target is normally found, max 12 words>","house":"<what the rooms found say about \
this building, max 25 words>","stage":"<how far the search has got, max 20 \
words>","pass":"first|second","nodes":[{"id":<int>,"why":"<reason, max 12 words>","p":<int 0-100>}, ...]}
"p": the probability in percent, from 0 (certain the target is not there) to 100 (certain it is).
"pass": "second" if re-checking rooms already scanned is now worth more than what is left unexplored, else \
"first".""",
    openings_supplement=OPENINGS_SUPPLEMENT,
    user_template="""TARGET: {target}

HOUSE: {house}

NODES ({n}):
{nodes}

For each of the {n} nodes: how likely is it that the target ({target}) is in the part of it not checked yet? Score them independently. JSON only.""")


def _registry(*versions: NodeOraclePrompt) -> Dict[str, NodeOraclePrompt]:
    """``{name: version}``; two versions under one name, or one wording under two names, is a mistake."""
    out = {}  # type: Dict[str, NodeOraclePrompt]
    digests = {}  # type: Dict[str, str]
    for version in versions:
        if version.name in out:
            raise ValueError("two node-oracle prompt versions are named %r" % version.name)
        if version.sha256 in digests:
            raise ValueError("node-oracle prompt versions %r and %r are the same text"
                             % (digests[version.sha256], version.name))
        out[version.name] = version
        digests[version.sha256] = version.name
    return out


#: Every registered version, by name.
PROMPTS = _registry(FACTS_QWEN_3B_V1)
#: The version a run uses unless its settings name another (``LoopSettings.oracle_prompt``).
DEFAULT_PROMPT = FACTS_QWEN_3B_V1.name


def prompt_version(version: Union[str, NodeOraclePrompt, None] = None) -> NodeOraclePrompt:
    """The registered version called ``version`` (the default for None); a :class:`NodeOraclePrompt` is returned as is.

    Raises:
        ValueError: No version is registered under that name; the message lists those that are.
    """
    if version is None:
        version = DEFAULT_PROMPT
    if isinstance(version, NodeOraclePrompt):
        return version
    if not isinstance(version, str) or version not in PROMPTS:
        raise ValueError("no node-oracle prompt version is named %r; registered: %s"
                         % (version, ", ".join(sorted(PROMPTS))))
    return PROMPTS[version]
