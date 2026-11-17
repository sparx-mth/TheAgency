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
version, and the digest shows it when that rule is broken (it was, once, on
instruction: the note at :data:`FACTS_QWEN_3B_V1` says when and gives the
digest of the text before).

**What every version holds to** (``tests/test_search_node_prompts.py`` runs
over the whole registry, so a new version cannot slip past it):

* It opens with whose judgement is wanted -- "You reason like a person who
  knows how homes are laid out." -- and the model is told to use its own
  reasoning and its own knowledge of standard homes and buildings, and to
  derive every probability from that.
* No knowledge of ours about where things are: no object named, no room type
  named as the place of anything, no probability, band or typical value, no
  list of a benchmark's categories, no worked example. The one object a
  model ever sees named is the target of the episode, on the ``TARGET:``
  line of every query.
* What the data means, never a value for it. A version says what each
  field of a line is, and -- since 2026-11-17 -- what the search so far
  means for a room: the more of it is mapped and the more actions were
  spent searching it, the less of it is left to hold the target (its
  unmapped floor and what lies behind its frontiers); a room never entered
  has not been searched, and grows likelier as the others are searched; a
  room's number is about the target being in it, not about walking through
  it. In words only: no number, no threshold, no band, no room, no object,
  no worked example.
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
offered as a node of its own: estimate the probability that the target is in the space behind it. "off room N (type=...)": the room it opens from. "glimpsed through \
it:" objects seen beyond it."""

#: Written against ``qwen2.5:3b-instruct`` (the model the laptop runs). **Edited in place on 2026-11-17**, on the
#: user's instruction, with the room line it explains (``search_node_oracle.format_node``): the name is the one of
#: 2026-11-16, the text is not. The wording of 2026-11-16 had the digest ``3ee6c00c94dc1753...`` and explained a line
#: of ten fields; a record that carries that digest was asked under it. What this text is:
#:
#: * it opens with whose judgement is wanted -- a person who knows how homes are laid out -- and tells the model to
#:   use its own knowledge of where things are; it names no object and no room, gives no number, band or example;
#: * it explains the FOUR things a room line holds (type, mapped floor, frontiers left, actions spent inside) and
#:   the HOUSE line, and nothing the code does not write;
#: * it says what the search so far MEANS, in three short paragraphs and without a number: the more of a room is
#:   mapped and the more actions were spent searching it, the less of it is left to hold the target; a room not
#:   entered has not been searched, and the more the others were searched the likelier it is; a room's number is
#:   about the target being IN it, not about walking through it. A 3B does not make these inferences from the
#:   fields alone -- the facts-only wording of 2026-11-16 left them to the model, and it scored the rooms it had
#:   entered above the rooms it had never entered (``core/mapping/topology/README.md``);
#: * the question is repeated, with the target's name and the number of nodes, at the end of the user prompt, and
#:   the reply format is on one line and says it ends with the nodes.
#:
#: Three things this wording was chosen for, each a difference of one sentence measured on a hundred recorded
#: queries (the README has the table). They are what that round showed and no more: one sentence moves a tenth of a
#: 3B's replies, so change a sentence here and measure again.
#:
#: * **Say it in the affirmative.** With "the search goes on, so it has not been found yet", 44 % of the zeros the
#:   model wrote gave "not found there" or "no evidence" as their reason (18 % under the text before) and a fifth of
#:   the replies put every room at 0. With "so the target is in a place the robot has not seen yet", 18 % -- and the
#:   numbers fell with the effort spent in a room again.
#: * **Ask about every node, by count.** A reply that leaves nodes out scores exactly one, the last one listed
#:   (every such reply, under every wording tried). A closing question in the singular ("is the target there?") got
#:   11 such replies in a hundred; "for each of the N nodes above ... score all N" got 5.
#: * **Name what the number is about**: "the part of that room the robot has not seen yet". With "in that room"
#:   alone a room never entered was put at 0 in a third of its lines; with this, in a quarter.
#:
#: A number the model writes is not the last word: ``search_node_oracle.SearchNodeOracle.score`` checks format, scale
#: and meaning before RPT* sees it (a room never entered is never zero), and the loop then weighs it by exploration
#: state.
FACTS_QWEN_3B_V1 = NodeOraclePrompt(
    name="facts-qwen2.5-3b-v1",
    tuned_on="qwen2.5:3b-instruct",
    system="""You reason like a person who knows how homes are laid out. You are the reasoning module of a robot that \
searches a building for one object. The TARGET line names it. The robot has mapped part of the building and \
divided the map into rooms. Each line under NODES is one room it can search next. A detector checks every \
camera frame for the target, so the target is in a place the robot has not seen yet. For every node, estimate \
the probability, in percent, that the target is in the part of that room the robot has not seen yet.

Use your own reasoning and your own knowledge of standard homes and buildings: in which rooms furniture and \
objects of each kind are normally found. Decide for yourself where this target makes sense, read what each line \
says about its room, and derive every probability from that. No numbers are supplied: the judgement is yours.

Fields of a room line:
type= the room's classification. A trailing "?": tentative. "unknown": not classified yet, so it may be any \
kind; the objects seen in it so far are listed in brackets ("xN": N of a kind).
mapped= the floor of the room the robot has already seen, out of all its floor, in square metres and as a \
percentage. Only floor the robot can reach counts.
frontier= how many places are left in the room where the mapped floor borders space not seen yet.
searched= how many actions the robot has already spent searching inside the room. "never": it has not entered \
the room.
The HOUSE line sums up the building so far: the types of the rooms found and whether each was entered or \
scanned (its search is complete), rooms not classified, openings (doorways and gaps) not yet looked into, \
whether unexplored frontier is reachable, and the actions used out of the budget.

What the search so far means:
The more of a room is mapped and the more actions were spent searching it, the less of it is left to hold the \
target: the target can only be on its floor not mapped yet or behind its frontiers.
A room never entered has not been searched, so all of it can still hold the target. The more the other rooms \
have been searched, the more likely the target is in a room not entered yet.
A room's number says only whether the target is in that room. Leave travel out: a planner weighs your \
probabilities against the distances, and it may pass through a searched room on the way to another.

Judge every node on its own: several may each hold a target, so the numbers are independent and need not sum to \
100. Score every id given, and no other.
Reply with ONLY this JSON, keys in this order, ending with one "nodes" entry per node:
{"home":"<where this target is normally found, max 12 words>","house":"<what the rooms found say about this \
building, max 25 words>","stage":"<how far the search has got, max 20 \
words>","pass":"first|second","nodes":[{"id":<int>,"why":"<reason, max 12 words>","p":<int 0-100>}, ...]}
"p": the probability in percent, from 0 (certain the target is not there) to 100 (certain it is).
"pass": "second" if re-checking rooms already scanned is now worth more than what is left unexplored, else \
"first".""",
    openings_supplement=OPENINGS_SUPPLEMENT,
    user_template="""TARGET: {target}

HOUSE: {house}

NODES ({n}):
{nodes}

For each of the {n} nodes above: how likely is it that the target ({target}) is in the part of that room not \
seen yet? Score all {n}, independently. JSON only.""")


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
