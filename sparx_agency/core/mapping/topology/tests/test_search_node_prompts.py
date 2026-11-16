"""The node oracle's prompt versions: the registry, and what EVERY registered version must hold to.

A version is a wording written against one model. These tests run over the
whole registry, so a version added for another model is held to the same
contract as the first without anyone remembering to list it.
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from sparx_agency.core.mapping.topology import search_node_oracle, search_node_prompts
from sparx_agency.core.mapping.topology.room_classifier import DEFAULT_LABEL_SET
from sparx_agency.core.mapping.topology.search_node_oracle import (
    OPENING, ROOM, SearchContext, SearchNode, SearchNodeOracle, format_node, format_prompt, system_prompt)
from sparx_agency.core.mapping.topology.search_node_prompts import (
    DEFAULT_PROMPT, PROMPTS, REPLY_KEYS, USER_TEMPLATE_FIELDS, NodeOraclePrompt, prompt_version)

VERSIONS = sorted(PROMPTS)
#: The least a wording under trial must carry to be built at all: the keys of the reply the parser reads.
TRIAL_REPLY = '{"home":"..","house":"..","stage":"..","pass":"..","nodes":[{"id":0,"why":"..","p":0}]}'
TRIAL_SYSTEM = "TRIAL SYSTEM " + TRIAL_REPLY

#: Every object a benchmark asks for (the six HM3D goal categories, the twenty-one of MP3D) and the furniture the
#: prompt of before 2026-11-16 named, as words: none may appear in what a model is told beside the facts.
OBJECT_WORDS = ("chair", "bed", "plant", "toilet", "tv", "monitor", "television", "sofa", "couch", "table", "sink",
                "bathtub", "shower", "cabinet", "counter", "cushion", "fireplace", "gym", "equipment", "picture",
                "stool", "towel", "chest", "drawers", "seating", "clothes", "refrigerator", "fridge", "oven", "stove",
                "cup", "book", "bottle", "wardrobe", "lamp", "desk", "mirror", "washer")
#: Every room type the labeller may write (``living room``, ``waiting area``, ...), and the ones the old prompt
#: named besides.
ROOM_WORDS = tuple(sorted({str(label).replace("_", " ") for label in DEFAULT_LABEL_SET if label != "unknown"}
                          | {"living", "dining", "corridor", "laundry", "storage", "en-suite", "ensuite", "basement",
                             "closet", "pantry", "bathrooms", "bedrooms"}))
#: The numbers a system prompt may hold: the word limits of the reply's three readings and of ``why``, the range
#: of ``p`` (twice: its range and its two ends), and "need not sum to 100". Anything else is a value we chose.
ALLOWED_NUMBERS = {"0", "100", "12", "20", "25"}

FULL = SearchNode(3, ROOM, "bedroom", tentative=True, footprint_m2=9.0, length_m=3.5, width_m=3.0, length_final=False,
                  walls_seen=3, bounded=False, covered_fraction=0.44, covered_m2=4.0, left_m2=5.0, open_perimeter_m=2.2,
                  frontier_clusters=1, exits=1, doorways=1, unseen_m2=12.0, objects=("bed",), connected=("hallway",),
                  glimpsed=("sink",), searched_s=12.0, last_inside_ago_s=3.0, target_seen=False)


def words_in(text, words):
    """The ``words`` that occur in ``text`` as whole words, singular or plural."""
    return sorted(word for word in words if re.search(r"(?<![a-z])%ss?(?![a-z])" % re.escape(word), text.lower()))


class Scripted:
    def __init__(self, reply):
        self.reply, self.systems, self.users = reply, [], []

    def chat_json(self, system, user, **kwargs):
        self.systems.append(system)
        self.users.append(user)
        return self.reply


# -- the registry ------------------------------------------------------------------------------------------------------
def test_the_default_is_a_registered_version_tuned_on_the_model_the_laptop_runs():
    assert DEFAULT_PROMPT in PROMPTS and prompt_version() is PROMPTS[DEFAULT_PROMPT] is prompt_version(None)
    assert prompt_version(DEFAULT_PROMPT).tuned_on == "qwen2.5:3b-instruct"
    assert search_node_oracle.SYSTEM_PROMPT == PROMPTS[DEFAULT_PROMPT].system, "the names the module always exported"
    assert search_node_oracle.USER_PROMPT_TEMPLATE == PROMPTS[DEFAULT_PROMPT].user_template
    assert search_node_oracle.OPENINGS_SUPPLEMENT == PROMPTS[DEFAULT_PROMPT].openings_supplement


def test_a_version_is_found_by_name_and_an_unregistered_name_is_refused_with_the_registered_ones_listed():
    for name in VERSIONS:
        assert prompt_version(name) is PROMPTS[name] and PROMPTS[name].name == name
    trial = dataclasses.replace(PROMPTS[DEFAULT_PROMPT], name="trial", system="another wording { " + TRIAL_REPLY)
    assert prompt_version(trial) is trial, "a wording under trial is used as given, registered or not"
    with pytest.raises(ValueError, match="no node-oracle prompt version is named 'facts-of-tomorrow'; registered: "
                                         + re.escape(", ".join(VERSIONS))):
        prompt_version("facts-of-tomorrow")
    with pytest.raises(ValueError, match="registered"):
        prompt_version(7)


def test_two_versions_cannot_share_a_name_or_a_text():
    first = PROMPTS[DEFAULT_PROMPT]
    with pytest.raises(ValueError, match="two node-oracle prompt versions are named"):
        search_node_prompts._registry(first, dataclasses.replace(first, system=first.system + " more"))
    with pytest.raises(ValueError, match="are the same text"):
        search_node_prompts._registry(first, dataclasses.replace(first, name="another-name"))
    with pytest.raises(ValueError, match="lacks {nodes}"):
        dataclasses.replace(first, user_template="TARGET: {target} HOUSE: {house} ({n})")
    with pytest.raises(ValueError, match="non-blank"):
        dataclasses.replace(first, system="  ")
    # A wording that forgets a key the parser reads is refused where it is written, not found by a failed run.
    with pytest.raises(ValueError, match="does not ask for the reply's pass"):
        dataclasses.replace(first, system=first.system.replace('"pass":', '"verdict":'))


def test_the_digest_follows_the_text_and_the_record_names_version_model_and_digest():
    first = PROMPTS[DEFAULT_PROMPT]
    assert re.fullmatch(r"[0-9a-f]{64}", first.sha256)
    assert dataclasses.replace(first, name="renamed").sha256 == first.sha256, "a name is not the wording"
    for field in ("system", "openings_supplement", "user_template"):
        changed = dataclasses.replace(first, **{field: getattr(first, field) + " "})
        assert changed.sha256 != first.sha256, "%s is part of what the model is told" % field
    assert first.describe() == {"version": first.name, "tuned_on": first.tuned_on, "sha256": first.sha256,
                                "system_chars": len(first.system)}


def test_the_oracle_sends_the_version_it_was_given_and_the_default_otherwise():
    reply = {"home": "a", "house": "b", "stage": "c", "pass": "first", "nodes": [{"id": 3, "why": "w", "p": 40}]}
    trial = NodeOraclePrompt(name="trial", tuned_on="some-model", system=TRIAL_SYSTEM,
                             openings_supplement=" TRIAL OPENINGS",
                             user_template="T={target} H={house} N={n}\n{nodes}\nEND")
    client = Scripted(reply)
    oracle = SearchNodeOracle(client, prompt=trial)
    assert oracle.version is trial and SearchNodeOracle(Scripted(reply)).version is PROMPTS[DEFAULT_PROMPT]
    assert SearchNodeOracle(Scripted(reply), prompt=DEFAULT_PROMPT).version is PROMPTS[DEFAULT_PROMPT]
    oracle.probabilities("bed", [FULL], SearchContext("nothing yet", action_time_s=1.0))
    assert client.systems == [TRIAL_SYSTEM]
    assert client.users[0].startswith("T=bed H=nothing yet N=1\nid=3  ROOM") and client.users[0].endswith("\nEND")
    door = SearchNode(200001, OPENING, "doorway")
    assert oracle.system([FULL, door]) == TRIAL_SYSTEM + " TRIAL OPENINGS" == system_prompt([FULL, door], None, trial)
    assert format_prompt("bed", [FULL], None, trial) == oracle.prompt("bed", [FULL])
    with pytest.raises(ValueError, match="registered"):
        SearchNodeOracle(client, prompt="no-such-version")


# -- what every version holds to -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", VERSIONS)
def test_a_version_tells_the_model_to_use_its_own_knowledge_of_homes_and_buildings(name):
    """The principle (2026-11-16): the model works out by itself where the object makes sense and derives each
    node's probability from that. Every version says so in so many words."""
    flat = " ".join(PROMPTS[name].system.split())
    for phrase in ("your own reasoning and your own knowledge of standard homes and buildings",
                   "in which rooms furniture and objects of each kind are normally found",
                   "the judgement is yours", "The TARGET line names it", "probability, in percent"):
        assert phrase in flat, phrase
    assert re.search(r"[Dd]erive every probability from that", flat)


@pytest.mark.parametrize("name", VERSIONS)
def test_a_version_holds_no_category_knowledge_no_number_of_ours_and_no_rule(name):
    """No object named, no room type named, no probability or band, no list of a benchmark's categories, no worked
    example, none of the old prompt's rules. Until 2026-11-16 the prompt listed the rooms of each HM3D goal
    category, the rooms a typical home has and a band of numbers per room status."""
    version = PROMPTS[name]
    for label, text in (("the system prompt", version.system), ("the openings supplement", version.openings_supplement),
                        ("the user template", version.user_template)):
        assert words_in(text, OBJECT_WORDS) == [], "%s of %s names an object" % (label, name)
        assert words_in(text, ROOM_WORDS) == [], "%s of %s names a room type" % (label, name)
        assert "%" not in text and "EXAMPLE" not in text.upper(), (label, name)
    assert set(re.findall(r"\d+", version.system)) <= ALLOWED_NUMBERS, re.findall(r"\d+", version.system)
    assert re.findall(r"\d+", version.openings_supplement) == [] and re.findall(r"\d+", version.user_template) == []
    for gone in ("STEP 1", "HOW TO JUDGE", "COVERAGE FIRST", "TIME IS SECONDARY", "SPREAD", "EXITS.", "Typical homes",
                 "never below", "or more", "at most HALF", "may have MISSED", "is FINAL", "no bigger",
                 "WHERE IT LIVES", "home type", "arrived_by", "STOREY", "storey", "stairs", "as good as scanned",
                 "is the best bet", "NEVER", "ALWAYS", "must", "should"):
        assert gone not in version.system and gone not in version.openings_supplement, (name, gone)


@pytest.mark.parametrize("name", VERSIONS)
def test_a_version_keeps_travel_out_and_says_nothing_of_where_the_robot_is(name):
    version = PROMPTS[name]
    flat = " ".join(version.system.split())
    assert "Leave travel out: a planner weighs your probabilities against the distances" in flat
    for word in ("here=", "metres away", "m away", "nearest", "closest", "far from", "close to", "route", "path length"):
        assert word not in version.system and word not in version.user_template, (name, word)


@pytest.mark.parametrize("name", VERSIONS)
def test_a_version_asks_for_the_reply_the_parser_reads_and_frames_the_query_with_the_targets_name(name):
    version = PROMPTS[name]
    flat = " ".join(version.system.split())
    for key in REPLY_KEYS:
        assert '"%s":' % key in flat, key
    order = [flat.index('"%s":' % key) for key in ("home", "house", "stage", "pass", "nodes")]
    assert order == sorted(order), "keys in the order the reading is written in"
    assert '"pass":"first|second"' in flat and "Reply with ONLY this JSON, keys in this order" in flat
    assert "independent" in flat and "need not sum to 100" in flat
    for field in USER_TEMPLATE_FIELDS:
        assert version.user_template.count(field) >= 1, field
    assert version.user_template.startswith("TARGET: {target}"), "the object of the episode is named first, in every query"
    reply = {"home": "a", "house": "b", "stage": "c", "pass": "second", "nodes": [{"id": 3, "why": "w", "p": 40}]}
    result = SearchNodeOracle(Scripted(reply), prompt=name).probabilities("potted plant", [FULL])
    assert result.source == "llm" and result.probs == {3: 0.4} and result.pass_verdict == "second"


@pytest.mark.parametrize("name", VERSIONS)
def test_a_version_explains_every_field_a_line_carries_and_nothing_the_code_does_not_write(name):
    system = PROMPTS[name].system
    line = format_node(FULL, 1.0)
    on_line = sorted(set(re.findall(r"  ([a-z_]+)=", line)))
    in_legend = sorted(set(key[:-1] for key in re.findall(r"(?m)^([a-z_]+=)", system)))
    assert on_line == in_legend, "every key on a room line has its line in the legend, and the legend no other"
    for label in ("connected to:", "seen:", "glimpsed through its exits:", "HOUSE"):
        assert label in system, label
    for word in ('"?"', '"all sides mapped"', '"at least"', '"+"', '"open edge"', "never_entered", "entered",
                 "scanned(<how>, <how long ago>)", '"xN"', '"+N more"'):
        assert word in system, word


@pytest.mark.parametrize("name", VERSIONS)
def test_a_version_is_short(name):
    assert len(PROMPTS[name].system) < 3800, "short and to the point: the prompt these replaced was 14,751 characters"
