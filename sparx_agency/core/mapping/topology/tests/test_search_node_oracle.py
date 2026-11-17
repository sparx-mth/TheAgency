"""The node oracle: the prompt, the parse, the contract the planner relies on."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.topology import search_node_oracle
from sparx_agency.core.mapping.topology.search_node_oracle import (
    OMITTED_PERCENT, OPENING, OPENINGS_SUPPLEMENT, P_CEILING, PASS_FIRST, PASS_SECOND, ROOM, SOURCE_FLAT,
    SOURCE_LLM, SYSTEM_PROMPT, UNEXPLORED_FLOOR, USABLE_SOURCES, USER_PROMPT_TEMPLATE, SearchContext, SearchNode,
    SearchNodeOracle, area_words, budget_line, coarse_duration, coarse_metres, coarse_seconds, counted,
    floor_unexplored, format_coverage, format_exits, format_mapped, format_node, format_prompt, format_size,
    keep_unentered, kinds, normalise, parse_pass, parse_reply, system_prompt, written_as_fractions)
from sparx_agency.core.mapping.topology.search_node_prompts import DEFAULT_PROMPT, PROMPTS


class Scripted:
    def __init__(self, reply=None, raise_=False):
        self.reply, self.raise_, self.calls, self.kwargs, self.systems, self.users = reply, raise_, 0, [], [], []

    def chat_json(self, system, user, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        self.systems.append(system)
        self.users.append(user)
        if self.raise_:
            raise TimeoutError("away")
        return self.reply


class Plain:
    """A client without the ``reasoning`` keyword, as the older ROS scene-graph stack has."""

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def chat_json(self, system, user):
        self.calls += 1
        return self.reply


KITCHEN = SearchNode(0, ROOM, "kitchen", area_m2=12.4, frontier_clusters=0, searched_s=43.0, last_inside_ago_s=31.0,
                     objects=("fridge", "sink", "Sink"), scanned="full rotation", scanned_ago_s=125.0)
LIVING = SearchNode(1, ROOM, "living_room", area_m2=25.0, frontier_clusters=1, searched_s=20.0, last_inside_ago_s=0.0,
                    here=True, objects=("sofa", "television"))
UNKNOWN = SearchNode(2, ROOM, "unknown", area_m2=18.0, frontier_clusters=3)
DOORWAY = SearchNode(200001, OPENING, "doorway", objects=("toilet", "Toilet", "sink"), via="room 11 (type=unknown)")
GAP = SearchNode(200002, OPENING, "gap")
NODES = [KITCHEN, LIVING, UNKNOWN, DOORWAY, GAP]
HOUSE = ("rooms found: kitchen (scanned), living_room (entered); 1 room unidentified; 0 openings not yet looked into; "
         "actions used about 150 of 500")

# -- the prompt: the default version's texts, and what went with the multi-storey supplement ---------------------------
# What every prompt version must hold to -- the model's own knowledge, no category knowledge, no rule, the reply
# format, the legend -- is checked over the whole registry in ``test_search_node_prompts.py``.
def test_the_module_exports_the_default_versions_texts_and_asks_whether_the_target_is_in_each_room():
    """What the one number is (since 2026-11-17): the probability that the target is in the room -- in the part
    of it the robot has not seen yet, since a detector checks every frame. A node is a room, no longer the room
    with the unseen space its exits open onto; and what the search so far means for a room is said in words,
    beside the fields -- facts and their meaning, never a value."""
    default = PROMPTS[DEFAULT_PROMPT]
    assert (SYSTEM_PROMPT, OPENINGS_SUPPLEMENT, USER_PROMPT_TEMPLATE) == (default.system, default.openings_supplement,
                                                                         default.user_template)
    flat = " ".join(SYSTEM_PROMPT.split())
    for phrase in ("You reason like a person who knows how homes are laid out.",
                   "Each line under NODES is one room it can search next.",
                   "A detector checks every camera frame for the target, so the target is in a place the robot has "
                   "not seen yet.",
                   "For every node, estimate the probability, in percent, that the target is in the part of that room "
                   "the robot has not seen yet.",
                   '"p": the probability in percent, from 0 (certain the target is not there) to 100 (certain it is).'):
        assert phrase in flat, phrase
    for gone in ("together with the unseen space its exits open onto", "the part of it not checked yet"):
        assert gone not in flat and gone not in USER_PROMPT_TEMPLATE, gone


def test_the_multi_storey_supplement_and_everything_it_asked_for_are_gone():
    for name in ("STAIRS", "STAIRS_SUPPLEMENT", "STOREY_LINES_TEMPLATE", "parse_home_here", "UNEXPLORED_ELSEWHERE",
                 "HOME_ELSEWHERE"):
        assert not hasattr(search_node_oracle, name), "%s went with the multi-storey supplement" % name
    for gone in ("STOREY", "storey", "stairs", "home_here"):
        assert gone not in SYSTEM_PROMPT and gone not in OPENINGS_SUPPLEMENT and gone not in USER_PROMPT_TEMPLATE, gone


def test_the_parser_reads_the_reply_the_prompt_asks_for():
    reply = {"home": "a", "house": "b", "stage": "c", "pass": "first", "nodes": [{"id": 0, "why": "w", "p": 40}]}
    result = SearchNodeOracle.score(reply, [KITCHEN])
    assert result.probs == {0: 0.4} and result.reading == {"home": "a", "house": "b", "stage": "c", "pass": "first"}


def test_travel_is_the_planners_and_nothing_on_a_line_says_where_the_robot_is():
    """The division of labour: the model gives the probability that the target is in each room; distance, and the
    weighing of distance against the probabilities, is RPT*'s. So no line carries a distance, a position or the
    robot's whereabouts -- ``here=yes`` was on the line until 2026-11-16."""
    assert "Leave travel out: a planner weighs your probabilities against the distances" in " ".join(SYSTEM_PROMPT.split())
    assert LIVING.here and "here" not in format_node(LIVING) and "here" not in format_node(LIVING, 1.0)
    assert format_node(LIVING) == format_node(SearchNode(1, ROOM, "living_room", area_m2=25.0, frontier_clusters=1,
                                                         searched_s=20.0, last_inside_ago_s=0.0,
                                                         objects=("sofa", "television"))), "standing in it changes nothing"
    # Nor does how long ago the robot left a room: it is read only for whether the room was ever entered.
    assert format_node(LIVING) == format_node(SearchNode(1, ROOM, "living_room", area_m2=25.0, frontier_clusters=1,
                                                         searched_s=20.0, last_inside_ago_s=400.0,
                                                         objects=("sofa", "television")))
    text = format_prompt("bed", NODES, SearchContext(HOUSE, action_time_s=1.0))
    for word in ("here=", "distance", " away", "metres from", "m from", "nearest", "closest"):
        assert word not in text, word
    assert "here=" not in SYSTEM_PROMPT and "distance to" not in SYSTEM_PROMPT


def test_every_call_names_the_object_the_episode_searches_for():
    client = Scripted({"nodes": [{"id": 0, "p": 20}]})
    oracle = SearchNodeOracle(client)
    for target in ("potted plant", "tv", "gym equipment"):
        oracle.probabilities(target, [KITCHEN], SearchContext(HOUSE))
        assert client.users[-1].startswith("TARGET: %s\n\nHOUSE: %s\n\nNODES (1):\n" % (target, HOUSE))
        assert client.users[-1].count(target) == 2, "named on its own line, and again in the question that ends the prompt"
        assert client.systems[-1] == SYSTEM_PROMPT and target not in client.systems[-1], "the system prompt names no object"
    assert client.calls == 3, "another target is another prompt"
    assert format_prompt("bed", [KITCHEN]).startswith("TARGET: bed\n\nHOUSE: nothing known yet\n")
    # The question is asked again where a small model reads last: after the node lines, with the target's name.
    assert format_prompt("bed", [KITCHEN]).endswith(
        "\n\nFor each of the 1 nodes above: how likely is it that the target (bed) is in the part of that room not "
        "seen yet? Score all 1, independently. JSON only.")


# -- the room line ---------------------------------------------------------------------------------------------------
def test_the_prompt_lists_every_node_in_the_format_the_legend_explains():
    """A room line holds four things (2026-11-17): its type, the floor mapped out of its floor, the frontiers left
    and the effort spent inside. An unclassified room -- and no other -- lists the objects seen in it."""
    text = format_prompt("bed", NODES, SearchContext(HOUSE))
    assert "TARGET: bed" in text and "NODES (5)" in text and "HOUSE: " + HOUSE in text
    assert format_node(KITCHEN) == "id=0  ROOM  type=kitchen  mapped=unknown of 12 m2  frontier=0  searched=40s"
    assert format_node(LIVING) == "id=1  ROOM  type=living_room  mapped=unknown of 25 m2  frontier=1  searched=20s"
    assert format_node(UNKNOWN) == ("id=2  ROOM  type=unknown (seen: nothing yet)  mapped=unknown of 18 m2  frontier=3  "
                                    "searched=never")
    assert all(line.startswith("id=") for line in text.split("NODES (5):\n")[1].split("\n\n")[0].splitlines())
    weak = SearchNode(4, ROOM, "bedroom", tentative=True, area_m2=14.0)
    assert "type=bedroom?  " in format_node(weak) and "type=unknown (seen: nothing yet)  " in format_node(replace_label(weak, "unknown"))
    # Everything else the scene graph stores about the room is on the node for the record and not on its line.
    full = SearchNode(3, ROOM, "bedroom", area_m2=12.4, observed_m2=6.1, frontier_clusters=1, searched_s=43.0,
                      last_inside_ago_s=5.0, objects=("bed", "wardrobe"), connected=("hallway",), exits=2, doorways=1,
                      unseen_m2=30.0, glimpsed=("sink",), scanned="full rotation", scanned_ago_s=60.0, footprint_m2=14.0,
                      length_m=4.0, width_m=3.5, walls_seen=4, bounded=True, covered_fraction=0.8, covered_m2=11.2,
                      left_m2=2.8, target_seen=True, here=True)
    assert format_node(full, 1.0) == "id=3  ROOM  type=bedroom  mapped=6 of 12 m2 (50%)  frontier=1  searched=40 actions"
    for gone in ("size=", "covered=", "exits=", "status=", "connected to", "seen:", "glimpsed", "target:", "here"):
        assert gone not in format_node(full, 1.0), gone


def replace_label(node, label):
    return SearchNode(node.id, node.kind, label, tentative=node.tentative, area_m2=node.area_m2)


def test_effort_is_written_in_actions_when_the_context_gives_the_action_time():
    """One unit on the whole prompt: the HOUSE line's budget is in actions, so the time spent in a room is too --
    the model can set one beside the other. A rig that does not count actions keeps time. A room the robot has
    never been inside reads ``never`` in either unit: that it was not entered is what its line says of the search."""
    assert format_node(KITCHEN, 1.0) == "id=0  ROOM  type=kitchen  mapped=unknown of 12 m2  frontier=0  searched=40 actions"
    assert "searched=10 actions" in format_node(LIVING, 2.0), "20 s at 2 s an action"
    assert format_node(UNKNOWN).endswith("searched=never") and format_node(UNKNOWN, 1.0).endswith("searched=never")
    stepped_in = SearchNode(5, ROOM, "hallway", area_m2=6.0, searched_s=3.0, last_inside_ago_s=40.0)
    assert format_node(stepped_in, 1.0).endswith("searched=0 actions"), "entered and crossed in three actions: not never"
    assert coarse_duration(43.0, 1.0) == "40 actions" and coarse_duration(46.0, 1.0) == "50 actions"
    assert coarse_duration(4.0, 1.0) == "0 actions" and coarse_duration(250.0, 0.5) == "480 actions", "4 min at 2 an s"
    assert coarse_duration(43.0) == "40s" and coarse_duration(150.0, None) == "2min" and coarse_duration(None, 1.0) == "never"
    # The steps are the time rendering's own, so the line changes exactly when it used to: tens under a minute's
    # worth, whole minutes' worth above.
    for seconds in (0, 4, 6, 43, 59, 60, 89, 91, 125, 128, 150, 151, 3600):
        text = coarse_seconds(seconds)
        expected = int(text[:-1]) if text.endswith("s") else 60 * int(text[:-3])
        assert coarse_duration(seconds, 1.0) == "%d actions" % expected, seconds
    text = format_prompt("bed", [KITCHEN], SearchContext(HOUSE, action_time_s=1.0))
    assert "searched=40 actions" in text and "40s" not in text
    assert "searched=40s" in format_prompt("bed", [KITCHEN], SearchContext(HOUSE))
    assert SearchNode(5, ROOM, "bedroom", scanned="fragment").status_words(1.0) == "scanned(fragment)", "no step known: no ago"
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="action_time_s"):
            SearchContext(HOUSE, action_time_s=bad)


def test_the_objects_of_an_unclassified_room_are_shown_with_their_counts_the_most_numerous_first():
    """The graph stores one entry per confirmed object; four chairs and a table say more than "chair, table". They
    are on the line of a room that has no classification yet, and then only (2026-11-17): a classified room's
    type stands for them."""
    assert counted(("chair", "table", "Chair", " chair ", "chair")) == "chair x4, table"
    assert counted(("sofa", "bed")) == "bed, sofa" and counted(()) == "" and counted(("", "  ")) == ""
    assert counted(("b", "a", "a", "c", "c"), limit=2) == "a x2, c x2, +1 more", "the rarest kind is the one cut, and the cut shows"
    assert counted(("b", "a"), limit=2) == "a, b" and counted(("a",), limit=0) == "+1 more"
    objects = ("chair",) * 6 + ("table", "plant")
    unnamed = SearchNode(6, ROOM, "unknown", area_m2=14.0, objects=objects)
    assert format_node(unnamed) == ("id=6  ROOM  type=unknown (seen: chair x6, plant, table)  mapped=unknown of 14 m2  "
                                    "frontier=0  searched=never")
    dining = SearchNode(6, ROOM, "dining_room", area_m2=14.0, objects=objects)
    assert format_node(dining) == "id=6  ROOM  type=dining_room  mapped=unknown of 14 m2  frontier=0  searched=never"
    assert "seen" not in format_node(SearchNode(6, ROOM, "dining_room", tentative=True, objects=objects)), "nor a tentative type's"
    crowded = SearchNode(7, ROOM, "unknown", objects=tuple("k%02d" % i for i in range(12)) + ("k11",))
    assert "type=unknown (seen: k11 x2, k00, k01, k02, k03, k04, k05, k06, +4 more)  " in format_node(crowded), (
        "eight kinds at most, and the model is told the list was cut: it reads the absence of an object from it")
    assert format_prompt("bed", [unnamed]) != format_prompt("bed", [SearchNode(6, ROOM, "unknown", area_m2=14.0)])
    assert format_prompt("bed", [dining]) == format_prompt("bed", [SearchNode(6, ROOM, "dining_room", area_m2=14.0)])
    # What was glimpsed through an exit is a kind, not a census (an opening node's line, the deprecated mode).
    assert kinds(("Sink ", "sink", "bed", "", " ")) == ("bed", "sink")


def test_what_the_planner_and_the_record_keep_is_not_on_the_line():
    """Removed from the line on 2026-11-17, and still on the node for the recording: the rooms a door joins it to
    (connections are the planner's, with the distances), its exits and what was glimpsed through them, its status
    word, the size of its footprint, and the note that the target's kind was detected in it and not reached (a
    confirmed target is a landmark node of the loop's own, at a fixed probability, without the model)."""
    plain = SearchNode(7, ROOM, "hallway", area_m2=6.0, observed_m2=3.0, frontier_clusters=1, last_inside_ago_s=4.0)
    line = format_node(plain)
    assert line == "id=7  ROOM  type=hallway  mapped=3.0 of 6.0 m2 (50%)  frontier=1  searched=0s"
    for extra in (dict(connected=("bedroom", "kitchen", "bedroom", "unknown")), dict(exits=2, doorways=1, unseen_m2=30.0),
                  dict(glimpsed=("toilet", "sink")), dict(scanned="full rotation", scanned_ago_s=40.0),
                  dict(target_seen=True), dict(objects=("sink", "toilet")), dict(here=True),
                  dict(footprint_m2=9.0, length_m=4.0, width_m=1.5, walls_seen=3, bounded=False, open_perimeter_m=2.0),
                  dict(covered_fraction=0.9, covered_m2=8.0, left_m2=1.0)):
        fuller = SearchNode(7, ROOM, "hallway", area_m2=6.0, observed_m2=3.0, frontier_clusters=1, last_inside_ago_s=4.0, **extra)
        assert format_node(fuller) == line, "%s is for the record, not for the model" % sorted(extra)
        assert format_prompt("toilet", [fuller]) == format_prompt("toilet", [plain]), "so it buys no call either"
    for word in ("detected here", "not reached", "connected to", "exits=", "status=", "glimpsed through its exits"):
        assert word not in SYSTEM_PROMPT, word
    # The helpers that wrote those fields are kept (the exits of a room are still counted and recorded).
    assert format_exits(0) == "0" and format_exits(3) == "3" and format_exits(3, 2) == "3 (2 doorways)"
    assert format_exits(1, 5) == "1 (1 doorway)", "no more doorways than exits"
    assert format_exits(2, 1, 30.4) == "2 (1 doorway, ~30m2 unseen behind them)"
    assert format_exits(2, 0, 7.6) == "2 (~8m2 unseen behind them)" and format_exits(0, 0, 12.0) == "0"
    assert format_exits(2, 0, 0.2) == "2", "under half a square metre is nothing to say"
    assert "RAISE" not in SYSTEM_PROMPT and "fits the target" not in SYSTEM_PROMPT, "no rule on what an exit is worth"
    assert "  OPENING  " not in SYSTEM_PROMPT, "no opening node line in the room-level prompt"
    assert system_prompt([plain, KITCHEN], SearchContext(HOUSE)) == SYSTEM_PROMPT, "no exit node offered: no supplement"


def test_the_mapped_floor_is_given_as_areas_and_as_a_share_the_model_reads_and_does_not_compute():
    """``mapped=6 of 12 m2 (50%)``: the floor the camera has seen from detection range out of the room's floor,
    and their share -- a room of 4 m2 and one of 40 m2 can both stand at 50 %, so the areas are said too. The
    floor is the free cells alone (floor the robot can stand on): no furniture island, no settled pocket."""
    def mapped(area, seen, **fields):
        return format_mapped(SearchNode(0, ROOM, "unknown", area_m2=area, observed_m2=seen, **fields))
    assert mapped(12.4, 6.1) == "6 of 12 m2 (50%)" and mapped(40.0, 20.0) == "20 of 40 m2 (50%)"
    assert mapped(4.0, 2.0) == "2.0 of 4.0 m2 (50%)", "the same share of a tenth of the floor"
    assert mapped(12.0, 0.0) == "0 of 12 m2 (0%)" and mapped(12.0, 12.0) == "12 of 12 m2 (100%)"
    # A hundred is kept for a room seen whole, and under it the mapped part never reads as the whole floor.
    assert mapped(12.0, 11.8) == "11 of 12 m2 (90%)" and mapped(6.2, 5.6) == "5.6 of 6.2 m2 (90%)"
    assert mapped(19.96, 9.98) == "10 of 20 m2 (50%)" and mapped(6.2, 6.16) == "6.1 of 6.2 m2 (90%)"
    assert mapped(12.0, 11.95) == "12 of 12 m2 (100%)" and mapped(12.0, 13.5) == "12 of 12 m2 (100%)", "never more than the room"
    assert mapped(12.0, -1.0) == "0 of 12 m2 (0%)"
    # The footprint (furniture and pockets included) is NOT the room's floor here: only where the robot can stand.
    assert mapped(8.0, 4.0, footprint_m2=11.0, covered_m2=7.0, covered_fraction=0.64, left_m2=4.0) == "4.0 of 8.0 m2 (50%)"
    # Under ten square metres a whole number says too little: one decimal, and never a room of nothing.
    assert mapped(1.2, 0.96) == "1.0 of 1.2 m2 (80%)" and mapped(0.4, 0.1) == "0.1 of 0.4 m2 (30%)"
    assert mapped(9.94, 5.0) == "5.0 of 9.9 m2 (50%)" and mapped(9.96, 5.0) == "5 of 10 m2 (50%)"
    assert mapped(0.03, 0.03) == "0.1 of 0.1 m2 (100%)" and mapped(0.03, 0.0) == "0.0 of 0.1 m2 (0%)"
    # Without the observation record (no sight ledger) the room's floor is said and the mapped part is not.
    assert mapped(12.0, None) == "unknown of 12 m2" and mapped(0.4, None) == "unknown of 0.4 m2"
    assert mapped(0.0, None) == "unknown" and mapped(0.0, 0.0) == "unknown", "a room of no known size"
    # The share is shown in tens, so a frame that saw three more cells does not change the prompt.
    a = format_prompt("toilet", [SearchNode(3, ROOM, "bedroom", area_m2=12.0, observed_m2=9.7)])
    b = format_prompt("toilet", [SearchNode(3, ROOM, "bedroom", area_m2=12.0, observed_m2=10.1)])
    assert a == b and a != format_prompt("toilet", [SearchNode(3, ROOM, "bedroom", area_m2=12.0, observed_m2=10.9)])
    flat = " ".join(SYSTEM_PROMPT.split())
    assert ("mapped= the floor of the room the robot has already seen, out of all its floor, in square metres and as a "
            "percentage. Only floor the robot can reach counts.") in flat
    assert ("frontier= how many places are left in the room where the mapped floor borders space not seen yet.") in flat
    assert ('searched= how many actions the robot has already spent searching inside the room. "never": it has not '
            "entered the room.") in flat





def _covered(covered_m2, fraction, left_m2, bounded):
    """The ``covered=`` words for a room with these numbers."""
    return format_coverage(SearchNode(0, ROOM, "unknown", bounded=bounded, covered_fraction=fraction,
                                      covered_m2=covered_m2, left_m2=left_m2))


def test_a_rooms_size_says_what_the_map_knows_and_does_not_call_it_final():
    """Room extents: the size reads ``all sides mapped`` when every side is located -- no part of the room lies
    outside the map -- and ``at least`` otherwise. Not "final": of 252 rooms the first wording declared final in 40
    HM3D recordings, 73 were over a quarter bigger by the end of the episode, 229 of the 233 growths being the
    partition joining or re-cutting rooms whose floor was mapped all along. The words claim what the reading knows.
    Since 2026-11-17 these words are no longer on a room's line (``mapped=`` is); the helpers are kept."""
    bounded = SearchNode(3, ROOM, "bedroom", area_m2=11.2, footprint_m2=12.6, length_m=4.1, width_m=3.05,
                         walls_seen=4, bounded=True, covered_fraction=0.87, covered_m2=10.9, left_m2=1.7, last_inside_ago_s=5.0)
    assert format_size(bounded) == "13m2 (4x3m, all sides mapped)"
    assert format_coverage(bounded) == "11m2 (80% of the room, 2m2 left)"
    partial = SearchNode(4, ROOM, "unknown", area_m2=6.0, footprint_m2=6.0, length_m=4.9, width_m=2.4,
                         length_final=False, width_final=True, walls_seen=3, bounded=False,
                         covered_fraction=0.43, covered_m2=2.6, left_m2=3.4, frontier_clusters=1)
    assert format_size(partial) == "at least 6m2 (5+ x 2.5m, 1 of 4 sides not mapped yet)"
    assert format_coverage(partial) == "3m2 (at most 50% of the room, at least 3m2 left)", "a bound rounds UP, a share DOWN"
    for line in (format_node(bounded), format_node(partial)):
        assert "size=" not in line and "covered=" not in line and "sides" not in line, line
    done = SearchNode(5, ROOM, "kitchen", footprint_m2=12.0, bounded=True, covered_fraction=0.996, covered_m2=11.95, left_m2=0.05)
    assert format_coverage(done) == "12m2 (all of it)"
    sliver = SearchNode(6, ROOM, "unknown", area_m2=0.3, footprint_m2=0.3, length_m=0.9, width_m=0.3,
                        length_final=False, width_final=False, walls_seen=1, bounded=False,
                        covered_fraction=1.0, covered_m2=0.3, left_m2=0.0)
    assert format_size(sliver) == "under 1m2 mapped so far (1+ x 0.5+m, 3 of 4 sides not mapped yet)"
    assert format_coverage(sliver) == "under 1m2 (all of the mapped part; the unmapped rest is unknown)"
    far = SearchNode(7, ROOM, "unknown", footprint_m2=7.0, bounded=False, covered_fraction=0.0, covered_m2=0.0, left_m2=7.0)
    assert format_coverage(far) == "0m2 (at most 0% of the room, at least 7m2 left)", "seen from beyond detection range"
    # The size's three shapes, as the extents of a 5 x 4 m room give them: closed, one side open, no side located.
    room = dict(footprint_m2=20.0, area_m2=20.0, length_m=5.0, width_m=4.0)
    assert format_size(SearchNode(0, ROOM, "unknown", walls_seen=4, bounded=True, **room)) == "20m2 (5x4m, all sides mapped)"
    assert format_size(SearchNode(0, ROOM, "unknown", walls_seen=3, bounded=False, length_final=False, **room)) == (
        "at least 20m2 (5+ x 4m, 1 of 4 sides not mapped yet)")
    assert format_size(SearchNode(0, ROOM, "unknown", walls_seen=0, bounded=False, length_final=False, width_final=False,
                                  **room)) == "at least 20m2 (5+ x 4+m, 4 of 4 sides not mapped yet)"
    # The length of boundary still open to unseen space rides in the brackets, from half a metre up.
    assert format_size(SearchNode(0, ROOM, "unknown", walls_seen=3, bounded=False, length_final=False,
                                  open_perimeter_m=2.2, **room)) == "at least 20m2 (5+ x 4m, 1 of 4 sides not mapped yet, open edge 2m)"
    assert "open edge" not in format_size(SearchNode(0, ROOM, "unknown", walls_seen=3, bounded=False, length_final=False,
                                                     open_perimeter_m=0.3, **room))
    assert "open edge" not in format_size(SearchNode(0, ROOM, "unknown", walls_seen=4, bounded=True,
                                                     open_perimeter_m=0.9, **room)), "a closed room has none to say"
    # The footprint (the floor under the furniture included) is the size; the labelled free cells alone are not.
    assert format_size(SearchNode(0, ROOM, "unknown", area_m2=18.5, footprint_m2=20.0, length_m=5.0, width_m=4.0,
                                  walls_seen=4, bounded=True)).startswith("20m2 ")
    assert coarse_metres(3.26) == "3.5" and coarse_metres(3.74) == "3.5" and coarse_metres(4.1) == "4"
    assert area_words(0.02) == "0m2" and area_words(0.6) == "under 1m2" and area_words(11.5) == "12m2"
    # Without the extents (an older rig, a storey before its first frame) the line carries the area alone.
    assert format_size(UNKNOWN) == "18m2" and format_coverage(UNKNOWN) is None
    assert format_size(SearchNode(6, ROOM, "unknown", area_m2=0.4)) == "under 1m2"
    assert format_size(SearchNode(7, ROOM, "unknown")) == "unknown size"
    # With the extents but no observation record (no sight ledger) the size is said and the coverage is not.
    unobserved = SearchNode(8, ROOM, "unknown", footprint_m2=12.0, length_m=4.0, width_m=3.0, walls_seen=4, bounded=True)
    assert format_size(unobserved) == "12m2 (4x3m, all sides mapped)" and format_coverage(unobserved) is None
    # No promise the reading cannot keep, on the line or in the prompt.
    for promise in ("FINAL", "final", "no bigger", "all walls seen", "walls not seen"):
        assert promise not in SYSTEM_PROMPT and promise not in format_node(bounded) and promise not in format_node(partial), promise


def test_the_coverage_words_hand_the_model_results_it_reads_and_does_not_compute():
    assert _covered(10.4, 0.87, 1.6, bounded=True) == "10m2 (80% of the room, 2m2 left)"
    assert _covered(11.95, 0.996, 0.05, bounded=True) == "12m2 (all of it)"
    assert _covered(0.0, 0.0, 12.0, bounded=True) == "0m2 (0% of the room, 12m2 left)"
    assert _covered(10.4, 0.87, 0.6, bounded=True) == "10m2 (80% of the room, under 1m2 left)"
    assert _covered(2.6, 0.43, 3.4, bounded=False) == "3m2 (at most 50% of the room, at least 3m2 left)"
    assert _covered(0.3, 1.0, 0.0, bounded=False) == "under 1m2 (all of the mapped part; the unmapped rest is unknown)"
    assert _covered(6.8, 0.97, 0.2, bounded=False) == "7m2 (all of the mapped part; the unmapped rest is unknown)", (
        "a bound that rounds up to 100% says nothing")
    assert _covered(5.5, 0.88, 0.7, bounded=False) == "6m2 (at most 90% of the room)", "no 'at least under 1m2'"
    assert _covered(0.0, 0.0, 7.0, bounded=False) == "0m2 (at most 0% of the room, at least 7m2 left)"
    assert _covered(3.0, 1.7, 0.0, bounded=True) == "3m2 (all of it)" and _covered(3.0, -0.2, 9.0, bounded=True) == (
        "3m2 (0% of the room, 9m2 left)"), "a share outside 0-1 is clamped, never printed"
    assert "coverage overrides" not in SYSTEM_PROMPT and "observed=" not in SYSTEM_PROMPT and "covered=" not in SYSTEM_PROMPT


def test_an_opening_node_is_the_deprecated_mode_and_brings_its_own_supplement():
    """Under ``exits_as_nodes`` (deprecated 2026-10-21) a doorway off the hallway with a toilet seen through it is a
    node named by the room it opens from and what the glimpse showed; the meaning of that line is appended to the
    system prompt only when such a node is on offer -- the meaning, and nothing on how to value it."""
    assert format_node(DOORWAY) == ("id=200001  OPENING  doorway off room 11 (type=unknown)  to space NOT seen yet  "
                                    "glimpsed through it: sink, toilet")
    assert format_node(GAP) == "id=200002  OPENING  gap off the mapped floor  to space NOT seen yet  glimpsed through it: nothing yet"
    text = format_prompt("couch", NODES, SearchContext(HOUSE))
    assert "NODES (5)" in text and "id=200001  OPENING" in text
    for part in ("An OPENING line is a doorway or gap at the edge of the mapped floor", "offered as a node of its own",
                 "estimate the probability that the target is in the space behind it",
                 '"glimpsed through it:" objects seen beyond it'):
        assert part in " ".join(OPENINGS_SUPPLEMENT.split()) and part not in SYSTEM_PROMPT, part
    assert "exits=" not in OPENINGS_SUPPLEMENT, "a room line carries no exits= since 2026-11-17"
    assert len(OPENINGS_SUPPLEMENT) < 400
    assert system_prompt([KITCHEN, GAP], SearchContext(HOUSE)) == SYSTEM_PROMPT + OPENINGS_SUPPLEMENT
    assert system_prompt([KITCHEN, LIVING], SearchContext(HOUSE)) == SYSTEM_PROMPT
    client = Scripted({"nodes": [{"id": 2, "p": 20}, {"id": 200002, "p": 60}]})
    SearchNodeOracle(client).probabilities("bed", [UNKNOWN, GAP])
    assert client.systems[0] == SYSTEM_PROMPT + OPENINGS_SUPPLEMENT


def test_there_is_no_staircase_node_and_no_storey_line():
    """The search is single-storey. The stair node, the multi-storey supplement, the storey lines of the user
    prompt and the ``home_here`` verdict they asked for were deleted on 2026-11-16, until the multi-storey work."""
    with pytest.raises(ValueError):
        SearchNode(100000, "stairs", "stairs up")
    for field in ("direction", "destination_visited", "destination", "arrived_by", "arrived_ago_s"):
        with pytest.raises(TypeError):
            SearchNode(0, ROOM, "kitchen", **{field: 1})
    with pytest.raises(TypeError):
        SearchContext(HOUSE, storey="upper")
    assert not hasattr(SearchContext(), "others") and not hasattr(SearchContext(), "multi_storey")
    text = format_prompt("bed", [KITCHEN, LIVING, UNKNOWN], SearchContext(HOUSE))
    assert "THIS STOREY" not in text and "OTHER STOREYS" not in text and "{storeys}" not in USER_PROMPT_TEMPLATE
    result = SearchNodeOracle.score({"home_here": "elsewhere", "storey": "upper", "nodes": [{"id": 2, "p": 20}]}, [UNKNOWN])
    assert not hasattr(result, "home_here") and not hasattr(result, "capped")
    assert result.reading == {} and result.probs[2] == pytest.approx(0.20), "a stray verdict is neither read nor applied"
    with pytest.raises(TypeError):
        SearchNodeOracle(Scripted({}), unexplored_elsewhere=0.1)


def test_the_room_status_words():
    assert SearchNode(5, ROOM, "unknown").status == "never_entered"
    assert SearchNode(5, ROOM, "unknown", last_inside_ago_s=0.0).status == "entered"
    assert SearchNode(5, ROOM, "bedroom", scanned="full rotation", scanned_ago_s=44.0).status == "scanned(full rotation, 40s ago)"
    assert SearchNode(5, ROOM, "bedroom", scanned="fragment").status == "scanned(fragment)", "no step known: no ago"
    scanned = SearchNode(5, ROOM, "unknown", scanned="seen from another room's scan", scanned_ago_s=10.0)
    assert not scanned.unexplored, "a scanned room is not an unknown place, whatever its type"
    assert scanned.status == "scanned(seen from another room's scan, 10s ago)"
    assert scanned.status_words(1.0) == "scanned(seen from another room's scan, 10 actions ago)"
    assert "status=" not in format_node(scanned) and "scanned" not in format_node(scanned, 1.0), "for the record since 2026-11-17"


def test_coarse_seconds_and_the_budget_line_keep_an_unchanged_map_an_unchanged_prompt():
    assert coarse_seconds(None) == "never"
    assert coarse_seconds(0) == "0s" and coarse_seconds(4) == "0s" and coarse_seconds(6) == "10s"
    assert coarse_seconds(43) == "40s" and coarse_seconds(59) == "60s"
    assert coarse_seconds(60) == "1min" and coarse_seconds(150) == "2min" and coarse_seconds(3600) == "60min"
    drifted = SearchNode(0, ROOM, "kitchen", area_m2=12.4, searched_s=44.0, last_inside_ago_s=33.0, objects=("fridge", "sink", "sink"),
                         scanned="full rotation", scanned_ago_s=128.0)
    assert format_prompt("bed", [KITCHEN]) == format_prompt("bed", [drifted]), "one more second changes nothing"
    context = SearchContext(HOUSE, action_time_s=1.0)
    assert format_prompt("bed", [KITCHEN], context) == format_prompt("bed", [drifted], context), "nor one more action"
    assert budget_line(0, 500) == "actions used about 0 of 500"
    assert budget_line(149, 500) == "actions used about 125 of 500" and budget_line(150, 500) == "actions used about 150 of 500"
    assert budget_line(174, 500) == budget_line(150, 500), "a 25-action bucket, rounded down"
    assert budget_line(600, 500) == "actions used about 500 of 500"


# -- the parse -----------------------------------------------------------------------------------------------------------
def test_a_good_reply_preserves_independent_node_probabilities_and_keeps_the_reading():
    reply = {"home": "bedroom", "house": "kitchen and living room found; one unidentified room", "stage": "early, first pass",
             "pass": "first",
             "nodes": [{"id": 0, "why": "kitchen, scanned, no beds", "p": 0},
                       {"id": 1, "why": "beds are not in living rooms", "p": 3},
                       {"id": 2, "why": "large unexplored room", "p": 27},
                       {"id": 200001, "why": "a bathroom behind it", "p": 62},
                       {"id": 200002, "why": "nothing seen", "p": 0},
                       {"id": 77, "why": "invented", "p": 40}],
             "elsewhere": 8}
    result = SearchNodeOracle.score(reply, NODES)
    assert result.source == "llm" and result.omitted == ()
    assert result.reading == {"home": "bedroom", "house": "kitchen and living room found; one unidentified room",
                              "stage": "early, first pass", "pass": "first"}
    assert result.pass_verdict == PASS_FIRST
    assert result.probs == pytest.approx({0: 0.0, 1: 0.03, 2: 0.27, 200001: 0.62, 200002: 0.0})
    assert result.elsewhere == pytest.approx(0.97 * 0.73 * 0.38)
    assert result.p_present == pytest.approx(1.0 - result.elsewhere)
    assert result.probability_model == "independent_search_success"
    assert 77 not in result.probs and result.reasons[200001] == "a bathroom behind it"
    assert result.spread == pytest.approx(0.62)
    assert result.floored == (), "no arithmetic touched the model's numbers"


def test_percentages_are_not_rescaled_and_omitted_nodes_get_a_small_probability():
    reply = {"nodes": [{"id": 0, "p": 50}, {"id": 2, "p": 150}], "elsewhere": 50}     # sums to 250, three nodes missing
    scores, elsewhere, reasons, omitted = parse_reply(reply, NODES)
    assert omitted == (1, 200001, 200002) and all(scores[nid] == OMITTED_PERCENT for nid in omitted)
    assert scores[2] == 100.0, "clamped into 0-100 before anything else"
    probs, rest = normalise(scores, elsewhere)
    assert probs[0] == 0.5 and probs[2] == P_CEILING
    assert rest == pytest.approx(0.5 * (1 - P_CEILING) * 0.99 ** 3)
    assert probs[2] > probs[0] > probs[1] > 0.0
    result = SearchNodeOracle.score(reply, NODES)
    assert result.omitted == (1, 200001, 200002) and result.reasons[1] == "(not scored by the model)"


def test_field_names_a_model_drifts_to_are_read():
    bare = SearchNodeOracle.score({"nodes": [{"id": 0, "p": 50}], "elsewhere": 50, "home": 7, "stage": ""}, [KITCHEN])
    assert bare.reading == {}, "a missing or non-string reading is simply absent"
    scores, _, _, _ = parse_reply({"nodes": [{"id": 0, "probability": 0.25}, {"id": 1, "percent": 30},
                                             {"id": 2, "score": 45}, {"id": 200001, "p": "10"}]}, NODES)
    assert scores[0] == 25.0 and scores[1] == 30.0 and scores[2] == 45.0 and scores[200001] == 10.0


def test_the_models_numbers_are_read_as_percentages_whatever_it_wrote_them_as():
    """The model need not write exactly what RPT* takes (2026-11-17): a number as text, a percent sign, an id as
    text, a whole reply in fractions of one. None of it is knowledge of a room or an object."""
    scores, _, _, omitted = parse_reply({"nodes": [{"id": "0", "p": "40%"}, {"id": 1, "p": " 35 % "}, {"id": 2, "p": "12.5"},
                                                   {"id": 200001, "p": 7}, {"id": 200002, "p": "a lot"}]}, NODES)
    assert scores[0] == 40.0 and scores[1] == 35.0 and scores[2] == 12.5 and scores[200001] == 7.0
    assert omitted == (200002,) and scores[200002] == OMITTED_PERCENT, "words are no number: forgotten, not ruled out"
    # Fractions of one, read off the reply as a whole: none above 1 and one strictly between 0 and 1.
    assert written_as_fractions([{"id": 0, "p": 0.4}, {"id": 1, "p": 0}, {"id": 2, "p": 1}])
    assert not written_as_fractions([{"id": 0, "p": 0.4}, {"id": 1, "p": 30}]), "one number above 1: percentages"
    assert not written_as_fractions([{"id": 0, "p": 1}, {"id": 1, "p": 0}]), "0 and 1 alone are the percentages asked for"
    assert not written_as_fractions([]) and not written_as_fractions([{"id": 0}, "prose"])
    scores, _, _, _ = parse_reply({"nodes": [{"id": 0, "p": 0.4}, {"id": 1, "p": 0}, {"id": 2, "p": "0.85"}]}, NODES[:3])
    assert scores == {0: 40.0, 1: 0.0, 2: 85.0}
    scores, _, _, _ = parse_reply({"nodes": [{"id": 0, "p": 1}, {"id": 1, "p": 0}, {"id": 2, "p": 1}]}, NODES[:3])
    assert scores == {0: 1.0, 1: 0.0, 2: 1.0}, "a model that means one percent is not read as certain"
    # A fraction is scaled once, whichever way it was recognised; an invented id does not vote on the scale.
    scores, _, _, _ = parse_reply({"nodes": [{"id": 0, "probability": 0.005}, {"id": 1, "probability": 0.5}]}, NODES[:2])
    assert scores == {0: 0.5, 1: 50.0}
    scores, _, _, _ = parse_reply({"nodes": [{"id": 0, "p": 30}, {"id": 1, "p": 60}, {"id": 99, "p": 0.5}]}, NODES[:2])
    assert scores == {0: 30.0, 1: 60.0}
    scores, _, _, _ = parse_reply({"nodes": [{"id": 0, "p": 0.4}, {"id": 1, "p": "90%"}, {"id": 2, "p": 0.1}]}, NODES[:3])
    assert scores == {0: 0.4, 1: 90.0, 2: 0.1}, "a number above 1 settles the scale: these are percentages"


@pytest.mark.parametrize("reply", [None, "prose", {}, {"nodes": "x"}, {"nodes": []},
                                   {"nodes": [{"id": 9, "p": 50}]}, {"nodes": [{"id": 0, "p": "many"}]}])
def test_an_unusable_reply_is_none_never_a_made_up_distribution(reply):
    assert parse_reply(reply, NODES) is None
    assert SearchNodeOracle.score(reply, NODES) is None


def test_all_zero_is_a_valid_belief_that_all_listed_searches_fail():
    assert parse_reply({"nodes": [{"id": 0, "p": 0}]}, [KITCHEN]) is not None
    scores, elsewhere, _, omitted = parse_reply({"nodes": [{"id": 0, "p": 0}], "elsewhere": 0}, [KITCHEN, LIVING])
    assert omitted == (1,) and scores[1] == OMITTED_PERCENT, "an omitted node is forgotten, not ruled out"
    all_out = SearchNodeOracle.score({"nodes": [{"id": 0, "p": 0}, {"id": 1, "p": 0}], "elsewhere": 100}, [KITCHEN, LIVING])
    assert all_out is not None and all_out.p_present == 0.0 and all_out.elsewhere == 1.0


def test_the_ceiling_keeps_rpt_star_from_dividing_by_zero():
    result = SearchNodeOracle.score({"nodes": [{"id": 0, "p": 100}], "elsewhere": 0}, [KITCHEN])
    assert result.probs[0] == P_CEILING < 1.0


# -- the oracle ------------------------------------------------------------------------------------------------------------
def test_the_oracle_routes_to_the_reasoning_model_and_reuses_an_unchanged_prompt():
    client = Scripted({"nodes": [{"id": 0, "p": 20}, {"id": 1, "p": 30}], "elsewhere": 50})
    oracle = SearchNodeOracle(client)
    first = oracle.probabilities("bed", [KITCHEN, LIVING])
    assert first.source == "llm" and not first.reused and client.calls == 1 and oracle.queries == 1
    assert client.kwargs[0] == {"reasoning": True}, "the one judgement per loop point goes to the reasoning model"
    assert client.systems[0] == SYSTEM_PROMPT, "rooms only: the system prompt, no supplement"
    again = oracle.probabilities("bed", [KITCHEN, LIVING])
    assert again.reused and client.calls == 1 and oracle.reuses == 1
    moved = oracle.probabilities("bed", [KITCHEN, SearchNode(1, ROOM, "living_room", area_m2=25.0, frontier_clusters=0)])
    assert not moved.reused and client.calls == 2, "a changed frontier count is a changed prompt"
    budget = oracle.probabilities("bed", [KITCHEN, LIVING], SearchContext(budget_line(300, 500)))
    assert not budget.reused and client.calls == 3, "a changed budget bucket is a changed prompt"
    with pytest.raises(ValueError):
        oracle.probabilities("bed", [])


def test_a_plain_client_without_the_reasoning_keyword_is_called_without_it():
    client = Plain({"nodes": [{"id": 0, "p": 60}], "elsewhere": 40})
    result = SearchNodeOracle(client).probabilities("bed", [KITCHEN])
    assert result.source == "llm" and client.calls == 1 and result.probs[0] == pytest.approx(0.6)


def test_model_trouble_degrades_to_a_uniform_the_caller_can_refuse():
    away = SearchNodeOracle(Scripted(raise_=True))
    result = away.probabilities("bed", NODES)
    assert result.source == "uniform_fallback" and result.raw_reply is None
    assert result.p_present + result.elsewhere == pytest.approx(1.0)
    junk = SearchNodeOracle(Scripted({"rooms": []}))
    result = junk.probabilities("bed", NODES)
    assert result.source == "uniform_fallback" and result.raw_reply == {"rooms": []}
    assert junk.probabilities("bed", NODES).source == "uniform_fallback", "a failed reply is never kept"


def test_the_flat_ablation_reads_every_node_at_one_probability_and_never_asks_the_model():
    """``flat_p``: what the search does without the oracle -- RPT* over travel cost alone -- measured on the same code."""
    client = Scripted({"nodes": [{"id": 0, "p": 90}, {"id": 1, "p": 5}]})
    oracle = SearchNodeOracle(client, flat_p=0.15)
    result = oracle.probabilities("bed", NODES, SearchContext(HOUSE))
    assert result.source == SOURCE_FLAT and SOURCE_FLAT in USABLE_SOURCES and "uniform_fallback" not in USABLE_SOURCES
    assert result.probs == {n.id: pytest.approx(0.15) for n in NODES} and result.spread == 0.0
    assert client.calls == 0 and oracle.queries == 0, "no call, so no answer to record or replay"
    assert result.pass_verdict is None and result.reading == {} and result.raw_reply is None
    assert result.elsewhere == pytest.approx(0.85 ** len(NODES)) and result.p_present == pytest.approx(1.0 - 0.85 ** len(NODES))
    assert SearchNodeOracle(client).probabilities("bed", [KITCHEN]).source == SOURCE_LLM, "None asks the model"
    for bad in (0.0, 1.0, -0.1, float("nan"), True):
        with pytest.raises(ValueError, match="flat_p"):
            SearchNodeOracle(client, flat_p=bad)


def test_search_node_refuses_an_unknown_kind():
    with pytest.raises(ValueError):
        SearchNode(0, "door", "door")


def test_two_likely_rooms_are_not_diluted_into_categorical_shares():
    result = SearchNodeOracle.score({"nodes": [{"id": 0, "p": 80}, {"id": 1, "p": 80}]}, [KITCHEN, LIVING])
    assert result.probs == {0: 0.8, 1: 0.8}
    assert result.elsewhere == pytest.approx(0.04)
    assert result.p_present == pytest.approx(0.96)
    assert "several may each hold a target, so the numbers are independent" in SYSTEM_PROMPT.replace("\n", " ")


# -- the pass verdict: shown and recorded (2026-10-07) -----------------------------------------------------------------
def test_the_pass_verdict_is_read_from_the_words_a_model_drifts_to():
    for text, verdict in (("first", PASS_FIRST), ("Second", PASS_SECOND), ("second pass", PASS_SECOND),
                          ("second, not first", PASS_SECOND), ("re-check the scanned rooms", PASS_SECOND),
                          ("revisit", PASS_SECOND), ("the home is covered", PASS_SECOND), ("explore first", PASS_FIRST),
                          ("1st", PASS_FIRST), ("2nd", PASS_SECOND), (1, PASS_FIRST), (2, PASS_SECOND),
                          ("", None), ("maybe", None), (None, None)):
        assert parse_pass({"pass": text}) == verdict, text
    assert parse_pass({}) is None and parse_pass("nonsense") is None, "a missing verdict is None: the first pass stands"
    second = SearchNodeOracle.score({"pass": "second", "nodes": [{"id": 0, "p": 30}]}, [KITCHEN])
    assert second.pass_verdict == PASS_SECOND and second.reading["pass"] == "second"
    assert SearchNodeOracle.score({"nodes": [{"id": 0, "p": 30}]}, [KITCHEN]).pass_verdict is None
    assert ('"pass": "second" if re-checking rooms already scanned is now worth more than what is left unexplored, else '
            '"first".') in " ".join(SYSTEM_PROMPT.split()), "the word is defined, not prescribed"


# -- a room never entered is never zero (2026-11-17) ----------------------------------------------------------------------
DOORWAY_WITH_TOILET = SearchNode(200003, OPENING, "doorway", objects=("toilet",), via="room 2 (type=unknown)")
UNKNOWN_ENTERED = SearchNode(3, ROOM, "unknown", area_m2=9.0, frontier_clusters=2, searched_s=12.0, last_inside_ago_s=20.0)
WEAK_BEDROOM_UNENTERED = SearchNode(4, ROOM, "bedroom", tentative=True, area_m2=14.0, frontier_clusters=1)


def test_a_room_never_entered_is_never_handed_on_at_zero_and_an_entered_rooms_zero_stands():
    """A zero says "certain the target is not there". Of a room the robot has been inside that is the model's
    verdict, and the room is then no node for the planner; of a room seen from outside only it is more than anybody
    knows, so the number is read as not given -- small, not nothing -- and the loop's floor for a room never
    entered then holds for it. The rule reads whether the robot was ever inside, and nothing else."""
    nodes = [KITCHEN, UNKNOWN, WEAK_BEDROOM_UNENTERED, UNKNOWN_ENTERED, GAP]
    reply = {"nodes": [{"id": 0, "why": "searched, none found", "p": 0}, {"id": 2, "why": "nothing seen there", "p": 0},
                       {"id": 4, "why": "not where it is kept", "p": 0}, {"id": 3, "why": "searched", "p": 0},
                       {"id": 200002, "why": "nothing glimpsed", "p": 0}]}
    scores, raised = keep_unentered(parse_reply(reply, nodes)[0], nodes)
    assert raised == (2, 4), "the two rooms never entered, whatever their type; no other node"
    assert scores[2] == scores[4] == OMITTED_PERCENT and scores[0] == scores[3] == scores[200002] == 0.0
    result = SearchNodeOracle.score(reply, nodes)
    assert result.probs == pytest.approx({0: 0.0, 2: 0.01, 4: 0.01, 3: 0.0, 200002: 0.0}) and result.floored == (2, 4)
    assert result.reasons[4] == "not where it is kept", "the model's words are kept beside the raised number"
    assert result.p_present > 0.0, "an all-zero reply no longer says that every listed search fails"
    # A number the model did give stands, however small; an omitted room was never zero.
    scored = {"nodes": [{"id": 2, "p": 1}, {"id": 4, "p": 3}]}
    assert keep_unentered(parse_reply(scored, [UNKNOWN, WEAK_BEDROOM_UNENTERED])[0], [UNKNOWN, WEAK_BEDROOM_UNENTERED]) == (
        {2: 1.0, 4: 3.0}, ())
    assert SearchNodeOracle.score({"nodes": [{"id": 2, "p": 40}]}, [UNKNOWN, WEAK_BEDROOM_UNENTERED]).floored == ()
    # A fraction of a percent on a room never entered is a zero in all but name.
    assert SearchNodeOracle.score({"nodes": [{"id": 2, "p": 0.4}, {"id": 0, "p": 30}]}, [UNKNOWN, KITCHEN]).probs[2] == pytest.approx(0.01)
    # It holds for a reply kept and re-read, and the flat ablation is untouched by it.
    client = Scripted({"nodes": [{"id": 2, "p": 0}, {"id": 0, "p": 0}]})
    oracle = SearchNodeOracle(client)
    first, again = oracle.probabilities("toilet", [UNKNOWN, KITCHEN]), oracle.probabilities("toilet", [UNKNOWN, KITCHEN])
    assert first.probs == again.probs == {2: pytest.approx(0.01), 0: 0.0} and again.reused and again.floored == (2,)


# -- the optional floor: off by default, the 3B era's arithmetic on request ------------------------------------------------


def test_which_nodes_are_unexplored():
    assert UNKNOWN.unexplored, "never entered, not identified"
    assert GAP.unexplored, "an opening nothing was glimpsed through"
    assert not DOORWAY_WITH_TOILET.unexplored, "a glimpse is a preliminary classification: the model's to value"
    assert not UNKNOWN_ENTERED.unexplored, "the robot stood in it; its search time speaks"
    assert not WEAK_BEDROOM_UNENTERED.unexplored, "a type was inferred from outside: the model's to value"
    assert not KITCHEN.unexplored


def test_the_floor_is_off_by_default_so_the_models_numbers_stand():
    assert UNEXPLORED_FLOOR == 0.0
    oracle = SearchNodeOracle(Scripted({}))
    assert oracle.unexplored_floor == 0.0 and not hasattr(oracle, "unexplored_elsewhere")
    nodes = [KITCHEN, UNKNOWN, GAP, DOORWAY_WITH_TOILET, WEAK_BEDROOM_UNENTERED]
    reply = {"nodes": [{"id": 0, "p": 0}, {"id": 2, "why": "small unknown room", "p": 1},
                       {"id": 200002, "why": "nothing glimpsed", "p": 0},
                       {"id": 200003, "p": 1}, {"id": 4, "why": "bedroom, no toilet", "p": 2}]}
    result = SearchNodeOracle.score(reply, nodes)
    assert result.floored == ()
    assert result.probs[2] == pytest.approx(0.01) and result.probs[200002] == 0.0, "written off is written off: the model's call"


def test_the_floor_raises_what_the_model_wrote_off_and_leaves_the_rest_alone_when_switched_on():
    nodes = [KITCHEN, UNKNOWN, GAP, DOORWAY_WITH_TOILET, WEAK_BEDROOM_UNENTERED]
    reply = {"nodes": [{"id": 0, "p": 0}, {"id": 2, "why": "small unknown room, no toilet fits", "p": 1},
                       {"id": 200002, "why": "no toilet glimpsed through gap", "p": 0},
                       {"id": 200003, "p": 1}, {"id": 4, "why": "bedroom, no toilet", "p": 2}]}
    scores, floored = floor_unexplored(parse_reply(reply, nodes)[0], nodes, 0.25)
    assert floored == (2, 200002)
    assert scores[2] == scores[200002] == pytest.approx(25.0)
    assert scores[0] == 0 and scores[200003] == 1 and scores[4] == 2
    result = SearchNodeOracle.score(reply, nodes, 0.25)
    assert result.floored == (2, 200002) and result.probs[2] == pytest.approx(0.25) and result.probs[200002] == pytest.approx(0.25)
    assert result.reasons[2] == "small unknown room, no toilet fits", "the model's words are kept beside the floored number"
    above = SearchNodeOracle.score({"nodes": [{"id": 2, "p": 40}, {"id": 200002, "p": 30}]}, [UNKNOWN, GAP], 0.25)
    assert above.floored == () and above.probs[2] == pytest.approx(0.40), "a serious valuation stands"


def test_the_oracle_takes_no_hand_written_prior_about_where_the_target_lives():
    """No table briefs the model (``room_priors.py`` was removed on 2026-11-15): the prompt carries no WHERE IT
    LIVES line, a node carries no ``home`` flag, and no floor raises a room for the objects standing in it."""
    text = format_prompt("toilet", [KITCHEN, LIVING, UNKNOWN], SearchContext(HOUSE))
    assert text.startswith("TARGET: toilet\n\nHOUSE: " + HOUSE)
    assert "WHERE IT LIVES" not in text and "WHERE IT LIVES" not in SYSTEM_PROMPT
    assert not hasattr(SearchContext(), "target_home") and not hasattr(KITCHEN, "home")
    with pytest.raises(TypeError):
        SearchContext(HOUSE, target_home="bathroom (most likely first)")
    with pytest.raises(TypeError):
        SearchNodeOracle(Scripted({}), home_floor=0.6)
    with pytest.raises(TypeError):
        SearchNode(1, ROOM, "bathroom", home=True)
    bathroom = SearchNode(1, ROOM, "bathroom", area_m2=4.0, frontier_clusters=0, objects=("bathtub",),
                          searched_s=20.0, last_inside_ago_s=60.0)
    nodes = [KITCHEN, bathroom, UNKNOWN]
    reply = {"nodes": [{"id": 0, "p": 0}, {"id": 1, "why": "bathroom, seen", "p": 0}, {"id": 2, "p": 30}]}
    result = SearchNodeOracle.score(reply, nodes, 0.25)
    assert result.probs[1] == 0.0 and 1 not in result.floored, "a bathtub in the room raises nothing: the model's number stands"
    assert result.probs[0] == 0 and result.probs[2] == pytest.approx(0.30)


def test_the_oracle_applies_a_switched_on_floor_to_fresh_and_reused_replies():
    client = Scripted({"nodes": [{"id": 2, "why": "too small", "p": 1}, {"id": 200003, "p": 60}]})
    oracle = SearchNodeOracle(client, unexplored_floor=0.3)
    first = oracle.probabilities("toilet", [UNKNOWN, DOORWAY_WITH_TOILET])
    assert first.probs[2] == pytest.approx(0.3) and first.floored == (2,) and first.probs[200003] == pytest.approx(0.6)
    again = oracle.probabilities("toilet", [UNKNOWN, DOORWAY_WITH_TOILET])
    assert again.reused and again.probs[2] == pytest.approx(0.3) and client.calls == 1
    with pytest.raises(ValueError):
        SearchNodeOracle(client, unexplored_floor=1.0)
