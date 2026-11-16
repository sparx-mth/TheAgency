"""The node oracle: the prompt, the parse, the contract the planner relies on."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.topology import search_node_oracle
from sparx_agency.core.mapping.topology.search_node_oracle import (
    OMITTED_PERCENT, OPENING, OPENINGS_SUPPLEMENT, P_CEILING, PASS_FIRST, PASS_SECOND, ROOM, SOURCE_FLAT,
    SOURCE_LLM, SYSTEM_PROMPT, UNEXPLORED_FLOOR, USABLE_SOURCES, USER_PROMPT_TEMPLATE, SearchContext, SearchNode,
    SearchNodeOracle, area_words, budget_line, coarse_duration, coarse_metres, coarse_seconds, counted,
    floor_unexplored, format_coverage, format_exits, format_node, format_prompt, format_size, kinds, normalise,
    parse_pass, parse_reply, system_prompt)
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
def test_the_module_exports_the_default_versions_texts_and_asks_about_the_part_of_a_node_not_checked_yet():
    """What the one number is: the probability that the target is in the part of the node not checked yet -- the
    floor of the room that is left, and the unseen space behind its exits. A node is a room together with that
    space, and the detector has looked at the covered floor without finding the target: facts, said as facts."""
    default = PROMPTS[DEFAULT_PROMPT]
    assert (SYSTEM_PROMPT, OPENINGS_SUPPLEMENT, USER_PROMPT_TEMPLATE) == (default.system, default.openings_supplement,
                                                                         default.user_template)
    flat = " ".join(SYSTEM_PROMPT.split())
    for phrase in ("Each line under NODES is one place it can search next: a room, together with the unseen space its "
                   "exits open onto.",
                   "A detector looks for the target in every camera frame; on the floor a line lists as covered it has "
                   "looked and not found it.",
                   "estimate the probability, in percent, that the target is in the part of it not checked yet: the floor "
                   "of the room that is left, and the unseen space behind its exits.",
                   '"p": the probability in percent, from 0 (certain the target is not there) to 100 (certain it is).'):
        assert phrase in flat, phrase


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
    # Nor does how long ago the robot left a room: it is read for the status word alone.
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
        "\n\nFor each of the 1 nodes: how likely is it that the target (bed) is in the part of it not checked yet? "
        "Score them independently. JSON only.")


# -- the room line ---------------------------------------------------------------------------------------------------
def test_the_prompt_lists_every_node_in_the_format_the_legend_explains():
    text = format_prompt("bed", NODES, SearchContext(HOUSE))
    assert "TARGET: bed" in text and "NODES (5)" in text and "HOUSE: " + HOUSE in text
    assert format_node(KITCHEN) == ("id=0  ROOM  type=kitchen  size=12m2  frontier=0  exits=0  "
                                    "status=scanned(full rotation, 2min ago)  searched=40s  seen: sink x2, fridge")
    assert format_node(LIVING).endswith("status=entered  searched=20s  seen: sofa, television")
    assert format_node(UNKNOWN) == ("id=2  ROOM  type=unknown  size=18m2  frontier=3  exits=0  status=never_entered  "
                                    "searched=0s  seen: nothing yet")
    assert all(line.startswith("id=") for line in text.split("NODES (5):\n")[1].split("\n\n")[0].splitlines())
    weak = SearchNode(4, ROOM, "bedroom", tentative=True, area_m2=14.0)
    assert "type=bedroom?" in format_node(weak) and "type=unknown  " in format_node(replace_label(weak, "unknown"))


def replace_label(node, label):
    return SearchNode(node.id, node.kind, label, tentative=node.tentative, area_m2=node.area_m2)


def test_effort_is_written_in_actions_when_the_context_gives_the_action_time():
    """One unit on the whole prompt: the HOUSE line's budget is in actions, so the time spent in a room and the age
    of its scan are too -- the model can set one beside the other. A rig that does not count actions keeps time."""
    assert format_node(KITCHEN, 1.0) == ("id=0  ROOM  type=kitchen  size=12m2  frontier=0  exits=0  "
                                         "status=scanned(full rotation, 120 actions ago)  searched=40 actions  "
                                         "seen: sink x2, fridge")
    assert "searched=10 actions" in format_node(LIVING, 2.0), "20 s at 2 s an action"
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
    assert "searched=40 actions" in text and "120 actions ago" in text and "2min" not in text
    assert "searched=40s" in format_prompt("bed", [KITCHEN], SearchContext(HOUSE))
    assert SearchNode(5, ROOM, "bedroom", scanned="fragment").status_words(1.0) == "scanned(fragment)", "no step known: no ago"
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="action_time_s"):
            SearchContext(HOUSE, action_time_s=bad)


def test_the_objects_of_a_room_are_shown_with_their_counts_the_most_numerous_first():
    """The graph stores one entry per confirmed object; four chairs and a table say more than "chair, table"."""
    assert counted(("chair", "table", "Chair", " chair ", "chair")) == "chair x4, table"
    assert counted(("sofa", "bed")) == "bed, sofa" and counted(()) == "" and counted(("", "  ")) == ""
    assert counted(("b", "a", "a", "c", "c"), limit=2) == "a x2, c x2, +1 more", "the rarest kind is the one cut, and the cut shows"
    assert counted(("b", "a"), limit=2) == "a, b" and counted(("a",), limit=0) == "+1 more"
    dining = SearchNode(6, ROOM, "dining_room", area_m2=14.0, objects=("chair",) * 6 + ("table", "plant"))
    assert format_node(dining).endswith("seen: chair x6, plant, table")
    crowded = SearchNode(7, ROOM, "unknown", objects=tuple("k%02d" % i for i in range(12)) + ("k11",))
    assert format_node(crowded).endswith("seen: k11 x2, k00, k01, k02, k03, k04, k05, k06, +4 more"), (
        "eight kinds at most, and the model is told the list was cut: it reads the absence of an object from it")
    # What was glimpsed through an exit is a kind, not a census: one name however many exits showed it.
    assert kinds(("Sink ", "sink", "bed", "", " ")) == ("bed", "sink")
    seen_through = SearchNode(8, ROOM, "hallway", exits=2, glimpsed=("sink", "Sink", "toilet"))
    assert format_node(seen_through).endswith("glimpsed through its exits: sink, toilet")


def test_a_room_in_which_the_target_was_detected_and_not_reached_says_so_and_no_other_line_does():
    """The prompt says the detector has looked at the covered floor WITHOUT finding the target. That is true of a
    room unless one of the target's own kind is among its confirmed objects (the closing let it go: no route, a
    failed inspection) -- and then the line says so, in words that need no legend. The 3B's version deliberately
    has none: explaining the note made that model read its absence from a line as "the target is not there" and
    put every node of a query at 0 (probed on 2026-11-16; ``core/mapping/topology/README.md``)."""
    found = SearchNode(3, ROOM, "bathroom", area_m2=5.0, objects=("sink", "toilet"), last_inside_ago_s=9.0, target_seen=True)
    assert format_node(found).endswith("seen: sink, toilet  target: detected here, not reached")
    for unsaid in (None, False):
        line = format_node(SearchNode(3, ROOM, "bathroom", area_m2=5.0, objects=("sink", "toilet"), target_seen=unsaid))
        assert "target:" not in line and line.endswith("seen: sink, toilet")
    assert "target:" not in format_node(SearchNode(200001, OPENING, "doorway", target_seen=True)), "a room's fact"
    assert "detected here" not in SYSTEM_PROMPT and "not reached" not in SYSTEM_PROMPT
    assert format_prompt("toilet", [found]) != format_prompt("toilet", [SearchNode(
        3, ROOM, "bathroom", area_m2=5.0, objects=("sink", "toilet"), last_inside_ago_s=9.0)])


def test_the_rooms_a_door_joins_it_to_are_on_its_line():
    """The scene graph's edges: a room's line names the types of the rooms a door or a doorway joins it to."""
    hall = SearchNode(7, ROOM, "hallway", area_m2=6.0, connected=("bedroom", "kitchen", "bedroom", "unknown"))
    assert format_node(hall) == ("id=7  ROOM  type=hallway  size=6m2  frontier=0  exits=0  status=never_entered  "
                                 "searched=0s  connected to: bedroom x2, kitchen, unknown  seen: nothing yet")
    assert "connected to" not in format_node(UNKNOWN), "no edge: unsaid"
    assert format_prompt("bed", [hall]) != format_prompt("bed", [SearchNode(7, ROOM, "hallway", area_m2=6.0)])


def test_a_rooms_exits_are_attributes_on_its_line():
    """Room-level GAPs (2026-10-21): a doorway or gap to unseen space is the room's attribute, never a node with a
    number of its own. What an exit promises is the model's to judge: the prompt says what the field is, no more."""
    hallway = SearchNode(7, ROOM, "hallway", area_m2=6.0, frontier_clusters=1, last_inside_ago_s=4.0,
                         exits=2, doorways=1, glimpsed=("toilet", "Toilet", "sink"))
    assert format_node(hallway) == ("id=7  ROOM  type=hallway  size=6m2  frontier=1  exits=2 (1 doorway)  status=entered  "
                                    "searched=0s  seen: nothing yet  glimpsed through its exits: sink, toilet")
    assert format_exits(0) == "0" and format_exits(3) == "3" and format_exits(3, 2) == "3 (2 doorways)"
    assert format_exits(1, 5) == "1 (1 doorway)", "no more doorways than exits"
    # Since 2026-10-22 the unseen floor behind the exits rides in the brackets: where the unexplored space is.
    assert format_exits(2, 1, 30.4) == "2 (1 doorway, ~30m2 unseen behind them)"
    assert format_exits(2, 0, 7.6) == "2 (~8m2 unseen behind them)" and format_exits(0, 0, 12.0) == "0"
    assert format_exits(2, 0, 0.2) == "2", "under half a square metre is nothing to say"
    wing = SearchNode(8, ROOM, "hallway", exits=2, unseen_m2=30.0)
    assert "exits=2 (~30m2 unseen behind them)" in format_node(wing)
    assert "glimpsed through its exits" not in format_node(SearchNode(7, ROOM, "hallway", exits=1)), "nothing glimpsed: unsaid"
    flat = " ".join(SYSTEM_PROMPT.split())
    assert ("exits= doorways and gaps on the room's edge that lead to space not seen yet; in brackets, how many are at a "
            "door frame and how much unseen floor lies behind them.") in flat
    assert "RAISE" not in SYSTEM_PROMPT and "fits the target" not in SYSTEM_PROMPT, "no rule on what an exit is worth"
    assert "  OPENING  " not in SYSTEM_PROMPT, "no opening node line in the room-level prompt"
    assert system_prompt([hallway, KITCHEN], SearchContext(HOUSE)) == SYSTEM_PROMPT, "no exit node offered: no supplement"
    # The attributes change the prompt the way any fact does, and an unchanged map keeps it byte-identical.
    assert format_prompt("toilet", [hallway]) != format_prompt("toilet", [SearchNode(7, ROOM, "hallway", frontier_clusters=1,
                                                                                      last_inside_ago_s=4.0)])
    assert format_prompt("toilet", [hallway]) == format_prompt("toilet", [hallway])


def _covered(covered_m2, fraction, left_m2, bounded):
    """The ``covered=`` words for a room with these numbers."""
    return format_coverage(SearchNode(0, ROOM, "unknown", bounded=bounded, covered_fraction=fraction,
                                      covered_m2=covered_m2, left_m2=left_m2))


def test_a_rooms_size_says_what_the_map_knows_and_does_not_call_it_final():
    """Room extents: the size reads ``all sides mapped`` when every side is located -- no part of the room lies
    outside the map -- and ``at least`` otherwise. Not "final": of 252 rooms the first wording declared final in 40
    HM3D recordings, 73 were over a quarter bigger by the end of the episode, 229 of the 233 growths being the
    partition joining or re-cutting rooms whose floor was mapped all along. The words claim what the reading knows."""
    bounded = SearchNode(3, ROOM, "bedroom", area_m2=11.2, footprint_m2=12.6, length_m=4.1, width_m=3.05,
                         walls_seen=4, bounded=True, covered_fraction=0.87, covered_m2=10.9, left_m2=1.7, last_inside_ago_s=5.0)
    assert format_node(bounded) == ("id=3  ROOM  type=bedroom  size=13m2 (4x3m, all sides mapped)  covered=11m2 (80% of the "
                                    "room, 2m2 left)  frontier=0  exits=0  status=entered  searched=0s  seen: nothing yet")
    partial = SearchNode(4, ROOM, "unknown", area_m2=6.0, footprint_m2=6.0, length_m=4.9, width_m=2.4,
                         length_final=False, width_final=True, walls_seen=3, bounded=False,
                         covered_fraction=0.43, covered_m2=2.6, left_m2=3.4, frontier_clusters=1)
    assert format_size(partial) == "at least 6m2 (5+ x 2.5m, 1 of 4 sides not mapped yet)"
    assert format_coverage(partial) == "3m2 (at most 50% of the room, at least 3m2 left)", "a bound rounds UP, a share DOWN"
    assert ("  size=at least 6m2 (5+ x 2.5m, 1 of 4 sides not mapped yet)  covered=3m2 (at most 50% of the room, at least "
            "3m2 left)  frontier=1  ") in format_node(partial)
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
    assert "covered=" not in format_node(UNKNOWN)
    assert format_size(SearchNode(6, ROOM, "unknown", area_m2=0.4)) == "under 1m2"
    assert format_size(SearchNode(7, ROOM, "unknown")) == "unknown size"
    # With the extents but no observation record (no sight ledger) the size is said and the coverage is not.
    unobserved = SearchNode(8, ROOM, "unknown", footprint_m2=12.0, length_m=4.0, width_m=3.0, walls_seen=4, bounded=True)
    assert "size=12m2 (4x3m, all sides mapped)  frontier=0" in format_node(unobserved)
    # Coverage is shown in tens, so a frame that saw three more cells does not change the prompt.
    a = format_prompt("toilet", [SearchNode(3, ROOM, "bedroom", bounded=True, covered_fraction=0.81, covered_m2=9.7, left_m2=2.3)])
    b = format_prompt("toilet", [SearchNode(3, ROOM, "bedroom", bounded=True, covered_fraction=0.89, covered_m2=9.7, left_m2=2.3)])
    assert a == b
    assert a != format_prompt("toilet", [SearchNode(3, ROOM, "bedroom", bounded=True, covered_fraction=0.90, covered_m2=9.7, left_m2=2.3)])
    # What the prompt says of it: no promise the reading cannot keep.
    flat = " ".join(SYSTEM_PROMPT.split())
    assert ('"all sides mapped": no part of the room lies outside the map, though the map\'s division into rooms may '
            "still be revised.") in flat
    assert '"at least": a side is not mapped yet and the room continues beyond it' in flat
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
    flat = " ".join(SYSTEM_PROMPT.split())
    assert ('covered= floor of the room the detector has already checked: its area and its share of the room. "left": '
            "floor it has not checked yet.") in flat
    assert "searched= how long the robot has spent inside the room so far." in flat
    assert "coverage overrides" not in flat and "observed=" not in SYSTEM_PROMPT, "what the two facts are worth is the model's"


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
                 "not counted in its room's exits=", "estimate the probability that the target is in the space behind it",
                 '"glimpsed through it:" objects seen beyond it'):
        assert part in " ".join(OPENINGS_SUPPLEMENT.split()) and part not in SYSTEM_PROMPT, part
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
    assert "status=scanned(seen from another room's scan, 10s ago)" in format_node(scanned)
    assert "status=scanned(seen from another room's scan, 10 actions ago)" in format_node(scanned, 1.0)


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


# -- the optional floor: off by default, the 3B era's arithmetic on request ------------------------------------------------
DOORWAY_WITH_TOILET = SearchNode(200003, OPENING, "doorway", objects=("toilet",), via="room 2 (type=unknown)")
UNKNOWN_ENTERED = SearchNode(3, ROOM, "unknown", area_m2=9.0, frontier_clusters=2, searched_s=12.0, last_inside_ago_s=20.0)
WEAK_BEDROOM_UNENTERED = SearchNode(4, ROOM, "bedroom", tentative=True, area_m2=14.0, frontier_clusters=1)


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
    bathroom = SearchNode(1, ROOM, "bathroom", area_m2=4.0, frontier_clusters=0, objects=("bathtub",))
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
