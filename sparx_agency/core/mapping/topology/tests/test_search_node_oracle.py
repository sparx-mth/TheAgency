"""The node oracle: the prompt, the parse, the contract the planner relies on."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.topology.search_node_oracle import (
    HOME_FLOOR, OMITTED_PERCENT, OPENING, P_CEILING, PASS_FIRST, PASS_SECOND, ROOM, STAIRS, STAIRS_SUPPLEMENT,
    SYSTEM_PROMPT, UNEXPLORED_ELSEWHERE, UNEXPLORED_FLOOR, SearchContext, SearchNode, SearchNodeOracle, budget_line,
    coarse_seconds, floor_unexplored, format_node, format_prompt, normalise, parse_home_here, parse_pass, parse_reply,
    system_prompt)


class Scripted:
    def __init__(self, reply=None, raise_=False):
        self.reply, self.raise_, self.calls, self.kwargs, self.systems = reply, raise_, 0, [], []

    def chat_json(self, system, user, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        self.systems.append(system)
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
UP = SearchNode(100000, STAIRS, "stairs up", direction=1)
DOWN = SearchNode(100001, STAIRS, "stairs down", direction=-1, destination_visited=True,
                  destination="storey F0: rooms found: kitchen; searched 2min; 1 room with frontier left",
                  arrived_by=True, arrived_ago_s=14.0)
NODES = [KITCHEN, LIVING, UNKNOWN, UP, DOWN]
HOUSE = ("rooms found: kitchen (scanned), living_room (entered); 1 room unidentified; 0 openings not yet looked into; "
         "actions used about 150 of 500")


def test_the_prompt_lists_every_node_in_the_format_the_system_prompt_explains():
    text = format_prompt("bed", NODES, SearchContext(HOUSE, "a storey with a kitchen and a living room", ("storey F0: ...",)))
    assert "TARGET: bed" in text and "NODES (5)" in text
    assert "HOUSE: " + HOUSE in text
    assert "THIS STOREY: a storey with a kitchen and a living room" in text and "OTHER STOREYS: storey F0: ..." in text
    assert format_node(KITCHEN) == ("id=0  ROOM  type=kitchen  size=12m2  frontier=0  status=scanned(full rotation, 2min ago)  "
                                    "searched=40s  seen: fridge, sink")
    assert format_node(LIVING).endswith("status=entered  searched=20s  here=yes  seen: sofa, television")
    assert format_node(UNKNOWN) == ("id=2  ROOM  type=unknown  size=18m2  frontier=3  status=never_entered  searched=0s  "
                                    "seen: nothing yet")
    assert format_node(UP) == "id=100000  STAIRS up  to a storey NOT visited yet"
    assert format_node(DOWN) == ("id=100001  STAIRS down  to a storey ALREADY visited (storey F0: rooms found: kitchen; "
                                 "searched 2min; 1 room with frontier left)  the robot came up these stairs 10s ago "
                                 "(arrived_by=yes)")
    for rule in ("type=unknown", "never_entered", "scanned(<how>, <ago>)", "NEVER use distance", "do NOT make them sum to 100",
                 "independent", 'STEP 1 "home"', 'STEP 2 "house"', 'STEP 3 "stage"', "never copy its numbers",
                 "en-suite", "ONE kitchen, ONE living room", "MISSED the target", '"pass":"first|second"',
                 "merged two rooms", "never write a room off for its size"):
        assert rule in SYSTEM_PROMPT, rule
    assert "no television fits" not in SYSTEM_PROMPT and "no toilet fits" not in SYSTEM_PROMPT, (
        "the example taught a 3B model to write 'no <target> fits' on every small unknown room (Hanson 2026-10-04)")


def test_the_single_storey_prompt_has_no_storey_lines_and_no_stairs_rules():
    """The standard benchmarks are single-storey: the model is not told about storeys or staircases at all."""
    rooms = [KITCHEN, LIVING, UNKNOWN]
    text = format_prompt("bed", rooms, SearchContext(HOUSE))
    assert "THIS STOREY" not in text and "OTHER STOREYS" not in text and "HOUSE: " + HOUSE in text
    assert system_prompt(rooms, SearchContext(HOUSE)) == SYSTEM_PROMPT
    assert "STAIRS" not in SYSTEM_PROMPT and "storey" not in SYSTEM_PROMPT.lower().replace("storeys", "")
    # A staircase on offer, or another storey known, brings the supplement and the storey lines.
    assert system_prompt(NODES, SearchContext(HOUSE)) == SYSTEM_PROMPT + STAIRS_SUPPLEMENT
    assert system_prompt(rooms, SearchContext(HOUSE, "upper", ("storey F0: ...",))) == SYSTEM_PROMPT + STAIRS_SUPPLEMENT
    with_stairs = format_prompt("bed", NODES, SearchContext(HOUSE))
    assert "THIS STOREY: the only storey mapped so far" in with_stairs and "OTHER STOREYS: none known" in with_stairs
    for rule in ("arrived_by=yes", '"home_here"', "ground floor", "planner adds travel and climbing"):
        assert rule in STAIRS_SUPPLEMENT, rule
    assert format_prompt("bed", rooms).startswith("TARGET: bed\n\nHOUSE: nothing known yet\n")


def test_an_opening_is_a_node_named_by_the_room_it_opens_from_and_what_was_glimpsed_through_it():
    """A doorway off the hallway with a toilet seen through it is the bathroom behind the door, not
    the hallway: the prompt says which mapped room it opens from and what the glimpse showed."""
    door = SearchNode(200001, OPENING, "doorway", objects=("toilet", "Toilet", "sink"), via="room 11 (type=unknown)")
    gap = SearchNode(200002, OPENING, "gap")
    assert format_node(door) == ("id=200001  OPENING  doorway off room 11 (type=unknown)  to space NOT seen yet  "
                                 "glimpsed through it: sink, toilet")
    assert format_node(gap) == "id=200002  OPENING  gap off the mapped floor  to space NOT seen yet  glimpsed through it: nothing yet"
    text = format_prompt("couch", NODES + [door, gap], SearchContext(HOUSE))
    assert "NODES (7)" in text and "id=200001  OPENING" in text
    for rule in ("OPENING: a doorway or gap", "quick look from the threshold", "room likely BEHIND it",
                 "glimpsed objects name the home type"):
        assert rule in SYSTEM_PROMPT, rule
    assert "id=200001  OPENING  doorway off room 7" in SYSTEM_PROMPT, "the worked example shows an opening valued by the home"


def test_the_room_status_words():
    assert SearchNode(5, ROOM, "unknown").status == "never_entered"
    assert SearchNode(5, ROOM, "unknown", last_inside_ago_s=0.0).status == "entered"
    assert SearchNode(5, ROOM, "bedroom", scanned="full rotation", scanned_ago_s=44.0).status == "scanned(full rotation, 40s ago)"
    assert SearchNode(5, ROOM, "bedroom", scanned="fragment").status == "scanned(fragment)", "no step known: no ago"
    scanned = SearchNode(5, ROOM, "unknown", scanned="seen from another room's scan", scanned_ago_s=10.0)
    assert not scanned.unexplored, "a scanned room is not an unknown place, whatever its type"
    assert "status=scanned(seen from another room's scan, 10s ago)" in format_node(scanned)


def test_coarse_seconds_and_the_budget_line_keep_an_unchanged_map_an_unchanged_prompt():
    assert coarse_seconds(None) == "never"
    assert coarse_seconds(0) == "0s" and coarse_seconds(4) == "0s" and coarse_seconds(6) == "10s"
    assert coarse_seconds(43) == "40s" and coarse_seconds(59) == "60s"
    assert coarse_seconds(60) == "1min" and coarse_seconds(150) == "2min" and coarse_seconds(3600) == "60min"
    drifted = SearchNode(0, ROOM, "kitchen", area_m2=12.4, searched_s=44.0, last_inside_ago_s=33.0, objects=("fridge", "sink"),
                         scanned="full rotation", scanned_ago_s=128.0)
    assert format_prompt("bed", [KITCHEN]) == format_prompt("bed", [drifted]), "one more second changes nothing"
    assert budget_line(0, 500) == "actions used about 0 of 500"
    assert budget_line(149, 500) == "actions used about 125 of 500" and budget_line(150, 500) == "actions used about 150 of 500"
    assert budget_line(174, 500) == budget_line(150, 500), "a 25-action bucket, rounded down"
    assert budget_line(600, 500) == "actions used about 500 of 500"


def test_a_good_reply_preserves_independent_node_probabilities_and_keeps_the_reading():
    reply = {"home": "bedroom", "house": "kitchen and living room found; one unidentified room", "stage": "early, first pass",
             "pass": "first",
             "nodes": [{"id": 0, "why": "kitchen, scanned, no beds", "p": 0},
                       {"id": 1, "why": "beds are not in living rooms", "p": 3},
                       {"id": 2, "why": "large unexplored room", "p": 27},
                       {"id": 100000, "why": "bedrooms upstairs", "p": 62},
                       {"id": 100001, "why": "just came from there", "p": 0},
                       {"id": 77, "why": "invented", "p": 40}],
             "elsewhere": 8}
    result = SearchNodeOracle.score(reply, NODES)
    assert result.source == "llm" and result.omitted == ()
    assert result.reading == {"home": "bedroom", "house": "kitchen and living room found; one unidentified room",
                              "stage": "early, first pass", "pass": "first"}
    assert result.pass_verdict == PASS_FIRST and result.home_here is None
    assert result.probs == pytest.approx({0: 0.0, 1: 0.03, 2: 0.27, 100000: 0.62, 100001: 0.0})
    assert result.elsewhere == pytest.approx(0.97 * 0.73 * 0.38)
    assert result.p_present == pytest.approx(1.0 - result.elsewhere)
    assert result.probability_model == "independent_search_success"
    assert 77 not in result.probs and result.reasons[100000] == "bedrooms upstairs"
    assert result.spread == pytest.approx(0.62)
    assert result.floored == () and result.capped == (), "no arithmetic touched the model's numbers"


def test_percentages_are_not_rescaled_and_omitted_nodes_get_a_small_probability():
    reply = {"nodes": [{"id": 0, "p": 50}, {"id": 2, "p": 150}], "elsewhere": 50}     # sums to 250, three nodes missing
    scores, elsewhere, reasons, omitted = parse_reply(reply, NODES)
    assert omitted == (1, 100000, 100001) and all(scores[nid] == OMITTED_PERCENT for nid in omitted)
    assert scores[2] == 100.0, "clamped into 0-100 before anything else"
    probs, rest = normalise(scores, elsewhere)
    assert probs[0] == 0.5 and probs[2] == P_CEILING
    assert rest == pytest.approx(0.5 * (1 - P_CEILING) * 0.99 ** 3)
    assert probs[2] > probs[0] > probs[1] > 0.0
    result = SearchNodeOracle.score(reply, NODES)
    assert result.omitted == (1, 100000, 100001) and result.reasons[1] == "(not scored by the model)"


def test_field_names_a_model_drifts_to_are_read():
    bare = SearchNodeOracle.score({"nodes": [{"id": 0, "p": 50}], "elsewhere": 50, "home": 7, "storey": ""}, [KITCHEN])
    assert bare.reading == {}, "a missing or non-string reading is simply absent"
    scores, _, _, _ = parse_reply({"nodes": [{"id": 0, "probability": 0.25}, {"id": 1, "percent": 30},
                                             {"id": 2, "score": 45}, {"id": 100000, "p": "10"}]}, NODES)
    assert scores[0] == 25.0 and scores[1] == 30.0 and scores[2] == 45.0 and scores[100000] == 10.0


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


def test_the_oracle_routes_to_the_reasoning_model_and_reuses_an_unchanged_prompt():
    client = Scripted({"nodes": [{"id": 0, "p": 20}, {"id": 1, "p": 30}], "elsewhere": 50})
    oracle = SearchNodeOracle(client)
    first = oracle.probabilities("bed", [KITCHEN, LIVING])
    assert first.source == "llm" and not first.reused and client.calls == 1 and oracle.queries == 1
    assert client.kwargs[0] == {"reasoning": True}, "the one judgement per loop point goes to the reasoning model"
    assert client.systems[0] == SYSTEM_PROMPT, "rooms only: the single-storey prompt, no stairs supplement"
    again = oracle.probabilities("bed", [KITCHEN, LIVING])
    assert again.reused and client.calls == 1 and oracle.reuses == 1
    moved = oracle.probabilities("bed", [KITCHEN, SearchNode(1, ROOM, "living_room", area_m2=25.0, frontier_clusters=0)])
    assert not moved.reused and client.calls == 2, "a changed frontier count is a changed prompt"
    budget = oracle.probabilities("bed", [KITCHEN, LIVING], SearchContext(budget_line(300, 500)))
    assert not budget.reused and client.calls == 3, "a changed budget bucket is a changed prompt"
    with pytest.raises(ValueError):
        oracle.probabilities("bed", [])


def test_a_staircase_on_offer_brings_the_stairs_supplement():
    client = Scripted({"nodes": [{"id": 2, "p": 20}, {"id": 100000, "p": 60}], "home_here": "elsewhere"})
    result = SearchNodeOracle(client).probabilities("bed", [UNKNOWN, UP])
    assert client.systems[0] == SYSTEM_PROMPT + STAIRS_SUPPLEMENT
    assert result.home_here == "elsewhere" and result.reading["home_here"] == "elsewhere"
    assert result.probs[2] == pytest.approx(0.20) and result.capped == (), "the elsewhere value is off by default"


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


def test_search_node_refuses_an_unknown_kind():
    with pytest.raises(ValueError):
        SearchNode(0, "door", "door")


def test_two_likely_rooms_are_not_diluted_into_categorical_shares():
    result = SearchNodeOracle.score({"nodes": [{"id": 0, "p": 80}, {"id": 1, "p": 80}]}, [KITCHEN, LIVING])
    assert result.probs == {0: 0.8, 1: 0.8}
    assert result.elsewhere == pytest.approx(0.04)
    assert result.p_present == pytest.approx(0.96)
    assert "NOT shares of one distribution" in SYSTEM_PROMPT.replace("\n", " ")


# -- the pass verdict: the model says when the home is covered (2026-10-07) --------------
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
    assert '"pass":"first"' in SYSTEM_PROMPT and 'the pass would be "second"' in SYSTEM_PROMPT, (
        "the worked example shows a first pass and says what turns it into a second")


# -- the optional floors: off by default, the 3B era's arithmetic on request ----------------
GAP = SearchNode(200002, OPENING, "gap", via="room 2 (type=unknown)")
DOORWAY_WITH_TOILET = SearchNode(200001, OPENING, "doorway", objects=("toilet",), via="room 2 (type=unknown)")
UNKNOWN_ENTERED = SearchNode(3, ROOM, "unknown", area_m2=9.0, frontier_clusters=2, searched_s=12.0, last_inside_ago_s=20.0)
WEAK_BEDROOM_UNENTERED = SearchNode(4, ROOM, "bedroom", tentative=True, area_m2=14.0, frontier_clusters=1)


def test_which_nodes_are_unexplored():
    assert UNKNOWN.unexplored, "never entered, not identified"
    assert GAP.unexplored, "an opening nothing was glimpsed through"
    assert not DOORWAY_WITH_TOILET.unexplored, "a glimpse is a preliminary classification: the model's to value"
    assert not UNKNOWN_ENTERED.unexplored, "the robot stood in it; its search time speaks"
    assert not WEAK_BEDROOM_UNENTERED.unexplored, "a type was inferred from outside: the model's to value"
    assert not KITCHEN.unexplored and not UP.unexplored and not DOWN.unexplored


def test_the_floors_are_off_by_default_so_the_models_numbers_stand():
    assert UNEXPLORED_FLOOR == 0.0 and UNEXPLORED_ELSEWHERE == 0.0 and HOME_FLOOR == 0.0
    oracle = SearchNodeOracle(Scripted({}))
    assert oracle.unexplored_floor == 0.0 and oracle.unexplored_elsewhere == 0.0 and oracle.home_floor == 0.0
    nodes = [KITCHEN, UNKNOWN, UP, GAP, DOORWAY_WITH_TOILET, WEAK_BEDROOM_UNENTERED]
    reply = {"home_here": "elsewhere",
             "nodes": [{"id": 0, "p": 0}, {"id": 2, "why": "small unknown room", "p": 1},
                       {"id": 100000, "p": 60}, {"id": 200002, "why": "nothing glimpsed", "p": 0},
                       {"id": 200001, "p": 1}, {"id": 4, "why": "bedroom, no toilet", "p": 2}]}
    result = SearchNodeOracle.score(reply, nodes)
    assert result.floored == () and result.capped == ()
    assert result.probs[2] == pytest.approx(0.01) and result.probs[200002] == 0.0, "written off is written off: the model's call"
    bathroom = SearchNode(1, ROOM, "bathroom", area_m2=4.0, frontier_clusters=0, objects=("bathtub",), home=True)
    home = SearchNodeOracle.score({"nodes": [{"id": 1, "p": 0}]}, [bathroom])
    assert home.probs[1] == 0.0 and home.floored == ()


def test_the_floor_raises_what_the_model_wrote_off_and_leaves_the_rest_alone_when_switched_on():
    nodes = [KITCHEN, UNKNOWN, UP, GAP, DOORWAY_WITH_TOILET, WEAK_BEDROOM_UNENTERED]
    reply = {"nodes": [{"id": 0, "p": 0}, {"id": 2, "why": "small unknown room, no toilet fits", "p": 1},
                       {"id": 100000, "p": 60}, {"id": 200002, "why": "no toilet glimpsed through gap", "p": 0},
                       {"id": 200001, "p": 1}, {"id": 4, "why": "bedroom, no toilet", "p": 2}]}
    scores, floored, capped = floor_unexplored(parse_reply(reply, nodes)[0], nodes, 0.25)
    assert floored == (2, 200002) and capped == ()
    assert scores[2] == scores[200002] == pytest.approx(25.0)
    assert scores[0] == 0 and scores[100000] == 60 and scores[200001] == 1 and scores[4] == 2
    result = SearchNodeOracle.score(reply, nodes, 0.25)
    assert result.floored == (2, 200002) and result.probs[2] == pytest.approx(0.25) and result.probs[200002] == pytest.approx(0.25)
    assert result.reasons[2] == "small unknown room, no toilet fits", "the model's words are kept beside the floored number"
    above = SearchNodeOracle.score({"nodes": [{"id": 2, "p": 40}, {"id": 200002, "p": 30}]}, [UNKNOWN, GAP], 0.25)
    assert above.floored == () and above.probs[2] == pytest.approx(0.40), "a serious valuation stands"


def test_a_room_holding_a_home_object_is_read_at_the_home_floor_when_switched_on():
    bathroom = SearchNode(1, ROOM, "bathroom", area_m2=4.0, frontier_clusters=0, objects=("bathtub",), home=True)
    nodes = [KITCHEN, bathroom, UNKNOWN]
    reply = {"home_here": "found", "nodes": [{"id": 0, "p": 0}, {"id": 1, "why": "bathroom, seen", "p": 0}, {"id": 2, "p": 30}]}
    result = SearchNodeOracle.score(reply, nodes, 0.25, 0.10, 0.60)
    assert result.probs[1] == pytest.approx(0.60) and 1 in result.floored
    assert result.probs[0] == 0 and result.probs[2] == pytest.approx(0.30), "the others keep the model's numbers"
    assert result.reasons[1] == "bathroom, seen", "the model's words stay beside the floored number"
    higher = SearchNodeOracle.score(dict(reply, nodes=[{"id": 1, "p": 85}]), [bathroom], 0.25, 0.10, 0.60)
    assert higher.probs[1] == pytest.approx(0.85) and higher.floored == (), "a valuation above the floor stands"
    # 'elsewhere' caps the unexplored places; the home room is still read at its floor.
    result = SearchNodeOracle.score(dict(reply, home_here="elsewhere"), nodes, 0.25, 0.10, 0.60)
    assert result.probs[1] == pytest.approx(0.60) and result.probs[2] == pytest.approx(0.10)
    assert 1 in result.floored and 1 not in result.capped
    with pytest.raises(ValueError):
        SearchNodeOracle(Scripted({}), home_floor=1.0)
    assert not KITCHEN.home and SearchNode(9, ROOM, "unknown").home is False


def test_when_the_model_says_the_home_type_is_elsewhere_unexplored_places_read_at_the_elsewhere_value_when_switched_on():
    nodes = [KITCHEN, UNKNOWN, UP, GAP, DOORWAY_WITH_TOILET, WEAK_BEDROOM_UNENTERED]
    reply = {"home": "living room", "storey": "upper floor (bedroom found); living rooms are downstairs",
             "home_here": "elsewhere",
             "nodes": [{"id": 0, "p": 0}, {"id": 2, "why": "small unknown room, never entered", "p": 25},
                       {"id": 100000, "p": 60}, {"id": 200002, "why": "unseen room, nothing known", "p": 25},
                       {"id": 200001, "why": "toilet glimpsed: a bathroom", "p": 1}, {"id": 4, "why": "bedroom", "p": 2}]}
    result = SearchNodeOracle.score(reply, nodes, 0.25, 0.10)
    assert result.home_here == "elsewhere" and result.reading["home_here"] == "elsewhere"
    assert result.capped == (2, 200002) and result.floored == ()
    assert result.probs[2] == result.probs[200002] == pytest.approx(0.10), "the unexplored upstairs places"
    assert result.probs[100000] == pytest.approx(0.60) and result.probs[200001] == pytest.approx(0.01) and result.probs[4] == pytest.approx(0.02)
    low = dict(reply, nodes=[{"id": 2, "p": 0}, {"id": 200002, "p": 1}, {"id": 100000, "p": 60}])
    result = SearchNodeOracle.score(low, [UNKNOWN, GAP, UP], 0.25, 0.10)
    assert result.floored == (2, 200002) and result.capped == () and result.probs[2] == pytest.approx(0.10)
    for verdict in ("found", "missing", None):
        same = dict(low, home_here=verdict) if verdict else {k: v for k, v in low.items() if k != "home_here"}
        result = SearchNodeOracle.score(same, [UNKNOWN, GAP, UP], 0.25, 0.10)
        assert result.home_here == verdict and result.probs[2] == pytest.approx(0.25) and result.capped == ()
    kept = SearchNodeOracle.score(reply, nodes, 0.25, 0.0)
    assert kept.probs[2] == pytest.approx(0.25) and kept.capped == () and kept.floored == ()


def test_the_home_here_verdict_is_read_from_the_words_a_model_drifts_to():
    for text, verdict in (("found", "found"), ("Elsewhere", "elsewhere"), ("missing", "missing"),
                          ("not found yet", "missing"), ("not here, downstairs", "elsewhere"),
                          ("on this storey", "found"), ("another floor", "elsewhere"), ("", None), (7, None),
                          ("none found", "missing"), ("no room of the home type found", "missing"),
                          ("nothing found yet", "missing"), ("no", "missing"), ("not found here", "missing"),
                          ("found elsewhere", "elsewhere"), ("yes", "found"), ("another level", "elsewhere")):
        assert parse_home_here({"home_here": text}) == verdict, text
    assert parse_home_here({}) is None and parse_home_here("nonsense") is None
    oracle = SearchNodeOracle(Scripted({"nodes": []}), unexplored_floor=0.25, unexplored_elsewhere=0.1)
    assert oracle.unexplored_elsewhere == 0.1
    with pytest.raises(ValueError):
        SearchNodeOracle(Scripted({}), unexplored_elsewhere=1.0)


def test_the_oracle_applies_a_switched_on_floor_to_fresh_and_reused_replies():
    client = Scripted({"nodes": [{"id": 2, "why": "too small", "p": 1}, {"id": 100000, "p": 60}]})
    oracle = SearchNodeOracle(client, unexplored_floor=0.3)
    first = oracle.probabilities("toilet", [UNKNOWN, UP])
    assert first.probs[2] == pytest.approx(0.3) and first.floored == (2,) and first.probs[100000] == pytest.approx(0.6)
    again = oracle.probabilities("toilet", [UNKNOWN, UP])
    assert again.reused and again.probs[2] == pytest.approx(0.3) and client.calls == 1
    with pytest.raises(ValueError):
        SearchNodeOracle(client, unexplored_floor=1.0)
