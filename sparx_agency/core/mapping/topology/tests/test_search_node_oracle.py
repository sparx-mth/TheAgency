"""The node oracle: the prompt, the parse, the contract the planner relies on."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.topology.search_node_oracle import (
    OMITTED_PERCENT, OPENING, P_CEILING, ROOM, STAIRS, SYSTEM_PROMPT, UNEXPLORED_ELSEWHERE, UNEXPLORED_FLOOR,
    SearchContext, SearchNode, SearchNodeOracle, coarse_seconds, floor_unexplored, format_node, format_prompt,
    normalise, parse_home_here, parse_reply)


class Scripted:
    def __init__(self, reply=None, raise_=False):
        self.reply, self.raise_, self.calls, self.kwargs = reply, raise_, 0, []

    def chat_json(self, system, user, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
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
                     objects=("fridge", "sink", "Sink"))
LIVING = SearchNode(1, ROOM, "living_room", area_m2=25.0, frontier_clusters=1, searched_s=20.0, last_inside_ago_s=0.0,
                    here=True, objects=("sofa", "television"))
UNKNOWN = SearchNode(2, ROOM, "unknown", area_m2=18.0, frontier_clusters=3)
UP = SearchNode(100000, STAIRS, "stairs up", direction=1)
DOWN = SearchNode(100001, STAIRS, "stairs down", direction=-1, destination_visited=True,
                  destination="storey F0: rooms found: kitchen; searched 2min; 1 room with frontier left",
                  arrived_by=True, arrived_ago_s=14.0)
NODES = [KITCHEN, LIVING, UNKNOWN, UP, DOWN]


def test_the_prompt_lists_every_node_in_the_format_the_system_prompt_explains():
    text = format_prompt("bed", NODES, SearchContext("a storey with a kitchen and a living room", ("storey F0: ...",)))
    assert "TARGET: bed" in text and "NODES (5)" in text
    assert "THIS STOREY: a storey with a kitchen and a living room" in text and "OTHER STOREYS: storey F0: ..." in text
    assert format_node(KITCHEN) == "id=0  ROOM  type=kitchen  size=12m2  frontier=0  searched=40s  ago=30s  seen: fridge, sink"
    assert format_node(LIVING).endswith("ago=0s  here=yes  seen: sofa, television")
    assert format_node(UNKNOWN) == "id=2  ROOM  type=unknown  size=18m2  frontier=3  searched=0s  entered=no  seen: nothing yet"
    assert format_node(UP) == "id=100000  STAIRS up  to a storey NOT visited yet"
    assert format_node(DOWN) == ("id=100001  STAIRS down  to a storey ALREADY visited (storey F0: rooms found: kitchen; "
                                 "searched 2min; 1 room with frontier left)  the robot came up these stairs 10s ago "
                                 "(arrived_by=yes)")
    for rule in ("type=unknown means NOT", "HOME MISSING HERE", "NEVER use distance", "planner adds travel and climbing",
                 "arrived_by=yes", "Do NOT make the scores sum to 100", "independently", 'STEP 1 -- "home"', 'STEP 2 -- "storey"',
                 "never copy its numbers", "entered=no", "UNEXPLORED places", "merged two rooms into one"):
        assert rule in SYSTEM_PROMPT
    assert "no television fits" not in SYSTEM_PROMPT, (
        "the example taught a 3B model to write 'no <target> fits' on every small unknown room (Hanson 2026-10-04)")


def test_an_opening_is_a_node_named_by_the_room_it_opens_from_and_what_was_glimpsed_through_it():
    """A doorway off the hallway with a toilet seen through it is the bathroom behind the door, not
    the hallway: the prompt says which mapped room it opens from and what the glimpse showed."""
    door = SearchNode(200001, OPENING, "doorway", objects=("toilet", "Toilet", "sink"), via="room 11 (type=unknown)")
    gap = SearchNode(200002, OPENING, "gap")
    assert format_node(door) == ("id=200001  OPENING  doorway off room 11 (type=unknown)  to space NOT seen yet  "
                                 "glimpsed through it: sink, toilet")
    assert format_node(gap) == "id=200002  OPENING  gap off the mapped floor  to space NOT seen yet  glimpsed through it: nothing yet"
    text = format_prompt("couch", NODES + [door, gap], SearchContext("upper floor", ()))
    assert "NODES (7)" in text and "id=200001  OPENING" in text
    for rule in ("OPENING nodes are doorways or gaps", "quick look in from the threshold", "room likely BEHIND it",
                 "an opening with nothing glimpsed is where it would be found"):
        assert rule in SYSTEM_PROMPT
    assert "id=200001  OPENING  doorway off room 11" in SYSTEM_PROMPT, "the worked example shows an opening valued by its glimpse"


def test_coarse_seconds_keeps_an_unchanged_map_an_unchanged_prompt():
    assert coarse_seconds(None) == "never"
    assert coarse_seconds(0) == "0s" and coarse_seconds(4) == "0s" and coarse_seconds(6) == "10s"
    assert coarse_seconds(43) == "40s" and coarse_seconds(59) == "60s"
    assert coarse_seconds(60) == "1min" and coarse_seconds(150) == "2min" and coarse_seconds(3600) == "60min"
    drifted = SearchNode(0, ROOM, "kitchen", area_m2=12.4, searched_s=44.0, last_inside_ago_s=33.0, objects=("fridge", "sink"))
    assert format_prompt("bed", [KITCHEN]) == format_prompt("bed", [drifted]), "one more second changes nothing"


def test_a_good_reply_preserves_independent_node_probabilities():
    reply = {"nodes": [{"id": 0, "why": "kitchen, fully seen, no beds", "p": 0},
                       {"id": 1, "why": "beds are not in living rooms", "p": 3},
                       {"id": 2, "why": "large unexplored room", "p": 27},
                       {"id": 100000, "why": "bedrooms upstairs", "p": 62},
                       {"id": 100001, "why": "just came from there", "p": 0},
                       {"id": 77, "why": "invented", "p": 40}],
             "elsewhere": 8}
    reply["home"] = "bedroom"
    reply["storey"] = "ground floor (kitchen, living room); no bedroom here; bedrooms are upstairs"
    result = SearchNodeOracle.score(reply, NODES)
    assert result.source == "llm" and result.omitted == ()
    assert result.reading == {"home": "bedroom",
                              "storey": "ground floor (kitchen, living room); no bedroom here; bedrooms are upstairs"}
    assert result.probs == pytest.approx({0: 0.0, 1: 0.03, 2: 0.27, 100000: 0.62, 100001: 0.0})
    assert result.elsewhere == pytest.approx(0.97 * 0.73 * 0.38)
    assert result.p_present == pytest.approx(1.0 - result.elsewhere)
    assert result.probability_model == "independent_search_success"
    assert 77 not in result.probs and result.reasons[100000] == "bedrooms upstairs"
    assert result.spread == pytest.approx(0.62)


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
    again = oracle.probabilities("bed", [KITCHEN, LIVING])
    assert again.reused and client.calls == 1 and oracle.reuses == 1
    moved = oracle.probabilities("bed", [KITCHEN, SearchNode(1, ROOM, "living_room", area_m2=25.0, frontier_clusters=0)])
    assert not moved.reused and client.calls == 2, "a changed frontier count is a changed prompt"
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


def test_search_node_refuses_an_unknown_kind():
    with pytest.raises(ValueError):
        SearchNode(0, "door", "door")


def test_two_likely_rooms_are_not_diluted_into_categorical_shares():
    result = SearchNodeOracle.score({"nodes": [{"id": 0, "p": 80}, {"id": 1, "p": 80}]}, [KITCHEN, LIVING])
    assert result.probs == {0: 0.8, 1: 0.8}
    assert result.elsewhere == pytest.approx(0.04)
    assert result.p_present == pytest.approx(0.96)
    assert "no fixed action or time horizon" in SYSTEM_PROMPT.replace("\n", " ")





# -- the uncertainty floor (2026-10-05) -----------------------------------------------
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


def test_the_floor_raises_what_the_model_wrote_off_and_leaves_the_rest_alone():
    nodes = [KITCHEN, UNKNOWN, UP, GAP, DOORWAY_WITH_TOILET, WEAK_BEDROOM_UNENTERED]
    reply = {"nodes": [{"id": 0, "p": 0}, {"id": 2, "why": "small unknown room, no toilet fits", "p": 1},
                       {"id": 100000, "p": 60}, {"id": 200002, "why": "no toilet glimpsed through gap", "p": 0},
                       {"id": 200001, "p": 1}, {"id": 4, "why": "bedroom, no toilet", "p": 2}]}
    scores, floored, capped = floor_unexplored(parse_reply(reply, nodes)[0], nodes, UNEXPLORED_FLOOR)
    assert floored == (2, 200002) and capped == ()
    assert scores[2] == scores[200002] == pytest.approx(25.0)
    assert scores[0] == 0 and scores[100000] == 60 and scores[200001] == 1 and scores[4] == 2
    result = SearchNodeOracle.score(reply, nodes, UNEXPLORED_FLOOR)
    assert result.floored == (2, 200002) and result.probs[2] == pytest.approx(0.25) and result.probs[200002] == pytest.approx(0.25)
    assert result.reasons[2] == "small unknown room, no toilet fits", "the model's words are kept beside the floored number"
    above = SearchNodeOracle.score({"nodes": [{"id": 2, "p": 40}, {"id": 200002, "p": 30}]}, [UNKNOWN, GAP], UNEXPLORED_FLOOR)
    assert above.floored == () and above.probs[2] == pytest.approx(0.40), "a serious valuation stands"
    off = SearchNodeOracle.score(reply, nodes, 0.0)
    assert off.floored == () and off.probs[2] == pytest.approx(0.01), "0 disables the floor"


# -- home_here: the model's STEP 2 as a structured verdict (2026-10-05, the Ranchester couch) -----
def test_when_the_model_says_the_home_type_is_elsewhere_unexplored_places_read_at_the_elsewhere_value():
    """Ranchester couch: 'living rooms are downstairs' at every loop point, 25 on every upstairs gap all
    the same (the example's number), 60 on the stairs -- and thirteen peeks before the stairs. Rule 2b's
    'less' is the code's to apply from the model's own verdict."""
    nodes = [KITCHEN, UNKNOWN, UP, GAP, DOORWAY_WITH_TOILET, WEAK_BEDROOM_UNENTERED]
    reply = {"home": "living room", "storey": "upper floor (bedroom found); living rooms are downstairs",
             "home_here": "elsewhere",
             "nodes": [{"id": 0, "p": 0}, {"id": 2, "why": "small unknown room, never entered", "p": 25},
                       {"id": 100000, "p": 60}, {"id": 200002, "why": "unseen room, nothing known", "p": 25},
                       {"id": 200001, "why": "toilet glimpsed: a bathroom", "p": 1}, {"id": 4, "why": "bedroom", "p": 2}]}
    result = SearchNodeOracle.score(reply, nodes, UNEXPLORED_FLOOR, UNEXPLORED_ELSEWHERE)
    assert result.home_here == "elsewhere" and result.reading["home_here"] == "elsewhere"
    assert result.capped == (2, 200002) and result.floored == ()
    assert result.probs[2] == result.probs[200002] == pytest.approx(0.10), "the unexplored upstairs places"
    assert result.probs[100000] == pytest.approx(0.60) and result.probs[200001] == pytest.approx(0.01) and result.probs[4] == pytest.approx(0.02), (
        "the stairs, a glimpsed opening and a typed room keep the model's numbers")
    # Written off instead (the 2026-10-04 failure): raised to the elsewhere value, not to the 0.25 floor.
    low = dict(reply, nodes=[{"id": 2, "p": 0}, {"id": 200002, "p": 1}, {"id": 100000, "p": 60}])
    result = SearchNodeOracle.score(low, [UNKNOWN, GAP, UP], UNEXPLORED_FLOOR, UNEXPLORED_ELSEWHERE)
    assert result.floored == (2, 200002) and result.capped == () and result.probs[2] == pytest.approx(0.10)
    # 'found' and 'missing' keep the floor; so does a reply without the field (an older model's).
    for verdict in ("found", "missing", None):
        same = dict(low, home_here=verdict) if verdict else {k: v for k, v in low.items() if k != "home_here"}
        result = SearchNodeOracle.score(same, [UNKNOWN, GAP, UP], UNEXPLORED_FLOOR, UNEXPLORED_ELSEWHERE)
        assert result.home_here == verdict and result.probs[2] == pytest.approx(0.25) and result.capped == ()
    # 0 leaves the model's numbers alone in the elsewhere case.
    kept = SearchNodeOracle.score(reply, nodes, UNEXPLORED_FLOOR, 0.0)
    assert kept.probs[2] == pytest.approx(0.25) and kept.capped == () and kept.floored == ()


def test_the_home_here_verdict_is_read_from_the_words_a_model_drifts_to():
    for text, verdict in (("found", "found"), ("Elsewhere", "elsewhere"), ("missing", "missing"),
                          ("not found yet", "missing"), ("not here, downstairs", "elsewhere"),
                          ("on this storey", "found"), ("another floor", "elsewhere"), ("", None), (7, None)):
        assert parse_home_here({"home_here": text}) == verdict, text
    assert parse_home_here({}) is None and parse_home_here("nonsense") is None


def test_the_prompt_asks_for_home_here_and_the_example_applies_rule_2b():
    assert '"home_here":"found|missing|elsewhere"' in SYSTEM_PROMPT
    assert '"home_here":"elsewhere"' in SYSTEM_PROMPT, "the worked example is an upper storey for a television"
    assert '"p":10}' in SYSTEM_PROMPT and 'about 10 when home_here is "elsewhere"' in SYSTEM_PROMPT
    oracle = SearchNodeOracle(Scripted({"nodes": []}), unexplored_floor=0.25, unexplored_elsewhere=0.1)
    assert oracle.unexplored_elsewhere == 0.1
    with pytest.raises(ValueError):
        SearchNodeOracle(Scripted({}), unexplored_elsewhere=1.0)


def test_the_oracle_applies_its_floor_to_fresh_and_reused_replies():
    client = Scripted({"nodes": [{"id": 2, "why": "too small", "p": 1}, {"id": 100000, "p": 60}]})
    oracle = SearchNodeOracle(client, unexplored_floor=0.3)
    first = oracle.probabilities("toilet", [UNKNOWN, UP])
    assert first.probs[2] == pytest.approx(0.3) and first.floored == (2,) and first.probs[100000] == pytest.approx(0.6)
    again = oracle.probabilities("toilet", [UNKNOWN, UP])
    assert again.reused and again.probs[2] == pytest.approx(0.3) and client.calls == 1
    with pytest.raises(ValueError):
        SearchNodeOracle(client, unexplored_floor=1.0)
