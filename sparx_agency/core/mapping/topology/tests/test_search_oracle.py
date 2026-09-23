"""Tests for the LLM search oracle (LLM client mocked).

The centre of gravity here is one flight failure. Over three equally
unsearched rooms a 3B model answered ``1.0 / 0.0 / 0.0``; the old parse
preserved that one-hot exactly, the search policy's ``min_prob`` filter then
dropped every room but one, and the "ranking" the whole method rests on had a
single candidate. :func:`test_a_one_hot_reply_becomes_a_ranking_not_a_verdict`
is that exact reply, and it now has to come out as a ranking.

Four groups:

* **the prompt** -- the exact lines the model is shown. The effort numbers are
  deliberately absent from it: a small model shown a number it was told to
  ignore double-counts it into the semantic judgement instead;
* **the scoring** -- clamp, shrink toward the mean, size, effort. Each term is
  pinned separately, because when the ordering comes out wrong in flight the
  question is always *which* term did it;
* **robustness** -- the shapes a small model actually emits: a missing room,
  an invented id, the old ``probability`` key, prose where a number should be;
* **the fallbacks** -- every path to uniform, which is the only honest answer
  when the model said nothing usable.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from sparx_agency.core.mapping.topology.search_oracle import (
    OracleResult,
    OracleRoom,
    OracleScoring,
    SYSTEM_PROMPT,
    SearchOracle,
    USER_PROMPT_TEMPLATE,
    affinity_weights,
    effort_factor,
    format_rooms_block,
    size_factor,
)


class FakeClient:
    def __init__(self, replies: List[Any]):
        self.replies = list(replies)
        self.calls: List[Dict[str, str]] = []

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


ROOMS = [
    OracleRoom(id=0, label="kitchen", searched_s=5.0, frontier_clusters=3,
               observed_classes=("refrigerator", "sink"), area_m2=18.0),
    OracleRoom(id=1, label="bedroom", searched_s=60.4, frontier_clusters=1,
               observed_classes=("bed", "nightstand", "bed"), area_m2=25.0),
    OracleRoom(id=2, label="hallway", searched_s=0.0, frontier_clusters=2,
               area_m2=6.0),
]

#: Three rooms identical in every way the code looks at, so a difference in
#: the output can only have come from the model's own scores.
EQUAL_ROOMS = [
    OracleRoom(id=0, label="unknown", searched_s=0.0, frontier_clusters=1,
               area_m2=20.0),
    OracleRoom(id=1, label="unknown", searched_s=0.0, frontier_clusters=1,
               area_m2=20.0),
    OracleRoom(id=2, label="unknown", searched_s=0.0, frontier_clusters=1,
               area_m2=20.0),
]


def _reply(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"rooms": entries}


def _scores(*pairs) -> Dict[str, Any]:
    return _reply([{"id": rid, "why": "because", "score": s}
                   for rid, s in pairs])


# ---------------------------------------------------------------------
#  The prompt
# ---------------------------------------------------------------------
def test_format_rooms_block_exact_lines():
    """No ordinal prefix: a leading '1.' makes a 3B answer with the ordinal."""
    assert format_rooms_block(ROOMS).split("\n") == [
        "id=0  type=kitchen  size=18m2  seen: refrigerator, sink",
        "id=1  type=bedroom  size=25m2  seen: bed, nightstand",
        "id=2  type=hallway  size=6m2  seen: nothing yet",
    ]


def test_the_prompt_never_shows_the_effort_numbers():
    """Told to ignore a number it can see, a small model uses it anyway."""
    block = format_rooms_block(ROOMS)
    assert "60" not in block, "searched seconds leaked into the prompt"
    assert "searched" not in block
    assert "frontier" not in block


def test_an_unknown_size_says_so_rather_than_showing_zero():
    room = OracleRoom(id=4, label="unknown", area_m2=0.0)
    assert "unknown size" in format_rooms_block([room])


def test_observed_classes_are_capped_lowercased_and_deduplicated():
    room = OracleRoom(id=1, label="ward", area_m2=30.0,
                      observed_classes=tuple(
                          ["Chair", "chair", "bed", "sink", "tv", "cart",
                           "desk", "lamp", "stool"]))
    line = format_rooms_block([room])
    assert "Chair" not in line
    assert line.count(",") == 5, "at most six classes reach the model"


def test_user_prompt_carries_target_block_and_count():
    client = FakeClient([_scores((0, 60), (1, 40), (2, 50))])
    SearchOracle(client).probabilities("car keys", ROOMS)
    call = client.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["user"] == USER_PROMPT_TEMPLATE.format(
        target="car keys",
        rooms_block=format_rooms_block(ROOMS),
        n_rooms=3,
    )


def test_the_system_prompt_forbids_the_two_degenerate_answers():
    assert "NEVER use 100 and NEVER use 0" in SYSTEM_PROMPT
    assert "not identified yet" in SYSTEM_PROMPT, "unknown is the common case"


# ---------------------------------------------------------------------
#  The scoring
# ---------------------------------------------------------------------
def test_a_one_hot_reply_becomes_a_ranking_not_a_verdict():
    """The exact flight failure: 100/0/0 over three equal rooms.

    Clamped to 90/3/3, shrunk toward their own mean of 32 at w=0.7, this is
    0.756 / 0.122 / 0.122 -- the model's ORDERING intact, its false certainty
    gone, and three candidates for the policy to draw from instead of one.
    """
    client = FakeClient([_scores((0, 100), (1, 0), (2, 0))])
    result = SearchOracle(client).probabilities("wheelchair", EQUAL_ROOMS)
    assert result.source == "llm"
    assert result.probs[0] == pytest.approx(0.75625)
    assert result.probs[1] == pytest.approx(0.121875)
    assert result.probs[2] == pytest.approx(0.121875)
    assert result.probs[0] > result.probs[1], "the ordering must survive"
    assert min(result.probs.values()) > 0.01, (
        "every room must stay above the policy's min_prob filter")


def test_the_raw_scores_are_kept_so_a_degenerate_tick_is_greppable():
    client = FakeClient([_scores((0, 100), (1, 0), (2, 0))])
    result = SearchOracle(client).probabilities("wheelchair", EQUAL_ROOMS)
    assert result.scores == {0: 100.0, 1: 0.0, 2: 0.0}
    assert result.spread > 0.0


def test_ordering_is_preserved_exactly():
    client = FakeClient([_scores((0, 30), (1, 85), (2, 55))])
    result = SearchOracle(client).probabilities("bed", EQUAL_ROOMS)
    assert result.probs[1] > result.probs[2] > result.probs[0]


def test_equal_scores_give_equal_probabilities():
    client = FakeClient([_scores((0, 50), (1, 50), (2, 50))])
    result = SearchOracle(client).probabilities("x", EQUAL_ROOMS)
    for rid in (0, 1, 2):
        assert result.probs[rid] == pytest.approx(1.0 / 3.0)


def test_a_missing_room_gets_the_mean_not_zero():
    """An unmentioned room is one the model forgot, not one it ruled out."""
    client = FakeClient([_scores((0, 80), (2, 40))])
    result = SearchOracle(client).probabilities("x", EQUAL_ROOMS)
    assert result.probs[1] > 0.0
    assert result.probs[0] > result.probs[1] > result.probs[2]


def test_a_bigger_room_outranks_a_smaller_one_at_equal_score():
    rooms = [OracleRoom(id=0, label="ward", area_m2=80.0, frontier_clusters=1),
             OracleRoom(id=1, label="ward", area_m2=5.0, frontier_clusters=1)]
    client = FakeClient([_scores((0, 60), (1, 60))])
    result = SearchOracle(client).probabilities("wheelchair", rooms)
    assert result.probs[0] > result.probs[1]


def test_a_searched_room_is_discounted_but_not_written_off():
    rooms = [OracleRoom(id=0, label="ward", area_m2=20.0, frontier_clusters=2,
                        searched_s=0.0),
             OracleRoom(id=1, label="ward", area_m2=20.0, frontier_clusters=2,
                        searched_s=600.0)]
    client = FakeClient([_scores((0, 60), (1, 60))])
    result = SearchOracle(client).probabilities("x", rooms)
    assert result.probs[0] > result.probs[1]
    assert result.probs[1] >= 0.15, (
        "frontier remains, so the room is unfinished, not cleared")


def test_an_exhausted_room_is_discounted_hard():
    rooms = [OracleRoom(id=0, label="ward", area_m2=20.0, frontier_clusters=2,
                        searched_s=60.0),
             OracleRoom(id=1, label="ward", area_m2=20.0, frontier_clusters=0,
                        searched_s=60.0)]
    client = FakeClient([_scores((0, 60), (1, 60))])
    result = SearchOracle(client).probabilities("x", rooms)
    assert result.probs[1] < 0.5 * result.probs[0]
    assert result.probs[1] > 0.0, "the detector can miss; never exactly zero"


def test_p_present_is_below_one_so_unmapped_space_keeps_its_mass():
    client = FakeClient([_scores((0, 50), (1, 50), (2, 50))])
    result = SearchOracle(client).probabilities("x", EQUAL_ROOMS)
    assert sum(result.probs.values()) == pytest.approx(1.0)
    assert result.p_present < 1.0


def test_a_big_unsearched_high_scoring_room_pushes_p_present_up():
    low = FakeClient([_scores((0, 10), (1, 10), (2, 10))])
    high = FakeClient([_scores((0, 90), (1, 90), (2, 90))])
    a = SearchOracle(low).probabilities("x", EQUAL_ROOMS)
    b = SearchOracle(high).probabilities("x", EQUAL_ROOMS)
    assert b.p_present > a.p_present


# -- the terms in isolation ----------------------------------------------
def test_affinity_weights_clamp_and_shrink():
    w = affinity_weights({0: 100.0, 1: 0.0}, [0, 1], OracleScoring())
    assert w[0] == pytest.approx(0.7 * 90.0 + 0.3 * 46.5)
    assert w[1] == pytest.approx(0.7 * 3.0 + 0.3 * 46.5)


def test_size_factor_is_one_at_the_reference_and_unknown_area():
    s = OracleScoring()
    assert size_factor(s.area_ref_m2, s) == pytest.approx(1.0)
    assert size_factor(0.0, s) == pytest.approx(1.0)


def test_size_factor_is_clamped_so_it_never_decides_alone():
    s = OracleScoring()
    assert size_factor(100000.0, s) == pytest.approx(s.area_clamp[1])
    assert size_factor(0.01, s) == pytest.approx(s.area_clamp[0])


def test_a_strong_semantic_match_beats_a_much_bigger_room():
    """Measured failure: a 6 m2 bathroom scored 85 lost to a 48 m2 ward at 30."""
    rooms = [OracleRoom(id=0, label="ward", area_m2=48.0, frontier_clusters=3),
             OracleRoom(id=1, label="bathroom", area_m2=6.0,
                        frontier_clusters=1)]
    client = FakeClient([_scores((0, 30), (1, 85))])
    result = SearchOracle(client).probabilities("toilet", rooms)
    assert result.probs[1] > result.probs[0], (
        "room size overturned the model's semantic judgement")


def test_effort_factor_floors_while_frontier_remains():
    s = OracleScoring()
    assert effort_factor(1e6, 3, 20.0, s) == pytest.approx(s.effort_floor)
    assert effort_factor(0.0, 3, 20.0, s) == pytest.approx(1.0)


def test_effort_half_life_scales_with_room_area():
    s = OracleScoring()
    small = effort_factor(90.0, 3, 20.0, s)
    large = effort_factor(90.0, 3, 200.0, s)
    assert large > small, "a ward must not be written off like a cupboard"


# ---------------------------------------------------------------------
#  Robustness to what a small model actually emits
# ---------------------------------------------------------------------
def test_the_old_probability_key_is_still_accepted():
    """A model that answers in the previous shape is usable, not discarded."""
    client = FakeClient([_reply([
        {"id": 0, "probability": 0.9},
        {"id": 1, "probability": 0.1},
        {"id": 2, "probability": 0.1},
    ])])
    result = SearchOracle(client).probabilities("apple", EQUAL_ROOMS)
    assert result.source == "llm"
    assert result.probs[0] > result.probs[1]


def test_invented_room_ids_are_dropped():
    client = FakeClient([_scores((0, 50), (99, 90))])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert set(result.probs.keys()) == {0, 1, 2}
    assert 99 not in result.scores


def test_malformed_entries_skipped():
    client = FakeClient([_reply([
        "not a dict",
        {"id": "zero", "score": 50},
        {"id": 1, "score": "high"},
        {"id": 2, "score": 70},
    ])])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert result.source == "llm"
    assert result.scores == {2: 70.0}


def test_reason_truncated_to_200_chars():
    client = FakeClient([_reply([{"id": 0, "score": 60, "why": "r" * 500}])])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert len(result.reasons[0]) == 200


def test_raw_reply_kept_for_debugging():
    reply = _scores((0, 60), (1, 40), (2, 50))
    client = FakeClient([reply])
    assert SearchOracle(client).probabilities("apple", ROOMS).raw_reply == reply


def test_scoring_knobs_are_injectable():
    """w_llm=1.0 trusts the model completely -- the pre-shrink behaviour."""
    client = FakeClient([_scores((0, 90), (1, 3), (2, 3))])
    result = SearchOracle(
        client, OracleScoring(w_llm=1.0)).probabilities("x", EQUAL_ROOMS)
    assert result.probs[0] == pytest.approx(90.0 / 96.0)


# ---------------------------------------------------------------------
#  Fallbacks
# ---------------------------------------------------------------------
def _assert_uniform(result: OracleResult) -> None:
    assert result.source == "uniform_fallback"
    assert sum(result.probs.values()) == pytest.approx(1.0)
    for rid in (0, 1, 2):
        assert result.probs[rid] == pytest.approx(1.0 / 3.0)
        assert result.reasons[rid] == ""


def test_llm_error_falls_back_to_uniform():
    client = FakeClient([RuntimeError("server down")])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    _assert_uniform(result)
    assert result.raw_reply is None


def test_all_zero_scores_are_a_ranking_not_a_fallback():
    """Zero means "surprising", not "impossible" -- it clamps to the floor."""
    reply = _scores((0, 0), (1, 0), (2, 0))
    client = FakeClient([reply])
    result = SearchOracle(client).probabilities("apple", EQUAL_ROOMS)
    assert result.source == "llm"
    for rid in (0, 1, 2):
        assert result.probs[rid] == pytest.approx(1.0 / 3.0)


def test_no_usable_entry_falls_back_to_uniform():
    client = FakeClient([_reply([{"id": "x"}, {"nope": 1}])])
    _assert_uniform(SearchOracle(client).probabilities("apple", ROOMS))


def test_missing_rooms_list_falls_back_to_uniform():
    client = FakeClient([{"answer": 42}])
    _assert_uniform(SearchOracle(client).probabilities("apple", ROOMS))


def test_empty_rooms_list_falls_back_to_uniform():
    client = FakeClient([_reply([])])
    _assert_uniform(SearchOracle(client).probabilities("apple", ROOMS))


def test_no_rooms_raises():
    client = FakeClient([])
    with pytest.raises(ValueError, match="at least one room"):
        SearchOracle(client).probabilities("apple", [])
    assert client.calls == []
