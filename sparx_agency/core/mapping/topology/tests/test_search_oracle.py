"""Tests for the LLM search oracle (LLM client mocked)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from sparx_agency.core.mapping.topology.search_oracle import (
    OracleResult,
    OracleRoom,
    SYSTEM_PROMPT,
    SearchOracle,
    USER_PROMPT_TEMPLATE,
    format_rooms_block,
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
               observed_classes=("refrigerator", "sink")),
    OracleRoom(id=1, label="bedroom", searched_s=60.4, frontier_clusters=1,
               observed_classes=("bed", "nightstand", "bed")),
    OracleRoom(id=2, label="hallway", searched_s=0.0, frontier_clusters=2),
]


def _reply(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"rooms": entries}


# ---------------------------------------------------------------------
#  Prompt formatting
# ---------------------------------------------------------------------
def test_format_rooms_block_exact_lines():
    block = format_rooms_block(ROOMS)
    assert block.split("\n") == [
        "1. Room id=0  type=kitchen  searched=5s  "
        "remaining_frontier_clusters=3. observed: refrigerator, sink",
        "2. Room id=1  type=bedroom  searched=60s  "
        "remaining_frontier_clusters=1. observed: bed, nightstand",
        "3. Room id=2  type=hallway  searched=0s  "
        "remaining_frontier_clusters=2. observed: (none)",
    ]


def test_user_prompt_carries_target_block_and_count():
    client = FakeClient([_reply([{"id": 0, "probability": 1.0}])])
    SearchOracle(client).probabilities("car keys", ROOMS)
    call = client.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["user"] == USER_PROMPT_TEMPLATE.format(
        target="car keys",
        rooms_block=format_rooms_block(ROOMS),
        n_rooms=3,
    )


# ---------------------------------------------------------------------
#  Happy path: clamp, zero-fill, normalize
# ---------------------------------------------------------------------
def test_probabilities_normalize_to_one():
    client = FakeClient([_reply([
        {"id": 0, "probability": 0.6, "reason": "keys land in kitchens"},
        {"id": 1, "probability": 0.3, "reason": "nightstand plausible"},
        {"id": 2, "probability": 0.3, "reason": "unsearched"},
    ])])
    result = SearchOracle(client).probabilities("car keys", ROOMS)
    assert result.source == "llm"
    assert sum(result.probs.values()) == pytest.approx(1.0)
    assert result.probs[0] == pytest.approx(0.5)
    assert result.probs[1] == pytest.approx(0.25)
    assert result.probs[2] == pytest.approx(0.25)
    assert result.reasons[0] == "keys land in kitchens"


def test_raw_values_clamped_to_unit_interval():
    client = FakeClient([_reply([
        {"id": 0, "probability": 5.0},     # clamps to 1.0
        {"id": 1, "probability": -0.3},    # clamps to 0.0
        {"id": 2, "probability": 1.0},
    ])])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert result.probs[0] == pytest.approx(0.5)
    assert result.probs[1] == pytest.approx(0.0)
    assert result.probs[2] == pytest.approx(0.5)


def test_missing_room_gets_zero():
    client = FakeClient([_reply([
        {"id": 0, "probability": 0.4},
        {"id": 2, "probability": 0.4},
        # id=1 omitted by the model
    ])])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert result.probs[1] == 0.0
    assert result.reasons[1] == ""
    assert sum(result.probs.values()) == pytest.approx(1.0)


def test_invented_room_ids_are_ignored():
    client = FakeClient([_reply([
        {"id": 0, "probability": 0.5},
        {"id": 99, "probability": 0.9},    # not a room we asked about
    ])])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert set(result.probs.keys()) == {0, 1, 2}
    assert result.probs[0] == pytest.approx(1.0)


def test_malformed_entries_skipped():
    client = FakeClient([_reply([
        "not a dict",
        {"id": "zero", "probability": 0.5},
        {"id": 1, "probability": "high"},
        {"id": 2, "probability": 0.5},
    ])])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert result.source == "llm"
    assert result.probs[2] == pytest.approx(1.0)


def test_reason_truncated_to_200_chars():
    client = FakeClient([_reply([
        {"id": 0, "probability": 1.0, "reason": "r" * 500}])])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert len(result.reasons[0]) == 200


def test_raw_reply_kept_for_debugging():
    reply = _reply([{"id": 0, "probability": 1.0}])
    client = FakeClient([reply])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    assert result.raw_reply == reply


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


def test_all_zero_probabilities_fall_back_to_uniform():
    reply = _reply([{"id": rid, "probability": 0.0} for rid in (0, 1, 2)])
    client = FakeClient([reply])
    result = SearchOracle(client).probabilities("apple", ROOMS)
    _assert_uniform(result)
    assert result.raw_reply == reply


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
