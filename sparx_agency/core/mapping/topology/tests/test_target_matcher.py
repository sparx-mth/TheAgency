"""Tests for the target-vs-class match ladder (LLM client mocked)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from sparx_agency.core.mapping.topology.target_matcher import (
    MATCH_SYSTEM,
    MATCH_USER_TEMPLATE,
    TargetMatcher,
    fallback_match,
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


class ExplodingClient:
    """A client that must never be consulted."""

    def chat_json(self, system: str, user: str):
        raise AssertionError("LLM was consulted when it must not be")


# ---------------------------------------------------------------------
#  Offline fallback matcher
# ---------------------------------------------------------------------
def test_fallback_exact():
    assert fallback_match("toilet", "toilet") is True


def test_fallback_token_overlap():
    assert fallback_match("car keys", "keys") is True
    assert fallback_match("toilet_seat", "toilet") is True   # underscore split


def test_fallback_substring():
    assert fallback_match("key", "keychain") is True


def test_fallback_documented_looseness_car_keys_vs_car():
    # Shared token 'car' matches offline even though the LLM prompt's
    # guidelines call this pair FALSE (accessory-of relation).
    assert fallback_match("car keys", "car") is True


def test_fallback_no_relation():
    assert fallback_match("apple", "fruit") is False
    assert fallback_match("laptop", "monitor") is False


def test_fallback_empty_strings():
    assert fallback_match("", "toilet") is False
    assert fallback_match("toilet", "  ") is False


# ---------------------------------------------------------------------
#  Ladder rung 1: exact short-circuit
# ---------------------------------------------------------------------
def test_exact_match_never_asks_llm():
    # Small local models were seen answering False for identical
    # strings — the short-circuit protects against that.
    matcher = TargetMatcher(client=ExplodingClient())
    result = matcher.matches("toilet", "toilet")
    assert result.match is True
    assert result.reason == "exact match"


def test_exact_match_is_case_and_whitespace_insensitive():
    matcher = TargetMatcher(client=ExplodingClient())
    assert matcher.matches(" Toilet ", "toilet").match is True


def test_empty_inputs_never_match():
    matcher = TargetMatcher(client=ExplodingClient())
    assert matcher.matches("", "").match is False
    assert matcher.matches("toilet", "").match is False


# ---------------------------------------------------------------------
#  Ladder rung 2+3: cache then LLM
# ---------------------------------------------------------------------
def test_llm_verdict_true_with_reason():
    client = FakeClient([{"match": True, "reason": "sofa is a couch"}])
    matcher = TargetMatcher(client=client)
    result = matcher.matches("couch", "sofa")
    assert result.match is True
    assert result.reason == "sofa is a couch"
    call = client.calls[0]
    assert call["system"] == MATCH_SYSTEM
    assert call["user"] == MATCH_USER_TEMPLATE.format(target="couch",
                                                      cname="sofa")


def test_llm_verdict_false_is_cached_too():
    client = FakeClient([{"match": False, "reason": "car is not keys"}])
    matcher = TargetMatcher(client=client)
    assert matcher.matches("car keys", "car").match is False
    assert matcher.matches("car keys", "car").match is False
    assert len(client.calls) == 1
    assert matcher.cache_size == 1


def test_cache_hits_skip_llm_on_repeats():
    client = FakeClient([{"match": True, "reason": "yes"}])
    matcher = TargetMatcher(client=client)
    matcher.matches("couch", "sofa")
    result = matcher.matches("Couch", " SOFA ")   # normalizes to same key
    assert result.match is True
    assert len(client.calls) == 1


def test_missing_match_field_defaults_false():
    client = FakeClient([{"reason": "confused"}])
    matcher = TargetMatcher(client=client)
    assert matcher.matches("couch", "sofa").match is False


def test_llm_reason_truncated_to_160_chars():
    client = FakeClient([{"match": True, "reason": "r" * 400}])
    matcher = TargetMatcher(client=client)
    assert len(matcher.matches("couch", "sofa").reason) == 160


def test_quoted_false_is_not_a_match():
    """A regression pinned by a real hospital flight.

    qwen2.5:3b-instruct answers ``{"match": "false"}`` with the word
    quoted. The verdict was read with ``bool()``, and ``bool("false")`` is
    True, so the target watcher counted every detection as a hit and
    latched ``/target_seen`` on the first object of the flight -- a shelf,
    carrying the model's own reason "CLASS is a different object".
    """
    client = FakeClient([{"match": "false",
                          "reason": "CLASS is a different object"}])
    matcher = TargetMatcher(client=client)
    assert matcher.matches("wheelchair", "shelf").match is False


@pytest.mark.parametrize("raw, expected", [
    (True, True), (False, False),
    ("true", True), ("false", False),
    ("True", True), ("FALSE", False),
    ("yes", True), ("no", False),
    (1, True), (0, False),
    ("", False), (None, False), ("perhaps", False),
])
def test_llm_verdict_coercion(raw, expected):
    """Every shape a small model spells its boolean in, and the default.

    Unrecognised values default to False: a target watcher that stops the
    search is a worse failure than one that keeps looking.
    """
    client = FakeClient([{"match": raw, "reason": "r"}])
    matcher = TargetMatcher(client=client)
    assert matcher.matches("wheelchair", "shelf").match is expected


# ---------------------------------------------------------------------
#  Ladder rung 4: offline fallback
# ---------------------------------------------------------------------
def test_llm_error_falls_back_to_token_overlap():
    client = FakeClient([RuntimeError("server down")])
    matcher = TargetMatcher(client=client)
    result = matcher.matches("car keys", "keys")
    assert result.match is True
    assert result.reason == "offline token-overlap fallback"
    assert matcher.cache_size == 1


def test_no_client_uses_fallback_directly():
    matcher = TargetMatcher(client=None)
    assert matcher.matches("car keys", "keys").match is True
    assert matcher.matches("apple", "fruit").match is False


def test_use_llm_false_ignores_client():
    matcher = TargetMatcher(client=ExplodingClient(), use_llm=False)
    result = matcher.matches("car keys", "keys")
    assert result.match is True
    assert result.reason == "offline token-overlap fallback"


def test_fallback_verdict_is_cached():
    matcher = TargetMatcher(client=None)
    matcher.matches("apple", "fruit")
    matcher.matches("apple", "fruit")
    assert matcher.cache_size == 1


def test_full_ladder_order():
    # 1) exact never consults the LLM; 2) a novel pair does; 3) the same
    # pair again hits the cache; 4) after an LLM error the fallback answers.
    client = FakeClient([{"match": True, "reason": "synonym"},
                         RuntimeError("down")])
    matcher = TargetMatcher(client=client)
    assert matcher.matches("mug", "mug").match is True          # exact
    assert matcher.matches("mug", "cup").match is True          # LLM
    assert matcher.matches("mug", "cup").match is True          # cache
    assert len(client.calls) == 1
    result = matcher.matches("mug", "glass")                    # LLM errors
    assert result.reason == "offline token-overlap fallback"
    assert result.match is False
    assert len(client.calls) == 2
