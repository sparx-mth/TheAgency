"""Tests for the LLM room-type classifier (LLM client mocked)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from sparx_agency.core.mapping.topology.room_classifier import (
    DEFAULT_LABEL_SET,
    RoomLabel,
    RoomTypeClassifier,
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    format_object_list,
)


class FakeClient:
    """Duck-typed LLMClient: scripted chat_json replies, call log."""

    def __init__(self, replies: List[Any]):
        self.replies = list(replies)
        self.calls: List[Dict[str, str]] = []

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


KITCHEN_REPLY = {"label": "kitchen", "confidence": 0.88,
                 "reasoning": "has refrigerator + sink"}


# ---------------------------------------------------------------------
#  format_object_list
# ---------------------------------------------------------------------
def test_format_object_list_collapses_duplicates_sorted():
    out = format_object_list(["sink", "chair", "sink", "sink", "chair"])
    assert out == "- chair x2\n- sink x3"


def test_format_object_list_empty():
    assert format_object_list([]) == "(no objects observed yet)"


# ---------------------------------------------------------------------
#  classify
# ---------------------------------------------------------------------
def test_classify_returns_parsed_room_label():
    client = FakeClient([KITCHEN_REPLY])
    clf = RoomTypeClassifier(client)
    result = clf.classify(["refrigerator", "sink"])
    assert result == RoomLabel(label="kitchen", confidence=0.88,
                               reasoning="has refrigerator + sink")
    assert len(client.calls) == 1


def test_prompts_carry_label_set_and_object_counts():
    client = FakeClient([KITCHEN_REPLY])
    clf = RoomTypeClassifier(client)
    clf.classify(["sink", "sink", "oven"])
    call = client.calls[0]
    assert call["system"] == SYSTEM_PROMPT_TEMPLATE.format(
        label_set=", ".join(DEFAULT_LABEL_SET))
    assert call["user"] == USER_PROMPT_TEMPLATE.format(
        obj_list="- oven x1\n- sink x2")


def test_min_objects_gate_skips_llm():
    client = FakeClient([])
    clf = RoomTypeClassifier(client, min_objects=1)
    result = clf.classify([])
    assert result.label == "unknown"
    assert result.confidence == 0.0
    assert result.reasoning == "no objects observed yet"
    assert client.calls == []


def test_min_objects_gate_counts_normalized_classes():
    # Empty / whitespace-only class names don't count toward the gate.
    client = FakeClient([])
    clf = RoomTypeClassifier(client, min_objects=2)
    result = clf.classify(["bed", "", "   "])
    assert result.label == "unknown"
    assert client.calls == []


def test_out_of_set_label_coerced_to_unknown():
    client = FakeClient([{"label": "spaceship_bridge", "confidence": 0.9,
                          "reasoning": "consoles everywhere"}])
    clf = RoomTypeClassifier(client)
    result = clf.classify(["console"])
    assert result.label == "unknown"
    assert result.confidence == pytest.approx(0.9)


def test_unparseable_confidence_becomes_zero():
    client = FakeClient([{"label": "kitchen", "confidence": "very high",
                          "reasoning": ""}])
    clf = RoomTypeClassifier(client)
    assert clf.classify(["sink"]).confidence == 0.0


def test_reasoning_truncated_to_200_chars():
    client = FakeClient([{"label": "kitchen", "confidence": 0.5,
                          "reasoning": "x" * 500}])
    clf = RoomTypeClassifier(client)
    assert len(clf.classify(["sink"]).reasoning) == 200


def test_custom_label_set():
    client = FakeClient([{"label": "cockpit", "confidence": 0.7,
                          "reasoning": ""}])
    clf = RoomTypeClassifier(client, label_set=["cockpit", "unknown"])
    assert clf.classify(["yoke"]).label == "cockpit"


# ---------------------------------------------------------------------
#  Signature cache
# ---------------------------------------------------------------------
def test_same_signature_calls_llm_once():
    client = FakeClient([KITCHEN_REPLY])
    clf = RoomTypeClassifier(client)
    a = clf.classify(["refrigerator", "sink"])
    b = clf.classify(["refrigerator", "sink"])
    assert a == b
    assert len(client.calls) == 1
    assert clf.cache_size == 1


def test_counts_change_but_classes_dont_no_recall():
    client = FakeClient([KITCHEN_REPLY])
    clf = RoomTypeClassifier(client)
    clf.classify(["sink", "refrigerator"])
    # More sinks appear; the *set* of classes is unchanged.
    result = clf.classify(["sink", "sink", "sink", "refrigerator"])
    assert result.label == "kitchen"
    assert len(client.calls) == 1


def test_case_and_whitespace_normalized_into_same_signature():
    client = FakeClient([KITCHEN_REPLY])
    clf = RoomTypeClassifier(client)
    clf.classify(["Sink", " Refrigerator "])
    clf.classify(["sink", "refrigerator"])
    assert len(client.calls) == 1


def test_new_class_added_triggers_new_call():
    client = FakeClient([KITCHEN_REPLY,
                         {"label": "dining_room", "confidence": 0.6,
                          "reasoning": ""}])
    clf = RoomTypeClassifier(client)
    clf.classify(["sink"])
    result = clf.classify(["sink", "dining_table"])
    assert result.label == "dining_room"
    assert len(client.calls) == 2
    assert clf.cache_size == 2


def test_llm_failure_propagates_and_is_not_cached():
    client = FakeClient([RuntimeError("server down"), KITCHEN_REPLY])
    clf = RoomTypeClassifier(client)
    with pytest.raises(RuntimeError, match="server down"):
        clf.classify(["sink"])
    assert clf.cache_size == 0
    # Retry after the failure reaches the LLM again and succeeds.
    assert clf.classify(["sink"]).label == "kitchen"
    assert len(client.calls) == 2


# ---------------------------------------------------------------------
#  Label set contract
# ---------------------------------------------------------------------
def test_default_label_set_has_15_labels_incl_unknown():
    assert len(DEFAULT_LABEL_SET) == 15
    assert "unknown" in DEFAULT_LABEL_SET
