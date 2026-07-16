"""Tests for the drone "thinking" message codec.

The codec is the contract between a dozen narrating nav nodes and the BEV
viewer's log, so the tests pin the round-trip and -- more importantly -- the
strictness: a mislabelled thought must fail at the publisher rather than reach
the operator wearing the wrong colour or severity.
"""
import json

import pytest

from sparx_agency.core.common.thought_message import (
    CATEGORIES, LEVELS, Thought, encode_thought, parse_thought_message)


def test_round_trip_preserves_every_field():
    wire = encode_thought("Aligning to waypoint 3 (x=1.0, y=2.0)", stamp=12.5,
                          category="nav", level="info", source="waypoint_follower")
    assert parse_thought_message(wire) == Thought(
        stamp=12.5, text="Aligning to waypoint 3 (x=1.0, y=2.0)",
        category="nav", level="info", source="waypoint_follower")


def test_encode_emits_the_documented_json_shape():
    payload = json.loads(encode_thought("Stopping to turn", stamp=1.0,
                                        source="waypoint_follower"))
    assert payload == {"stamp": 1.0, "text": "Stopping to turn",
                       "category": "nav", "level": "info",
                       "source": "waypoint_follower"}


def test_encode_strips_surrounding_whitespace():
    assert parse_thought_message(
        encode_thought("  Stopping to turn\n", stamp=1.0)).text == "Stopping to turn"


@pytest.mark.parametrize("text", ["", "   ", "\n", None])
def test_encode_rejects_empty_text(text):
    with pytest.raises(ValueError, match="empty"):
        encode_thought(text, stamp=1.0)


def test_encode_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown thought category"):
        encode_thought("hi", stamp=1.0, category="weather")


def test_encode_rejects_unknown_level():
    with pytest.raises(ValueError, match="unknown thought level"):
        encode_thought("hi", stamp=1.0, level="fatal")


@pytest.mark.parametrize("category", CATEGORIES)
def test_every_declared_category_round_trips(category):
    wire = encode_thought("thinking", stamp=0.0, category=category)
    assert parse_thought_message(wire).category == category


@pytest.mark.parametrize("level", LEVELS)
def test_every_declared_level_round_trips(level):
    wire = encode_thought("thinking", stamp=0.0, level=level)
    assert parse_thought_message(wire).level == level


def test_parse_rejects_non_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_thought_message("Stopping to turn")


def test_parse_rejects_non_object_json():
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_thought_message('["Stopping to turn"]')


def test_parse_rejects_missing_text():
    with pytest.raises(ValueError, match="missing 'text'"):
        parse_thought_message('{"stamp": 1.0}')


@pytest.mark.parametrize("text", ["null", "0", "{\"a\": 1}", "[1, 2]", "true"])
def test_parse_rejects_a_non_string_text(text):
    # str() would coerce these into real-looking lines ("None", "0", "{'a': 1}")
    # and put them in front of the operator as if the drone had thought them.
    with pytest.raises(ValueError, match="'text' must be a string"):
        parse_thought_message('{"text": %s, "stamp": 1.0}' % text)


def test_parse_rejects_a_non_string_source():
    with pytest.raises(ValueError, match="'source' must be a string"):
        parse_thought_message('{"text": "hi", "stamp": 1.0, "source": 3}')


def test_parse_rejects_malformed_stamp():
    with pytest.raises(ValueError, match="'stamp' is malformed"):
        parse_thought_message('{"text": "hi", "stamp": "soon"}')


def test_parse_uses_default_stamp_when_absent():
    assert parse_thought_message('{"text": "hi"}', default_stamp=7.5).stamp == 7.5


def test_parse_requires_a_stamp_when_no_default():
    with pytest.raises(ValueError, match="missing 'stamp'"):
        parse_thought_message('{"text": "hi"}')


def test_parse_defaults_category_level_and_source():
    got = parse_thought_message('{"text": "hi", "stamp": 1.0}')
    assert (got.category, got.level, got.source) == ("nav", "info", "")


def test_parse_rejects_unknown_category_from_the_wire():
    with pytest.raises(ValueError, match="unknown thought category"):
        parse_thought_message('{"text": "hi", "stamp": 1.0, "category": "weather"}')


def test_parse_rejects_unknown_level_from_the_wire():
    with pytest.raises(ValueError, match="unknown thought level"):
        parse_thought_message('{"text": "hi", "stamp": 1.0, "level": "fatal"}')
