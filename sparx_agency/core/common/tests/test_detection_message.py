"""Tests for the object-approach detections wire format."""
import json

import pytest

from sparx_agency.core.common.detection_message import (
    encode_detections,
    parse_detections_message,
)
from sparx_agency.core.common.types.perception import Detection2D


def _det(label="refrigerator", score=0.83, bbox=(10, 20, 30, 40), w=504, h=294):
    return Detection2D(label=label, score=score, bbox_xyxy=bbox, frame_w=w, frame_h=h)


def test_round_trip_preserves_everything():
    dets = [_det(), _det(label="hat", score=0.4, bbox=(1, 2, 3, 4))]
    parsed = parse_detections_message(encode_detections(dets, 12.5, 504, 294))
    assert parsed.stamp == 12.5
    assert (parsed.width, parsed.height) == (504, 294)
    assert [d.label for d in parsed.detections] == ["refrigerator", "hat"]
    assert parsed.detections[1].bbox_xyxy == (1, 2, 3, 4)
    assert parsed.detections[0].score == pytest.approx(0.83)


def test_round_trip_of_empty_detections():
    parsed = parse_detections_message(encode_detections([], 1.0, 8, 6))
    assert parsed.detections == []
    assert (parsed.width, parsed.height) == (8, 6)


def test_frame_size_is_stamped_onto_each_detection():
    # The consumer normalises boxes against the DETECTOR's frame, not its own.
    parsed = parse_detections_message(encode_detections([_det()], 0.0, 640, 480))
    assert (parsed.detections[0].frame_w, parsed.detections[0].frame_h) == (640, 480)


def test_labels_are_normalised_on_parse():
    payload = json.dumps({"stamp": 0.0, "w": 4, "h": 4,
                          "detections": [{"label": "  ReFrigerator ", "score": 0.5,
                                          "bbox": [0, 0, 1, 1]}]})
    assert parse_detections_message(payload).detections[0].label == "refrigerator"


def test_defaults_fill_absent_fields():
    payload = json.dumps({"detections": []})
    parsed = parse_detections_message(payload, default_width=504,
                                      default_height=294, default_stamp=7.0)
    assert (parsed.width, parsed.height, parsed.stamp) == (504, 294, 7.0)


def test_payload_wins_over_defaults():
    payload = json.dumps({"stamp": 1.0, "w": 8, "h": 6, "detections": []})
    parsed = parse_detections_message(payload, default_width=504,
                                      default_height=294, default_stamp=7.0)
    assert (parsed.width, parsed.height, parsed.stamp) == (8, 6, 1.0)


def test_missing_field_without_default_raises():
    with pytest.raises(ValueError, match="missing 'w'"):
        parse_detections_message(json.dumps({"stamp": 0.0, "h": 4, "detections": []}))


def test_bad_json_raises():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_detections_message("{not json")


def test_non_object_payload_raises():
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_detections_message("[1, 2, 3]")


def test_detections_not_a_list_raises():
    payload = json.dumps({"stamp": 0.0, "w": 4, "h": 4, "detections": {"a": 1}})
    with pytest.raises(ValueError, match="must be a list"):
        parse_detections_message(payload)


def test_missing_detection_field_raises():
    payload = json.dumps({"stamp": 0.0, "w": 4, "h": 4,
                          "detections": [{"label": "x", "score": 0.5}]})
    with pytest.raises(ValueError, match="missing field"):
        parse_detections_message(payload)


def test_malformed_bbox_raises():
    payload = json.dumps({"stamp": 0.0, "w": 4, "h": 4,
                          "detections": [{"label": "x", "score": 0.5,
                                          "bbox": ["a", 0, 1, 1]}]})
    with pytest.raises(ValueError, match="malformed"):
        parse_detections_message(payload)


def test_short_bbox_raises():
    payload = json.dumps({"stamp": 0.0, "w": 4, "h": 4,
                          "detections": [{"label": "x", "score": 0.5, "bbox": [0, 1]}]})
    with pytest.raises(ValueError, match="4 values"):
        parse_detections_message(payload)


def test_malformed_frame_size_raises():
    payload = json.dumps({"stamp": 0.0, "w": "wide", "h": 4, "detections": []})
    with pytest.raises(ValueError, match="'w' is malformed"):
        parse_detections_message(payload)
