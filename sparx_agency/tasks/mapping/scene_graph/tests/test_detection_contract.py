"""Tests for the scene-graph detection wire contract.

Covers the three legs of the protocol: JPEG frame round-trip (lossy, so
tolerance-based), detection JSON round-trip (exact), and the ported hospital
vocabulary (length + sentinels). Runs in the plain ``.venv`` — the contract
must never need torch.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.tasks.mapping.scene_graph.serve.contract import (
    DEFAULT_HOSPITAL_VOCABULARY,
    DEFAULT_PORT,
    DetectionWire,
    decode_frame,
    detections_from_json,
    detections_to_json,
    encode_frame,
)


def _synthetic_bgr(h: int = 48, w: int = 64) -> np.ndarray:
    """A smooth gradient frame (JPEG-friendly, so lossiness stays small)."""
    yy, xx = np.mgrid[0:h, 0:w]
    b = (255.0 * xx / max(w - 1, 1)).astype(np.uint8)
    g = (255.0 * yy / max(h - 1, 1)).astype(np.uint8)
    r = np.full((h, w), 96, dtype=np.uint8)
    return np.dstack([b, g, r])


class TestFrameRoundTrip:
    def test_encode_decode_preserves_shape_dtype_and_content(self):
        frame = _synthetic_bgr()
        data = encode_frame(frame)
        assert isinstance(data, bytes) and len(data) > 0
        back = decode_frame(data)
        assert back.shape == frame.shape
        assert back.dtype == np.uint8
        # JPEG is lossy; a smooth gradient should survive within a few counts.
        mean_err = np.mean(np.abs(back.astype(np.int16) - frame.astype(np.int16)))
        assert mean_err < 3.0, "JPEG round-trip drifted %.2f counts" % mean_err

    def test_encode_rejects_non_bgr_input(self):
        with pytest.raises(ValueError):
            encode_frame(np.zeros((10, 10), dtype=np.uint8))        # grayscale
        with pytest.raises(ValueError):
            encode_frame(np.zeros((10, 10, 3), dtype=np.float32))   # wrong dtype

    def test_decode_rejects_garbage_bytes(self):
        with pytest.raises(ValueError):
            decode_frame(b"definitely not a jpeg")


class TestDetectionJsonRoundTrip:
    def test_round_trip_is_exact(self):
        dets = [
            DetectionWire(cls="hospital bed", conf=0.875,
                          xyxy=(1.5, 2.25, 300.0, 240.75)),
            DetectionWire(cls="iv stand", conf=0.25, xyxy=(0.0, 0.0, 10.0, 20.0)),
            DetectionWire(cls="person", conf=1.0, xyxy=(5.0, 6.0, 7.0, 8.0)),
        ]
        # Through an actual JSON string, exactly as the wire carries it.
        wire = json.loads(json.dumps({"detections": detections_to_json(dets)}))
        assert detections_from_json(wire["detections"]) == dets

    def test_empty_list_round_trips(self):
        assert detections_from_json(detections_to_json([])) == []

    def test_json_shape_matches_wire_spec(self):
        [item] = detections_to_json(
            [DetectionWire(cls="chair", conf=0.5, xyxy=(1.0, 2.0, 3.0, 4.0))])
        assert set(item.keys()) == {"cls", "conf", "xyxy"}
        assert item["xyxy"] == [1.0, 2.0, 3.0, 4.0]

    def test_from_json_rejects_malformed_items(self):
        with pytest.raises(ValueError):
            detections_from_json([{"cls": "chair", "conf": 0.5}])   # no xyxy
        with pytest.raises(ValueError):
            detections_from_json([{"cls": "chair", "conf": 0.5,
                                   "xyxy": [1.0, 2.0, 3.0]}])       # 3 elements


class TestVocabularyAndPort:
    def test_vocabulary_has_27_terms(self):
        assert len(DEFAULT_HOSPITAL_VOCABULARY) == 27
        assert len(set(DEFAULT_HOSPITAL_VOCABULARY)) == 27          # no dupes

    def test_sentinel_entries_ported_verbatim(self):
        assert DEFAULT_HOSPITAL_VOCABULARY[0] == "person"
        for sentinel in ("wheelchair", "hospital bed", "iv stand",
                         "x-ray machine", "blood pressure monitor"):
            assert sentinel in DEFAULT_HOSPITAL_VOCABULARY

    def test_default_port(self):
        assert DEFAULT_PORT == 8092
