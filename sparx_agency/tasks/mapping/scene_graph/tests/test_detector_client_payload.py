"""Tests for the detector client's reply handling.

``ros2/detection_payload.py`` is the seam between the conda/GPU detection
server and the ROS2 client node, and it carries one easily-lost invariant: the
``/perception/detections`` stamp is the **source RGB header** stamp, never the
server's own. Downstream the object mapper joins detections with depth and
pose by that value, so a stamp taken from the reply would back-project every
box against the wrong pose — silently, with plausible-looking output.

The rest is loud failure (a half-parsed detection frame must raise rather than
reach the mapper) and the debug overlay's shape/purity. All rclpy-free, so it
runs in the plain ``.venv``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.tasks.mapping.scene_graph.ros2.detection_payload import (
    build_detections_payload,
    draw_detections,
)

SOURCE_STAMP = 1234.5
BOX_BGR = (0, 200, 0)


def server_reply(detections=None, **overrides):
    """A well-formed ``POST /detect`` reply body, as ``json.loads`` returns it.

    The ``stamp`` field is deliberately present and deliberately wrong: the
    server timestamps its own reply, and that value must never be published.
    """
    reply = {"stamp": 999.0, "w": 640, "h": 480, "ms": 12.5,
             "detections": [{"cls": "hospital bed", "conf": 0.87,
                             "xyxy": [10, 20, 300, 400]}]
             if detections is None else detections}
    reply.update(overrides)
    return reply


class TestStampProvenance:
    """The invariant that makes the whole depth/pose join correct."""

    def test_payload_carries_the_source_image_stamp(self):
        payload = build_detections_payload(SOURCE_STAMP, server_reply())
        assert payload["stamp"] == SOURCE_STAMP

    def test_the_servers_own_stamp_never_leaks_through(self):
        reply = server_reply(stamp=999.0)
        payload = build_detections_payload(SOURCE_STAMP, reply)
        assert payload["stamp"] != reply["stamp"]

    def test_a_reply_without_any_stamp_is_still_fine(self):
        reply = server_reply()
        reply.pop("stamp")
        assert build_detections_payload(SOURCE_STAMP, reply)["stamp"] \
            == SOURCE_STAMP

    def test_a_numpy_stamp_is_coerced_so_json_dumps_cannot_raise(self):
        payload = build_detections_payload(np.float32(2.5), server_reply())
        assert type(payload["stamp"]) is float
        json.dumps(payload)


class TestPayloadContract:
    """``{"stamp", "w", "h", "ms", "detections": [{cls, conf, xyxy}]}``."""

    def test_exact_top_level_keys(self):
        payload = build_detections_payload(SOURCE_STAMP, server_reply())
        assert set(payload.keys()) == {"stamp", "w", "h", "ms", "detections"}

    def test_geometry_and_timing_pass_through(self):
        payload = build_detections_payload(
            SOURCE_STAMP, server_reply(w=1280, h=720, ms=33.75))
        assert payload["w"] == 1280 and type(payload["w"]) is int
        assert payload["h"] == 720 and type(payload["h"]) is int
        assert payload["ms"] == 33.75 and type(payload["ms"]) is float

    def test_detections_are_canonicalized(self):
        """Integer pixel boxes arrive as ints and leave as a 4-float list."""
        payload = build_detections_payload(SOURCE_STAMP, server_reply())
        [det] = payload["detections"]
        assert set(det.keys()) == {"cls", "conf", "xyxy"}
        assert det["cls"] == "hospital bed"
        assert det["conf"] == pytest.approx(0.87)
        assert det["xyxy"] == [10.0, 20.0, 300.0, 400.0]
        assert all(type(v) is float for v in det["xyxy"])

    def test_an_empty_detection_frame_is_a_valid_payload(self):
        payload = build_detections_payload(SOURCE_STAMP,
                                           server_reply(detections=[]))
        assert payload["detections"] == []

    def test_extra_server_fields_are_dropped(self):
        payload = build_detections_payload(
            SOURCE_STAMP, server_reply(model="yolov8s-worldv2", device="cuda:0"))
        assert "model" not in payload and "device" not in payload

    def test_the_payload_round_trips_through_json(self):
        payload = build_detections_payload(
            SOURCE_STAMP,
            server_reply(detections=[
                {"cls": "chair", "conf": 0.5, "xyxy": [1.5, 2.5, 3.5, 4.5]},
                {"cls": "iv stand", "conf": 0.25, "xyxy": [0, 0, 10, 20]}]))
        assert json.loads(json.dumps(payload)) == payload


class TestMalformedRepliesRaise:
    """Loud failure: junk must never be published as a detection frame."""

    @pytest.mark.parametrize("reply", [
        None,
        [],
        "not json object",
        42,
    ], ids=["none", "list", "string", "number"])
    def test_a_reply_that_is_not_an_object_raises(self, reply):
        with pytest.raises(ValueError):
            build_detections_payload(SOURCE_STAMP, reply)

    @pytest.mark.parametrize("missing", ["w", "h", "ms", "detections"])
    def test_a_missing_field_raises(self, missing):
        reply = server_reply()
        reply.pop(missing)
        with pytest.raises(ValueError):
            build_detections_payload(SOURCE_STAMP, reply)

    @pytest.mark.parametrize("overrides", [
        {"w": "wide"},
        {"h": None},
        {"ms": "fast"},
    ], ids=["w-not-a-number", "h-none", "ms-not-a-number"])
    def test_an_unparseable_field_raises(self, overrides):
        with pytest.raises(ValueError):
            build_detections_payload(SOURCE_STAMP, server_reply(**overrides))

    def test_a_detections_field_that_is_not_a_list_raises(self):
        with pytest.raises(ValueError):
            build_detections_payload(SOURCE_STAMP,
                                     server_reply(detections={"cls": "chair"}))

    @pytest.mark.parametrize("item", [
        {"cls": "chair", "conf": 0.5},                       # no box
        {"cls": "chair", "xyxy": [1, 2, 3, 4]},              # no confidence
        {"conf": 0.5, "xyxy": [1, 2, 3, 4]},                 # no class
        {"cls": "chair", "conf": 0.5, "xyxy": [1, 2, 3]},    # short box
        {"cls": "chair", "conf": 0.5, "xyxy": [1, 2, 3, 4, 5]},   # long box
        {"cls": "chair", "conf": 0.5, "xyxy": "1,2,3,4"},    # box not numbers
        {"cls": "chair", "conf": "high", "xyxy": [1, 2, 3, 4]},
    ], ids=["no-box", "no-conf", "no-cls", "short-box", "long-box",
            "box-string", "conf-string"])
    def test_a_malformed_detection_entry_raises(self, item):
        with pytest.raises(ValueError):
            build_detections_payload(SOURCE_STAMP,
                                     server_reply(detections=[item]))

    def test_one_bad_entry_poisons_the_whole_frame(self):
        """No partial publish: a good box beside a bad one still raises."""
        with pytest.raises(ValueError):
            build_detections_payload(SOURCE_STAMP, server_reply(detections=[
                {"cls": "chair", "conf": 0.9, "xyxy": [1, 2, 3, 4]},
                {"cls": "chair", "conf": 0.9}]))


class TestDebugOverlay:
    """The overlay is a debug view; it must not alter the frame it is given."""

    @staticmethod
    def frame(h=48, w=64):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def payload_detections(self):
        return build_detections_payload(SOURCE_STAMP, server_reply(detections=[
            {"cls": "chair", "conf": 0.9, "xyxy": [5, 6, 40, 30]}]))[
                "detections"]

    def test_overlay_has_the_input_shape_and_dtype(self):
        img = self.frame()
        out = draw_detections(img, self.payload_detections())
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_overlay_does_not_modify_the_input_frame(self):
        img = self.frame()
        before = img.copy()
        draw_detections(img, self.payload_detections())
        assert np.array_equal(img, before)

    def test_boxes_are_actually_drawn(self):
        out = draw_detections(self.frame(), self.payload_detections())
        painted = int(np.sum(np.all(out == np.array(BOX_BGR, dtype=np.uint8),
                                    axis=2)))
        assert painted > 0, "no box pixels on the overlay"

    def test_no_detections_returns_an_unmarked_copy(self):
        img = self.frame()
        out = draw_detections(img, [])
        assert np.array_equal(out, img)
        assert out is not img

    def test_a_non_contiguous_frame_is_accepted(self):
        """The node hands over slices/flips of the decoded image."""
        img = self.frame()[:, ::-1]
        assert not img.flags["C_CONTIGUOUS"]
        out = draw_detections(img, self.payload_detections())
        assert out.shape == img.shape

    def test_boxes_outside_the_frame_do_not_raise(self):
        out = draw_detections(self.frame(), [
            {"cls": "chair", "conf": 0.9, "xyxy": [-500, -500, 5000, 5000]},
            {"cls": "person", "conf": 0.1, "xyxy": [-20, -20, -5, -5]}])
        assert out.shape == (48, 64, 3)

    def test_a_full_size_frame_keeps_its_shape(self):
        img = self.frame(h=480, w=640)
        out = draw_detections(img, self.payload_detections())
        assert out.shape == (480, 640, 3)
