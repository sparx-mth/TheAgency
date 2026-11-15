"""The segmentation wire contract and service, driven through the runtime's client; no torch.

The server is the real ``ThreadingHTTPServer`` from ``segmentation_server``
bound to a fake :class:`SegmentationModel` that paints each box's interior
minus a hole, so a mask that came back in the wrong order, lossy, or for the
wrong frame would be caught. Runs in the plain ``.venv``.
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from sparx_agency.core.mapping.interfaces.segmentation_model import SegmentationModel
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.mapping.scene_graph.serve.contract import encode_frame
from sparx_agency.tasks.mapping.scene_graph.serve.segmentation_contract import (
    SEGMENTATION_BACKEND, decode_mask, encode_mask, masks_from_json, masks_to_json, parse_segment_request,
    segment_request)
from sparx_agency.tasks.mapping.scene_graph.serve.segmentation_server import _make_server, _SegmentationContext
from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_segmenter import HttpSegmenter


def frame(h=48, w=64):
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (255.0 * xx / (w - 1)).astype(np.uint8)
    rgb[..., 1] = (255.0 * yy / (h - 1)).astype(np.uint8)
    rgb[..., 2] = 90
    return rgb


class PaintingSegmenter(SegmentationModel):
    """The interior of each box with its central pixel left out: a box-specific, non-rectangular mask."""

    def __init__(self):
        self.calls = []

    def segment(self, rgb, boxes_xyxy):
        self.calls.append((rgb.shape, [tuple(b) for b in boxes_xyxy]))
        masks = []
        for x1, y1, x2, y2 in boxes_xyxy:
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[int(y1):int(y2), int(x1):int(x2)] = True
            mask[int((y1 + y2) // 2), int((x1 + x2) // 2)] = False
            masks.append(mask)
        return masks


METADATA = {"backend": SEGMENTATION_BACKEND, "checkpoint_sha256": "abc", "segmenter_config": {"model_path": "mobile_sam.pt"}}


@pytest.fixture
def service():
    fake = PaintingSegmenter()
    ctx = _SegmentationContext(fake, model_name="mobile_sam.pt", device="cpu", metadata=METADATA)
    server = _make_server(ctx, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_address[1], ctx, fake
    finally:
        server.shutdown()
        server.server_close()


# ── the codec ────────────────────────────────────────────────────────────────

def test_a_mask_round_trips_losslessly_through_png():
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:30, 5:40] = True
    mask[20, 20] = False
    assert np.array_equal(decode_mask(encode_mask(mask), (48, 64)), mask)


def test_decoding_checks_the_shape_and_the_bytes():
    mask = np.zeros((48, 64), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        decode_mask(encode_mask(mask), (47, 64))
    with pytest.raises(ValueError):
        decode_mask("not base64!", (48, 64))
    with pytest.raises(ValueError):
        encode_mask(np.zeros((2, 2, 2), dtype=bool))


def test_a_request_round_trips_and_refuses_bad_boxes():
    bgr = frame()[..., ::-1]
    body = segment_request(bgr, [(1, 2, 10, 20)])
    decoded, jpeg, boxes = parse_segment_request(json.loads(json.dumps(body)))
    assert decoded.shape == (48, 64, 3) and jpeg == encode_frame(np.ascontiguousarray(bgr)) and boxes == [[1.0, 2.0, 10.0, 20.0]]
    for bad in ([(0, 0, 70, 10)], [(5, 5, 5, 10)], [(0, 0, 1)], ["x"]):
        with pytest.raises(ValueError):
            parse_segment_request(segment_request(bgr, bad) if bad != ["x"] else dict(body, boxes=["x"]))
    with pytest.raises(ValueError):
        parse_segment_request(dict(body, boxes="no"))
    with pytest.raises(ValueError):
        parse_segment_request({"boxes": []})


def test_mask_items_carry_their_pixel_count_and_are_checked_against_it():
    masks = PaintingSegmenter().segment(frame(), [(0, 0, 10, 10)])
    items = masks_to_json(masks, [[0, 0, 10, 10]])
    assert items[0]["area_px"] == 99 and items[0]["xyxy"] == [0.0, 0.0, 10.0, 10.0]
    assert np.array_equal(masks_from_json(items, (48, 64))[0], masks[0])
    with pytest.raises(ValueError, match="disagrees"):
        masks_from_json([dict(items[0], area_px=100)], (48, 64))
    with pytest.raises(ValueError):
        masks_to_json(masks, [])


# ── the service through the client ──────────────────────────────────────────

def test_the_client_gets_one_mask_per_box_in_box_order_and_pins_the_identity(service):
    url, ctx, fake = service
    client = HttpSegmenter(url)
    identity = client.health()
    assert identity["model"] == "mobile_sam.pt" and identity["metadata"]["backend"] == SEGMENTATION_BACKEND
    rgb = frame()
    masks = client.segment(rgb, [(40, 5, 60, 45), (10, 10, 20, 30)])
    assert len(masks) == 2 and all(m.shape == (48, 64) and m.dtype == bool for m in masks)
    assert masks[0].sum() == 20 * 40 - 1 and masks[0][5:45, 40:60].sum() == 20 * 40 - 1
    assert masks[1].sum() == 10 * 20 - 1 and masks[1][10:30, 10:20].sum() == 10 * 20 - 1
    assert fake.calls[0][1] == [(40.0, 5.0, 60.0, 45.0), (10.0, 10.0, 20.0, 30.0)]
    assert ctx.frames_served == 1 and client.last_inference_ms is not None
    assert client.segment(rgb, []) == [] and ctx.frames_served == 1          # no boxes, no request


def test_the_client_refuses_a_service_that_is_not_the_sam_service(service):
    url, ctx, _ = service
    ctx.metadata = dict(METADATA, backend="yolo_world")
    with pytest.raises(ObjNavInternalError, match="not the box-prompted SAM service"):
        HttpSegmenter(url).health()
    ctx.metadata = {"backend": SEGMENTATION_BACKEND}
    with pytest.raises(ObjNavInternalError, match="provenance"):
        HttpSegmenter(url).health()
    ctx.metadata = dict(METADATA)
    with pytest.raises(ObjNavInternalError, match="not the requested"):
        HttpSegmenter(url, expected_model="sam2.1_t.pt").health()


def test_identity_drift_mid_run_ends_the_run_without_a_transport_cause(service):
    url, ctx, _ = service
    client = HttpSegmenter(url)
    client.segment(frame(), [(0, 0, 10, 10)])
    ctx.metadata = dict(METADATA, checkpoint_sha256="other")
    with pytest.raises(ObjNavInternalError, match="reconfigured") as info:
        client.segment(frame(), [(0, 0, 10, 10)])
    assert info.value.__cause__ is None
    with pytest.raises(ObjNavInternalError, match="changed during"):
        client.health()


def test_a_bad_box_or_a_dead_service_is_a_frame_failure_with_a_cause(service):
    url, _, _ = service
    client = HttpSegmenter(url)
    with pytest.raises(ObjNavInternalError, match="Segmenter service failed") as info:
        client.segment(frame(), [(0, 0, 100, 10)])                       # outside the 64-wide frame: HTTP 400
    assert info.value.__cause__ is not None
    dead = HttpSegmenter("http://127.0.0.1:9", timeout_s=0.5)
    dead._identity = {"model": "mobile_sam.pt", "device": "cpu", "metadata": METADATA}
    with pytest.raises(ObjNavInternalError, match="Segmenter service failed") as info:
        dead.segment(frame(), [(0, 0, 10, 10)])
    assert info.value.__cause__ is not None


def test_unknown_routes_and_non_json_bodies_are_refused(service):
    url, _, _ = service
    import requests
    assert requests.get(url + "/detect", timeout=5).status_code == 404
    assert requests.post(url + "/detect", data=b"x", timeout=5).status_code == 404
    reply = requests.post(url + "/segment", data=b"not json", timeout=5)
    assert reply.status_code == 400 and "not JSON" in reply.json()["error"]


def test_the_timeout_must_be_positive_and_finite():
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            HttpSegmenter("http://127.0.0.1:1", timeout_s=value)
