"""Numpy-only tests for the YOLO-World preprocess + postprocess (no torch/TRT).

These exercise the parts of the pipeline that run on the dev laptop: the letterbox
transform round-trip, NMS, and decoding a synthetic raw head tensor back into
original-frame :class:`Detection2D`. They pin the maths so a refactor of the CNN
build path cannot silently break the geometry.
"""
import numpy as np
import pytest

from sparx_agency.tasks.mapping.yolo_world_trt import postprocess, preprocess
from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import parse_imgsz


def test_parse_imgsz_forms_and_validation():
    assert parse_imgsz("288x512") == (288, 512)
    assert parse_imgsz(640) == (640, 640)
    assert parse_imgsz([256, 448]) == (256, 448)
    with pytest.raises(ValueError):
        parse_imgsz("300x512")          # 300 not a multiple of 32


def test_letterbox_shape_and_center_roundtrip():
    rgb = np.zeros((294, 504, 3), np.uint8)
    padded, tr = preprocess.letterbox(rgb, (288, 512))
    assert padded.shape == (288, 512, 3)
    # A box covering the whole original frame maps to ~the whole frame back.
    full = np.array([[0, 0, 504, 294]], np.float32)
    # forward map into engine space, then undo -> original (within a pixel).
    fwd = full.copy()
    fwd[:, [0, 2]] = full[:, [0, 2]] * tr.scale + tr.pad_x
    fwd[:, [1, 3]] = full[:, [1, 3]] * tr.scale + tr.pad_y
    back = tr.undo_xyxy(fwd)
    assert np.allclose(back[0], [0, 0, 503, 293], atol=1.5)


def test_to_engine_tensor_layout_and_range():
    padded = np.full((288, 512, 3), 255, np.uint8)
    t = preprocess.to_engine_tensor(padded)
    assert t.shape == (1, 3, 288, 512)
    assert t.dtype == np.float32
    assert np.isclose(t.max(), 1.0) and t.min() >= 0.0


def test_nms_suppresses_overlap_keeps_distinct():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 120, 120]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = postprocess.nms(boxes, scores, iou_thresh=0.5)
    assert keep[0] == 0                # highest kept
    assert 1 not in keep               # heavy overlap suppressed
    assert 2 in keep                   # disjoint box kept


def _raw_head(boxes_xywh, class_scores):
    """Build a synthetic ``[1, 4+nc, A]`` head tensor from boxes + score rows."""
    nc = class_scores.shape[1]
    a = boxes_xywh.shape[0]
    arr = np.zeros((1, 4 + nc, a), np.float32)
    arr[0, :4, :] = boxes_xywh.T
    arr[0, 4:, :] = class_scores.T
    return arr


def test_decode_maps_back_to_original_and_labels():
    labels = ["refrigerator", "chair"]
    rgb = np.zeros((294, 504, 3), np.uint8)
    _padded, tr = preprocess.letterbox(rgb, (288, 512))
    # One strong 'chair' box centred in the engine frame.
    cx, cy, w, h = 256.0, 144.0, 40.0, 60.0
    boxes = np.array([[cx, cy, w, h]], np.float32)
    scores = np.array([[0.1, 0.92]], np.float32)     # class 1 = chair
    raw = _raw_head(boxes, scores)

    dets = postprocess.decode(raw, labels, tr, conf_thresh=0.25, iou_thresh=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "chair"
    assert d.score == pytest.approx(0.92, abs=1e-4)
    assert d.frame_w == 504 and d.frame_h == 294
    x1, y1, x2, y2 = d.bbox_xyxy
    assert 0 <= x1 < x2 <= 504 and 0 <= y1 < y2 <= 294


def test_decode_accepts_transposed_output():
    labels = ["a", "b"]
    _p, tr = preprocess.letterbox(np.zeros((294, 504, 3), np.uint8), (288, 512))
    boxes = np.array([[256.0, 144.0, 20.0, 20.0]], np.float32)
    scores = np.array([[0.8, 0.1]], np.float32)
    raw = _raw_head(boxes, scores)[0].T[None]        # [1, A, 4+nc]
    dets = postprocess.decode(raw, labels, tr, conf_thresh=0.25)
    assert len(dets) == 1 and dets[0].label == "a"


def test_decode_below_threshold_is_empty():
    labels = ["a"]
    _p, tr = preprocess.letterbox(np.zeros((294, 504, 3), np.uint8), (288, 512))
    raw = _raw_head(np.array([[10, 10, 5, 5]], np.float32),
                    np.array([[0.05]], np.float32))
    assert postprocess.decode(raw, labels, tr, conf_thresh=0.25) == []
