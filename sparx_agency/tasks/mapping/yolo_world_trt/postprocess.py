"""Decode the raw YOLOv8 / YOLO-World head output into detections (torch-free).

The engine is exported NMS-free, so its single output is the raw head tensor
``[1, 4 + nc, num_anchors]``: the first 4 rows are the box centre/size
``(cx, cy, w, h)`` already decoded to *pixels in the letterboxed engine input*,
and the remaining ``nc`` rows are per-class confidences (post-sigmoid). This
module thresholds, runs class-wise Non-Maximum Suppression, maps boxes back to the
original frame via a :class:`~...preprocess.LetterboxTransform`, and returns core
:class:`Detection2D` objects.

Kept pure numpy (no torch, no tensorrt) so it is unit-testable on any box and so
the runtime does not drag torch onto the Noetic / Python-3.8 side.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.tasks.mapping.yolo_world_trt.preprocess import LetterboxTransform


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """``[N,4]`` (cx,cy,w,h) -> (x1,y1,x2,y2)."""
    out = np.empty_like(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Greedy single-class NMS. Returns kept indices, highest score first.

    Args:
        boxes: ``[N,4]`` xyxy.
        scores: ``[N]`` confidences.
        iou_thresh: suppress a box overlapping a kept box by more than this.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return keep


def decode(raw: np.ndarray, labels: Sequence[str], transform: LetterboxTransform,
           conf_thresh: float = 0.25, iou_thresh: float = 0.5,
           max_det: int = 100) -> List[Detection2D]:
    """Decode one raw head tensor into original-frame :class:`Detection2D` list.

    Args:
        raw: ``[1, 4+nc, A]`` or ``[4+nc, A]`` (or its transpose) head output.
        labels: baked class prompts, indexed by the head's class rows.
        transform: the letterbox transform used to build the engine input.
        conf_thresh: minimum class confidence to keep a box.
        iou_thresh: NMS IoU threshold (applied per class).
        max_det: cap on returned detections (highest score first).

    Returns:
        Detections in ORIGINAL-frame pixel coordinates.
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    nc = len(labels)
    # Orient to [4+nc, A]: the anchor axis (~thousands) is the long one.
    if arr.shape[0] != 4 + nc and arr.shape[1] == 4 + nc:
        arr = arr.T
    if arr.shape[0] != 4 + nc:
        raise ValueError("raw head has %s rows, expected %d (4 + nc=%d)"
                         % (arr.shape[0], 4 + nc, nc))

    boxes_in = _xywh_to_xyxy(arr[:4].T)          # [A,4] in engine-input pixels
    cls_scores = arr[4:].T                        # [A, nc]
    cls_id = cls_scores.argmax(axis=1)
    conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

    keep_mask = conf >= conf_thresh
    if not np.any(keep_mask):
        return []
    boxes_in, conf, cls_id = boxes_in[keep_mask], conf[keep_mask], cls_id[keep_mask]

    boxes_orig = transform.undo_xyxy(boxes_in)

    kept: List[int] = []
    for c in np.unique(cls_id):
        idx = np.where(cls_id == c)[0]
        local = nms(boxes_orig[idx], conf[idx], iou_thresh)
        kept.extend(int(idx[j]) for j in local)

    kept.sort(key=lambda i: conf[i], reverse=True)
    kept = kept[:max_det]

    dets: List[Detection2D] = []
    for i in kept:
        x1, y1, x2, y2 = boxes_orig[i]
        c = int(cls_id[i])
        label = str(labels[c]).strip().lower() if c < len(labels) else str(c)
        dets.append(Detection2D(
            label=label,
            score=float(conf[i]),
            bbox_xyxy=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            frame_w=transform.orig_w,
            frame_h=transform.orig_h,
        ))
    return dets
