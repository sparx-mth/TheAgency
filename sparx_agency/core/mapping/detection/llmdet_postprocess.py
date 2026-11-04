"""Numpy-only category decoding for LLMDet's grounded token logits.

Use the mean of sigmoid probabilities over each category's positive token map,
as in MMDetection's ``convert_grounding_to_cls_scores``. Do NOT decode arbitrary
above-threshold token fragments into new labels ("plant" != "potted plant").
This is detector postprocessing, not a second detector or learned score model.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D


def validate_prompts(prompts: Sequence[str]) -> List[str]:
    """Preserve exact wire labels, rejecting ambiguous/empty category lists."""
    if isinstance(prompts, str) or not prompts:
        raise ValueError("Provide a non-empty sequence of category names")
    cleaned = []
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Every category must be a non-empty string")
        label = prompt.strip()
        if "." in label:
            raise ValueError("Category names cannot contain the caption separator '.'")
        cleaned.append(label)
    if len({p.lower() for p in cleaned}) != len(cleaned):
        raise ValueError("Duplicate category names after case normalization")
    return cleaned


def caption_and_spans(prompts: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Build the dot-delimited BERT caption and exact category character spans."""
    caption, spans = "", []
    for label in prompts:
        text = label.lower()
        start = len(caption)
        caption += text
        spans.append((start, len(caption)))
        caption += " . "
    return caption, spans


def tokens_for_spans(offsets, spans) -> List[List[int]]:
    """Map complete category spans to tokens; refuse lost/truncated categories."""
    groups = []
    for start, end in spans:
        indices = [i for i, (left, right) in enumerate(offsets)
                   if right > left and left < end and right > start]
        if not indices or offsets[indices[-1]][1] < end:
            raise ValueError("Category was lost or truncated by the tokenizer")
        groups.append(indices)
    return groups


def decode_token_detections(probabilities, boxes_cxcywh, token_groups, prompts,
                            frame_hw, threshold, max_det) -> List[Detection2D]:
    """Decode normalized model boxes to clipped, original-frame pixel boxes.

    Keep top scoring (query, category) pairs, matching MM-GDINO's category-mode
    top-k protocol. NMS is NOT added to this DETR backend; the existing downstream
    alias-aware suppression and distinct-frame landmark evidence remain in charge.
    """
    probs = np.asarray(probabilities)
    boxes = np.asarray(boxes_cxcywh)
    if (probs.ndim != 2 or boxes.shape != (len(probs), 4)
            or len(token_groups) != len(prompts) or not prompts):
        raise ValueError("Malformed LLMDet output or category token map")
    if (not np.isfinite(probs).all() or not np.isfinite(boxes).all()
            or np.any(probs < 0) or np.any(probs > 1)
            or np.any(boxes[:, 2:] < 0)):
        raise ValueError("Non-finite/invalid LLMDet probabilities or boxes")
    scores = []
    for indices in token_groups:
        if not indices or min(indices) < 0 or max(indices) >= probs.shape[1]:
            raise ValueError("Category token map exceeds the model output")
        scores.append(probs[:, indices].mean(axis=1))
    scores = np.stack(scores, axis=1)
    # Stable ordering makes tied scores and repeated runs reproducible.
    order = np.argsort(-scores.reshape(-1), kind="stable")
    h, w = frame_hw
    result = []
    for flat_index in order:
        query, category = divmod(int(flat_index), len(prompts))
        score = float(scores[query, category])
        if score < threshold or len(result) >= max_det:
            break
        cx, cy, bw, bh = boxes[query]
        xyxy = np.array([cx - bw / 2, cy - bh / 2,
                         cx + bw / 2, cy + bh / 2]) * [w, h, w, h]
        xyxy = np.clip(xyxy, 0, [w, h, w, h]).astype(int)
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        if x1 >= x2 or y1 >= y2:
            continue
        result.append(Detection2D(
            label=prompts[category], score=score,
            bbox_xyxy=(x1, y1, x2, y2), frame_w=w, frame_h=h))
    return result
