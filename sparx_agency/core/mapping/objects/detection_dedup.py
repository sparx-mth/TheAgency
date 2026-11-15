"""Alias-aware deduplication of one frame's detections, before any association.

An open-vocabulary detector fires several prompts on one object: ``couch``
and ``sofa`` on the same cushion, two overlapping ``chair`` boxes on one
seat. Folding them here -- the alias table collapses the prompt to its
canonical class, a box that overlaps a kept box of the same class by half
or any kept box almost entirely is dropped -- keeps repeated prompts from
manufacturing confirmation downstream. Exact aliases only: this is not fuzzy
category matching, and distinct classes never become one goal.

Works on any detection record with ``cls``, ``conf`` and ``xyxy`` fields
that :func:`dataclasses.replace` accepts
(:class:`~sparx_agency.core.planning.object_search.interfaces.Detection`,
the detection server's wire record).
"""
from __future__ import annotations

from dataclasses import replace

from sparx_agency.core.common.math.bbox import iou

#: Detector prompt -> the canonical class the method reasons in.
ALIASES = {"couch": "sofa", "tv": "television", "monitor": "television", "armchair": "chair"}


def deduplicate_detections(detections):
    """Suppress duplicate prompts without broad fuzzy category matching."""
    kept = []
    for detection in sorted(detections, key=lambda d: -d.conf):
        canonical = ALIASES.get(detection.cls, detection.cls)
        candidate = replace(detection, cls=canonical)
        if any((old.cls == canonical and iou(old.xyxy, candidate.xyxy) >= 0.5)
               or iou(old.xyxy, candidate.xyxy) >= 0.9 for old in kept):
            continue
        kept.append(candidate)
    return kept
