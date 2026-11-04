"""Coherent RGB-D projection of detector boxes to world-frame object centroids.

The one place a detection becomes a position. Each box above the confidence
floor is measured on the depth it covers -- the whole box
(:func:`~sparx_agency.core.mapping.objects.geometry.robust_bbox_depth`) or,
with the ``segmentation`` setting, the object's own pixels
(:func:`~sparx_agency.core.mapping.objects.geometry.robust_mask_depth`) --
and only a box whose central patch is ONE connected surface at that depth
yields a centroid: a percentile depth attached to the centre ray is not a
position. The camera's actual pitch is folded in through
:func:`~sparx_agency.core.planning.objnav.camera_geometry.world_T_camera_optical`.

Every box gets a diagnostics row (``status``: ``low_confidence``,
``invalid_box``, ``segmentation_unavailable``, ``insufficient_depth``,
``empty_mask``, ``mixed_depth`` or ``projected``) so the recording can say
why a frame did or did not place an object.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage

from sparx_agency.core.mapping.objects.box_extent import footprint_radius_m
from sparx_agency.core.mapping.objects.box_occluder import depth_band_m, occluder_in_front
from sparx_agency.core.mapping.objects.geometry import robust_bbox_depth, robust_mask_depth
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical

#: Detector labels that name a staircase: terrain context for the building coordinator, never an object.
STAIR_LABELS = frozenset(("stairs", "staircase"))


def observed_objects(observation, detections, confidence, diagnostics=None, masks=None):
    """Yield coherent centroids, not percentile depths attached to centre rays.

    Args:
        observation: The frame.
        detections: The boxes to project.
        confidence: Boxes below it are recorded as ``low_confidence`` and skipped.
        diagnostics: Optional list that receives one row per detection.
        masks: Optional mapping ``detection -> HxW boolean mask`` from the
            segmenter (the ``segmentation`` setting). A detection WITH a mask
            is measured on the object's own pixels
            (:func:`~sparx_agency.core.mapping.objects.geometry.robust_mask_depth`);
            one mapped to ``None`` was due a mask the service could not
            deliver this frame and is NOT measured (``segmentation_unavailable``)
            -- with segmentation on, no box of the target is ever read the old
            way, so the comparison between the two pipelines stays clean; one
            absent from the mapping (another class) is measured on its box as
            always. ``None`` (the default) is the box pipeline throughout.
    """
    camera = observation.camera
    k = camera.intrinsics
    transform = world_T_camera_optical(observation.pose, camera)
    for detection in detections:
        row = {"label": detection.cls, "confidence": float(detection.conf), "xyxy": list(detection.xyxy),
               "step": observation.step, "status": "low_confidence"}
        if diagnostics is not None:
            diagnostics.append(row)
        if detection.conf < confidence:
            continue
        x1, y1, x2, y2 = detection.xyxy
        row["status"] = "invalid_box"
        if not (0 <= x1 < x2 <= k.width and 0 <= y1 < y2 <= k.height):
            continue
        mask = None
        if masks is not None and detection in masks:
            mask = masks[detection]
            row["segmented"] = mask is not None
            if mask is None:
                row["status"] = "segmentation_unavailable"
                continue
        xyz = _supported_centroid(observation, detection.xyxy, transform, row, mask)
        if xyz is not None:
            yield detection.cls, xyz


def _supported_centroid(observation, box, transform, row, mask=None):
    """Measure connected support before downsampling; a stride can alias holes.

    Without a mask the depth is the box's (``robust_bbox_depth``: a low
    percentile of the shrunken box) and the support is sought in the box's
    central quarter. With a mask the depth is the object's own
    (``robust_mask_depth``) and the support is sought over the whole box, on
    the mask's pixels only: the occluder that stands inside the box -- and
    that the box read takes for the near surface of the object -- is not on
    the mask, so neither the centroid nor the near range can land on it.

    **The box read has its own answer to an occluder it can tell from the
    depth** (since 2026-10-15, ``box_occluder.occluder_in_front``): when a
    second surface spans the whole box behind a near one that covers less of
    it and leaves its top band alone -- a seat back's rail at camera height
    in front of a television -- the near surface's pixels are set aside and
    the box is read again without them; ``occluder_depth_m`` records the
    range that was set aside. A near surface that fills the box, reaches its
    top, or has nothing behind it is the object, as before.
    """
    camera, k = observation.camera, observation.camera.intrinsics
    row["status"] = "insufficient_depth"
    x1, y1, x2, y2 = box
    excluded = None
    if mask is None:
        depth = robust_bbox_depth(observation.depth_m, box, min_depth_m=camera.min_depth_m, max_depth_m=camera.max_depth_m)
        if depth is not None:
            excluded = occluder_in_front(observation.depth_m, box, depth, camera.min_depth_m, camera.max_depth_m)
            if excluded is not None:
                row.update(occluder_depth_m=float(depth), occluder_px=int(excluded.sum()))
                depth = robust_bbox_depth(np.where(excluded, np.nan, observation.depth_m), box,
                                          min_depth_m=camera.min_depth_m, max_depth_m=camera.max_depth_m)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left, right = int(cx - (x2 - x1) / 4), int(cx + (x2 - x1) / 4)
        top, bottom = int(cy - (y2 - y1) / 4), int(cy + (y2 - y1) / 4)
        selected = None
    else:
        left, right, top, bottom = int(x1), int(x2), int(y1), int(y2)
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != observation.depth_m.shape:
            raise ValueError("mask shape %s does not match the depth %s" % (selected.shape, observation.depth_m.shape))
        row["mask_pixels"] = int(np.count_nonzero(selected[top:bottom, left:right]))
        row["mask_fraction"] = row["mask_pixels"] / max(1, (right - left) * (bottom - top))
        if row["mask_pixels"] == 0:
            row["status"] = "empty_mask"
            return None
        depth = robust_mask_depth(observation.depth_m, selected, box, min_depth_m=camera.min_depth_m, max_depth_m=camera.max_depth_m)
    if depth is None:
        return None
    stride = max(1, int(math.sqrt((right - left) * (bottom - top) / 900)))
    patch = observation.depth_m[top:bottom, left:right]
    valid = np.isfinite(patch) & (patch > camera.min_depth_m) & (patch < camera.max_depth_m)
    if selected is not None:
        valid &= selected[top:bottom, left:right]
    if excluded is not None:
        valid &= ~excluded[top:bottom, left:right]
    supported = valid & (np.abs(patch - depth) <= depth_band_m(depth))
    components, count = ndimage.label(supported)
    if not count:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component = components == int(np.argmax(sizes))
    fraction = float(component.sum()) / max(1, int(valid.sum()))
    row.update(supported_fraction=fraction, valid_pixels=int(valid.sum()), status="mixed_depth")
    # The coherence gate (at least 35 % of the patch within the band) asks the box's
    # central quarter whether it is ONE surface. The mask has answered that: its pixels
    # are the object's, and a deep object seen whole -- a bed from its foot, a sofa at an
    # angle -- spans more than the band, so on the mask path the fraction is recorded
    # and not gated; the near-surface component still carries the centroid and range.
    if component.sum() < 8 or (selected is None and fraction < 0.35):
        return None
    v, u = np.nonzero(component[::stride, ::stride])
    if len(v) < 8:
        return None
    d = patch[v * stride, u * stride]
    optical = np.stack(((left + u * stride - k.cx) * d / k.fx, (top + v * stride - k.cy) * d / k.fy, d), axis=1)
    points = optical @ transform[:3, :3].T + transform[:3, 3]
    xyz = np.median(points, axis=0)
    # The range to the object's NEAR surface, not to its middle: a 10th-percentile
    # horizontal distance of the supported points, robust to a few stray pixels.
    # The benchmark's success radius is measured to the object, so the terminal
    # test of the target closing reads this beside the centroid.
    pose = observation.pose
    near = float(np.percentile(np.hypot(points[:, 0] - pose.x, points[:, 1] - pose.y), 10))
    row.update(status="projected", xyz=xyz.tolist(), height_range_m=[float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
               radius_m=footprint_radius_m(box, float(depth), k.fx), range_near_m=near)
    return xyz


