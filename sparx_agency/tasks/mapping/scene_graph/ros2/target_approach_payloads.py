"""Wire seam for the target approach: parse in, measure, report out.

Everything ``target_approach_node`` does that can be decided without a clock,
a camera or an rclpy context lives here, for the same reason
:mod:`...ros2.room_search_payloads` exists beside ``room_search_node``: none
of it can be verified by watching the aircraft. A ``/target_seen/info``
payload whose class key was renamed produces a node that locks onto the empty
string and simply never confirms; a bbox handed to the depth camera without
the intrinsics rescale produces a range that is merely *wrong*, so the drone
lands early or never; a status payload with a renamed key produces a blank
operator panel. Each of those is asserted in ``tests/test_target_approach.py``.

Three jobs, in the order the node uses them:

1. :func:`target_info_from_json` — which class to lock onto, out of the
   latched ``/target_seen/info`` JSON the target watcher publishes.
2. :func:`detections_to_core` / :func:`bbox_range_m` — the detection server's
   wire boxes into :class:`Detection2D` in the node's own pixel space, and the
   tracked box into a metric range through the *depth* camera's intrinsics.
3. :func:`approach_info_payload` — the latched ``/target_approach/info``
   status a dashboard reads.

**The RGB and depth cameras are not the same camera.** Both front sensors on
this aircraft render 600x600, so a box copied straight across looks plausible
and is silently wrong -- the RGB sensor has the wider FOV, so its pixels
subtend a different angle. :func:`bbox_range_m` therefore goes through
:func:`~sparx_agency.core.mapping.objects.geometry.rescale_bbox_between_intrinsics`,
exactly as ``object_mapper_node`` does for the same pair of cameras.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.math.bbox import rescale_xyxy
from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.objects.geometry import (
    PinholeK, rescale_bbox_between_intrinsics, robust_bbox_depth)
from sparx_agency.tasks.mapping.scene_graph.serve.contract import (
    detections_from_json)


@dataclass(frozen=True)
class TargetInfo:
    """What the latched ``/target_seen/info`` payload says was found.

    Attributes:
        target: The mission's target text as configured on the watcher
            (``target_object``), e.g. ``"wheelchair"``.
        matched_class: The detector *vocabulary class* that actually matched
            it, e.g. ``"hospital bed"`` for a target of ``"bed"``. Possibly
            empty when the watcher published no class.
        lock_class: The class the approach must lock onto: ``matched_class``
            when the watcher named one, else ``target``. This is the one that
            matters -- :class:`...visual_servo.TargetConfirmationGate` matches
            against detector labels, and the LLM matcher may well have mapped
            a target word onto a vocabulary prompt that does not contain it.
        object_id: The landmark id that matched, or ``-1``.
        xy: The landmark's world ENU position (metres). Carried for the
            operator status only -- the approach is purely visual and never
            flies to a coordinate.
        count: How many observations confirmed the landmark.
    """

    target: str
    matched_class: str
    lock_class: str
    object_id: int
    xy: Tuple[float, float]
    count: int


def target_info_from_json(text: str) -> TargetInfo:
    """Parse ``/target_seen/info`` into the class the approach locks onto.

    Args:
        text: The raw ``std_msgs/String`` data published (latched) by
            ``target_watcher_node`` on the first match.

    Returns:
        The parsed :class:`TargetInfo`, with ``lock_class`` already resolved.

    Raises:
        ValueError: If the payload is not a JSON object, or names neither a
            ``matched_class`` nor a ``target`` -- there is then no class to
            lock onto, and a node that shrugged this off would take the
            aircraft and hunt for nothing. The caller must refuse to engage.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_seen/info is not JSON: %s" % (exc,))
    if not isinstance(data, dict):
        raise ValueError("target_seen/info must be a JSON object, got %s"
                         % (type(data).__name__,))

    target = str(data.get("target", "") or "").strip()
    matched = str(data.get("matched_class", "") or "").strip()
    lock = matched or target
    if not lock:
        raise ValueError(
            "target_seen/info names neither matched_class nor target: %r"
            % (text,))

    xy_raw = data.get("xy", [0.0, 0.0])
    try:
        xy = (float(xy_raw[0]), float(xy_raw[1]))
    except (IndexError, KeyError, TypeError, ValueError):
        # A missing/odd position is not fatal: the approach is visual and
        # never flies to this coordinate, it only reports it.
        xy = (0.0, 0.0)
    try:
        object_id = int(data.get("object_id", -1))
    except (TypeError, ValueError):
        object_id = -1
    try:
        count = int(data.get("count", 0))
    except (TypeError, ValueError):
        count = 0

    return TargetInfo(target=target, matched_class=matched,
                      lock_class=lock.lower(), object_id=object_id, xy=xy,
                      count=count)


def detections_to_core(items: Sequence[Dict[str, Any]], src_w: int, src_h: int,
                       dst_w: int, dst_h: int):
    """Detection-server wire boxes -> :class:`Detection2D` in our pixel space.

    The server reports boxes in the pixel space of the JPEG it was posted,
    which is the frame the node encoded. That is normally the node's own
    camera geometry, but it is rescaled anyway (as the FALCON reference node
    does): a resized or cropped post would otherwise hand the servo an offset
    it reads as a real bearing error and yaws the aircraft onto.

    Args:
        items: The ``detections`` list of a ``POST /detect`` reply.
        src_w: Width of the posted frame (px).
        src_h: Height of the posted frame (px).
        dst_w: Width of the node's intrinsics (px).
        dst_h: Height of the node's intrinsics (px).

    Returns:
        One :class:`Detection2D` per item, boxes in ``dst`` pixels.

    Raises:
        ValueError: If an item is malformed, or a frame size is not positive.
    """
    for name, value in (("src_w", src_w), ("src_h", src_h),
                        ("dst_w", dst_w), ("dst_h", dst_h)):
        if int(value) <= 0:
            raise ValueError("%s must be > 0, got %r" % (name, value))
    out = []
    for wire in detections_from_json(items):
        box = rescale_xyxy(wire.xyxy, int(src_w), int(src_h),
                           int(dst_w), int(dst_h))
        out.append(Detection2D(label=str(wire.cls), score=float(wire.conf),
                               bbox_xyxy=tuple(int(round(v)) for v in box),
                               frame_w=int(dst_w), frame_h=int(dst_h)))
    return out


def bbox_range_m(depth_img: np.ndarray, bbox_rgb, rgb_k: PinholeK,
                 depth_k: PinholeK, min_depth_m: float = 0.30,
                 max_depth_m: float = 8.0) -> Optional[float]:
    """Metric range (m) to a tracked RGB box, measured on the depth camera.

    The two-step the node must never shortcut: move the box through the depth
    camera's pinhole first (the two front sensors share a pose but not a FOV),
    then take the robust low-percentile depth of the shrunken box -- the same
    measurement ``object_mapper_node`` places landmarks with, so the range the
    approach lands on and the position the map recorded come from one routine.

    Args:
        depth_img: ``(H, W)`` float depth in metres (the SJTU front depth
            camera publishes ``32FC1`` metres, not millimetres).
        bbox_rgb: The tracked box ``(x1, y1, x2, y2)`` in RGB pixels.
        rgb_k: RGB intrinsics ``(fx, fy, cx, cy)``.
        depth_k: Depth intrinsics ``(fx, fy, cx, cy)``.
        min_depth_m: Reject depths below this (near-clip artefacts).
        max_depth_m: Reject depths above this (far background).

    Returns:
        Range in metres, or ``None`` when the box has too few valid depth
        pixels -- a legitimate "no measurement this frame", which the state
        machine treats as a broken land streak rather than as an arrival.

    Raises:
        ValueError: If ``depth_img`` is not 2D or an intrinsic is degenerate.
    """
    depth_bbox = rescale_bbox_between_intrinsics(
        tuple(float(v) for v in bbox_rgb), rgb_k, depth_k)
    return robust_bbox_depth(depth_img, depth_bbox, min_depth_m=float(min_depth_m),
                             max_depth_m=float(max_depth_m))


def approach_info_payload(stamp: float, state: str, target: str,
                          lock_class: str, engaged: bool, confirmed: bool,
                          streak: int, tracking: bool,
                          range_m: Optional[float], ticks: int,
                          elapsed_s: float, ended: bool,
                          reason: str) -> Dict[str, Any]:
    """The latched ``/target_approach/info`` status payload.

    Every value is coerced to a plain Python scalar here, so a numpy float
    smuggled in from the depth measurement cannot make ``json.dumps`` raise at
    publish time -- which would lose the one message that says how the mission
    ended.

    Args:
        stamp: Seconds (node clock) this status describes.
        state: The approach state machine's label, or ``"IDLE"`` before the
            target was seen and ``"DONE"`` once the node has finished.
        target: The mission target text.
        lock_class: The detector class being locked onto.
        engaged: True while this node owns ``cmd_vel`` (the follower is muted).
        confirmed: The confirmation gate has reached its streak.
        streak: Consecutive matching detector frames.
        tracking: The tracker holds a box this tick.
        range_m: Metric range to the target, or None when unmeasured.
        ticks: Control ticks run since arming.
        elapsed_s: Seconds since arming.
        ended: True once the node has stopped for good (landed or gave up).
        reason: Why it is in this state -- and, once ``ended``, why it ended.

    Returns:
        The wire dict. Renaming a key here blanks the operator panel.
    """
    return {
        "stamp": float(stamp),
        "state": str(state),
        "target": str(target),
        "lock_class": str(lock_class),
        "engaged": bool(engaged),
        "confirmed": bool(confirmed),
        "streak": int(streak),
        "tracking": bool(tracking),
        "range_m": None if range_m is None else float(range_m),
        "ticks": int(ticks),
        "elapsed_s": float(elapsed_s),
        "ended": bool(ended),
        "reason": str(reason),
    }
