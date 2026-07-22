"""One frame's AprilTag diagnostics: which tags were seen, and how good each was.

The localization filter only ever publishes an *aggregate* pose confidence. That
is the right signal for the controller, but it cannot answer the question an
operator setting up a room actually has: *which tag is letting me down?* A tag
can be poorly placed (rarely in view, or only ever seen at a grazing angle from
far away) or mis-mapped (its recorded pose/size is wrong, so it drags every fix
it joins). Both are invisible in the aggregate and both are fixable on the wall.

This module turns the raw material the provider already has each frame -- the
detector's output, the tag map, and the solved :class:`CameraPoseResult` -- into
a flat per-tag record a logger can append. It is pure (no ROS, no file I/O, no
OpenCV): the provider builds the diagnostic, the ROS layer decides whether and
where to persist it.

Python 3.8 compatible: the provider that calls this runs under the ROS2 node and,
via the FALCON Noetic adapter, under 3.8.
"""
from __future__ import annotations

from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

from sparx_agency.tasks.localization.common.apriltag_pnp import CameraPoseResult


class TagObservation(NamedTuple):
    """How one tag showed up in one frame.

    Attributes:
        tag_id: AprilTag id.
        in_map: True if this id exists in the tag map. A detected tag that is NOT
            in the map is a tag on the wall the map does not know about -- it
            contributes nothing and is worth spotting.
        used: True if the tag actually contributed to the published fix (it was
            in the map, survived margin hysteresis, and was not dropped as a
            reprojection outlier). Detected-but-not-used, frame after frame, is
            the signature of a mis-mapped tag.
        decision_margin: The detector's own confidence in the detection. Low means
            a hard-to-read tag: poor print, glare, motion blur, or steep angle.
        apparent_px: Largest image extent of the tag (px) -- how big it looked.
        center_x: Tag centre x in the image (px); near the frame edge means it is
            about to leave view.
        center_y: Tag centre y in the image (px).
        dist_m: Camera-to-tag distance for this fix (m); None when the tag was
            detected but not used, so no pose placed it.
        reproj_rms_px: This tag's own reprojection residual under the shared pose
            (px); None when the tag was not used. THE mis-map signal.
    """

    tag_id: int
    in_map: bool
    used: bool
    decision_margin: float
    apparent_px: float
    center_x: float
    center_y: float
    dist_m: Optional[float]
    reproj_rms_px: Optional[float]


class FrameDiag(NamedTuple):
    """Every tag seen in one frame, plus the fix they produced.

    Attributes:
        stamp_sec: Frame timestamp (s).
        source: ``apriltag`` (a real fix), ``apriltag_coast`` (dead reckoning, no
            usable tag), ``blind`` (no tag detected at all), or ``rejected`` (a
            fix computed but thrown out, e.g. an implausible jump).
        confidence: The published fix confidence, 0..1 (0 when there is no fix).
        pos_std_m: The published position-error estimate (m).
        n_detected: How many tags the detector found this frame (mapped or not).
        n_used: How many tags contributed to the fix.
        geometry: Fix geometry score 0..1 (see :class:`CameraPoseResult`).
        ambiguity: Fix planar-ambiguity score 0..1.
        reproj_rms_px: Pooled reprojection RMS of the fix (px).
        tags: One :class:`TagObservation` per DETECTED tag.
    """

    stamp_sec: float
    source: str
    confidence: float
    pos_std_m: float
    n_detected: int
    n_used: int
    geometry: float
    ambiguity: float
    reproj_rms_px: float
    tags: Tuple[TagObservation, ...]


def _apparent_px(corners) -> float:
    """Largest image extent of a tag from its 4 corners (px). Mirrors apriltag_pnp."""
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def build_frame_diag(detections: Sequence[object],
                     result: Optional[CameraPoseResult],
                     mapped_ids: Iterable[int],
                     confidence: float,
                     pos_std_m: float,
                     source: str,
                     stamp_sec: float) -> FrameDiag:
    """Assemble one :class:`FrameDiag` from a frame's raw material.

    Args:
        detections: Raw detector output for this frame. Each item must expose
            ``tag_id``, ``corners`` (4x2) and ``decision_margin``, and optionally
            ``center`` -- i.e. a ``pupil_apriltags`` detection or the provider's
            ``RawDet``. May be empty (a blind frame).
        result: The solved :class:`CameraPoseResult`, or None when no fix was
            produced (blind or coast). Supplies per-tag residual/distance and the
            used-tag set.
        mapped_ids: The tag ids present in the tag map.
        confidence: Published fix confidence for this frame.
        pos_std_m: Published position-error estimate for this frame.
        source: Provenance string (see :class:`FrameDiag`).
        stamp_sec: Frame timestamp.

    Returns:
        A fully-populated, immutable :class:`FrameDiag`.
    """
    mapped = set(int(i) for i in mapped_ids)
    used_ids = set(int(i) for i in result.used_tag_ids) if result else set()
    by_id = {int(s.tag_id): s for s in (result.per_tag if result else ())}

    tags: List[TagObservation] = []
    for det in detections:
        tid = int(det.tag_id)
        corners = det.corners
        center = getattr(det, "center", None)
        if center is not None:
            cx, cy = float(center[0]), float(center[1])
        else:
            cx = sum(float(c[0]) for c in corners) / 4.0
            cy = sum(float(c[1]) for c in corners) / 4.0
        stat = by_id.get(tid)
        tags.append(TagObservation(
            tag_id=tid,
            in_map=tid in mapped,
            used=tid in used_ids,
            decision_margin=float(getattr(det, "decision_margin", 0.0)),
            apparent_px=_apparent_px(corners),
            center_x=cx,
            center_y=cy,
            dist_m=(stat.dist_m if stat else None),
            reproj_rms_px=(stat.reproj_rms_px if stat else None),
        ))

    return FrameDiag(
        stamp_sec=float(stamp_sec),
        source=str(source),
        confidence=float(confidence),
        pos_std_m=float(pos_std_m),
        n_detected=len(tags),
        n_used=len(used_ids),
        geometry=float(result.geometry) if result else 0.0,
        ambiguity=float(result.ambiguity) if result else 0.0,
        reproj_rms_px=float(result.reproj_rms_px) if result else 0.0,
        tags=tuple(tags),
    )
