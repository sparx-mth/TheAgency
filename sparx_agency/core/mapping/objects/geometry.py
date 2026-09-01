"""Detection-to-world geometry: bbox intrinsics rescale, robust depth, back-projection.

Pure numpy, ROS-free, Python-3.8-safe. Ported from the SJTU stack's
``semantic_mapper/object_mapper_node.py`` (``_det_cb`` / ``_robust_depth``),
with one deliberate fix — see :func:`backproject_bbox_to_world`.

Frames (the repo's four-frames doctrine)
----------------------------------------
* **Image**: pixels, origin top-left, ``u`` right, ``v`` down. Boxes are
  ``(x1, y1, x2, y2)`` as in :mod:`sparx_agency.core.common.math.bbox`.
* **Camera optical** (OpenCV): ``x`` right, ``y`` down, ``z`` forward — what a
  raw pinhole back-projection yields. Converted to FLU immediately, never
  propagated further.
* **Body FLU** (REP-103): ``x`` forward, ``y`` left, ``z`` up.
* **World ENU**: ``z`` up, right-handed — the frame of ``Pose2D`` and the BEV
  grid; landmark XY lives here.

Intrinsics are plain ``(fx, fy, cx, cy)`` tuples throughout (pass an
``Intrinsics`` object as ``(K.fx, K.fy, K.cx, K.cy)``).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.math.bbox import BBox, xyxy_center, xyxy_size
from sparx_agency.core.mapping.depth.depth_bbox_fusion import valid_depth_mask

PinholeK = Tuple[float, float, float, float]
"""Pinhole intrinsics ``(fx, fy, cx, cy)`` in pixels."""

R_BODY_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)
"""Standard camera-optical -> body-FLU rotation (REP-103/105).

``v_body = R_BODY_OPTICAL @ v_optical``: optical ``z`` (forward) -> body ``x``,
optical ``x`` (right) -> body ``-y``, optical ``y`` (down) -> body ``-z``.
Determinant +1.

Deliberately not shared with
:func:`sparx_agency.core.mapping.pipeline.mapping_pipeline.optical_xyz_to_base_xyz`:
that helper maps optical ``y`` to base ``z`` with a **positive** sign (its own
BEV convention, where the height component is discarded downstream) and casts
to float32; this constant is the exact orthonormal rotation, needed here
because the world ``z`` of a landmark is kept.
"""


def rescale_bbox_between_intrinsics(bbox_xyxy: BBox, src_k: PinholeK,
                                    dst_k: PinholeK) -> BBox:
    """Map a pixel bbox from one camera's intrinsics into another's.

    Each corner moves through normalized image coordinates:
    ``x' = (x - cx_src) * (fx_dst / fx_src) + cx_dst`` (same for ``y`` with
    ``fy``/``cy``), i.e. the same viewing ray re-projected through the second
    pinhole. This is exact only when the two cameras are coaxial and
    co-located (same pose and optical axis, different intrinsics) — the
    RGB-vs-depth situation of the SJTU Gazebo drone, where the RGB camera has
    a much wider FOV than the depth camera and a plain resolution rescale
    (:func:`sparx_agency.core.mapping.depth.depth_bbox_fusion.rescale_bbox_to_depth`,
    which serves the same-camera-different-resolution case) would be wrong.

    Args:
        bbox_xyxy: ``(x1, y1, x2, y2)`` in source-camera pixels.
        src_k: Source intrinsics ``(fx, fy, cx, cy)``.
        dst_k: Destination intrinsics ``(fx, fy, cx, cy)``.

    Returns:
        ``(x1, y1, x2, y2)`` in destination-camera pixels (unclipped; clip with
        :func:`sparx_agency.core.common.math.bbox.clip_xyxy` if needed).

    Raises:
        ValueError: If a focal length is not strictly positive.
    """
    fx_s, fy_s, cx_s, cy_s = (float(v) for v in src_k)
    fx_d, fy_d, cx_d, cy_d = (float(v) for v in dst_k)
    if min(fx_s, fy_s, fx_d, fy_d) <= 0.0:
        raise ValueError(
            "focal lengths must be positive, got src=(%r, %r) dst=(%r, %r)"
            % (fx_s, fy_s, fx_d, fy_d))
    kx = fx_d / fx_s
    ky = fy_d / fy_s
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    return ((x1 - cx_s) * kx + cx_d, (y1 - cy_s) * ky + cy_d,
            (x2 - cx_s) * kx + cx_d, (y2 - cy_s) * ky + cy_d)


def robust_bbox_depth(depth_img: np.ndarray, bbox_xyxy: BBox,
                      min_depth_m: float = 0.30, max_depth_m: float = 5.0,
                      shrink: float = 0.50, percentile: float = 30.0,
                      min_valid_px: int = 20) -> Optional[float]:
    """Robust metric depth of a detection: low percentile of the shrunken bbox.

    Ports ``ObjectMapper._robust_depth`` semantics exactly: shrink the box
    about its center (a detection box overhangs its object, so the rim pixels
    are background), keep only finite depths within ``[min_depth_m,
    max_depth_m]``, require at least ``min_valid_px`` of them, and take a low
    percentile — near-side-of-object, robust to background bleed-through.

    Args:
        depth_img: ``(H, W)`` float metric depth in meters (optical ``z``).
            NaN/inf holes are allowed and ignored.
        bbox_xyxy: ``(x1, y1, x2, y2)`` in ``depth_img`` pixels.
        min_depth_m: Reject depths below this (sensor near-clip artifacts).
        max_depth_m: Reject depths above this (far background).
        shrink: Box scale factor about the center, clamped to ``[0.05, 1.0]``.
        percentile: Percentile of the valid depths to return.
        min_valid_px: Minimum number of valid pixels; fewer means no
            measurement.

    Returns:
        Depth in meters, or ``None`` when the shrunken box is degenerate /
        outside the image or has fewer than ``min_valid_px`` valid pixels
        (a legitimate "no measurement", not an error).
    """
    depth = np.asarray(depth_img)
    if depth.ndim != 2:
        raise ValueError("depth_img must be (H, W), got shape %r"
                         % (depth.shape,))
    height, width = depth.shape
    u, v = xyxy_center(bbox_xyxy)
    bw, bh = xyxy_size(bbox_xyxy)
    hw, hh = 0.5 * bw, 0.5 * bh
    s = max(0.05, min(1.0, float(shrink)))
    x0, x1 = int(max(0.0, u - hw * s)), int(min(float(width), u + hw * s))
    y0, y1 = int(max(0.0, v - hh * s)), int(min(float(height), v + hh * s))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth[y0:y1, x0:x1]
    valid = valid_depth_mask(patch, min_depth=float(min_depth_m),
                             max_depth=float(max_depth_m))
    if int(valid.sum()) < int(min_valid_px):
        return None
    return float(np.percentile(patch[valid], float(percentile)))


def backproject_bbox_to_world(
    bbox_xyxy: BBox,
    depth_m: float,
    depth_k: PinholeK,
    rotation_world_body: np.ndarray,
    translation_world: Sequence[float],
    camera_offset_body: Sequence[float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """Back-project a bbox center at a known depth into world ENU.

    The chain, with every frame hop explicit::

        v_optical = [(u - cx) / fx * d,  (v - cy) / fy * d,  d]
        p_world   = R_world_body @ (R_BODY_OPTICAL @ v_optical
                                    + camera_offset_body) + t_world

    **Bug fixed relative to the source stack**: the SJTU
    ``semantic_mapper/object_mapper_node.py`` (``_det_cb``) computed
    ``Pw = cam_R @ [Xc, Yc, depth] + cam_p`` — rotating the CAMERA-OPTICAL ray
    (``x`` right, ``y`` down, ``z`` forward) directly by the BODY pose
    quaternion, skipping the optical->FLU conversion. With a level,
    identity-yaw pose that places an object seen dead ahead at ``d`` meters
    *above* the drone instead of ``d`` meters in front of it; landmarks only
    looked plausible because the map was flat and yaw errors partially
    cancelled. Here the ray is rotated into body FLU first
    (:data:`R_BODY_OPTICAL`), matching the repo's frames doctrine.

    The camera mount is **rotation-only plus a fixed body-frame lever arm**:
    ``camera_offset_body`` covers a camera mounted off the body origin, but
    there is no camera-tilt (body-pitch) parameter — no existing rotation
    helper in the tree provides a rotation about the body ``y`` axis
    (``core.control.flatness.rotations`` has only ``rotation_about_z``), so
    per CLAUDE.md's no-hand-rolled-frame-math rule the tilt is left out until
    such a helper exists. A tilted camera needs its ``R_body_camera`` folded
    in by the caller.

    Args:
        bbox_xyxy: ``(x1, y1, x2, y2)`` in **depth-camera** pixels (rescale
            RGB-space boxes with :func:`rescale_bbox_between_intrinsics`
            first).
        depth_m: Metric depth of the object along the optical axis, e.g. from
            :func:`robust_bbox_depth`. Must be finite and positive.
        depth_k: Depth-camera intrinsics ``(fx, fy, cx, cy)``.
        rotation_world_body: ``(3, 3)`` world-from-body rotation. From a ROS
            ``[x, y, z, w]`` pose quaternion use
            ``sparx_agency.core.common.math.se3.quaternion_matrix(q)[:3, :3]``.
        translation_world: Body position ``(x, y, z)`` in world ENU.
        camera_offset_body: Camera position in the body FLU frame, meters.
            Default zero (camera at the body origin).

    Returns:
        ``(x, y, z)`` of the object in world ENU.

    Raises:
        ValueError: If ``depth_m`` is not finite and positive, a focal length
            is not positive, or ``rotation_world_body`` is not ``(3, 3)``.
    """
    d = float(depth_m)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("depth_m must be finite and positive, got %r"
                         % (depth_m,))
    fx, fy, cx, cy = (float(v) for v in depth_k)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive, got (%r, %r)"
                         % (fx, fy))
    rotation = np.asarray(rotation_world_body, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_world_body must be (3, 3), got shape %r"
                         % (rotation.shape,))
    u, v = xyxy_center(bbox_xyxy)
    v_optical = np.array([(u - cx) / fx * d, (v - cy) / fy * d, d],
                         dtype=np.float64)
    v_body = R_BODY_OPTICAL.dot(v_optical) + np.asarray(
        camera_offset_body, dtype=np.float64).reshape(3)
    p_world = rotation.dot(v_body) + np.asarray(
        translation_world, dtype=np.float64).reshape(3)
    return (float(p_world[0]), float(p_world[1]), float(p_world[2]))
