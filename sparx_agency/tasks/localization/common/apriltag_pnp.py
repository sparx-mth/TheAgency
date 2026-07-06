"""Robust camera-pose estimation from AprilTag detections (OpenCV, no ROS).

This module replaces the "solve each tag independently, then average" approach
with two improvements that address the two classic failure modes of AprilTag
localization:

1. **Single-tag planar ambiguity (the "pose jumps" problem).**
   A single square tag is four coplanar points, so ``solvePnP`` always has *two*
   near-degenerate solutions that are mirror images across the fronto-parallel
   plane. ``cv2.solvePnP`` silently returns only the lower-reprojection-error one;
   when the tag is small/far/fronto-parallel the two errors are within pixel noise
   and the winner flips frame-to-frame, mirroring the recovered tilt and throwing
   the camera position to a mirrored location. Here we evaluate **both** branches
   with ``cv2.solvePnPGeneric`` and, when they are ambiguous, disambiguate with a
   physical prior (camera is roughly gravity-upright, and temporal continuity)
   instead of trusting the raw reprojection winner.

2. **Multiple tags under-used (weak fusion).**
   Instead of solving each tag separately and averaging the camera positions
   (keeping only one tag's orientation), we run a **single joint PnP** over the
   pooled corners of *all* visible tags expressed in the world frame. When the
   tags span more than one wall the pooled points are non-coplanar, so the
   ambiguity disappears entirely and every corner constrains both position and
   orientation jointly. A mis-mapped/outlier tag is detected by reprojection and
   dropped.

All poses are returned as ``world_T_cam`` in the OpenCV optical camera frame
(X right, Y down, Z forward), matching the legacy estimator, so downstream
CV->ROS conversion and filtering are unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparx_agency.core.localization.tag_triangulation import (
    TagWorldPose,
    world_T_tag_from_pose,
)
from sparx_agency.tasks.localization.common.apriltag_cv_common import tag_object_points

# --- Tuning constants ---------------------------------------------------------
# Above this ratio of best/second reprojection error the two solutions are
# considered "ambiguous" and the physical prior is used to pick the branch.
# Below it, the reprojection winner is trusted outright.
_AMBIGUITY_GATE = 0.7
# Weight (radians per metre) trading off temporal continuity against uprightness
# when scoring the two ambiguous branches.
_TEMPORAL_WEIGHT = 0.5
# A tag whose corners reproject worse than this (px) AND far worse than the
# median tag is treated as an outlier and dropped from a multi-tag solve.
_OUTLIER_PX = 4.0


@dataclass(frozen=True)
class TagDetection:
    """A single detected tag ready for PnP.

    Attributes:
        tag_id: AprilTag id.
        corners: (4, 2) float image corners in ``pupil_apriltags`` order, i.e.
            tag points (-1, 1), (1, 1), (1, -1), (-1, -1) — the same order as
            :func:`tag_object_points`.
        size_m: Physical tag edge length in metres.
    """

    tag_id: int
    corners: np.ndarray
    size_m: float


@dataclass(frozen=True)
class CameraPoseResult:
    """Estimated camera pose plus quality metrics for confidence gating.

    Attributes:
        world_T_cam: (4, 4) camera pose in world, OpenCV optical convention.
        used_tag_ids: Tag ids that contributed to this estimate.
        n_tags: Number of tags used.
        reproj_rms_px: RMS reprojection error over all used corners (px).
        ambiguity: 0 (unique/robust) .. 1 (two branches equally good). Only a
            single, fronto-parallel tag is typically near 1.
    """

    world_T_cam: np.ndarray
    used_tag_ids: List[int]
    n_tags: int
    reproj_rms_px: float
    ambiguity: float


# --- small SE(3) / reprojection helpers --------------------------------------

def _rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def _reproj_rms(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray,
                D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(
        obj_pts.reshape(-1, 1, 3).astype(np.float64),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1), K, D)
    d = proj.reshape(-1, 2) - img_pts.reshape(-1, 2)
    return float(math.sqrt(np.mean(np.sum(d * d, axis=1))))


def _refine_vvs(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray,
                D: np.ndarray, rvec: np.ndarray,
                tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Virtual Visual Servoing refinement; returns the input unchanged on error."""
    try:
        r = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
        t = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
        r, t = cv2.solvePnPRefineVVS(
            obj_pts.reshape(-1, 1, 3).astype(np.float64),
            img_pts.reshape(-1, 1, 2).astype(np.float64), K, D, r, t)
        return r, t
    except cv2.error:
        return rvec, tvec


def _world_corners(det: TagDetection, world_T_tag: np.ndarray) -> np.ndarray:
    """Return the tag's 4 corners in world coordinates, (4, 3)."""
    obj = tag_object_points(det.size_m)
    homog = np.concatenate([obj, np.ones((4, 1))], axis=1)  # (4, 4)
    return (world_T_tag @ homog.T).T[:, :3]


# --- branch disambiguation ----------------------------------------------------

def _branch_score(world_T_cam: np.ndarray, prev_cam_pos: Optional[np.ndarray],
                  gravity_down: np.ndarray) -> float:
    """Physical implausibility cost for one candidate camera pose (lower = better).

    Uses two priors that cleanly separate the true pose from its planar mirror:

    * Uprightness: the camera optical Y axis (image "down") should point roughly
      along world gravity. The mirror solution introduces a large spurious tilt,
      so it scores much worse. Robust to moderate drone pitch.
    * Temporal continuity (optional): closeness to the previous camera position.
    """
    cam_down = world_T_cam[:3, 1]  # optical Y expressed in world
    cam_down = cam_down / (np.linalg.norm(cam_down) + 1e-12)
    cos = float(np.clip(cam_down @ gravity_down, -1.0, 1.0))
    cost = math.acos(cos)  # radians from perfectly upright
    if prev_cam_pos is not None:
        cost += _TEMPORAL_WEIGHT * float(np.linalg.norm(world_T_cam[:3, 3] - prev_cam_pos))
    return cost


def _pick_branch(candidates: List[Tuple[np.ndarray, float, np.ndarray, np.ndarray]],
                 prev_cam_pos: Optional[np.ndarray],
                 gravity_down: np.ndarray) -> Tuple[Tuple, float]:
    """Choose among PnP candidates (each: world_T_cam, err, rvec, tvec).

    Returns (winning_candidate, ambiguity). ``candidates`` must be sorted by
    reprojection error ascending.
    """
    if len(candidates) == 1:
        return candidates[0], 0.0

    e0, e1 = candidates[0][1], candidates[1][1]
    ambiguity = 1.0 if e1 <= 1e-9 else float(min(1.0, e0 / e1))

    # Reprojection clearly prefers one branch -> trust it.
    if ambiguity < _AMBIGUITY_GATE:
        return candidates[0], ambiguity

    # Ambiguous -> let the physical prior decide.
    best = min(candidates, key=lambda c: _branch_score(c[0], prev_cam_pos, gravity_down))
    return best, ambiguity


# --- single- and multi-tag solvers -------------------------------------------

def _solve_single(det: TagDetection, world_T_tag: np.ndarray, K: np.ndarray,
                  D: np.ndarray, prev_cam_pos: Optional[np.ndarray],
                  gravity_down: np.ndarray) -> Optional[CameraPoseResult]:
    obj = tag_object_points(det.size_m).astype(np.float64)
    img = np.ascontiguousarray(det.corners, dtype=np.float64).reshape(4, 2)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    if n < 1:
        return None

    cands = []
    for i in range(n):
        world_T_cam = world_T_tag @ _inv_T(_rt_to_T(rvecs[i], tvecs[i]))
        err = float(np.asarray(errs[i]).reshape(-1)[0])
        cands.append((world_T_cam, err, rvecs[i], tvecs[i]))
    cands.sort(key=lambda c: c[1])

    (_, _, rvec, tvec), ambiguity = _pick_branch(cands, prev_cam_pos, gravity_down)
    rvec, tvec = _refine_vvs(obj, img, K, D, rvec, tvec)
    world_T_cam = world_T_tag @ _inv_T(_rt_to_T(rvec, tvec))
    rms = _reproj_rms(obj, img, K, D, rvec, tvec)
    return CameraPoseResult(world_T_cam, [det.tag_id], 1, rms, ambiguity)


def _solve_multi(dets: Sequence[TagDetection], poses: Dict[int, TagWorldPose],
                 K: np.ndarray, D: np.ndarray, prev_cam_pos: Optional[np.ndarray],
                 gravity_down: np.ndarray) -> Optional[CameraPoseResult]:
    world_pts: List[np.ndarray] = []
    img_pts: List[np.ndarray] = []
    used: List[int] = []
    for det in dets:
        wt = world_T_tag_from_pose(poses[det.tag_id])
        world_pts.append(_world_corners(det, wt))
        img_pts.append(np.asarray(det.corners, dtype=np.float64).reshape(4, 2))
        used.append(det.tag_id)

    world_pts_arr = np.ascontiguousarray(np.vstack(world_pts), dtype=np.float64)
    img_pts_arr = np.ascontiguousarray(np.vstack(img_pts), dtype=np.float64)

    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            world_pts_arr, img_pts_arr, K, D, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if n < 1:
        return None

    cands = []
    for i in range(n):
        world_T_cam = _inv_T(_rt_to_T(rvecs[i], tvecs[i]))  # solve gives cam_T_world
        err = float(np.asarray(errs[i]).reshape(-1)[0])
        cands.append((world_T_cam, err, rvecs[i], tvecs[i]))
    cands.sort(key=lambda c: c[1])

    (_, _, rvec, tvec), ambiguity = _pick_branch(cands, prev_cam_pos, gravity_down)
    rvec, tvec = _refine_vvs(world_pts_arr, img_pts_arr, K, D, rvec, tvec)
    world_T_cam = _inv_T(_rt_to_T(rvec, tvec))
    rms = _reproj_rms(world_pts_arr, img_pts_arr, K, D, rvec, tvec)
    return CameraPoseResult(world_T_cam, list(used), len(used), rms, ambiguity)


def estimate_camera_pose(
    detections: Sequence[TagDetection],
    tag_world_poses: Dict[int, TagWorldPose],
    K: np.ndarray,
    D: np.ndarray,
    prev_cam_pos_world: Optional[np.ndarray] = None,
    gravity_down_world: Tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> Optional[CameraPoseResult]:
    """Estimate ``world_T_cam`` from one or more tag detections.

    A single tag uses the disambiguated IPPE-square solve; two or more tags use a
    single joint PnP over all pooled corners, with reprojection-based outlier
    rejection. Detections whose id is absent from ``tag_world_poses`` are ignored.

    Args:
        detections: Detected tags (corners in ``pupil_apriltags`` order).
        tag_world_poses: Known tag poses in world (from the tag map).
        K: (3, 3) camera matrix.
        D: distortion coefficients.
        prev_cam_pos_world: Previous camera position in world for temporal
            disambiguation (optional).
        gravity_down_world: World "down" unit vector; default assumes +Z is up.

    Returns:
        A :class:`CameraPoseResult`, or ``None`` if no known tag was usable.
    """
    dets = [d for d in detections if d.tag_id in tag_world_poses]
    if not dets:
        return None

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    gravity = np.asarray(gravity_down_world, dtype=np.float64)
    gravity = gravity / (np.linalg.norm(gravity) + 1e-12)

    if len(dets) == 1:
        wt = world_T_tag_from_pose(tag_world_poses[dets[0].tag_id])
        return _solve_single(dets[0], wt, K, D, prev_cam_pos_world, gravity)

    result = _solve_multi(dets, tag_world_poses, K, D, prev_cam_pos_world, gravity)
    if result is None:
        return None

    # Reprojection-based outlier rejection by exhaustive leave-one-out. Only
    # attempted with >= 3 tags: a bad tag inflates *every* residual (least
    # squares compromises) so a per-tag ratio test picks the wrong one, and
    # dropping to a single tag always reprojects near-zero (a lone tag trivially
    # fits its own corners) so it proves nothing. With >= 3 tags the kept set
    # (>= 2) stays mutually constrained, so the drop giving the lowest — and much
    # lower — kept RMS is real evidence the removed tag was the outlier.
    if len(dets) >= 3 and result.reproj_rms_px > _OUTLIER_PX:
        best_kept: Optional[List[TagDetection]] = None
        best_rms = result.reproj_rms_px
        for dropped in dets:
            kept = [d for d in dets if d.tag_id != dropped.tag_id]
            alt = _solve_multi(kept, tag_world_poses, K, D, prev_cam_pos_world, gravity)
            if alt is not None and alt.reproj_rms_px < best_rms:
                best_rms, best_kept = alt.reproj_rms_px, kept
        if best_kept is not None and best_rms <= _OUTLIER_PX and best_rms < 0.4 * result.reproj_rms_px:
            # Recurse to re-clean (multiple outliers) and produce a refined result.
            return estimate_camera_pose(
                best_kept, tag_world_poses, K, D, prev_cam_pos_world, gravity_down_world)

    return result


def pose_confidence(result: CameraPoseResult) -> float:
    """Map a :class:`CameraPoseResult` to a 0..1 confidence.

    More tags, lower reprojection error and lower ambiguity all raise confidence.
    An ambiguous single tag is strongly penalised so it does not inject jumps.
    """
    tag_term = min(1.0, 0.4 + 0.3 * result.n_tags)          # 1 tag -> 0.7, >=2 -> 1.0
    rms_term = 1.0 / (1.0 + (result.reproj_rms_px / 1.5))    # 0px -> 1.0, 1.5px -> 0.5
    amb_term = 1.0 - 0.5 * result.ambiguity                 # ambiguous -> halve
    return float(max(0.0, min(1.0, tag_term * rms_term * amb_term)))
