"""Synthetic-scene tests for :mod:`apriltag_pnp`.

The scene mirrors the real deployment (end-of-hallway wall tags viewed by the
640x360 front camera, fx~149). It reproduces the single-tag planar ambiguity
and verifies that (a) disambiguation beats the raw reprojection winner and
(b) the joint multi-tag solve is markedly more accurate and stable.
"""
import numpy as np
import pytest

from sparx_agency.core.localization.tag_triangulation import (
    TagWorldPose,
    world_T_tag_from_pose,
)
from sparx_agency.tasks.localization.common.apriltag_cv_common import tag_object_points
from sparx_agency.tasks.localization.common.apriltag_pnp import (
    TagDetection,
    estimate_camera_pose,
)

# Real front-camera intrinsics (config/front_camera_calib.yaml).
K = np.array([[149.2, 0.0, 319.5],
              [0.0, 180.0, 179.5],
              [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.zeros(5)

# Camera optical (CV) frame in world: looks along +X world (down the hallway).
_R_WC = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]], dtype=np.float64)

# End-wall tags (from tag_map_path_ALL.yaml).
TAG_POSES = {
    0: TagWorldPose(xyz=(4.5, 0.0, 0.65), rpy=(-1.5708, 0.0, -1.5708)),
    7: TagWorldPose(xyz=(4.5, -0.25, 1.30), rpy=(-1.5708, 0.0, -1.5708)),
    8: TagWorldPose(xyz=(0.15, -4.7, 1.0), rpy=(-1.5708, 0.0, -3.1416)),
}
TAG_SIZE = {0: 0.20, 7: 0.15, 8: 0.20}


def _world_T_cam(cam_pos):
    T = np.eye(4)
    T[:3, :3] = _R_WC
    T[:3, 3] = cam_pos
    return T


def _project(world_pts, cam_pos, rng, noise_px):
    import cv2
    cam_T_world = np.linalg.inv(_world_T_cam(cam_pos))
    rvec, _ = cv2.Rodrigues(cam_T_world[:3, :3])
    img, _ = cv2.projectPoints(world_pts.reshape(-1, 1, 3), rvec,
                               cam_T_world[:3, 3], K, D)
    img = img.reshape(-1, 2)
    if noise_px:
        img = img + rng.normal(0.0, noise_px, img.shape)
    return img


def _detection(tag_id, cam_pos, rng, noise_px=0.0):
    obj = tag_object_points(TAG_SIZE[tag_id])
    wt = world_T_tag_from_pose(TAG_POSES[tag_id])
    world_corners = (wt @ np.c_[obj, np.ones(4)].T).T[:, :3]
    corners = _project(world_corners, cam_pos, rng, noise_px)
    return TagDetection(tag_id=tag_id, corners=corners, size_m=TAG_SIZE[tag_id])


def test_returns_none_without_known_tags():
    rng = np.random.default_rng(0)
    det = TagDetection(tag_id=999, corners=np.zeros((4, 2)), size_m=0.2)
    assert estimate_camera_pose([det], TAG_POSES, K, D) is None


def test_single_tag_close_is_accurate():
    """A close, well-conditioned tag should localise to a few cm."""
    rng = np.random.default_rng(1)
    cam_pos = np.array([3.2, -0.9, 1.1])  # ~1.6 m, off-axis
    errs = []
    for _ in range(30):
        res = estimate_camera_pose([_detection(0, cam_pos, rng, 0.4)],
                                   TAG_POSES, K, D)
        assert res is not None and res.n_tags == 1
        errs.append(np.linalg.norm(res.world_T_cam[:3, 3] - cam_pos))
    assert np.mean(errs) < 0.10


def test_two_tags_beat_single_tag_and_are_accurate():
    """Joint multi-tag PnP must be clearly more accurate than a single tag."""
    rng = np.random.default_rng(2)
    cam_pos = np.array([2.5, 0.0, 1.0])  # far, fronto-parallel: worst case
    single_errs, joint_errs = [], []
    for _ in range(60):
        d0 = _detection(0, cam_pos, rng, 0.4)
        d7 = _detection(7, cam_pos, rng, 0.4)
        r1 = estimate_camera_pose([d0], TAG_POSES, K, D)
        r2 = estimate_camera_pose([d0, d7], TAG_POSES, K, D)
        assert r1 is not None and r2 is not None
        assert r2.n_tags == 2
        single_errs.append(np.linalg.norm(r1.world_T_cam[:3, 3] - cam_pos))
        joint_errs.append(np.linalg.norm(r2.world_T_cam[:3, 3] - cam_pos))
    assert np.mean(joint_errs) < 0.30
    assert np.mean(joint_errs) < 0.6 * np.mean(single_errs)


def test_two_tags_across_walls_are_very_accurate():
    """Non-coplanar tags (different walls) remove the ambiguity: sub-15 cm."""
    rng = np.random.default_rng(3)
    cam_pos = np.array([1.5, -2.0, 1.0])
    errs = []
    for _ in range(40):
        d0 = _detection(0, cam_pos, rng, 0.4)   # end wall
        d8 = _detection(8, cam_pos, rng, 0.4)   # side wall (different normal)
        res = estimate_camera_pose([d0, d8], TAG_POSES, K, D)
        assert res is not None and res.n_tags == 2
        errs.append(np.linalg.norm(res.world_T_cam[:3, 3] - cam_pos))
    assert np.mean(errs) < 0.15


def test_ambiguity_flag_is_high_for_far_single_tag():
    rng = np.random.default_rng(4)
    cam_pos = np.array([2.5, 0.0, 1.0])
    ambs = [estimate_camera_pose([_detection(0, cam_pos, rng, 0.4)],
                                 TAG_POSES, K, D).ambiguity for _ in range(30)]
    assert np.mean(ambs) > 0.5  # near-degenerate branches


def test_outlier_tag_is_dropped():
    """A tag whose corners are inconsistent with the map is rejected."""
    rng = np.random.default_rng(5)
    cam_pos = np.array([1.5, -2.0, 1.0])
    d0 = _detection(0, cam_pos, rng, 0.2)
    d8 = _detection(8, cam_pos, rng, 0.2)
    # tag 7 corners produced as if the camera were somewhere else (bad detection)
    bad7 = _detection(7, np.array([0.0, 0.0, 1.0]), rng, 0.2)
    res = estimate_camera_pose([d0, d8, bad7], TAG_POSES, K, D)
    assert res is not None
    assert 7 not in res.used_tag_ids
    assert np.linalg.norm(res.world_T_cam[:3, 3] - cam_pos) < 0.20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
