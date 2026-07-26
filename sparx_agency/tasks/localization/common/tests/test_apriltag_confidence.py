"""Confidence must track real accuracy, because the filter uses it as its gain.

The pose filter weights each fix by the error this confidence implies, so a
confidence that is merely plausible is not good enough: if it says a lone distant
tag is as good as three near ones, the filter will let that tag drag the drone's
reported position around. These tests pin the ordering that makes the gain safe.

The scene is the deployed hallway geometry (tags on more than one wall, mixed
tag sizes), rendered through the real 504x294 intrinsics.
"""
import math

import cv2
import numpy as np
import pytest

from sparx_agency.core.localization.tag_triangulation import (
    TagWorldPose, world_T_tag_from_pose,
)
from sparx_agency.tasks.localization.common.apriltag_cv_common import tag_object_points
from sparx_agency.tasks.localization.common.apriltag_pnp import (
    TagDetection, estimate_camera_pose, pose_confidence,
)

# Real deployed intrinsics (camera_xtend_ros_calib_504_294_resize.yaml).
K = np.array([[322.635108347494793, 0.0, 242.064796586797144],
              [0.0, 323.389330714117420, 90.030190766806044],
              [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.zeros(5)

# A subset of the real map: end-of-hallway wall plus a side wall.
POSES = {
    5: TagWorldPose(xyz=(4.5, 0.0, 1.33), rpy=(-1.5708, 0.0, -1.5708)),
    10: TagWorldPose(xyz=(3.13, -0.82, 1.03), rpy=(-1.5708, 0.0, -1.5708)),
    8: TagWorldPose(xyz=(3.13, -1.75, 1.03), rpy=(-1.5708, 0.0, -1.5708)),
    2: TagWorldPose(xyz=(2.55, -2.3, 0.97), rpy=(-1.5708, 0.0, -3.1416)),
}
SIZES = {5: 0.29, 10: 0.2, 8: 0.15, 2: 0.2}


def _world_T_cam(pos, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[s, 0.0, c], [-c, 0.0, s], [0.0, -1.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _detect(tag_id, pos, yaw, rng, noise_px=0.5):
    """Render one tag as it would appear from (pos, yaw)."""
    wt = world_T_tag_from_pose(POSES[tag_id])
    wc = (wt @ np.c_[tag_object_points(SIZES[tag_id]), np.ones(4)].T).T[:, :3]
    cTw = np.linalg.inv(_world_T_cam(pos, yaw))
    rvec, _ = cv2.Rodrigues(cTw[:3, :3])
    img, _ = cv2.projectPoints(wc.reshape(-1, 1, 3), rvec, cTw[:3, 3], K, D)
    img = img.reshape(4, 2) + rng.normal(0.0, noise_px, (4, 2))
    return TagDetection(tag_id=tag_id, corners=img, size_m=SIZES[tag_id])


POS, YAW = np.array([1.0, -0.3, 1.0]), 0.0


def _conf(tag_ids, seed=0, noise_px=0.5):
    rng = np.random.default_rng(seed)
    dets = [_detect(t, POS, YAW, rng, noise_px) for t in tag_ids]
    est = estimate_camera_pose(dets, POSES, K, D)
    assert est is not None
    return pose_confidence(est), est


def test_more_tags_raise_confidence():
    """The dominant term: one tag can be off by tens of cm, three by ~1 cm."""
    c1, _ = _conf([8])
    c2, _ = _conf([8, 10])
    c3, _ = _conf([8, 10, 5])
    assert c1 < c2 < c3, (c1, c2, c3)


def test_a_lone_tag_is_never_highly_confident():
    """However crisp a single tag looks, its pose is not worth a high gain.

    A lone tag is coplanar with itself and has no baseline, so the geometry that
    would pin the pose simply is not there -- noise-free corners do not change
    that. This is the guard that stops the filter chasing a single tag.
    """
    for tag in (5, 10, 8, 2):
        c, est = _conf([tag], noise_px=0.0)
        assert est.n_tags == 1
        assert c <= 0.5, "tag %d scored %.2f with perfect corners" % (tag, c)


def test_confidence_falls_when_a_tag_contradicts_the_map():
    """A mis-measured tag map entry must show up as lower confidence.

    This is the failure that silently poisons a fix: the pooled RMS averages one
    bad tag away, so the per-tag residual is what has to catch it.
    """
    rng = np.random.default_rng(1)
    good = [_detect(t, POS, YAW, rng, 0.3) for t in (8, 10, 5)]
    c_good, est_good = _conf([8, 10, 5], seed=1, noise_px=0.3)

    # Same tags, but tag 5's map entry is wrong by 20 cm.
    bad_map = dict(POSES)
    bad_map[5] = TagWorldPose(xyz=(4.5, -0.20, 1.33), rpy=POSES[5].rpy)
    est_bad = estimate_camera_pose(good, bad_map, K, D)
    assert est_bad is not None
    c_bad = pose_confidence(est_bad)

    assert est_bad.worst_tag_rms_px > est_good.worst_tag_rms_px
    assert c_bad < c_good, (c_bad, c_good)


def test_confidence_is_bounded():
    for ids in ([8], [8, 10], [8, 10, 5], [8, 10, 5, 2]):
        c, _ = _conf(ids)
        assert 0.0 <= c <= 1.0


def test_geometry_score_rewards_a_spread_of_tags():
    """Tags far apart constrain the pose better than tags side by side."""
    _, close = _conf([10, 8])          # neighbours on the same wall
    _, spread = _conf([5, 10, 8, 2])   # across two walls, much wider
    assert spread.geometry > close.geometry


def test_quality_metrics_are_populated():
    """The filter and the confidence topic both read these; empty means silent
    mis-gating rather than a visible failure."""
    _, est = _conf([8, 10, 5])
    assert est.min_tag_px > 0.0
    assert est.max_tag_dist_m > 0.0
    assert est.worst_tag_rms_px >= 0.0
    assert 0.0 <= est.geometry <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
