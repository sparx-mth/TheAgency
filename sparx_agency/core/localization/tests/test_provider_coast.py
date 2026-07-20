"""Blind-frame coasting: the provider's contract when NO tag is in view.

Not every wall carries a tag, so blind stretches are part of every route. For a
few frames the provider dead-reckons on the earned-trust command prior — and
these tests pin the promises that make that safe: a coasted estimate can never
look like a fix (source + collapsing confidence), the budget is hard, a stuck
drone coasts in place, and disabling either the coast or the command feed
restores publish-nothing-when-blind exactly.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from sparx_agency.core.common.types.perception import Observation, RGBFrame
from sparx_agency.core.localization.providers.apriltag_provider import (
    AprilTagLocalizationProvider,
)
from sparx_agency.core.localization.tag_triangulation import world_T_tag_from_pose
from sparx_agency.tasks.localization.common.apriltag_cv_common import tag_object_points

# Two tags on two different walls (like the deployed map) so fixes are confident
# enough (conf > conf_floor 0.3) for the command model to learn from them.
MAP_YAML = """
tags:
  1:
    xyz: [4.0, 0.0, 1.0]
    rpy: [-1.5708, 0, -1.5708]
    size: 0.25
  2:
    xyz: [2.5, -2.0, 1.0]
    rpy: [-1.5708, 0, -3.1416]
    size: 0.25
"""

CALIB_YAML = """
camera_matrix:
  rows: 3
  cols: 3
  data: [322.6, 0.0, 242.1, 0.0, 323.4, 90.0, 0.0, 0.0, 1.0]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.0, 0.0, 0.0, 0.0, 0.0]
"""

K = np.array([[322.6, 0.0, 242.1], [0.0, 323.4, 90.0], [0.0, 0.0, 1.0]])
GRAY = np.zeros((294, 504), dtype=np.uint8)
DT = 0.1


@dataclass
class FakeDet:
    tag_id: int
    corners: np.ndarray
    decision_margin: float = 60.0


class FakeDetector:
    """Replays scripted per-frame detections instead of reading pixels."""

    def __init__(self):
        self.next_dets = []

    def detect(self, gray):
        return self.next_dets


def cam_pose(pos, yaw):
    """world_T_cam, optical convention (X right, Y down, Z forward = +x@yaw=0)."""
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[s, 0.0, c], [-c, 0.0, s], [0.0, -1.0, 0.0]])
    T[:3, 3] = pos
    return T


def detections_from(pos, yaw, tag_ids=(1, 2)):
    """Perfect corner detections of the map tags as seen from (pos, yaw)."""
    poses = {1: ((4.0, 0.0, 1.0), (-1.5708, 0, -1.5708)),
             2: ((2.5, -2.0, 1.0), (-1.5708, 0, -3.1416))}
    cTw = np.linalg.inv(cam_pose(np.asarray(pos, float), yaw))
    rvec, _ = cv2.Rodrigues(cTw[:3, :3])
    out = []
    for tid in tag_ids:
        from sparx_agency.core.localization.tag_triangulation import TagWorldPose
        wt = world_T_tag_from_pose(TagWorldPose(xyz=poses[tid][0], rpy=poses[tid][1]))
        wc = (wt @ np.c_[tag_object_points(0.25), np.ones(4)].T).T[:, :3]
        img, _ = cv2.projectPoints(wc.reshape(-1, 1, 3), rvec, cTw[:3, 3],
                                   K, np.zeros(5))
        out.append(FakeDet(tid, img.reshape(4, 2)))
    return out


@pytest.fixture
def provider(tmp_path):
    mp = tmp_path / "map.yaml"
    cp = tmp_path / "calib.yaml"
    mp.write_text(MAP_YAML)
    cp.write_text(CALIB_YAML)

    def make(**kw):
        kw.setdefault("roi_rescue", False)      # detection is scripted here
        p = AprilTagLocalizationProvider(
            tag_map_path=str(mp), camera_calib_path=str(cp), **kw)
        p._detector = FakeDetector()
        return p

    return make


def fly(prov, pos, yaw, t, vx=0.0, wz=0.0, blind=False):
    """One tick: command, (maybe) see the tags, update."""
    prov.set_command(vx, 0.0, wz, t)
    prov._detector.next_dets = [] if blind else detections_from(pos, yaw)
    return prov.update(Observation(rgb=RGBFrame(image=GRAY, stamp_sec=t)))


def teach_moving(prov, n=25, vx=0.25):
    """Advance with commands matched by motion, so effectiveness learns up."""
    pos, t = np.array([0.5, -0.5, 1.0]), 0.0
    for _ in range(n):
        est = fly(prov, pos, 0.0, t, vx=vx)
        assert est is not None
        pos[0] += vx * DT
        t += DT
    return pos, t


def test_blind_frames_coast_then_go_silent(provider):
    prov = provider(coast_frames=3)
    pos, t = teach_moving(prov)
    outs = [fly(prov, pos, 0.0, t + i * DT, vx=0.25, blind=True) for i in range(6)]
    assert [o is not None for o in outs] == [True, True, True, False, False, False]


def test_coast_is_unmistakable_for_a_fix(provider):
    """source says coast, confidence collapses and stays far below a real fix."""
    prov = provider(coast_frames=4)
    pos, t = teach_moving(prov)
    confs = []
    for i in range(4):
        est = fly(prov, pos, 0.0, t + i * DT, vx=0.25, blind=True)
        assert est.source == "apriltag_coast"
        confs.append(est.confidence)
    assert all(c <= 0.25 for c in confs)
    assert confs == sorted(confs, reverse=True)          # strictly collapsing
    assert confs[-1] < 0.5 * confs[0]


def test_coast_follows_proven_commands(provider):
    """Effectiveness ~1 after real motion: the coasted pose keeps advancing."""
    prov = provider(coast_frames=5)
    pos, t = teach_moving(prov)
    x0 = None
    est = None
    for i in range(5):
        est = fly(prov, pos, 0.0, t + i * DT, vx=0.25, blind=True)
        if x0 is None:
            x0 = est.pose.x
    advanced = est.pose.x - x0
    # 4 further coast frames of 0.25 m/s at trust<=0.7: clearly forward, bounded.
    assert 0.03 < advanced < 0.10


def test_stuck_drone_coasts_in_place(provider):
    """Commands that provably do nothing must not move the coasted pose."""
    prov = provider(coast_frames=5)
    pos, t = teach_moving(prov, n=10)
    for i in range(15):                                   # commanded, not moving
        fly(prov, pos, 0.0, t, vx=0.25)
        t += DT
    est0 = None
    est = None
    for i in range(5):
        est = fly(prov, pos, 0.0, t + i * DT, vx=0.25, blind=True)
        est0 = est0 or est
    assert abs(est.pose.x - est0.pose.x) < 0.02
    assert est.cmd_effectiveness < 0.15


def test_reacquire_resets_the_budget(provider):
    prov = provider(coast_frames=2)
    pos, t = teach_moving(prov)
    assert fly(prov, pos, 0.0, t + 0 * DT, blind=True) is not None
    assert fly(prov, pos, 0.0, t + 1 * DT, blind=True) is not None
    assert fly(prov, pos, 0.0, t + 2 * DT, blind=True) is None
    assert fly(prov, pos, 0.0, t + 3 * DT) is not None            # tags back
    assert fly(prov, pos, 0.0, t + 4 * DT, blind=True) is not None  # budget fresh


def test_coast_zero_restores_silence(provider):
    prov = provider(coast_frames=0)
    pos, t = teach_moving(prov)
    assert fly(prov, pos, 0.0, t, vx=0.25, blind=True) is None


def test_no_command_feed_means_no_coast(provider):
    """cmd_trust_max=0: a coast would repeat the pose, not inform -- stay silent."""
    prov = provider(coast_frames=5, cmd_trust_max=0.0)
    pos, t = teach_moving(prov)
    assert fly(prov, pos, 0.0, t, blind=True) is None


def test_never_coasts_before_the_first_fix(provider):
    prov = provider(coast_frames=5)
    assert fly(prov, np.array([0.5, -0.5, 1.0]), 0.0, 0.0, blind=True) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
