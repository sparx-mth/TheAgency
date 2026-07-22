"""Tests for RollAssistFollower: navigation is untouched, ROLL is layered on."""
import math

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.trackers.waypoint_follower import (
    WaypointFollower,
    WaypointFollowerParams,
    FollowerState,
)
from sparx_agency.core.planning.trackers.roll_assist_follower import (
    RollAssistFollower,
)

# Straight corridor along +x with several waypoints so a leg survives re-anchoring.
PATH = [Pose2D(0.0, 0.0), Pose2D(2.0, 0.0), Pose2D(4.0, 0.0), Pose2D(6.0, 0.0)]


def _wrap(**base_kw):
    base = WaypointFollower(WaypointFollowerParams(**base_kw))
    return RollAssistFollower(base), base


def test_delegates_navigation_surface():
    foll, base = _wrap()
    foll.set_path(PATH, Pose2D(0.0, 0.0, 0.0))
    assert foll.state == base.state
    assert foll.required_axis() == base.required_axis()
    assert foll.done == base.done
    assert foll.settle_map_updates_required == base.settle_map_updates_required
    assert foll.params is base.params


def test_advance_adds_lateral_correction_only():
    # forward_only base -> straight into ADVANCE; drone drifted 0.3 m left (+y).
    foll, _ = _wrap(forward_only=True)
    pose = Pose2D(1.0, 0.3, 0.0)
    foll.set_path(PATH, pose)
    cmd = None
    for _ in range(8):
        cmd = foll.step(pose, 0.1)
    assert cmd.state == FollowerState.ADVANCE
    assert cmd.vx > 0.0                     # base still drives forward
    assert cmd.vy < 0.0                     # ROLL pulls back toward the line (right)
    assert abs(cmd.wz) < 1e-9               # no yaw introduced


def test_hold_gate_suppresses_correction():
    foll, _ = _wrap(forward_only=True)
    pose = Pose2D(1.0, 0.3, 0.0)
    foll.set_path(PATH, pose)
    cmd = foll.step(pose, 0.1, hold=True)
    assert cmd.vx == 0.0 and cmd.vy == 0.0 and cmd.wz == 0.0


def test_unconfirmed_axis_suppresses_correction():
    # Base needs FORWARD confirmed; unconfirmed -> gated -> no ROLL injected.
    foll, base = _wrap(forward_only=True)
    pose = Pose2D(1.0, 0.3, 0.0)
    foll.set_path(PATH, pose)
    assert base.required_axis() == foll.required_axis()   # FORWARD
    cmd = foll.step(pose, 0.1, axis_confirmed=False)
    assert cmd.vy == 0.0 and cmd.vx == 0.0


def test_navigation_stream_matches_bare_follower():
    # The ROLL layer must never change the base's yaw or state decisions. Drive a
    # wrapped follower and a bare one through the SAME pose sequence and compare.
    wrapped, _ = _wrap()
    bare = WaypointFollower(WaypointFollowerParams())
    start = Pose2D(0.0, 0.0, 0.0)
    wrapped.set_path(PATH, start)
    bare.set_path(PATH, start)
    # A pose sequence that needs a turn (goal off to the side) then holds.
    poses = [Pose2D(0.0, 0.0, math.radians(a)) for a in (0, 5, 10, 15, 20, 25, 30)]
    for pose in poses:
        wc = wrapped.step(pose, 0.1)
        bc = bare.step(pose, 0.1)
        assert wc.state == bc.state
        assert abs(wc.wz - bc.wz) < 1e-9    # yaw identical (pure base)
        assert wc.wp_idx == bc.wp_idx


def test_reset_clears_both_layers():
    foll, _ = _wrap(forward_only=True)
    pose = Pose2D(1.0, 0.4, 0.0)
    foll.set_path(PATH, pose)
    for _ in range(6):
        foll.step(pose, 0.1)
    foll.reset()
    # After reset the corrector memory is clear: the very next step ramps vy from 0
    # (bounded by one accel step), not from the previously-built-up value.
    foll.set_path(PATH, pose)
    cmd = foll.step(pose, 0.1)
    assert abs(cmd.vy) <= 0.1 + 1e-9        # <= accel_limit(1.0) * dt(0.1)
