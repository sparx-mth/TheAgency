"""Unit tests for the ROS-free waypoint follower state machine."""
import math

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.trackers.waypoint_follower import (
    ControlAxis,
    FollowerState,
    WaypointFollower,
    WaypointFollowerParams,
)
from sparx_agency.core.planning.trackers.waypoint_follower import algorithm as alg

DT = 0.2


def _drive(follower, pose_fn, steps, **step_kw):
    """Step the follower `steps` times, feeding pose_fn(i) each tick."""
    last = None
    for i in range(steps):
        last = follower.step(pose_fn(i), DT, **step_kw)
    return last


def test_platform_invariant_never_both_axes():
    f = WaypointFollower()
    f.set_path([Pose2D(0, 0), Pose2D(5, 5)], Pose2D(0, 0, 0.0))
    for i in range(60):
        cmd = f.step(Pose2D(0.0, 0.0, 0.0), DT)
        assert cmd.command.y == 0.0 and cmd.command.z == 0.0
        assert abs(cmd.vx) < 1e-6 or abs(cmd.wz) < 1e-6


def test_yaw_align_then_advance_for_axis_target():
    # Target straight ahead (+x); aligned -> should advance, not rotate.
    f = WaypointFollower(WaypointFollowerParams(forward_only=False))
    f.set_path([Pose2D(0, 0), Pose2D(10, 0)], Pose2D(0, 0, 0.0))
    assert f.state == FollowerState.YAW_ALIGN
    cmd = f.step(Pose2D(0.0, 0.0, 0.0), DT)
    # Already aligned -> immediately satisfied -> ADVANCE.
    assert f.state == FollowerState.ADVANCE
    cmd = f.step(Pose2D(0.0, 0.0, 0.0), DT)
    assert cmd.vx > 0.0 and abs(cmd.wz) < 1e-6


def test_yaw_align_rotates_toward_offaxis_target():
    f = WaypointFollower(WaypointFollowerParams(yaw_lead_pct=0.0))
    # Target at +90 deg; robot facing +x. Must rotate (positive wz).
    f.set_path([Pose2D(0, 0), Pose2D(0, 10)], Pose2D(0, 0, 0.0))
    cmd = f.step(Pose2D(0.0, 0.0, 0.0), DT)
    assert f.state == FollowerState.YAW_ALIGN
    assert cmd.wz > 0.0 and cmd.vx == 0.0
    assert cmd.required_axis == ControlAxis.YAW


def test_axis_not_confirmed_holds_zero_and_does_not_transition():
    f = WaypointFollower()
    f.set_path([Pose2D(0, 0), Pose2D(0, 10)], Pose2D(0, 0, 0.0))
    for _ in range(20):
        cmd = f.step(Pose2D(0.0, 0.0, 0.0), DT, axis_confirmed=False)
    assert f.state == FollowerState.YAW_ALIGN  # never advanced
    assert cmd.vx == 0.0 and cmd.wz == 0.0


def test_hold_suppresses_motion():
    f = WaypointFollower()
    f.set_path([Pose2D(0, 0), Pose2D(10, 0)], Pose2D(0, 0, 0.0))
    cmd = f.step(Pose2D(0.0, 0.0, 0.0), DT, hold=True)
    assert cmd.vx == 0.0 and cmd.wz == 0.0
    assert f.state == FollowerState.YAW_ALIGN


def test_forward_only_skips_yaw_align():
    f = WaypointFollower(WaypointFollowerParams(forward_only=True))
    f.set_path([Pose2D(0, 0), Pose2D(0, 10)], Pose2D(0, 0, 0.0))
    assert f.state == FollowerState.ADVANCE


def test_reaches_goal_along_straight_path():
    f = WaypointFollower()
    goal = 4.0
    f.set_path([Pose2D(0, 0), Pose2D(goal, 0)], Pose2D(0, 0, 0.0))
    # Simulate the drone advancing in +x at a fixed rate while facing +x.
    x = 0.0
    state_seen = set()
    for _ in range(400):
        cmd = f.step(Pose2D(x, 0.0, 0.0), DT)
        x += max(cmd.vx, 0.0) * DT
        state_seen.add(f.state)
        if f.done:
            break
    assert f.done
    assert FollowerState.ADVANCE in state_seen
    assert FollowerState.BRAKE in state_seen


def test_reanchor_drops_passed_waypoints():
    pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    out = alg.reanchor_path(pts, Pose2D(1.5, 0.0, 0.0), pos_radius=0.35)
    # Closest segment is index 1 (1->2); keep from index 2 onward.
    assert out[0] == (2.0, 0.0)
    assert out[-1] == (3.0, 0.0)


def test_reanchor_keeps_last_when_all_passed():
    pts = [(0.0, 0.0), (1.0, 0.0)]
    out = alg.reanchor_path(pts, Pose2D(5.0, 0.0, 0.0), pos_radius=0.35)
    assert out == [(1.0, 0.0)]


def test_slew_and_saturate():
    assert alg.saturate(5.0, 1.25) == 1.25
    assert alg.saturate(-5.0, 1.25) == -1.25
    assert alg.slew(1.0, 0.0, 0.3) == 0.3
    assert math.isclose(alg.slew(0.1, 0.0, 0.3), 0.1)
