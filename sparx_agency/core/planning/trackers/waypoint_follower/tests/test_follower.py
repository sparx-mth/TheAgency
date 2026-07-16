"""Unit tests for the ROS-free waypoint follower state machine."""
import math

from sparx_agency.core.common.types import Pose2D, normalize_angle
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


# ── Retiring waypoints against the whole route ──────────────────────
def test_overshot_waypoint_does_not_make_the_drone_turn_around():
    """Having sailed past a waypoint, aim at the next leg -- never turn around.

    The ADVANCE gate is predictive, so a waypoint is routinely retired without
    the drone ever entering pos_radius of it, and on a cut corner the drone can
    end up level with (or past) the one after it too. Stepping the target index
    by one then hands it a point BEHIND it: the follower brakes and rotates all
    the way back to fly to a waypoint it has already been to.

    Drives the real state machine with the position held fixed -- the only
    question is which way it decides to point. Aiming at the passed waypoint
    swings it ~112 deg off course; aiming at the leg ahead costs ~20 deg.
    """
    path = [Pose2D(0, 0), Pose2D(1.5, 0), Pose2D(3.0, 0), Pose2D(4.5, 0)]
    f = WaypointFollower(WaypointFollowerParams(pos_radius=0.35))
    f.set_path(path, Pose2D(0.0, 0.0, 0.0))

    x, y, yaw = 3.2, 0.5, 0.0          # flown past w1, already level with w2
    worst = 0.0
    for _ in range(60):
        cmd = f.step(Pose2D(x, y, yaw), DT)
        yaw = normalize_angle(yaw + cmd.wz * DT)
        worst = max(worst, abs(math.degrees(yaw)))
    assert worst < 60.0, (
        "swung %.0f deg off course -- the drone turned back toward a waypoint it "
        "had already passed instead of flying on to the next leg" % worst)


def test_live_index_never_walks_backwards():
    """The target only ever moves forward, whatever the pose says."""
    pts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0), (4.5, 0.0)]
    back_at_the_start = Pose2D(0.1, 0.0, 0.0)
    assert alg.live_waypoint_index(pts, back_at_the_start, 0.35, 3) == 3


def test_live_index_keeps_a_waypoint_that_is_still_ahead():
    """Do not skip a point merely because the heading is off -- only if it is
    genuinely behind on the route. A drone facing the wrong way mid-turn must
    still fly to the waypoint in front of it."""
    pts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    facing_backwards = Pose2D(0.2, 0.0, math.pi)   # on leg 0, pointed back
    assert alg.live_waypoint_index(pts, facing_backwards, 0.35, 1) == 1


def test_live_index_skips_a_waypoint_already_stood_on():
    pts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    on_top_of_w1 = Pose2D(1.5, 0.05, 0.0)
    assert alg.live_waypoint_index(pts, on_top_of_w1, 0.35, 1) == 2


def test_live_index_reports_the_route_is_finished():
    pts = [(0.0, 0.0), (1.5, 0.0)]
    past_the_end = Pose2D(2.0, 0.0, 0.0)
    assert alg.live_waypoint_index(pts, past_the_end, 0.35, 2) == 2   # == len(pts)


def test_gate_wider_than_the_capture_radius_is_refused():
    """A gate that overshoots pos_radius means waypoints are never acquired.

    Both numbers look reasonable alone; only the pair is wrong, and the symptom
    in the air (the drone stops reaching waypoints and retires them 100 deg late)
    points at neither. So it is refused at construction rather than flown.
    """
    import pytest
    with pytest.raises(ValueError, match="must be < pos_radius"):
        WaypointFollowerParams(yaw_capture_tol_m=0.51, pos_radius=0.30)
    WaypointFollowerParams(yaw_capture_tol_m=0.20, pos_radius=0.30)   # the real pair
