"""Unit tests for the pulse -> settle -> re-measure yaw behaviour.

These exercise the YAW_SETTLE state, the open-loop burst, the gentle predictive
ADVANCE gate and convergence (no yaw ping-pong). Run without pytest via the
project venv by importing this module and calling the ``test_*`` functions.
"""
import math

from sparx_agency.core.common.types import Pose2D, normalize_angle
from sparx_agency.core.planning.trackers.waypoint_follower import (
    FollowerState,
    WaypointFollower,
    WaypointFollowerParams,
)

DT = 0.2


def _simulate(follower, start, path, steps, *, turn=True, drive=True):
    """Closed-loop rollout against a simple instantaneous unicycle plant.

    Integrates the commanded yaw rate / forward speed straight into the pose
    (no inertia), which is enough to test the state machine. Returns the list of
    per-tick records and the final pose.
    """
    follower.set_path([Pose2D(*p) for p in path], start)
    x, y, yaw = start.x, start.y, start.yaw
    recs = []
    for _ in range(steps):
        pose = Pose2D(x, y, yaw)
        cmd = follower.step(pose, DT)
        recs.append({
            "state": follower.state, "freeze": cmd.freeze, "vx": cmd.vx,
            "wz": cmd.wz, "axis": cmd.required_axis, "pose": pose,
        })
        if turn:
            yaw = normalize_angle(yaw + cmd.wz * DT)
        if drive:
            x += cmd.vx * math.cos(yaw) * DT
            y += cmd.vx * math.sin(yaw) * DT
        if follower.done:
            break
    return recs, Pose2D(x, y, yaw)


def _states(recs):
    return [r["state"] for r in recs]


def _motion_runs(recs, eps=1e-6):
    """Maximal runs of the same motion label: F=forward, T=turn, S=stop."""
    labels = []
    for r in recs:
        if abs(r["vx"]) > eps:
            labels.append("F")
        elif abs(r["wz"]) > eps:
            labels.append("T")
        else:
            labels.append("S")
    runs = []
    for lab in labels:
        if runs and runs[-1][0] == lab:
            runs[-1][1] += 1
        else:
            runs.append([lab, 1])
    return runs


def test_no_extra_yaw_when_aligned():
    """A target dead ahead never enters a burst or YAW_SETTLE."""
    f = WaypointFollower()
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), (5, 0)], 40)
    seen = set(_states(recs))
    assert FollowerState.YAW_SETTLE not in seen
    # No tick ever commanded a yaw rate.
    assert all(abs(r["wz"]) < 1e-9 for r in recs)


def test_burst_then_settle_then_advance_sequence():
    """Off-axis target: rotate, settle, then advance (bounded bursts)."""
    f = WaypointFollower()
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)], 200)
    states = _states(recs)
    assert FollowerState.YAW_ALIGN in states
    assert FollowerState.YAW_SETTLE in states
    assert FollowerState.ADVANCE in states
    # The first YAW_SETTLE must come before the first ADVANCE.
    assert states.index(FollowerState.YAW_SETTLE) < states.index(FollowerState.ADVANCE)
    # A 90 deg turn is split into ~25 deg increments (yaw_burst_max_rad), so it
    # takes several bursts, each ending by entering YAW_SETTLE (stop + voxel
    # update + re-measure). Expect more than one, but a bounded number.
    bursts = sum(1 for a, b in zip(states[:-1], states[1:])
                 if a == FollowerState.YAW_ALIGN and b == FollowerState.YAW_SETTLE)
    assert 2 <= bursts <= 8, bursts


def test_open_loop_burst_ignores_mid_rotation_pose():
    """During a burst the commanded yaw rate ignores the (noisy) live pose."""
    f = WaypointFollower(WaypointFollowerParams(yaw_burst_max_ticks=50))
    f.set_path([Pose2D(0, 0), Pose2D(0, 10)], Pose2D(0, 0, 0.0))
    # First tick decides + starts the burst.
    c0 = f.step(Pose2D(0, 0, 0.0), DT)
    assert f.state == FollowerState.YAW_ALIGN and c0.wz > 0
    # Feed deliberately garbage yaws; the burst must keep commanding +yaw_rate.
    garbage = [3.0, -3.0, 1.0, -2.0, 0.5]
    for g in garbage:
        c = f.step(Pose2D(5.0, -5.0, g), DT)
        if f.state != FollowerState.YAW_ALIGN:
            break
        assert abs(c.wz - 0.7) < 1e-6, (g, c.wz)


def test_yaw_settle_coast_freezes_then_dwell_unfreezes():
    """YAW_SETTLE freezes while coasting (|wz|>=eps) then unfreezes to dwell;
    required_axis is None throughout, and it eventually leaves to YAW_ALIGN."""
    # Small yaw accel => the command takes several ticks to slew to zero, so the
    # coast sub-phase spans multiple ticks and is observable.
    p = WaypointFollowerParams(yaw_accel_limit=1.0, yaw_settle_dwell_s=0.6)
    f = WaypointFollower(p)
    f.set_path([Pose2D(0, 0), Pose2D(0, 6)], Pose2D(0, 0, 0.0))
    # Drive (with turning) until we first reach YAW_SETTLE.
    yaw = 0.0
    for _ in range(80):
        c = f.step(Pose2D(0, 0, yaw), DT)
        yaw = normalize_angle(yaw + c.wz * DT)
        if f.state == FollowerState.YAW_SETTLE:
            break
    assert f.state == FollowerState.YAW_SETTLE
    # Hold the pose fixed (drone stopped) and watch the settle play out.
    saw_coast_freeze = saw_dwell_unfreeze = left_settle = False
    for _ in range(20):
        c = f.step(Pose2D(0.0, 0.0, yaw), DT)
        if f.state == FollowerState.YAW_SETTLE:
            assert c.required_axis is None
            if abs(f._last_wz) >= p.yaw_settle_eps:
                assert c.freeze is True
                saw_coast_freeze = True
            else:
                assert c.freeze is False
                saw_dwell_unfreeze = True
        else:
            left_settle = True
            break
    assert saw_coast_freeze and saw_dwell_unfreeze and left_settle


def test_yaw_settle_blocks_until_map_ready():
    """YAW_SETTLE will not start the next rotation until map_ready is True, even
    after the dwell elapses -- so the drone never moves on stale, pre-stop voxels."""
    f = WaypointFollower()
    f.set_path([Pose2D(0, 0), Pose2D(0, 6)], Pose2D(0, 0, 0.0))
    yaw = 0.0
    for _ in range(80):
        c = f.step(Pose2D(0, 0, yaw), DT)
        yaw = normalize_angle(yaw + c.wz * DT)
        if f.state == FollowerState.YAW_SETTLE:
            break
    assert f.state == FollowerState.YAW_SETTLE
    # Dwell far past yaw_settle_dwell_s with no map update: must stay settling.
    for _ in range(40):
        f.step(Pose2D(0, 0, yaw), DT, map_ready=False)
    assert f.state == FollowerState.YAW_SETTLE
    # Once the map update lands, it may continue.
    left = False
    for _ in range(10):
        f.step(Pose2D(0, 0, yaw), DT, map_ready=True)
        if f.state != FollowerState.YAW_SETTLE:
            left = True
            break
    assert left


def test_gentle_gate_advances_close_enough():
    """The 70deg-target / 58deg-heading case: ~12deg error advances immediately."""
    f = WaypointFollower()
    err = math.radians(12.0)
    wp = (2.0 * math.cos(err), 2.0 * math.sin(err))
    f.set_path([Pose2D(0, 0), Pose2D(*wp)], Pose2D(0, 0, 0.0))
    c = f.step(Pose2D(0, 0, 0.0), DT)
    assert f.state == FollowerState.ADVANCE
    assert abs(c.wz) < 1e-9


def test_gate_far_needs_alignment_near_is_loose():
    """Same heading error: a far waypoint must align; a near one may advance."""
    err = math.radians(20.0)
    # Far: cross-track miss = 5*sin(20) ~ 1.7m >> capture_tol -> must rotate.
    far = WaypointFollower()
    far.set_path([Pose2D(0, 0), Pose2D(5 * math.cos(err), 5 * math.sin(err))],
                 Pose2D(0, 0, 0.0))
    cf = far.step(Pose2D(0, 0, 0.0), DT)
    assert far.state == FollowerState.YAW_ALIGN and abs(cf.wz) > 0
    # Near: cross-track miss = 0.5*sin(20) ~ 0.17m < capture_tol -> advance.
    near = WaypointFollower()
    near.set_path([Pose2D(0, 0), Pose2D(0.5 * math.cos(err), 0.5 * math.sin(err))],
                  Pose2D(0, 0, 0.0))
    near.step(Pose2D(0, 0, 0.0), DT)
    assert near.state == FollowerState.ADVANCE


def test_convergence_no_oscillation():
    """Starting just past the accept floor converges to ADVANCE, never loops."""
    f = WaypointFollower()
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0),
                        [(0, 0), (3.0 * math.cos(0.35), 3.0 * math.sin(0.35))], 300)
    assert FollowerState.ADVANCE in _states(recs)


def test_set_path_mid_burst_enters_settle():
    """A new path arriving mid-rotation settles first instead of re-bursting."""
    f = WaypointFollower()
    f.set_path([Pose2D(0, 0), Pose2D(0, 10)], Pose2D(0, 0, 0.0))
    for _ in range(3):
        f.step(Pose2D(0, 0, 0.0), DT)        # ramp into the burst
    assert abs(f._last_wz) > 0.1
    f.set_path([Pose2D(0, 0), Pose2D(10, 0)], Pose2D(0, 0, 1.0))
    assert f.state == FollowerState.YAW_SETTLE


def test_no_lone_motion_command_turn_or_forward():
    """Every motion run (forward OR turn) is >= min_motion_ticks: a lone 5 Hz
    pulse can't overcome the deadband, so the follower never emits just one."""
    f = WaypointFollower()
    minc = f.params.min_motion_ticks
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0),
                        [(0, 0), (2, 0), (2, 2), (0, 2)], 600)
    bad = [(lab, n) for lab, n in _motion_runs(recs) if lab in ("F", "T") and n < minc]
    assert not bad, bad


def test_forward_never_lone_tick_near_waypoint():
    """A waypoint just past the capture radius (one tick would reach it) still
    gets at least two forward commands, not a single wasted pulse."""
    f = WaypointFollower()
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), (0.40, 0)], 40)
    fwd = [n for lab, n in _motion_runs(recs) if lab == "F"]
    assert fwd and all(n >= f.params.min_motion_ticks for n in fwd), _motion_runs(recs)


def test_forward_only_never_settles():
    """forward_only takes corners via BRAKE->ADVANCE, never YAW_SETTLE/burst."""
    f = WaypointFollower(WaypointFollowerParams(forward_only=True))
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0),
                        [(0, 0), (4, 0), (4, 4)], 300)
    seen = set(_states(recs))
    assert FollowerState.YAW_SETTLE not in seen
    assert all(abs(r["wz"]) < 1e-9 for r in recs)
