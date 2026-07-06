"""Tests for the graded-pulse / mid-burst-feedback / anti-deadlock yaw upgrades.

These exercise the new params (all default OFF, so the existing suites prove the
change is inert). Operating point: 10 Hz (DT=0.1) => 0.7 rad/s = 4 deg/tick.
"""
import math

from sparx_agency.core.common.types import Pose2D, normalize_angle
from sparx_agency.core.planning.trackers.waypoint_follower import (
    FollowerState, WaypointFollower, WaypointFollowerParams)
from sparx_agency.core.planning.trackers.waypoint_follower import algorithm as alg

DT = 0.1   # 10 Hz operating point


def _graded(**over):
    base = dict(yaw_graded_pulses=True, yaw_burst_grade_max_ticks=6,
                yaw_settle_dwell_per_tick=0.1)
    base.update(over)
    return WaypointFollowerParams(**base)


def _roll(follower, start, path, steps, *, dt=DT, yaw_gain=1.0, fed_noise=None):
    """Closed-loop unicycle rollout. yaw_gain>1 makes the plant rotate FASTER than
    commanded (forces mid-burst overshoot); fed_noise[i] perturbs only the yaw the
    follower SEES (not the true plant yaw), to test noise rejection."""
    follower.set_path([Pose2D(*p) for p in path], start)
    x, y, yaw = start.x, start.y, start.yaw
    recs = []
    for i in range(steps):
        seen = yaw + (fed_noise[i] if fed_noise and i < len(fed_noise) else 0.0)
        cmd = follower.step(Pose2D(x, y, seen), dt)
        recs.append({"state": follower.state, "wz": cmd.wz, "vx": cmd.vx,
                     "freeze": cmd.freeze, "yaw": yaw})
        yaw = normalize_angle(yaw + cmd.wz * dt * yaw_gain)
        x += cmd.vx * math.cos(yaw) * dt
        y += cmd.vx * math.sin(yaw) * dt
        if follower.done:
            break
    return recs


def _turn_runs(recs, eps=1e-6):
    """Lengths of maximal consecutive BURST runs (one per burst).

    Counts only YAW_ALIGN ticks that command yaw — i.e. the actual burst budget,
    excluding the YAW_SETTLE coast-down (where wz slews to zero) which would
    otherwise inflate the count by a tick or two.
    """
    runs, cur = [], 0
    for r in recs:
        if r["state"] == FollowerState.YAW_ALIGN and abs(r["wz"]) > eps:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs


# --------------------------------------------------------------------------
# helper math
# --------------------------------------------------------------------------
def test_burst_tick_count_grades_to_2_4_6():
    ta = 0.7 * DT                       # 4 deg/tick
    coast = math.radians(15.0)
    big = alg.burst_tick_count(math.radians(85), coast, ta, 2, 6)
    mid = alg.burst_tick_count(math.radians(33), coast, ta, 2, 6)
    small = alg.burst_tick_count(math.radians(20), coast, ta, 2, 6)
    assert big == 6 and mid == 4 and small == 2
    assert alg.burst_tick_count(math.radians(5), coast, ta, 2, 6) == 2   # floor


def test_settle_dwell_scales_with_ticks():
    assert abs(alg.settle_dwell(0.8, 0.1, 2, 6) - 1.0) < 1e-9
    assert abs(alg.settle_dwell(0.8, 0.1, 6, 6) - 1.4) < 1e-9
    assert abs(alg.settle_dwell(0.8, 0.1, 10, 6) - 1.4) < 1e-9   # capped at 6
    assert alg.settle_dwell(0.8, 0.0, 6, 6) == 0.8              # per_tick 0 = fixed


def test_accept_with_reversals_widens_and_locks():
    a0, l0 = alg.accept_with_reversals(0.20, 0, math.radians(4), 3)
    a2, l2 = alg.accept_with_reversals(0.20, 2, math.radians(4), 3)
    a3, l3 = alg.accept_with_reversals(0.20, 3, math.radians(4), 3)
    assert a2 > a0 and not l0 and not l2 and l3
    assert alg.accept_with_reversals(0.20, 9, 0.0, 0)[1] is False   # 0 = lock off


# --------------------------------------------------------------------------
# graded bursts + dwell
# --------------------------------------------------------------------------
def test_graded_bursts_are_capped_and_converge():
    f = WaypointFollower(_graded())
    recs = _roll(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)], 400)   # 90 deg left
    runs = _turn_runs(recs)
    assert runs, "expected at least one yaw burst"
    assert all(n <= 6 for n in runs), runs                      # hard 6-tick cap
    assert runs[0] == 6                                         # big turn -> strong
    states = [r["state"] for r in recs]
    assert FollowerState.ADVANCE in states                     # it converges


def test_inertia_proportional_dwell_longer_after_bigger_burst():
    # First settle after a 6-tick burst should dwell longer than the fixed base.
    f = WaypointFollower(_graded(yaw_settle_dwell_per_tick=0.1))
    recs = _roll(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)], 400)
    # count consecutive unfrozen (freeze False) stop ticks in the FIRST settle
    in_settle = False
    dwell_ticks = 0
    for r in recs:
        if r["state"] == FollowerState.YAW_SETTLE and r["freeze"] is False:
            in_settle = True
            dwell_ticks += 1
        elif in_settle:
            break
    # base 0.8s + 0.1*6 = 1.4s at DT=0.1 => ~14 unfrozen ticks; comfortably > the
    # ~8 a fixed 0.8s base (8 ticks) would give.
    assert dwell_ticks >= 11, dwell_ticks


# --------------------------------------------------------------------------
# mid-burst live feedback
# --------------------------------------------------------------------------
def test_midburst_cut_on_confirmed_overshoot():
    # 50 deg target; the plant rotates 5x commanded, so the live heading reaches
    # the target inside the 6-tick budget and the burst must CUT to YAW_SETTLE
    # early (one-way: never an opposite-sign command in the same run).
    e = math.radians(50.0)
    wp = (6 * math.cos(e), 6 * math.sin(e))
    f = WaypointFollower(_graded(yaw_burst_live_feedback=True, yaw_fb_confirm_ticks=2))
    recs = _roll(f, Pose2D(0, 0, 0.0), [(0, 0), wp], 60, yaw_gain=5.0)
    first = _turn_runs(recs)[0]
    assert first < 6, first                       # cut before the full 6-tick budget
    # after the cut it must be settling, not commanding the opposite direction
    yaw_signs = [1 if r["wz"] > 1e-6 else (-1 if r["wz"] < -1e-6 else 0)
                 for r in recs if r["state"] == FollowerState.YAW_ALIGN and abs(r["wz"]) > 1e-6]
    assert all(s == yaw_signs[0] for s in yaw_signs[:first])   # one-way burst


def test_midburst_single_noise_frame_does_not_cut():
    # yaw_gain=1 (no real overshoot) but one fed-noise frame momentarily reads
    # past the target. With confirm=2 the burst must NOT cut on that single frame.
    noise = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # one wild frame at tick 2
    f = WaypointFollower(_graded(yaw_burst_live_feedback=True, yaw_fb_confirm_ticks=2,
                                 yaw_burst_grade_max_ticks=6))
    recs = _roll(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)], 12, fed_noise=noise)
    first = _turn_runs(recs)[0]
    assert first >= 4, first      # single noisy frame did not chop the burst short


# --------------------------------------------------------------------------
# anti-deadlock
# --------------------------------------------------------------------------
def test_no_ping_pong_on_10deg_error():
    # 10 deg error at a FAR waypoint: inside the 11.5 deg accept floor at 10 Hz,
    # so it advances on the first tick WITHOUT ever bursting -> no oscillation.
    err = math.radians(10.0)
    f = WaypointFollower(_graded(yaw_max_reversals=3, yaw_accept_growth_rad=math.radians(4)))
    recs = _roll(f, Pose2D(0, 0, 0.0),
                 [(0, 0), (6 * math.cos(err), 6 * math.sin(err))], 200)
    assert all(abs(r["wz"]) < 1e-9 for r in recs)            # never commanded yaw
    assert FollowerState.ADVANCE in [r["state"] for r in recs]


def test_anti_deadlock_locks_to_advance():
    # A plant with huge coast (yaw_gain=6) overshoots every burst, so the residual
    # keeps flipping sign. The reversal lock must force ADVANCE within
    # yaw_max_reversals and never loop forever.
    f = WaypointFollower(_graded(yaw_max_reversals=3,
                                 yaw_accept_growth_rad=math.radians(4)))
    recs = _roll(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)], 500, yaw_gain=6.0)
    assert FollowerState.ADVANCE in [r["state"] for r in recs] or recs[-1]["state"] == FollowerState.DONE
    assert f._reversals <= 3                                  # bounded, never runaway
