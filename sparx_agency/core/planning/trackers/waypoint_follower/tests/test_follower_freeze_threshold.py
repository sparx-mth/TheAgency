"""Tests for the angle-gated map freeze.

A turn larger than ``freeze_yaw_thresh_rad`` freezes the map for the whole
burst-and-coast and then re-observes the scene ``settle_map_updates`` times while
stopped before moving on; a smaller heading correction stays live (map keeps
updating, no stationary re-observation forced). The freeze decision is latched
per alignment episode and clears on reaching ADVANCE.

Run without pytest via the project venv by importing this module and calling the
``test_*`` functions.
"""
import math

from sparx_agency.core.common.types import Pose2D, normalize_angle
from sparx_agency.core.planning.trackers.waypoint_follower import (
    FollowerState,
    WaypointFollower,
    WaypointFollowerParams,
)

DT = 0.2


def _roll_until_advance(follower, start, path, steps=400, *, map_ready=True):
    """Closed-loop unicycle rollout; records each tick until ADVANCE/DONE."""
    follower.set_path([Pose2D(*p) for p in path], start)
    x, y, yaw = start.x, start.y, start.yaw
    recs = []
    for _ in range(steps):
        pose = Pose2D(x, y, yaw)
        cmd = follower.step(pose, DT, map_ready=map_ready)
        recs.append({
            "state": follower.state, "freeze": cmd.freeze, "wz": cmd.wz,
            "vx": cmd.vx, "need": follower.settle_map_updates_required,
        })
        yaw = normalize_angle(yaw + cmd.wz * DT)
        x += cmd.vx * math.cos(yaw) * DT
        y += cmd.vx * math.sin(yaw) * DT
        if follower.state == FollowerState.ADVANCE or follower.done:
            break
    return recs


def _burst_ticks(recs):
    """Ticks that actually commanded a yaw (the frozen/unfrozen burst body)."""
    return [r for r in recs if r["state"] == FollowerState.YAW_ALIGN
            and abs(r["wz"]) > 1e-6]


def _settle_ticks(recs):
    return [r for r in recs if r["state"] == FollowerState.YAW_SETTLE]


# --------------------------------------------------------------------------
# below threshold -> live
# --------------------------------------------------------------------------
def test_turn_below_threshold_stays_live():
    """A turn under freeze_yaw_thresh_rad never freezes and forces no re-observe.

    Uses a high threshold so the well-exercised 90 deg geometry counts as a
    'small correction' -- the same bursts, but live."""
    p = WaypointFollowerParams(freeze_yaw_thresh_rad=math.radians(120.0))
    f = WaypointFollower(p)
    recs = _roll_until_advance(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)])
    bursts = _burst_ticks(recs)
    assert bursts, "expected at least one yaw burst"
    assert all(r["freeze"] is False for r in bursts)          # never froze
    settles = _settle_ticks(recs)
    assert settles, "expected a YAW_SETTLE"
    assert all(r["freeze"] is False for r in settles)         # coast not frozen
    assert all(r["need"] == 0 for r in settles)               # no re-observe forced


# --------------------------------------------------------------------------
# above threshold -> frozen + re-observe
# --------------------------------------------------------------------------
def test_turn_above_threshold_freezes_and_reobserves():
    """A turn past the default 20 deg threshold freezes the burst + coast and asks
    for settle_map_updates fresh re-observations in YAW_SETTLE."""
    f = WaypointFollower()   # default freeze_yaw_thresh_rad = 20 deg
    recs = _roll_until_advance(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)])
    bursts = _burst_ticks(recs)
    assert bursts and all(r["freeze"] is True for r in bursts)  # every burst frozen
    settles = _settle_ticks(recs)
    # The coast sub-phase (wz still slewing) must be frozen; the required
    # re-observation count in the frozen settle is the param value.
    assert any(r["freeze"] is True for r in settles), "coast should stay frozen"
    assert all(r["need"] == f.params.settle_map_updates for r in settles)
    assert f.params.settle_map_updates >= 2


def test_frozen_settle_blocks_until_reobserved():
    """A frozen turn's YAW_SETTLE will not leave while map_ready is False (the
    adapter withholds it until settle_map_updates fresh voxels have landed).

    This is requirement #6: >=2 stationary re-observations after a real turn.
    """
    f = WaypointFollower()
    f.set_path([Pose2D(0, 0), Pose2D(0, 6)], Pose2D(0, 0, 0.0))
    yaw = 0.0
    for _ in range(80):
        c = f.step(Pose2D(0, 0, yaw), DT)
        yaw = normalize_angle(yaw + c.wz * DT)
        if f.state == FollowerState.YAW_SETTLE:
            break
    assert f.state == FollowerState.YAW_SETTLE
    assert f.settle_map_updates_required == f.params.settle_map_updates
    for _ in range(40):                                   # dwell past the time gate
        f.step(Pose2D(0, 0, yaw), DT, map_ready=False)
    assert f.state == FollowerState.YAW_SETTLE            # still waiting to re-observe


# --------------------------------------------------------------------------
# threshold = 0 -> freeze every turn (legacy)
# --------------------------------------------------------------------------
def test_master_switch_off_keeps_map_live():
    """freeze_on_rotation=False never freezes, even on a big turn, and forces no
    stationary re-observation (map stays live throughout)."""
    p = WaypointFollowerParams(freeze_on_rotation=False)
    f = WaypointFollower(p)
    recs = _roll_until_advance(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)])  # 90 deg
    assert all((r["freeze"] in (False, None)) for r in recs)
    assert all(r["need"] == 0 for r in recs)
    assert f.state == FollowerState.ADVANCE


def test_threshold_zero_freezes_small_turn():
    """freeze_yaw_thresh_rad=0 freezes any turn, however small."""
    p = WaypointFollowerParams(freeze_yaw_thresh_rad=0.0)
    f = WaypointFollower(p)
    # ~30 deg off-axis, far enough that the predictive gate still bursts.
    e = math.radians(30.0)
    recs = _roll_until_advance(f, Pose2D(0, 0, 0.0),
                               [(0, 0), (6 * math.cos(e), 6 * math.sin(e))])
    bursts = _burst_ticks(recs)
    assert bursts and all(r["freeze"] is True for r in bursts)


# --------------------------------------------------------------------------
# latch: a big turn stays frozen through its shrinking residual bursts
# --------------------------------------------------------------------------
def test_episode_freeze_latches_until_advance():
    """Once a big turn latches frozen, every later (smaller) burst of the SAME
    episode stays frozen -- the map never un-freezes mid-turn -- until ADVANCE."""
    f = WaypointFollower()
    recs = _roll_until_advance(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)])  # 90 deg
    # No YAW_ALIGN burst tick before ADVANCE may be unfrozen.
    assert all(r["freeze"] is True for r in _burst_ticks(recs))
    # Reaching ADVANCE clears the latch for the next episode.
    assert f.state == FollowerState.ADVANCE
    assert f._episode_freeze is False
    assert f.settle_map_updates_required == 0            # advancing needs no wait


# --------------------------------------------------------------------------
# a genuinely small correction that still needs a burst is live
# --------------------------------------------------------------------------
def test_small_burst_correction_is_live_by_default():
    """With default 20 deg threshold, an ~18 deg off-axis far waypoint bursts (the
    predictive gate can't glide to it) yet stays live -- freeze False, need 0."""
    f = WaypointFollower()
    e = math.radians(18.0)                               # < 20 deg threshold
    recs = _roll_until_advance(f, Pose2D(0, 0, 0.0),
                               [(0, 0), (6 * math.cos(e), 6 * math.sin(e))])
    bursts = _burst_ticks(recs)
    if bursts:                                           # only assert if it bursts
        assert all(r["freeze"] is False for r in bursts)
        assert all(r["need"] == 0 for r in _settle_ticks(recs))
    # Either way it must converge to ADVANCE without ever freezing.
    assert all((r["freeze"] in (False, None)) for r in recs)
    assert f.state == FollowerState.ADVANCE


# --------------------------------------------------------------------------
# latch does NOT stick across a re-plan from standstill (review finding #1)
# --------------------------------------------------------------------------
def _roll_into_stopped_settle(follower, start, path, steps=200):
    """Roll a turn until the follower sits stopped in a YAW_SETTLE dwell."""
    follower.set_path([Pose2D(*p) for p in path], start)
    yaw = start.yaw
    for _ in range(steps):
        c = follower.step(Pose2D(start.x, start.y, yaw), DT, map_ready=False)
        yaw = normalize_angle(yaw + c.wz * DT)
        if (follower.state == FollowerState.YAW_SETTLE
                and abs(follower._last_wz) <= follower.params.yaw_settle_eps):
            return Pose2D(start.x, start.y, yaw)
    raise AssertionError("did not reach a stopped YAW_SETTLE dwell")


def test_replan_from_standstill_clears_stale_freeze_latch():
    """A big turn latches frozen; if a re-plan arrives while STOPPED mid-turn, the
    next small correction must NOT inherit the frozen ritual (req: small stays
    live). Guards review finding #1."""
    f = WaypointFollower()   # default 20 deg threshold
    pose = _roll_into_stopped_settle(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 6)])  # 90 deg
    assert f._episode_freeze is True                     # big turn latched frozen
    # Re-plan from this standstill toward a small (<20 deg) heading correction.
    e = math.radians(15.0)
    f.set_path([Pose2D(0, 0), Pose2D(6 * math.cos(e), 6 * math.sin(e))], pose)
    assert f._episode_freeze is False                    # latch cleared (stopped)


def test_replan_while_turning_keeps_freeze_latch():
    """A re-plan that lands WHILE the drone is still physically turning keeps the
    freeze (the coast into the settle must stay frozen)."""
    f = WaypointFollower()
    f.set_path([Pose2D(0, 0), Pose2D(0, 10)], Pose2D(0, 0, 0.0))   # 90 deg
    for _ in range(3):
        f.step(Pose2D(0, 0, 0.0), DT)                    # ramp into the burst
    assert abs(f._last_wz) > f.params.yaw_settle_eps and f._episode_freeze is True
    f.set_path([Pose2D(0, 0), Pose2D(10, 0)], Pose2D(0, 0, 1.0))   # re-plan mid-turn
    assert f.state == FollowerState.YAW_SETTLE
    assert f._episode_freeze is True                     # still turning -> stays frozen
