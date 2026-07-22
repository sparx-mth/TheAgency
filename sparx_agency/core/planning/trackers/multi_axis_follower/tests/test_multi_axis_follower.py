"""Unit + closed-loop tests for the multi-axis (vx + vy + yaw) follower.

Covers the allocation math (deadband-with-snap, travel cone, yaw hysteresis,
approach-speed ramp) and the end-to-end behaviour against a simple instantaneous
holonomic plant: crabbing for small offsets without yaw, engaging yaw for large
ones, reaching the goal, never commanding altitude, the minimum-force invariant
(including the decel transient), and the station-keeping deadband.

Run without pytest via the project venv by importing this module and calling the
``test_*`` functions.
"""
import math

from sparx_agency.core.common.types import Pose2D, normalize_angle
from sparx_agency.core.planning.trackers.multi_axis_follower import (
    MultiAxisFollower,
    MultiAxisFollowerParams,
    MultiAxisState,
    predict_trajectory,
)
from sparx_agency.core.planning.trackers.multi_axis_follower import allocation as alloc

DT = 0.1


def _simulate(follower, start, path, steps, dt=DT):
    """Closed-loop rollout against an instantaneous holonomic plant.

    Integrates the commanded (vx, vy, wz) straight into the pose (no inertia),
    which is enough to exercise the control logic. Returns per-tick records and
    the final pose.
    """
    follower.set_path([Pose2D(*p) for p in path], start)
    x, y, yaw = start.x, start.y, start.yaw
    recs = []
    for _ in range(steps):
        pose = Pose2D(x, y, yaw)
        cmd = follower.step(pose, dt)
        recs.append({"state": follower.state, "vx": cmd.vx, "vy": cmd.vy,
                     "wz": cmd.wz, "vz": cmd.command.z, "pose": pose,
                     "done": follower.done})
        yaw = normalize_angle(yaw + cmd.wz * dt)
        x += (cmd.vx * math.cos(yaw) - cmd.vy * math.sin(yaw)) * dt
        y += (cmd.vx * math.sin(yaw) + cmd.vy * math.cos(yaw)) * dt
        if follower.done:
            break
    return recs, Pose2D(x, y, yaw)


def _rollout_emitted(f, start, path, steps, dt=DT, hold_pose=None, hold_steps=0):
    """Rollout returning every emitted (vx, vy, wz)."""
    f.set_path([Pose2D(*p) for p in path], start)
    x, y, yaw = start.x, start.y, start.yaw
    out = []
    for _ in range(steps):
        c = f.step(Pose2D(x, y, yaw), dt)
        out.append((c.vx, c.vy, c.wz))
        yaw = normalize_angle(yaw + c.wz * dt)
        x += (c.vx * math.cos(yaw) - c.vy * math.sin(yaw)) * dt
        y += (c.vx * math.sin(yaw) + c.vy * math.cos(yaw)) * dt
        if f.done:
            break
    if hold_pose is not None:
        for _ in range(hold_steps):
            c = f.step(Pose2D(*hold_pose), dt)
            out.append((c.vx, c.vy, c.wz))
    return out


def _axis_ok(v, mn):
    return abs(v) < 1e-12 or abs(v) >= mn - 1e-9


# ─── Allocation math (pure) ──────────────────────────────────────
def test_shape_axis_deadband_snap_passthrough():
    """Below release -> 0; between release and min -> min; above min -> as-is."""
    s = alloc.shape_axis
    assert s(0.01, 0.06, 0.5, 1e-3) == 0.0                  # below 0.03 -> drop
    assert abs(s(0.04, 0.06, 0.5, 1e-3) - 0.06) < 1e-12     # snap up to min
    assert abs(s(-0.04, 0.06, 0.5, 1e-3) + 0.06) < 1e-12    # sign preserved
    assert abs(s(0.20, 0.06, 0.5, 1e-3) - 0.20) < 1e-12     # pass through
    assert s(0.0, 0.06, 0.5, 1e-3) == 0.0


def test_body_error_angles():
    """Body error encodes the heading error: atan2(e_lat, e_fwd) == eyaw."""
    e_fwd, e_lat, dist, eyaw = alloc.body_error(0.0, 0.0, 0.0, 0.0, 2.0)
    assert abs(eyaw - math.pi / 2) < 1e-9          # target to the left
    assert abs(e_fwd) < 1e-9 and abs(e_lat - 2.0) < 1e-9 and abs(dist - 2.0) < 1e-9
    e_fwd, e_lat, _, eyaw = alloc.body_error(0.0, 0.0, math.pi / 2, 0.0, 2.0)
    assert abs(eyaw) < 1e-9 and abs(e_fwd - 2.0) < 1e-9 and abs(e_lat) < 1e-9


def test_yaw_hysteresis_latch():
    """Engages above engage, stays engaged until below release."""
    eng, rel = math.radians(25), math.radians(10)
    assert alloc.yaw_engaged(False, math.radians(20), eng, rel) is False
    assert alloc.yaw_engaged(False, math.radians(30), eng, rel) is True
    assert alloc.yaw_engaged(True, math.radians(15), eng, rel) is True   # still on
    assert alloc.yaw_engaged(True, math.radians(8), eng, rel) is False   # released


def test_travel_cone_clamp():
    cone = math.radians(80)
    assert abs(alloc.clamp_travel_angle(math.radians(120), cone) - cone) < 1e-12
    assert abs(alloc.clamp_travel_angle(-math.radians(120), cone) + cone) < 1e-12
    assert abs(alloc.clamp_travel_angle(math.radians(30), cone) - math.radians(30)) < 1e-12


def test_approach_speed_ramp():
    """Gentle-arrival law: 0 captured, cruise far, linear ramp to arrive_min at
    pos_radius, ramp disabled when slow_radius<=pos_radius (glide-through)."""
    a = alloc.approach_speed
    pr, slow, cruise, amin = 0.35, 0.8, 0.3, 0.08
    assert a(0.30, pr, slow, cruise, amin) == 0.0                  # captured
    assert abs(a(1.0, pr, slow, cruise, amin) - cruise) < 1e-12    # far -> cruise
    assert abs(a(slow, pr, slow, cruise, amin) - cruise) < 1e-12   # at slow_radius
    mid = a(0.575, pr, slow, cruise, amin)                         # halfway ramp
    assert abs(mid - (amin + (cruise - amin) * 0.5)) < 1e-9
    ramp = [a(d, pr, slow, cruise, amin) for d in (0.8, 0.7, 0.6, 0.5, 0.4, 0.36)]
    assert all(ramp[i] >= ramp[i + 1] for i in range(len(ramp) - 1))  # monotone down
    assert all(v >= amin - 1e-12 for v in ramp)                  # never below floor
    assert abs(a(0.5, pr, pr, cruise, amin) - cruise) < 1e-12    # slow<=pos disables ramp


def test_params_reject_bad_hysteresis():
    raised = False
    try:
        MultiAxisFollowerParams(yaw_engage_rad=0.1, yaw_release_rad=0.2)
    except ValueError:
        raised = True
    assert raised


def test_params_reject_stall_ring_arrive_speed():
    """arrive_speed_min below release_frac*min_vx (would stall just outside
    pos_radius) is rejected."""
    raised = False
    try:
        MultiAxisFollowerParams(arrive_speed_min=0.0)
    except ValueError:
        raised = True
    assert raised


# ─── Closed-loop behaviour ───────────────────────────────────────
def test_straight_ahead_pure_forward():
    """A target dead ahead is flown with pure forward: no lateral, no yaw."""
    f = MultiAxisFollower()
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), (4, 0)], 200)
    assert all(abs(r["wz"]) < 1e-9 for r in recs)
    assert all(abs(r["vy"]) < 1e-9 for r in recs)
    assert any(r["vx"] > 0.0 for r in recs)
    assert f.state == MultiAxisState.HOLD


def test_small_offset_crabs_without_yaw():
    """A small off-axis target (within the yaw deadband) is reached by crabbing:
    lateral motion is used, yaw never engages."""
    f = MultiAxisFollower()
    ang = math.radians(15.0)            # < yaw_engage (25 deg)
    wp = (5.0 * math.cos(ang), 5.0 * math.sin(ang))
    recs, end = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), wp], 400)
    assert all(abs(r["wz"]) < 1e-9 for r in recs), "must not yaw for a small offset"
    assert any(abs(r["vy"]) > 1e-3 for r in recs), "must crab laterally"
    assert f.state == MultiAxisState.HOLD
    assert math.hypot(end.x - wp[0], end.y - wp[1]) < f.params.pos_radius + 0.1


def test_large_offset_engages_yaw_and_reaches():
    """A target far off-axis engages yaw (with translation) and is reached."""
    f = MultiAxisFollower()
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), (0, 5)], 600)  # 90 deg left
    assert any(abs(r["wz"]) > 1e-6 for r in recs), "must yaw for a large offset"
    assert f.state == MultiAxisState.HOLD


def test_goal_behind_turns_around_and_reaches():
    """A goal directly behind is reached by turning around (never skipped)."""
    f = MultiAxisFollower()
    recs, _ = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), (-4, 0)], 800)
    assert any(abs(r["wz"]) > 1e-6 for r in recs)
    assert f.state == MultiAxisState.HOLD


def test_never_commands_altitude():
    """vz is always exactly 0 (fixed altitude)."""
    f = MultiAxisFollower()
    recs, _ = _simulate(f, Pose2D(0, 0, 0.3), [(0, 0), (3, 2), (1, 4)], 800)
    assert all(r["vz"] == 0.0 for r in recs)


def test_multi_waypoint_glides_to_goal():
    """A multi-waypoint path is followed through to HOLD at the final goal."""
    f = MultiAxisFollower()
    path = [(0, 0), (3, 0), (3, 3), (0, 3)]
    recs, end = _simulate(f, Pose2D(0, 0, 0.0), path, 1500)
    assert f.state == MultiAxisState.HOLD
    assert math.hypot(end.x - 0.0, end.y - 3.0) < f.params.pos_radius + 0.2


def test_min_force_invariant_including_decel():
    """Every emitted command (cruise, decel into HOLD) is 0 or >= the axis min."""
    f = MultiAxisFollower()
    p = f.params
    out = _rollout_emitted(f, Pose2D(0, 0, 0.0), [(0, 0), (2, 0)], 400,
                           hold_pose=(2.0, 0.0, 0.0), hold_steps=10)
    assert all(_axis_ok(vx, p.min_vx) and _axis_ok(vy, p.min_vy)
               and _axis_ok(wz, p.min_wz) for vx, vy, wz in out)


def test_min_force_invariant_fine_dt():
    """At a fast rate (accel_limit*dt < min_vx) the ramp-in still never emits a
    sub-threshold command -- shaping is applied AFTER slew."""
    f = MultiAxisFollower()
    p = f.params
    out = _rollout_emitted(f, Pose2D(0, 0, 0.0), [(0, 0), (3, 0)], 40, dt=0.05)
    assert 0.05 * p.accel_limit < p.min_vx          # precondition: ramp step < min
    assert all(_axis_ok(vx, p.min_vx) and _axis_ok(vy, p.min_vy)
               and _axis_ok(wz, p.min_wz) for vx, vy, wz in out)


def test_hold_deadband_commands_zero_when_settled():
    """Sitting on the goal in HOLD settles to a zero command (rides out noise)."""
    f = MultiAxisFollower()
    f.set_path([Pose2D(0, 0), Pose2D(2, 0)], Pose2D(0, 0, 0.0))
    x, y, yaw = 0.0, 0.0, 0.0
    for _ in range(400):
        c = f.step(Pose2D(x, y, yaw), DT)
        yaw = normalize_angle(yaw + c.wz * DT)
        x += (c.vx * math.cos(yaw) - c.vy * math.sin(yaw)) * DT
        y += (c.vx * math.sin(yaw) + c.vy * math.cos(yaw)) * DT
        if f.done:
            break
    assert f.state == MultiAxisState.HOLD
    c = None
    for _ in range(10):
        c = f.step(Pose2D(2.0, 0.0, yaw), DT)
    assert abs(c.vx) < 1e-9 and abs(c.vy) < 1e-9 and abs(c.wz) < 1e-9


def test_hold_crabs_back_to_behind_nose_goal():
    """In HOLD a goal that drifted just behind the nose is crabbed back to (not
    pushed away past the capture radius), and the drone never yaws."""
    f = MultiAxisFollower()
    f.set_path([Pose2D(0, 0), Pose2D(2, 0)], Pose2D(0, 0, 0.0))
    x, y, yaw = 0.0, 0.0, 0.0
    for _ in range(400):
        c = f.step(Pose2D(x, y, yaw), DT)
        yaw = normalize_angle(yaw + c.wz * DT)
        x += (c.vx * math.cos(yaw) - c.vy * math.sin(yaw)) * DT
        y += (c.vx * math.sin(yaw) + c.vy * math.cos(yaw)) * DT
        if f.done:
            break
    assert f.state == MultiAxisState.HOLD
    gx, gy = 2.0, 0.0
    x, y, yaw = gx + 0.30, 0.0, 0.0           # goal directly behind, in nudge band
    dist0 = math.hypot(x - gx, y - gy)
    yaws = []
    for _ in range(80):
        c = f.step(Pose2D(x, y, yaw), DT)
        yaws.append(c.wz)
        yaw = normalize_angle(yaw + c.wz * DT)
        x += (c.vx * math.cos(yaw) - c.vy * math.sin(yaw)) * DT
        y += (c.vx * math.sin(yaw) + c.vy * math.cos(yaw)) * DT
    distf = math.hypot(x - gx, y - gy)
    assert distf < dist0, (dist0, distf)                 # converged, not diverged
    assert distf <= f.params.hold_deadband + 0.05
    assert all(abs(w) < 1e-12 for w in yaws)             # never yawed in HOLD
    assert f.state == MultiAxisState.HOLD


def test_hold_reacquires_on_large_drift():
    """A big drift out of the capture radius drops HOLD back to RUN pursuit."""
    f = MultiAxisFollower()
    f.set_path([Pose2D(0, 0), Pose2D(2, 0)], Pose2D(0, 0, 0.0))
    x, y, yaw = 0.0, 0.0, 0.0
    for _ in range(400):
        c = f.step(Pose2D(x, y, yaw), DT)
        yaw = normalize_angle(yaw + c.wz * DT)
        x += (c.vx * math.cos(yaw) - c.vy * math.sin(yaw)) * DT
        y += (c.vx * math.sin(yaw) + c.vy * math.cos(yaw)) * DT
        if f.done:
            break
    assert f.state == MultiAxisState.HOLD
    c = f.step(Pose2D(2.0 - 1.5, 0.0, yaw), DT)
    assert f.state == MultiAxisState.RUN
    assert abs(c.vx) > 0.0 or abs(c.vy) > 0.0


def test_idle_without_path_holds_zero():
    f = MultiAxisFollower()
    c = f.step(Pose2D(0, 0, 0.0), DT)
    assert f.state == MultiAxisState.IDLE
    assert c.vx == 0.0 and c.vy == 0.0 and c.wz == 0.0 and not f.done


def test_hold_flag_suppresses_motion():
    f = MultiAxisFollower()
    f.set_path([Pose2D(0, 0), Pose2D(5, 0)], Pose2D(0, 0, 0.0))
    c = f.step(Pose2D(0, 0, 0.0), DT, hold=True)
    assert c.vx == 0.0 and c.vy == 0.0 and c.wz == 0.0
    c2 = f.step(Pose2D(0, 0, 0.0), DT, axis_confirmed=False)
    assert c2.vx == 0.0 and c2.vy == 0.0 and c2.wz == 0.0


def test_predict_reaches_goal():
    """The holonomic rollout reaches the goal and reports it."""
    params = MultiAxisFollowerParams()
    res = predict_trajectory(params, Pose2D(0, 0, 0.0),
                             [Pose2D(0, 0), Pose2D(4, 1)], DT, 60.0)
    assert res.reaches_goal
    assert res.end_gap < params.pos_radius + 0.2
    assert len(res.poses) >= 2


def test_predict_collision_flag():
    """The predictor flags obstacles at the start pose and along the rollout."""
    params = MultiAxisFollowerParams()
    start, path = Pose2D(0, 0, 0.0), [Pose2D(0, 0), Pose2D(4, 0)]
    res = predict_trajectory(params, start, path, DT, 60.0,
                             occupied_fn=lambda x, y: 1.5 <= x <= 2.0)
    assert res.collides is True
    res2 = predict_trajectory(params, start, path, DT, 60.0,
                              occupied_fn=lambda x, y: False)
    assert res2.collides is False and res2.reaches_goal and res2.n_stops == 1
    res3 = predict_trajectory(params, start, path, DT, 60.0,
                              occupied_fn=lambda x, y: True)
    assert res3.collides is True            # occupied at the start pose


def test_turn_coordination_confines_translation_to_a_forward_cone():
    """While the yaw axis is active: no backward, at least the floor forward,
    and the lateral capped inside the sideslip cone. A quiet yaw passes the
    translation through untouched."""
    cone = math.radians(45.0)   # tan = 1: the cap equals the forward speed
    # Quiet yaw: nothing is touched, backward included.
    assert alloc.turn_coordination(-0.1, 0.2, 0.01, 0.05, 0.08, cone) == (-0.1, 0.2)
    # Active yaw: backward becomes the floor, the roll is capped at tan*vx.
    vx, vy = alloc.turn_coordination(-0.1, 0.2, 0.3, 0.05, 0.08, cone)
    assert vx == 0.08
    assert abs(vy - 0.08) < 1e-9
    # No floor: never backward, and with vx 0 the cone closes fully.
    vx, vy = alloc.turn_coordination(-0.1, -0.2, 0.3, 0.05, 0.0, cone)
    assert vx == 0.0 and vy == 0.0
    # A forward speed above the floor is left alone; only the roll is shaped.
    vx, vy = alloc.turn_coordination(0.3, -0.4, -0.3, 0.05, 0.08, cone)
    assert vx == 0.3
    assert abs(vy - (-0.3)) < 1e-9


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print("all %d tests passed" % len(fns))
