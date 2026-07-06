"""Unit + closed-loop tests for the spline-then-Pure-Pursuit task adapter.

ROS-free: drives ``PurePursuitFollower`` (which composes the core HermiteSmoother
and PurePursuitTracker) against a simple instantaneous holonomic plant. The
adapter module lives next to the ROS nodes in ``scripts/``, so the sibling dir is
put on the path before importing it.
"""
import math
import pathlib
import sys

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from sparx_agency.core.common.types import KinematicLimits, Pose2D  # noqa: E402
from sparx_agency.core.planning.smoothers.hermite import (  # noqa: E402
    HermiteParams,
    HermiteSmoother,
)
from sparx_agency.core.planning.trackers.pure_pursuit import (  # noqa: E402
    PurePursuitParams,
    PurePursuitTracker,
)

from pure_pursuit_follower import (  # noqa: E402
    PurePursuitFollower,
    PurePursuitState,
    _shape_axis,
)

DT = 0.1


def _make(**kw):
    limits = KinematicLimits(max_speed_xy=0.6, max_yaw_rate=0.8)
    tracker = PurePursuitTracker(PurePursuitParams(holonomic=True), default_limits=limits)
    smoother = HermiteSmoother(HermiteParams())
    return PurePursuitFollower(tracker, smoother, limits=limits, **kw)


def _simulate(f, start, path, steps, dt=DT):
    """Closed-loop rollout against an instantaneous holonomic plant."""
    f.set_path([Pose2D(*p) for p in path], start)
    x, y, yaw = start.x, start.y, start.yaw
    recs = []
    for _ in range(steps):
        c = f.step(Pose2D(x, y, yaw), dt)
        recs.append((c.vx, c.vy, c.wz, c.state, c.done))
        yaw = yaw + c.wz * dt
        x += (c.vx * math.cos(yaw) - c.vy * math.sin(yaw)) * dt
        y += (c.vx * math.sin(yaw) + c.vy * math.cos(yaw)) * dt
        if f.done:
            break
    return recs, (x, y, yaw)


# ─── unit ────────────────────────────────────────────────────────
def test_shape_axis():
    assert _shape_axis(0.01, 0.06, 0.5, 1e-3) == 0.0
    assert abs(_shape_axis(0.04, 0.06, 0.5, 1e-3) - 0.06) < 1e-12
    assert abs(_shape_axis(0.2, 0.06, 0.5, 1e-3) - 0.2) < 1e-12
    assert abs(_shape_axis(0.2, 0.0, 0.5, 1e-3) - 0.2) < 1e-12   # min=0 -> inert
    assert _shape_axis(0.0005, 0.0, 0.5, 1e-3) == 0.0            # dust still zeroed


def test_set_path_splines_and_caches():
    f = _make()
    f.set_path([Pose2D(0, 0), Pose2D(2, 0.3), Pose2D(4, 1.0), Pose2D(6, 1.0)],
               Pose2D(0, 0, 0.0))
    assert f.state == PurePursuitState.RUN
    assert len(f.smooth_xy) > 10                         # densely sampled spline
    # The spline starts and ends at the path endpoints.
    assert math.hypot(f.smooth_xy[0][0] - 0.0, f.smooth_xy[0][1] - 0.0) < 0.05
    assert math.hypot(f.smooth_xy[-1][0] - 6.0, f.smooth_xy[-1][1] - 1.0) < 0.05


def test_short_path_goes_idle():
    f = _make()
    f.set_path([Pose2D(0, 0)], Pose2D(0, 0, 0.0))        # < 2 distinct points
    assert f.state == PurePursuitState.IDLE
    c = f.step(Pose2D(0, 0, 0.0), DT)
    assert c.vx == 0.0 and c.vy == 0.0 and c.wz == 0.0


def test_step_moves_and_sets_lookahead():
    f = _make()
    f.set_path([Pose2D(0, 0), Pose2D(4, 0)], Pose2D(0, 0, 0.0))
    c = f.step(Pose2D(0, 0, 0.0), DT)
    assert c.vx > 0.0                                    # advancing forward
    assert f.lookahead is not None and f.lookahead[0] > 0.0
    assert not c.done


def test_reaches_goal_straight():
    f = _make()
    recs, end = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), (5, 0)], 400)
    assert f.state == PurePursuitState.DONE and f.done
    assert math.hypot(end[0] - 5.0, end[1] - 0.0) < 0.3


def test_reaches_goal_curved():
    f = _make()
    recs, end = _simulate(f, Pose2D(0, 0, 0.0),
                          [(0, 0), (2, 1.0), (4, 1.0), (6, 0.0)], 800)
    assert f.done
    assert math.hypot(end[0] - 6.0, end[1] - 0.0) < 0.4


def test_hold_and_unconfirmed_axis_suppress_motion():
    f = _make()
    f.set_path([Pose2D(0, 0), Pose2D(5, 0)], Pose2D(0, 0, 0.0))
    c1 = f.step(Pose2D(0, 0, 0.0), DT, hold=True)
    assert c1.vx == 0.0 and c1.vy == 0.0 and c1.wz == 0.0
    c2 = f.step(Pose2D(0, 0, 0.0), DT, axis_confirmed=False)
    assert c2.vx == 0.0 and c2.vy == 0.0 and c2.wz == 0.0


def test_crab_vy_sign_and_magnitude():
    """Holonomic crab: a left-curving goal yields +vy (left, REP-103), a
    right-curving goal yields -vy, and the magnitude is non-trivial. Pins both
    the crab feature and the sign convention (a vy zero/flip would pass the
    reach-goal tests, which the plant can satisfy via yaw+vx alone)."""
    left = _make()
    left.set_path([Pose2D(0, 0), Pose2D(2, 2)], Pose2D(0, 0, 0.0))
    cl = left.step(Pose2D(0, 0, 0.0), DT)
    assert cl.vy > 0.02, cl.vy
    right = _make()
    right.set_path([Pose2D(0, 0), Pose2D(2, -2)], Pose2D(0, 0, 0.0))
    cr = right.step(Pose2D(0, 0, 0.0), DT)
    assert cr.vy < -0.02, cr.vy


def test_short_unsmootheable_path_is_graceful():
    """A 2-point path too short to spline (sub-sample length) drops to IDLE and
    holds zero -- it does not raise out of set_path or keep flying a stale path."""
    f = _make()
    f.set_path([Pose2D(0, 0), Pose2D(4, 0)], Pose2D(0, 0, 0.0))   # valid first
    assert f.state == PurePursuitState.RUN
    f.set_path([Pose2D(1.0, 1.0), Pose2D(1.005, 1.0)], Pose2D(1, 1, 0.0))  # ~5 mm
    assert f.state == PurePursuitState.IDLE
    c = f.step(Pose2D(1, 1, 0.0), DT)
    assert c.vx == 0.0 and c.vy == 0.0 and c.wz == 0.0


def test_all_duplicate_path_is_idle():
    f = _make()
    f.set_path([Pose2D(1, 1), Pose2D(1, 1), Pose2D(1, 1)], Pose2D(0, 0, 0.0))
    assert f.state == PurePursuitState.IDLE


def test_min_force_too_large_rejected():
    """A forward min-force floor that would stall the near-goal slow-down is
    rejected at construction (release_frac*min_vx >= tracker min_speed)."""
    raised = False
    try:
        _make(min_vx=0.5)   # default tracker min_speed 0.1 <= 0.5*0.5 = 0.25
    except ValueError:
        raised = True
    assert raised


def test_default_min_force_reaches_goal():
    """A valid platform min-force config (node defaults) still reaches the goal
    -- the floor never snaps the approach to a stall."""
    f = _make(min_vx=0.06, min_vy=0.06, min_wz=math.radians(8))
    recs, end = _simulate(f, Pose2D(0, 0, 0.0), [(0, 0), (5, 0)], 400)
    assert f.done and math.hypot(end[0] - 5.0, end[1] - 0.0) < 0.4


def test_reset_clears():
    f = _make()
    f.set_path([Pose2D(0, 0), Pose2D(4, 0)], Pose2D(0, 0, 0.0))
    f.step(Pose2D(0, 0, 0.0), DT)
    f.reset()
    assert f.state == PurePursuitState.IDLE and not f.done
    assert f.smooth_xy == [] and f.lookahead is None


if __name__ == "__main__":
    mod = sys.modules[__name__]
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print("all %d tests passed" % len(fns))
