"""Unit tests for the cross-track ROLL corrector control law."""
from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.trackers.roll_assist_follower import (
    CrossTrackRollCorrector,
    CrossTrackRollParams,
)

STRAIGHT = [(0.0, 0.0), (5.0, 0.0)]   # trajectory along +x, y = 0


def _settle(corr, pose, **kw):
    """Run the corrector for enough ticks that the slew ramp reaches steady state."""
    vy = vx = 0.0
    for _ in range(60):
        vy, vx = corr.correct(pose, STRAIGHT, 0, dt=0.1, **kw)
    return vy, vx


def test_advancing_drift_left_pulls_right():
    # Drone 0.3 m to the LEFT of the line (y=+0.3), facing +x -> ROLL right (vy<0).
    corr = CrossTrackRollCorrector()
    vy, vx = _settle(corr, Pose2D(2.0, 0.3, 0.0), advancing=True, yaw_active=False)
    assert vy < 0.0
    assert vx == 0.0                       # no forward correction while advancing


def test_advancing_drift_right_pulls_left():
    corr = CrossTrackRollCorrector()
    vy, vx = _settle(corr, Pose2D(2.0, -0.3, 0.0), advancing=True, yaw_active=False)
    assert vy > 0.0
    assert vx == 0.0


def test_on_track_no_correction():
    corr = CrossTrackRollCorrector()
    vy, vx = _settle(corr, Pose2D(2.0, 0.0, 0.0), advancing=True, yaw_active=False)
    assert vy == 0.0 and vx == 0.0


def test_deadband_ignores_tiny_drift():
    # 3 cm drift is inside the 5 cm deadband -> nothing.
    corr = CrossTrackRollCorrector()
    vy, vx = _settle(corr, Pose2D(2.0, 0.03, 0.0), advancing=True, yaw_active=False)
    assert vy == 0.0 and vx == 0.0


def test_turn_correction_is_weaker_than_advance():
    pose = Pose2D(2.0, 0.3, 0.0)
    adv = CrossTrackRollCorrector()
    turn = CrossTrackRollCorrector()
    vy_adv, _ = _settle(adv, pose, advancing=True, yaw_active=False)
    vy_turn, _ = _settle(turn, pose, advancing=False, yaw_active=True)
    # Both pull the same way (right / negative) but the turn correction is smaller.
    assert vy_turn < 0.0 and vy_adv < 0.0
    assert abs(vy_turn) < abs(vy_adv)


def test_hold_correction_is_small():
    pose = Pose2D(2.0, 0.3, 0.0)
    adv = CrossTrackRollCorrector()
    hold = CrossTrackRollCorrector()
    vy_adv, _ = _settle(adv, pose, advancing=True, yaw_active=False)
    vy_hold, _ = _settle(hold, pose, advancing=False, yaw_active=False)
    assert abs(vy_hold) < abs(vy_adv)


def test_turn_adds_along_track_correction():
    # Drone BEHIND the segment start while turning -> a forward nudge is allowed.
    corr = CrossTrackRollCorrector()
    vy, vx = _settle(corr, Pose2D(-0.5, 0.0, 0.0), advancing=False, yaw_active=True)
    assert vx > 0.0                        # closest point is ahead -> push forward


def test_advance_never_pushes_forward_or_back():
    # Same behind-the-start pose, but advancing: the base drives forward, so the
    # corrector must not add its own along-track command.
    corr = CrossTrackRollCorrector()
    _, vx = _settle(corr, Pose2D(-0.5, 0.0, 0.0), advancing=True, yaw_active=False)
    assert vx == 0.0


def test_relax_decays_to_zero():
    corr = CrossTrackRollCorrector()
    _settle(corr, Pose2D(2.0, 0.4, 0.0), advancing=True, yaw_active=False)
    vy = vx = None
    for _ in range(60):
        vy, vx = corr.relax(0.1)
    assert vy == 0.0 and vx == 0.0


def test_min_force_floor_and_saturation():
    # Huge drift saturates at lateral_speed_max; the shaped output never dribbles
    # below min_vy.
    p = CrossTrackRollParams(lateral_speed_max=0.25, min_vy=0.06)
    corr = CrossTrackRollCorrector(p)
    vy, _ = _settle(corr, Pose2D(2.0, 5.0, 0.0), advancing=True, yaw_active=False)
    assert abs(abs(vy) - 0.25) < 1e-6      # saturated
    assert abs(vy) >= 0.06                 # above the min-force floor


def test_empty_path_relaxes():
    corr = CrossTrackRollCorrector()
    vy, vx = corr.correct(Pose2D(0.0, 0.0, 0.0), [(0.0, 0.0)], 0,
                          advancing=True, yaw_active=False, dt=0.1)
    assert vy == 0.0 and vx == 0.0


def test_params_validation():
    ok = False
    try:
        CrossTrackRollParams(release_frac=1.5)
    except ValueError:
        ok = True
    assert ok
    ok = False
    try:
        CrossTrackRollParams(lateral_speed_max=0.0)
    except ValueError:
        ok = True
    assert ok
