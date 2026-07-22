"""Unit tests for the follower trajectory predictor (rollout)."""
import math

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.trackers.waypoint_follower import (
    MotionModelParams,
    WaypointFollowerParams,
    predict_trajectory,
    prediction_score,
)

DT = 0.2


def test_predictor_reaches_goal_straight():
    res = predict_trajectory(WaypointFollowerParams(), Pose2D(0, 0, 0.0),
                             [Pose2D(0, 0), Pose2D(4, 0)], DT, horizon_s=40.0)
    assert res.reaches_goal
    assert res.end_gap < 0.6
    assert not res.collides
    assert len(res.poses) > 2
    assert res.total_yaw < 0.2          # essentially no turning on a straight run


def test_predictor_detects_collision():
    def wall(x, y):
        return 1.5 <= x <= 2.5          # a wall straddling the straight path

    res = predict_trajectory(WaypointFollowerParams(), Pose2D(0, 0, 0.0),
                             [Pose2D(0, 0), Pose2D(4, 0)], DT, horizon_s=40.0,
                             occupied_fn=wall)
    assert res.collides


def test_predictor_raises_on_bad_input():
    for bad in (
        dict(path=[Pose2D(0, 0)], dt=DT, horizon_s=10.0),
        dict(path=[Pose2D(0, 0), Pose2D(1, 0)], dt=0.0, horizon_s=10.0),
        dict(path=[Pose2D(0, 0), Pose2D(1, 0)], dt=DT, horizon_s=0.0),
    ):
        try:
            predict_trajectory(WaypointFollowerParams(), Pose2D(0, 0, 0.0), **bad)
            assert False, "expected ValueError for %r" % bad
        except ValueError:
            pass


def test_predictor_corner_is_deterministic_and_turns():
    path = [Pose2D(0, 0), Pose2D(3, 0), Pose2D(3, 3)]
    kw = dict(dt=DT, horizon_s=60.0, motion=MotionModelParams(yaw_tau_s=1.2))
    a = predict_trajectory(WaypointFollowerParams(), Pose2D(0, 0, 0.0), path, **kw)
    b = predict_trajectory(WaypointFollowerParams(), Pose2D(0, 0, 0.0), path, **kw)
    assert a.poses == b.poses                      # no randomness
    assert a.reaches_goal
    assert a.n_stops >= 1                           # the corner forces a pause
    assert a.total_yaw > 1.3                        # ~90 deg turn (>= geometric)


def test_prediction_score_high_for_clean_run_zero_for_collision():
    clean = predict_trajectory(WaypointFollowerParams(), Pose2D(0, 0, 0.0),
                               [Pose2D(0, 0), Pose2D(4, 0)], DT, horizon_s=40.0)
    assert prediction_score(clean) > 0.6

    hit = predict_trajectory(WaypointFollowerParams(), Pose2D(0, 0, 0.0),
                             [Pose2D(0, 0), Pose2D(4, 0)], DT, horizon_s=40.0,
                             occupied_fn=lambda x, y: 1.5 <= x <= 2.5)
    assert prediction_score(hit) == 0.0
