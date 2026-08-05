"""Tests for the rig itself — a measuring instrument has to be checked too.

Nothing here asserts that the anticipation is *good*; that is a judgement about
the drone and it belongs in the README next to the numbers. These pin the
things that would make the numbers meaningless: an airframe model that does not
match the measured table, a tuning loader that silently falls back to defaults,
and a comparison that is not actually comparing two different controllers.

The survey routes are not exercised here: they pull OMPL through
``core.planning.planners``, which corrupts the heap at interpreter exit and
would take the whole suite's exit code with it.
"""
from __future__ import annotations

from math import degrees, radians

import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.tasks.planning.turn_anticipation_rig.airframe import (
    Airframe, AirframeParams,
)
from sparx_agency.tasks.planning.turn_anticipation_rig.compare import run, summarise
from sparx_agency.tasks.planning.turn_anticipation_rig.flight import fly
from sparx_agency.tasks.planning.turn_anticipation_rig.routes import (
    CORRIDORS, corner_angles,
)
from sparx_agency.tasks.planning.turn_anticipation_rig.tuning import (
    MISSION_YAML, anticipation, controller_params, deployed_dials,
)

RIGHT_TURN = [Pose2D(0.0, 0.0), Pose2D(5.0, 0.0), Pose2D(5.0, -3.0)]


# ── the airframe model ───────────────────────────────────────────
def test_the_yaw_delivery_matches_the_measured_table():
    """~11% standing still, 30-68% flying, inverted going backwards."""
    drone = Airframe()
    assert drone.yaw_delivery(0.0) == pytest.approx(0.11)
    assert drone.yaw_delivery(0.20) == pytest.approx(0.68)
    assert 0.11 < drone.yaw_delivery(0.05) < 0.68
    assert drone.yaw_delivery(-0.10) < 0.0, (
        "a backward translation INVERTED the delivered yaw on the real drone")


def test_the_coupling_can_be_switched_off_for_the_second_number():
    ideal = Airframe(AirframeParams(yaw_bite=False))
    for vx in (-0.2, 0.0, 0.3):
        assert ideal.yaw_delivery(vx) == 1.0


def test_a_standing_drone_barely_rotates():
    """The whole premise, end to end through the model."""
    spinning = Airframe(AirframeParams(drift_vy=0.0))
    flying = Airframe(AirframeParams(drift_vy=0.0))
    for _ in range(40):
        spinning.step(0.0, 0.0, 0.4, 0.1)
        flying.step(0.25, 0.0, 0.4, 0.1)
    assert abs(flying.pose.yaw) > 3.0 * abs(spinning.pose.yaw)


# ── the tuning loader ────────────────────────────────────────────
def test_the_rig_flies_the_deployed_tuning_not_a_copy_of_it():
    dials = deployed_dials()
    assert MISSION_YAML.exists(), "the deployed mission config moved"
    assert dials, "mission.yaml parsed but carried no dp_* dials"
    params = controller_params(dials=dials)
    assert params.cruise_speed == pytest.approx(float(dials["dp_cruise_speed"]))
    assert params.envelope.max_vy == pytest.approx(float(dials["dp_max_vy"]))
    assert params.track_yaw_rate == pytest.approx(
        float(dials["dp_track_yaw_rate"]))


def test_a_missing_config_falls_back_rather_than_crashing():
    params = controller_params(dials={})
    assert params.cruise_speed > 0.0


def test_the_anticipation_is_forced_on_and_inherits_the_yaw_budget():
    """It is off in mission.yaml; the rig must be able to fly it anyway."""
    yl = anticipation(dials={"dp_track_yaw_rate": 0.21})
    assert yl.enabled is True
    assert yl.rate == pytest.approx(0.21), (
        "the schedule must inherit the tracking yaw cap, as the adapter does")


def test_overrides_reach_the_schedule():
    yl = anticipation(dials={}, start_m=4.25)
    assert yl.start_m == 4.25


# ── routes and scoring ───────────────────────────────────────────
def test_every_corridor_carries_the_corner_it_advertises():
    for name, waypoints in CORRIDORS:
        assert len(waypoints) >= 3, name
        turns = [degrees(t) for t in corner_angles(waypoints)]
        assert turns, name
        assert max(abs(t) for t in turns) >= 25.0, (
            "%s has no corner worth anticipating" % name)


def test_corner_angles_are_signed():
    left = corner_angles([Pose2D(0, 0), Pose2D(1, 0), Pose2D(1, 1)])
    right = corner_angles([Pose2D(0, 0), Pose2D(1, 0), Pose2D(1, -1)])
    assert left[0] == pytest.approx(radians(90.0))
    assert right[0] == pytest.approx(radians(-90.0))


def test_a_flight_reaches_the_goal_and_scores_it():
    result = fly(RIGHT_TURN, controller_params())
    assert result.reached
    assert result.seconds > 0.0
    assert result.arrive_err_m < 0.35
    assert len(result.track) == len(result.leads)
    assert result.peak_lead_deg == 0.0, "the anticipation was not asked for"


def test_the_comparison_really_flies_two_different_controllers():
    results = run([CORRIDORS[0]])
    name, off, on = results[0]
    assert off.peak_lead_deg == 0.0
    assert on.peak_lead_deg > 30.0, "the 'ON' run never led its nose anywhere"
    assert on.turn_ticks < off.turn_ticks, (
        "the anticipation did not reduce time spent stopped to rotate")


def test_the_summary_reports_both_runs_and_a_total():
    lines = summarise(run([CORRIDORS[0], CORRIDORS[4]]))
    assert sum(1 for line in lines if " off " in line) == 2
    assert sum(1 for line in lines if " ON " in line) == 2
    assert lines[-1].startswith("TOTAL")
