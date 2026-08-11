"""Fly the servo closed-loop against a lagging airframe and measure the result.

These are not tick-by-tick assertions on intermediate quantities. The claim the
module makes is about *tracking error over a flight*, so that is what is
measured: fly a trajectory, record how far the aircraft ever got from where the
plan said it should be, and assert on the number.

The comparison against ``use_feedforward_lead=False`` is the load-bearing test.
It is the controller the previous stack flew -- velocity feedforward plus a
position P term -- and if inverting the plant does not beat it, the whole design
argument is wrong.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.velocity_servo import VelocityServo, VelocityServoParams
from sparx_agency.core.control.velocity_servo.tests.airframe import LaggingAirframe
from sparx_agency.core.planning.trajectories.bspline import (
    BsplineTrajectory, NonUniformBspline,
)

CONTROL_DT = 0.02
"""50 Hz, the rate the ROS node runs at."""


def straight_trajectory(speed=0.6, seconds=6.0, altitude=1.5, knot_dt=0.5, heading=0.0):
    # type: (float, float, float, float, float) -> BsplineTrajectory
    """A constant-speed run along the aircraft's +x, at a fixed heading."""
    count = int(seconds / knot_dt) + 4
    points = np.zeros((count, 3), dtype=float)
    points[:, 0] = np.arange(count) * speed * knot_dt * math.cos(heading)
    points[:, 1] = np.arange(count) * speed * knot_dt * math.sin(heading)
    points[:, 2] = altitude
    yaw_points = np.full((count, 1), heading, dtype=float)
    return BsplineTrajectory(NonUniformBspline(points, 3, knot_dt),
                             NonUniformBspline(yaw_points, 3, knot_dt),
                             start_time_s=0.0, traj_id=1)


def cornering_trajectory(speed=0.6, knot_dt=0.5, altitude=1.5):
    # type: (float, float, float) -> BsplineTrajectory
    """An L: out along +x, then a left turn onto +y, with the nose following."""
    legs = []
    headings = []
    step = speed * knot_dt
    for i in range(8):
        legs.append((i * step, 0.0, altitude))
        headings.append(0.0)
    x = 7 * step
    for i in range(1, 9):
        legs.append((x, i * step, altitude))
        headings.append(math.pi / 2.0)
    points = np.asarray(legs, dtype=float)
    yaw_points = np.asarray(headings, dtype=float).reshape(-1, 1)
    return BsplineTrajectory(NonUniformBspline(points, 3, knot_dt),
                             NonUniformBspline(yaw_points, 3, knot_dt),
                             start_time_s=0.0, traj_id=1)


def fly(trajectory, params=None, airframe=None, seconds=None, start_offset=(0.0, 0.0, 0.0)):
    # type: (object, object, object, float, object) -> dict
    """Fly a trajectory closed-loop and return the error record.

    The aircraft starts on the curve's own first point unless displaced, so the
    numbers describe tracking rather than an initial acquisition transient.
    """
    servo = VelocityServo(params)
    plant = airframe or LaggingAirframe()
    first = trajectory.sample(0.0)
    # Started in trim, on the plan, at the plan's own speed. Otherwise every
    # number below is dominated by the aircraft accelerating from rest, which
    # is an acquisition transient and not what tracking means.
    plant.place(np.array([first.x, first.y, first.z]) + np.asarray(start_offset, dtype=float),
                yaw=first.yaw or 0.0,
                velocity=(first.vx, first.vy, first.vz), dt=CONTROL_DT)
    servo.set_trajectory(trajectory)

    horizon = trajectory.duration if seconds is None else float(seconds)
    ticks = int(horizon / CONTROL_DT)
    gaps, crosses, speeds, yaw_errors = [], [], [], []
    for i in range(ticks):
        now = (i + 1) * CONTROL_DT
        command = servo.update(plant.position, plant.velocity, plant.yaw,
                               CONTROL_DT, now)
        plant.step(command.body_velocity(), command.yaw_rate, CONTROL_DT)
        if now > 0.5:                       # let the first transient pass
            gaps.append(command.position_error_m)
            crosses.append(abs(command.cross_track_error_m))
            speeds.append(float(np.linalg.norm(plant.velocity)))
            yaw_errors.append(abs(command.yaw_error_rad))
    return {"max_gap": max(gaps), "mean_gap": sum(gaps) / len(gaps),
            "max_cross": max(crosses), "max_speed": max(speeds),
            "max_yaw_error": max(yaw_errors), "servo": servo, "plant": plant}


def test_straight_run_tracks_to_centimetres():
    """On a plan the airframe can fly, the aircraft should be on it."""
    result = fly(straight_trajectory())
    assert result["max_gap"] < 0.10, result["max_gap"]
    assert result["mean_gap"] < 0.05, result["mean_gap"]


def test_inverting_the_plant_beats_not_inverting_it():
    """The lead term is the design's central claim. Measure it.

    Without it the aircraft settles at the standing lag the plant's own delay
    and time constant impose -- which is what the previous stack flew and what
    this backend exists to remove.

    Measured on a **cornering** route, deliberately. A constant-velocity
    straight line has zero acceleration, so ``tau * a_plan`` is zero and the two
    configurations are identical by construction -- the lead term only has
    anything to do where the plan is changing speed or direction, which is
    exactly where an exploration route spends its time and where the tracking
    error that hits walls is generated.
    """
    plan = cornering_trajectory()
    with_lead = fly(plan, VelocityServoParams(use_feedforward_lead=True,
                                              predict_reference=True))
    without = fly(plan, VelocityServoParams(use_feedforward_lead=False,
                                            predict_reference=False))
    assert with_lead["mean_gap"] < 0.6 * without["mean_gap"], (
        "lead %.3f m vs plain %.3f m" % (with_lead["mean_gap"], without["mean_gap"]))
    assert with_lead["max_cross"] < without["max_cross"], (
        "cross-track: lead %.3f m vs plain %.3f m"
        % (with_lead["max_cross"], without["max_cross"]))


def test_speed_never_runs_away_from_the_plan():
    """The clearance FALCON planned for is spent if the aircraft flies faster."""
    params = VelocityServoParams()
    result = fly(straight_trajectory(speed=0.6), params, start_offset=(-1.5, 0.0, 0.0))
    assert result["max_speed"] <= 0.6 + params.max_overspeed + 0.05, result["max_speed"]


def test_a_corner_is_rounded_not_cut():
    """Cross-track is the error that hits walls; a bend must not spend it."""
    result = fly(cornering_trajectory())
    assert result["max_cross"] < 0.25, result["max_cross"]


def test_heading_is_servoed_not_merely_fed_forward():
    """A yaw-rate airframe integrates open-loop unless the loop is closed.

    The aircraft is started 40 degrees off the plan's heading; a controller that
    only feeds the plan's yaw rate forward never recovers that.
    """
    plan = straight_trajectory(seconds=8.0)
    servo = VelocityServo()
    plant = LaggingAirframe()
    first = plan.sample(0.0)
    plant.place((first.x, first.y, first.z), yaw=math.radians(40.0))
    servo.set_trajectory(plan)
    for i in range(int(6.0 / CONTROL_DT)):
        command = servo.update(plant.position, plant.velocity, plant.yaw,
                               CONTROL_DT, (i + 1) * CONTROL_DT)
        plant.step(command.body_velocity(), command.yaw_rate, CONTROL_DT)
    assert abs(plant.yaw) < math.radians(3.0), math.degrees(plant.yaw)


def test_a_standing_disturbance_is_learned():
    """The integrator is the disturbance observer; prove it observes."""
    plan = straight_trajectory(seconds=14.0)
    pushed = LaggingAirframe(drift=(0.0, 0.12, 0.0))
    result = fly(plan, airframe=pushed, seconds=12.0)
    assert result["max_cross"] < 0.30, result["max_cross"]


def test_hold_station_returns_to_the_latched_point():
    """`follow=False` must stop where it was told, not drift."""
    servo = VelocityServo()
    plant = LaggingAirframe()
    plant.place((1.0, 2.0, 1.5))
    for i in range(int(4.0 / CONTROL_DT)):
        command = servo.update(plant.position, plant.velocity, plant.yaw,
                               CONTROL_DT, (i + 1) * CONTROL_DT, follow=False)
        assert command.holding
        plant.step(command.body_velocity(), command.yaw_rate, CONTROL_DT)
    assert np.linalg.norm(plant.position - np.array([1.0, 2.0, 1.5])) < 0.05


def test_the_command_is_body_frame():
    """A world command must arrive rotated, or the aircraft flies a mirror."""
    servo = VelocityServo()
    plan = straight_trajectory(heading=0.0)
    servo.set_trajectory(plan)
    # Facing 90 degrees left of the route: a +x world command must come out as
    # -y in the body (the route is to the aircraft's right).
    command = servo.update((0.3, 0.0, 1.5), (0.0, 0.0, 0.0), math.pi / 2.0,
                           CONTROL_DT, 0.5)
    assert command.world_vx > 0.1
    assert command.vy < -0.1, command
    assert abs(command.vx) < 0.05, command


def test_dt_must_be_positive():
    servo = VelocityServo()
    with pytest.raises(ValueError):
        servo.update((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0, 1.0)


def test_a_resent_trajectory_is_rejected():
    """A re-send must not restart a curve the aircraft is halfway along."""
    servo = VelocityServo()
    plan = straight_trajectory()
    assert servo.set_trajectory(plan)
    assert not servo.set_trajectory(plan)
