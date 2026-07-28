"""Tests for the 3D reference tracker.

The interesting properties are behavioural, so most of these fly a toy aircraft:
a first-order velocity lag, which is what a well-tuned inner loop looks like from
outside. That is enough to show the difference between a controller that only
replays the plan and one that closes on it.
"""
import math

import pytest

from sparx_agency.core.common.types import KinematicLimits, TrajectoryPoint
from sparx_agency.core.planning.trackers.drift_pid.pid import PidGains
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)

DT = 0.02


def _reference(t, speed=1.0, z=1.5, yaw=0.0):
    """A straight, constant-speed reference along +x."""
    return TrajectoryPoint(t=t, x=speed * t, y=0.0, z=z,
                           vx=speed, vy=0.0, vz=0.0, yaw=yaw)


class _LaggingAircraft:
    """First-order velocity lag with an optional constant disturbance.

    ``tau`` is how long the inner loop takes to reach a commanded velocity;
    ``drift`` is a steady push the controller is not told about, which is what
    the integral term exists to learn.
    """

    def __init__(self, position=(0.0, 0.0, 1.5), tau=0.25, drift=(0.0, 0.0, 0.0)):
        self.position = list(position)
        self.velocity = [0.0, 0.0, 0.0]
        self.tau = tau
        self.drift = drift
        self.yaw = 0.0

    def step(self, command, yaw_cmd, dt):
        alpha = dt / (self.tau + dt)
        for i in range(3):
            self.velocity[i] += alpha * (command[i] - self.velocity[i])
            self.position[i] += (self.velocity[i] + self.drift[i]) * dt
        self.yaw = yaw_cmd


def test_perfect_tracking_emits_the_reference_velocity():
    """With no error, the command is the plan -- feed-forward, not feedback."""
    tracker = ReferenceTracker3D()
    reference = _reference(1.0, speed=1.0)
    out = tracker.update(reference, (reference.x, reference.y, reference.z),
                         yaw=0.0, dt=DT)
    assert out.vx == pytest.approx(1.0, abs=1e-9)
    assert out.vy == pytest.approx(0.0, abs=1e-9)
    assert out.vz == pytest.approx(0.0, abs=1e-9)
    assert out.position_error_m == pytest.approx(0.0, abs=1e-9)
    assert not out.holding


def test_acceleration_is_led_onto_the_velocity_command():
    """The reference acceleration shows up as a velocity lead, not as nothing."""
    params = ReferenceTrackerParams(accel_lead_s=0.4)
    tracker = ReferenceTracker3D(params)
    reference = TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=1.0,
                                vx=0.5, vy=0.0, vz=0.0, ax=1.0, yaw=0.0)
    out = tracker.update(reference, (0.0, 0.0, 1.0), yaw=0.0, dt=DT)
    assert out.vx == pytest.approx(0.5 + 0.4 * 1.0, abs=1e-9)


def _fly(tracker, aircraft, steps, reference_of):
    """Run a closed loop and return the last tracker output."""
    out = None
    for step in range(steps):
        out = tracker.update(reference_of(step * DT), tuple(aircraft.position),
                             aircraft.yaw, DT)
        aircraft.step(out.velocity(), out.yaw, DT)
    return out


def test_feedback_closes_a_standing_position_error():
    """A step offset from the plan is flown out, not held."""
    tracker = ReferenceTracker3D()
    aircraft = _LaggingAircraft(position=(0.0, -1.0, 1.5))
    _fly(tracker, aircraft, 2500, lambda t: _reference(t, speed=0.0, z=1.5))
    # Settles inside the loop's own deadband (1 cm) plus a little.
    assert abs(aircraft.position[1]) < 0.02


def test_integral_separation_keeps_a_large_step_from_overshooting():
    """Integrating a 1 m catch-up would charge a correction that arrives late."""
    tracker = ReferenceTracker3D()
    aircraft = _LaggingAircraft(position=(0.0, -1.0, 1.5))
    overshoot = 0.0
    for step in range(1500):
        out = tracker.update(_reference(step * DT, speed=0.0, z=1.5),
                             tuple(aircraft.position), aircraft.yaw, DT)
        aircraft.step(out.velocity(), out.yaw, DT)
        overshoot = max(overshoot, aircraft.position[1])   # past the reference
    assert overshoot < 0.06


def test_integral_learns_a_constant_disturbance():
    """A steady push is cancelled to a much smaller offset than P alone allows."""
    tracker = ReferenceTracker3D()
    aircraft = _LaggingAircraft(drift=(0.0, 0.15, 0.0))
    _fly(tracker, aircraft, 2500, lambda t: _reference(t, speed=0.0, z=1.5))
    # Proportional-only settles at drift / kp = 0.15 m. The integral does an
    # order of magnitude better, which is the whole reason it is here.
    assert abs(aircraft.position[1]) < 0.05


def test_tracks_a_moving_reference_within_a_handspan():
    """Flying the plan, not chasing it: the lag stays small at cruise speed."""
    tracker = ReferenceTracker3D()
    aircraft = _LaggingAircraft()
    worst = 0.0
    for step in range(500):
        t = step * DT
        out = tracker.update(_reference(t, speed=1.0), tuple(aircraft.position),
                             aircraft.yaw, DT)
        aircraft.step(out.velocity(), out.yaw, DT)
        if t > 1.0:  # after the initial acceleration transient
            worst = max(worst, out.position_error_m)
    assert worst < 0.20


def test_stale_reference_holds_station_rather_than_flying_on():
    """Silence from the planner stops the aircraft where it is."""
    tracker = ReferenceTracker3D()
    reference = _reference(1.0, speed=1.0)
    tracker.update(reference, (1.0, 0.0, 1.5), yaw=0.0, dt=DT)
    out = tracker.update(reference, (1.0, 0.0, 1.5), yaw=0.0, dt=DT,
                         reference_age=5.0)
    assert out.holding
    assert out.velocity() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_hold_point_is_latched_not_re_measured():
    """A drifting aircraft is flown back to the hold point, not tracked away."""
    tracker = ReferenceTracker3D()
    tracker.update(None, (0.0, 0.0, 1.5), yaw=0.0, dt=DT)
    out = tracker.update(None, (0.5, 0.0, 1.5), yaw=0.0, dt=DT)
    assert out.vx < -0.1   # commanded back toward x=0
    assert out.holding


def test_none_reference_holds_and_does_not_raise():
    tracker = ReferenceTracker3D()
    out = tracker.update(None, (1.0, 2.0, 3.0), yaw=0.5, dt=DT)
    assert out.holding
    assert out.yaw == pytest.approx(0.5)


def test_horizontal_speed_is_capped_without_turning_the_command():
    """Over-speed is scaled, so the commanded direction survives the clamp."""
    params = ReferenceTrackerParams(
        limits=KinematicLimits(max_speed_xy=1.0, max_speed_z=0.5,
                               max_yaw_rate=1.0))
    tracker = ReferenceTracker3D(params)
    reference = TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=1.0, vx=3.0, vy=4.0, yaw=0.0)
    out = tracker.update(reference, (0.0, 0.0, 1.0), yaw=0.0, dt=DT)
    assert math.hypot(out.vx, out.vy) == pytest.approx(1.0, abs=1e-9)
    assert math.atan2(out.vy, out.vx) == pytest.approx(math.atan2(4.0, 3.0), abs=1e-9)


def test_climb_rate_is_capped():
    params = ReferenceTrackerParams(
        limits=KinematicLimits(max_speed_xy=2.0, max_speed_z=0.4, max_yaw_rate=1.0))
    tracker = ReferenceTracker3D(params)
    reference = TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=5.0, vz=3.0, yaw=0.0)
    out = tracker.update(reference, (0.0, 0.0, 1.0), yaw=0.0, dt=DT)
    assert out.vz == pytest.approx(0.4)


def test_yaw_passes_through_when_the_planner_respects_its_own_rate():
    """A planner-rate yaw ramp is not re-limited by the tracker."""
    params = ReferenceTrackerParams(
        limits=KinematicLimits(max_speed_xy=2.0, max_speed_z=1.0,
                               max_yaw_rate=1.0),
        yaw_rate_margin=1.5)
    tracker = ReferenceTracker3D(params)
    yaw = 0.0
    for step in range(50):
        yaw += 1.0 * DT       # exactly the planner's own ceiling
        out = tracker.update(_reference(step * DT, speed=0.0, yaw=yaw),
                             (0.0, 0.0, 1.5), yaw=0.0, dt=DT)
    assert out.yaw == pytest.approx(yaw, abs=1e-6)


def test_yaw_jump_is_rate_limited():
    """A discontinuous reference heading is slewed, not stepped."""
    params = ReferenceTrackerParams(
        limits=KinematicLimits(max_speed_xy=2.0, max_speed_z=1.0,
                               max_yaw_rate=1.0),
        yaw_rate_margin=1.0)
    tracker = ReferenceTracker3D(params)
    tracker.update(_reference(0.0, speed=0.0, yaw=0.0), (0.0, 0.0, 1.5), 0.0, DT)
    out = tracker.update(_reference(DT, speed=0.0, yaw=math.pi), (0.0, 0.0, 1.5),
                         0.0, DT)
    assert abs(out.yaw) <= 1.0 * DT + 1e-9


def test_yaw_slew_takes_the_short_way_round():
    params = ReferenceTrackerParams(
        limits=KinematicLimits(max_speed_xy=2.0, max_speed_z=1.0, max_yaw_rate=1.0),
        yaw_rate_margin=1.0)
    tracker = ReferenceTracker3D(params)
    tracker.reset(yaw=math.pi - 0.05)
    out = tracker.update(_reference(0.0, speed=0.0, yaw=-math.pi + 0.05),
                         (0.0, 0.0, 1.5), yaw=math.pi - 0.05, dt=DT)
    assert out.yaw > 0.0   # went up through +pi, not down through 0


def test_divergence_is_reported_but_not_enforced():
    params = ReferenceTrackerParams(max_position_error_m=1.0)
    tracker = ReferenceTracker3D(params)
    out = tracker.update(_reference(0.0, speed=0.0), (0.0, 5.0, 1.5), 0.0, DT)
    assert out.diverged
    assert out.position_error_m == pytest.approx(5.0)
    assert abs(out.vy) > 0.0   # still trying


def test_error_splits_into_lag_and_cross_track():
    """Behind on a +x reference reads as lag; beside it reads as cross-track."""
    tracker = ReferenceTracker3D()
    reference = _reference(1.0, speed=1.0)   # at x=1, travelling +x
    behind = tracker.update(reference, (0.7, 0.0, 1.5), 0.0, DT)
    assert behind.along_track_lag_m == pytest.approx(0.3, abs=1e-6)
    assert behind.cross_track_error_m == pytest.approx(0.0, abs=1e-6)

    tracker.reset()
    beside = tracker.update(reference, (1.0, -0.3, 1.5), 0.0, DT)
    assert beside.along_track_lag_m == pytest.approx(0.0, abs=1e-6)
    assert beside.cross_track_error_m == pytest.approx(0.3, abs=1e-6)


def test_stationary_reference_reports_all_error_as_cross_track():
    """A hover point has no direction of travel, so nothing is 'late'."""
    tracker = ReferenceTracker3D()
    out = tracker.update(_reference(0.0, speed=0.0), (0.4, 0.3, 1.5), 0.0, DT)
    assert out.along_track_lag_m == 0.0
    assert out.cross_track_error_m == pytest.approx(0.5, abs=1e-6)


def test_reset_clears_the_learned_bias():
    tracker = ReferenceTracker3D()
    for _ in range(200):
        tracker.update(_reference(0.0, speed=0.0), (0.0, -0.5, 1.5), 0.0, DT)
    charged = tracker.update(_reference(0.0, speed=0.0), (0.0, 0.0, 1.5), 0.0, DT)
    tracker.reset(yaw=0.0)
    cleared = tracker.update(_reference(0.0, speed=0.0), (0.0, 0.0, 1.5), 0.0, DT)
    assert abs(cleared.vy) < abs(charged.vy)
    assert cleared.vy == pytest.approx(0.0, abs=1e-9)


def test_non_positive_dt_is_rejected():
    tracker = ReferenceTracker3D()
    with pytest.raises(ValueError):
        tracker.update(_reference(0.0), (0.0, 0.0, 1.5), 0.0, 0.0)


@pytest.mark.parametrize("kwargs", [
    {"accel_lead_s": -0.1},
    {"max_position_error_m": 0.0},
    {"reference_timeout_s": 0.0},
    {"yaw_rate_margin": 0.0},
])
def test_invalid_params_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ReferenceTrackerParams(**kwargs)


def test_params_default_speed_ceiling_exceeds_a_typical_planner_limit():
    """The clamp must not clip a trajectory the planner thought was feasible."""
    params = ReferenceTrackerParams()
    assert params.limits.max_speed_xy > 1.0


def test_gains_reject_an_integral_that_could_saturate_the_axis_alone():
    with pytest.raises(ValueError):
        PidGains(kp=1.0, ki=0.1, i_limit=2.0, out_limit=1.0)
