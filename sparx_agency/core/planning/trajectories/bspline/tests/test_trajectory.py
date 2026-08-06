"""The position-plus-yaw trajectory, and the endpoint behaviour a controller relies on."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory


def _straight_run(length=10, spacing=0.6, knot_span=0.4, yaw_values=None):
    """A straight, evenly spaced trajectory, built the way FALCON transmits one.

    Args:
        length: Number of position control points.
        spacing: Distance between them, metres.
        knot_span: Uniform knot spacing, seconds.
        yaw_values: Yaw control points; a constant heading by default.

    Returns:
        A :class:`BsplineTrajectory`.
    """
    points = [[float(i) * spacing, 0.0, 1.4] for i in range(length)]
    knots = [(-3 + i) * knot_span for i in range(length + 4)]
    yaws = yaw_values if yaw_values is not None else [0.25] * 6
    return BsplineTrajectory.from_falcon(
        order=3, knots=knots, position_points=points, yaw_points=yaws,
        yaw_dt=knot_span, start_time_s=100.0, traj_id=7)


def test_duration_follows_the_transmitted_knots():
    """The trajectory lasts as long as its position curve's usable span."""
    trajectory = _straight_run(length=10, knot_span=0.4)
    assert trajectory.duration == pytest.approx((10 - 3) * 0.4)


def test_mid_flight_sample_carries_every_derivative():
    """Inside the span the reference has velocity, acceleration and jerk."""
    trajectory = _straight_run()
    reference = trajectory.sample(1.0)
    assert reference.vx > 0.0
    assert reference.vy == pytest.approx(0.0, abs=1e-9)
    # A constant-speed straight line has no acceleration, but the fields exist
    # and are finite -- which is what the attitude feedforward will read.
    for name in ("ax", "ay", "az", "jx", "jy", "jz"):
        assert math.isfinite(getattr(reference, name))


def test_speed_is_spacing_over_knot_span():
    """Evenly spaced control points fly at a predictable, checkable speed."""
    trajectory = _straight_run(spacing=0.6, knot_span=0.4)
    reference = trajectory.sample(1.0)
    assert reference.vx == pytest.approx(0.6 / 0.4, abs=1e-6)


def test_past_the_end_holds_position_and_zeroes_every_derivative():
    """Overrunning a trajectory brakes to a hover on its last point.

    Extrapolating a B-spline past its end sends the aircraft into space FALCON
    never checked against the map, so this is a safety property rather than a
    convenience.
    """
    trajectory = _straight_run()
    last = trajectory.sample(trajectory.duration)
    beyond = trajectory.sample(trajectory.duration + 5.0)
    assert (beyond.x, beyond.y, beyond.z) == pytest.approx((last.x, last.y, last.z))
    assert (beyond.vx, beyond.vy, beyond.vz) == (0.0, 0.0, 0.0)
    assert (beyond.ax, beyond.ay, beyond.az) == (0.0, 0.0, 0.0)
    assert (beyond.jx, beyond.jy, beyond.jz) == (0.0, 0.0, 0.0)
    assert beyond.yaw_rate == 0.0


def test_before_the_start_holds_the_first_point():
    """A trajectory that has not begun yet reads as its own start, at rest.

    FALCON deliberately starts each new curve a planning-time in the future, so
    a negative elapsed time is routine rather than a fault.
    """
    trajectory = _straight_run()
    first = trajectory.sample(0.0)
    early = trajectory.sample(-0.4)
    assert (early.x, early.y, early.z) == pytest.approx((first.x, first.y, first.z))
    assert (early.vx, early.vy, early.vz) == (0.0, 0.0, 0.0)


def test_yaw_is_wrapped_into_the_principal_range():
    """A yaw curve wandering past pi comes back wrapped, as the C++ does it."""
    trajectory = _straight_run(yaw_values=[3.0, 3.4, 3.8, 4.2, 4.6, 5.0])
    for t in np.linspace(0.0, trajectory.duration, 20):
        assert -math.pi <= trajectory.sample(float(t)).yaw <= math.pi


def test_wall_clock_sampling_uses_the_start_time():
    """``sample_at`` and ``sample`` agree once the start time is subtracted."""
    trajectory = _straight_run()
    assert trajectory.start_time_s == 100.0
    by_clock = trajectory.sample_at(101.25)
    by_elapsed = trajectory.sample(1.25)
    assert (by_clock.x, by_clock.vx) == pytest.approx((by_elapsed.x, by_elapsed.vx))


def test_activity_window():
    """A trajectory is active only between its start and its end."""
    trajectory = _straight_run()
    assert not trajectory.is_active(99.0)
    assert trajectory.is_active(100.5)
    assert not trajectory.is_active(100.0 + trajectory.duration + 0.1)


def test_dimension_mistakes_are_refused():
    """A 2D position curve or a 3D yaw curve is a wiring bug, caught at build."""
    from sparx_agency.core.planning.trajectories.bspline import NonUniformBspline
    flat = NonUniformBspline(np.zeros((6, 2)), 3)
    yaw = NonUniformBspline(np.zeros((6, 1)), 3)
    with pytest.raises(ValueError, match="position curve must be 3D"):
        BsplineTrajectory(flat, yaw, 0.0, 1)
    with pytest.raises(ValueError, match="yaw curve must be 1D"):
        BsplineTrajectory(NonUniformBspline(np.zeros((6, 3)), 3), flat, 0.0, 1)
