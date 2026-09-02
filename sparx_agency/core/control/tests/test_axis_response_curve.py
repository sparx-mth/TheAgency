"""Tests for the measured stick-response curve."""

import pytest

from sparx_agency.core.control.axis_response_curve import AxisResponseCurve

# A miniature expo-shaped curve in the Rooster's format.
POINTS = [(0, 0.0), (250, 0.026), (500, 0.22), (700, 0.79), (900, 1.566)]


def test_rejects_a_curve_that_does_not_start_at_zero():
    with pytest.raises(ValueError):
        AxisResponseCurve([(100, 0.1), (900, 1.5)])


def test_rejects_non_monotone_points():
    with pytest.raises(ValueError):
        AxisResponseCurve([(0, 0.0), (500, 0.5), (400, 0.6)])
    with pytest.raises(ValueError):
        AxisResponseCurve([(0, 0.0), (400, 0.5), (500, 0.4)])


def test_measured_points_round_trip_exactly():
    curve = AxisResponseCurve(POINTS)
    for counts, speed in POINTS[1:]:
        assert curve.speed_at(counts) == pytest.approx(speed)
        assert curve.axis_for(speed) == pytest.approx(counts)


def test_interpolation_is_linear_between_points():
    curve = AxisResponseCurve(POINTS)
    assert curve.speed_at(600) == pytest.approx((0.22 + 0.79) / 2.0)
    assert curve.axis_for((0.22 + 0.79) / 2.0) == pytest.approx(600.0)


def test_sign_is_carried_through_both_ways():
    curve = AxisResponseCurve(POINTS)
    assert curve.speed_at(-700) == pytest.approx(-0.79)
    assert curve.axis_for(-0.79) == pytest.approx(-700.0)


def test_requests_beyond_the_last_point_clamp_to_it():
    """The last measured point is the ceiling: never extrapolate past it."""
    curve = AxisResponseCurve(POINTS)
    assert curve.axis_for(5.0) == pytest.approx(900.0)
    assert curve.axis_for(-5.0) == pytest.approx(-900.0)
    assert curve.speed_at(1000) == pytest.approx(1.566)
    assert curve.max_counts == 900.0
    assert curve.max_speed == 1.566


def test_zero_maps_to_zero():
    curve = AxisResponseCurve(POINTS)
    assert curve.speed_at(0.0) == 0.0
    assert curve.axis_for(0.0) == 0.0


def test_the_rooster_curve_module_is_consistent():
    from sparx_agency.robots.ROBOTICAN.rooster_axis_curve import (
        ROOSTER_HORIZONTAL_CURVE,
    )
    assert ROOSTER_HORIZONTAL_CURVE.max_counts == 900.0
    # 0.8 m/s (FALCON's plan ceiling) must be comfortably under the axis cap.
    assert ROOSTER_HORIZONTAL_CURVE.axis_for(0.8) < 750.0
    # The measured 400-count level: ~0.1 m/s, nowhere near any "dead band".
    assert 0.05 < ROOSTER_HORIZONTAL_CURVE.speed_at(400) < 0.15
