"""The de Boor evaluator, checked against closed-form cubic B-spline basis functions.

The point of these tests is not that the code runs -- it is that this evaluator
produces the *same curve* FALCON's C++ would, because the two are evaluating one
trajectory on opposite sides of a socket and any disagreement shows up as a
tracking error nobody can attribute.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.trajectories.bspline import NonUniformBspline


def _uniform_cubic_reference(control_points, s):
    """One span of a uniform cubic B-spline, from the textbook basis functions.

    An independent implementation with no code shared with the thing it checks.

    Args:
        control_points: The four control points influencing this span.
        s: Local parameter in [0, 1].

    Returns:
        The point on the curve.
    """
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in control_points)
    basis = (
        (1.0 - s) ** 3,
        3.0 * s ** 3 - 6.0 * s ** 2 + 4.0,
        -3.0 * s ** 3 + 3.0 * s ** 2 + 3.0 * s + 1.0,
        s ** 3,
    )
    return (basis[0] * p0 + basis[1] * p1 + basis[2] * p2 + basis[3] * p3) / 6.0


def test_matches_closed_form_cubic_basis():
    """Every point of the first span agrees with the textbook basis functions."""
    points = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 1.5], [3.0, 1.0, 2.0], [4.5, -1.0, 1.0]])
    spline = NonUniformBspline(points, 3, knot_span=1.0)
    for s in np.linspace(0.0, 1.0, 21):
        expected = _uniform_cubic_reference(points, s)
        assert np.allclose(spline.evaluate_at_time(s), expected, atol=1e-12)


def test_duration_of_a_uniform_spline():
    """A uniform curve lasts one knot span per control point past the degree."""
    points = np.zeros((7, 3))
    spline = NonUniformBspline(points, 3, knot_span=0.4)
    assert spline.duration == pytest.approx((7 - 3) * 0.4)


def test_straight_line_control_points_give_constant_velocity():
    """Evenly spaced collinear control points are a constant-velocity line.

    The strongest single check on the indexing: any off-by-one in the knot
    vector shows up as a curve that is not straight or not evenly timed.
    """
    points = np.array([[float(i), 0.0, 0.0] for i in range(8)])
    spline = NonUniformBspline(points, 3, knot_span=0.5)
    velocity = spline.derivative()
    for t in np.linspace(0.0, spline.duration, 25):
        assert velocity.evaluate_at_time(t) == pytest.approx([2.0, 0.0, 0.0], abs=1e-9)


def test_derivative_agrees_with_finite_difference():
    """The analytic derivative curve matches a numerical one everywhere inside."""
    rng = np.random.RandomState(7)
    points = rng.uniform(-4.0, 4.0, size=(9, 3))
    spline = NonUniformBspline(points, 3, knot_span=0.35)
    velocity = spline.derivative()
    step = 1e-6
    for t in np.linspace(0.2, spline.duration - 0.2, 15):
        numerical = ((spline.evaluate_at_time(t + step) - spline.evaluate_at_time(t - step))
                     / (2.0 * step))
        assert velocity.evaluate_at_time(t) == pytest.approx(numerical, abs=1e-5)


def test_second_and_third_derivatives_exist_and_agree():
    """Acceleration and jerk are available and consistent with the curve above.

    Sampled at knot-interval *midpoints*. The jerk of a cubic is degree 0 and so
    genuinely piecewise constant -- it steps at every knot, and a central
    difference straddling one of those steps measures the average of two
    segments rather than either.
    """
    rng = np.random.RandomState(11)
    span = 0.3
    points = rng.uniform(-2.0, 2.0, size=(10, 3))
    spline = NonUniformBspline(points, 3, knot_span=span)
    acceleration = spline.derivative().derivative()
    jerk = acceleration.derivative()
    assert jerk.degree == 0
    step = 1e-6
    midpoints = np.arange(0.5 * span, spline.duration - 0.5 * span, span)
    assert midpoints.size >= 5
    for t in midpoints:
        numerical = ((acceleration.evaluate_at_time(t + step)
                      - acceleration.evaluate_at_time(t - step)) / (2.0 * step))
        assert jerk.evaluate_at_time(t) == pytest.approx(numerical, abs=1e-3)


def test_evaluation_is_clamped_outside_the_span():
    """Past either end the endpoint is held rather than extrapolated."""
    points = np.array([[float(i), float(i * i), 0.0] for i in range(6)])
    spline = NonUniformBspline(points, 3, knot_span=0.5)
    assert np.allclose(spline.evaluate_at_time(-5.0), spline.evaluate_at_time(0.0))
    assert np.allclose(spline.evaluate_at_time(spline.duration + 5.0),
                       spline.evaluate_at_time(spline.duration))


def test_explicit_knots_replace_the_uniform_vector():
    """A transmitted knot vector of the right length is accepted and used.

    FALCON reparameterises the position curve to respect the velocity limit, so
    its knots are *not* evenly spaced -- ignoring them would fly the right shape
    at the wrong speed.
    """
    points = np.array([[float(i), 0.0, 0.0] for i in range(6)])
    stretched = np.array([-1.5, -1.0, -0.5, 0.0, 1.0, 2.5, 4.5, 5.0, 5.5, 6.0])
    spline = NonUniformBspline(points, 3, knot_span=0.5, knots=stretched)
    assert spline.duration == pytest.approx(4.5)
    uniform = NonUniformBspline(points, 3, knot_span=0.5)
    assert not np.allclose(spline.evaluate_at_time(1.0), uniform.evaluate_at_time(1.0))


def test_wrong_knot_count_is_refused():
    """A knot vector of the wrong length is a corrupt message, not a warning."""
    points = np.zeros((6, 3))
    with pytest.raises(ValueError, match="expected 10 knots"):
        NonUniformBspline(points, 3, knot_span=0.5, knots=np.zeros(9))


def test_too_few_control_points_is_refused():
    """A degree-3 curve needs four control points."""
    with pytest.raises(ValueError, match="at least 4 control points"):
        NonUniformBspline(np.zeros((3, 3)), 3)


def test_degree_zero_has_no_derivative():
    """The jerk curve of a cubic is degree 0 and cannot be differentiated again."""
    spline = NonUniformBspline(np.zeros((4, 3)), 0)
    with pytest.raises(ValueError, match="degree-0"):
        spline.derivative()


def test_one_dimensional_curves_are_supported():
    """The yaw curve is 1D and travels through the same class."""
    spline = NonUniformBspline([0.0, 0.5, 1.0, 1.5, 2.0], 3, knot_span=0.25)
    assert spline.dimension == 1
    assert spline.evaluate_at_time(0.0).shape == (1,)
