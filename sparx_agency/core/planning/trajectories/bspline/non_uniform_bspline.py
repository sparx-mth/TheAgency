"""A non-uniform B-spline, evaluated by de Boor's algorithm.

This is a port of FALCON's ``fast_planner::NonUniformBspline`` (its
``trajectory`` package), and it is deliberately a *faithful* one rather than a
tidier reimplementation: the whole point is to evaluate, on this side of the
link, exactly the curve FALCON's own ``traj_server`` would have evaluated on
that side. Any disagreement between the two is a silent tracking error that
looks like a control problem, so the indexing below mirrors the C++ line for
line and the tests check the two against each other's construction rules.

Why evaluate it here at all, when ``traj_server`` already publishes 100 Hz
samples of it: a sample stream can only answer "where should I be at time *t*".
Holding the curve answers three further questions that a tracking controller
needs and a sample cannot provide --

* **jerk**, the third derivative, which is the attitude feedforward and simply
  is not in the ``PositionCommand`` message;
* **the nearest point on the curve**, which is what stops a lagging aircraft
  cutting corners (see :mod:`~.projection`);
* **any time at all**, on the evaluator's own clock, which decouples the
  aircraft from ``traj_server``'s wall-clock timer.

Frames and units are whatever the control points are given in; nothing here is
aware of a coordinate convention.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class NonUniformBspline:
    """A B-spline curve of arbitrary degree and dimension.

    The curve is defined by its control points, its degree and its knot vector.
    Construction takes a uniform knot span and an explicit knot vector may then
    replace it -- which is exactly how FALCON transmits a trajectory, and why
    the two are separable here.

    Note that ``n``/``m`` are derived from the *control point count* at
    construction and are **not** recomputed when the knots are replaced. That
    matches the C++, and it is load-bearing: the transmitted knot vector is
    longer than the uniform one it replaces on a derivative curve.

    Args:
        control_points: ``(N, D)`` control points -- N of them, in D dimensions.
        degree: Spline degree (FALCON publishes 3 for position and yaw).
        knot_span: Spacing of the uniform knot vector built at construction.
        knots: Optional explicit knot vector, replacing the uniform one. Must
            have ``N + degree + 1`` entries.

    Raises:
        ValueError: If there are too few control points for the degree, if the
            knot span is not positive, or if an explicit knot vector is the
            wrong length.
    """

    def __init__(self, control_points, degree, knot_span=1.0, knots=None):
        # type: (Sequence, int, float, Optional[Sequence]) -> None
        points = np.asarray(control_points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if points.ndim != 2:
            raise ValueError("control_points must be (N, D), got shape %r" % (points.shape,))
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be >= 0, got %r" % (degree,))
        if points.shape[0] < degree + 1:
            raise ValueError(
                "a degree-%d spline needs at least %d control points, got %d"
                % (degree, degree + 1, points.shape[0]))
        if knot_span <= 0.0:
            raise ValueError("knot_span must be > 0, got %r" % (knot_span,))

        self.control_points = points
        self.degree = degree
        self.knot_span = float(knot_span)
        # n and m follow the C++ naming: n is the last control-point index and
        # m the last knot index, so the knot vector holds m + 1 entries.
        self._n = points.shape[0] - 1
        self._m = self._n + degree + 1
        self.knots = self._uniform_knots()
        if knots is not None:
            self.set_knots(knots)

    def _uniform_knots(self):
        # type: () -> np.ndarray
        """The uniform knot vector, starting at ``-degree * knot_span``.

        The negative start is what makes ``t = 0`` mean "the beginning of the
        usable span" rather than "the beginning of the knot vector" -- see
        :meth:`evaluate_at_time`.
        """
        knots = np.empty(self._m + 1, dtype=float)
        for index in range(self._m + 1):
            if index <= self.degree:
                knots[index] = float(-self.degree + index) * self.knot_span
            else:
                knots[index] = knots[index - 1] + self.knot_span
        return knots

    def set_knots(self, knots):
        # type: (Sequence) -> None
        """Replace the knot vector, keeping the control points and degree.

        Args:
            knots: ``N + degree + 1`` non-decreasing knot values.

        Raises:
            ValueError: If the length is wrong.
        """
        values = np.asarray(knots, dtype=float).reshape(-1)
        if values.size != self._m + 1:
            raise ValueError("expected %d knots for %d control points at degree %d, got %d"
                             % (self._m + 1, self._n + 1, self.degree, values.size))
        self.knots = values

    @property
    def dimension(self):
        # type: () -> int
        """How many components each control point has."""
        return int(self.control_points.shape[1])

    @property
    def duration(self):
        # type: () -> float
        """Length of the usable parameter span, in the knot vector's units.

        This is FALCON's ``getTimeSum()``. The curve is only defined over
        ``[knots[degree], knots[m - degree]]``; outside it there are too few
        control points to influence the result.
        """
        return float(self.knots[self._m - self.degree] - self.knots[self.degree])

    def evaluate(self, u):
        # type: (float) -> np.ndarray
        """Evaluate the curve at a raw knot-vector parameter.

        The parameter is clamped into the usable span rather than raising, which
        is what the C++ does and what a controller wants: a reference that runs
        a millisecond past the end of a trajectory should hold the endpoint, not
        fail.

        Args:
            u: Parameter in the knot vector's own units.

        Returns:
            A ``(D,)`` array -- the point on the curve.
        """
        low = self.knots[self.degree]
        high = self.knots[self._m - self.degree]
        bounded = min(max(low, float(u)), high)

        # Find the knot span [knots[k], knots[k+1]) containing the parameter.
        span = self.degree
        while self.knots[span + 1] < bounded and span < self._m - self.degree - 1:
            span += 1

        # de Boor's algorithm: repeatedly interpolate the degree + 1 control
        # points that influence this span, until one point remains.
        working = [self.control_points[span - self.degree + i].copy()
                   for i in range(self.degree + 1)]
        for level in range(1, self.degree + 1):
            for i in range(self.degree, level - 1, -1):
                lower = self.knots[i + span - self.degree]
                upper = self.knots[i + 1 + span - level]
                spread = upper - lower
                # A repeated knot collapses the span; the C++ would divide by
                # zero here, and taking the left point is the limit value.
                alpha = 0.0 if spread <= 0.0 else (bounded - lower) / spread
                working[i] = (1.0 - alpha) * working[i - 1] + alpha * working[i]
        return working[self.degree]

    def evaluate_at_time(self, t):
        # type: (float) -> np.ndarray
        """Evaluate at a time measured from the start of the usable span.

        This is the form every caller wants: ``t = 0`` is the beginning of the
        trajectory and ``t = duration`` its end, whatever the knot vector's
        absolute values happen to be.

        Args:
            t: Seconds since the trajectory started.

        Returns:
            A ``(D,)`` array.
        """
        return self.evaluate(float(t) + self.knots[self.degree])

    def derivative(self):
        # type: () -> NonUniformBspline
        """The curve's derivative, itself a B-spline of one degree lower.

        Exact, not a finite difference. Cheap enough to take three times when a
        trajectory arrives and then never again.

        Raises:
            ValueError: If the curve is already degree 0, which has no
                B-spline derivative.
        """
        if self.degree < 1:
            raise ValueError("a degree-0 spline has no B-spline derivative")
        rows = self.control_points.shape[0] - 1
        points = np.zeros((rows, self.dimension), dtype=float)
        for i in range(rows):
            spread = self.knots[i + self.degree + 1] - self.knots[i + 1]
            if spread <= 0.0:
                continue
            points[i] = (self.degree
                         * (self.control_points[i + 1] - self.control_points[i]) / spread)
        # The derivative lives on the interior of the parent's knot vector, so
        # one knot comes off each end.
        return NonUniformBspline(points, self.degree - 1, self.knot_span,
                                 knots=self.knots[1:-1])
