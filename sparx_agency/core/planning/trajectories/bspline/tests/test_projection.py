"""Nearest-point projection: the thing that stops a lagging aircraft cutting corners."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.trajectories.bspline import (
    BsplineTrajectory, ProjectionParams, TrajectoryProjector,
)


def _line(length=14, spacing=0.5, knot_span=0.25):
    """A straight trajectory along +x at 2 m/s."""
    points = [[float(i) * spacing, 0.0, 1.0] for i in range(length)]
    knots = [(-3 + i) * knot_span for i in range(length + 4)]
    return BsplineTrajectory.from_falcon(3, knots, points, [0.0] * 6, knot_span, 0.0, 1)


def _corner(knot_span=0.3):
    """A trajectory that runs east then turns hard north."""
    points = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0],
              [3.0, 1.0, 1.0], [3.0, 2.0, 1.0], [3.0, 3.0, 1.0], [3.0, 4.0, 1.0]]
    knots = [(-3 + i) * knot_span for i in range(len(points) + 4)]
    return BsplineTrajectory.from_falcon(3, knots, points, [0.0] * 6, knot_span, 0.0, 1)


def test_a_point_on_the_curve_projects_to_itself():
    """The trivial case, and the one every other result is measured against."""
    trajectory = _line()
    projector = TrajectoryProjector()
    for t in (0.3, 0.9, 1.4):
        projector.reset(t - 0.2)
        on_curve = trajectory.position_at(t)
        assert projector.project(trajectory, on_curve) == pytest.approx(t, abs=1e-3)


def test_perpendicular_offset_projects_to_the_foot_of_the_perpendicular():
    """Being sideways of the path does not move the projection along it.

    This is the whole point: cross-track error is reported as cross-track,
    rather than being smeared into a reference further down the route.
    """
    trajectory = _line()
    projector = TrajectoryProjector()
    projector.reset(0.8)
    foot = trajectory.position_at(1.0)
    offset = foot + np.array([0.0, 0.7, 0.0])
    assert projector.project(trajectory, offset) == pytest.approx(1.0, abs=5e-3)


def test_a_lagging_aircraft_projects_behind_the_time_based_reference():
    """The correction that removes corner cutting, stated as a test.

    The aircraft sits where the plan wanted it half a second ago. Time-indexed,
    the reference is already half a second further on and the error vector
    points down the route; projected, the reference is right where the aircraft
    is and the error is honestly zero.
    """
    trajectory = _line()
    elapsed = 1.5
    lagging = trajectory.position_at(elapsed - 0.5)
    projector = TrajectoryProjector()
    projector.reset(elapsed)
    projected = projector.project(trajectory, lagging)
    assert projected == pytest.approx(elapsed - 0.5, abs=1e-2)
    assert projected < elapsed


def test_the_inside_of_a_corner_is_not_a_shortcut():
    """A drone cutting the inside of a turn is pulled back onto the arc.

    Time-indexed tracking rewards the shortcut -- the reference is already round
    the corner, so the error points across it. Projection puts the reference on
    the corner itself.
    """
    trajectory = _corner()
    inside = np.array([2.7, 0.7, 1.0])
    projector = TrajectoryProjector()
    projector.reset(0.9)
    projected = projector.project(trajectory, inside)
    nearest = trajectory.position_at(projected)
    # Whatever point it picked, nothing on the curve is closer.
    scan = np.linspace(0.0, trajectory.duration, 400)
    best = min(float(np.linalg.norm(trajectory.position_at(float(t)) - inside)) for t in scan)
    assert float(np.linalg.norm(nearest - inside)) == pytest.approx(best, abs=2e-2)


def test_the_window_stops_a_self_crossing_route_snapping_backwards():
    """On a route that revisits a room, the search stays local.

    A global nearest-point search would hand back the leg flown a minute ago and
    fly the aircraft back down it.
    """
    span = 0.3
    points = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0],
              [3.0, 1.5, 1.0], [2.0, 1.5, 1.0], [1.0, 1.5, 1.0], [0.0, 1.5, 1.0],
              [0.0, 0.2, 1.0], [1.0, 0.2, 1.0], [2.0, 0.2, 1.0], [3.0, 0.2, 1.0]]
    knots = [(-3 + i) * span for i in range(len(points) + 4)]
    trajectory = BsplineTrajectory.from_falcon(3, knots, points, [0.0] * 6, span, 0.0, 1)

    late = trajectory.duration - 0.4
    here = trajectory.position_at(late)
    projector = TrajectoryProjector()
    projector.reset(late)
    assert projector.project(trajectory, here) == pytest.approx(late, abs=1e-2)


def test_the_projection_advances_with_the_aircraft():
    """Stepping along the curve gives a monotonically advancing projection."""
    trajectory = _line()
    projector = TrajectoryProjector()
    projector.reset(0.0)
    previous = -1.0
    for t in np.arange(0.0, trajectory.duration, 0.1):
        result = projector.project(trajectory, trajectory.position_at(float(t)))
        assert result >= previous - 1e-6
        previous = result


def test_reference_time_leans_forward_and_clamps_at_the_end():
    """The lookahead is added on top of the projection and never overruns."""
    trajectory = _line()
    projector = TrajectoryProjector(ProjectionParams(lookahead_s=0.2))
    projector.reset(1.0)
    on_curve = trajectory.position_at(1.0)
    assert projector.reference_time(trajectory, on_curve) == pytest.approx(1.2, abs=1e-2)

    projector.reset(trajectory.duration)
    end = trajectory.position_at(trajectory.duration)
    assert projector.reference_time(trajectory, end) == pytest.approx(trajectory.duration)


def test_bad_parameters_are_refused():
    """The search window has to be a window."""
    with pytest.raises(ValueError, match="search_ahead_s"):
        ProjectionParams(search_ahead_s=0.0)
    with pytest.raises(ValueError, match="coarse_step_s"):
        ProjectionParams(coarse_step_s=-0.1)
    with pytest.raises(ValueError, match="lookahead_s"):
        ProjectionParams(lookahead_s=-0.1)
