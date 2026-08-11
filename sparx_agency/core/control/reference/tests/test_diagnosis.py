"""Pin the split between "late" and "sideways", including its sign.

There is nothing stateful or closed-loop to fly here: ``decompose_error`` is one
projection of one vector, so these tests are exact identities rather than
tolerances on a flight. That is the point of the function being pure -- the two
control backends both report these numbers, a mission decides what to do about
them, and every one of those consumers is entitled to an unambiguous convention.

The convention is the deliverable. Being late is benign and being sideways is
what hits walls, so a sign error here does not degrade tracking, it inverts
which of the two the aircraft thinks it has.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.reference import decompose_error

FORWARD = (1.0, 0.0, 0.0)
"""A plan travelling along world +x, which every fixture below flies."""


def test_the_two_components_always_recombine_into_the_gap():
    """``along**2 + cross**2 == gap**2``, so a gap is fully attributable.

    Checked over random offsets and random directions rather than a hand-picked
    pair, because the property that matters is that no displacement can go
    missing between the two halves -- a decomposition that dropped part of the
    gap would let a controller believe it was closer to the plan than it is.
    """
    rng = np.random.RandomState(20260810)
    for _ in range(200):
        offset = rng.uniform(-5.0, 5.0, 3)
        direction = rng.uniform(-3.0, 3.0, 3)
        gap, along, cross = decompose_error(offset, direction)
        assert gap == pytest.approx(float(np.linalg.norm(offset)), abs=1e-12)
        assert math.hypot(along, cross) == pytest.approx(gap, abs=1e-9)
        assert cross >= 0.0


def test_an_aircraft_directly_behind_the_plan_is_all_lag():
    """Trailing the reference along its own track is being late, nothing else.

    This is the reading the along-track catch-up term acts on, and the one an
    earlier "distance to the nearest point on the curve" definition of
    cross-track got exactly backwards.
    """
    gap, along, cross = decompose_error((1.3, 0.0, 0.0), FORWARD)
    assert gap == pytest.approx(1.3)
    assert along == pytest.approx(1.3)
    assert cross == pytest.approx(0.0, abs=1e-12)


def test_an_aircraft_displaced_sideways_is_all_cross_track():
    """Off the path but on schedule: the expensive error, with no lag in it.

    Vertical counts as cross-track too -- it is perpendicular to the direction
    of travel, and a controller that treated an altitude offset as lateness
    would try to fix it by flying forwards.
    """
    gap, along, cross = decompose_error((0.0, 0.4, -0.3), FORWARD)
    assert gap == pytest.approx(0.5)
    assert along == pytest.approx(0.0, abs=1e-12)
    assert cross == pytest.approx(0.5)


def test_positive_along_track_means_late():
    """The sign convention everything downstream is built on, pinned explicitly.

    The offset argument runs **from the aircraft to the point the plan says it
    should occupy**, so an aircraft that has not got there yet produces a
    positive along-track component. Flip this and the catch-up term brakes a
    lagging aircraft and accelerates one that is already ahead of its schedule,
    which is a runaway rather than a tracking error.
    """
    # The plan's point is 1.0 m further along +x than the aircraft: late.
    _, behind, _ = decompose_error((1.0, 0.0, 0.0), FORWARD)
    assert behind > 0.0
    # The plan's point is 1.0 m *behind* the aircraft: early, and it must read
    # as a negative lag rather than as a second kind of lateness.
    _, ahead, _ = decompose_error((-1.0, 0.0, 0.0), FORWARD)
    assert ahead == pytest.approx(-1.0)


def test_a_stationary_reference_reports_the_whole_gap_as_cross_track():
    """With no direction of travel there is no such thing as being late.

    A hover point has no track to be behind, so attributing the offset to lag
    would invite the catch-up term to "recover schedule" against a reference
    that is not going anywhere. Reporting it all as cross-track is the
    conservative reading: it is the half the controller is allowed to close
    hard.
    """
    gap, along, cross = decompose_error((0.6, -0.8, 0.0), (0.0, 0.0, 0.0))
    assert gap == pytest.approx(1.0)
    assert along == 0.0
    assert cross == pytest.approx(1.0)

    # And the same just below the stationary threshold, so that numerical dust
    # on a nearly-stopped reference cannot be mistaken for a heading.
    _, dust_along, dust_cross = decompose_error((0.6, -0.8, 0.0), (1e-9, 0.0, 0.0))
    assert dust_along == 0.0
    assert dust_cross == pytest.approx(1.0)


def test_the_direction_need_not_be_normalised():
    """Callers pass a velocity, not a unit vector; only its bearing is used.

    The feed hands in the curve's own velocity, whose magnitude is the planned
    speed and varies along the route. If the magnitude leaked into the answer,
    the reported lag would scale with how fast the plan happened to be going at
    that point -- wrong by a factor that changes mid-flight.
    """
    offset = (0.9, 0.6, -0.2)
    reference = decompose_error(offset, (1.0, 0.0, 0.0))
    for scale in (1e-3, 0.25, 17.3, 1e4):
        scaled = decompose_error(offset, (scale, 0.0, 0.0))
        assert scaled == pytest.approx(reference, abs=1e-12)


def test_a_diagonal_direction_splits_the_offset_by_geometry():
    """A worked case with an answer known independently of the implementation.

    Offset (1, 1, 0) against a direction 45 degrees off it: the projection is
    ``cos(45) * sqrt(2) = 1``, and the remainder is the same again.
    """
    gap, along, cross = decompose_error((1.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    assert gap == pytest.approx(math.sqrt(2.0))
    assert along == pytest.approx(1.0)
    assert cross == pytest.approx(1.0)
