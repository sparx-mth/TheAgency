"""The plan-holding half of the control stack, tested where it can lie.

``TrajectoryFeed`` has no gains and commands nothing, so there is no flight to
measure here. What it can get wrong is narrower and nastier: adopting a curve
before it starts, adopting a stale one, keeping a dead one alive, forgetting to
let go of one, or -- the regression that gives this module most of its length --
reporting the *wrong half* of the error at the one moment the aircraft is
furthest behind.

Trajectories are built the way ``velocity_servo``'s tests build them, from
control points laid out by hand, so the curve's geometry is known and an
assertion can be exact rather than a tolerance around a fixture.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.reference import ReferenceParams, TrajectoryFeed
from sparx_agency.core.planning.trajectories.bspline import (
    BsplineTrajectory, NonUniformBspline,
)


def straight_trajectory(speed=0.6, seconds=6.0, altitude=1.5, knot_dt=0.5,
                        heading=0.0, start_time_s=0.0, traj_id=1):
    # type: (float, float, float, float, float, float, int) -> BsplineTrajectory
    """A constant-speed run along a fixed bearing, at a fixed heading.

    Evenly spaced control points, so the velocity curve is constant and the
    direction of travel is the same everywhere -- including at the very end,
    which is what the past-the-end test needs to be unambiguous about.
    """
    count = int(seconds / knot_dt) + 4
    points = np.zeros((count, 3), dtype=float)
    points[:, 0] = np.arange(count) * speed * knot_dt * math.cos(heading)
    points[:, 1] = np.arange(count) * speed * knot_dt * math.sin(heading)
    points[:, 2] = altitude
    yaw_points = np.full((count, 1), heading, dtype=float)
    return BsplineTrajectory(NonUniformBspline(points, 3, knot_dt),
                             NonUniformBspline(yaw_points, 3, knot_dt),
                             start_time_s=start_time_s, traj_id=traj_id)


def test_a_new_trajectory_is_queued_rather_than_adopted():
    """Handing a curve over must not change what is being flown that instant.

    FALCON begins each curve a planning-time in the future so it joins smoothly
    onto the one still being flown. Adopting on arrival would jump the reference
    to a point the aircraft has not reached, and do it several times a second.
    """
    feed = TrajectoryFeed()
    assert feed.set_trajectory(straight_trajectory(traj_id=1, start_time_s=0.0))
    assert feed.trajectory is None
    assert feed.trajectory_id == -1


def test_a_queued_trajectory_waits_for_its_own_start_time():
    """Promotion happens on the plan's clock, not on arrival.

    The first curve is the exception and deliberately so: with nothing being
    flown there is no reference to jump *from*, so it is adopted at once and
    :meth:`TrajectoryFeed.usable` is what keeps the aircraft off it until its
    start time comes up. Every curve after that has something to protect and
    waits.
    """
    feed = TrajectoryFeed()
    feed.set_trajectory(straight_trajectory(traj_id=1, start_time_s=0.0))
    assert feed.promote(0.0)
    assert feed.trajectory_id == 1

    feed.set_trajectory(straight_trajectory(traj_id=2, start_time_s=4.0))
    assert not feed.promote(3.5)
    assert feed.trajectory_id == 1, "adopted a curve that has not started"
    assert not feed.promote(3.999)
    assert feed.trajectory_id == 1

    assert feed.promote(4.0)
    assert feed.trajectory_id == 2


def test_promote_reports_the_swap_on_exactly_the_tick_it_happens():
    """Callers drop derivative state on a True, so a stuck True re-drops it.

    The new curve is a different parameterisation, so a backend uses this return
    to discard anything it had differenced across ticks. Returning True while
    nothing changed would throw that away every tick and leave the backend
    permanently in its first-tick state.
    """
    feed = TrajectoryFeed()
    assert not feed.promote(0.0), "promoted with nothing queued"

    feed.set_trajectory(straight_trajectory(traj_id=1, start_time_s=0.0))
    assert feed.promote(0.1)
    assert not feed.promote(0.2)
    assert not feed.promote(5.0)


def test_a_trajectory_that_is_not_newer_is_refused():
    """A re-send or a misordered message would restart a curve mid-flight."""
    feed = TrajectoryFeed()
    assert feed.set_trajectory(straight_trajectory(traj_id=5, start_time_s=0.0))
    assert not feed.set_trajectory(straight_trajectory(traj_id=5, start_time_s=0.0))
    assert not feed.set_trajectory(straight_trajectory(traj_id=4, start_time_s=0.0))

    feed.promote(0.0)
    assert feed.trajectory_id == 5
    # Refused against what is being flown, too, not only against the queue.
    assert not feed.set_trajectory(straight_trajectory(traj_id=5, start_time_s=1.0))
    assert feed.set_trajectory(straight_trajectory(traj_id=6, start_time_s=1.0))


def test_a_refused_trajectory_does_not_displace_the_queued_one():
    """Rejection must be inert, not destructive.

    The comparison is against the newest curve *held*, which is the queued one
    when there is a queue. If a refused message still overwrote the queue, a
    duplicate of an old id -- exactly what a re-send is -- would silently
    discard the replan waiting to be flown.
    """
    feed = TrajectoryFeed()
    feed.set_trajectory(straight_trajectory(traj_id=7, start_time_s=0.0))
    assert not feed.set_trajectory(straight_trajectory(traj_id=7, start_time_s=99.0))
    assert not feed.set_trajectory(straight_trajectory(traj_id=3, start_time_s=99.0))

    assert feed.promote(0.0)
    assert feed.trajectory_id == 7
    # The start time is how the two id-7 curves are told apart: the original is
    # still the one loaded.
    assert feed.trajectory.start_time_s == pytest.approx(0.0)


def test_usable_spans_the_curve_and_a_bounded_grace_period():
    """When the plan is worth flying, at the three edges that matter.

    Before its start time there is nothing to fly to. Past its end the aircraft
    keeps flying to the final point for ``max_trajectory_age_s``, which covers
    the normal gap between replans. Beyond that the planner has stopped talking,
    and flying to a stale endpoint is a guess about a world that has moved.
    """
    feed = TrajectoryFeed(ReferenceParams(max_trajectory_age_s=2.0))
    plan = straight_trajectory(start_time_s=10.0, traj_id=1)
    feed.set_trajectory(plan)
    feed.promote(10.0)
    duration = plan.duration

    assert not feed.usable(9.5), "flew a curve that had not started"
    assert feed.usable(10.0)
    assert feed.usable(10.0 + duration / 2.0)
    assert feed.usable(10.0 + duration + 1.9)
    assert not feed.usable(10.0 + duration + 2.1)


def test_usable_is_false_with_no_trajectory_at_all():
    """The empty state is a normal one -- before the first plan arrives."""
    assert not TrajectoryFeed().usable(0.0)


def test_resolve_returns_nothing_when_no_plan_is_usable():
    """A backend must be told there is no reference, not handed a stale one.

    None is the signal to hold station. Returning the last known sample instead
    would have the aircraft chase a point from a plan that has expired.
    """
    feed = TrajectoryFeed()
    assert feed.resolve((0.0, 0.0, 1.5), 0.0) is None

    plan = straight_trajectory(start_time_s=10.0)
    feed.set_trajectory(plan)
    feed.promote(10.0)
    assert feed.resolve((0.0, 0.0, 1.5), 9.0) is None, "resolved before the start time"
    assert feed.resolve((0.0, 0.0, 1.5), 11.0) is not None
    assert feed.resolve((0.0, 0.0, 1.5), 10.0 + plan.duration + 5.0) is None


def test_reset_drops_the_queued_curve_as_well_as_the_live_one():
    """"Stop flying the plan" must not mean "stop until the next tick".

    Clearing only the live curve leaves the queued one to be promoted on the
    very next call, so the one method a caller has for standing the aircraft
    down would hand the plan straight back to it.
    """
    feed = TrajectoryFeed()
    feed.set_trajectory(straight_trajectory(traj_id=1, start_time_s=0.0))
    feed.promote(0.0)
    feed.set_trajectory(straight_trajectory(traj_id=2, start_time_s=1.0))
    assert feed.trajectory_id == 1

    feed.reset()
    assert feed.trajectory_id == -1
    assert feed.trajectory is None
    assert not feed.promote(5.0), "the queued curve survived the reset"
    assert feed.trajectory_id == -1
    assert not feed.usable(5.0)
    assert feed.resolve((0.0, 0.0, 1.5), 5.0) is None


def test_an_offset_from_the_plan_splits_into_lag_and_cross_track():
    """The nominal case, so the past-the-end one below is a comparison.

    Mid-curve the aircraft is placed 0.5 m behind where the schedule says it
    should be and 0.3 m to the left of the route: those must come back as lag
    and cross-track respectively, and recombine into the gap.
    """
    feed = TrajectoryFeed()
    plan = straight_trajectory(start_time_s=0.0)
    feed.set_trajectory(plan)
    feed.promote(0.0)

    on_schedule = plan.position_at(3.0)
    sample = feed.resolve((on_schedule[0] - 0.5, on_schedule[1] - 0.3, on_schedule[2]), 3.0)
    assert sample.along_track_lag_m == pytest.approx(0.5, abs=1e-6)
    assert sample.cross_track_error_m == pytest.approx(0.3, abs=1e-6)
    assert sample.gap_m == pytest.approx(math.hypot(0.5, 0.3), abs=1e-6)
    assert not sample.past_end
    assert sample.trajectory_id == 1
    assert sample.duration_s == pytest.approx(plan.duration)


def test_lag_past_the_end_of_a_curve_is_not_reported_as_cross_track():
    """The regression: the error split inverted exactly where it matters most.

    ``BsplineTrajectory.sample`` zeroes every derivative outside the curve's
    span -- correctly, so an overrunning aircraft brakes to a hover on the final
    point instead of extrapolating into unmapped space. But the error
    decomposition used to take its direction of travel from that sampled
    velocity, so past the end it had none, and the fallback for "no direction"
    is to report the whole gap as cross-track.

    The result: an aircraft trailing a finished trajectory -- which is the
    normal state during the second between replans, and the state in which the
    aircraft is furthest behind -- reported its entire lag as the one error the
    docs call the one that flies into walls. Benign lateness read as a metre of
    sideways departure, and any consumer acting on cross-track acted on a lie.

    The fix reads the tangent off the velocity *curve*, which is defined
    everywhere. Here the aircraft sits 0.8 m directly behind the final point,
    two seconds after the schedule ran out, so the honest answer is unambiguous:
    all lag, no cross-track.
    """
    feed = TrajectoryFeed(ReferenceParams(max_trajectory_age_s=3.0))
    plan = straight_trajectory(start_time_s=0.0)
    feed.set_trajectory(plan)
    feed.promote(0.0)

    now = plan.duration + 2.0
    assert feed.usable(now), "fixture is past the grace period, not past the end"
    endpoint = plan.position_at(plan.duration)
    sample = feed.resolve((endpoint[0] - 0.8, endpoint[1], endpoint[2]), now)

    assert sample.past_end
    assert sample.along_track_lag_m == pytest.approx(0.8, abs=1e-6)
    assert sample.cross_track_error_m == pytest.approx(0.0, abs=1e-6)
    assert sample.gap_m == pytest.approx(0.8, abs=1e-6)
    # The reference itself is pinned to the stopped endpoint, so the aircraft is
    # pulled to the end of the plan rather than through it at cruise.
    assert sample.reference_time_s == pytest.approx(plan.duration)
    assert sample.velocity == pytest.approx((0.0, 0.0, 0.0))


def test_the_past_end_split_holds_on_a_route_that_does_not_run_along_an_axis():
    """The same regression on a diagonal, so no axis alignment is doing the work.

    A curve along +x can be split correctly by accident -- a bug that returned
    the x component would pass. On a 40-degree bearing only a genuine tangent
    gives the right answer.
    """
    heading = math.radians(40.0)
    feed = TrajectoryFeed(ReferenceParams(max_trajectory_age_s=3.0))
    plan = straight_trajectory(heading=heading, start_time_s=0.0)
    feed.set_trajectory(plan)
    feed.promote(0.0)

    endpoint = plan.position_at(plan.duration)
    behind = (endpoint[0] - 0.8 * math.cos(heading),
              endpoint[1] - 0.8 * math.sin(heading),
              endpoint[2])
    sample = feed.resolve(behind, plan.duration + 1.0)
    assert sample.along_track_lag_m == pytest.approx(0.8, abs=1e-6)
    assert sample.cross_track_error_m == pytest.approx(0.0, abs=1e-6)


def test_the_reference_time_is_reported_with_projection_disabled():
    """``last_reference_time`` must be truthful in both indexing modes.

    With projection off the reference is the point at the current time, and the
    projector is never asked for anything -- so unless it is re-anchored to that
    answer it keeps reporting the 0.0 it was constructed with, for the entire
    flight. Everything that reads this number for logging, telemetry or a
    progress check then sees an aircraft permanently at the start of its plan.
    """
    feed = TrajectoryFeed(ReferenceParams(use_projection=False))
    plan = straight_trajectory(start_time_s=10.0)
    feed.set_trajectory(plan)
    feed.promote(10.0)
    assert feed.last_reference_time == pytest.approx(0.0)

    for elapsed in (0.5, 1.7, 3.2):
        on_schedule = plan.position_at(elapsed)
        sample = feed.resolve(on_schedule, 10.0 + elapsed)
        assert sample.reference_time_s == pytest.approx(elapsed, abs=1e-9)
        assert feed.last_reference_time == pytest.approx(elapsed, abs=1e-9)

    # Past the end it tracks the clamp rather than the elapsed time: there is no
    # curve beyond the duration to have a position on.
    feed.resolve(plan.position_at(plan.duration), 10.0 + plan.duration + 1.0)
    assert feed.last_reference_time == pytest.approx(plan.duration, abs=1e-9)


def test_projection_puts_the_reference_where_the_aircraft_is_not_where_the_clock_is():
    """The two indexing modes must actually differ, or the flag is decoration.

    An aircraft held back on the route has a reference at its own position under
    projection and one seconds further down the route under time indexing. That
    difference is the entire reason the parameter exists.
    """
    plan = straight_trajectory(start_time_s=0.0)
    stalled = plan.position_at(1.0)

    projected = TrajectoryFeed(ReferenceParams(use_projection=True))
    projected.set_trajectory(plan)
    projected.promote(0.0)
    # Walk the projector forward the way a flight would, so its search window
    # sits where the aircraft is rather than at the start of the curve.
    for tick in range(1, 41):
        projected.resolve(plan.position_at(min(0.025 * tick, 1.0)), 0.025 * tick)
    near = projected.resolve(stalled, 3.0)

    timed = TrajectoryFeed(ReferenceParams(use_projection=False))
    timed.set_trajectory(plan)
    timed.promote(0.0)
    ahead = timed.resolve(stalled, 3.0)

    assert near.reference_time_s == pytest.approx(1.0, abs=0.05)
    assert ahead.reference_time_s == pytest.approx(3.0, abs=1e-9)
    # Both report the same schedule lag: that is measured against the point the
    # plan names for *now*, and does not depend on how the reference is indexed.
    assert near.along_track_lag_m == pytest.approx(ahead.along_track_lag_m, abs=1e-6)


def test_promotion_re_anchors_the_projection_onto_the_new_curve():
    """A new curve is a new parameterisation; last tick's position is meaningless.

    Carrying the old projected time over would open the search window at an
    arbitrary point of the replacement -- and the window is deliberately narrow,
    so the search cannot walk back out of a bad anchor.
    """
    feed = TrajectoryFeed()
    first = straight_trajectory(traj_id=1, start_time_s=0.0)
    feed.set_trajectory(first)
    feed.promote(0.0)
    for tick in range(1, 121):
        feed.resolve(first.position_at(0.025 * tick), 0.025 * tick)
    assert feed.last_reference_time > 1.0

    feed.set_trajectory(straight_trajectory(traj_id=2, start_time_s=3.0))
    assert feed.promote(3.0)
    assert feed.last_reference_time == pytest.approx(0.0)
