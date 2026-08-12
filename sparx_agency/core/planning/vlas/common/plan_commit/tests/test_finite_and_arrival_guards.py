"""The guards that keep a bad number from being flown, and arrival from being cheap.

Two families, both found by adversarial review of the commitment executor:

* **Finiteness.** A policy head that has diverged, an estimator that has lost
  its fix, or an uninitialised pose message all deliver NaN. Unchecked it
  reaches the follower as a ``(nan, nan)`` setpoint with ``replan_reason`` of
  ``None`` -- "keep flying toward this" -- which in velocity control is a lost
  aircraft. The invariant is that a non-finite value never survives as far as a
  target.
* **Arrival.** Being near the commit point is only half of having flown a
  commitment. The other half is arc, and the allowance has to scale, or the
  shortest legal commitment is handed back mostly unflown.
"""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.vlas.common.plan_commit.committed_plan import (
    CommittedPlan,
    anchor_plan,
)
from sparx_agency.core.planning.vlas.common.plan_commit.executor import (
    FLOWN,
    CommitSpec,
    PlanCommitExecutor,
)


def straight(count=24, step=0.2):
    """A body-frame prediction running straight forward, NavDP's shape and span."""
    return np.stack([np.arange(1, count + 1) * step, np.zeros(count)], axis=1)


def flying(**overrides):
    """An executor holding a straight commitment, already asked once."""
    engine = PlanCommitExecutor(CommitSpec(**overrides))
    engine.mark_attempt(0.0)
    engine.commit(straight(), (0.0, 0.0, 0.0), 0.0)
    return engine


# ── the plan ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_waypoint_is_refused(bad):
    """The policy is the likeliest source: a diverged head emits NaN, not zeros."""
    body = straight()
    body[5, 1] = bad
    with pytest.raises(ValueError):
        anchor_plan(body, (0.0, 0.0, 0.0), 0.0, 0.5)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_non_finite_anchor_pose_is_refused(index):
    """A NaN yaw rotates every waypoint to NaN; ``Pose2D`` guards the same triple."""
    pose = [0.0, 0.0, 0.0]
    pose[index] = float("nan")
    with pytest.raises(ValueError):
        anchor_plan(straight(), pose, 0.0, 0.5)


def test_a_non_finite_clock_is_refused():
    """``now - issued > max_commit_s`` is False forever against NaN, which deletes
    EXPIRED -- the only guard that ends a commitment whose arc has stalled."""
    with pytest.raises(ValueError):
        anchor_plan(straight(), (0.0, 0.0, 0.0), float("nan"), 0.5)


def test_the_invariant_belongs_to_the_type_not_the_factory():
    """``CommittedPlan`` is exported, so constructing one directly must not be a
    way around the check that ``anchor_plan`` performs."""
    with pytest.raises(ValueError):
        CommittedPlan(np.array([[0.0, 0.0], [0.2, 0.0], [float("nan"), 0.0]]),
                      (0.0, 0.0, 0.0), 0.0, 2)


def test_waypoints_that_overflow_the_arc_are_refused():
    """Finite inputs are not enough: ``cumulative_arc`` squares the components, so
    coordinates past ``sqrt(DBL_MAX)`` yield an infinite arc and a ``(nan, nan)``
    carrot while every input value is finite. The derived arc is what must be
    checked."""
    huge = 1.35e154
    # The overflow is the point of the test, so numpy's warning about it is not
    # a finding; the ValueError is.
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        anchor_plan(np.array([[huge, 0.0], [2 * huge, 0.0]]), (0.0, 0.0, 0.0), 0.0, 1.0)


@pytest.mark.parametrize("bad", [np.array([]), np.zeros((5, 1)), np.array([1.0])])
def test_a_trajectory_without_two_columns_raises_value_error(bad):
    """``atleast_2d`` reshapes ``(0,)`` to ``(1, 0)`` and ``(N,)`` to ``(1, N)``, so a
    row-count test alone lets these reach an ``IndexError`` -- which a caller
    following the documented ``Raises:`` would not catch."""
    with pytest.raises(ValueError):
        anchor_plan(bad, (0.0, 0.0, 0.0), 0.0, 0.5)


# ── the aircraft ─────────────────────────────────────────────────────
@pytest.mark.parametrize("x, y", [(float("nan"), 0.0), (0.0, float("nan")),
                                  (float("inf"), 0.0)])
def test_a_non_finite_aircraft_pose_is_refused(x, y):
    """The plan is validated once per inference; the pose arrives every control
    step and is the likelier NaN. Unchecked it projects to a NaN arc, ``argmin``
    picks index 0, and the carrot is ``(nan, nan)`` with ``replan_reason`` None."""
    engine = flying()
    engine.tick(0.4, 0.0, 0.1)
    with pytest.raises(ValueError):
        engine.tick(x, y, 0.2)


def test_a_non_finite_clock_on_a_tick_is_refused():
    """``now_s`` feeds both the expiry test and the rate floor."""
    engine = flying()
    with pytest.raises(ValueError):
        engine.tick(0.4, 0.0, float("nan"))


# ── arrival is not cheap ─────────────────────────────────────────────
def test_the_shortest_legal_commitment_is_not_arrived_at_quarter_way():
    """The arc allowance scales with the commitment, not just with the radius.

    A flat ``arrive_radius_m`` allowance is most of a short commitment: at the
    shipped defaults the shortest legal one (``min_commit_m`` = 0.40 m) left only
    0.10 m to cover, and was declared FLOWN after 0.105 m.
    """
    engine = PlanCommitExecutor(CommitSpec())
    engine.mark_attempt(0.0)
    plan = engine.commit(straight(count=4), (0.0, 0.0, 0.0), 0.0)
    assert math.isclose(plan.commit_arc_m, 0.40, abs_tol=1e-9)
    flown_at = None
    x = 0.0
    while x < plan.commit_arc_m - 1e-9:
        x += 0.005
        if engine.tick(x, 0.0, 1.0).replan_reason == FLOWN:
            flown_at = x
            break
    assert flown_at is not None, "the commitment must end once it is flown"
    assert flown_at >= 0.7 * plan.commit_arc_m, (
        "declared FLOWN after %.3f m of a %.2f m commitment" % (flown_at, plan.commit_arc_m))


def test_an_arrival_radius_that_swallows_the_shortest_commitment_is_rejected():
    """Tuning the two knobs into overlap restores per-frame inference, which is
    the same class of misconfiguration ``fraction <= 0`` is rejected for."""
    CommitSpec()                       # the shipped defaults stay legal
    with pytest.raises(ValueError):
        CommitSpec(arrive_radius_m=0.5, min_commit_m=0.4)


def out_and_back(reach=1.4, offset=0.03, per_leg=6, tail=12):
    """Out ``reach`` and straight back to the anchor, then a tail past it.

    The turn is early enough that the whole hairpin fits inside the committed
    half, so the commit point lands back under the anchor -- unlike
    ``test_executor.hairpin``, where the turn *is* the commit point. A corridor
    the policy enters and reverses out of within one prediction has this shape.
    """
    out = np.stack([np.linspace(reach / per_leg, reach, per_leg),
                    np.zeros(per_leg)], axis=1)
    back = np.stack([np.linspace(reach * (per_leg - 1) / per_leg, 0.0, per_leg),
                     np.full(per_leg, offset)], axis=1)
    ahead = np.stack([np.linspace(0.2, 2.6, tail), np.full(tail, offset)], axis=1)
    return np.concatenate([out, back, ahead], axis=0)


def test_an_out_and_back_within_the_committed_half_is_not_flown_at_the_anchor():
    """Sitting within ``arrive_radius_m`` of the commit point is not arrival; the
    arc has to have been flown as well.

    Here the commit point is 3 cm from the anchor with 2.8 m of route in
    between, so proximity alone would hand the commitment back on the first
    tick, before the aircraft had moved -- per-frame inference again, at the one
    place a commitment is worth most.
    """
    engine = PlanCommitExecutor(CommitSpec())
    engine.mark_attempt(0.0)
    plan = engine.commit(out_and_back(), (0.0, 0.0, 0.0), 0.0)
    assert plan.commit_index == 12
    assert plan.commit_arc_m > 2.0                    # a real route to fly ...
    assert math.hypot(*plan.commit_point) < 0.05      # ... ending at the anchor

    standing = engine.tick(0.0, 0.0, 1.0)
    assert standing.arc_m < 0.1, "nothing has been flown yet"
    assert standing.replan_reason is None, (
        "declared %r having flown %.3f m of a %.2f m commitment"
        % (standing.replan_reason, standing.arc_m, plan.commit_arc_m))

    # And it still ends once the route really has been flown: refusing the
    # standing start must not turn into a commitment that can never finish.
    ticks = [engine.tick(float(p[0]), float(p[1]), 1.0) for p in plan.committed_xy]
    flown = [tick for tick in ticks if tick.replan_reason == FLOWN]
    assert flown, "a commitment the aircraft has flown must still end"
    assert flown[0].fraction > 0.8


def corner(turn_index, waypoints, leg_m, jog_m):
    """Straight out ``leg_m``, a right-angle jog of ``jog_m``, then a tail.

    The commit point lands on or just past the jog, which is the geometry a
    corner cut happens on: the aircraft rounds the inside of the turn and passes
    within the arrival radius of a point its projected arc never quite reaches.
    """
    step = leg_m / (turn_index - 1)
    points = [(step * (k + 1), 0.0) for k in range(turn_index - 1)]
    points.append((points[-1][0], jog_m))
    while len(points) < waypoints:
        points.append((points[-1][0] + step, jog_m))
    return np.array(points)


@pytest.mark.parametrize("cut", [0.1, 0.2, 0.28])
@pytest.mark.parametrize("turn_index, waypoints, leg_m, jog_m",
                         [(3, 8, 0.5, 0.25), (4, 10, 0.6, 0.3)])
def test_short_commit_corner_cuts_still_arrive(cut, turn_index, waypoints,
                                               leg_m, jog_m):
    """Corner-cutting still arrives where the arc allowance is a quarter of the
    commitment rather than the whole arrival radius.

    Scaling that allowance is what stops a short commitment being handed back
    three quarters unflown, and the risk in scaling it is the opposite failure:
    a genuine corner cut that never arrives, stalls, and ends as EXPIRED
    seconds later with the aircraft holding a stale plan through the turn.
    """
    engine = PlanCommitExecutor(CommitSpec())
    engine.mark_attempt(0.0)
    plan = engine.commit(corner(turn_index, waypoints, leg_m, jog_m),
                         (0.0, 0.0, 0.0), 0.0)
    assert 0.25 * plan.commit_arc_m < engine.spec.arrive_radius_m, (
        "a %.2f m commitment is not short enough for the quarter to bind"
        % plan.commit_arc_m)

    for point in plan.committed_xy[1:-1]:             # fly the route in ...
        engine.tick(float(point[0]), float(point[1]), 1.0)
    commit_x, commit_y = plan.commit_point
    tick = engine.tick(commit_x, commit_y - cut, 2.0)  # ... then cut inside
    assert tick.replan_reason == FLOWN, (
        "a %.2f m cut inside a %.2f m commitment did not arrive (reason %r)"
        % (cut, plan.commit_arc_m, tick.replan_reason))


# ── a refused commitment changes nothing ───────────────────────────
def test_a_failed_recommit_leaves_the_standing_commitment_intact():
    """A rejected ``commit()`` is atomic: the standing plan, the progress already
    flown against it and the commitment count all survive the ``ValueError``.

    Both consumers catch it and go on flying the route they had. A half-replaced
    commitment would either leave them with no target at all in mid-corridor, or
    credit the aircraft with none of the metres it has already flown and send it
    round the same route again.
    """
    engine = flying()
    standing = engine.plan
    for step in range(16):
        engine.tick(0.1 * step, 0.0, 0.1)

    with pytest.raises(ValueError):
        engine.commit(np.array([[float("nan"), 0.0], [0.2, 0.0]]),
                      (0.0, 0.0, 0.0), 0.2)

    assert engine.plan is standing, "the old plan must be untouched"
    assert engine.commitments == 1, "a refused commitment is not a commitment"
    held = engine.tick(1.5, 0.0, 1.0)
    assert held.arc_m == pytest.approx(1.5), "progress must be preserved"
    assert held.replan_reason is None
    assert held.target is not None
