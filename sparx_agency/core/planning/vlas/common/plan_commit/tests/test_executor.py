"""When a commitment stands, when it is over, and what that buys in the air."""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.vlas.common.plan_commit.progress import project
from sparx_agency.core.planning.vlas.common.plan_commit.executor import (
    EXPIRED,
    FLOWN,
    NO_PLAN,
    OFF_ROUTE,
    TOO_SHORT,
    CommitSpec,
    PlanCommitExecutor,
)


def straight_ahead(count=24, step=0.2):
    """A body-frame prediction running straight forward, NavDP's shape and span."""
    return np.stack([np.arange(1, count + 1) * step, np.zeros(count)], axis=1)


def executor(**overrides):
    spec = CommitSpec(**overrides)
    return PlanCommitExecutor(spec)


def walk(engine, points, now_s):
    """Tick the executor along ``points`` in order, returning the last tick.

    Progress is windowed (:data:`~...progress.CURSOR_WINDOW`), so an aircraft
    cannot appear several metres down a route it has not flown -- which is the
    point, and which means a test that wants the aircraft further along has to
    fly it there rather than teleport it.
    """
    tick = None
    for x, y in points:
        tick = engine.tick(float(x), float(y), now_s)
    return tick


def straight_walk(engine, to_x, now_s, step=0.1):
    """Fly straight along ``y = 0`` from the origin to ``to_x``."""
    count = max(1, int(round(to_x / step)))
    return walk(engine, [(i * step, 0.0) for i in range(count + 1)], now_s)


# ── the commitment ───────────────────────────────────────────────────
def test_with_nothing_committed_there_is_no_target_and_a_plan_is_due():
    tick = executor().tick(0.0, 0.0, 0.0)
    assert tick.target is None
    assert tick.replan_reason == NO_PLAN


def test_a_fresh_commitment_is_flown_not_replaced():
    engine = executor()
    engine.mark_attempt(0.0)
    engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 0.0)
    tick = engine.tick(0.0, 0.0, 1.0)
    assert tick.replan_reason is None
    assert tick.commit_arc_m == pytest.approx(2.4)
    assert tick.target[0] == pytest.approx(1.2, abs=0.05)


def test_halfway_down_the_prediction_ends_the_commitment():
    engine = executor()
    engine.mark_attempt(0.0)
    engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 0.0)
    assert straight_walk(engine, 1.9, 1.0).replan_reason is None
    assert straight_walk(engine, 2.45, 1.1).replan_reason == FLOWN


def test_cutting_the_corner_still_counts_as_arriving():
    """The aircraft can pass inside the commit point without its projected arc
    ever reaching it; the arrival radius is what catches that.

    It has to be flown there rather than teleported. Arrival also requires the
    commitment to be all but complete in arc terms, so that a long route whose
    commit *point* merely happens to lie near the aircraft -- a loop, or a
    corridor entered and reversed out of -- cannot be declared flown from a
    standing start.
    """
    turning = np.stack([np.linspace(0.2, 3.2, 16),
                        -np.linspace(0.0, 2.0, 16) ** 2 / 4.0], axis=1)
    engine = executor(arrive_radius_m=0.3)
    engine.mark_attempt(0.0)
    plan = engine.commit(turning, (0.0, 0.0, 0.0), 0.0)
    commit_x, commit_y = plan.commit_point
    assert walk(engine, plan.world_xy[1:6], 1.0).replan_reason is None
    tick = engine.tick(commit_x - 0.2, commit_y + 0.1, 1.0)
    assert tick.arc_m < plan.commit_arc_m, "the arc test must not be what fires"
    assert tick.replan_reason == FLOWN


def test_progress_is_a_high_water_mark():
    """A commitment whose measured progress can fall can never be finished, so
    an aircraft pushed backwards along the route does not un-fly it."""
    engine = executor()
    engine.mark_attempt(0.0)
    engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 0.0)
    assert straight_walk(engine, 2.0, 1.0).arc_m == pytest.approx(2.0)
    assert engine.tick(0.5, 0.0, 1.5).arc_m == pytest.approx(2.0)


def hairpin(reach=2.4, offset=0.05, per_leg=12):
    """Out and back, the return leg passing ``offset`` from the outbound one.

    A corridor the policy decides against halfway down produces exactly this,
    and it is the shape that breaks a naive nearest-point projection.
    """
    out = np.stack([np.linspace(reach / per_leg, reach, per_leg),
                    np.zeros(per_leg)], axis=1)
    back = np.stack([np.linspace(reach, 0.0, per_leg),
                     np.full(per_leg, offset)], axis=1)
    return np.concatenate([out, back], axis=0)


def test_a_route_that_doubles_back_is_not_finished_before_it_is_flown():
    """The return leg passes 5 cm from the start, so it is the *nearest* segment
    to an aircraft that has drifted 4 cm sideways off the anchor. Unwindowed,
    that reads as 4.8 m of progress against a 2.4 m commitment -- finished
    having flown nothing, with the carrot pointing back the way it came."""
    engine = executor()
    engine.mark_attempt(0.0)
    plan = engine.commit(hairpin(), (0.0, 0.0, 0.0), 0.0)
    assert plan.commit_arc_m == pytest.approx(2.4, abs=0.01)

    tick = engine.tick(0.02, 0.04, 1.0)          # 4 cm off, nothing flown
    assert tick.arc_m < 0.2
    assert tick.replan_reason is None
    assert tick.target[0] > 0.0                  # aimed forward, not back


def test_the_cursor_only_moves_forward_along_a_doubling_back_route():
    """Once the aircraft is on the return leg, drifting near the outbound one
    must not read as having lost two metres of progress."""
    engine = executor(max_deviation_m=5.0)
    engine.mark_attempt(0.0)
    engine.commit(hairpin(), (0.0, 0.0, 0.0), 0.0)
    out = [(x, 0.0) for x in np.arange(0.0, 2.45, 0.1)]
    back = [(x, 0.05) for x in np.arange(2.4, 1.15, -0.1)]
    walk(engine, out + back, 2.0)
    far = engine.tick(1.2, 0.05, 2.0).arc_m      # on the way back
    assert far > 2.4
    assert engine.tick(1.2, 0.01, 3.0).arc_m == pytest.approx(far)


# ── the four guards ──────────────────────────────────────────────────
def test_a_near_stationary_prediction_is_not_flown():
    engine = executor(min_commit_m=0.4)
    engine.mark_attempt(0.0)
    engine.commit(straight_ahead(step=0.01), (0.0, 0.0, 0.0), 0.0)
    assert engine.tick(0.0, 0.0, 1.0).replan_reason == TOO_SHORT


def test_a_commitment_the_aircraft_stopped_tracking_is_abandoned():
    engine = executor(max_deviation_m=2.0)
    engine.mark_attempt(0.0)
    engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 0.0)
    assert engine.tick(1.0, 2.5, 1.0).replan_reason == OFF_ROUTE


def test_a_commitment_that_takes_too_long_is_abandoned():
    engine = executor(max_commit_s=8.0)
    engine.mark_attempt(0.0)
    engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 0.0)
    assert engine.tick(0.1, 0.0, 7.0).replan_reason is None
    assert engine.tick(0.1, 0.0, 9.0).replan_reason == EXPIRED


def test_the_rate_ceiling_outranks_every_reason():
    """A fast server must not be able to reintroduce per-frame inference
    through one of the guards."""
    engine = executor(min_period_s=1.0)
    engine.mark_attempt(10.0)
    engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 10.0)
    assert straight_walk(engine, 3.0, 10.5).replan_reason is None
    assert engine.tick(3.0, 0.0, 11.5).replan_reason == FLOWN


def test_a_dropped_inference_still_costs_a_period():
    """mark_attempt without commit: a server that is down is not hammered."""
    engine = executor(min_period_s=1.0)
    engine.mark_attempt(0.0)
    assert engine.tick(0.0, 0.0, 0.5).replan_reason is None
    assert engine.tick(0.0, 0.0, 1.5).replan_reason == NO_PLAN


def test_reset_forgets_the_commitment():
    engine = executor()
    engine.mark_attempt(0.0)
    engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 0.0)
    engine.reset()
    assert engine.tick(0.0, 0.0, 0.0).replan_reason == NO_PLAN
    assert engine.commitments == 0


# ── what it buys ─────────────────────────────────────────────────────
def fly(engine, metres=20.0, speed=1.0, dt=0.02):
    """Fly a straight corridor, asking a policy that always predicts straight on.

    Returns:
        ``(x, inferences)`` -- how far the aircraft got and how many times the
        policy was asked to get it there.
    """
    x, y, clock, inferences = 0.0, 0.0, 0.0, 0
    while clock < metres / speed * 1.5 and x < metres:
        tick = engine.tick(x, y, clock)
        if tick.replan_reason is not None:
            engine.mark_attempt(clock)
            engine.commit(straight_ahead(), (x, y, 0.0), clock)
            inferences += 1
            tick = engine.tick(x, y, clock)
        if tick.target is not None:
            dx, dy = tick.target[0] - x, tick.target[1] - y
            norm = math.hypot(dx, dy)
            if norm > 1e-9:
                x += speed * dt * dx / norm
                y += speed * dt * dy / norm
        clock += dt
    return x, inferences


def test_committing_flies_the_corridor_on_a_handful_of_inferences():
    """Twenty metres on a 2.4 m commitment is about nine inferences. The same
    twenty metres at a 3 Hz schedule is sixty -- and each of those sixty
    replaces the plan before a third of a metre of it has been flown."""
    x, inferences = fly(executor(min_period_s=0.33))
    assert x >= 20.0
    assert inferences <= 12


def test_the_carrot_follows_the_shape_of_the_plan_not_just_its_end():
    """The whole point of committing, and the one thing a straight corridor
    cannot show: on a plan that turns, the target must trace the turn rather
    than cut across it to the commit point."""
    angles = np.linspace(0.0, math.pi / 2, 24)
    body = np.stack([3.0 * np.sin(angles), 3.0 * (1.0 - np.cos(angles))], axis=1)
    engine = executor(lookahead_m=0.6)
    engine.mark_attempt(0.0)
    plan = engine.commit(body, (0.0, 0.0, 0.0), 0.0)

    # Walk the aircraft along the committed arc and collect where it is aimed.
    targets = [engine.tick(float(p[0]), float(p[1]), 1.0).target
               for p in plan.committed_xy[:-2]]
    # Measured with the window OFF: this is the test's own instrument, and it
    # has to be able to find a carrot anywhere on the route, not only near the
    # executor's cursor.
    reached = [project(plan.world_xy, tx, ty, 0, window=None) for tx, ty in targets]

    # On the route, not across it: a carrot that cut the corner would leave the
    # polyline, and on a 3 m-radius quarter circle it would leave it by tens of
    # centimetres.
    for (_, lateral, _), target in zip(reached, targets):
        assert lateral < 0.01, "carrot %s is off the route by %.3f m" % (target, lateral)
    # And it advances with the aircraft. A fixed aim at the commit point -- the
    # behaviour this package replaced -- would sit at one arc length throughout.
    arcs = [arc for arc, _, _ in reached]
    assert all(later >= earlier - 1e-9 for earlier, later in zip(arcs[:-1], arcs[1:]))
    assert arcs[-1] - arcs[0] > 1.0
    assert arcs[0] < plan.commit_arc_m < arcs[-1]


# ── the legal extremes ───────────────────────────────────────────────
def test_a_single_waypoint_prediction_is_committed_to_and_flown():
    """The shortest legal prediction is a commitment like any other.

    A one-waypoint answer -- a policy that has seen enough to offer one step and
    no more -- must anchor a route the aircraft can be measured against. Clamped
    to nothing, or dismissed as unflyable while it is a metre long, it would be
    re-inferred on the very next tick, which is the per-frame inference this
    package exists to stop.
    """
    engine = executor()
    engine.mark_attempt(0.0)
    plan = engine.commit(np.array([[1.0, 0.0]]), (0.0, 0.0, 0.0), 0.0)
    assert plan.waypoints == 1
    assert plan.commit_index == 1
    assert plan.commit_arc_m == pytest.approx(1.0)          # well over min_commit_m
    assert engine.tick(0.0, 0.0, 1.0).replan_reason is None
    flown = straight_walk(engine, 1.0, 1.0)
    assert flown.replan_reason == FLOWN
    assert flown.fraction == pytest.approx(1.0)


def test_committing_the_whole_prediction_ends_only_once_all_of_it_is_flown():
    """``fraction=1.0`` commits every waypoint, so the commitment ends where the
    prediction does rather than at the half the default flies.

    The far end of a learned trajectory is the part the world has had the most
    time to change under, which is why the default stops at half -- but the
    extreme is legal, and it must lengthen the commitment rather than quietly
    behave like the default.
    """
    engine = executor(fraction=1.0)
    engine.mark_attempt(0.0)
    plan = engine.commit(straight_ahead(), (0.0, 0.0, 0.0), 0.0)
    assert plan.commit_index == 24
    assert plan.commit_arc_m == pytest.approx(4.8)

    # Nor does the whole-prediction case get an exemption from the travel cap:
    # arc credit cannot exceed the ground actually covered, so a first tick two
    # metres down a route it has not flown is credited the arrival slack alone.
    early = engine.tick(2.0, 0.0, 1.0)
    assert early.arc_m == pytest.approx(0.3)                # arrive_radius_m, no more
    assert early.replan_reason is None

    # What a default commitment would have finished at is not half of this one.
    assert straight_walk(engine, 2.4, 1.0).replan_reason is None
    assert straight_walk(engine, 4.8, 1.1).replan_reason == FLOWN


# ── a route that folds back on itself ────────────────────────────────
def fold_back(tail=19, spacing=0.1):
    """Out half a metre and back to 3 cm from the anchor by waypoint 5.

    The fold sits at the *start* of the route rather than at its middle, which
    is what a policy answers with when it looks past an obstacle it has just
    decided to go around. At a small ``fraction`` the commit point lands inside
    the projection window, so the cursor -- which stops a hairpin being finished
    from a standing start only because the cursor must walk every segment of the
    way out first -- is no defence here.
    """
    fold = [(0.25, 0.0), (0.5, 0.0), (0.5, 0.06), (0.25, 0.06), (0.03, 0.06)]
    return np.array(fold + [(0.03, 0.06 + spacing * k)
                            for k in range(1, tail + 1)])


def test_a_fold_is_not_finished_before_the_ground_has_been_covered():
    """A commitment whose commit point falls inside the projection window is not
    declared flown until the aircraft has actually covered the distance.

    Flown by its own carrot from the anchor, this route projects straight onto
    its commit point: on projected arc alone a 1.03 m commitment reads as
    complete after four centimetres, and the aircraft infers every frame for the
    rest of the flight. Arc credit is capped by distance travelled, so the
    earliest FLOWN can arrive is ``commit_arc_m - arrive_radius_m`` of ground.
    """
    engine = executor(fraction=0.2)
    engine.mark_attempt(0.0)
    plan = engine.commit(fold_back(), (0.0, 0.0, 0.0), 0.0)
    assert plan.commit_index == 5                   # inside the projection window
    assert plan.commit_arc_m == pytest.approx(1.03)

    x, y, travelled, flown_after = 0.0, 0.0, 0.0, None
    for step_index in range(60):
        tick = engine.tick(x, y, 1.0 + step_index * 0.05)
        if tick.replan_reason == FLOWN:
            flown_after = travelled
            break
        assert tick.replan_reason is None, tick.replan_reason
        dx, dy = tick.target[0] - x, tick.target[1] - y
        norm = math.hypot(dx, dy)
        if norm > 1e-9:
            step = min(0.02, norm)                  # 0.4 m/s at the 20 Hz tick
            x += step * dx / norm
            y += step * dy / norm
            travelled += step
    assert flown_after is not None, "the commitment was never finished"
    assert flown_after >= plan.commit_arc_m - engine.spec.arrive_radius_m


# ── a commitment's deadline is sized by how far it actually is ───────────
#
# `max_commit_s` has to be long enough for the longest route the policy emits.
# On a policy that mixes a 2.5 m curve with a 0.25 m step rendered from a
# discrete action, that makes it ten times too long for most commitments: a
# short step whose arrival never registers -- blocked, braking, changing
# altitude -- then sits out the whole ceiling doing nothing. Measured in the
# hospital at max_commit_s 12: five twelve-second stalls in ninety seconds.


def _scaled_spec(**kw):
    base = dict(fraction=1.0, lookahead_m=1.0, arrive_radius_m=0.15,
                min_commit_m=0.20, max_commit_s=12.0, max_deviation_m=1.5,
                min_period_s=0.5, expected_speed_mps=0.4, commit_grace_s=2.0)
    base.update(kw)
    return CommitSpec(**base)


def test_a_short_commitment_expires_long_before_a_long_one():
    ex = PlanCommitExecutor(_scaled_spec())
    assert ex._deadline_s(0.25) == pytest.approx(0.25 / 0.4 + 2.0)
    assert ex._deadline_s(2.50) == pytest.approx(2.50 / 0.4 + 2.0)
    assert ex._deadline_s(0.25) < ex._deadline_s(2.50)


def test_the_flat_ceiling_is_still_the_backstop():
    ex = PlanCommitExecutor(_scaled_spec())
    assert ex._deadline_s(100.0) == pytest.approx(12.0)


def test_zero_speed_restores_the_flat_ceiling():
    ex = PlanCommitExecutor(_scaled_spec(expected_speed_mps=0.0))
    for arc in (0.25, 2.5, 100.0):
        assert ex._deadline_s(arc) == pytest.approx(12.0)


def test_a_stalled_short_commitment_is_released_on_its_own_deadline():
    """The aircraft never moves, so only the deadline can free it."""
    ex = PlanCommitExecutor(_scaled_spec())
    ex.commit(np.array([[0.25, 0.0]]), (0.0, 0.0, 0.0), 0.0)
    assert ex.tick(0.0, 0.0, 1.0).replan_reason is None, "freed too early"
    assert ex.tick(0.0, 0.0, 3.0).replan_reason == EXPIRED
