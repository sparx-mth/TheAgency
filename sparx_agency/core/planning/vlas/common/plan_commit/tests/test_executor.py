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
    assert engine.tick(1.9, 0.0, 1.0).replan_reason is None
    assert engine.tick(2.45, 0.0, 1.1).replan_reason == FLOWN


def test_cutting_the_corner_still_counts_as_arriving():
    """The aircraft can pass inside the commit point without its projected arc
    ever reaching it; the arrival radius is what catches that."""
    turning = np.stack([np.linspace(0.2, 3.2, 16),
                        -np.linspace(0.0, 2.0, 16) ** 2 / 4.0], axis=1)
    engine = executor(arrive_radius_m=0.3)
    engine.mark_attempt(0.0)
    plan = engine.commit(turning, (0.0, 0.0, 0.0), 0.0)
    commit_x, commit_y = plan.commit_point
    assert engine.tick(commit_x - 0.2, commit_y + 0.1, 1.0).replan_reason == FLOWN


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
