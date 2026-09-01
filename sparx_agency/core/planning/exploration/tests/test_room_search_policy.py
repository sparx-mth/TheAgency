"""Tests for the ranked-room search state machine.

Everything here is a fact about a flight that was expensive to learn, replayed
in microseconds because the policy holds no clock: a monotonic ``now`` is
handed in, so a sixty-second timeout is one function call away rather than one
minute away.

Four groups:

* **the draw** -- that the ``min_prob`` filter and the renormalisation over
  survivors do what the flown sampler did, and that an injected
  :class:`random.Random` makes the whole thing reproducible. A search you
  cannot replay is a search you cannot debug from a recording;
* **the transitions** -- arrival, the planner that never answered, the clock,
  and the dwell that releases the room. Each of the three exits from a pursuit
  was a separate flight failure in the original stack;
* **the cooldown** -- the one deliberate departure from the flown version,
  including the case that must NOT deadlock: every candidate cooling;
* **the no-ops** -- an empty ranking, a ranking that is entirely filtered out,
  and no pose yet. All three happen for the first several seconds of every
  flight, before the oracle and the mapper have said anything.
"""
from __future__ import annotations

import random

import pytest

from sparx_agency.core.planning.exploration.room_search_policy import (
    DWELL,
    IDLE,
    PURSUING,
    Hold,
    PublishGoal,
    ReSample,
    RoomOption,
    RoomSearchParams,
    RoomSearchPolicy,
)

HERE = (0.0, 0.0)


def rooms(*specs):
    """``(room_id, prob, xy)`` triples as :class:`RoomOption` s."""
    return [RoomOption(room_id=rid, prob=prob, xy=xy, label="R%d" % rid)
            for rid, prob, xy in specs]


def policy(seed=7, **overrides):
    """A policy with a seeded generator, so every draw here is reproducible."""
    return RoomSearchPolicy(RoomSearchParams(**overrides),
                            rng=random.Random(seed))


# -- the draw --------------------------------------------------------------

def test_first_tick_draws_a_room_and_asks_for_a_goal():
    state = policy().update(rooms((1, 0.7, (5.0, 0.0)), (2, 0.3, (0.0, 5.0))),
                            HERE, now=0.0)
    assert state.state == PURSUING
    assert state.changed is True
    assert isinstance(state.action, PublishGoal)
    assert state.action.room_id in (1, 2)
    assert state.action.xy == state.goal_xy


def test_the_same_seed_draws_the_same_rooms():
    ranking = rooms((1, 0.5, (5.0, 0.0)), (2, 0.3, (0.0, 5.0)),
                    (3, 0.2, (-5.0, 0.0)))
    picked = []
    for _ in range(2):
        # Cooldown off: with it on, a replay would be shaped by visit history
        # as well as by the generator, which is not what determinism means here.
        # No ``last_plan_s`` either, so every pursuit is abandoned after the
        # grace and the run is a sequence of draws rather than a single one.
        search = policy(seed=11, visit_cooldown=False, plan_grace_s=0.5)
        run = []
        now = 0.0
        for _ in range(12):
            state = search.update(ranking, HERE, now=now)
            if isinstance(state.action, PublishGoal):
                run.append(state.action.room_id)
            now += 1.0
        picked.append(run)
    assert picked[0] == picked[1]
    assert len(picked[0]) >= 4
    assert len(set(picked[0])) > 1


def test_rooms_below_min_prob_are_never_drawn():
    search = policy(min_prob=0.2)
    state = search.update(rooms((1, 0.99, (5.0, 0.0)), (2, 0.01, (0.0, 5.0))),
                          HERE, now=0.0)
    assert [c.room_id for c in state.candidates] == [1]
    assert state.action.room_id == 1


def test_a_room_without_a_centroid_is_not_a_candidate():
    search = policy()
    state = search.update([RoomOption(room_id=1, prob=0.9, xy=None),
                           RoomOption(room_id=2, prob=0.1, xy=(3.0, 0.0))],
                          HERE, now=0.0)
    assert [c.room_id for c in state.candidates] == [2]
    assert state.action.room_id == 2


def test_survivors_are_renormalised_among_themselves():
    # Half the published mass is on rooms that cannot be flown to; the two that
    # can must still share 1.0 between them, in their original ratio.
    search = policy(min_prob=0.05)
    state = search.update(rooms((1, 0.3, (5.0, 0.0)), (2, 0.1, (0.0, 5.0)),
                                (3, 0.01, (1.0, 1.0)), (4, 0.59, None)),
                          HERE, now=0.0)
    renorm = {c.room_id: c.prob_renorm for c in state.candidates}
    assert set(renorm) == {1, 2}
    assert renorm[1] == pytest.approx(0.75)
    assert renorm[2] == pytest.approx(0.25)
    assert sum(renorm.values()) == pytest.approx(1.0)


def test_the_draw_follows_the_renormalised_weights():
    # An overwhelming favourite is drawn overwhelmingly often; a 5% room is
    # drawn sometimes. Both halves matter -- an argmax would never pick R2.
    ranking = rooms((1, 0.95, (5.0, 0.0)), (2, 0.05, (0.0, 5.0)))
    counts = {1: 0, 2: 0}
    for seed in range(300):
        state = policy(seed=seed).update(ranking, HERE, now=0.0)
        counts[state.action.room_id] += 1
    assert counts[1] > counts[2] * 4
    assert counts[2] > 0


# -- the transitions -------------------------------------------------------

def test_arriving_inside_the_tolerance_starts_the_dwell():
    search = policy(arrival_tol_m=0.6, dwell_after_arrival_s=15.0)
    goal = search.update(rooms((1, 1.0, (5.0, 0.0))), HERE, now=0.0)
    near = (goal.goal_xy[0] - 0.5, goal.goal_xy[1])
    state = search.update(rooms((1, 1.0, (5.0, 0.0))), near, now=1.0,
                          last_plan_s=0.0)
    assert state.state == DWELL
    assert isinstance(state.action, Hold)
    assert state.dwell_left_s == pytest.approx(15.0)


def test_the_dwell_expires_into_a_resample():
    search = policy(arrival_tol_m=0.6, dwell_after_arrival_s=15.0)
    ranking = rooms((1, 1.0, (5.0, 0.0)))
    search.update(ranking, HERE, now=0.0)
    search.update(ranking, (4.7, 0.0), now=1.0, last_plan_s=0.0)
    assert isinstance(search.update(ranking, (4.7, 0.0), now=10.0).action, Hold)
    released = search.update(ranking, (4.7, 0.0), now=16.1)
    assert isinstance(released.action, ReSample)
    assert released.state == IDLE
    assert search.stats["dwell_completes"] == 1


def test_no_plan_within_the_grace_gives_the_room_up():
    # The caller never produced a route: the centroid is unreachable, and the
    # flown sampler's answer is to draw a different room rather than wait out
    # the sixty-second clock.
    search = policy(plan_grace_s=5.0, max_pursue_s=60.0)
    ranking = rooms((1, 1.0, (50.0, 0.0)))
    search.update(ranking, HERE, now=0.0)
    assert isinstance(search.update(ranking, HERE, now=4.0).action, Hold)
    state = search.update(ranking, HERE, now=5.1)
    assert isinstance(state.action, ReSample)
    assert state.state == IDLE
    assert search.stats["plan_fails"] == 1


def test_one_plan_after_the_goal_survives_the_grace():
    search = policy(plan_grace_s=5.0, max_pursue_s=60.0)
    ranking = rooms((1, 1.0, (50.0, 0.0)))
    search.update(ranking, HERE, now=0.0)
    state = search.update(ranking, HERE, now=5.1, last_plan_s=1.0)
    assert isinstance(state.action, Hold)
    assert state.state == PURSUING


def test_a_plan_made_before_the_goal_does_not_count():
    # The snapshot at goal time is the whole point: a route planned for the
    # PREVIOUS room must not vouch for this one.
    search = policy(plan_grace_s=5.0)
    ranking = rooms((1, 1.0, (50.0, 0.0)))
    search.update(ranking, HERE, now=10.0, last_plan_s=9.0)
    state = search.update(ranking, HERE, now=15.2, last_plan_s=9.0)
    assert isinstance(state.action, ReSample)


def test_the_pursue_clock_gives_up_on_a_route_that_never_arrives():
    search = policy(plan_grace_s=5.0, max_pursue_s=60.0)
    ranking = rooms((1, 1.0, (50.0, 0.0)))
    search.update(ranking, HERE, now=0.0)
    assert isinstance(
        search.update(ranking, HERE, now=59.0, last_plan_s=58.0).action, Hold)
    state = search.update(ranking, HERE, now=60.5, last_plan_s=60.0)
    assert isinstance(state.action, ReSample)
    assert search.stats["timeouts"] == 1
    assert search.stats["plan_fails"] == 0


def test_a_resample_is_followed_by_a_fresh_draw_on_the_next_tick():
    search = policy(plan_grace_s=5.0, visit_cooldown=False)
    ranking = rooms((1, 1.0, (50.0, 0.0)))
    search.update(ranking, HERE, now=0.0)
    search.update(ranking, HERE, now=5.1)
    state = search.update(ranking, HERE, now=6.1)
    assert isinstance(state.action, PublishGoal)
    assert state.changed is True


# -- the cooldown ----------------------------------------------------------

def test_a_searched_room_is_skipped_while_it_is_cooling():
    search = policy(arrival_tol_m=0.6, dwell_after_arrival_s=1.0,
                    visit_cooldown=True, visit_cooldown_s=120.0)
    ranking = rooms((1, 0.9, (5.0, 0.0)), (2, 0.1, (0.0, 5.0)))
    first = search.update(ranking, HERE, now=0.0)
    reached = first.goal_xy
    search.update(ranking, reached, now=1.0, last_plan_s=0.0)   # arrive
    search.update(ranking, reached, now=2.5)                    # dwell expires
    second = search.update(ranking, reached, now=3.5)
    assert isinstance(second.action, PublishGoal)
    assert second.action.room_id != first.action.room_id


def test_the_cooldown_expires():
    search = policy(arrival_tol_m=0.6, dwell_after_arrival_s=1.0,
                    visit_cooldown_s=10.0)
    ranking = rooms((1, 1.0, (5.0, 0.0)))
    search.update(ranking, HERE, now=0.0)
    search.update(ranking, (5.0, 0.0), now=1.0, last_plan_s=0.0)
    search.update(ranking, (5.0, 0.0), now=2.5)
    # Still cooling: the only room is drawn anyway (see the deadlock test), so
    # assert on the candidate set once the cooldown has genuinely lapsed.
    late = search.update(rooms((1, 1.0, (5.0, 0.0)), (2, 0.5, (0.0, 5.0))),
                         (5.0, 0.0), now=30.0)
    assert {c.room_id for c in late.candidates} == {1, 2}


def test_every_candidate_cooling_does_not_deadlock_the_search():
    search = policy(arrival_tol_m=0.6, dwell_after_arrival_s=1.0,
                    visit_cooldown_s=1e6)
    ranking = rooms((1, 1.0, (5.0, 0.0)))
    search.update(ranking, HERE, now=0.0)
    search.update(ranking, (5.0, 0.0), now=1.0, last_plan_s=0.0)
    search.update(ranking, (5.0, 0.0), now=2.5)
    state = search.update(ranking, (5.0, 0.0), now=3.5)
    assert isinstance(state.action, PublishGoal)
    assert state.action.room_id == 1


def test_forget_visits_clears_the_cooldown_when_the_ids_restart():
    # The segmentation renumbers every room whenever the map is reshaped, so a
    # cooldown carried across the renumbering would skip whichever room
    # INHERITED the id rather than the room that was actually searched.
    search = policy(arrival_tol_m=0.6, dwell_after_arrival_s=1.0,
                    plan_grace_s=5.0, visit_cooldown_s=1e6)
    ranking = rooms((1, 0.9, (5.0, 0.0)), (2, 0.1, (0.0, 5.0)))
    first = search.update(ranking, HERE, now=0.0)
    goal = first.goal_xy
    search.update(ranking, goal, now=1.0, last_plan_s=0.0)   # arrive -> dwell
    search.update(ranking, goal, now=2.5)                    # dwell out -> idle
    cooling = search.update(ranking, goal, now=3.5)          # the next draw
    assert first.room_id not in {c.room_id for c in cooling.candidates}

    search.update(ranking, goal, now=9.6)                    # no plan -> idle
    search.forget_visits()
    after = search.update(ranking, goal, now=10.0)
    assert {c.room_id for c in after.candidates} == {1, 2}


def test_cooldown_off_reproduces_the_flown_sampler():
    # The flown version had no memory: the room just searched is immediately
    # eligible again, and with one candidate it is drawn again.
    search = policy(arrival_tol_m=0.6, dwell_after_arrival_s=1.0,
                    visit_cooldown=False)
    ranking = rooms((1, 0.9, (5.0, 0.0)), (2, 0.1, (0.0, 5.0)))
    search.update(ranking, HERE, now=0.0)
    search.update(ranking, (5.0, 0.0), now=1.0, last_plan_s=0.0)
    search.update(ranking, (5.0, 0.0), now=2.5)
    state = search.update(ranking, (5.0, 0.0), now=3.5)
    assert {c.room_id for c in state.candidates} == {1, 2}


def test_an_abandoned_room_cools_as_well_as_a_searched_one():
    search = policy(plan_grace_s=5.0, visit_cooldown_s=120.0)
    ranking = rooms((1, 0.9, (50.0, 0.0)), (2, 0.1, (0.0, 5.0)))
    first = search.update(ranking, HERE, now=0.0)
    search.update(ranking, HERE, now=5.1)          # no route -> abandoned
    second = search.update(ranking, HERE, now=6.1)
    assert second.action.room_id != first.action.room_id


# -- the no-ops ------------------------------------------------------------

def test_an_empty_ranking_is_a_safe_no_op():
    state = policy().update([], HERE, now=0.0)
    assert state.state == IDLE
    assert isinstance(state.action, Hold)
    assert state.candidates == ()
    assert state.goal_xy is None


def test_a_ranking_filtered_to_nothing_is_a_safe_no_op():
    state = policy(min_prob=0.5).update(
        rooms((1, 0.2, (5.0, 0.0)), (2, 0.1, (0.0, 5.0))), HERE, now=0.0)
    assert state.state == IDLE
    assert isinstance(state.action, Hold)


def test_no_pose_holds_without_touching_the_pursuit():
    search = policy()
    ranking = rooms((1, 1.0, (5.0, 0.0)))
    search.update(ranking, HERE, now=0.0)
    state = search.update(ranking, None, now=1000.0)
    assert state.state == PURSUING
    assert isinstance(state.action, Hold)
    assert search.stats["timeouts"] == 0
