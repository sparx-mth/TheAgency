"""Tests for the object-search loop: select, transit, map the room, repeat.

Every fact here is a fact about a flight, replayed in microseconds because the
machine holds no clock -- a ninety-second room budget is one function call
away rather than ninety seconds away.

Six groups:

* **the loop** -- that a full turn runs select -> transit -> search -> select
  and lands on the next room, which is the whole pipeline in one test;
* **the transit exits** -- arrival, a planner that never answered, the clock,
  and a follower wedged on the way in. Each one was a separate flight failure
  in the stack this replaces;
* **the search exits** -- mapped, stalled and budget-spent, plus the grace
  window that stops the first tick retiring a room it has not entered. All
  three exits are needed: a room whose far corner is permanently occluded
  never reaches zero frontier clusters;
* **the solver seam** -- that an injected order is followed in order, that a
  room which drops out of the ranking is skipped rather than flown to, and
  that the default stub still flies without RPT* existing;
* **the memory** -- cooldown, repeated failure and the deferral that follows,
  and the renumbering that must wipe all of it;
* **the no-ops** -- no pose, not airborne, an empty ranking, and the target
  latch that ends everything. All four happen on a real flight.
"""
from __future__ import annotations

import random

import pytest

from sparx_agency.core.planning.exploration.object_search_supervisor import (
    BLOCKED,
    BUDGET_SPENT,
    FOUND,
    MAPPED,
    SEARCH,
    SELECT,
    STALLED,
    TRANSIT,
    TRANSIT_TIMEOUT,
    UNREACHABLE,
    FlyTo,
    ObjectSearchParams,
    ObjectSearchSupervisor,
    Release,
    RoomFacts,
    SearchRoom,
    StandDown,
)
from sparx_agency.core.planning.exploration.room_search_policy import (
    Hold, RoomOption)

HERE = (0.0, 0.0)


def rooms(*specs):
    """``(room_id, prob, xy)`` triples as :class:`RoomOption` s."""
    return [RoomOption(room_id=r, prob=p, xy=xy, label="R%d" % r)
            for r, p, xy in specs]


def facts(**counts):
    """``{room_id: RoomFacts}`` from ``r1=3`` style frontier counts."""
    return dict((int(k[1:]), RoomFacts(room_id=int(k[1:]),
                                       frontier_clusters=int(v)))
                for k, v in counts.items())


def fixed_solver(*order):
    """A solver that always returns the same order, filtered to what is live."""
    def solve(candidates, instance=None):
        live = {c.room_id for c in candidates}
        return [r for r in order if r in live]
    return solve


def supervisor(params=None, solver=None, seed=7):
    return ObjectSearchSupervisor(
        params or ObjectSearchParams(),
        solver=solver, rng=random.Random(seed))


TWO_ROOMS = rooms((1, 0.7, (10.0, 0.0)), (2, 0.3, (20.0, 0.0)))


# -- the loop -------------------------------------------------------------
def test_a_whole_turn_runs_select_transit_search_and_moves_on():
    sup = supervisor(solver=fixed_solver(1, 2))

    first = sup.update(TWO_ROOMS, facts(r1=2, r2=2), HERE, now=0.0, last_plan_s=0.1)
    assert first.state == TRANSIT
    assert isinstance(first.action, FlyTo)
    assert first.action.room_id == 1
    assert first.changed is True

    holding = sup.update(TWO_ROOMS, facts(r1=2, r2=2), (5.0, 0.0), now=1.0,
                         last_plan_s=0.9)
    assert holding.state == TRANSIT
    assert isinstance(holding.action, Hold)

    arrived = sup.update(TWO_ROOMS, facts(r1=2, r2=2), (10.0, 0.0), now=2.0,
                         last_plan_s=1.9)
    assert arrived.state == SEARCH
    assert isinstance(arrived.action, SearchRoom)
    assert arrived.action.room_id == 1
    assert arrived.action.deadline_s == pytest.approx(2.0 + 90.0)

    # Mapped: zero clusters, held for the full confirmation run, past grace.
    for t in (12.0, 13.0, 14.0):
        done = sup.update(TWO_ROOMS, facts(r1=0, r2=2), (10.0, 0.0), now=t,
                          last_plan_s=t)
    assert isinstance(done.action, Release)
    assert done.action.verdict == MAPPED
    assert done.completed == (1, MAPPED)
    assert done.state == SELECT
    assert done.rooms_done == 1

    nxt = sup.update(TWO_ROOMS, facts(r1=0, r2=2), (10.0, 0.0), now=15.0,
                     last_plan_s=15.0)
    assert nxt.state == TRANSIT
    assert nxt.action.room_id == 2, "the order should advance to the next room"


# -- transit exits --------------------------------------------------------
def test_transit_gives_up_when_no_route_is_ever_produced():
    sup = supervisor(solver=fixed_solver(1))
    sup.update(TWO_ROOMS, facts(r1=1), HERE, now=0.0, last_plan_s=None)
    out = sup.update(TWO_ROOMS, facts(r1=1), HERE, now=6.0, last_plan_s=None)
    assert isinstance(out.action, Release)
    assert out.action.verdict == UNREACHABLE
    assert sup.stats["plan_fails"] == 1


def test_transit_gives_up_on_its_own_clock():
    sup = supervisor(ObjectSearchParams(max_transit_s=20.0),
                     solver=fixed_solver(1))
    sup.update(TWO_ROOMS, facts(r1=1), HERE, now=0.0, last_plan_s=0.0)
    out = sup.update(TWO_ROOMS, facts(r1=1), (1.0, 0.0), now=21.0, last_plan_s=20.0)
    assert out.action.verdict == TRANSIT_TIMEOUT


def test_transit_abandons_a_wedged_follower():
    sup = supervisor(ObjectSearchParams(blocked_abandon_s=5.0),
                     solver=fixed_solver(1))
    sup.update(TWO_ROOMS, facts(r1=1), HERE, now=0.0, last_plan_s=0.0)
    fine = sup.update(TWO_ROOMS, facts(r1=1), (1.0, 0.0), now=2.0,
                      last_plan_s=1.9, blocked_since=1.0)
    assert isinstance(fine.action, Hold), "a brief block is not an abandonment"
    out = sup.update(TWO_ROOMS, facts(r1=1), (1.0, 0.0), now=7.0,
                     last_plan_s=6.9, blocked_since=1.0)
    assert out.action.verdict == BLOCKED


# -- search exits ---------------------------------------------------------
def arrive(sup, room_xy=(10.0, 0.0), t=0.0, f=None):
    """Drive the machine from SELECT to SEARCH and return the arrival tick."""
    f = f if f is not None else facts(r1=3, r2=3)
    sup.update(TWO_ROOMS, f, HERE, now=t, last_plan_s=t)
    return sup.update(TWO_ROOMS, f, room_xy, now=t + 1.0, last_plan_s=t + 0.9)


def finish_room(sup, rooms_in=None, f=None, room_xy=(10.0, 0.0), t=0.0):
    """Select a room, arrive, and map it -- one whole productive turn.

    The grace window is 8 s, so the confirmation run cannot start until then;
    a test that ticks at t+1, t+2, t+3 is still inside the window and retires
    nothing, which is the mistake this helper exists to stop repeating.
    """
    rooms_in = rooms_in if rooms_in is not None else TWO_ROOMS
    f = f if f is not None else facts(r1=0, r2=1)
    sup.update(rooms_in, f, HERE, now=t, last_plan_s=t)
    sup.update(rooms_in, f, room_xy, now=t + 1.0, last_plan_s=t + 0.9)
    for step in (10.0, 11.0, 12.0, 13.0, 14.0):
        out = sup.update(rooms_in, f, room_xy, now=t + step, last_plan_s=t + step)
        if isinstance(out.action, Release):
            return out
    raise AssertionError("the room never finished: %r" % (out,))


def test_the_grace_window_stops_the_arrival_tick_retiring_the_room():
    """On arrival the room's frontier count is still its PRE-arrival value."""
    sup = supervisor(ObjectSearchParams(search_grace_s=8.0),
                     solver=fixed_solver(1))
    arrive(sup, f=facts(r1=0, r2=3))
    for t in (2.0, 3.0, 4.0):
        out = sup.update(TWO_ROOMS, facts(r1=0, r2=3), (10.0, 0.0), now=t,
                         last_plan_s=t)
    assert out.state == SEARCH, "retired inside the grace window"
    assert isinstance(out.action, Hold)


def test_a_single_zero_tick_is_not_enough_to_call_a_room_mapped():
    sup = supervisor(ObjectSearchParams(frontier_clear_ticks=3),
                     solver=fixed_solver(1))
    arrive(sup)
    out = sup.update(TWO_ROOMS, facts(r1=0, r2=3), (10.0, 0.0), now=12.0,
                     last_plan_s=12.0)
    assert out.state == SEARCH
    out = sup.update(TWO_ROOMS, facts(r1=3, r2=3), (10.0, 0.0), now=13.0,
                     last_plan_s=13.0)
    assert out.state == SEARCH, "a dropout must reset the confirmation run"


def test_a_room_that_stops_improving_is_called_stalled():
    """A permanently occluded corner keeps one cluster for ever."""
    sup = supervisor(ObjectSearchParams(frontier_stall_s=20.0),
                     solver=fixed_solver(1))
    arrive(sup)
    for t in (12.0, 20.0, 25.0):
        out = sup.update(TWO_ROOMS, facts(r1=1, r2=3), (10.0, 0.0), now=t,
                         last_plan_s=t)
        assert out.state == SEARCH
    out = sup.update(TWO_ROOMS, facts(r1=1, r2=3), (10.0, 0.0), now=40.0,
                     last_plan_s=40.0)
    assert out.action.verdict == STALLED


def test_progress_resets_the_stall_clock():
    sup = supervisor(ObjectSearchParams(frontier_stall_s=20.0),
                     solver=fixed_solver(1))
    arrive(sup, f=facts(r1=6, r2=3))
    sup.update(TWO_ROOMS, facts(r1=5, r2=3), (10.0, 0.0), now=15.0, last_plan_s=15.0)
    sup.update(TWO_ROOMS, facts(r1=4, r2=3), (10.0, 0.0), now=30.0, last_plan_s=30.0)
    out = sup.update(TWO_ROOMS, facts(r1=3, r2=3), (10.0, 0.0), now=45.0,
                     last_plan_s=45.0)
    assert out.state == SEARCH, "the count is still falling; this is not a stall"


def test_the_budget_is_the_backstop_when_frontier_never_clears():
    sup = supervisor(
        ObjectSearchParams(search_timeout_s=30.0, frontier_stall_s=1e9),
        solver=fixed_solver(1))
    arrive(sup)
    out = sup.update(TWO_ROOMS, facts(r1=4, r2=3), (10.0, 0.0), now=40.0,
                     last_plan_s=40.0)
    assert out.action.verdict == BUDGET_SPENT


def test_search_survives_a_room_that_vanishes_from_the_facts():
    """A renumbered pid stops appearing; the budget must still end the turn."""
    sup = supervisor(ObjectSearchParams(search_timeout_s=30.0),
                     solver=fixed_solver(1))
    arrive(sup)
    out = sup.update(TWO_ROOMS, {}, (10.0, 0.0), now=40.0, last_plan_s=40.0)
    assert out.action.verdict == BUDGET_SPENT
    assert out.state == SELECT


# -- the solver seam ------------------------------------------------------
def test_the_injected_order_is_followed_in_order():
    sup = supervisor(solver=fixed_solver(2, 1))
    first = sup.update(TWO_ROOMS, facts(r1=1, r2=1), HERE, now=0.0, last_plan_s=0.0)
    assert first.action.room_id == 2, "the solver's order beats the probability"
    assert first.order == (2, 1)


def test_a_room_that_leaves_the_ranking_is_skipped_not_flown_to():
    sup = supervisor(solver=fixed_solver(1, 2))
    done = finish_room(sup)
    assert done.action.verdict == MAPPED
    only_two = rooms((2, 1.0, (20.0, 0.0)))
    out = sup.update(only_two, facts(r2=1), (10.0, 0.0), now=15.0, last_plan_s=15.0)
    assert out.action.room_id == 2


def test_the_solver_is_re_asked_when_the_room_set_changes():
    sup = supervisor(solver=fixed_solver(1, 2))
    finish_room(sup)
    calls = sup.stats["solver_calls"]
    three = TWO_ROOMS + rooms((3, 0.5, (30.0, 0.0)))
    sup.update(three, facts(r1=0, r2=1, r3=1), (10.0, 0.0), now=15.0,
               last_plan_s=15.0)
    assert sup.stats["solver_calls"] > calls, (
        "a new room appeared; the committed order is stale")


def test_the_default_stub_flies_without_a_solver():
    """The machine must work before RPT* exists."""
    sup = supervisor()
    out = sup.update(TWO_ROOMS, facts(r1=1, r2=1), HERE, now=0.0, last_plan_s=0.0)
    assert out.state == TRANSIT
    assert out.action.room_id in (1, 2)
    assert len(out.order) == 1, "the stub commits to one room, not a tour"


def test_the_default_stub_is_reproducible_from_a_seed():
    a = supervisor(seed=42).update(TWO_ROOMS, facts(r1=1, r2=1), HERE, now=0.0,
                                   last_plan_s=0.0)
    b = supervisor(seed=42).update(TWO_ROOMS, facts(r1=1, r2=1), HERE, now=0.0,
                                   last_plan_s=0.0)
    assert a.action.room_id == b.action.room_id


def test_the_instance_is_handed_to_the_solver_untouched():
    seen = {}

    def solve(candidates, instance=None):
        seen["instance"] = instance
        return [candidates[0].room_id]

    sentinel = object()
    supervisor(solver=solve).update(TWO_ROOMS, facts(r1=1), HERE, now=0.0,
                                    last_plan_s=0.0, instance=sentinel)
    assert seen["instance"] is sentinel


# -- the memory -----------------------------------------------------------
def test_a_finished_room_cools_before_it_can_be_chosen_again():
    sup = supervisor(ObjectSearchParams(visit_cooldown_s=100.0),
                     solver=fixed_solver(1, 2))
    finish_room(sup)
    out = sup.update(TWO_ROOMS, facts(r1=0, r2=1), (10.0, 0.0), now=15.0,
                     last_plan_s=15.0)
    assert out.action.room_id == 2
    assert 1 not in {c.room_id for c in out.candidates}


def test_every_room_cooling_does_not_stop_the_search():
    sup = supervisor(ObjectSearchParams(visit_cooldown_s=1e9),
                     solver=fixed_solver(1))
    one = rooms((1, 1.0, (10.0, 0.0)))
    finish_room(sup, rooms_in=one, f=facts(r1=0))
    out = sup.update(one, facts(r1=0), (10.0, 0.0), now=15.0, last_plan_s=15.0)
    assert out.state == TRANSIT, "a search that stops is worse than one that repeats"


def test_repeated_failure_defers_a_room():
    """A room the map cannot route to is not worth a fourth attempt."""
    def stubborn(candidates, instance=None):
        """Keeps proposing R1 alone, so each failure is a fresh attempt at it.

        A solver that returned [1, 2] would not retry: the committed order
        simply advances to R2 and R1 never sees a second attempt.
        """
        live = [c.room_id for c in candidates]
        return [1] if 1 in live else live

    sup = supervisor(
        ObjectSearchParams(max_attempts=2, defer_s=500.0, visit_cooldown=False),
        solver=stubborn)
    t = 0.0
    for _ in range(2):
        chosen = sup.update(TWO_ROOMS, facts(r1=1, r2=1), HERE, now=t,
                            last_plan_s=None)
        assert chosen.action.room_id == 1
        sup.update(TWO_ROOMS, facts(r1=1, r2=1), HERE, now=t + 6.0,
                   last_plan_s=None)
        t += 20.0
    out = sup.update(TWO_ROOMS, facts(r1=1, r2=1), HERE, now=t, last_plan_s=t)
    assert out.action.room_id == 2
    assert 1 not in {c.room_id for c in out.candidates}


def test_a_productive_verdict_clears_the_attempt_count():
    sup = supervisor(ObjectSearchParams(max_attempts=2, visit_cooldown=False),
                     solver=fixed_solver(1))
    sup.update(TWO_ROOMS, facts(r1=1), HERE, now=0.0, last_plan_s=None)
    fail = sup.update(TWO_ROOMS, facts(r1=1), HERE, now=6.0, last_plan_s=None)
    assert fail.action.verdict == UNREACHABLE
    done = finish_room(sup, f=facts(r1=0, r2=1), t=10.0)
    assert done.action.verdict == MAPPED
    assert sup._attempts.get(1) is None


def test_forget_rooms_wipes_every_per_room_memory():
    sup = supervisor(solver=fixed_solver(1, 2))
    finish_room(sup)
    sup.forget_rooms()
    out = sup.update(TWO_ROOMS, facts(r1=0, r2=1), (10.0, 0.0), now=15.0,
                     last_plan_s=15.0)
    assert 1 in {c.room_id for c in out.candidates}


def test_the_history_records_every_verdict():
    sup = supervisor(solver=fixed_solver(1))
    sup.update(TWO_ROOMS, facts(r1=1), HERE, now=0.0, last_plan_s=None)
    sup.update(TWO_ROOMS, facts(r1=1), HERE, now=6.0, last_plan_s=None)
    assert sup.history == [(1, UNREACHABLE, 6.0)]


# -- the no-ops -----------------------------------------------------------
def test_no_pose_holds():
    out = supervisor().update(TWO_ROOMS, facts(r1=1), None, now=0.0)
    assert isinstance(out.action, Hold)
    assert out.state == SELECT


def test_not_airborne_holds():
    """Taking the aircraft during a climb strands it on its skids."""
    out = supervisor().update(TWO_ROOMS, facts(r1=1), HERE, now=0.0,
                              airborne=False)
    assert isinstance(out.action, Hold)
    assert "airborne" in out.action.note


def test_an_empty_ranking_holds():
    out = supervisor().update([], {}, HERE, now=0.0)
    assert isinstance(out.action, Hold)


def test_a_ranking_entirely_below_min_prob_holds():
    out = supervisor(ObjectSearchParams(min_prob=0.5)).update(
        rooms((1, 0.1, (1.0, 0.0))), facts(r1=1), HERE, now=0.0)
    assert isinstance(out.action, Hold)


def test_a_room_with_no_centre_is_never_flown_to():
    out = supervisor().update(
        [RoomOption(room_id=1, prob=1.0, xy=None, label="R1")],
        facts(r1=1), HERE, now=0.0)
    assert isinstance(out.action, Hold)


def test_target_seen_stands_down_from_any_state_and_is_terminal():
    sup = supervisor(solver=fixed_solver(1))
    sup.update(TWO_ROOMS, facts(r1=1), HERE, now=0.0, last_plan_s=0.0)
    out = sup.update(TWO_ROOMS, facts(r1=1), (5.0, 0.0), now=1.0,
                     last_plan_s=1.0, target_seen=True)
    assert out.state == FOUND
    assert isinstance(out.action, StandDown)
    assert out.changed is True
    # Terminal: the latch never un-latches, so neither does this.
    again = sup.update(TWO_ROOMS, facts(r1=1), (5.0, 0.0), now=2.0,
                       last_plan_s=2.0, target_seen=False)
    assert again.state == FOUND
