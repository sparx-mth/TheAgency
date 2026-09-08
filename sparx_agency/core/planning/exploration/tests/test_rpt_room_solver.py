"""Tests for the RPT* adapter that orders the object search's rooms.

The adapter is a translation, and every way a translation fails is silent: a
tour planned from the wrong vertex still flies, a room quietly missing from
the problem is simply never visited, and a solver that raises inside a ROS
timer stops the mission with a stack trace nobody reads until afterwards. So
the tests are grouped by what would go unnoticed:

* **the answer** -- that the order really is RPT*'s and not one of the two
  obvious ones, pinned against the package's own nearest-first and
  likeliest-first baselines. If this passes and nothing else does, the
  contribution is present;
* **the depot** -- that the aircraft's own vertex, which the instance appends
  LAST and the routing problem expects FIRST, is moved. Pinned twice: once on
  numbers small enough to check by hand, and once on a real numpy instance
  built by ``room_costs``;
* **the intersection** -- that the rooms planned over are exactly those the
  supervisor still considers eligible AND the map can still reach;
* **the degenerate shapes** -- one room, no rooms, no instance;
* **the failures** -- that every one of them lands on the weighted draw and
  none of them escapes, because the supervisor marks a room set as solved-for
  before it looks at what came back, so an empty order parks the aircraft
  until the ranking itself changes;
* **the cap** -- that the vertex set is bounded, because past fifteen rooms
  the exact search times out and RPT* hands back a nearest-first tour that
  never reads the probabilities. Uncapped, this whole module would deliver the
  baseline it exists to beat;
* **the loop** -- that the supervisor really does walk the whole tour.

The hand-built instances are plain Python lists rather than numpy arrays, on
purpose: the adapter reads the instance through ``tolist()`` and never with a
numpy call, so a list of lists has to work as well as the real thing. Only the
two tests that deliberately drive ``room_costs.build_instance`` -- the ones
pinning the depot placement and the unreachable room against real geometry --
touch numpy at all.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from sparx_agency.core.planning.exploration.object_search_supervisor import (
    ObjectSearchParams, ObjectSearchSupervisor, RoomFacts)
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.exploration.room_search_policy import (
    RoomCandidate, RoomOption)
from sparx_agency.core.planning.exploration.rpt_room_solver import (
    DEFAULT_MAX_ROOMS, FALLBACK, NEAREST_RESERVED, NOT_ATTEMPTED, RPT_STAR,
    RptStarRoomSolver)
from sparx_agency.core.planning.exploration.tests.test_room_costs import (
    CORRIDOR, ROOM_A, ROOM_B, ROOM_C, corridor_world, flat_cost)
from sparx_agency.core.planning.routing.rpt_star import (
    RouteProblem, RouteVertex, RptStarParams, costs_from_points,
    expected_cost, greedy_probability_order, nearest_neighbour_order)

EXACT = RptStarParams(epsilon=None, time_budget_s=None)
"""No clock, so every assertion about the ORDER is about the algorithm."""

#: A hand-checkable instance in which the two obvious answers are both wrong.
#: Room 1 is far but very likely, room 2 is far and unlikely, room 3 is near
#: and fairly likely. Nearest-first opens 3, 2, 1; likeliest-first opens
#: 1, 3, 2; the cheapest expected order is 3, 1, 2 and is neither.
DEPOT_XY = (0.0, 0.0)
ROOM_XY = {1: (10.0, 0.0), 2: (2.0, 10.0), 3: (2.0, 2.0)}
ROOM_P = {1: 0.8, 2: 0.1, 3: 0.6}
BEST_ORDER = [3, 1, 2]


def instance(rooms=(1, 2, 3), depot_xy=DEPOT_XY, probs=None, units="metres"):
    """An HppPtInstance stand-in: rooms in pid order, the aircraft LAST.

    Plain lists, not numpy, and no ``tolist`` -- which is exactly the shape
    the adapter has to cope with as well as the real thing.
    """
    probs = ROOM_P if probs is None else probs
    points = [ROOM_XY[pid] for pid in rooms] + [depot_xy]
    cost = [[math.hypot(a[0] - b[0], a[1] - b[1]) for b in points]
            for a in points]
    return SimpleNamespace(
        C=cost,
        p=[float(probs.get(pid, 0.0)) for pid in rooms] + [0.0],
        depot=len(rooms),
        index_to_pid=tuple(rooms) + (-1,),
        units=units)


def candidates(*room_ids):
    """Eligible rooms, renormalised over themselves as the supervisor does."""
    total = sum(ROOM_P.get(pid, 0.0) for pid in room_ids) or 1.0
    return tuple(
        RoomCandidate(room_id=pid, label="R%d" % pid,
                      prob=ROOM_P.get(pid, 0.0),
                      prob_renorm=ROOM_P.get(pid, 0.0) / total,
                      xy=ROOM_XY.get(pid, (0.0, 0.0)))
        for pid in room_ids)


def reference(depot_xy=DEPOT_XY, rooms=(1, 2, 3)):
    """The same problem posed straight to RPT*, for the baselines to score."""
    vertices = [RouteVertex(id=pid, prob=ROOM_P[pid], label="R%d" % pid)
                for pid in rooms]
    problem = RouteProblem.with_external_start(vertices)
    matrix = costs_from_points([depot_xy] + [ROOM_XY[pid] for pid in rooms])
    return problem, matrix


# -- the answer -----------------------------------------------------------
def test_the_order_is_neither_the_nearest_room_nor_the_likeliest():
    """The whole contribution, in one assertion.

    Both obvious policies are available from the package's own baselines, and
    the point of RPT* is that it agrees with neither. An adapter that quietly
    lost the probabilities would return the nearest-first tour; one that lost
    the costs would return the likeliest-first one.
    """
    order = RptStarRoomSolver(params=EXACT)(candidates(1, 2, 3), instance())
    problem, matrix = reference()
    nearest = nearest_neighbour_order(problem, matrix)
    likeliest = greedy_probability_order(problem, matrix)

    assert order == BEST_ORDER
    assert list(problem.ids_of(nearest))[1:] != order, "matched nearest-first"
    assert list(problem.ids_of(likeliest))[1:] != order, "matched likeliest"

    ours = expected_cost((0,) + tuple(problem.index(r) for r in order),
                         problem.probs, matrix)
    assert ours < expected_cost(nearest, problem.probs, matrix)
    assert ours < expected_cost(likeliest, problem.probs, matrix)


def test_the_whole_remaining_order_is_returned_not_just_the_head():
    """One room per solve would re-solve on a noisy ranking and oscillate."""
    order = RptStarRoomSolver(params=EXACT)(candidates(1, 2, 3), instance())
    assert len(order) == 3, "the tour, not the head of it"
    assert sorted(order) == [1, 2, 3]


def test_the_record_carries_the_optimality_claim():
    solver = RptStarRoomSolver(params=EXACT)
    solver(candidates(1, 2, 3), instance())
    last = solver.last
    assert last.source == RPT_STAR
    assert last.status == "solved"
    assert last.guarantee == "optimal"
    assert last.route_source == "search"
    assert last.rooms == 3
    assert last.expected_cost == pytest.approx(7.1514, abs=1e-3)
    assert last.bound_ratio == pytest.approx(1.0)
    assert last.expansions > 0
    assert last.units == "metres"
    assert "rpt*" in last.summary()


def test_the_probabilities_come_from_the_instance_not_from_prob_renorm():
    """``prob_renorm`` is exactly 1.0 for a lone candidate, which RPT* refuses.

    It also asserts the target is certainly in a room already mapped, which
    is the thing both halves deliberately do not do.
    """
    solver = RptStarRoomSolver(params=EXACT)
    order = solver(candidates(2), instance())
    assert order == [2], "a lone candidate must not trip the p < 1 contract"
    assert solver.last.source == RPT_STAR


# -- the depot ------------------------------------------------------------
def test_where_the_aircraft_is_changes_the_order():
    """The instance appends the depot LAST; the problem wants it FIRST.

    An adapter that assumed the two index spaces line up would read some
    room's row as the aircraft's and return the same tour wherever the
    aircraft actually was.
    """
    solver = RptStarRoomSolver(params=EXACT)
    from_origin = solver(candidates(1, 2, 3), instance())
    at_room_2 = solver(candidates(1, 2, 3),
                       instance(depot_xy=ROOM_XY[2]))
    at_room_1 = solver(candidates(1, 2, 3),
                       instance(depot_xy=ROOM_XY[1]))

    assert from_origin == [3, 1, 2]
    assert at_room_2 == [2, 3, 1], "parked in a room, search it first"
    assert at_room_1 == [1, 3, 2]


def test_a_real_numpy_instance_is_read_with_the_aircraft_in_the_right_place():
    """The same fact again, through ``build_instance`` and its numpy arrays.

    Equal probabilities everywhere, so the ONLY thing that can order the
    three rooms is where the aircraft is standing.
    """
    world = corridor_world()
    cost = flat_cost(world)
    centres = {1: ROOM_A, 2: ROOM_B, 3: ROOM_C}
    flat = {1: 0.3, 2: 0.3, 3: 0.3}
    solver = RptStarRoomSolver(params=EXACT)

    near_a, _ = build_instance(world, cost, centres, flat,
                               depot_xy=(1.0, 1.25), cruise_speed_mps=0.0)
    near_c, _ = build_instance(world, cost, centres, flat,
                               depot_xy=(4.0, 1.25), cruise_speed_mps=0.0)
    assert near_a.index_to_pid == (1, 2, 3, -1), "the depot is appended last"
    assert near_a.depot == 3

    assert solver(candidates(1, 2, 3), near_a)[0] == 1
    assert solver(candidates(1, 2, 3), near_c)[0] == 3


# -- the intersection -----------------------------------------------------
def test_only_rooms_in_both_the_candidates_and_the_instance_are_planned_over():
    """The instance maps every room; the policy has already refused some."""
    solver = RptStarRoomSolver(params=EXACT)
    order = solver(candidates(1, 3), instance(rooms=(1, 2, 3)))
    assert sorted(order) == [1, 3], "room 2 was not eligible this tick"
    assert solver.last.rooms == 2
    assert solver.last.skipped == ()


def test_a_candidate_the_instance_never_mapped_is_reported_not_priced():
    """A room withheld as unreachable has no arc weight, so it has no vertex.

    Inventing one would plan a tour through a room the map says it cannot
    reach; dropping it silently would hide that from the operator.
    """
    solver = RptStarRoomSolver(params=EXACT)
    order = solver(candidates(1, 2, 3), instance(rooms=(1, 3)))
    assert sorted(order) == [1, 3]
    assert solver.last.skipped == (2,)
    assert solver.last.source == RPT_STAR, "one missing room is not a failure"


def test_an_unreachable_room_is_withheld_by_the_instance_and_skipped_here():
    """End to end: a room the map cannot reach, through the real builder."""
    world = corridor_world()
    grid = world.grid.copy()
    grid[1:3, 6] = 100                  # seal room C off from the corridor
    world = type(world)(grid, world.params, values=world.values)
    cost = flat_cost(world)

    inst, dropped = build_instance(
        world, cost, {1: ROOM_A, 2: ROOM_B, 3: ROOM_C},
        {1: 0.3, 2: 0.3, 3: 0.3}, depot_xy=CORRIDOR, cruise_speed_mps=0.0)
    assert 3 in dropped, "room C should be unreachable once sealed"

    solver = RptStarRoomSolver(params=EXACT)
    order = solver(candidates(1, 2, 3), inst)
    assert 3 not in order
    assert sorted(order) == [1, 2]
    assert solver.last.skipped == (3,)


# -- the degenerate shapes ------------------------------------------------
def test_a_single_room_instance_still_answers():
    solver = RptStarRoomSolver(params=EXACT)
    assert solver(candidates(3), instance(rooms=(3,))) == [3]
    assert solver.last.source == RPT_STAR
    assert solver.last.rooms == 1


def test_an_instance_holding_no_room_at_all_falls_back():
    """The aircraft's own vertex is not somewhere to fly to."""
    empty = SimpleNamespace(C=[[0.0]], p=[0.0], depot=0,
                            index_to_pid=(-1,), units="metres")
    solver = RptStarRoomSolver(params=EXACT, rng=random.Random(1))
    order = solver(candidates(1, 2, 3), empty)
    assert order, "an empty order parks the aircraft"
    assert solver.last.source == FALLBACK


def test_no_candidates_is_an_empty_order():
    solver = RptStarRoomSolver(params=EXACT)
    assert solver((), instance()) == []
    assert solver.last.order == ()


# -- the failures ---------------------------------------------------------
def test_no_instance_yet_falls_back_to_the_draw():
    """The first minute of every flight, before the mapper has a room."""
    solver = RptStarRoomSolver(params=EXACT, rng=random.Random(4))
    order = solver(candidates(1, 2, 3), None)
    assert len(order) == 1, "the stub commits to one room, not a tour"
    assert order[0] in (1, 2, 3)
    assert solver.last.source == FALLBACK
    assert solver.last.reason == "no usable instance"


def test_a_solver_failure_falls_back_rather_than_grounding_the_aircraft():
    """A disconnected matrix is fatal to RPT* and must not be fatal here."""
    broken = instance()
    broken.C[0][1] = float("inf")
    broken.C[1][0] = float("inf")
    solver = RptStarRoomSolver(params=EXACT, rng=random.Random(5))
    order = solver(candidates(1, 2, 3), broken)
    assert len(order) == 1
    assert solver.last.source == FALLBACK
    assert "DisconnectedGraphError" in solver.last.reason


def test_a_nan_probability_falls_back_instead_of_raising():
    """A NaN is a bug upstream; clamping it into range would hide it."""
    bad = instance()
    bad.p[0] = float("nan")
    solver = RptStarRoomSolver(params=EXACT, rng=random.Random(6))
    order = solver(candidates(1, 2, 3), bad)
    assert len(order) == 1
    assert solver.last.source == FALLBACK
    assert "InvalidProblemError" in solver.last.reason


def test_an_instance_that_raises_on_being_read_does_not_escape():
    """Nothing may propagate: the caller is a ROS timer with no guard."""

    class Hostile(object):
        @property
        def C(self):
            raise RuntimeError("the arrays went away")

    solver = RptStarRoomSolver(params=EXACT, rng=random.Random(7))
    order = solver(candidates(1, 2, 3), Hostile())
    assert order, "a failure must still leave somewhere to fly"
    assert solver.last.source == FALLBACK


def test_the_fallback_is_reproducible_from_a_seed():
    seeded = [RptStarRoomSolver(params=EXACT, rng=random.Random(9))(
        candidates(1, 2, 3), None) for _ in range(4)]
    assert len(set(tuple(o) for o in seeded)) == 1


def test_a_budget_too_small_still_returns_a_flyable_order():
    """RPT* answers with a constructive tour rather than with nothing."""
    solver = RptStarRoomSolver(params=RptStarParams(max_expansions=1))
    order = solver(candidates(1, 2, 3), instance())
    assert sorted(order) == [1, 2, 3]
    assert solver.last.source == RPT_STAR
    assert solver.last.status == "budget_exceeded"
    assert solver.last.guarantee == "none"
    assert solver.last.route_source == "fallback"


def test_the_probabilities_are_not_renormalised_on_the_way_in():
    """Normalising over mapped rooms alone suppresses unmapped space.

    RPT* remarks on a belief that does not sum to one, so its own warning is
    the evidence that the adapter handed the numbers over untouched -- these
    sum to 1.5, and an adapter that "fixed" that would silence the warning.
    """
    assert sum(ROOM_P.values()) == pytest.approx(1.5)
    solver = RptStarRoomSolver(params=EXACT)
    solver(candidates(1, 2, 3), instance())
    assert any("1.5000" in warning for warning in solver.last.warnings), (
        "RPT* was handed a renormalised belief: %r" % (solver.last.warnings,))


# -- stability ------------------------------------------------------------
def test_the_order_does_not_depend_on_the_order_the_ranking_arrived_in():
    """The ranking churns; a tour that churns with it re-flies the aircraft."""
    solver = RptStarRoomSolver(params=EXACT)
    first = solver(candidates(1, 2, 3), instance())
    for shuffled in ((3, 1, 2), (2, 3, 1), (3, 2, 1)):
        assert solver(candidates(*shuffled), instance()) == first


def test_a_tie_is_broken_the_same_way_however_the_ranking_arrived():
    """Where the optimum is NOT unique, only the canonical order saves us.

    Four rooms on a square with the aircraft in the middle and a flat belief:
    every tour costs the same, so which one comes back is decided purely by
    the order the search happened to expand in. That is the shape the LLM
    produces when it has nothing to say, and the supervisor commits to the
    head of whatever comes back -- so an order that depends on the ranking's
    arrival order sends the aircraft somewhere different every time an
    unrelated room appears.
    """
    square = {1: (0.0, 4.0), 2: (4.0, 0.0), 3: (0.0, -4.0), 4: (-4.0, 0.0)}
    flat = dict((pid, 0.25) for pid in square)
    solver = RptStarRoomSolver(params=EXACT)

    # The instance is built in ascending pid order and does NOT churn --
    # room_costs builds it that way every tick. Only the ranking churns.
    order_pids = sorted(square)
    points = [square[pid] for pid in order_pids] + [(0.0, 0.0)]
    tied = SimpleNamespace(
        C=[[math.hypot(a[0] - b[0], a[1] - b[1]) for b in points]
           for a in points],
        p=[flat[pid] for pid in order_pids] + [0.0],
        depot=len(order_pids), index_to_pid=tuple(order_pids) + (-1,),
        units="metres")

    def tied_candidates(rooms):
        return tuple(RoomCandidate(room_id=pid, label="R%d" % pid, prob=0.25,
                                   prob_renorm=0.25, xy=square[pid])
                     for pid in rooms)

    arrivals = ((1, 2, 3, 4), (4, 3, 2, 1), (2, 4, 1, 3), (3, 1, 4, 2))
    for capped in (None, 3):
        solver = RptStarRoomSolver(params=EXACT, max_rooms=capped)
        orders = set(tuple(solver(tied_candidates(a), tied)) for a in arrivals)
        assert len(orders) == 1, (
            "max_rooms=%r: a tied instance returned %d different tours: %r"
            % (capped, len(orders), orders))


# -- the cap --------------------------------------------------------------
def wide(count, probs=None):
    """``count`` rooms on a circle, the aircraft at the centre."""
    xy = dict((pid, (math.cos(pid) * (5.0 + pid % 3),
                     math.sin(pid) * (5.0 + pid % 3)))
              for pid in range(1, count + 1))
    prob = probs or dict((pid, 0.9 / pid) for pid in xy)
    points = [xy[pid] for pid in sorted(xy)] + [(0.0, 0.0)]
    inst = SimpleNamespace(
        C=[[math.hypot(a[0] - b[0], a[1] - b[1]) for b in points]
           for a in points],
        p=[prob[pid] for pid in sorted(xy)] + [0.0],
        depot=count, index_to_pid=tuple(sorted(xy)) + (-1,), units="metres")
    cands = tuple(RoomCandidate(room_id=pid, label="R%d" % pid,
                                prob=prob[pid], prob_renorm=prob[pid] / count,
                                xy=xy[pid])
                  for pid in sorted(xy))
    return cands, inst


def test_the_search_is_capped_so_it_never_degrades_to_nearest_first():
    """Past ~15 rooms the exact search times out and gives up the belief.

    Measured on the hospital BEV: from 16 rooms up, every solve returned
    ``route_source='fallback'`` -- nearest-first, which never reads the
    probabilities -- and at 20 rooms the order was identical for a peaked, a
    spread and a flat belief. The cap is what keeps the guarantee.
    """
    cands, inst = wide(24)
    solver = RptStarRoomSolver()          # the SHIPPED params, budget included
    order = solver(cands, inst)
    last = solver.last

    assert last.rooms == DEFAULT_MAX_ROOMS
    assert len(last.capped) == 24 - DEFAULT_MAX_ROOMS
    assert len(order) == DEFAULT_MAX_ROOMS
    assert last.status == "solved", "the cap exists to stop the timeout"
    assert last.guarantee == "optimal"
    assert last.route_source == "search", "nearest-first is the failure"
    assert last.solve_ms < 1000.0
    assert set(order).isdisjoint(last.capped)


def test_the_cap_keeps_the_likeliest_rooms():
    cands, inst = wide(20)
    solver = RptStarRoomSolver(params=EXACT, max_rooms=5)
    solver(cands, inst)
    # prob is 0.9/pid, so the five likeliest are rooms 1..5.
    assert solver.last.capped == tuple(range(6, 21))


def test_the_cap_falls_through_to_the_nearest_rooms_on_a_flat_belief():
    """A flat ranking carries no information, so distance is all that is left.

    Not hypothetical: the oracle publishes a uniform ranking under
    ``source='uniform_fallback'`` whenever the LLM is down. The cap must not
    additionally make the choice arbitrary.
    """
    cands, inst = wide(12, probs=dict((pid, 0.25) for pid in range(1, 13)))
    solver = RptStarRoomSolver(params=EXACT, max_rooms=4)
    solver(cands, inst)
    depot_row = inst.C[inst.depot]
    kept = [pid for pid in range(1, 13) if pid not in solver.last.capped]
    nearest = sorted(range(1, 13), key=lambda pid: depot_row[pid - 1])[:4]
    assert sorted(kept) == sorted(nearest)


def test_the_cap_never_drops_the_room_the_aircraft_is_next_to():
    """A cap ranked on probability alone is greedy-probability as a prefilter.

    That is the baseline RPT* exists to beat, and it would strip out the
    cheap room under the aircraft before the solver could weigh it. Measured
    in the closed loop on the hospital fixture, holding one slot back for the
    nearest room takes expected time-to-find to 0.88 of not holding one.
    """
    # Room 1 is right next to the aircraft but ranked last; rooms 2..6 are
    # far away and likelier. With four slots and no reserve, room 1 goes.
    xy = {1: (0.5, 0.0), 2: (40.0, 0.0), 3: (41.0, 0.0),
          4: (42.0, 0.0), 5: (43.0, 0.0), 6: (44.0, 0.0)}
    prob = {1: 0.01, 2: 0.30, 3: 0.25, 4: 0.20, 5: 0.15, 6: 0.09}
    points = [xy[pid] for pid in sorted(xy)] + [(0.0, 0.0)]
    inst = SimpleNamespace(
        C=[[abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in points]
           for a in points],
        p=[prob[pid] for pid in sorted(xy)] + [0.0],
        depot=len(xy), index_to_pid=tuple(sorted(xy)) + (-1,), units="metres")
    cands = tuple(RoomCandidate(room_id=pid, label="R%d" % pid,
                                prob=prob[pid], prob_renorm=prob[pid],
                                xy=xy[pid])
                  for pid in sorted(xy))

    solver = RptStarRoomSolver(params=EXACT, max_rooms=4)
    order = solver(cands, inst)
    assert NEAREST_RESERVED >= 1
    assert 1 in order, (
        "the nearest room was capped out by the belief: %r" % (order,))
    assert 1 not in solver.last.capped
    assert order[0] == 1, "and it is free to reach, so it goes first"
    # The reserve costs exactly one slot, not more: the three likeliest are
    # still in.
    assert set(order) == {1, 2, 3, 4}


def test_a_record_the_solver_never_produced_does_not_claim_no_route():
    """``no_route`` is something RPT* says after looking, not before."""
    solver = RptStarRoomSolver(params=EXACT, rng=random.Random(2))
    solver(candidates(1, 2, 3), None)
    assert solver.last.source == FALLBACK
    assert solver.last.status == NOT_ATTEMPTED


def test_the_cap_can_be_lifted_and_says_so_when_it_did_not_bite():
    cands, inst = wide(6)
    solver = RptStarRoomSolver(params=EXACT, max_rooms=None)
    assert len(solver(cands, inst)) == 6
    assert solver.last.capped == ()


@pytest.mark.parametrize("off", [0, None, -1, -5])
def test_every_way_of_turning_the_cap_off_keeps_every_room(off):
    """A negative would otherwise slice from the far end.

    ``ranked[:-1]`` is valid Python and drops the LEAST likely room while
    silently keeping the rest -- but ``ranked[:-5]`` on six rooms keeps one,
    and it is the wrong one. Better that every non-positive value means the
    same thing the docstring says zero means.
    """
    cands, inst = wide(6)
    solver = RptStarRoomSolver(params=EXACT, max_rooms=off)
    assert sorted(solver(cands, inst)) == [1, 2, 3, 4, 5, 6]
    assert solver.last.capped == ()


def test_the_record_names_the_vertex_the_tour_was_planned_from():
    """A depot pid other than -1 means the aircraft could not be placed."""
    solver = RptStarRoomSolver(params=EXACT)
    solver(candidates(1, 2, 3), instance())
    assert solver.last.depot_pid == -1, "the aircraft has its own vertex"

    # build_instance falls back to the lowest-pid ROOM when it cannot snap
    # the pose, and then the tour is planned from an assumption.
    fallen_back = instance()
    fallen_back.depot = 0
    fallen_back.index_to_pid = (1, 2, 3, -1)
    solver(candidates(1, 2, 3), fallen_back)
    assert solver.last.depot_pid == 1


# -- the loop -------------------------------------------------------------
def test_the_supervisor_walks_the_whole_rpt_order():
    """The seam itself: injected as the solver, followed room by room."""
    rooms = [RoomOption(room_id=pid, label="R%d" % pid, prob=ROOM_P[pid],
                        xy=ROOM_XY[pid])
             for pid in (1, 2, 3)]
    facts = dict((pid, RoomFacts(room_id=pid, frontier_clusters=1))
                 for pid in (1, 2, 3))
    sup = ObjectSearchSupervisor(ObjectSearchParams(),
                                 solver=RptStarRoomSolver(params=EXACT),
                                 rng=random.Random(7))

    state = sup.update(rooms, facts, DEPOT_XY, now=0.0, last_plan_s=0.1,
                       instance=instance())
    assert state.order == tuple(BEST_ORDER), "the whole tour is committed"
    assert state.action.room_id == BEST_ORDER[0]
    assert sup.stats["solver_calls"] == 1
