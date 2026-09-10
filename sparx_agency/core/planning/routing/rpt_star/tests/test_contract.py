"""What the package refuses, what it merely grumbles about, and the odd shapes.

Three groups, and the split between the first two is the design:

* **refusals** -- inputs that would make the answer wrong rather than merely
  strange. Every one of these raises, because the alternative is a route that
  looks fine and is not optimal, which nothing downstream can detect;
* **grumbles** -- inputs that are unusual and perfectly legal. The important
  one is probabilities that do not sum to one: the paper is explicit that the
  algorithm does not depend on it (Remark 1, p.3), a language model's ranking
  breaks it constantly, and refusing it would make the package unusable for the
  thing it is for;
* **degenerate shapes** -- one vertex, two vertices, everything equally likely,
  nothing likely at all. These are where a search quietly returns nonsense
  rather than failing.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.routing.rpt_star import (
    GUARANTEE_NONE,
    GUARANTEE_OPTIMAL,
    STATUS_BUDGET_EXCEEDED,
    STATUS_SOLVED,
    DisconnectedGraphError,
    InvalidCostError,
    InvalidProblemError,
    NO_EDGE,
    RouteProblem,
    RouteVertex,
    RptStarParams,
    TriangleInequalityError,
    costs_from_pairs,
    costs_from_points,
    dense_costs,
    metric_closure,
    nearest_neighbour_order,
    solve,
)

EXACT = RptStarParams(epsilon=None, time_budget_s=None)


def square(probs):
    """Four places at the corners of a unit square, with the given beliefs."""
    vertices = [RouteVertex(id="v%d" % i, prob=p) for i, p in enumerate(probs)]
    problem = RouteProblem(vertices, start_id="v0")
    matrix = costs_from_points([(0, 0), (1, 0), (1, 1), (0, 1)])
    return problem, matrix


# -- refusals -------------------------------------------------------------

def test_probability_of_one_is_refused():
    """``p = 1`` is outside the paper's range, and the heuristic divides by 1-p.

    Not a boundary case to handle gracefully -- a certainty is not a search
    problem, and admitting it would divide by zero in Eq. (6).
    """
    with pytest.raises(InvalidProblemError, match=r"\[0.0, 1.0\)"):
        RouteProblem([RouteVertex(id="a", prob=1.0),
                      RouteVertex(id="b", prob=0.0)], start_id="a")


def test_negative_probability_is_refused():
    with pytest.raises(InvalidProblemError):
        RouteProblem([RouteVertex(id="a", prob=-0.1),
                      RouteVertex(id="b", prob=0.2)], start_id="a")


def test_duplicate_vertex_ids_are_refused():
    """Two places with one name would make the returned order ambiguous."""
    with pytest.raises(InvalidProblemError, match="duplicate"):
        RouteProblem([RouteVertex(id="a", prob=0.1),
                      RouteVertex(id="a", prob=0.2)], start_id="a")


def test_unknown_start_is_refused_and_says_what_to_use_instead():
    with pytest.raises(InvalidProblemError, match="with_external_start"):
        RouteProblem([RouteVertex(id="a", prob=0.1)], start_id="robot")


def test_empty_problem_is_refused():
    with pytest.raises(InvalidProblemError):
        RouteProblem([], start_id="a")


def test_an_unreachable_place_is_refused_rather_than_priced():
    """A missing edge makes the instance infeasible, not merely expensive.

    The tempting alternative -- a very large finite cost -- looks feasible and
    silently distorts the ordering of everything else, so the package will not
    accept it by omission.
    """
    problem = RouteProblem(
        [RouteVertex(id="a", prob=0.1), RouteVertex(id="b", prob=0.2),
         RouteVertex(id="c", prob=0.3)], start_id="a")
    matrix = costs_from_pairs(3, {(0, 1): 1.0, (1, 0): 1.0, (0, 2): 1.0,
                                  (2, 0): 1.0, (1, 2): 1.0})
    with pytest.raises(DisconnectedGraphError, match="infeasible"):
        solve(problem, matrix, EXACT)


def test_a_detour_that_beats_going_direct_is_refused():
    """The triangle inequality is what makes dominance pruning sound.

    Violating it does not crash the search; it makes the answer quietly
    sub-optimal, which is the worst possible failure mode. So it is checked.
    """
    problem = RouteProblem(
        [RouteVertex(id="a", prob=0.1), RouteVertex(id="b", prob=0.2),
         RouteVertex(id="c", prob=0.3)], start_id="a")
    rows = [[0.0, 1.0, 100.0], [1.0, 0.0, 1.0], [100.0, 1.0, 0.0]]
    with pytest.raises(TriangleInequalityError, match="triangle inequality"):
        solve(problem, dense_costs(rows), EXACT)


def test_the_triangle_error_names_the_offending_places_by_caller_id():
    """The message has to be actionable, so it carries ids and not indices."""
    problem = RouteProblem(
        [RouteVertex(id="kitchen", prob=0.1), RouteVertex(id="hall", prob=0.2),
         RouteVertex(id="ward", prob=0.3)], start_id="kitchen")
    rows = [[0.0, 1.0, 100.0], [1.0, 0.0, 1.0], [100.0, 1.0, 0.0]]
    with pytest.raises(TriangleInequalityError) as caught:
        solve(problem, dense_costs(rows), EXACT)
    assert set(caught.value.triple) == {"kitchen", "hall", "ward"}
    assert caught.value.excess == pytest.approx(98.0)


def test_metric_closure_repairs_a_matrix_the_checker_rejects():
    """The offered fix actually works, and it is the caller who applies it."""
    problem = RouteProblem(
        [RouteVertex(id="a", prob=0.1), RouteVertex(id="b", prob=0.2),
         RouteVertex(id="c", prob=0.3)], start_id="a")
    rows = [[0.0, 1.0, 100.0], [1.0, 0.0, 1.0], [100.0, 1.0, 0.0]]
    repaired = metric_closure(dense_costs(rows))
    assert repaired[0][2] == pytest.approx(2.0)     # a->b->c beats a->c
    solution = solve(problem, repaired, EXACT)
    assert solution.guarantee == GUARANTEE_OPTIMAL


def test_a_non_square_matrix_is_refused():
    with pytest.raises(InvalidCostError, match="square"):
        dense_costs([[0.0, 1.0], [1.0]])


def test_a_negative_cost_is_refused():
    """A negative edge would let flying further reduce the expected cost."""
    with pytest.raises(InvalidCostError, match="negative"):
        dense_costs([[0.0, -1.0], [1.0, 0.0]])


# -- grumbles -------------------------------------------------------------

def test_probabilities_that_do_not_sum_to_one_are_allowed_but_noted():
    """Remark 1 (p.3): the algorithm does not depend on normalisation.

    A language model's ranking almost never sums to one after filtering, so
    refusing it would make the package useless for its actual purpose.
    """
    problem, matrix = square([0.0, 0.9, 0.9, 0.9])
    solution = solve(problem, matrix, EXACT)
    assert solution.found
    assert any("sum to" in w for w in solution.warnings)


def test_a_normalised_ranking_produces_no_warning():
    problem, matrix = square([0.0, 0.5, 0.3, 0.2])
    assert solve(problem, matrix, EXACT).warnings == ()


def test_all_zero_probabilities_reduce_to_the_shortest_path_and_say_so():
    """With no belief, HPP-PT is the plain Hamiltonian path problem (p.1).

    So the answer must agree with a distance-only ordering on a instance where
    the shortest route is unambiguous -- the corners of a square, in order.
    """
    problem, matrix = square([0.0, 0.0, 0.0, 0.0])
    solution = solve(problem, matrix, EXACT)
    # Three unit sides of the square: no route around four corners is shorter.
    assert solution.expected_cost == pytest.approx(3.0)
    # And with no belief to weigh, going to the nearest place each time is
    # already optimal -- which is the sense in which the problem has reduced.
    nearest = nearest_neighbour_order(problem, matrix)
    from sparx_agency.core.planning.routing.rpt_star import expected_cost
    assert expected_cost(nearest, problem.probs, matrix) == pytest.approx(
        solution.expected_cost)
    assert any("plain shortest Hamiltonian path" in w
               for w in solution.warnings)


# -- degenerate shapes ----------------------------------------------------

def test_a_single_place_is_already_solved():
    """The robot is standing on the only place there is. Cost zero, no move."""
    problem = RouteProblem([RouteVertex(id="here", prob=0.5)], start_id="here")
    solution = solve(problem, dense_costs([[0.0]]), EXACT)
    assert solution.status == STATUS_SOLVED
    assert solution.guarantee == GUARANTEE_OPTIMAL
    assert solution.order == ("here",)
    assert solution.next_id is None
    assert solution.expected_cost == 0.0


def test_two_places_have_exactly_one_route():
    """There is nothing to optimise, and the cost must still be right."""
    problem = RouteProblem(
        [RouteVertex(id="here", prob=0.0), RouteVertex(id="there", prob=0.4)],
        start_id="here")
    solution = solve(problem, dense_costs([[0.0, 7.0], [7.0, 0.0]]), EXACT)
    assert solution.order == ("here", "there")
    assert solution.next_id == "there"
    assert solution.expected_cost == pytest.approx(7.0)


def test_an_external_start_costs_nothing_to_be_at():
    """The robot's own position takes no probability mass and no travel.

    So the expected cost of the whole route equals the cost of the route from
    the first real place onwards, plus the leg that gets there -- which is what
    makes costs comparable between replans from different positions.
    """
    rooms = [RouteVertex(id="a", prob=0.5), RouteVertex(id="b", prob=0.3)]
    problem = RouteProblem.with_external_start(rooms, payload=(0.0, 0.0))
    assert problem.n == 3
    assert problem.start == 0
    assert problem.probs[0] == 0.0
    assert problem.vertex(0).payload == (0.0, 0.0)
    matrix = costs_from_points([(0, 0), (1, 0), (2, 0)])
    solution = solve(problem, matrix, EXACT)
    # start -> a costs 1 at survival 1.0, a -> b costs 1 at survival 0.5.
    assert solution.expected_cost == pytest.approx(1.0 + 0.5)


def test_an_external_start_may_not_collide_with_a_candidate():
    with pytest.raises(InvalidProblemError, match="duplicate"):
        RouteProblem.with_external_start(
            [RouteVertex(id="__start__", prob=0.5)])


def test_payload_and_label_survive_the_round_trip():
    """The solver never reads them, so it must never lose them either."""
    rooms = [RouteVertex(id="a", prob=0.5, label="kitchen", payload={"x": 1}),
             RouteVertex(id="b", prob=0.3, label="ward", payload={"x": 2})]
    problem = RouteProblem.with_external_start(rooms)
    assert problem.vertex(1).label == "kitchen"
    assert problem.vertex(1).payload == {"x": 1}


# -- the budget -----------------------------------------------------------

def test_an_exhausted_budget_returns_a_route_and_withdraws_the_guarantee():
    """Running out of time must not mean handing a mission node ``None``.

    But it must also not look like success: the status says what happened, the
    guarantee is withdrawn, and the lower bound still says how good the answer
    could possibly have been.
    """
    problem, matrix = square([0.0, 0.01, 0.01, 0.01])
    solution = solve(problem, matrix,
                     RptStarParams(epsilon=None, max_expansions=1,
                                   time_budget_s=None))
    assert solution.status == STATUS_BUDGET_EXCEEDED
    assert solution.guarantee == GUARANTEE_NONE
    assert solution.lower_bound <= solution.expected_cost + 1e-12


def test_a_budget_that_is_never_reached_leaves_the_guarantee_intact():
    problem, matrix = square([0.0, 0.4, 0.3, 0.2])
    solution = solve(problem, matrix,
                     RptStarParams(epsilon=None, max_expansions=10 ** 6,
                                   time_budget_s=None))
    assert solution.status == STATUS_SOLVED
    assert solution.guarantee == GUARANTEE_OPTIMAL


def test_the_lower_bound_is_a_real_bound_even_when_the_search_is_cut_short():
    """Whatever happened, no ordering of these places can cost less."""
    problem, matrix = square([0.0, 0.02, 0.02, 0.02])
    cut = solve(problem, matrix, RptStarParams(epsilon=None, max_expansions=1,
                                               time_budget_s=None))
    full = solve(problem, matrix, EXACT)
    assert cut.lower_bound <= full.expected_cost + 1e-9
