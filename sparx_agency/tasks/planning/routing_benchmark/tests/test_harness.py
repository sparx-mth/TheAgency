"""The benchmark has to be right before its numbers mean anything.

A measuring instrument that is subtly wrong does not produce obvious nonsense;
it produces confident, plausible, wrong conclusions, and there is nothing in
the output to warn you. So the harness gets the same treatment as the solver.

Four things are checked, and the third is the one that would otherwise ruin
everything quietly:

* **the buildings are legal problems** -- distances metric, symmetric,
  positive. RPT* refuses a cost matrix that violates the triangle inequality,
  so a generator that produced one would be testing the validator instead of
  the search;
* **the beliefs are distributions**, and the oracle regimes actually differ in
  the way they claim to;
* **the scoring is right**, checked against a literal simulation of walking the
  route. This is where a benchmark goes wrong: an off-by-one in which leg is
  charged would shift every planner by one hop and change the winner;
* **the planners are wired up correctly** -- and in particular that RPT*
  really does achieve the lowest cost *under the belief it was given*, which
  is the only thing it promises. If it does not, the harness is feeding it the
  wrong problem.
"""
from __future__ import annotations

import itertools
import random

import pytest

from sparx_agency.core.planning.routing.rpt_star import (
    RouteProblem,
    RouteVertex,
    dense_costs,
    expected_cost,
    validate,
)
from sparx_agency.tasks.planning.routing_benchmark.buildings import (
    TOPOLOGIES,
    generate_building,
)
from sparx_agency.tasks.planning.routing_benchmark.metrics import (
    clairvoyant_distance,
    evaluate,
)
from sparx_agency.tasks.planning.routing_benchmark.oracle import (
    ORACLE_MODELS,
    make_belief,
)
from sparx_agency.tasks.planning.routing_benchmark.planners import PLANNERS
from sparx_agency.tasks.planning.routing_benchmark.scenarios import (
    SIZES,
    build_scenario,
)

SMALL = (6, 9)


def scenarios(sizes=SMALL, seeds=(0, 1)):
    """A spread across every topology and oracle, kept small enough to be fast."""
    for topology in TOPOLOGIES:
        for n_rooms in sizes:
            for model in ORACLE_MODELS:
                for seed in seeds:
                    yield build_scenario(topology, n_rooms, model, seed)


# -- the buildings --------------------------------------------------------

def test_every_generated_building_is_a_legal_routing_problem():
    """Distances must be metric, or the solver rejects them outright.

    The strongest possible form of this test: hand the matrix to the real
    validator the solver uses. If a building ever failed here, every number the
    benchmark produced from it would be meaningless.
    """
    for scenario in scenarios():
        vertices = [RouteVertex(id=i, prob=p)
                    for i, p in enumerate(scenario.belief.belief)]
        vertices.append(RouteVertex(id=scenario.n_rooms, prob=0.0))
        problem = RouteProblem(vertices, start_id=scenario.n_rooms)
        validate(problem, dense_costs(scenario.distance),
                 require_triangle_inequality=True)


def test_distances_are_symmetric_and_positive():
    """Walking there and back costs the same, and nothing is free or negative."""
    for scenario in scenarios():
        matrix = scenario.distance
        size = len(matrix)
        for a in range(size):
            assert matrix[a][a] == 0.0
            for b in range(size):
                if a == b:
                    continue
                assert matrix[a][b] > 0.0
                assert matrix[a][b] == pytest.approx(matrix[b][a], rel=1e-12)


def test_the_topologies_actually_differ():
    """A cross is not a corridor: dead-end wings cost more to cover.

    Guards against every topology quietly collapsing to the same generator,
    which would make the whole factor meaningless while still producing a full
    results table.
    """
    spread = {}
    for topology in TOPOLOGIES:
        totals = []
        for seed in range(4):
            scenario = build_scenario(topology, 12, ORACLE_MODELS[1], seed)
            matrix = scenario.distance
            totals.append(sum(matrix[scenario.entrance][r]
                              for r in range(scenario.n_rooms)))
        spread[topology] = sum(totals) / len(totals)
    assert len(set(round(v, 3) for v in spread.values())) == len(TOPOLOGIES)


def test_a_building_needs_at_least_two_rooms():
    with pytest.raises(ValueError):
        generate_building("corridor", 1, random.Random(0))


def test_an_unknown_topology_is_refused():
    with pytest.raises(ValueError, match="unknown topology"):
        generate_building("spiral", 8, random.Random(0))


# -- the beliefs ----------------------------------------------------------

def test_truth_and_belief_are_both_distributions():
    for scenario in scenarios():
        assert sum(scenario.belief.truth) == pytest.approx(1.0)
        assert sum(scenario.belief.belief) == pytest.approx(1.0)
        assert all(p >= 0.0 for p in scenario.belief.belief)
        assert all(p < 1.0 for p in scenario.belief.belief), \
            "a probability of exactly 1.0 is outside the solver's domain"


def test_the_oracle_regimes_are_ordered_by_how_right_they_are():
    """A perfect oracle must agree with the truth more than a hopeless one.

    Without this the regime labels would be decoration, and a chart grouped by
    them would be grouping by nothing.
    """
    agreement = {}
    for model in ORACLE_MODELS:
        scores = [build_scenario(t, 12, model, s).belief.agreement()
                  for t in TOPOLOGIES for s in range(3)]
        agreement[model.name] = sum(scores) / len(scores)
    # Not exactly 1.0: FLOOR guards the logarithm, which perturbs the
    # reconstruction in the eighth decimal place.
    assert agreement["perfect"] == pytest.approx(1.0, abs=1e-6)
    assert agreement["perfect"] > agreement["accurate"]
    assert agreement["accurate"] > agreement["adversarial"]
    assert agreement["decoy"] < agreement["accurate"]


def test_the_decoy_lands_somewhere_expensive_and_wrong():
    """The planted peak has to be a costly mistake, not a cheap one.

    A decoy next door would be indistinguishable from noise; the paper's
    misleading prior puts its second peak far away (p.13), which is what makes
    following it expensive.
    """
    for topology in TOPOLOGIES:
        scenario = build_scenario(topology, 12, ORACLE_MODELS[4], 0)
        belief, truth = scenario.belief.belief, scenario.belief.truth
        decoy = max(range(scenario.n_rooms), key=lambda r: belief[r])
        assert truth[decoy] < 1.0 / scenario.n_rooms, "decoy is a likely room"
        from_start = scenario.distance[scenario.entrance]
        median = sorted(from_start[:scenario.n_rooms])[scenario.n_rooms // 2]
        assert from_start[decoy] >= median, "decoy is not far away"


# -- the scoring ----------------------------------------------------------

def test_expected_distance_matches_walking_the_route_room_by_room():
    """The exact expectation must equal the literal simulation of the walk.

    Computed here the slow, obvious way -- for each possible true room, walk
    the order until you reach it and add up the legs -- and compared against
    the closed form the benchmark actually uses.
    """
    for scenario in scenarios(sizes=(6,), seeds=(0,)):
        matrix = scenario.distance
        truth = scenario.belief.truth
        for _, planner in PLANNERS:
            order = planner(scenario.belief.belief, matrix).order
            literal = 0.0
            for target, probability in enumerate(truth):
                walked = 0.0
                for position in range(1, len(order)):
                    walked += matrix[order[position - 1]][order[position]]
                    if order[position] == target:
                        break
                literal += probability * walked
            assert evaluate(order, truth, matrix).distance == pytest.approx(
                literal, rel=1e-12)


def test_nobody_beats_the_clairvoyant_bound():
    """Knowing the answer in advance is unbeatable, so it bounds everything."""
    for scenario in scenarios():
        bound = clairvoyant_distance(scenario.belief.truth, scenario.distance,
                                     scenario.entrance)
        for _, planner in PLANNERS:
            order = planner(scenario.belief.belief, scenario.distance).order
            assert evaluate(order, scenario.belief.truth,
                            scenario.distance).distance >= bound - 1e-9


def test_rooms_searched_counts_rooms_and_not_the_entrance():
    """The entrance is walked through, never searched."""
    for scenario in scenarios(sizes=(6,), seeds=(0,)):
        outcome = evaluate(tuple([scenario.entrance]
                                 + list(range(scenario.n_rooms))),
                           scenario.belief.truth, scenario.distance)
        assert 1.0 <= outcome.rooms_searched <= scenario.n_rooms


# -- the planners ---------------------------------------------------------

def test_every_planner_returns_a_full_tour_from_the_entrance():
    """A planner that skipped a room would be scored as if it had searched it."""
    for scenario in scenarios():
        for name, planner in PLANNERS:
            order = planner(scenario.belief.belief, scenario.distance).order
            assert order[0] == scenario.entrance, name
            assert sorted(order) == list(range(scenario.n_rooms + 1)), name


def test_rpt_star_really_does_minimise_the_cost_it_was_given():
    """The only thing RPT* promises: the lowest expected cost *under the belief*.

    This is the test that catches the harness feeding the solver the wrong
    problem -- swapped probabilities, a transposed matrix, the entrance in the
    wrong place. Every one of those would still produce a plausible-looking
    results table, and every one would make RPT* lose to a baseline on its own
    objective, which cannot happen if it is wired up right.
    """
    for scenario in scenarios(sizes=(6, 9), seeds=(0, 1)):
        probs = list(scenario.belief.belief) + [0.0]
        costs = scenario.distance
        best = None
        for name, planner in PLANNERS:
            order = planner(scenario.belief.belief, costs).order
            believed = expected_cost(order, probs, costs)
            if name == "rpt_star":
                best = believed
            else:
                assert best is not None
                assert best <= believed + 1e-9, (
                    "%s beat rpt_star on rpt_star's own objective in %s"
                    % (name, scenario.key))


def test_rpt_star_matches_brute_force_on_the_smallest_scenarios():
    """And that its answer really is the optimum, on this harness's own data."""
    from sparx_agency.core.planning.routing.rpt_star import brute_force_order

    for scenario in scenarios(sizes=(6,), seeds=(0, 1)):
        probs = list(scenario.belief.belief) + [0.0]
        vertices = [RouteVertex(id=i, prob=p) for i, p in enumerate(probs)]
        problem = RouteProblem(vertices, start_id=scenario.entrance)
        matrix = dense_costs(scenario.distance)
        _, optimum = brute_force_order(problem, matrix)
        order = PLANNERS[0][1](scenario.belief.belief,
                               scenario.distance).order
        assert expected_cost(order, probs, scenario.distance) == pytest.approx(
            optimum, abs=1e-9)
