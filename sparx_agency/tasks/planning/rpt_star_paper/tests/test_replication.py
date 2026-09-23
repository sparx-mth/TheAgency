"""The replication's own tests: does it measure what it claims to measure?

The studies in :mod:`~sparx_agency.tasks.planning.rpt_star_paper.experiments`
produce numbers, and numbers are easy to produce and hard to trust. These check
the machinery underneath them -- that the datasets follow the paper's stated
recipe, that the belief dial does what its name says, that the metrics are the
exact expectations they are documented to be, and that the planners are
distinguishable rather than five names for the same ordering.

The paper's own theorems are checked by
:func:`~sparx_agency.tasks.planning.rpt_star_paper.experiments.run_fidelity`
against exhaustive enumeration; :func:`test_fidelity_study_passes` runs it and
insists every claim holds.

Run with::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \\
        sparx_agency/tasks/planning/rpt_star_paper/tests/ -q
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.routing.rpt_star import (
    RptStarParams,
    brute_force_order,
    expected_cost,
    lkh_style_order,
    nearest_neighbour_order,
    path_length,
    solve,
)
from sparx_agency.tasks.planning.rpt_star_paper import (
    beliefs,
    datasets,
    experiments,
    metrics,
    planners,
    report,
)


# -- datasets: the paper's stated recipe ---------------------------------


@pytest.mark.parametrize("n", [8, 12, 20])
def test_synthetic_respects_minimum_separation(n):
    """Sec. VII: "the distance between any two vertices must exceed 5"."""
    instance = datasets.synthetic_instance(n, seed=3)
    assert instance.points is not None
    for i in range(n):
        for j in range(i + 1, n):
            gap = math.hypot(instance.points[i][0] - instance.points[j][0],
                             instance.points[i][1] - instance.points[j][1])
            assert gap > datasets.MIN_SEPARATION


def test_synthetic_sampling_area_switches_at_forty_vertices():
    """Sec. VII: 500x500 for |V| <= 40, 5000x5000 above it."""
    small = datasets.synthetic_instance(40, seed=1)
    assert max(max(p) for p in small.points) <= datasets.SMALL_EXTENT
    large = datasets.synthetic_instance(41, seed=1)
    assert max(max(p) for p in large.points) > datasets.SMALL_EXTENT


def test_synthetic_is_metric():
    """Euclidean costs satisfy the triangle inequality Sec. III assumes."""
    instance = datasets.synthetic_instance(12, seed=5)
    count, worst = datasets.triangle_violations(instance.matrix)
    assert count == 0
    assert worst == 0.0


def test_synthetic_is_reproducible():
    """A seed names an instance, so a result can be gone back to."""
    a = datasets.synthetic_instance(10, seed=42)
    b = datasets.synthetic_instance(10, seed=42)
    assert a.matrix == b.matrix


@pytest.mark.parametrize("name", datasets.TSPLIB_INSTANCES)
def test_tsplib_loads_and_is_symmetric_with_zero_diagonal(name):
    """The five instances parse, and look like distance matrices."""
    instance = datasets.tsplib_instance(name)
    assert instance.n == int(name[-2:])
    for i in range(instance.n):
        assert instance.matrix[i][i] == 0.0
        for j in range(instance.n):
            assert instance.matrix[i][j] == instance.matrix[j][i]


@pytest.mark.parametrize("name", datasets.TSPLIB_INSTANCES)
def test_every_paper_tsplib_instance_violates_the_triangle_inequality(name):
    """The finding, pinned as a test so it cannot quietly stop being true.

    Sec. III states the edge costs satisfy the triangle inequality, and Lemma 6
    -- the soundness of dominance pruning -- needs it. Every instance the paper
    benchmarks on breaks it, so Theorem 2 does not apply to the results in
    Table I.
    """
    instance = datasets.tsplib_instance(name)
    count, worst = datasets.triangle_violations(instance.matrix)
    assert count > 0
    assert worst > 0.0


@pytest.mark.parametrize("name", datasets.TSPLIB_INSTANCES)
def test_metric_closure_repairs_every_instance(name):
    """The closure is metric, which is what makes it the honest way to run."""
    instance = datasets.tsplib_instance(name)
    closure = datasets.metric_closure_matrix(instance.matrix)
    count, _ = datasets.triangle_violations(closure)
    assert count == 0


def test_metric_closure_never_lengthens_an_edge():
    """A shortest path is at most the direct edge, by construction."""
    instance = datasets.tsplib_instance("gr17")
    closure = datasets.metric_closure_matrix(instance.matrix)
    for i in range(instance.n):
        for j in range(instance.n):
            assert closure[i][j] <= instance.matrix[i][j] + 1e-9


# -- beliefs: the dial means what it says --------------------------------


def test_truth_sums_to_one_and_stays_in_domain():
    """Remark 1's single-target case, and the paper's p(v) in [0, 1)."""
    for sharpness in (4.0, 1.0, 0.05, 0.01):
        truth = beliefs.sample_truth(12, seed=1, sharpness=sharpness)
        assert abs(sum(truth) - 1.0) < 1e-9
        assert all(0.0 <= p < 1.0 for p in truth)
        assert max(truth) <= beliefs.MAX_PROB + 1e-12


def test_perfect_reliability_returns_the_truth():
    truth = beliefs.sample_truth(10, seed=2, sharpness=0.5)
    belief = beliefs.belief_from_truth(truth, 1.0)
    assert all(abs(a - b) < 1e-9 for a, b in zip(truth, belief))


def test_zero_reliability_is_uniform_from_both_directions():
    """The dial is continuous at zero, so the sweep has no discontinuity."""
    truth = beliefs.sample_truth(10, seed=3, sharpness=0.5)
    for level in (0.0, -0.0):
        belief = beliefs.belief_from_truth(truth, level)
        assert all(abs(p - 0.1) < 1e-9 for p in belief)


def test_adversarial_reverses_the_ranking():
    """The most likely place must become the least likely, not merely flatter."""
    truth = beliefs.sample_truth(10, seed=4, sharpness=0.4)
    belief = beliefs.belief_from_truth(truth, -1.0)
    best_true = max(range(10), key=lambda v: truth[v])
    worst_belief = min(range(10), key=lambda v: belief[v])
    assert best_true == worst_belief


def test_reliability_moves_belief_monotonically_away_from_truth():
    """Lower reliability must mean a bigger gap, or the axis is meaningless."""
    truth = beliefs.sample_truth(12, seed=5, sharpness=0.5)
    gaps = []
    for level in (1.0, 0.75, 0.5, 0.25, 0.0):
        belief = beliefs.belief_from_truth(truth, level)
        gaps.append(sum(abs(a - b) for a, b in zip(truth, belief)))
    assert gaps == sorted(gaps)


def test_decoy_places_its_peak_far_from_the_truth():
    """The paper's misleading prior: confidently wrong, and wrong far away."""
    instance = datasets.synthetic_instance(12, seed=6)
    truth = beliefs.sample_truth(12, seed=6, sharpness=0.4)
    belief = beliefs.decoy_belief(truth, instance.matrix, seed=6)

    true_peak = max(range(12), key=lambda v: truth[v])
    decoy_peak = max(range(12), key=lambda v: belief[v])
    assert decoy_peak != true_peak
    # And it is the furthest vertex, not merely a different one.
    furthest = max(range(12), key=lambda v: instance.matrix[true_peak][v])
    assert decoy_peak == furthest


def test_every_belief_is_a_legal_problem():
    """Whatever the dial is set to, the solver must accept the result."""
    instance = datasets.synthetic_instance(10, seed=7)
    truth = beliefs.sample_truth(10, seed=7, sharpness=0.2)
    candidates = [beliefs.belief_from_truth(truth, level)
                  for level in beliefs.RELIABILITY_LEVELS]
    candidates.append(beliefs.decoy_belief(truth, instance.matrix, seed=7))
    for belief in candidates:
        assert abs(sum(belief) - 1.0) < 1e-6
        planners.build_problem(belief)          # must not raise


# -- metrics: exact expectations, not proxies ----------------------------


def test_distance_under_truth_matches_a_hand_computation():
    """Three places, a known truth: the expectation is short enough to write."""
    matrix = [[0.0, 10.0, 20.0],
              [10.0, 0.0, 5.0],
              [20.0, 5.0, 0.0]]
    truth = [0.2, 0.3, 0.5]
    order = (0, 1, 2)
    # reached: 0 at v0, 10 at v1, 15 at v2
    expected = 0.2 * 0.0 + 0.3 * 10.0 + 0.5 * 15.0
    assert abs(metrics.expected_distance_under_truth(order, truth, matrix)
               - expected) < 1e-12


def test_places_searched_matches_a_hand_computation():
    truth = [0.2, 0.3, 0.5]
    expected = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
    assert abs(metrics.expected_places_searched((0, 1, 2), truth) - expected) \
        < 1e-12


def test_flight_time_is_distance_over_speed_plus_dwell():
    matrix = [[0.0, 10.0], [10.0, 0.0]]
    truth = [0.0, 1.0]
    seconds = metrics.expected_flight_time((0, 1), truth, matrix,
                                           speed_mps=2.0, dwell_s=3.0)
    assert abs(seconds - (10.0 / 2.0 + 3.0 * 2)) < 1e-12


def test_belief_cost_equals_the_solver_objective():
    """The benchmark must not carry its own copy of the objective."""
    instance = datasets.synthetic_instance(9, seed=8)
    belief = beliefs.sample_truth(9, seed=8, sharpness=0.6)
    order = tuple(range(9))
    assert abs(metrics.expected_cost_under_belief(order, belief,
                                                  instance.matrix)
               - expected_cost(order, belief, instance.matrix)) < 1e-12


def test_paper_objective_is_not_expected_search_distance_for_one_target():
    """Remark 1 is wrong, and this is the proof by counterexample.

    Eq. 1 weights the ``i``-th edge by ``prod (1 - p_k)``, the chance of
    independently missing at each place so far. Remark 1 (p.3) says the
    single-target case merely adds ``sum p(v) = 1`` and that the approach still
    applies. It does not: with one target the misses are mutually exclusive, so
    the weight must be ``1 - sum p_k``, which is strictly smaller whenever two
    or more places carry probability.
    """
    probs = [0.0, 0.3, 0.3, 0.4]
    matrix = [[0.0, 1.0, 1.0, 1.0],
              [1.0, 0.0, 1.0, 1.0],
              [1.0, 1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0, 0.0]]
    order = (0, 1, 2, 3)
    paper = expected_cost(order, probs, matrix)
    corrected = metrics.expected_cost_single_target(order, probs, matrix)
    assert paper > corrected
    # q_2 = 1 * 0.7 = 0.70 under the paper; 1 - 0.3 = 0.70 corrected -- equal
    # after one place. The gap opens at the third edge: 0.7*0.7 = 0.49 against
    # 1 - 0.6 = 0.40.
    assert abs(paper - (1.0 + 0.7 + 0.49)) < 1e-12
    assert abs(corrected - (1.0 + 0.7 + 0.40)) < 1e-12


def test_corrected_objective_equals_expected_search_distance():
    """The corrected weight is exactly the distance metric, rearranged."""
    for seed in range(6):
        instance = datasets.synthetic_instance(9, seed=seed)
        truth = beliefs.sample_truth(9, seed=seed, sharpness=0.7)
        order = tuple(range(9))
        assert abs(metrics.expected_cost_single_target(order, truth,
                                                       instance.matrix)
                   - metrics.expected_distance_under_truth(order, truth,
                                                           instance.matrix)) \
            < 1e-9


def test_eq1_route_can_fly_further_than_necessary_with_a_perfect_prior():
    """The cost of Remark 1's error, on real instances.

    Even when the belief *is* the truth, the ordering that minimises Eq. 1 is
    not always the ordering that finds the object soonest. This asserts the
    phenomenon exists; :func:`test_objective_study_quantifies_the_gap` bounds
    how big it gets.
    """
    import itertools
    found_a_gap = False
    for seed in range(12):
        instance = datasets.synthetic_instance(8, seed=seed)
        truth = beliefs.sample_truth(8, seed=seed + 5000, sharpness=1.0)
        others = [v for v in range(8) if v != 0]

        best_paper = min(itertools.permutations(others),
                         key=lambda p: expected_cost((0,) + p, truth,
                                                     instance.matrix))
        best_true = min(itertools.permutations(others),
                        key=lambda p: metrics.expected_distance_under_truth(
                            (0,) + p, truth, instance.matrix))
        paper_distance = metrics.expected_distance_under_truth(
            (0,) + best_paper, truth, instance.matrix)
        true_distance = metrics.expected_distance_under_truth(
            (0,) + best_true, truth, instance.matrix)

        # The Eq. 1 route can never beat the true optimum of the distance it
        # is being scored on.
        assert paper_distance >= true_distance - 1e-9
        if paper_distance > true_distance + 1e-6:
            found_a_gap = True
    assert found_a_gap, "expected at least one instance where Eq. 1 misleads"


def test_objective_study_quantifies_the_gap():
    """The gap is real, and it grows as the belief sharpens."""
    rows = experiments.run_objective(sizes=(7,), seeds=range(10),
                                     sharpnesses=(4.0, 0.25))
    grouped = report.group_by(rows, "sharpness")
    flat = report.median([r["excess_pct"] for r in grouped[(4.0,)]])
    peaked = report.median([r["excess_pct"] for r in grouped[(0.25,)]])
    assert flat >= 0.0 and peaked >= 0.0
    assert peaked > flat


# -- planners: distinct, and behaving as described -----------------------


def test_greedy_visits_in_descending_probability():
    """Sec. VII-C-1: "the largest probability value among the unvisited"."""
    belief = [0.0, 0.1, 0.5, 0.2, 0.2]
    problem = planners.build_problem(belief)
    order = planners.plan_greedy(problem, None).order
    tail = order[1:]
    assert [belief[v] for v in tail] == sorted((belief[v] for v in tail),
                                              reverse=True)


def test_lkh_never_loses_to_nearest_neighbour_on_distance():
    """The stand-in has to be strong, or the comparison flatters RPT*."""
    for seed in range(6):
        instance = datasets.synthetic_instance(11, seed=seed)
        belief = beliefs.sample_truth(11, seed=seed, sharpness=1.0)
        problem = planners.build_problem(belief)
        lkh = lkh_style_order(problem, instance.matrix)
        nn = nearest_neighbour_order(problem, instance.matrix)
        assert path_length(lkh, instance.matrix) \
            <= path_length(nn, instance.matrix) + 1e-9


def test_every_planner_returns_a_permutation_starting_at_the_start():
    """Def. 2: a path from v_s visiting every vertex exactly once."""
    instance = datasets.synthetic_instance(10, seed=11)
    belief = beliefs.sample_truth(10, seed=11, sharpness=0.6)
    problem = planners.build_problem(belief)
    for name, planner in planners.default_planners(include_optimal=True):
        order = planner(problem, instance.matrix).order
        assert order[0] == problem.start, name
        assert sorted(order) == list(range(10)), name


def test_zero_heuristic_ablation_still_returns_the_optimum():
    """RPT*_noh is exact -- h = 0 is admissible, so Theorem 2 still holds."""
    for seed in range(5):
        instance = datasets.synthetic_instance(9, seed=seed)
        belief = beliefs.sample_truth(9, seed=seed, sharpness=0.7)
        problem = planners.build_problem(belief)
        _, optimum = brute_force_order(problem, instance.matrix)
        result = planners.plan_rpt_star(problem, instance.matrix,
                                        use_heuristic=False)
        assert abs(expected_cost(result.order, belief, instance.matrix)
                   - optimum) < 1e-9


def test_heuristic_reduces_expansions():
    """Sec. VII-B-1's claim, as an inequality rather than a percentage."""
    total_with, total_without = 0, 0
    for seed in range(8):
        instance = datasets.synthetic_instance(10, seed=seed)
        belief = beliefs.sample_truth(10, seed=seed, sharpness=0.8)
        problem = planners.build_problem(belief)
        total_with += planners.plan_rpt_star(problem,
                                             instance.matrix).expansions
        total_without += planners.plan_rpt_star(
            problem, instance.matrix, use_heuristic=False).expansions
    assert total_with < total_without


def test_focal_search_honours_its_bound():
    """Theorem 3, on data rather than on paper."""
    for seed in range(5):
        instance = datasets.synthetic_instance(9, seed=seed)
        belief = beliefs.sample_truth(9, seed=seed, sharpness=0.7)
        problem = planners.build_problem(belief)
        _, optimum = brute_force_order(problem, instance.matrix)
        for epsilon in planners.PAPER_EPSILONS:
            result = planners.plan_rpt_star(problem, instance.matrix,
                                            epsilon=epsilon)
            cost = expected_cost(result.order, belief, instance.matrix)
            assert cost <= (1.0 + epsilon) * optimum + 1e-9


# -- the studies run, and say what they are supposed to say --------------


def test_fidelity_study_passes():
    """Every theorem the paper states, checked against enumeration."""
    rows = experiments.run_fidelity(sizes=(6, 7), seeds=range(4))
    assert rows
    for row in rows:
        assert row["rpt_is_optimal"]
        assert row["noh_is_optimal"]
        assert row["lemma1_holds"]
        assert row["heuristic_admissible"]
        assert all(f["within_bound"] for f in row["focal"])


def test_reliability_study_shows_greedy_degrading_faster_than_rpt_star():
    """The headline: a wrong prior costs Greedy far more than it costs RPT*.

    This is the paper's Table II/III finding, and it is the one result here
    that reproduces cleanly.
    """
    rows = experiments.run_reliability(n=10, seeds=range(8))
    grouped = report.group_by(rows, "planner", "regime")

    def value(planner, regime):
        return report.median([r["distance"]
                              for r in grouped[(planner, regime)]])

    rpt_penalty = value("RPT*", "adversarial") / value("RPT*", "perfect")
    greedy_penalty = value("Greedy", "adversarial") / value("Greedy", "perfect")
    assert greedy_penalty > rpt_penalty


def test_reliability_study_shows_distance_only_winning_when_the_prior_is_wrong():
    """The contradiction, pinned: below uninformative, ignoring the belief wins.

    The paper claims RPT* "yields the best performance in the presence of
    misleading prior knowledge" (Sec. VII-E). For the single-shot HPP-PT it
    defines, that is not what happens -- a planner that ignores the belief
    outright flies less far, because RPT* faithfully optimises against an
    input that is actively wrong.
    """
    rows = experiments.run_reliability(n=10, seeds=range(8))
    grouped = report.group_by(rows, "planner", "regime")

    def value(planner, regime):
        return report.median([r["distance"]
                              for r in grouped[(planner, regime)]])

    assert value("RPT*", "perfect") < value("LKH", "perfect")
    assert value("LKH", "adversarial") < value("RPT*", "adversarial")


def test_sharpness_study_shows_the_baselines_moving_in_opposite_directions():
    """Why no single belief reproduces both of the paper's headline numbers.

    As the belief sharpens, ``Greedy`` improves and ``LKH`` degrades. The
    paper reports ``Greedy`` at 2-3x and ``LKH`` at 1.5-1.8x as though from one
    experiment; those live at opposite ends of this axis.
    """
    rows = experiments.run_sharpness(n=10, seeds=range(8),
                                     sharpnesses=(2.0, 0.05))
    grouped = report.group_by(rows, "planner", "sharpness")

    def ratio(planner, sharpness):
        return report.median([r["ratio_to_best"]
                              for r in grouped[(planner, sharpness)]])

    assert ratio("Greedy", 2.0) > ratio("Greedy", 0.05)
    assert ratio("LKH", 2.0) < ratio("LKH", 0.05)


def test_tsplib_study_withdraws_the_guarantee_on_the_published_matrix():
    """The violation is real and the solver refuses to certify the result.

    Run on gr17 only -- the smallest -- because exact search on a 29-vertex
    instance in pure Python is not a unit test.
    """
    rows = experiments.run_tsplib(instances=("gr17",), seeds=range(1),
                                  time_limit_s=20.0)
    published = [r for r in rows if r["matrix"] == "as-published"]
    assert published
    for row in published:
        assert row["triangle_violations"] > 0
        # No optimality claim survives a waived precondition.
        assert row["guarantee"] == "none"

    closed = [r for r in rows if r["matrix"] == "metric-closure"]
    for row in closed:
        assert row["triangle_violations"] == 0
        # The reference route is its own reference, so no penalty by
        # construction; this guards the bookkeeping, not the algorithm.
        assert abs(row["penalty_pct"]) < 1e-9


def test_report_tables_render():
    """The summaries must survive real rows, including the NaN reliability."""
    rows = experiments.run_reliability(n=8, seeds=range(3))
    for column in ("distance", "flight_time_s", "planning_s",
                   "places_searched"):
        assert report.summarise_reliability(rows, column)

