"""The paper's own experiments, plus the one it left out.

Five studies. Four replicate something Section VII claims; the fifth measures
the axis Section VII never varies.

``fidelity``
    Not in the paper -- this is the check that the implementation earns the
    right to be called a replication. RPT* against exhaustive enumeration,
    Lemma 1 against Eq. 1, the heuristic against the true cost-to-go, and the
    F-RPT* bound against the optimum it claims to be near.

``ablation``
    Sec. VII-B-1 and VII-B-3. What the Eq. 6 heuristic is worth (the paper
    says 50-70% of the runtime), and what varying ``epsilon`` does (the paper
    says very little).

``baselines``
    Sec. VII-C. RPT* against ``Greedy`` and ``LKH`` on solution cost and
    runtime. The paper reports ``Greedy`` at two to three times optimal and
    ``LKH`` at 50-80% above it.

``reliability``
    **Not in the paper.** Section VII fixes one belief per instance and never
    says what it was. This sweeps the belief from perfect to adversarial and
    measures flight distance, flight time and planning time at each step. It is
    the experiment that decides whether RPT* is worth flying, because a
    planner that only wins when its input is already correct is not solving the
    problem it was motivated by.

``tsplib``
    Sec. VII-A on the five named instances -- and the finding that all five
    violate the triangle inequality that Sec. III assumes and Lemma 6 needs.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.routing.rpt_star import (
    RptStarParams,
    brute_force_order,
    expected_cost,
    expected_cost_literal,
    solve,
)
from sparx_agency.tasks.planning.rpt_star_paper import beliefs as belief_mod
from sparx_agency.tasks.planning.rpt_star_paper import datasets, metrics
from sparx_agency.tasks.planning.rpt_star_paper.planners import (
    PAPER_EPSILONS,
    PAPER_TIME_LIMIT_S,
    build_problem,
    default_planners,
    plan_greedy,
    plan_lkh,
    plan_nearest_neighbour,
    plan_optimal,
    plan_rpt_star,
)


def run_fidelity(sizes=(6, 7, 8, 9), seeds=range(12), sharpness=1.0):
    # type: (Sequence[int], Sequence[int], float) -> List[Dict]
    """Check the implementation against the paper's own theorems.

    Every instance is small enough to enumerate exhaustively, so each claim is
    checked against ground truth rather than against another implementation of
    the same idea.

    Args:
        sizes: Vertex counts to test.
        seeds: Instance seeds per size.
        sharpness: Dirichlet concentration for the belief.

    Returns:
        One row per instance, each carrying the four checks as booleans and the
        numbers behind them.
    """
    rows = []                                   # type: List[Dict]
    for n in sizes:
        for seed in seeds:
            instance = datasets.synthetic_instance(n, seed)
            matrix = instance.matrix
            belief = belief_mod.sample_truth(n, seed + 5000, sharpness)
            problem = build_problem(belief)

            best_order, best_cost = brute_force_order(problem, matrix)
            exact = plan_rpt_star(problem, matrix)
            unguided = plan_rpt_star(problem, matrix, use_heuristic=False)

            exact_cost = expected_cost(exact.order, belief, matrix)
            unguided_cost = expected_cost(unguided.order, belief, matrix)

            # Lemma 1: the q-weighted sum equals Eq. 1 as printed.
            literal = expected_cost_literal(exact.order, belief, matrix)

            # Lemma 5: h never exceeds the true cost-to-go, checked at the
            # initial state where the true remaining cost is the optimum.
            from sparx_agency.core.planning.routing.rpt_star import GammaTable
            gamma = GammaTable(belief, matrix)
            h_start = gamma.estimate(problem.start, 1.0 - belief[problem.start],
                                     1)
            admissible = h_start <= best_cost + 1e-9

            bound_rows = []
            for epsilon in PAPER_EPSILONS:
                focal = plan_rpt_star(problem, matrix, epsilon=epsilon)
                focal_cost = expected_cost(focal.order, belief, matrix)
                bound_rows.append({
                    "epsilon": epsilon,
                    "cost": focal_cost,
                    "within_bound": focal_cost <= (1.0 + epsilon) * best_cost
                                    + 1e-9,
                })

            rows.append({
                "instance": instance.name,
                "n": n,
                "optimal_cost": best_cost,
                "rpt_cost": exact_cost,
                "rpt_is_optimal": abs(exact_cost - best_cost) <= 1e-9,
                "noh_is_optimal": abs(unguided_cost - best_cost) <= 1e-9,
                "lemma1_holds": abs(literal - exact_cost) <= 1e-6 * max(
                    1.0, abs(exact_cost)),
                "heuristic_admissible": admissible,
                "focal": bound_rows,
                "expansions": exact.expansions,
                "expansions_noh": unguided.expansions,
            })
    return rows


def run_ablation(sizes=(8, 10, 12, 14, 16), seeds=range(5), sharpness=1.0,
                 time_limit_s=PAPER_TIME_LIMIT_S):
    # type: (Sequence[int], Sequence[int], float, float) -> List[Dict]
    """Sec. VII-B: what the heuristic buys, and what epsilon buys.

    The paper uses 20 instances per size up to 40 vertices in C++. This is
    pure Python and the search is exponential, so the defaults are smaller;
    the shape of the result is what transfers, not the absolute seconds.

    Args:
        sizes: Vertex counts.
        seeds: Instances per size.
        sharpness: Dirichlet concentration for the belief.
        time_limit_s: Per-instance budget.

    Returns:
        One row per (size, seed, variant).
    """
    rows = []                                   # type: List[Dict]
    variants = [("RPT*", dict(use_heuristic=True)),
                ("RPT*_noh", dict(use_heuristic=False))]
    for epsilon in PAPER_EPSILONS:
        variants.append(("F-RPT*(%g)" % epsilon, dict(epsilon=epsilon)))

    for n in sizes:
        for seed in seeds:
            instance = datasets.synthetic_instance(n, seed)
            belief = belief_mod.sample_truth(n, seed + 5000, sharpness)
            problem = build_problem(belief)
            for name, kwargs in variants:
                result = plan_rpt_star(problem, instance.matrix,
                                       time_limit_s=time_limit_s, **kwargs)
                rows.append({
                    "instance": instance.name,
                    "n": n,
                    "seed": seed,
                    "variant": name,
                    "planning_s": result.planning_s,
                    "expansions": result.expansions,
                    "solved": result.solved,
                    "cost": expected_cost(result.order, belief,
                                          instance.matrix)
                            if result.order else float("inf"),
                })
    return rows


def run_baselines(sizes=(8, 10, 12, 14, 16), seeds=range(5), sharpness=1.0,
                  time_limit_s=PAPER_TIME_LIMIT_S):
    # type: (Sequence[int], Sequence[int], float, float) -> List[Dict]
    """Sec. VII-C: RPT* against Greedy and LKH on cost and runtime.

    Scored under the belief, which is the comparison the paper makes -- it
    reports "solution cost", meaning the HPP-PT objective. The truth-scored
    version of the same question is :func:`run_reliability`.

    Args:
        sizes: Vertex counts.
        seeds: Instances per size.
        sharpness: Dirichlet concentration for the belief.
        time_limit_s: Per-instance budget.

    Returns:
        One row per (size, seed, planner), carrying the ratio to the best
        cost seen on that instance.
    """
    rows = []                                   # type: List[Dict]
    planners = default_planners(epsilons=(0.01,))
    for n in sizes:
        for seed in seeds:
            instance = datasets.synthetic_instance(n, seed)
            belief = belief_mod.sample_truth(n, seed + 5000, sharpness)
            problem = build_problem(belief)

            local = []                          # type: List[Dict]
            for name, planner in planners:
                result = planner(problem, instance.matrix)
                cost = (expected_cost(result.order, belief, instance.matrix)
                        if result.order else float("inf"))
                local.append({
                    "instance": instance.name,
                    "n": n,
                    "seed": seed,
                    "planner": name,
                    "cost": cost,
                    "planning_s": result.planning_s,
                    "expansions": result.expansions,
                    "solved": result.solved,
                })
            best = min(row["cost"] for row in local)
            for row in local:
                row["ratio_to_best"] = (row["cost"] / best if best > 0.0
                                        else 1.0)
            rows.extend(local)
    return rows


def run_reliability(n=12, seeds=range(20), sharpness=0.7,
                    levels=belief_mod.RELIABILITY_LEVELS,
                    names=belief_mod.RELIABILITY_NAMES,
                    time_limit_s=PAPER_TIME_LIMIT_S):
    # type: (int, Sequence[int], float, Sequence[float], Sequence[str], float) -> List[Dict]
    """The experiment the paper does not run: vary how right the prior is.

    Each scenario fixes a truth, derives a belief at every reliability level
    plus the paper's two-peaked decoy, plans with every planner under that
    belief, and scores every route against the truth. So the flight distance
    reported is what the robot really expects to fly, not what the planner
    believed when it chose the route.

    Args:
        n: Vertex count. Held fixed so reliability is the only thing moving.
        seeds: Instances.
        sharpness: Dirichlet concentration for the truth. Below one gives a
            peaked truth, which is the interesting case -- a flat truth makes
            every ordering nearly equivalent and there is nothing to win.
        levels: Reliability values to sweep.
        names: Human names for those values.
        time_limit_s: Per-instance budget.

    Returns:
        One row per (seed, regime, planner).
    """
    rows = []                                   # type: List[Dict]
    planners = default_planners(epsilons=(0.01,))

    for seed in seeds:
        instance = datasets.synthetic_instance(n, seed)
        matrix = instance.matrix
        truth = belief_mod.sample_truth(n, seed + 5000, sharpness)

        regimes = [(names[i], levels[i],
                    belief_mod.belief_from_truth(truth, levels[i]))
                   for i in range(len(levels))]
        regimes.append(("decoy", float("nan"),
                        belief_mod.decoy_belief(truth, matrix, seed + 9000)))

        for regime_name, level, belief in regimes:
            problem = build_problem(belief)
            for planner_name, planner in planners:
                result = planner(problem, matrix)
                if not result.order:
                    continue
                outcome = metrics.evaluate(
                    planner=planner_name,
                    order=result.order,
                    belief=belief,
                    truth=truth,
                    matrix=matrix,
                    planning_s=result.planning_s,
                    expansions=result.expansions,
                    status=result.status,
                    guarantee=result.guarantee,
                    solved=result.solved,
                )
                row = outcome.as_row()
                row.update({
                    "instance": instance.name,
                    "n": n,
                    "seed": seed,
                    "regime": regime_name,
                    "reliability": level,
                })
                rows.append(row)
    return rows


def run_sharpness(n=12, seeds=range(20),
                  sharpnesses=(4.0, 2.0, 1.0, 0.5, 0.25, 0.1),
                  time_limit_s=PAPER_TIME_LIMIT_S):
    # type: (int, Sequence[int], Sequence[float], float) -> List[Dict]
    """How peaked the belief is -- the hidden variable behind every headline.

    Section VII-C reports ``Greedy`` at two to three times optimal and ``LKH``
    at 50-80% above it, as though those were properties of the algorithms. They
    are not. They are properties of the belief, and the paper never states
    which belief it used.

    The reason is structural. ``q`` is a product of ``(1 - p)`` terms, so when
    every ``p`` is small ``q`` stays near one, every edge is charged at nearly
    full weight, and the HPP-PT objective degenerates into plain path length --
    at which point a distance-only planner is *already solving the right
    problem* and LKH is near-optimal by construction. Only when the belief is
    concentrated does the ordering start to matter, and only then does RPT*
    have anything to win.

    So this sweeps the Dirichlet concentration from flat to spiked and reports
    what each baseline costs relative to the optimum at each. It is the
    experiment that says when RPT* is worth running at all.

    Args:
        n: Vertex count, held fixed.
        seeds: Instances per sharpness.
        sharpnesses: Dirichlet concentrations. Large is flat, small is peaked.
        time_limit_s: Per-instance budget.

    Returns:
        One row per (sharpness, seed, planner), with the ratio to RPT* and the
        largest single probability in the belief, which is the readable summary
        of how peaked it was.
    """
    rows = []                                   # type: List[Dict]
    planners = default_planners(epsilons=())
    for sharpness in sharpnesses:
        for seed in seeds:
            instance = datasets.synthetic_instance(n, seed)
            belief = belief_mod.sample_truth(n, seed + 5000, sharpness)
            problem = build_problem(belief)

            local = []                          # type: List[Dict]
            for name, planner in planners:
                result = planner(problem, instance.matrix)
                local.append({
                    "n": n,
                    "seed": seed,
                    "sharpness": sharpness,
                    "max_prob": max(belief),
                    "planner": name,
                    "cost": expected_cost(result.order, belief, instance.matrix)
                            if result.order else float("inf"),
                    "planning_s": result.planning_s,
                    "expansions": result.expansions,
                    "solved": result.solved,
                })
            best = min(row["cost"] for row in local)
            for row in local:
                row["ratio_to_best"] = (row["cost"] / best if best > 0.0
                                        else 1.0)
            rows.extend(local)
    return rows


def run_objective(sizes=(7, 8, 9), seeds=range(15),
                  sharpnesses=(4.0, 1.0, 0.5, 0.25)):
    # type: (Sequence[int], Sequence[int], Sequence[float]) -> List[Dict]
    """What Eq. 1 costs when there is exactly one target -- Remark 1's error.

    Remark 1 (p.3) says the single-target case just adds the constraint
    ``sum p(v) = 1``, and that "our approaches ... do not rely on those
    constraints and is applicable to all these problem variants". The second
    half of that sentence is false, and this measures by how much.

    Eq. 1 charges the ``i``-th edge at ``prod_{k<=i} (1 - p_k)``: the chance of
    independently missing at each of the first ``i`` places. With one target
    that certainly exists those events are mutually exclusive, and the correct
    weight is ``1 - sum_{k<=i} p_k``. The paper's weight is always the larger
    of the two, so Eq. 1 over-values the far end of the route and can prefer an
    ordering that is not the one that finds the object soonest.

    Every instance here is enumerated exhaustively, so both optima are exact
    and the gap between them is not a search artefact.

    Args:
        sizes: Vertex counts. Kept small -- this enumerates twice.
        seeds: Instances per cell.
        sharpnesses: Dirichlet concentrations for the belief.

    Returns:
        One row per instance, carrying the distance flown by the route Eq. 1
        prefers and by the route that is actually best, and the excess.
    """
    import itertools

    rows = []                                   # type: List[Dict]
    for sharpness in sharpnesses:
        for n in sizes:
            for seed in seeds:
                instance = datasets.synthetic_instance(n, seed)
                matrix = instance.matrix
                # Belief is exactly the truth: this gap is not about a bad
                # prior, it is about the objective being the wrong one even
                # when the prior is perfect.
                truth = belief_mod.sample_truth(n, seed + 5000, sharpness)
                others = [v for v in range(n) if v != 0]

                best_paper = min(
                    itertools.permutations(others),
                    key=lambda perm: expected_cost((0,) + perm, truth, matrix))
                best_true = min(
                    itertools.permutations(others),
                    key=lambda perm: metrics.expected_distance_under_truth(
                        (0,) + perm, truth, matrix))

                paper_distance = metrics.expected_distance_under_truth(
                    (0,) + best_paper, truth, matrix)
                true_distance = metrics.expected_distance_under_truth(
                    (0,) + best_true, truth, matrix)

                rows.append({
                    "instance": instance.name,
                    "n": n,
                    "seed": seed,
                    "sharpness": sharpness,
                    "max_prob": max(truth),
                    "paper_distance": paper_distance,
                    "true_distance": true_distance,
                    "excess_pct": ((paper_distance - true_distance)
                                   / true_distance * 100.0)
                                  if true_distance > 0.0 else 0.0,
                    "same_route": tuple(best_paper) == tuple(best_true),
                })
    return rows


def run_tsplib(instances=datasets.TSPLIB_INSTANCES, seeds=range(3),
               sharpness=1.0, time_limit_s=PAPER_TIME_LIMIT_S):
    # type: (Sequence[str], Sequence[int], float, float) -> List[Dict]
    """Sec. VII-A's five instances -- and the assumption they break.

    Each is run twice: on the matrix as published, and on its metric closure.
    The closure is what the problem statement assumes the input already is, so
    a difference between the two is a measure of how far the paper's own
    benchmark sits outside its own preconditions.

    **Both routes are then scored under the closure**, which is the only
    coherent cost model -- one in which a direct edge is never dearer than a
    detour. Comparing each route's cost on its own matrix would compare two
    different objectives and report a difference that is mostly just the
    closure having shorter edges. The number that means something is: *did
    planning on the broken matrix cost the robot anything?*

    Args:
        instances: Which TSPLIB instances to load.
        seeds: Belief seeds per instance -- the files carry no probabilities,
            so one has to be invented, and inventing several shows the answer
            is not an artefact of one.
        sharpness: Dirichlet concentration for the belief.
        time_limit_s: Per-instance budget.

    Returns:
        One row per (instance, seed, matrix variant). The ``as-published`` rows
        carry ``penalty_pct``: how much more the route chosen on the broken
        matrix costs, measured on the coherent one.
    """
    rows = []                                   # type: List[Dict]
    for name in instances:
        instance = datasets.tsplib_instance(name)
        violations, worst = datasets.triangle_violations(instance.matrix)
        closure = datasets.metric_closure_matrix(instance.matrix)
        closure_violations, _ = datasets.triangle_violations(closure)

        for seed in seeds:
            belief = belief_mod.sample_truth(instance.n, seed + 5000, sharpness)
            problem = build_problem(belief)

            solutions = {}
            for variant, matrix in (("as-published", instance.matrix),
                                    ("metric-closure", closure)):
                # The as-published matrix breaks the precondition, so the
                # check has to be waived to run at all -- which is exactly the
                # situation the paper is silently in.
                params = RptStarParams(
                    time_budget_s=time_limit_s,
                    require_triangle_inequality=(variant == "metric-closure"),
                )
                started = time.monotonic()
                solution = solve(problem, matrix, params)
                elapsed = time.monotonic() - started
                solutions[variant] = (solution, elapsed)

            reference = expected_cost(
                solutions["metric-closure"][0].order_indices, belief, closure)

            for variant, (solution, elapsed) in solutions.items():
                on_closure = expected_cost(solution.order_indices, belief,
                                           closure)
                rows.append({
                    "instance": name,
                    "n": instance.n,
                    "seed": seed,
                    "matrix": variant,
                    "triangle_violations": (violations
                                            if variant == "as-published"
                                            else closure_violations),
                    "worst_violation": worst if variant == "as-published" else 0.0,
                    "cost": solution.expected_cost,
                    "cost_on_closure": on_closure,
                    "penalty_pct": ((on_closure - reference) / reference * 100.0
                                    if reference > 0.0 else 0.0),
                    "status": solution.status,
                    "guarantee": solution.guarantee,
                    "planning_s": elapsed,
                    "expansions": solution.stats.expansions,
                    "order": list(solution.order_indices),
                })
    return rows



