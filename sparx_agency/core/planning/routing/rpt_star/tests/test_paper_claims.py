"""The paper's lemmas, restated as things that either hold or fail loudly.

Each test here corresponds to a numbered claim in arXiv:2601.12701. They are
worth writing out because the whole package is a translation of that paper, and
a translation is only trustworthy if the claims survive the trip.

* **Lemma 1** (p.4) -- the reformulation with the survival probability gives
  the same number as Definition 1. This is the substitution the entire method
  rests on; if it were wrong, the search would be optimising something else.
* **Lemma 3** (p.4) -- visiting more places means a smaller survival
  probability, which is why dominance never has to compare it.
* **Lemma 4** (p.5) -- the heuristic table. The paper's stated complexity is
  wrong and the test says so.
* **Lemma 5** (p.6) -- the heuristic never overestimates. Checked against the
  true cost-to-go computed exhaustively, which is the definition of admissible
  rather than a proxy for it.
"""
from __future__ import annotations

import itertools
import random

import pytest

from sparx_agency.core.planning.routing.rpt_star import (
    GammaTable,
    RouteProblem,
    RouteVertex,
    costs_from_points,
    expected_cost,
    expected_cost_literal,
)


def instance(rng, n, max_prob=0.6):
    """A random Euclidean problem."""
    points = [(rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0))
              for _ in range(n)]
    vertices = [RouteVertex(id=i, prob=rng.uniform(0.0, max_prob))
                for i in range(n)]
    return RouteProblem(vertices, start_id=0), costs_from_points(points)


# -- Lemma 1 --------------------------------------------------------------

def test_lemma_1_the_two_cost_formulas_agree():
    """Definition 1 as printed, and the survival-probability form, are equal.

    Over random routes rather than random *optimal* routes: the identity is
    claimed for every path, not only for good ones.
    """
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randint(2, 9)
        problem, matrix = instance(rng, n)
        order = [0] + rng.sample(range(1, n), n - 1)
        assert expected_cost(order, problem.probs, matrix) == pytest.approx(
            expected_cost_literal(order, problem.probs, matrix), abs=1e-12)


def test_lemma_1_holds_when_probabilities_are_zero():
    """With no belief anywhere, the expected cost is just the path length.

    The degenerate case the paper notes on p.1: HPP-PT reduces to the plain
    Hamiltonian path problem.
    """
    probs = [0.0, 0.0, 0.0, 0.0]
    matrix = costs_from_points([(0, 0), (1, 0), (2, 0), (3, 0)])
    order = (0, 1, 2, 3)
    assert expected_cost(order, probs, matrix) == pytest.approx(3.0)
    assert expected_cost_literal(order, probs, matrix) == pytest.approx(3.0)


def test_the_final_leg_is_flown_regardless_of_its_probability():
    """The last term of Eq. (1) carries no ``p``, and that is not a typo.

    If nothing was found at the first N-1 places, the robot goes to the last
    one whatever the chance it is there -- so the last leg's contribution must
    not depend on that probability at all.
    """
    matrix = costs_from_points([(0, 0), (1, 0), (2, 0)])
    order = (0, 1, 2)
    cheap = expected_cost(order, [0.0, 0.5, 0.01], matrix)
    rich = expected_cost(order, [0.0, 0.5, 0.99], matrix)
    assert cheap == pytest.approx(rich)


# -- Lemma 3 --------------------------------------------------------------

def test_lemma_3_more_visited_means_lower_survival():
    """A superset of visited places always has the smaller survival product.

    This is why Definition 3 compares only cost and the visited set: the
    survival probability comes along for free, and comparing it as well would
    be redundant at best and inconsistent at worst.
    """
    rng = random.Random(2)
    probs = [rng.uniform(0.0, 0.9) for _ in range(10)]
    for _ in range(300):
        subset = rng.sample(range(10), rng.randint(1, 6))
        extra = [v for v in range(10) if v not in subset]
        superset = subset + rng.sample(extra, rng.randint(1, len(extra)))
        survival_small = 1.0
        for v in subset:
            survival_small *= (1.0 - probs[v])
        survival_large = 1.0
        for v in superset:
            survival_large *= (1.0 - probs[v])
        assert survival_large <= survival_small + 1e-15


# -- Lemma 4 --------------------------------------------------------------

class _CountingRow(tuple):
    """A cost row that tallies how many entries are read out of it."""

    reads = [0]

    def __getitem__(self, index):
        _CountingRow.reads[0] += 1
        return tuple.__getitem__(self, index)


def _counting_matrix(matrix):
    """The same costs, instrumented to count reads during a table build."""
    return tuple(_CountingRow(row) for row in matrix)


def test_lemma_4_the_table_is_cubic_not_quadratic():
    """The paper says the table costs O(|V|^2). It costs O(|V|^3).

    The table holds |V| x |V| entries and each one minimises over the other
    |V| - 1 vertices; the proof on p.5 counts the entries and forgets the
    minimisation inside them.

    Counted rather than timed, so the result does not depend on the machine,
    and expressed as a growth ratio rather than an absolute, so it does not
    depend on constant factors either: doubling the vertex count must multiply
    the reads by about eight. Quadratic work would multiply them by four.
    """
    rng = random.Random(3)
    reads = []
    for n in (20, 40):
        problem, matrix = instance(rng, n)
        _CountingRow.reads[0] = 0
        GammaTable(problem.probs, _counting_matrix(matrix))
        reads.append(_CountingRow.reads[0])
    ratio = reads[1] / float(reads[0])
    assert ratio > 6.0, (
        "doubling |V| multiplied the reads by %.2f; quadratic would be ~4 and "
        "cubic ~8, and the paper claims quadratic" % ratio)
    # And the count is exactly what a cubic build predicts: for each of the
    # |V| - 1 table rows above the base case, for each of the |V| vertices,
    # one read per one of the |V| - 1 candidate successors.
    assert reads[0] == (20 - 1) * 20 * (20 - 1)


def test_gamma_row_zero_is_zero_everywhere():
    """No steps left, nothing left to fly."""
    rng = random.Random(4)
    problem, matrix = instance(rng, 6)
    gamma = GammaTable(problem.probs, matrix)
    for vertex in range(6):
        assert gamma.steps(vertex, 0) == 0.0


def test_gamma_has_exactly_n_rows_which_is_why_the_start_must_be_visited():
    """The table stops at ``|V| - 1`` steps, and the initial state needs that.

    With the start counted as visited the first heuristic call asks for row
    ``|V| - 1``, the last one there is. With the paper's literal ``A = {}`` it
    would ask for row ``|V|``, which does not exist -- which is the mechanical
    proof that Algorithm 1's line 1 is a typo.
    """
    rng = random.Random(5)
    problem, matrix = instance(rng, 5)
    gamma = GammaTable(problem.probs, matrix)
    assert gamma.steps(0, problem.n - 1) >= 0.0
    with pytest.raises(IndexError):
        gamma.steps(0, problem.n)


# -- Lemma 5 --------------------------------------------------------------

def test_lemma_5_the_heuristic_never_overestimates():
    """``h`` is never more than the true cheapest completion.

    The true value is computed exhaustively over every completion of every
    partial route, which is the definition of admissibility rather than a
    stand-in for it. Small instances only, because that enumeration is
    factorial.
    """
    rng = random.Random(6)
    for _ in range(25):
        n = rng.randint(3, 6)
        problem, matrix = instance(rng, n, max_prob=0.7)
        probs = problem.probs
        gamma = GammaTable(probs, matrix)
        for prefix_length in range(1, n):
            for prefix in itertools.permutations(range(n), prefix_length):
                survival = 1.0
                for vertex in prefix:
                    survival *= (1.0 - probs[vertex])
                estimate = gamma.estimate(prefix[-1], survival, len(prefix))
                spent = expected_cost(prefix, probs, matrix)
                remaining = [v for v in range(n) if v not in prefix]
                true_best = min(
                    expected_cost(prefix + tail, probs, matrix) - spent
                    for tail in itertools.permutations(remaining))
                assert estimate <= true_best + 1e-9, (
                    "heuristic overestimated: h=%r true=%r prefix=%r"
                    % (estimate, true_best, prefix))
