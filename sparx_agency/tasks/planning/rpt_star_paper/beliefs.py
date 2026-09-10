"""Where the target really is, and what the planner was told -- kept apart.

**This is the axis the paper does not have.** Section VII never states how
``p(v)`` was generated for its synthetic instances, and the TSPLIB files carry
no probabilities at all. Yet ``q`` is a product of ``(1 - p)`` terms, so the
belief shape drives both the runtime and the size of the win far harder than
``|V|`` does. Reporting a replication against vertex count alone would repeat
the paper's own blind spot.

So every scenario here carries two distributions, and never confuses them:

* **truth** -- where the target actually is. Never shown to any planner. Used
  only to score what a route really costs to fly.
* **belief** -- ``p(v)``, what the planner is handed. Derived from the truth
  through a reliability dial.

Scoring a planner against the same distribution it was optimising is circular:
RPT* provably minimises expected cost under its input, so it would win by
definition and the number would mean nothing. The gap between truth and belief
is the entire experiment.

The dial runs from ``+1`` to ``-1``:

===========  ==========================================================
reliability  what the planner is told
===========  ==========================================================
``+1.0``     the truth exactly
``+0.5``     the truth, half-diluted toward uniform
``0.0``      uniform -- "no clue where the target might be", which is the
             paper's own stated fallback (footnote 1, p.9)
``-0.5``     half-inverted: likely places look unlikely
``-1.0``     fully inverted, the adversarial case
===========  ==========================================================

Separately, :func:`decoy_belief` reproduces the paper's *misleading prior*
(Sec. VII-E, p.13): "a mixture of two Gaussians, with one peak matching the
target location and the other being far away from the true target location".
That is not a point on the dial -- it is confidently right and confidently
wrong at the same time, which is the failure mode a greedy planner cannot
survive and the one RPT* is claimed to handle.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import random
from typing import List, Sequence

#: The largest probability any vertex may be given. ``p(v) in [0, 1)`` is the
#: paper's domain (p.3) and the heuristic divides by ``1 - p``, so a belief that
#: collapses onto one vertex has to stop short of certainty.
MAX_PROB = 0.98

#: Reliability values swept by default, from perfect to adversarial.
RELIABILITY_LEVELS = (1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0)

#: Human names for those levels, in the same order.
RELIABILITY_NAMES = ("perfect", "good", "fair", "weak", "uninformative",
                     "half-inverted", "adversarial")


def sample_truth(n, seed, sharpness=1.0):
    # type: (int, int, float) -> List[float]
    """Draw where the target actually is, as a distribution over vertices.

    A symmetric Dirichlet, which is the standard one-parameter family over the
    simplex: ``sharpness`` large gives a flat, boring truth where every place
    is about as likely as every other, and ``sharpness`` small gives a peaked
    one where the target is probably in a particular place.

    Sums to one. Remark 1 (p.3) is explicit that this is the right normalisation
    when there is a single target known to be somewhere in the graph, which is
    the target-search scenario the whole paper is motivated by.

    At very small ``sharpness`` the Dirichlet collapses onto a single vertex and
    would return a probability of exactly one, which is outside the paper's
    ``[0, 1)`` domain and which the heuristic divides by. The result is
    therefore passed through the same clamp as every other belief here, so a
    sharpness sweep can be pushed as far as it likes without falling out of the
    problem definition.

    Args:
        n: The vertex count.
        seed: Seed, so a scenario is reproducible from its name.
        sharpness: The Dirichlet concentration. Must be positive.

    Returns:
        ``n`` probabilities summing to one, each strictly below
        :data:`MAX_PROB`.
    """
    rng = random.Random(seed)
    weights = [rng.gammavariate(sharpness, 1.0) for _ in range(n)]
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / n] * n
    return _normalise(weights)


def belief_from_truth(truth, reliability):
    # type: (Sequence[float], float) -> List[float]
    """Degrade a truth into the belief a planner is handed.

    For ``reliability >= 0`` the truth is blended toward uniform; for
    ``reliability < 0`` it is blended toward its own inversion, so that likely
    places come to look unlikely. Both branches meet at uniform when the dial
    is zero, so the sweep is continuous across it.

    The inversion is by *rank*, not by ``1 - p``. Subtracting from one leaves a
    near-uniform distribution near-uniform and would make the adversarial case
    indistinguishable from the uninformative one; reversing the ranking moves
    the mass that was on the most likely place onto the least likely, which is
    what "misled" has to mean for the comparison to be worth anything.

    Args:
        truth: The true distribution.
        reliability: In ``[-1, 1]``. See the module docstring.

    Returns:
        A belief summing to one, every entry strictly below :data:`MAX_PROB`.
    """
    n = len(truth)
    uniform = 1.0 / n
    weight = abs(float(reliability))
    if weight > 1.0:
        weight = 1.0

    if reliability >= 0.0:
        target = list(truth)
    else:
        target = _invert_by_rank(truth)

    blended = [weight * target[i] + (1.0 - weight) * uniform for i in range(n)]
    return _normalise(blended)


def decoy_belief(truth, matrix, seed, decoy_mass=0.55, true_mass=0.25):
    # type: (Sequence[float], Sequence[Sequence[float]], int, float, float) -> List[float]
    """The paper's misleading prior: right about the target, louder about a lie.

    Section VII-E (p.13) builds it as "a mixture of two Gaussians, with one peak
    matching the target location and the other being far away from the true
    target location", and Table III is the result -- greedy's mission time
    roughly triples while RPT*'s barely moves.

    The decoy is placed at the vertex *furthest* from the true peak, because a
    decoy next door costs nothing to check and would not test anything.

    Args:
        truth: The true distribution. Its argmax is the true target location.
        matrix: The cost matrix, used to find the furthest vertex.
        seed: Seed for the residual mass, so the result is reproducible.
        decoy_mass: Belief mass placed on the false peak. Larger than
            ``true_mass`` on purpose -- the prior is confidently wrong.
        true_mass: Belief mass left on the real location.

    Returns:
        A belief summing to one, every entry strictly below :data:`MAX_PROB`.
    """
    n = len(truth)
    true_peak = max(range(n), key=lambda v: truth[v])
    decoy = max(range(n), key=lambda v: matrix[true_peak][v])

    rng = random.Random(seed)
    residual = max(0.0, 1.0 - decoy_mass - true_mass)
    scatter = [rng.random() for _ in range(n)]
    scatter[true_peak] = 0.0
    scatter[decoy] = 0.0
    scatter_total = sum(scatter)

    belief = [0.0] * n
    for v in range(n):
        if scatter_total > 0.0:
            belief[v] = residual * scatter[v] / scatter_total
    belief[true_peak] += true_mass
    belief[decoy] += decoy_mass
    return _normalise(belief)


def _invert_by_rank(truth):
    # type: (Sequence[float]) -> List[float]
    """Reassign the probability values in reverse order of likelihood.

    The most likely vertex receives the mass of the least likely and vice
    versa, so the shape of the distribution is preserved exactly while the
    ordering it implies is reversed.
    """
    n = len(truth)
    order = sorted(range(n), key=lambda v: (truth[v], v))
    values = sorted(truth)
    inverted = [0.0] * n
    for rank, vertex in enumerate(order):
        inverted[vertex] = values[n - 1 - rank]
    return inverted


def _normalise(values):
    # type: (Sequence[float]) -> List[float]
    """Scale to sum one, then clamp anything at or above :data:`MAX_PROB`.

    Clamping can only bite when the belief has collapsed onto a single vertex,
    and it redistributes what it removes across the others rather than leaving
    the total short -- a belief that does not sum to one is legal for the
    solver (Remark 1) but would quietly change what the truth-scored metrics
    mean.
    """
    total = sum(values)
    n = len(values)
    if total <= 0.0:
        return [1.0 / n] * n
    scaled = [v / total for v in values]

    excess = 0.0
    for i, value in enumerate(scaled):
        if value >= MAX_PROB:
            excess += value - MAX_PROB
            scaled[i] = MAX_PROB
    if excess > 0.0:
        room = [i for i, value in enumerate(scaled) if value < MAX_PROB]
        if room:
            share = excess / len(room)
            for i in room:
                scaled[i] = min(MAX_PROB, scaled[i] + share)
    return scaled

