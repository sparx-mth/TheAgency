"""The two datasets the paper runs on, built to its own stated recipe.

Section VII (p.11) names both, and one of them it specifies completely while
the other it specifies only by name:

* **Synthetic.** "graphs with varying number of vertices that are randomly
  sampled from a 2D plane. The sampling area is set to a 500x500 region for
  small graphs (|V| <= 40) and expanded to 5000x5000 for larger instances
  (|V| > 40). In all cases, the distance between any two vertices must exceed
  5." That is a complete recipe and :func:`synthetic_instance` follows it
  exactly, including the rejection sampling implied by the minimum separation.

* **TSPLIB.** Five named instances -- ``gr17``, ``gr21``, ``gr24``, ``fri26``,
  ``bays29`` (Sec. VII-A-1, p.11; Table I writes ``bays29`` where the body text
  writes ``bayes29``). The files themselves are vendored under ``data/``.

**What the paper does not specify, and it matters more than anything it does:**
how ``p(v)`` is chosen. Section VII never states the distribution used to
generate vertex probabilities for the synthetic instances, and the TSPLIB files
carry no probabilities at all -- they are pure distance matrices. Since the
survival probability ``q`` is a product of ``(1 - p)`` terms, the belief shape
governs the runtime far more strongly than ``|V|`` does, so a replication that
guessed here would be reporting its own guess. Belief generation is therefore
kept out of this module entirely and made an explicit, swept axis in
:mod:`~sparx_agency.tasks.planning.rpt_star_paper.beliefs`.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
import pathlib
import random
from typing import Dict, List, Optional, Sequence, Tuple

#: Where the vendored TSPLIB files live.
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

#: The five TSPLIB instances of Sec. VII-A-1, in the paper's own order.
TSPLIB_INSTANCES = ("gr17", "gr21", "gr24", "fri26", "bays29")

#: Sampling half-width for |V| <= 40 (Sec. VII, p.11).
SMALL_EXTENT = 500.0

#: Sampling half-width for |V| > 40 (Sec. VII, p.11).
LARGE_EXTENT = 5000.0

#: Above this vertex count the paper switches to the larger sampling area.
SMALL_GRAPH_MAX = 40

#: "the distance between any two vertices must exceed 5" (Sec. VII, p.11).
MIN_SEPARATION = 5.0


class Instance(object):
    """One HPP-PT instance without its belief: geometry and distances only.

    Attributes:
        name: A stable identifier, used as the row key in results.
        n: The vertex count.
        matrix: The full cost matrix, ``matrix[i][j]`` being ``c(v_i, v_j)``.
        points: The 2D coordinates, or ``None`` for a TSPLIB instance given as
            an explicit matrix with no usable embedding.
        source: ``"synthetic"`` or ``"tsplib"``.
    """

    __slots__ = ("name", "n", "matrix", "points", "source")

    def __init__(self, name, n, matrix, points=None, source="synthetic"):
        # type: (str, int, List[List[float]], Optional[List[Tuple[float, float]]], str) -> None
        self.name = name
        self.n = n
        self.matrix = matrix
        self.points = points
        self.source = source

    def __repr__(self):
        # type: () -> str
        return "Instance(%r, n=%d, source=%r)" % (self.name, self.n, self.source)


def synthetic_instance(n, seed):
    # type: (int, int) -> Instance
    """Sample one synthetic instance exactly as Sec. VII describes it.

    Points are drawn uniformly from a square whose size depends on ``n``, and
    resampled until every pair is more than :data:`MIN_SEPARATION` apart. Costs
    are Euclidean, which makes them metric by construction -- so RPT*'s
    dominance pruning is sound here, which is not true of the TSPLIB half of
    the paper's own benchmark. See :func:`triangle_violations`.

    Args:
        n: The vertex count.
        seed: Seed for the generator, so an instance is reproducible from its
            name alone.

    Returns:
        The instance, with ``points`` populated.

    Raises:
        ValueError: If ``n`` points cannot be placed with the required
            separation after a generous number of attempts, which means the
            requested count does not fit in the paper's own sampling area.
    """
    extent = SMALL_EXTENT if n <= SMALL_GRAPH_MAX else LARGE_EXTENT
    rng = random.Random(seed)
    points = []                                 # type: List[Tuple[float, float]]
    attempts = 0
    limit = 20000 * max(1, n)
    while len(points) < n:
        attempts += 1
        if attempts > limit:
            raise ValueError(
                "could not place %d points more than %.1f apart in a %.0fx%.0f "
                "square after %d attempts; the paper's sampling area does not "
                "fit this many vertices" % (n, MIN_SEPARATION, extent, extent,
                                            limit))
        candidate = (rng.uniform(0.0, extent), rng.uniform(0.0, extent))
        for existing in points:
            if math.hypot(candidate[0] - existing[0],
                          candidate[1] - existing[1]) <= MIN_SEPARATION:
                break
        else:
            points.append(candidate)

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = math.hypot(points[i][0] - points[j][0],
                                          points[i][1] - points[j][1])
    return Instance("synthetic-n%d-s%d" % (n, seed), n, matrix, points,
                    "synthetic")


def tsplib_instance(name):
    # type: (str) -> Instance
    """Load one of the paper's five TSPLIB instances from ``data/``.

    Args:
        name: One of :data:`TSPLIB_INSTANCES`.

    Returns:
        The instance. ``points`` is ``None``: these are explicit distance
        matrices, and the ``DISPLAY_DATA_SECTION`` some of them carry is a
        drawing aid, not an embedding that reproduces the distances.

    Raises:
        FileNotFoundError: If the vendored file is missing.
        ValueError: If the file uses an edge-weight format not handled here.
    """
    path = DATA_DIR / ("%s.tsp" % name)
    with path.open() as handle:
        return _parse_tsplib(name, handle.read())


def _parse_tsplib(name, text):
    # type: (str, str) -> Instance
    """Parse the ``EXPLICIT`` subset of TSPLIB that the paper's five need.

    Only ``FULL_MATRIX`` and ``LOWER_DIAG_ROW`` appear among them, so only
    those are supported; anything else raises rather than guessing.
    """
    spec = {}                                   # type: Dict[str, str]
    weights = []                                # type: List[float]
    section = None                              # type: Optional[str]
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        if line.endswith("SECTION"):
            section = line
            continue
        if section is None and ":" in line:
            key, value = line.split(":", 1)
            spec[key.strip()] = value.strip()
            continue
        if section == "EDGE_WEIGHT_SECTION":
            weights.extend(float(token) for token in line.split())

    n = int(spec["DIMENSION"])
    fmt = spec.get("EDGE_WEIGHT_FORMAT", "")
    matrix = [[0.0] * n for _ in range(n)]
    if fmt == "FULL_MATRIX":
        for i in range(n):
            for j in range(n):
                matrix[i][j] = weights[i * n + j]
    elif fmt == "LOWER_DIAG_ROW":
        cursor = 0
        for i in range(n):
            for j in range(i + 1):
                matrix[i][j] = matrix[j][i] = weights[cursor]
                cursor += 1
    else:
        raise ValueError(
            "%s uses EDGE_WEIGHT_FORMAT %r, which this loader does not handle. "
            "The paper's five instances are all FULL_MATRIX or LOWER_DIAG_ROW."
            % (name, fmt))
    return Instance(name, n, matrix, None, "tsplib")


def triangle_violations(matrix):
    # type: (Sequence[Sequence[float]]) -> Tuple[int, float]
    """Count how badly a cost matrix breaks the triangle inequality.

    Section III states as part of the problem definition that "The edge costs
    satisfy the triangle inequality", and Lemma 6 -- the proof that dominance
    pruning never discards an optimal completion -- consumes that assumption
    directly, by shortcutting a repeated vertex and needing
    ``c(a,b) + c(b,c) >= c(a,c)`` for the shortcut not to cost more.

    So this is not a stylistic check. On a matrix that violates it, RPT* still
    returns a route and still reports a cost computed correctly for that route,
    but Theorem 2 no longer applies and the route may not be optimal, with
    nothing in the output to say so.

    Args:
        matrix: The cost matrix.

    Returns:
        The number of ordered triples ``(i, k, j)`` for which going direct
        beats the detour, and the largest amount by which it does.
    """
    n = len(matrix)
    count = 0
    worst = 0.0
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            if i == j:
                continue
            direct = row[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                gap = direct - (row[k] + matrix[k][j])
                if gap > 1e-9:
                    count += 1
                    if gap > worst:
                        worst = gap
    return count, worst


def metric_closure_matrix(matrix):
    # type: (Sequence[Sequence[float]]) -> List[List[float]]
    """Repair a matrix into its shortest-path closure (Floyd-Warshall).

    The result is metric by construction, so the pruning is sound on it. This
    is the honest way to run RPT* on the paper's TSPLIB instances, and it
    changes the answer -- which is the point of measuring it rather than
    assuming it does not.

    Args:
        matrix: The cost matrix.

    Returns:
        A new matrix in which ``m[i][j]`` is the cheapest route from ``i`` to
        ``j``, direct or not.
    """
    n = len(matrix)
    out = [list(row) for row in matrix]
    for k in range(n):
        row_k = out[k]
        for i in range(n):
            row_i = out[i]
            through = row_i[k]
            for j in range(n):
                candidate = through + row_k[j]
                if candidate < row_i[j]:
                    row_i[j] = candidate
    for i in range(n):
        out[i][i] = 0.0
    return out

