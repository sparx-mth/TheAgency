"""The places worth searching, how likely each is, and where the robot starts.

This module owns the *only* translation between the ids a caller thinks in --
``("room", 7)``, ``("frontier", 3)`` -- and the dense integer indices the search
runs on. That bijection is touched exactly twice in a solve: once here at
ingest, once when the solution is decorated. Everywhere in between is integers
and bitmasks, because the dominance test is the hot loop and it should compare
machine words, not tuples.

**The start is a vertex like any other.** The paper fixes ``v_1 = v_s`` and
leaves the terminal free (Def. 2, p.3), so the depot is simply the first
element of the permutation and still has to be "visited". When the robot's
current position is not one of the candidate places -- the usual case, since a
drone is rarely standing exactly on a room centroid --
:meth:`RouteProblem.with_external_start` adds it as a vertex with probability
zero. Zero is fully inside the contract: it leaves the survival probability
unchanged, contributes no term to the expected cost, and is never divided by
(every division in the heuristic is by ``1 - p``, not by ``p``). Visiting it
first therefore costs nothing, which is exactly right -- the robot is already
there.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.routing.rpt_star.errors import (
    InvalidProblemError,
)

#: Anything hashable and stable for the life of one solve. A tuple that names
#: its own kind -- ``("room", 7)`` -- is the shape that survives merging two id
#: spaces into one vertex set, which is what a rooms-plus-frontiers search does.
VertexId = Hashable

#: The largest probability accepted. The paper defines ``p(v) in [0, 1)``
#: (p.3) and the heuristic divides by ``1 - p(v)`` (Eq. 6, p.5), so 1.0 is not
#: a boundary case to handle -- it is outside the domain.
MAX_PROB = 1.0


@dataclass(frozen=True)
class RouteVertex:
    """One place worth searching, and how likely the target is to be there.

    Attributes:
        id: The caller's own identifier. Hashable, and unique within a problem.
        prob: ``p(v)``, the probability that the robot finds the target here
            and therefore stops here. In ``[0.0, 1.0)`` -- see :data:`MAX_PROB`.
            Not required to sum to one across the problem; see
            :func:`~sparx_agency.core.planning.routing.rpt_star.validation.validate`.
        label: A human name, carried into the solution untouched so a log can
            say "kitchen" rather than "v4".
        payload: Anything the caller wants handed back with the answer -- a
            centroid, an entry pose, a room record. The solver never reads it.
    """

    id: VertexId
    prob: float
    label: str = "?"
    payload: Any = None


class RouteProblem(object):
    """A vertex set with a fixed start, and the id-to-index bijection for it.

    Args:
        vertices: The places, including the start. Order is preserved and
            becomes the index order, so a caller who wants to compare against
            the paper's own indexing can control it.
        start_id: Which of them the robot begins at.

    Raises:
        InvalidProblemError: On an empty vertex set, a duplicate id, an
            unknown ``start_id``, or a probability outside ``[0, 1)``.
    """

    def __init__(self, vertices, start_id):
        # type: (Sequence[RouteVertex], VertexId) -> None
        vertices = list(vertices)
        if not vertices:
            raise InvalidProblemError(
                "a routing problem needs at least one vertex, got none")
        index_of = {}       # type: Dict[VertexId, int]
        for index, vertex in enumerate(vertices):
            if vertex.id in index_of:
                raise InvalidProblemError(
                    "duplicate vertex id %r at positions %d and %d -- ids must "
                    "be unique, and two places at the same spot are still two "
                    "places" % (vertex.id, index_of[vertex.id], index))
            probability = float(vertex.prob)
            if not (0.0 <= probability < MAX_PROB):
                raise InvalidProblemError(
                    "vertex %r has prob=%r, outside the half-open range "
                    "[0.0, 1.0) the paper defines (p.3). 1.0 is excluded "
                    "because the heuristic divides by (1 - p)."
                    % (vertex.id, vertex.prob))
            index_of[vertex.id] = index
        if start_id not in index_of:
            raise InvalidProblemError(
                "start_id %r is not one of the %d vertices. Use "
                "RouteProblem.with_external_start() when the robot's position "
                "is not itself a candidate place." % (start_id, len(vertices)))
        self._vertices = tuple(vertices)
        self._index_of = index_of
        self._start = index_of[start_id]
        self._probs = tuple(float(v.prob) for v in vertices)

    @classmethod
    def with_external_start(cls, candidates, start_id="__start__",
                            label="start", payload=None):
        # type: (Sequence[RouteVertex], VertexId, str, Any) -> RouteProblem
        """Build a problem whose start is the robot, not a candidate place.

        The synthetic start is prepended with ``prob=0.0``, so it consumes no
        probability mass, adds no term to the expected cost, and is visited
        first for free. Its cost row and column must still be supplied -- the
        distance from the robot to each candidate is exactly what makes the
        ordering depend on where the robot is standing.

        Args:
            candidates: The real places, none of which may already use
                ``start_id``.
            start_id: The id to give the robot's position.
            label: Its human name.
            payload: Anything to carry through, e.g. the pose itself.

        Returns:
            A problem of ``len(candidates) + 1`` vertices, start at index 0.

        Raises:
            InvalidProblemError: If ``start_id`` collides with a candidate.
        """
        start = RouteVertex(id=start_id, prob=0.0, label=label, payload=payload)
        return cls([start] + list(candidates), start_id)

    # -- the vertex set ---------------------------------------------------

    @property
    def n(self):
        # type: () -> int
        """``|V|``, the start included."""
        return len(self._vertices)

    @property
    def start(self):
        # type: () -> int
        """The start vertex, as a dense index."""
        return self._start

    @property
    def vertices(self):
        # type: () -> Tuple[RouteVertex, ...]
        """The vertices in index order."""
        return self._vertices

    @property
    def probs(self):
        # type: () -> Tuple[float, ...]
        """``p(v)`` in index order -- what the search actually reads."""
        return self._probs

    def index(self, vertex_id):
        # type: (VertexId) -> int
        """The dense index of a caller id.

        Raises:
            InvalidProblemError: If the id is not in this problem.
        """
        try:
            return self._index_of[vertex_id]
        except KeyError:
            raise InvalidProblemError(
                "unknown vertex id %r" % (vertex_id,))

    def vertex(self, index):
        # type: (int) -> RouteVertex
        """The vertex at a dense index."""
        return self._vertices[index]

    def id_of(self, index):
        # type: (int) -> VertexId
        """The caller id at a dense index."""
        return self._vertices[index].id

    def ids_of(self, indices):
        # type: (Sequence[int]) -> Tuple[VertexId, ...]
        """Caller ids for a whole route, in order."""
        return tuple(self._vertices[i].id for i in indices)

    def __len__(self):
        # type: () -> int
        return len(self._vertices)

    def __repr__(self):
        # type: () -> str
        return ("RouteProblem(n=%d, start=%r, sum_p=%.3f)"
                % (self.n, self.id_of(self._start), sum(self._probs)))
