"""The room order for the object search, solved by RPT* instead of drawn.

:mod:`~sparx_agency.core.planning.exploration.object_search_supervisor` asks a
``solver(candidates, instance) -> Sequence[int]`` for the rooms to visit and
then walks that order. Its built-in stub draws ONE room weighted by
probability and never looks at what a room costs to reach. This module is the
real answer: it turns the supervisor's eligible rooms plus the arc weights in
an :class:`~sparx_agency.core.planning.exploration.room_costs.HppPtInstance`
into the Hamiltonian-path-with-probabilistic-terminals problem that
:mod:`~sparx_agency.core.planning.routing.rpt_star` solves, and hands back the
whole tour.

The difference is the entire point of the search: "most likely" and "closest"
are both wrong, and the ordering that minimises the *expected* time to find the
target is neither of them.

**Why this module and not the solver package.** ``rpt_star`` is standard
library only and knows nothing of rooms, metres or numpy; ``room_costs`` is
numpy and scipy and knows nothing of routing. Neither may grow a dependency on
the other, so the translation lives here, on the exploration side, next to both
of the exploration types it speaks. It is itself **standard library only** --
the instance's arrays are read through ``tolist()`` and never with a numpy
call -- so it stays importable anywhere ``object_search_supervisor`` is, and a
test can drive it with plain lists.

Six things the translation has to get right, every one of which fails quietly:

1. **The depot sits at the opposite end.** ``HppPtInstance`` APPENDS the
   aircraft's own vertex last; ``RouteProblem.with_external_start`` PREPENDS a
   synthetic start at index 0. The rows and columns are reordered here so the
   two agree. Assuming they line up plans the tour from whichever room happens
   to be last in the index space.
2. **The candidates are a subset of the instance.** The instance covers every
   room the map can reach; the supervisor has already dropped rooms below
   ``min_prob``, rooms on visit cooldown and rooms deferred after repeated
   failure. Only the intersection becomes a vertex, so the solver cannot plan
   over a room the policy already refused. A candidate the instance never
   mapped -- withheld as unreachable -- is reported in
   :attr:`SolveRecord.skipped` rather than priced with a made-up weight.
3. **The probabilities are the instance's, and they are not normalised.**
   ``RoomCandidate.prob_renorm`` divides by the surviving mass, which asserts
   the target is certainly in a room we have already mapped and structurally
   suppresses flying toward unmapped space -- and which is exactly 1.0 when a
   single candidate survives, a value RPT* rejects because its heuristic
   divides by ``1 - p``. So ``p`` comes from the instance, un-normalised, as
   both halves intend.
4. **The order is returned whole, not just its head.** The supervisor commits
   to an order and re-asks only when it is spent, when its head stops being
   eligible, or when the room set changes. Returning one room would re-solve
   against a noisy ranking every time a room finishes and flip the aircraft
   between two near-equal rooms without ever searching either -- the failure
   the supervisor's own docstring exists to prevent.
5. **Failure must not ground the aircraft.** The supervisor marks the room set
   as solved-for *before* it inspects what came back, so a solver that returns
   nothing parks the aircraft until the ranking itself changes. Every failure
   here therefore falls back to the weighted draw and says why in
   :attr:`SolveRecord.reason`; nothing propagates out into the ROS timer.
6. **The vertex set has to be capped, and the cap is the difference between
   RPT\\* and nearest-first.** The search is exponential, and measured on the
   committed hospital BEV (29 watershed rooms, the node's own defaults --
   0.30 m/s cruise, no folded budget, a 2 s budget, over four belief shapes
   and six aircraft positions) the cliff is sharp and close: 13 rooms solve
   optimally in at most 177 ms and 14 in 750 ms, but at 15 every shape except
   a sharply peaked one runs out of budget. Past that the search stops
   returning its own answer at all and falls through to
   :func:`~sparx_agency.core.planning.routing.rpt_star.baselines.nearest_neighbour_order`,
   which never reads the probabilities: at 20 rooms the returned order was
   byte-identical across peaked, spread and flat beliefs. Handing the solver
   all 29 rooms therefore does not give a slightly worse tour, it silently
   gives up the entire contribution. So :data:`DEFAULT_MAX_ROOMS` rooms are
   chosen first -- see :func:`_cap`, which is careful not to become the
   greedy-by-probability baseline while doing it -- and the search is exact
   over those. This costs less than it sounds: a room takes a mapping budget
   to search, so a mission never flies more than a handful before the
   eligible set changes and the order is asked for again.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.exploration.object_search_supervisor import (
    weighted_order)
from sparx_agency.core.planning.routing.rpt_star import (
    RouteProblem, RouteVertex, RptStarParams, dense_costs, solve)

#: Where the order came from.
RPT_STAR = "rpt_star"
FALLBACK = "fallback"

NOT_ATTEMPTED = "not_attempted"
"""``status`` on a record RPT* never got to produce.

Distinct from the package's own ``no_route``, which is a thing the search
says after looking. Reporting a fallback as ``no_route`` would claim the
solver ruled the rooms out when in fact it was never asked.
"""

DEFAULT_TIME_BUDGET_S = 2.0
"""Seconds one solve may take before it settles for its constructive route.

Lower than the package default of 5.0 because the caller here is a 1 Hz ROS
timer on a single-threaded executor and a solve blocks it. RPT* does not
return nothing when the clock beats it -- it returns a nearest-neighbour tour
stamped ``route_source='fallback'`` and ``guarantee='none'`` -- so the cost of
this being too small is a worse order, never a stalled loop. Watch
:attr:`SolveRecord.status` to find out which happened.
"""

DEFAULT_MAX_ROOMS = 12
"""How many rooms one solve may range over. Measured, not guessed.

Worst case over four belief shapes and six aircraft positions on the committed
hospital BEV, at the node's own defaults: 96 ms at twelve rooms, 177 ms at
thirteen, 750 ms at fourteen, and at fifteen everything but a sharply peaked
belief hits the 2 s budget and loses its guarantee. Twelve keeps a factor of
twenty in hand, which is what a solve sharing a 1 Hz tick with a planner and a
mapper needs -- the numbers above were taken on an idle machine. Raise it only
while a flight's ``guarantee`` stays ``optimal``.
"""

NEAREST_RESERVED = 1
"""Slots the cap holds back for the closest room, whatever the belief says.

Without this the cap is greedy-by-probability -- the very baseline RPT* exists
to beat -- applied as a prefilter, so the room the aircraft is standing next to
can be dropped before the solver ever gets to weigh its cost. Measured in the
closed loop on the hospital fixture (24 flights, a noisy zipf belief, a random
start, four rooms deep, re-solving after each as the loop really does),
reserving one slot cuts the expected time-to-find to 0.88 of reserving none
(median 145.9 s -> 106.3 s). Reserving two or three is worse than one (0.90,
0.91): past the first, slots spent on proximity are slots taken from belief.
"""

P_CEILING = 1.0 - 1e-6
"""Largest probability handed to RPT*, whose heuristic divides by ``1 - p``.

A deliberate second copy of ``room_costs.P_CLAMP_DEFAULT`` rather than an
import of it: that module needs numpy and scipy, and this one must not. The
instance clamps to the same value already, so this only ever bites an instance
somebody built by hand.
"""


@dataclass(frozen=True)
class SolveRecord:
    """What the last call decided, and how much of it is worth believing.

    Nothing in the flight loop reads this -- it exists so an operator watching
    a heartbeat, or a campaign scoring a run afterwards, can tell an optimal
    tour from a timed-out guess. An optimality claim nobody can audit is not a
    claim.

    Attributes:
        source: :data:`RPT_STAR` or :data:`FALLBACK`.
        reason: Why the fallback was taken, empty when it was not.
        order: The room ids returned, in visit order.
        rooms: How many rooms the solve ranged over, the aircraft excluded.
        skipped: Candidate room ids the instance had no vertex for.
        capped: Candidate room ids dropped to keep the search inside its
            budget -- the least likely, and among equals the furthest away.
        depot_pid: The room id of the vertex the tour was planned from, or
            ``-1`` when that vertex is the aircraft's own pose. A room id
            here means ``build_instance`` could not place the aircraft and
            fell back to a room, so the origin is an assumption.
        expected_cost: Expected cost of the returned order, in
            :attr:`units` -- the quantity RPT* minimises, not the tour length.
        lower_bound: No ordering of these rooms can cost less than this.
        bound_ratio: ``expected_cost / lower_bound``.
        status: ``solved`` / ``budget_exceeded`` / ``no_route``.
        guarantee: ``optimal`` / ``bounded`` / ``none``. Separate from
            ``status`` on purpose: a search can finish cleanly and still owe
            nothing.
        route_source: ``search`` or ``fallback`` -- RPT*'s own word for
            whether it searched the order out or constructed it after the
            budget expired.
        expansions: States expanded, the honest measure of search effort.
        solve_ms: Wall time this call took, the translation included.
        units: ``seconds`` or ``metres``, from the instance.
        warnings: RPT*'s own remarks on the problem it was given.
    """

    source: str = FALLBACK
    reason: str = ""
    order: Tuple[int, ...] = ()
    rooms: int = 0
    skipped: Tuple[int, ...] = ()
    capped: Tuple[int, ...] = ()
    depot_pid: int = -1
    expected_cost: float = float("inf")
    lower_bound: float = 0.0
    bound_ratio: float = float("inf")
    status: str = NOT_ATTEMPTED
    guarantee: str = "none"
    route_source: str = ""
    expansions: int = 0
    solve_ms: float = 0.0
    units: str = "?"
    warnings: Tuple[str, ...] = ()

    def summary(self):
        # type: () -> str
        """One line for a heartbeat."""
        if self.source != RPT_STAR:
            return "fallback draw (%s)" % (self.reason or "no reason given",)
        return ("rpt* %d room(s)%s -> %s, E=%.1f%s bound=%.1f (%s/%s, "
                "%d exp, %.0f ms)"
                % (self.rooms,
                   " of %d" % (self.rooms + len(self.capped)) if self.capped
                   else "",
                   list(self.order), self.expected_cost,
                   "s" if self.units == "seconds" else "m", self.lower_bound,
                   self.status, self.guarantee, self.expansions,
                   self.solve_ms))


class RptStarRoomSolver:
    """A supervisor solver that orders rooms by expected time-to-find.

    Call it as the supervisor does -- ``solver(candidates, instance)``, both
    positional -- and it returns the room ids to visit, in order. It is a
    class rather than a closure only so :attr:`last` can be read afterwards.

    Args:
        params: Tuning for the search. The default runs the exact search of
            the paper under :data:`DEFAULT_TIME_BUDGET_S`, so an order that
            comes back ``status='solved'`` is provably the cheapest there is.
        rng: The generator the fallback draw uses. Injected so a test, and a
            seeded flight, are reproducible even along the failure path.
        fallback: ``fallback(candidates, instance) -> Sequence[int]``, used
            whenever RPT* cannot answer. Defaults to the supervisor's own
            weighted draw, which is what the loop did before this existed.
        max_rooms: The most rooms one solve may range over; see
            :data:`DEFAULT_MAX_ROOMS` for why there has to be a limit and why
            this is where it sits. Zero, None or a negative removes the cap,
            which on a fully mapped building means every order is
            nearest-first.
    """

    def __init__(self, params=None, rng=None, fallback=None,
                 max_rooms=DEFAULT_MAX_ROOMS):
        # type: (Optional[RptStarParams], Optional[random.Random], Optional[Callable], Optional[int]) -> None
        self.params = (params if params is not None
                       else RptStarParams(time_budget_s=DEFAULT_TIME_BUDGET_S))
        self.rng = rng
        self.max_rooms = max_rooms
        self._fallback = fallback
        self._last = SolveRecord(reason="never called")
        self.calls = 0

    @property
    def last(self):
        # type: () -> SolveRecord
        """What the most recent call decided."""
        return self._last

    def __call__(self, candidates, instance=None):
        # type: (Sequence[Any], Any) -> List[int]
        """The order to visit these rooms in.

        Args:
            candidates: The supervisor's eligible rooms. Only ``room_id`` and
                ``label`` are read -- the probability comes from the instance,
                un-normalised, for the reason in this module's docstring.
            instance: The ``HppPtInstance`` the arc weights live in. ``None``
                for the first minute of a flight, before the mapper has
                produced a room -- not an error, and not a reason to stop.

        Returns:
            Room ids in visit order. Never empty while any candidate exists,
            because an empty order parks the aircraft until the ranking
            itself changes.
        """
        self.calls += 1
        started = time.monotonic()
        if not candidates:
            self._last = SolveRecord(reason="no candidates")
            return []
        try:
            record = self._solve(candidates, instance, started)
        except Exception as failure:            # noqa: BLE001 -- see below
            # Deliberately every exception, not just RoutingError: this is
            # called from a ROS timer with no guard of its own, so anything
            # that escapes here stops the whole loop. The reason is recorded
            # rather than swallowed, and the next call tries RPT* again.
            record = None
            reason = "%s: %s" % (type(failure).__name__, failure)
        else:
            reason = ""
        if record is not None:
            self._last = record
            return list(record.order)
        return self._draw(candidates, instance, reason, started)

    # -- the translation ---------------------------------------------------
    def _solve(self, candidates, instance, started):
        # type: (Sequence[Any], Any, float) -> Optional[SolveRecord]
        """Build the routing problem, solve it, and dress the answer.

        Returns:
            The record, or None when there is nothing for RPT* to decide and
            the draw should have it instead.
        """
        if instance is None:
            return None
        kept, skipped = _intersect(candidates, instance)
        if not kept:
            return None

        rows = _as_rows(instance.C)
        probs = _as_list(instance.p)
        depot = int(instance.depot)
        kept, capped = _cap(kept, rows[depot], probs, self.max_rooms)
        # The depot leads, because with_external_start prepends its synthetic
        # start at index 0 while the instance appended the aircraft last.
        keep = [depot] + [index for _, index in kept]
        matrix = dense_costs([[rows[a][b] for b in keep] for a in keep])

        rooms = [RouteVertex(id=int(candidate.room_id),
                             prob=_clamped(probs[index]),
                             label=str(candidate.label),
                             payload=candidate)
                 for candidate, index in kept]
        problem = RouteProblem.with_external_start(rooms, label="aircraft")

        solution = solve(problem, matrix, self.params)
        # order[0] is the aircraft's synthetic start, which is not a room.
        order = tuple(int(room_id) for room_id in solution.order[1:])
        if not order:
            return None
        return SolveRecord(
            source=RPT_STAR,
            order=order,
            rooms=len(rooms),
            skipped=skipped,
            capped=capped,
            depot_pid=int(instance.index_to_pid[depot]),
            expected_cost=float(solution.expected_cost),
            lower_bound=float(solution.lower_bound),
            bound_ratio=float(solution.bound_ratio),
            status=str(solution.status),
            guarantee=str(solution.guarantee),
            route_source=str(solution.route_source),
            expansions=int(solution.stats.expansions),
            solve_ms=(time.monotonic() - started) * 1e3,
            units=str(getattr(instance, "units", "?")),
            warnings=tuple(solution.warnings))

    def _draw(self, candidates, instance, reason, started):
        # type: (Sequence[Any], Any, str, float) -> List[int]
        """Fall back to the weighted single draw, and say why."""
        if self._fallback is not None:
            order = list(self._fallback(candidates, instance) or [])
        else:
            order = weighted_order(candidates, instance, self.rng)
        self._last = SolveRecord(
            source=FALLBACK,
            reason=reason or "no usable instance",
            order=tuple(int(room_id) for room_id in order),
            skipped=(),
            solve_ms=(time.monotonic() - started) * 1e3,
            units=str(getattr(instance, "units", "?")))
        return order


# -- reading the instance without importing numpy -------------------------
def _intersect(candidates, instance):
    # type: (Sequence[Any], Any) -> Tuple[List[Tuple[Any, int]], Tuple[int, ...]]
    """Pair each candidate with its vertex, and name the ones with none.

    The pairs come back sorted by vertex index -- ascending room id, the
    order ``room_costs`` builds the index space in -- rather than in the
    ranking's order. The optimum does not depend on it, but tie-breaking
    does, and a canonical order is what stops a re-solve after an unrelated
    room appears from returning a different-looking tour of equal cost.

    Args:
        candidates: The supervisor's eligible rooms.
        instance: The ``HppPtInstance``.

    Returns:
        ``(kept, skipped)``, kept as ``(candidate, vertex index)`` pairs and
        skipped as the room ids the instance had no vertex for.
    """
    index_of = {}                               # type: Dict[int, int]
    for index, pid in enumerate(instance.index_to_pid):
        pid = int(pid)
        if pid < 0:
            # The aircraft's own vertex. It is the depot, never a room to
            # visit -- and a room whose pid really is the depot's index is
            # still a room, so this tests the pid and not the index.
            continue
        index_of.setdefault(pid, index)

    kept = []                                   # type: List[Tuple[Any, int]]
    skipped = []                                # type: List[int]
    for candidate in candidates:
        room_id = int(candidate.room_id)
        index = index_of.get(room_id)
        if index is None:
            skipped.append(room_id)
        else:
            kept.append((candidate, index))
    kept.sort(key=lambda pair: pair[1])
    return kept, tuple(sorted(skipped))


def _cap(kept, depot_row, probs, limit, reserved=NEAREST_RESERVED):
    # type: (List[Tuple[Any, int]], Sequence[float], Sequence[float], Optional[int], int) -> Tuple[List[Tuple[Any, int]], Tuple[int, ...]]
    """The rooms most worth deciding between, and the ones that had to go.

    :data:`NEAREST_RESERVED` slots go to the closest rooms and the rest to the
    likeliest, because a cap that ranked on probability alone would be
    ``greedy_probability_order`` used as a prefilter -- the baseline RPT*
    exists to beat, applied to the very rooms it was about to weigh against
    cost, and able to drop the room the aircraft is standing next to before
    the solver ever sees it. Both keys carry the other as their tie-break and
    the vertex index last, so a FLAT belief -- a real operating state, not a
    hypothetical, since the oracle publishes a uniform ranking under
    ``source='uniform_fallback'`` whenever the LLM is down -- falls through to
    the nearest rooms, and nothing depends on the order the ranking arrived
    in.

    Args:
        kept: ``(candidate, vertex index)`` pairs from :func:`_intersect`.
        depot_row: The instance's cost row out of the depot.
        probs: The instance's probability vector.
        limit: The most to keep. Zero, None or a negative keeps
            everything -- a negative would otherwise slice from the far end
            and silently drop the likeliest rooms instead of the least.
        reserved: Slots held for the nearest rooms. See
            :data:`NEAREST_RESERVED`.

    Returns:
        ``(kept, capped)``, kept back in vertex-index order and capped as the
        room ids that did not make it. The re-sort is for legibility and for
        :func:`_intersect`'s stated contract, NOT for determinism -- the
        matrix is sliced from this same list either way, and the indices in
        the sort keys already make the selection independent of the order the
        candidates arrived in.
    """
    if not limit or int(limit) < 0 or len(kept) <= int(limit):
        return kept, ()
    likeliest = sorted(kept, key=lambda pair: (-float(probs[pair[1]]),
                                               float(depot_row[pair[1]]),
                                               pair[1]))
    nearest = sorted(kept, key=lambda pair: (float(depot_row[pair[1]]),
                                             -float(probs[pair[1]]),
                                             pair[1]))
    chosen = []                                 # type: List[Tuple[Any, int]]
    taken = set()                               # type: set
    for pair in nearest[:max(0, int(reserved))]:
        chosen.append(pair)
        taken.add(pair[1])
    for pair in likeliest:
        if len(chosen) >= int(limit):
            break
        if pair[1] not in taken:
            chosen.append(pair)
            taken.add(pair[1])
    capped = tuple(sorted(int(candidate.room_id)
                          for candidate, index in kept if index not in taken))
    return sorted(chosen, key=lambda pair: pair[1]), capped


def _as_rows(matrix):
    # type: (Any) -> Sequence[Sequence[float]]
    """The cost matrix as nested sequences, whatever it arrived as.

    ``tolist()`` on the way in is the whole of this module's numpy story: one
    C-level conversion, and everything after it is plain Python. A matrix that
    is already a list of lists passes straight through, which is what lets a
    test build an instance without numpy.
    """
    to_list = getattr(matrix, "tolist", None)
    return to_list() if callable(to_list) else matrix


def _as_list(values):
    # type: (Any) -> Sequence[float]
    """The probability vector as a flat sequence."""
    to_list = getattr(values, "tolist", None)
    return to_list() if callable(to_list) else values


def _clamped(probability):
    # type: (Any) -> float
    """A probability RPT* will accept, or one it will refuse loudly.

    Clamps into ``[0.0, P_CEILING]`` exactly as ``build_instance`` already
    does, so this changes nothing about an instance built the normal way. A
    NaN survives untouched and is refused by ``RouteProblem``, because a NaN
    probability is a bug upstream and quietly turning it into a number would
    hide it.
    """
    probability = float(probability)
    if probability < 0.0:
        return 0.0
    if probability > P_CEILING:
        return P_CEILING
    return probability
