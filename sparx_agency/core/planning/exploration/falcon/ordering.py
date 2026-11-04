"""FALCON Sec. V-A/B/C: open tours, precedence and layered refinement.

The optimization problems are retained, not LKH's implementation. Exact subset
DP handles small local domains; a deterministic beam bounds larger problems.
No external solver process, random seed, ROS service or unbounded retry exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


class PlanningDeadline(RuntimeError):
    """A computation limit, never an exploration or simulator action limit."""


class PlanningCapacity(PlanningDeadline):
    """A deterministic input/work cap reached before starting a native search."""


class Deadline:
    def __init__(self, seconds, clock=time.monotonic):
        self.clock = clock
        self.end = clock() + seconds

    def check(self):
        if self.clock() >= self.end:
            raise PlanningDeadline("FALCON planning computation deadline")


@dataclass(frozen=True)
class Tour:
    order: tuple
    cost: float
    status: str
    expanded: int


def solve_order(cost, precedence=(), *, exact_limit=10, beam_width=128, deadline=None):
    """Minimum-cost Hamiltonian OPEN path from vertex 0 with hard precedence.

    A zero-cost return column in upstream ATSP gives this identical objective.
    ``precedence`` contains (before, after) pairs, not negative edge weights.
    Unlike upstream magic costs, unreachable edges stay infinite. A beam result
    is explicitly approximate. A timeout is a visible failure, not a new policy.
    """
    cost = np.asarray(cost, dtype=float)
    n = len(cost)
    if cost.shape != (n, n) or n == 0 or np.isnan(cost).any() or (cost < 0).any():
        raise ValueError("Need a nonnegative square cost matrix")
    required = [0] * n
    for before, after in precedence:
        if not 0 <= before < n or not 0 < after < n or before == after:
            raise ValueError("Invalid precedence")
        required[after] |= 1 << before
    states = {(1, 0): (0.0, (0,))}
    expanded, pruned = 0, False
    for _ in range(1, n):
        if deadline is not None:
            deadline.check()
        following = {}
        for (mask, last), (total, order) in sorted(states.items()):
            for nxt in range(1, n):
                if mask & (1 << nxt) or required[nxt] & mask != required[nxt]:
                    continue
                edge = float(cost[last, nxt])
                if not math.isfinite(edge):
                    continue
                key = (mask | (1 << nxt), nxt)
                candidate = (total + edge, order + (nxt,))
                if key not in following or candidate < following[key]:
                    following[key] = candidate
                expanded += 1
                if deadline is not None and expanded % 256 == 0:
                    deadline.check()
        if not following:
            return Tour((), math.inf, "infeasible", expanded)
        if n > exact_limit and len(following) > beam_width:
            # Optimizing the complete path objective, not a one-frontier score.
            ranked = sorted(following, key=lambda k: following[k])[:beam_width]
            following = {k: following[k] for k in ranked}
            pruned = True
        states = following
    total, order = min(states.values())
    return Tour(order, total, "beam" if pruned else "optimal", expanded)


def refine_layers(layers, edge_cost, deadline):
    """Shortest path through ordered candidate layers (upstream Dijkstra DAG).

    A layer is one frontier's qualified viewpoints. The first layer holds the
    measured start. Free-space reachability is mandatory, including the first
    leg; no upstream 'subtract failure penalty' relaxation is reproduced.
    """
    states = [(0.0, (point,)) for point in layers[0]]
    for layer in layers[1:]:
        deadline.check()
        following = []
        for point in layer:
            choices = [(value + edge_cost(path[-1], point), path + (point,))
                       for value, path in states]
            if choices:
                best = min(choices, key=lambda item: item[0])
                if math.isfinite(best[0]):
                    following.append(best)
        if not following:
            return ()
        states = following
    return min(states, key=lambda item: item[0])[1]
