# RPT* — the visiting order that finds a thing soonest

You have a set of places, a belief about how likely the target is to be at each,
and the cost of flying between them. In what order should you look?

Neither obvious answer is right. Going to the most likely place first ignores
that it might be across the building. Flying the shortest tour ignores
everything you believe. The cheapest ordering trades the two off, and it does so
in a way that is not greedy in either: **an edge late in the route is only paid
for if nothing was found earlier**, so the cost of a route depends on the order
it was built in, not just on which edges it uses.

That history dependence is what makes the problem hard, and this package
implements the paper that solves it exactly:

> Yunpeng Lyu, Chao Cao, Ji Zhang, Howie Choset, Zhongqiang Ren.
> *RPT\*: Global Planning with Probabilistic Terminals for Target Search in
> Complex Environments.* arXiv:2601.12701, January 2026.

**No code was released with that paper.** This is a clean-room implementation
from the text. The correspondence table below says where every equation lives,
and the corrections section records the three places where the paper is wrong.

```python
from sparx_agency.core.planning.routing.rpt_star import (
    RouteProblem, RouteVertex, costs_from_points, solve)

rooms = [RouteVertex(id="kitchen", prob=0.5, label="kitchen"),
         RouteVertex(id="ward",    prob=0.3, label="ward"),
         RouteVertex(id="store",   prob=0.2, label="store")]

problem = RouteProblem.with_external_start(rooms, payload=robot_pose)
matrix = costs_from_points([robot_xy, kitchen_xy, ward_xy, store_xy])

solution = solve(problem, matrix)
fly_to(solution.next_id)        # then learn something, and solve again
```

Python 3.8, standard library only. No ROS, no numpy, no scipy — it is imported
by the FALCON adapters inside the Noetic container, and a test enforces that.

## How it works, in one paragraph

Carry one extra number along the route: `q`, the probability of still having
found nothing. With it, the expected cost stops being a sum of long probability
products and becomes a plain sum in which each edge is weighted by the `q` in
force when you traverse it. That makes the cost *additive along the route*,
which is exactly what an A\* g-value has to be, so a search state
`(where I am, what it cost, q, where I have been)` depends only on itself and
not on its history — and ordinary A\* applies. That substitution is the whole
idea. Everything else is standard machinery: a heuristic from a relaxed problem,
a dominance rule for pruning, and focal search for a bounded-suboptimal variant.

## Where the paper lives in the code

| paper | here |
|---|---|
| Eq. 1 / Def. 1, the expected cost as printed (p.3) | `objective.py: expected_cost_literal` |
| Lemma 1, the same cost with `q` (p.4) | `objective.py: expected_cost` |
| The search state (p.4) | `state.py: SearchState` |
| Eqs. 2–4, successor generation (p.4) | `state.py: successors` |
| Alg. 1 line 1, the initial state (p.5) | `state.py: initial_state` — **corrected, see below** |
| Def. 3 + Lemma 3, dominance (p.4) | `dominance.py: dominates` |
| `IsPruned`, `FilterAndAddFront` (p.5) | `dominance.py: VertexFrontier` |
| Algorithm 1, the main loop (p.5) | `search.py: search` |
| Eq. 5 + Lemma 4, the `gamma` table (p.5) | `heuristic.py: GammaTable._build` |
| Eq. 6 + Lemma 5, `h(s)` (p.5–6) | `heuristic.py: GammaTable.estimate` |
| Sec. IV-D + Thm 3, F-RPT\* (p.6–7) | `focal_list.py: FocalList` |
| Assumption 1 + Lemma 6, why pruning is sound (p.6–7) | `validation.py: _require_triangle_inequality` |
| Sec. VII-C, the Greedy and LKH baselines (p.12) | `baselines.py` |
| Thm 2, that the answer is optimal | `brute_force.py`, and `tests/test_optimality.py` |

## Three corrections to the paper

**1. Algorithm 1's initial state is wrong, and provably so.** Line 1 (p.5)
prints `s_o <- (v = v_s, g = 0, q = 1 - p(v_s), A = {})` — an empty visited set.
It cannot be empty. Eq. 5 builds the heuristic table with rows `k = 0 .. |V|-1`,
and Eq. 6 indexes it with `k(s) = |V| - |A(s)|`. An empty `A` asks for row `|V|`,
one past the last row the paper itself defines: the very first heuristic
evaluation is out of bounds. It would also let the start vertex be revisited and
make the goal test `A(s) = V` unreachable. With `A = {v_s}` everything lines up,
and `q = 1 - p(v_s)` matches `q_1` in Lemma 1 — the start has been searched, so
its probability is consumed. This implementation uses `A = {v_s}` and reproduces
the exhaustive optimum on hundreds of random instances.

**2. Lemma 4's complexity is wrong.** It claims the table costs `O(|V|^2)`,
counting the `|V| x |V|` entries and forgetting that *each* entry minimises over
the other `|V|-1` vertices. It is `O(|V|^3)`. Harmless at these sizes — 200
places is eight million operations — but a caller sizing a budget from the
paper's figure would be wrong by two orders of magnitude.
`tests/test_paper_claims.py` measures this by counting matrix reads.

**3. Two figures call the algorithm "PRT\*".** Fig. 5's block and Fig. 8's
caption (p.9, p.11). Cosmetic, noted so a reader is not confused.

## What actually governs the runtime — and it is not the number of places

The paper reports scaling only against vertex count and never says what
probability distribution its instances used. That omission hides the dominant
factor. Measured here, on identical geometry with a 30 000-expansion cap:

| probabilities | N=10 | N=14 | N=18 | N=22 | N=26 | N=30 |
|---|---|---|---|---|---|---|
| `~U(0, 0.50)` | 91 exp | 206 | 180 | 134 | 1 320 | 1 714 |
| `~U(0, 0.10)` | 542 | 1 542 | **over cap** | — | — | — |
| `~U(0, 0.02)` | 837 | 18 924 | **over cap** | — | — | — |

`q` is a product of `(1 - p)` terms. **Large probabilities make it collapse**,
the cost stops growing, the heuristic becomes nearly exact, and the search is
trivial. **Small probabilities keep it near one**, the problem degenerates
towards a plain travelling-salesman instance, and it explodes.

### What this means for the default, and for your vertex set

`RptStarParams.epsilon` **defaults to `None`, meaning exact.** The paper's own
suggested `0.01` (p.12) was measured here against exact search across belief
shapes and sizes and returned *the same routes, slightly more slowly*, while
downgrading the guarantee from optimal to bounded. Its band is too narrow to
admit anything exact search would not have expanded anyway. Epsilon is worth
setting at around `0.5`, and only in a narrow window — a concentrated belief
over roughly 18–24 places, where it roughly triples how often the search
finishes inside its budget.

**RPT\* is worth most exactly where it is cheapest.** Measured at 12 places,
against the true optimum:

| belief | largest `p` | nearest-neighbour | greedy-on-probability |
|---|---|---|---|
| very flat | 0.11 | 1.06× optimal | 2.19× |
| flat | 0.17 | 1.05× | 1.99× |
| peaked | 0.35 | 1.11× | 1.75× |
| very peaked | 0.53 | 1.15× | 1.60× |

The flatter the belief, the less there is to gain — simply flying to the
nearest unvisited place is within about 5% of optimal, and the search is at its
most expensive. The more concentrated the belief, the more RPT\* wins *and* the
faster it runs, because a large `p` is what makes `q` collapse. Going to the
most likely place regardless of distance is bad everywhere, and worst when the
belief is flat.

**So the practical guidance is to keep the vertex set small and the belief
sharp.** Beyond about 24 places with a flat belief nothing finishes, and the
fallback route takes over. If you are building a vertex set by merging
categories — rooms plus frontier clusters, say — merge the clusters
aggressively rather than handing the solver fifty near-identical candidates: at
that size the belief is necessarily flat, the problem is a travelling-salesman
instance in disguise, and the search cannot help you.

## Four things the caller must get right

1. **The triangle inequality.** Dominance pruning is only sound if a detour
   through a third place never beats going direct (Lemma 6). Violate it and the
   search still returns a legal route with an honestly computed cost — just not
   the cheapest one, with nothing anywhere to say so. So it is checked, and a
   violation raises. **Shortest-path costs satisfy it by construction**, so a
   caller using A\* path lengths between centroids — which is what the paper's
   own system does (p.12) — will never see this. `metric_closure()` is the fix
   if you need one; it is offered rather than applied, because a cost silently
   rewritten under you is a cost you can no longer reason about.
2. **Every place must be reachable from every other.** RPT\* routes through
   *all* of them, so one unreachable place makes the instance infeasible rather
   than merely expensive. Drop it from the problem. Do not give it a large
   finite cost — that looks feasible and silently distorts everything else.
3. **`p(v)` is in `[0, 1)`.** One is excluded, not clamped: the heuristic
   divides by `1 - p`, and a certainty is not a search problem.
4. **Probabilities need not sum to one.** Remark 1 (p.3) is explicit that the
   algorithm does not depend on it, and nothing in the search ever sums them. A
   language model's ranking breaks it constantly and is perfectly usable. It is
   a warning, never an error.

## Design decisions worth knowing

- **One entry point.** `solve(problem, matrix, params)`; `epsilon=None` runs
  the exact search, a number runs F-RPT\*. The variants differ in one line of
  Algorithm 1 and share validation, heuristic, pruning and result type — two
  front doors would let the pruning drift, and a drifted pruning rule does not
  crash, it just quietly stops being optimal.
- **`status` and `guarantee` are two fields, not one.** A search can terminate
  perfectly cleanly and still owe no quality claim — because the budget expired
  and the route returned is the best complete one that happened to turn up. One
  "success" flag reports that as success.
- **Every result carries `lower_bound`,** so it certifies itself: no ordering of
  these places costs less than that, whatever happened during the search. It is
  the only thing a timed-out run can still say for certain.
- **An exhausted budget returns a route, it does not raise.** Handing a mission
  node `None` at the moment it needs a goal is worse than handing it an
  unguaranteed route plus fields saying so. When the budget expires before the
  search has completed *any* ordering — the normal outcome on a hard instance —
  a nearest-neighbour route is returned instead, and `route_source` says so.
  That is somewhere to fly, not an answer: `status` still says the budget ran
  out and `guarantee` is still withdrawn.
- **The dominance rule has exactly one definition.** `VertexFrontier` calls
  `dominates()` rather than inlining the comparison, even though the inline
  version is marginally faster in the hot loop. A second copy of a pruning rule
  is the drift this repository has been bitten by before, and it hides well:
  both copies agree, every test passes, and mutating the canonical one changes
  nothing — which is how you find out nothing calls it.
- **The reconstructed route is always re-scored** against the g-value the search
  terminated on. Linear against an exponential search, and it closes the entire
  family of bugs where the state machine and the objective drift apart.
- **Ties break on insertion order.** A planner that returns a different route
  each time it is asked the same question cannot be debugged from a recording.
- **The dominance frontier is a plain list.** A sorted structure was measured
  and bought about ten percent, which is not worth obscuring the one piece of
  code the whole optimality proof rests on.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/planning/routing/rpt_star/tests/ -q
```

- `test_optimality.py` — the search against exhaustive enumeration over hundreds
  of random instances in three probability regimes; the F-RPT\* bound at four
  epsilons; that `epsilon=0` recovers the optimum exactly.
- `test_paper_claims.py` — Lemmas 1, 3, 4 and 5 as executable claims.
  Admissibility is checked against the true cost-to-go computed exhaustively,
  which is the definition rather than a proxy for it.
- `test_contract.py` — what is refused, what is merely warned about, and the
  degenerate shapes: one place, two places, no belief at all, an exhausted
  budget.
- `test_python38_contract.py` — an AST scan for anything that would not import
  in the Noetic container. The package has also been imported and run under real
  Python 3.8.10 in `falcon-ros-custom:v1`.

## Not in here, on purpose

- **Building the cost matrix from a map.** That is `routing/`'s planned second
  inhabitant. Until it exists, `costs_from_row_callback` lets a caller supply
  one whole row per graph search — twenty places cost twenty searches, not 380.
- **Caching costs across replans.** The caller knows when its map changed; this
  package must not guess.
- **A cost for searching *at* a place.** The objective charges travel only
  (Def. 1 / Eq. 1, p.3). Folding a constant dwell into the outgoing
  edges is exact; a dwell that scales with room size is not, and that change
  would alter Eq. 1 and take the paper correspondence with it.
- **A ROS node.** `core/` is ROS-free; wiring belongs beside its consumer under
  `tasks/`.
