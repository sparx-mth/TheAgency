# RPT\* paper replication — the algorithm alone, before any pipeline

Reproduces the experiments of

> Yunpeng Lyu, Chao Cao, Ji Zhang, Howie Choset, Zhongqiang Ren.
> *RPT\*: Global Planning with Probabilistic Terminals for Target Search in
> Complex Environments.* arXiv:2601.12701, 19 January 2026.

against [`core/planning/routing/rpt_star`](../../../core/planning/routing/rpt_star/),
on the paper's own datasets and against the paper's own baselines. No ROS, no
map, no detector, no robot — the point is to know whether the algorithm does
what it says before anything is wired to it.

```bash
# everything — a couple of minutes except tsplib
.venv/bin/python -m sparx_agency.tasks.planning.rpt_star_paper.run

# one study
.venv/bin/python -m sparx_agency.tasks.planning.rpt_star_paper.run reliability

# closer to the paper's sizes, much slower
.venv/bin/python -m sparx_agency.tasks.planning.rpt_star_paper.run --full
```

Raw rows land in `~/rpt_star_paper/`, outside the repository.

## The short version

**The implementation is faithful and correct.** Every theorem the paper states
is checked against exhaustive enumeration and holds: RPT\* returns the exact
optimum of Eq. 1, the Lemma 1 rewrite equals Eq. 1 as printed, the heuristic is
admissible, and F-RPT\* honours its bound at every ε the paper uses.

**Two of the paper's claims reproduce.** The heuristic saves 50–70% of the
runtime, and a greedy planner collapses under a misleading prior while RPT\*
does not.

**Four do not, and one of them is a mathematical error that matters for us.**

| # | finding | consequence |
|---|---|---|
| 1 | **Eq. 1 is the wrong objective when there is exactly one target**, which is the case Remark 1 explicitly endorses and the case our pipeline has. | Even with a *perfect* prior, RPT\*'s route flies a median **3.9% further** than necessary overall &mdash; **8.5%** once the belief is concentrated, and **44%** at worst. |
| 2 | **All five TSPLIB instances violate the triangle inequality** that Sec. III assumes and Lemma 6 needs. | Theorem 2 does not apply to Table I; the guarantee is void on the paper's headline dataset. (Measured: it got away with it — no route penalty.) |
| 3 | **The baseline gaps are governed by belief sharpness, which the paper never states.** | Greedy at 2–3× and LKH at 1.5–1.8× cannot come from the same experiment; they sit at opposite ends of an unreported axis. |
| 4 | **When the prior is worse than useless, ignoring it beats RPT\*.** | Contradicts Sec. VII-E. RPT\* faithfully optimises against an input that is actively wrong. |

Everything below is measured, reproducible from a seed, and pinned by a test.

---

## 0. Fidelity — is this really RPT\*?

```
claim                                result
-----------------------------------  ------  ----
RPT* == exhaustive optimum (Thm. 2)   24/24  PASS
RPT*_noh == exhaustive optimum        24/24  PASS
Lemma 1 == Eq. 1                      24/24  PASS
h admissible (Lemma 5)                24/24  PASS
F-RPT* within (1+eps) (Thm. 3)        96/96  PASS
heuristic expansion saving            1.46x
```

Ground truth is every ordering enumerated, which shares no code with the
search, so no bug can be common to both. The implementation departs from the
printed paper in exactly three places, all documented in the
[solver's README](../../../core/planning/routing/rpt_star/README.md): the
initial state's `A = {}` (which is provably a typo — it indexes one row past
the end of the paper's own heuristic table), Lemma 4's complexity (`O(|V|³)`,
not `O(|V|²)`), and two figures spelling the algorithm "PRT\*".

## 1. The objective is wrong for a single target

**This is the finding that matters most for integration, and it is not a
matter of opinion.**

Equation 1 weights the *i*-th edge by

```
q_i = (1 - p_1)(1 - p_2) ... (1 - p_i)
```

the probability of *independently* failing to find a target at each of the
first *i* places. That is exactly right for Remark 1's multi-target case —
"trash bins", where each room independently may or may not contain one, and
the `p(v)` need not sum to anything.

Remark 1 then says (p.3):

> If there is only one target object (such as an ID card) in G and it is known
> that the object must exist in G, then an additional constraint that
> Σ p(v) = 1 should be imposed. **Our approaches ... do not rely on those
> constraints and is applicable to all these problem variants.**

The bolded claim is false. With one object that certainly exists, "it is in
room A" and "it is in room B" are **mutually exclusive**, not independent. The
probability of still searching after *i* places is

```
1 - (p_1 + p_2 + ... + p_i)
```

which is not the product. Two rooms at `p = 0.3` give `1 - 0.6 = 0.40`, where
Eq. 1 uses `0.7 × 0.7 = 0.49`. The paper's weight is always the larger, and by
more the further along the route it is applied — so **Eq. 1 systematically
over-charges the tail of the route**, and the ordering minimising it is not the
ordering that finds a single object soonest.

Measured by enumerating both optima exactly, with the belief set equal to the
truth so that nothing here is about a bad prior:

| Dirichlet α | median max p | median excess | mean excess | worst excess | same route |
|---|---|---|---|---|---|
| 4 | 0.23 | 0.00% | 2.06% | 13.95% | 18/30 |
| 1 | 0.32 | 2.46% | 4.10% | 36.10% | 9/30 |
| 0.5 | 0.48 | 6.98% | 9.61% | 44.24% | 6/30 |
| 0.25 | 0.53 | 8.54% | 9.14% | 25.75% | 2/30 |

Read the last column: with a concentrated belief, **the route Eq. 1 prefers is
the right one in 2 instances out of 30**, and the robot flies a median 8.5%
further for it.

Three things make this worse rather than academic:

- **It gets worse exactly where RPT\* is otherwise most attractive.** A sharper
  belief is where RPT\* beats the baselines by most (§3) — and it is where this
  error is largest. (It is *not* where the search is cheapest: with the belief
  normalised, sharpening it makes the search harder, not easier — 7.4k
  expansions at 16 places on a flat belief against 24.9k on a sharp one.)
- **The paper's own system runs in this regime.** HATS-L's Bayesian filter
  normalises the belief to sum to one (Eq. 25: "η is the normalization factor
  so that the posterior belief sums to one"), and the motivating example in the
  introduction is a single key.
- **It is the regime we have.** A VLM or LLM ranking candidate rooms for one
  object produces a normalised, peaked belief.

The fix is one line — weight by `1 - Σp` instead of `Π(1-p)` — and it is
deliberately **not applied to the solver**, which stays faithful to the paper as
instructed. It is implemented as `metrics.expected_cost_single_target` for
measurement, and it is the first thing to decide at integration. Note the
corrected weight is still additive along the route, so the whole RPT\* machinery
— Markovian state, dominance, focal search — carries over unchanged; only the
`q` update and the γ recurrence need editing.

## 2. Every TSPLIB instance breaks the paper's own assumption

Section III states, as part of the problem definition, "The edge costs satisfy
the triangle inequality". Lemma 6 — the proof that dominance pruning never
discards an optimal completion — consumes it directly, by shortcutting a
repeated vertex and needing `c(a,b) + c(b,c) ≥ c(a,c)`.

All five instances the paper benchmarks on violate it:

| instance | n | violating triples | worst violation | route penalty | guarantee |
|---|---|---|---|---|---|
| gr17 | 17 | 134 | 67 | +0.00% | **none** |
| gr21 | 21 | 208 | 68 | +0.00% | **none** |
| gr24 | 24 | 560 | 111 | +0.00% | **none** |
| fri26 | 26 | 26 | 1 | +0.00% | **none** |
| bays29 | 29 | 492 | 100 | +0.00% | **none** |

So on the entire dataset behind Table I, **Theorem 2 does not apply**. The
solver knows it: with the precondition waived, it withdraws the guarantee and
returns `none` on every instance. RPT\* still returns a route and still costs
it correctly — it simply cannot promise the route is the cheapest one.

**The honest ending, which sharpens the point rather than blunting it:** the
route penalty is zero everywhere. Planning on the broken matrix and planning on
its metric closure yield routes of identical cost, measured consistently under
the closure — the only coherent cost model, in which a direct edge is never
dearer than a detour. So on these instances the pruning did not in fact discard
the optimum.

The finding is therefore about **risk, not damage**. The paper runs its
headline experiment outside the assumption its optimality proof requires, and
gets away with it on this data. Table I's claim is that RPT\* beats Gurobi by
0.004%–0.062%, attributing the gap to floating-point error in the IP solver;
those margins are far smaller than the guarantee that was silently dropped to
obtain them. Nothing in the paper acknowledges the trade, and nothing in a
re-run would warn you.

Comparing each route's cost on *its own* matrix — the obvious thing to do —
shows a spurious +1.41% on gr24. That is not a penalty, it is two different
objectives being subtracted, and it is why the study scores both routes under
one cost model instead.

**A scaling note from the same run.** At the paper's own 60 s limit, only
**6 of 15** exact runs on these instances finished here. The paper reports
exact solutions up to 40 vertices; that is C++, and this is Python against an
exponential search. The gap is the language, not the algorithm — but it fixes
the practical ceiling for *this* implementation at roughly 16–18 places, which
is what matters for planning a mission around it.

## 3. The baseline comparison is governed by an unreported variable

Section VII-C reports Greedy at "two to three times" optimal and LKH at "50–80%
more expensive", as if these were properties of the algorithms. They are
properties of the *belief*, and Section VII never says what belief it used.

The mechanism is structural: `q` is a product of `(1 - p)` terms, so when every
`p` is small `q` stays near one, every edge is charged at nearly full weight,
and **HPP-PT degenerates into plain shortest-path** — at which point a
distance-only planner is already solving the right problem.

Cost ratio to optimal, sweeping only the belief concentration:

| | max p=0.28 | 0.53 | 0.67 | 0.79 | 0.97 | 0.98 |
|---|---|---|---|---|---|---|
| **Greedy** | 1.96× | 1.75× | 1.62× | 1.57× | 1.20× | 1.11× |
| **LKH** | 1.04× | 1.05× | 1.21× | 1.41× | 1.42× | 1.47× |

**The two baselines move in opposite directions.** Greedy is worst on a flat
belief; LKH is worst on a sharp one. The paper's two numbers — Greedy at 2–3×
and LKH at 1.5–1.8× — do not coexist anywhere on this axis. Greedy reaches 2×
at `max p ≈ 0.28`, where LKH is 1.04×, not 1.5×.

The practical reading: **if your belief is flat, do not run RPT\*.** At
`max p = 0.28` it buys 4% over a good TSP heuristic and costs orders of
magnitude more compute. Its value is concentrated entirely in sharp beliefs.

## 4. Reliability — the experiment the paper does not run

Fixed geometry, 12 places, 20 instances. The truth is held constant and the
belief handed to the planner is degraded from perfect to inverted. Every route
is scored against the **truth**, so these are metres really flown, not what the
planner believed.

**Expected flight distance (m), ratio to RPT\*:**

| planner | perfect | good | fair | weak | uninformative | half-inverted | adversarial | decoy |
|---|---|---|---|---|---|---|---|---|
| RPT\* | 515 (1.00×) | 515 | 515 | 515 | 615 | 741 | 769 | 685 |
| F-RPT\*(0.01) | 509 (0.99×) | 515 | 515 | 551 | 615 | 741 | 769 | 685 |
| Greedy | 705 (1.37×) | 705 | 705 | 705 | 1340 (2.18×) | 2199 (2.97×) | 2199 (2.86×) | 1346 (1.96×) |
| **LKH** | 616 (1.20×) | 616 | 616 | 616 | 616 (1.00×) | **616 (0.83×)** | **616 (0.80×)** | **616 (0.90×)** |
| NN | 633 (1.23×) | 633 | 633 | 633 | 633 | 633 (0.85×) | 633 (0.82×) | 633 (0.92×) |

**What reproduces.** Greedy degrades catastrophically — 705 m to 2199 m, a
factor of 3.1 — while RPT\* moves only 515 m to 769 m, a factor of 1.5. This
is the paper's Table II/III finding and it holds cleanly. RPT\* really does
hedge, and hedging really does pay when the prior is wrong.

**What does not.** Section VII-E claims RPT\* "yields the best performance in
the presence of misleading prior knowledge". It does not. Once the belief is
worse than useless, **the planner that ignores the belief entirely wins**: at
`adversarial`, LKH flies 616 m against RPT\*'s 769 m — RPT\* flies **25%
further**. The reason is exactly what RPT\* is for: it faithfully optimises
against the distribution it is given, and that distribution is now actively
lying.

Note LKH's row is constant at 616 m across every regime — it never reads the
belief, so it cannot be misled. That is the entire point. RPT\* beats it by 20%
when the prior is good and loses to it by 20% when the prior is inverted; the
crossover is at `uninformative`, where they are level.

**Expected flight time (s)** tells the same story with a 30 s dwell charged per
room opened — Greedy's advantage with a good prior (it opens 4 rooms against
RPT\*'s 6) narrows the gap to 1.15× but never reverses it.

**Planning time**, median milliseconds at 12 places — the other half of the
trade:

| planner | planning time | vs RPT\* |
|---|---|---|
| RPT\* | 7.95 ms | 1.00× |
| F-RPT\*(0.01) | 11.28 ms | 1.42× |
| LKH | 3.03 ms | 0.38× |
| NN | 0.01 ms | ~0 |
| Greedy | <0.01 ms | ~0 |

Against a 30 s dwell per room, 8 ms is not a consideration at this size — RPT\*
saves ~100 m of flight for 8 ms of thought. At 18+ places it becomes one; see
the ablation below. Note F-RPT\*(0.01) is **slower** than exact, consistently,
which is covered in §5.

**Places opened** is where Greedy earns its one honest win: with a good prior
it opens 3.90 rooms against RPT\*'s 5.69, because it drives straight at the
answer. That is the paper's Table II result, and it is why Greedy beat RPT\* on
mission duration there. It reverses completely at `adversarial`, where Greedy
opens 9.62 of 12.

## 5. Ablation — the heuristic, and ε

Median planning time in ms, with success rate inside a 60 s budget:

| variant | n=8 | n=10 | n=12 | n=14 |
|---|---|---|---|---|
| RPT\* | 0.2 (100%) | 1.9 (100%) | 8.3 (100%) | 62.8 (100%) |
| RPT\*_noh | 0.4 (100%) | 4.2 (100%) | 23.4 (100%) | 112.4 (100%) |
| F-RPT\*(0.001) | 0.3 | 3.0 | 12.9 | 99.5 |
| F-RPT\*(0.01) | 0.3 | 2.7 | 12.2 | 94.5 |
| F-RPT\*(0.1) | 0.3 | 2.2 | 12.1 | 84.6 |
| F-RPT\*(0.2) | 0.2 | 1.3 | 8.4 | 78.8 |

**The heuristic claim reproduces.** 50–70% saved at n=10–12 (4.2→1.9 ms is 55%,
23.4→8.3 ms is 65%), against the paper's stated 50–70%.

**The ε claim reproduces, and is a warning.** "Varying ε does not affect the
runtime very much" — correct, and here every ε is *slower than exact*. Focal
search's bookkeeping costs more than its band saves at these sizes, while
downgrading the guarantee from optimal to bounded. The paper's suggested
ε = 0.01 is a strict loss below ~18 places. F-RPT\* earns its keep only where
exact search stops finishing at all.

Note the runtime growth: ×7.6 from n=12 to n=14. This is pure Python against
the paper's C++, so absolute numbers are not comparable — but the exponent is,
and it says the practical ceiling here is around 16–18 places, not the paper's
40.

---

## How a scenario is built

**Datasets follow the paper exactly** where it specifies them (`datasets.py`):
points uniform in 500×500 for `|V| ≤ 40` and 5000×5000 above, resampled until
every pair is more than 5 apart (Sec. VII). The five TSPLIB instances are
vendored under `data/`.

**Beliefs are the axis the paper omits** (`beliefs.py`). Each scenario carries
a `truth` — never shown to any planner — and a `belief` derived from it through
a reliability dial from `+1` (the truth) through `0` (uniform, the paper's own
stated fallback for "no clue") to `-1` (rank-inverted). The `decoy` regime is
the paper's misleading prior: a confident peak on the true location and a
larger one on the furthest vertex from it.

Scoring a planner against the distribution it was optimising would be circular —
RPT\* provably minimises that — so the gap between truth and belief *is* the
experiment.

**Every score is an exact expectation, never sampled.** Once the ordering is
fixed the target's location is the only random quantity and it ranges over a
finite set, so each metric is a finite weighted sum. A 1% difference between two
planners is a real 1%.

**The LKH baseline is deliberately strong.** LKH itself is out of scope to
vendor, so `lkh_style_order` runs nearest-neighbour construction from every
possible first hop, each improved to a local optimum under 2-opt *and* Or-opt.
Substituting plain nearest-neighbour would land 15–25% above the optimal tour
and credit RPT\* with a win that belongs to the baseline being bad — the `NN`
row is kept in the tables to show exactly how much that would have been.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
    sparx_agency/tasks/planning/rpt_star_paper/tests/ -q
```

48 tests. They check the datasets follow the paper's recipe, that the belief
dial is monotone and continuous at zero, that the metrics match hand
computations, and that each finding above still holds — including that every
TSPLIB instance still violates the triangle inequality, and that Eq. 1 still
prefers the wrong route for a single target.

## What this does not cover

- **HATS** (Sec. VI). The Bayesian filter, frontier clustering and mean-shift
  are a system around the planner, not the planner. They belong with the
  integration.
- **The integer programs** (Sec. V). Both need Gurobi, and the paper's own
  conclusion is to abandon them.
- **Sizes above ~18 places.** Pure Python against an exponential search. The
  paper's 40-vertex exact ceiling and 200-vertex focal ceiling are C++ numbers.
- **Whether any of this survives contact with a real map.** That is next, and
  it is where finding 1 has to be decided rather than merely recorded.

