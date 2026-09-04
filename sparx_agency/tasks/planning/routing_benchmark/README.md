# Routing benchmark — is an optimal search order worth it?

Measures what [`core/planning/routing/rpt_star`](../../../core/planning/routing/rpt_star/)
actually buys, against the alternatives, with no simulator involved. Buildings
are generated, beliefs are modelled, and every score is an exact expectation.
A full sweep runs in minutes on one core.

```bash
# the factorial sweep: 480 scenarios x 6 planners
.venv/bin/python -m sparx_agency.tasks.planning.routing_benchmark.run_benchmark

# the two questions underneath it: how good must the oracle be, and how much
# does a sharper belief buy?
.venv/bin/python -m sparx_agency.tasks.planning.routing_benchmark.crossover
```

Results land in `~/rpt_star_benchmark/`, outside the repository.

## The question it answers, and the one it refuses to answer

**It answers:** given a building, a belief about where an object is, and the
cost of walking between rooms, how much less does the robot fly if it visits
the rooms in the optimal order rather than the obvious one — and what does
computing that order cost?

**It refuses to answer** "is RPT\* optimal", because that is not a benchmark
question. RPT\* provably minimises expected cost *under the distribution it is
handed*; scoring it against its own input would confirm only that it is not
broken. So every planner here is given the oracle's **belief** and scored
against the **truth**, and the gap between those two is the whole experiment.

## How a scenario is built

**Buildings are corridor graphs, not point clouds.** Scattering points in a
square is the standard shortcut and it is a bad test of *this* algorithm: in a
square, every place is about as reachable as every other and the ordering
barely matters. In a building it decides everything — two rooms either side of
a wall are three metres apart and forty metres of walking. So rooms hang off a
corridor spine through doors, and the cost between two rooms is the shortest
path out, along, and in. That also makes the distances **exactly metric**,
which is what RPT\*'s pruning requires; a generator that broke the triangle
inequality would be testing the validator instead of the search.

Four shapes, each making the ordering hard in a different way: a straight
`corridor`, a `ring` (two ways round, so committing to the wrong direction is
expensive — the shape of the hospital floor in the paper's own Fig. 12), a
`cross` of dead-end wings, and a `suite` of rooms chained through one another.

**Beliefs come from a model of an oracle, with the truth kept separate.** Each
room gets a semantic kind; the truth is derived from those kinds; the oracle
sees the truth through a filter with three dials — how much of it survives
(`skill`), how noisily (`noise`), and whether a confident decoy is planted far
away. Six named regimes span perfect to adversarial. The `decoy` regime is the
paper's own misleading prior: mostly right, with a second confident peak
somewhere expensive.

**Every score is an exact expectation, never a sample.** Once an ordering is
fixed, the expected distance is a finite sum over rooms. There is no
Monte-Carlo error anywhere, so a one-percent difference between two planners is
a real difference. A thousand sampled trials would still carry several percent
of standard error — the size of the effects being measured.

## The design

Factorial, not random: 4 topologies × 4 sizes × 6 oracle regimes × 5 seeds =
**480 scenarios**, every cell populated. That makes "RPT\* wins on ring-shaped
buildings when the oracle is misleading" a question with an answer, rather than
a subgroup of eleven instances, and it lets a difference be attributed to the
factor that actually varied.

## Who competes

| planner | pays attention to |
|---|---|
| `rpt_star` | belief and distance, weighed optimally |
| `f_rpt_star` | the same, allowed to stop just short of optimal |
| `nearest_2opt` | distance only, and genuinely well — the one to beat |
| `nearest` | distance only, greedily |
| `greedy` | belief only — the paper's `Greedy` baseline |
| `random` | nothing, so the other numbers have a scale |
| *clairvoyant* | not a planner: it already knows, and bounds everyone |

`nearest_2opt` is included because beating a *weak* distance baseline would
prove very little. On these buildings 2-opt improves nearest-neighbour by only
0.04%, which is itself worth knowing: walking a corridor in order is already
near-optimal for distance, so any win RPT\* records comes from using the
belief and not from tour-shortening.

## What the answers turn out to be

Two things decide whether this is worth flying, and they are independent.

**How right the oracle is decides *whether* it helps.** Below roughly 0.56
agreement between the belief and the truth, a planner that ignores the belief
does better — following a bad prior is worse than having none. Above it, RPT\*
pulls ahead and keeps pulling.

**How concentrated the world is decides *how much*.** With the object nearly
equally likely anywhere there is no ordering to get right and nothing to win.
As the belief sharpens, the saving grows to around 40% against distance-only
search, and RPT\* wins nine scenarios in ten.

`greedy` is far behind everywhere except at extreme concentration, where "go to
the one likely room" is simply correct and every method agrees.

## Files

| file | what it does |
|---|---|
| `buildings.py` | procedural floor plans as corridor graphs, with metric distances |
| `oracle.py` | the truth, and the model of a language model looking at it |
| `scenarios.py` | the factorial design |
| `planners.py` | the competitors behind one signature |
| `metrics.py` | expected distance under the truth, exactly |
| `run_benchmark.py` | the sweep |
| `crossover.py` | the two one-factor sweeps: oracle quality, and world concentration |
| `analysis.py` | grouping and summarising, no plotting |
| `tests/` | the harness's own tests |

## The harness is tested, because a broken benchmark lies quietly

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/tasks/planning/routing_benchmark/tests/ -q
```

The tests worth knowing about: every generated building is handed to the real
validator, so a scenario can never be an illegal problem; the scoring is
checked against a literal room-by-room simulation of the walk, which is where
an off-by-one would otherwise shift every planner by one hop and change the
winner; and RPT\* is asserted to achieve the lowest cost *under the belief it
was given* on every scenario — the check that catches the harness feeding the
solver a transposed matrix or the entrance in the wrong place, each of which
would still produce a plausible-looking results table.
