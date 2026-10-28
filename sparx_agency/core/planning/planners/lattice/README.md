# lattice — kinematic planning on the discrete-action state lattice

The 2D grid planner (`../astar/weighted_planner_2d.py`) says *where* a route
goes. This package says *exactly which actions* walk it: an A\* over the state
space (x, y, heading) whose only edges are the benchmark's own
`MOVE_FORWARD` (0.25 m along the heading), `TURN_LEFT` and `TURN_RIGHT`
(30 degrees in place) -- Hybrid A\* with the discrete action set for motion
primitives. Every pose on the plan is one the agent will stand at, to the
millimetre, if the simulator executes every action; the policy's converter
executes the plan verbatim (`NavigationCommand.actions`).

Why it exists (2026-10-28): the converter's waypoint ladder re-decides from
the true pose every step with a heading quantised to half a turn, and on the
HM3D recordings that is where the actions went -- oscillating onto and off a
path through a doorway, clipping the jamb with a 15-degree heading error, and
a refused step painted on the map sealed the doorway for the rest of the
episode. A plan made on the lattice has no such error to correct: its turns
are where the path needs them and its straight runs are straight.

```text
 2D route (WeightedAStarPlanner2D)            lattice (this package)
 ───────────────────────────────              ──────────────────────
 Path2D points, cost grid, clearance ──────►  route_field.py   cost-to-go of the route, inf outside its corridor
                                              primitives.py    the spec's step and turn, the collision samples
                                              search.py        A* over (x, y, k): MOVE_FORWARD checked on the EDT,
                                                               TURNs priced as actions, heuristic from the route field
                                              planner.py       DiscreteActionLatticePlanner: refused-move memory, stats
 LatticePlan(actions, poses, cost) ◄───────── 
```

| module | holds |
|---|---|
| `params.py` | `LatticePlannerParams`: action costs, the corridor, the goal tolerances, the expansion cap, the heuristic weight, the refused-move horizon |
| `primitives.py` | `MotionPrimitives` from a `DiscreteActionSpec`: headings per circle, the lattice heading of an index, the forward end point, the sample fractions along a step |
| `route_field.py` | `route_corridor_fields` → `(cost_to_go, offset)`: every cell within `corridor_m` of the route gets the route's remaining cost (its length weighted by the cost grid) from the nearest route point plus the offset, and the offset alone; `inf` beyond. `route_cost_to_go` is the first of the two |
| `search.py` | `lattice_search` and `LatticePlan`; the statuses `success`, `no_plan`, `expansion_cap`, `start_outside_corridor`, `goal_outside_grid` |
| `planner.py` | `DiscreteActionLatticePlanner.plan(start, route, origin, resolution, cost, clearance, footprint_radius_m, ...)`, `note_refused`, `move_refused`, `diagnostics` |

## How the search works

- **State.** The exact continuous pose, merged for the closed set by the grid
  cell of the position and the lattice heading index (`yaw0 + k * turn`): the
  standard Hybrid A\* trade, a 0.1 m cell with 12 headings.
- **Edges.** `MOVE_FORWARD` is sampled every `sample_spacing_m` (5 cm) along the
  step and feasible only where the cost grid is finite and the clearance
  (the exact Euclidean distance transform the 2D planner already computes) is
  at least `footprint_radius_m` -- the standoff the route was planned at. Its
  cost is `forward_action_cost` times the mean per-cell cost it crosses, so
  the 2D planner's clearance and medial-axis layers price the lattice too. A
  TURN is always feasible and costs `turn_action_cost`. A turn that undoes the
  previous turn is pruned.
- **Objective.** Actions, not metres: a benchmark with a step budget charges a
  TURN what it charges a MOVE_FORWARD.
- **Heuristic and domain.** The route's cost-to-go field, divided by the step
  and weighted by `heuristic_weight` (1.5). A state outside the corridor
  (`corridor_m`, 0.4 m) is never expanded, and every forward step pays
  `route_offset_cost` (2 actions per metre) for the lateral offset of its end
  point from the route (`route_corridor_fields`), so the plan rides the
  route's own line -- the medial axis the 2D planner's centreline layer put
  it on -- and leaves it only to line up a heading. The remaining cost is the route's length *weighted by the
  cost grid*, because a doorway cell costs several times an open one and a
  heuristic that counts it as one step is surprised by every doorway: on a
  32 m synthetic route through three 0.8 m doorways the unweighted heuristic
  flooded 60,000 states without a plan, the weighted one planned in 6,400
  (23 ms) with the 1.5 m corridor and in 500 (5 ms) riding the route. Ties on
  f go to the deeper node.
- **Goal.** Within `goal_tolerance_m` of the route's last point (the
  converter's 0.25 m, so arrival agrees), and within half a turn of
  `goal_yaw` when one is given.
- **The relaxed start.** Within `start_free_radius_m` of the start a sample
  in a lethal-but-not-occupied cell is passable: the body stands inside a
  wall's inflation skirt without touching it, as the 2D planner's own start
  relaxation allows.

## Refused moves

The simulator refused a MOVE_FORWARD from a pose: `note_refused(x, y, yaw,
step)`. For `refused_move_actions` actions (120; 0 for the episode) every
forward primitive from within `refused_move_radius_m` (0.15 m) of that pose at
the same lattice heading is excluded, and nothing else -- the cell ahead stays
whatever the map says, every other heading through it stays open, the way
back stays open. The next plan turns first, or lines up a few centimetres
over. This replaces, inside doorways, the occupancy-grid mark that
`ObservedMap.notify_blocked` writes (see
`core/planning/object_search/navigation/scrape_guard.py`).

## What it is not

Not a `BasePlanner`: it refines a route the 2D planner has already found
through the same cost grid, so its inputs are the route and the grid's cost
and clearance fields, not a start and a goal on a map. It plans in the plane:
a storey at a time, as the converter walks. And it is Python 3.8 syntax on the
numpy 1.17 API with no scipy, like the rest of `planners/`, though today only
the object-search policy (`core/planning/object_search/navigation`) calls it.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/planning/planners/lattice -q
```

The doorway rig: a 0.8 m doorway in a wall, the plan passes within 15 cm of
its axis at the full standoff, is executable by the converter's own transition
model, and spends fewer actions and turns than the ladder's rollout on the same
route; a refused pose makes the next plan turn first; failures are named.
