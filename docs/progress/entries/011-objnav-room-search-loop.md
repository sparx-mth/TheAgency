# 011 - ObjectNav room-search loop (bounded local exploration, global re-plan)

**Branch:** `feat/objnav-grounded-vlm-nadav` (continues 010; the user decides on a split)
**Status:** in-progress
**Roadmap item:** [ObjectNav room-search loop](../ROADMAP.md)

## Goal

The frontier/RPT ObjectNav policy follows one explicit loop, step for step as
specified on 2026-09-28:

- **Background, every action:** detection, landmarks, and a scene graph with
  object→room associations, doors/stairs/floor transitions, current room labels,
  cumulative search time per room and remaining frontiers per room.
- **1. Local exploration, bounded to the room in force.** No door, no stairs.
  Ends after `local_steps` actions (10) or when nothing reachable is left.
- **2. Global re-classification** of every known room from every confirmed
  object so far.
- **3. Per-room probability** that the target is there *and* the search ends
  there, from remaining frontiers, cumulative search time and current distance.
- **4. Visit order** from a joint function of probability and travel cost (RPT*).
- **5. A\*** to the head room.
- **6. Direct transit** to the *closest frontier inside* that room.
- **7. Reset** the local counter on arrival and loop.

## Why

The pieces already existed (supervisor state machine, RPT* solver, LLM oracle,
frontier ranking) but the control flow around them did not match the loop: the
LLM ran on a fixed 10-action timer regardless of phase, a room's turn ended on
clocks, a released room went on a 120-action cooldown even when only its budget
had run out, the transit goal was the room *centroid*, the committed visit order
was walked to its end instead of being re-solved after each room, and nothing
kept an in-room route from leaving the room. The FALCON adaptation
(`falcon_policy._reason/_select`) already ran this exact burst→reason→order
pattern; the frontier method now does too, through the supervisor.

## Steps

- [x] Core supervisor: `arrived` and `budget_spent` caller-driven exits,
      `frontier_exhausted` honoured in TRANSIT, `cooldown_verdicts` and
      `resolve_on_release` params (defaults keep the flight behaviour).
      Added along the way: `repeat_verdicts` (see Notes 3).
- [x] Core frontier ranking: `frontier_goals_by_room` -- one extraction, one
      Dijkstra, clusters assigned to rooms by the same majority rule the count uses.
      `room_costs.frontier_clusters` carries member cells for the vote.
- [x] Runtime `room_search_loop.py`: the seven steps, explicit and logged; room
      confinement by blocking every other room's cells in the planning copy of the map;
      entry goal = nearest reachable frontier inside the room (centroid fallback);
      re-plan on the same action a room ends; LLM only at the loop point.
- [x] Runtime `frontier_sweep.py` reduced to goal generation, goal persistence and the
      optional look-around (default 0 turns, per the termination rule).
- [x] `rpt_policy`: scene-graph geometry refreshed every action (`graph_period_steps`
      1) without an LLM call; the loop calls `graph.reason()`; RPT* charges the local
      budget as per-room service time; configuration/diagnostics report the loop.
- [x] Regressions: supervisor flags and params (13 new); by-room goals (4); loop (27,
      on a hand-drawn two-room world with the real supervisor, A* and ranking) --
      budget release re-selects without cooldown, confinement, entry frontier, arrival
      by mask, exhausted-in-transit, LLM once per loop point, round guard, floor
      travel; oracle reuse (4). 667 pass across the runtime, exploration and
      topology suites.
- [x] README (runtime, topology), CHANGELOG, this entry; LESSONS for the three churn
      mechanisms the first trace exposed.
- [ ] Re-run Ranchester and compare (user decides when).

## Open questions

- Whether `local_steps=10` is the right N: it also sets the LLM cadence (one
  reasoning round per local burst, plus one per room release).
- Distance is an *input* to step 3 as specified, but it is consumed by the
  RPT* objective (step 4) rather than multiplied into the probability -- doing
  both would double-count travel. The estimate record carries it per room.
- Rooms with no frontier left keep a centroid entry so the detector can still
  get a close look; the oracle's exhausted factor and the cooldown bound that.
- Scene-graph geometry every action costs ~65-80 ms on the 80 m / 0.1 m map
  (measured with one room; the watershed runs over the whole 800x800 grid).
  `policy_decision` is ~180-190 ms median with fake services. Cropping the
  segmentation to the known bounding box would recover most of it; not done here.

## Notes

1. The plan grew three things the entry did not foresee, all found by running the
   loop end to end (`tests/trace_room_search_loop.py`) rather than by its unit tests:
   the frontier goal moved from the cluster centroid to the member cell nearest it
   (the centroid of an arc lies in seen floor and was retired the action it was
   adopted); the in-room sweep now draws goals by the count's majority vote instead
   of the eroded watershed mask; and `SearchOracle` reuses a reply whose prompt is
   unchanged, since a loop point over an unchanged map otherwise spends a model
   call to get the same scores. LESSONS has the full account.
2. The N-step check moved from inside the in-room sweep to before the first
   supervisor tick of an action: judged after a tick it cost a round, and the
   same-action chain budget -> reselect -> arrive -> exhausted -> reselect overran
   the 3-round guard. The guard is 5 and an overrun is counted and logged.
3. `repeat_verdicts`: the supervisor's every-room-cooling escape hatch re-entered a
   room the live map had just declared finished, on every action. The loop keeps
   `EXHAUSTED`/`MAPPED` rooms out of it and falls back to the floor-wide frontier;
   the aircraft default (repeat anything) is unchanged.
4. The loop also reasons when a room the estimate has never seen appears while the
   supervisor is holding -- a room without a probability can never be chosen, and no
   release is coming to give it one. Bounded by the number of rooms, not by actions.
5. `graph_period_steps` is now geometry-only and defaults to 1; a loop point refreshes
   the geometry itself if the background did not on that action, so raising the
   period to save time does not stale the estimate.

## Result

Implemented on `feat/objnav-grounded-vlm-nadav`, not yet flown. On the headless probe
with a route-following executor: transit, arrival at step 10, exactly ten local
steps, budget release at step 20 with re-reason, re-solve and transit to the room's
nearest frontier on that action; a refused transit route ends the room as
`unreachable` and re-plans at once; 3 LLM rounds in 60 actions (was 6 in 12 before
the fixes in Notes 1-3). Ranchester comparison pending.
