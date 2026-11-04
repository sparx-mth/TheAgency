# 013 - ObjectNav: the node oracle, staircases as RPT* nodes, committed climbs, the search column
**Branch:** `feat/objnav-habitat-gibson-nadav` (continues 012)
**Status:** in-progress
**Roadmap item:** [ObjectNav robustness](../ROADMAP.md)
## Goal
The user's model of the search, built into the existing seven-step loop:
1. **No step budget decides anything.** A staircase is a node of the RPT* instance like a room:
   the LLM gives it a probability, the planner charges it the real travel -- the walk to its foot,
   the flight, and the fixed cost of a storey change on every arc into and out of it, because
   going upstairs distances the robot from every room down here -- and the order says when to
   climb. Once the climb is decided it is finished.
2. **The LLM values every node from the whole state**, in one call per loop point: irrelevant
   rooms and fully observed rooms at zero, searched rooms low by how long and how recently,
   `unknown` rooms as exploration nodes worth a look by their size and frontier and the room
   types still missing, staircases by what this storey turned out to be. A precise prompt, and
   a model strong enough to follow it (Qwen-3B is not).
3. **One object names a room**, the name is revised the action a new kind of object appears, and
   a changed name is a fact the oracle has not valued -- so the room's turn ends and the estimate
   and the order are redone at once.
4. **The recording shows the reasoning**: every node's probability and the model's reason, the
   RPT* order, the node in force or in transit, on the video and in the trace.
## Why
- The previous cut (this entry's first draft) reached the user's behaviour through hand-written
  rules -- an affinity table, a "passage prior", an "irrelevant" verdict, a target-aware
  allowance, a "stairs first" preference. Each was a judgement the LLM should make, coded around
  a model too small to make it, and each came with a threshold to tune. The user's feedback was
  exact: the decision is a probability per node and a distance per arc, the LLM owns the first,
  the planner the second, and runtime is not a constraint.
- The old oracle (`search_oracle.py`) hid frontiers and search time from the model because a 3B
  double-counted them; a capable model wants them, and cannot weigh a staircase without them.
## Steps
- [x] `core/mapping/topology/llm_client.py`: a REASONING route (`LLM_REASONING_MODEL`,
      `LLM_REASONING_TIMEOUT_S`, `LLM_REASONING_MAX_TOKENS`) beside the default model;
      `chat_json(..., reasoning=True)`; `LLMConfig.models()`; `VerifiedLLMClient` checks both.
- [x] `core/mapping/topology/search_node_oracle.py`: `SearchNode` (room / stairs), `SearchContext`,
      the system prompt with the judgement spelled out and distance forbidden, `SearchNodeOracle`
      (parse, omitted → small share, rescale, clamp, refuse, reuse); `methods/oracle_retry.py`
      repairs one malformed reply. Tests.
- [x] `core/planning/exploration/room_costs.build_instance(leaves=...)`: a per-node leaf charged
      on every incident arc, metric by construction. Test.
- [x] `core/planning/exploration/object_search_supervisor.py`: `RECLASSIFIED` (neutral, no
      cooldown, no attempt), `TRAVERSED` (productive), `room_reclassified` exit, `finish()` for the
      turn the tick cannot see, `is_cooling()`. Tests.
- [x] `methods/stair_nodes.py`: portal → node (`stair_node_id`, facts, leaf via `stair_cost_m`,
      approach through `floor_decision.approach_points`), storey summaries for the prompt.
- [x] `methods/scene_graph.py`: `nodes()` with `last_inside`, `stair_probs`, `reason(extra_nodes,
      context, here_xy)`, tentative labels shown as `type=kitchen?`.
- [x] `methods/room_search_loop.py`: stairs offered at SELECT, valued with the rooms, in the
      instance with their leaves; `_transit_stairs` → `building.commit`; `stairs_taken`;
      `_reclassify` in `_local` and `_transit`; `estimate_events`; no affinity, discovery,
      irrelevance or passage machinery.
- [x] `methods/multifloor_policy.py`: `commit()`, `stairs_taken` on the climb, `arrived_by` /
      `arrived_step`, no allowance / storey pull / stairs-first; `floor_decision.py` is the
      fallback rule (`rule: fallback`) sharing `stair_cost_m` with the leaf; `MultiFloorParams
      .floor_change_cost_m`; the way back is never cooled.
- [x] Committed traversal (from the first draft, kept): tight following on a block, 12 blocks /
      30 stalled actions before a retreat, no retreat on the budget clock, `FloorAtlas.settle`.
- [x] `search_panel.py`, `visualization.search_snapshot`, `floor_panels`: rooms AND stair nodes,
      probability and reason, elsewhere, climb cost; `recording.py` live status.
- [x] Tests updated to the design (1545+ pass across the runtime and touched core suites);
      README, MULTISTORY, topology README, QUICKSTART, CHANGELOG, this entry.
- [x] Provision the reasoning model. `qwen2.5:14b-instruct` (9 GB Q4_K_M) pulled into the
      laptop's `ollama-scene-graph` container (CPU-only, 11434); it is the code default. 32B does
      not fit beside Habitat in 30 GB of RAM. `tests/probe_node_oracle.py` against the live model:
      ~25-30 s per call warm, 74 s cold; JSON always well-formed; the first run exposed two prompt
      faults (size read as "bigger is better" -- a toilet's room is small; a known kitchen on the
      other storey undervalued against an unknown room; the few-shot example parroted), fixed in
      the prompt and re-probed.
- [ ] Re-run Ranchester 000000 and one multi-story episode with the 14B; read `estimate_events`,
      the `portal_selected` sources and the `stairs_taken` events against this entry (user
      decides when).
## Open questions
- `floor_change_cost_m` (8 m ≈ 30 actions) is what the Ranchester flights spent per completed
  transition beyond the polyline; a spiral or landing-broken flight would be under-charged by its
  polyline length alone. The right number is a measurement over more flights.
- The oracle's `elsewhere` is recorded but not yet used by the loop; RPT*'s objective does not
  need it. It could gate the exploration fallback ("nothing listed is worth anything, but 60 % is
  elsewhere: go to the frontier") once a run shows the model using it sensibly.
- The classifier still runs on the default (small) model; a wrong single-object name costs one
  reclassify round. Routing it to the reasoning model too is a one-line change if the recordings
  show it mattering.
- `RptStarRoomSolver.max_rooms` (12) caps the nodes per solve; staircases count against it. A
  large floor with three flights leaves nine slots for rooms.
## Notes
- 2026-09-29: The trace probe's first-waypoint executor dithered for forty actions at a route
  whose first point lay behind the agent -- at HEAD as well as after the change. The probe now
  runs the real `DiscreteActionConverter`; the dithering was the probe's, not the loop's.
- 2026-09-29: The first clue-relabel fired on every new landmark and after every watershed
  re-partition: 22 classifier calls in 60 actions. Kinds, not counts, are the clue, and a verdict
  for a set of kinds is reused -- the label is a function of the evidence, not of the region: 5
  calls, identical decisions (LESSONS 2026-09-29).
- 2026-09-29: The first draft's affinity table, passage prior, irrelevant verdict, target-aware
  allowance and stairs-first preference were removed in the same session on the user's review:
  every one of them was the oracle's judgement coded around a 3B model. What survives of that
  draft is the committed traversal, the one-object label with the kind-level cache, the
  `finish`/`is_cooling` supervisor API and the dashboard.
## Result
Implemented and unit-tested; not yet flown with a capable model. The Ranchester re-runs with a
30B-class reasoning model are the next step.
