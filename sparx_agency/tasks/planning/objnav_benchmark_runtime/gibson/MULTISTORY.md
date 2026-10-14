# Multi-story ZSON: persistent floor graphs and stair connections

**Measured status:** [The complete 15-episode campaign](MULTISTORY_RESULTS.md)
ran without runtime errors and produced all five recordings, but scored 0/15
annotated-target successes. Multi-story search quality is not solved.

## Where the stairs come from (`RPTSettings.multifloor.stair_source`)

Two sources, one setting; the atlas, the per-floor contexts and the portal
bookkeeping are the same under both.

**Whether they are taken at all is a separate setting:**
`RPTSettings.allow_stair_traversal`, **false by default** because the
published single-storey benchmarks are scored on the spawn floor
(`methods/spawn_floor_guard.py`). The multi-storey entrypoints lift it
explicitly -- `run_development` for a manifest of this protocol's schema and
`stair_diagnostic` always set `allow_stair_traversal: true` below the policy
config, and the frozen `method` block records the value -- so every command in
this file runs with stairs takeable. A `--policy-config` that sets it false
wins and turns a multi-storey campaign into a guaranteed spawn-floor search.

**`ground_truth` -- the default.** The evaluator reads the scene's navmesh once
per scene (`gibson/stair_connectors.py`): the area-supported storey heights
(the generator's own rule -- 8 m² within ±0.30 m, 1.5 m apart) and, between
them, every cluster of off-level walkable surface as one **connector** -- its
bottom and top floor anchors, the storey heights it joins and the polyline
walked along it (the band's centreline; the navmesh shortest path repairs it
where the centreline is not walkable). The block rides on the episode metadata
(`stair_connectors`, `floor_levels`); it names no goal, no distance and no
target floor, and the harness's privileged-key tripwire accepts it. The
building coordinator then:

- makes every connector touching the storey in force a portal once the camera
  has SEEN it (`stair_sightings.py`; a flight just walked is seen from both
  ends), oriented from that storey (`+1` up / `-1` down; `stair_ground_truth.py`).
  What the oracle is told before any flight is seen is the building's storey
  count and this storey's rank among them (`stair_nodes.storey_position`),
  which comes from the same metadata and is disclosed under `ground_truth_stairs`;
- **offers every portal to the room-search loop as a NODE** (`methods/stair_nodes.py`)
  once its foot is on the observed passable map and it is not cooling after a
  failed approach. The loop hands the staircase to the node oracle beside the
  rooms -- up or down, whether the other storey was visited and what was found
  and searched there, whether the robot arrived by it and how long ago -- and
  the LLM answers the probability that going there next finds the target, in
  the same distribution as the rooms. RPT* then orders rooms and staircases
  together, the staircase sitting at the foot of its flight with a LEAF of the
  flight's length plus `MultiFloorParams.floor_change_cost_m` (8 m, about
  thirty actions -- the climb's turns, the settle and the exit stub) charged
  on every arc into AND out of it, so going upstairs puts every room down
  here that much further away. **Nothing decides a floor change by a clock:**
  `floor_search_actions` is the observed mode's allowance only. When the order
  puts a staircase first the loop calls `MultiFloorSearch.commit`; the
  coordinator approaches the foot of the flight, starts the traversal there
  and tells the loop (`stairs_taken`, verdict `traversed`), so the floor's
  supervisor is clean when the robot comes back down and the way back up is
  a node like any other (`arrived_by` in its facts, no cooldown). Events:
  `portal_selected` with `solver_source: rpt_star`, `traversal_started` with
  `selected_by`;
- **keeps the explicit rule as the exploration fallback's last resort**
  (`floor_decision.py`, `rule: fallback`): asked only when the loop has
  nothing to offer -- the room LLM is away, or no node on the floor is worth
  anything -- and the floor is exhausted. Eligible = not cooling; reachable
  now = entry anchor snaps onto the observed passable map within 1.5 m
  (`snap_entry`: the nearest passable cell with a clear Bresenham line to the
  anchor through what the map knows -- unknown allowed, occupied not; the
  Euclidean-nearest cell of a stair head is as likely in the room behind the
  corridor's wall as in the corridor, and was, in the first recorded
  campaign), else re-checked in 10 actions; worth it = the other storey is
  unvisited, or visited with frontier left; score = worth per metre, worth 1.0
  unvisited / 0.5 visited-with-frontier, cost = the walk to the entry plus the
  same `stair_cost_m` the RPT* leaf uses. Every call is a `floor_decision`
  event naming the winner, its direction, its score and each loser's verdict
  -- one record per distinct verdict. A chosen connector whose approach cannot
  be planned is re-snapped onto the passable map and retried
  `approach_failures` (3) times before it is deferred; an approach that ENDS
  short of the entry (more than 0.65 m from it) is re-snapped on the map as
  it stands from there and extended while that brings the agent nearer
  (`approach_extended` events), so the flight is walked from as close to its
  foot as the observed map allows;
- **walks the connector's own polyline** (`ground_truth_traversal.py`): a
  FOLLOW command on every action, led in from where the agent stands -- the
  first point of every command is the agent's position when its next vertex
  was decided, held fixed until that vertex changes, so the converter's
  forward-only progress runs along the lead-in first and can reach the far
  leg of a folded flight (a U-shaped staircase's top anchor passes within a
  metre of its bottom one in XY) only by walking there; past the far anchor
  to step clear of the stair head; a vertex is reached within 0.26 m in plan
  view and 0.45 m in height (`REACHED_XY_M` / `REACHED_Z_M`; the flight above a
  switchback is a hand's breadth away in plan view); the atlas confirms the
  storey only at the far anchor at the destination height, and a storey first
  reached this way is pinned to the connector's navmesh height (since
  2026-10-05: the poses the atlas settled on lay on the eased last treads, 0.17 m
  above the storey in Hanson, and no staircase seen up there ever became a
  portal). **Once begun, a transition is finished**: a blocked forward step
  first tightens the following to a straight line at the next vertex (no
  lookahead to cut the corner into the banister the shortest path grazes) and
  otherwise falls back to the last vertex reached rather than skipping; turning
  back takes `commit_failures` (12) blocked steps or `commit_stall_actions`
  (30) in which the distance still to walk along the route never shrank by
  4 cm (displacement is not progress: an agent skidding along a wall moves
  every action and gets nowhere), and the transition budget no longer turns
  the agent round -- it lets the atlas settle wherever the agent stands while
  the following goes on; at the destination height by the far anchor, the
  storey is confirmed outright after `confirm_actions` (24) if the atlas's
  translated-plateau test has not fired (`destination_forced` -- a landing
  walled on three sides). A retreat goes back down the vertices walked and
  then off the flight to where the approach ended -- or, when no vertex was
  ever reached, straight to where the traversal began -- and the return is
  settled the action it is reached rather than left to a plateau test an
  agent turning on the spot never passes. A spent retreat never halts the
  episode either;
- **starts an unplanned transition only ON a connector**: a height departure
  beyond `departure_m` (0.45 m) within `near_connector_m` (0.75 m) of a
  polyline completes the climb in the direction already taken; any other
  departure is logged once (`height_departure_ignored`) and left to the atlas.
  The 13 cm raised bathroom floor and the 12 cm wobble at a stair head that
  the observed mode took for first treads never reach the threshold;
- runs **no look-down inspections** and applies **no depth veto** on the
  committed route -- the stairs are known.

`configuration()["ground_truth_stairs"]` is `True` for such a run; it is not an
observed-only run, and its results must say so. RGB stair detections (YOLO
`stairs`/`staircase`) are context labels only, as before.

**`observed`** -- the former RGB-D discovery, kept for comparison
(`{"multifloor": {"stair_source": "observed"}}`). Everything in the next
section describes it.

## Architecture

The building is a **hierarchical graph**: each discovered floor retains the
existing occupancy map, room/door scene graph, LLM beliefs, landmarks, target
verification history, exploration progress and rejection memory. Inter-floor
edges contain actually traversed 3D stair paths. IDs are qualified by floor;
rooms or objects at the same XY coordinates on different floors cannot merge.
Floor IDs are discovery IDs, not surveyed floor numbers.

The detector, revisable room labels, RPT* objective, frontier/FALCON local search,
weighted A*, route commitment and multi-view target confirmation are preserved.
The building coordinator schedules stair portals after bounded floor search or
local exhaustion -- in observed mode with the RPT* solver over observed travel
costs, in ground-truth mode with the explicit rule above. Unknown floors retain
explicit prior probability; a local room ranking does not assert the whole
building has been searched.

Floor changes require a translated, stable-height plateau. Turning on a stair
or crossing successive treads does not allocate a floor. In observed mode new
floors also require at least 3 m² of connected, observed near-level support,
excluding small stair landings; in ground-truth mode the storey heights are
known and the 1.5 m separation rule alone keeps landings from becoming floors.
A return to an existing height restores the saved context and revalidates
routes. Intermediate stair observations do not contaminate floor semantics.
Room clocks pause off-floor; the episode action clock never pauses or refills.

In observed mode stairs are proposed from RGB-D support surfaces, with
organized-depth normals, step-height-limited connectivity and body-height
obstacle clearance. Traversal uses the same MOVE_FORWARD and TURN actions,
optionally inspecting below the horizon with existing LOOK actions (up to 60
degrees down). Near-field support can reuse previously observed free space on
the same floor, never unknown cells. The underfoot height is measured from depth
rather than assuming a fixed offset between a smoothed navmesh base and a
scanned tread. Steering checks legal discrete headings, and retreat commits to a
fixed observed reverse trail. Failed proposals have bounded attempts and
cooldowns. In that mode the policy never receives scene geometry, GT heights,
goal maps, navmesh queries, evaluator distances or success-region entry signals;
in ground-truth mode it receives the storey heights and stair connectors and
nothing else.

Configuration is under `RPTSettings.multifloor` (`enabled: true`,
`stair_source: "ground_truth"` by default).
`{"multifloor": {"enabled": false}}` explicitly selects the historical destructive
floor-reset ablation. Both `--explorer frontier` and `--explorer falcon` support
the building coordinator. The frontier mode remains the default; enabling
multi-story support does not silently replace it with another explorer.

## Honest evaluation boundary

The installed Gibson train assets contain full multi-story geometry, but the
SemExp semantic maps annotate only their reference floor. It would be wrong to
report their planar FMM score as a full-building multi-story benchmark: a
position upstairs can overlap a goal downstairs in XY.

The separate `sparx-gibson-multistory-development/2` protocol therefore:

- selects training buildings with at least two area-supported levels, separated
  by at least 1.5 m, and verifies connected cross-floor starts;
- rejects a start with less than 0.35 m of navmesh clearance
  (`generate_development.MIN_START_CLEARANCE_M`, since 2026-10-05; recorded as
  `start_clearance_m` in the generation audit): the navmesh admits a point
  0.18 m from a bed, and Hanson/000002 spawned there -- the camera saw nothing
  but the bed and the agent could not plan a step out. Manifests frozen before
  this keep their starts; a new manifest is a new frozen set;
- generates all episodes deterministically before the policy runs (no replacing
  difficult buildings based on success);
- starts on another storey from the annotated category's success region;
- requires height-qualified reference-floor goal-region membership for success;
- uses evaluator-only 3D navmesh geodesics to 0.20 m-spaced success-region samples
  and independently cross-checked 3D executed path length for SPL;
- retains the 500-action limit, 0.25 m forward step, 30-degree turns, RGB-D camera,
  native collision handling, and no-STOP-required success convention;
- explicitly permits bounded camera inspection with 30-degree LOOK actions;
- bounds native vertical motion at 0.60 m per forward action, retaining the
  separate planar, turn, pitch and path-length checks. A policy-independent
  audit of 240,000 original-navmesh moves measured a 0.51932 m maximum in
  Coffeen. This simulator-validation bound does not change the policy's 0.24 m
  observed tread limit or the published validation kinematics;
- records the original Habitat 0.2.4 versus reference 0.1.5 mismatch.

**These are development integration results, not published Gibson validation,
not an untouched holdout, and not a complete full-building ObjectNav accuracy
claim.** A real target of the same category on an unannotated floor can be
scored as a failure. Complete semantic annotation is required for that stronger
claim. The original validation scorer and protocol remain available unchanged.

Version 1's 0.30 m vertical bound interrupted the first campaign on a native
0.305 m Coffeen step. That partial attempt is retained, not merged into the
version-2 results. The corrected campaign repeats all fifteen identical starts;
no navigation policy or detector parameters were tuned in response to scores.

## Reproduce five buildings × three episodes

Use the existing Habitat environment and licensed local assets; no download or
GPU model service is required. The detector and room LLM run on CPU. Replace
paths for the workstation and keep the rendering GPU exclusive to Habitat.

Generate the immutable, policy-independent starts:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.generate_development \
  --train-info "$HOME/datasets/objectnav/gibson/objectnav/gibson/v1.1/train/train_info.pbz2" \
  --archive "$HOME/Downloads/gibson_habitat_trainval.zip" \
  --scenes-dir "$HOME/datasets/gibson/multistory/scenes" \
  --output "$HOME/objnav_benchmark/multistory/episodes.json" \
  --multistory --buildings 5 --episodes-per-building 3 --seed 17
```

`--categories couch` (since 2026-10-05) restricts the goals to the categories named
(comma-separated Gibson goal names) and skips a building without one annotated on
its reference floor: a couch is the one category never upstairs in a house, so a
couch-only cross-floor campaign makes the storey change the test rather than a
coincidence of the annotation (the Hanson toilet, chair and plant episodes of
2026-10-05 all ended at real upstairs instances the reference-floor annotation does
not know, SR=0 by construction). Recorded as `generation.goal_categories`.

`--start-storey same` (since 2026-10-05) puts every start on the annotated
reference storey instead of another one -- the SemExp-style same-floor protocol
on this harness, whose 3D geodesic region metrics then score a STOP at an
annotated instance. The 4-60 m geodesic band, the 2 m separation between a
building's starts, the 0.35 m start clearance and the different category per
episode are unchanged; a single-storey building is eligible, and the agent may
still change storeys on its own. Recorded as `generation.start_storey` and in
the run manifest as `start_storey` / `cross_floor_required=false`. The default,
`other`, is the cross-floor protocol above.

After the [detector and CPU LLM setup](README.md#data-and-running), run the
existing frozen campaign runner with one explorer and one preselected recording
per building:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distinct_buildings \
  --manifest "$HOME/objnav_benchmark/multistory/episodes.json" \
  --output "$HOME/objnav_benchmark/multistory/campaign" \
  --explorers frontier --record-first --seed 17 \
  --detector-url http://127.0.0.1:18095 --detector-backend yolo_world \
  --allow-sim-version-mismatch
```

Every job is preflighted and frozen before rendering. Changing source, data,
models or configuration invalidates resume; use a new output directory rather
than mixing attempts. Runs are sequential and retain all episode outcomes.
`run_development --limit 1` is an explicitly labelled smoke subset, not the
15-episode campaign. `--summarize-only` rebuilds the gallery from complete runs.

Outputs:

- `index.html`: the five preselected episode videos (RGB detections, depth,
  active-floor map with the rooms and stair nodes named and valued and the
  RPT* order drawn through them, the search column -- visit order, next node,
  per-node probability and the model's reason, the climb's cost, objects,
  loop events -- route, executed floor trail, floor heights/links);
- `RESULTS.md`, `statistics.json`: every outcome, SPL, action count, vertical
  range, observed floor/connection counts, native collisions and failures;
- `campaign.json`, `locks/`: frozen job, source, model, sensor and data identities;
- per-building `episodes.jsonl`, `evaluation_diagnostics.jsonl` and `run.json`;
- per-recording `video.mp4`, `steps.jsonl`, `trajectory.csv` and final images.

The first episode is selected for recording **before outcomes are known**.
Coverage is an observed occupancy-area proxy, deduplicated by stable floor ID;
transient stair maps do not count as new semantic-floor area. Video and replay
never feed interpolated or evaluator-only information back to the policy.

## Tests and limitations

Run the normal ObjectNav/runtime regressions plus `tests/test_multifloor.py`.
The latter covers height hysteresis, stair landings/turns, return edges,
independent grids and target memory, paused floor clocks, coverage deduplication,
step-connected terrain versus cliffs, height-safe scoring and selective recording.
`tests/test_stair_ground_truth.py` covers the navmesh reading on a synthetic
two-storey house and the policy's orientation of connectors;
`tests/test_ground_truth_floor_transitions.py` covers the fallback rule (the
climb charged, a dearer connector losing at equal worth, the approach sweep),
the stair-node builder (ids, facts, leaf, the way back marked `arrived_by`,
the storey summaries the prompt shows), the coordinator deciding nothing by a
clock, `commit` and the climb ending the node's turn, the re-snapped approach,
the polyline traversal and its commitment (tight following before a skip, many
blocked steps before a retreat, the transition budget no longer turning the
climb round, the forced confirmation at a boxed stair head), the raised-floor
and accidental-descent cases and the arrival that settles a new floor;
`tests/test_room_search_loop.py` covers the staircase as a node of the order
end to end. The observed-mode stair tests in
`tests/test_multistory_repair.py` select `stair_source: "observed"` explicitly.
Actual simulator smoke tests remain necessary: synthetic tests alone cannot
establish navigation success. The connector reading itself was checked against
the real Ranchester navmesh: storeys at 0.05 m and 2.64 m, one connector,
bottom anchor (13.35, 1.56) to top anchor (15.21, 1.77) ENU, 5.5 m long -- the
flight the recorded episodes climbed and descended.

`gibson.stair_diagnostic` runs a bounded, recorded component test from a pose in
an existing observation trajectory. Its optional `--spent-floor-budget` setting
isolates stair selection and is explicitly recorded as a diagnostic override.
These tests are **not** scored campaign episodes. Ranchester traversal was
verified in both directions (0.04 m ↔ 2.64 m) with the actual detector, room LLM,
converter and native collision checks; the intermediate landing remained a
transition. Both recordings are retained separately from the campaign.

Scan holes, narrow stairs, fragmented navmeshes, occluded descending flights,
noisy semantic detections and unannotated target floors remain failure modes.
The implementation does not promise successful navigation in every episode or
claim statistical superiority from fifteen development episodes.



