# Multi-story ZSON: persistent floor graphs and stair connections

**Measured status:** [The complete 15-episode campaign](MULTISTORY_RESULTS.md)
ran without runtime errors and produced all five recordings, but scored 0/15
annotated-target successes. Multi-story search quality is not solved.

## Where the stairs come from (`RPTSettings.multifloor.stair_source`)

**From the robot's own depth, and nowhere else.** `stair_source` is
`"observed"` -- the default since 2026-10-21 and the only source since
2026-11-15. The method's stairs are what its own RGB-D measures
(`world_model/stair_terrain.py`: a reachable run of support surfaces rising
`stair_min_rise_m`, 0.40 m, within 5 m in a flight's profile --
`stair_min_risers` (3) risers of `stair_riser_min_m` (0.10) to `max_step_m`
with no tread between them deeper than `stair_max_tread_m` (0.55),
`stair_profile`, since 2026-10-22 -- is a portal of the storey, placed from
the measurement and masked on the planning grid by the same capsule as any
seen flight); no simulator geometry is read, which is the footing of ApexNav,
SG-Nav and OSG-Nav. Everything in the "Architecture" section describes it.

**Whether they are taken at all is a separate setting:**
`RPTSettings.allow_stair_traversal`, **false by default** because the
published single-storey benchmarks are scored on the spawn floor
(`core/planning/object_search/multifloor/spawn_floor_guard.py`). The multi-storey entrypoints lift it
explicitly -- `run_development` for a manifest of this protocol's schema and
`stair_diagnostic` always set `allow_stair_traversal: true` below the policy
config, and the frozen `method` block records the value -- so every command in
this file runs with stairs takeable. A `--policy-config` that sets it false
wins and turns a multi-storey campaign into a guaranteed spawn-floor search.

With traversal allowed, a floor change is the building coordinator's decision
(`MultiFloorSearch.plan`): once a floor's `floor_search_actions` allowance is
spent, or at once when the exploration fallback reports the floor exhausted,
RPT\* over the seen portals picks one (`_select`); the approach is planned on
the observed map, the flight is walked on measured tread support
(`stair_traversal.py`), and look-down inspections look for stairs at natural
pauses (`_inspect`).

### Removed on 2026-11-15: the `ground_truth` stair source

Until then `{"multifloor": {"stair_source": "ground_truth"}}` selected an
ablation in which the evaluator read each scene's navmesh
(`habitat/stair_connectors.py`: storey heights and, between them, every
stair connector with its anchors, its centreline and a sample of its
surface), attached that block to `ObjNavEpisode.metadata`, and the building
coordinator used it: a "perfect stair detector" made a connector a portal once
its surface was in frame, the node oracle was told how many storeys the
building has and this storey's rank among them, every such portal was a NODE
of the room-search loop's RPT\* instance, the fallback had an explicit up/down
rule (`floor_decision.py`), and the climb followed the connector's polyline
(`ground_truth_traversal.py`). It was the default from its introduction (progress
entry 012) until 2026-10-21.

It was privileged scene geometry -- no published zero-shot ObjectNav method is
told where the stairs are or how many storeys a building has -- and it is
gone: the modules, the settings that only it read (`near_connector_m`,
`commit_failures`, `commit_stall_actions`, `confirm_actions`), the adapters'
metadata, the storey count and rank in the prompt. Asking for it raises
(`ValueError: ... was removed`); the harness refuses an episode whose metadata
carries storeys, stairs or a navmesh; and
`core/planning/object_search/tests/test_no_privileged_geometry.py` fails if
`core` can reach any of it again. What this leaves, and what it does not:

- **`habitat/stair_connectors.py` stays, for analysis only** -- the
  after-the-fact breakdown of results by the storey a goal is on
  (`hm3d/episode_levels.py`). It is never handed to a policy.
- **Staircases as nodes have no stair source.** The loop's side
  (`rooms/stair_nodes.py`, `MultiFloorSearch.commit`, the way-back hold, the
  leaf of flight length plus `floor_change_cost_m`) is kept and tested on
  hand-placed portals, but a staircase is offered only when its portal says
  which storey the flight leads to, and the observed discovery measures a
  flight's treads, not where it ends. Giving the observed source that
  knowledge -- from the flight's measured rise, or from a storey once it has
  been stood on -- is the open work.
- **Results recorded with it stay what they were.** The six tracked Gibson
  validation records of 2026-10-08 to 2026-10-18 (`gibson/results/*/run.json`:
  `"stair_source": "ground_truth"`, `"ground_truth_stairs": true`) were run
  while that source was the default: the staircases their search knew and
  masked were the navmesh's connectors, not flights the robot had measured.
  They cannot be reproduced with this code; read them as such. The 15-episode
  campaign of [MULTISTORY_RESULTS.md](MULTISTORY_RESULTS.md) (2026-09-17)
  predates the source and never used it.

RGB stair detections (YOLO `stairs`/`staircase`) are context labels only, as
before.

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
local exhaustion, with the RPT* solver over observed travel costs. Unknown floors retain
explicit prior probability; a local room ranking does not assert the whole
building has been searched.

Floor changes require a translated, stable-height plateau. Turning on a stair
or crossing successive treads does not allocate a floor. New floors also
require at least 3 m² of connected, observed near-level support, excluding
small stair landings: nobody tells the atlas the storey heights.
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
cooldowns. The policy never receives scene geometry, GT heights, goal maps,
navmesh queries, evaluator distances or success-region entry signals.

Configuration is under `RPTSettings.multifloor` (`enabled: true`,
`stair_source: "observed"`, the only source; `"ground_truth"` raises).
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
two-storey house (the analysis module, nothing a policy sees);
`tests/test_stair_nodes.py` covers the loop's side of staircases as nodes on
hand-placed portals -- ids, facts, the leaf, the approach point that must see
the entry, the way back marked `arrived_by`, `commit` and the climb ending the
node's turn -- and that a flight the robot only measured is not a node and
`stair_source: "ground_truth"` is refused; `tests/test_room_search_loop.py`
covers the staircase as a node of the order end to end on the same portals.
The observed stair tests are in `tests/test_multistory_repair.py`,
`tests/test_multifloor.py` and `tests/test_spawn_floor_guard.py`.
Actual simulator smoke tests remain necessary: synthetic tests alone cannot
establish navigation success. The navmesh reading itself (the analysis
module) was checked against the real Ranchester navmesh: storeys at 0.05 m
and 2.64 m, one connector, bottom anchor (13.35, 1.56) to top anchor
(15.21, 1.77) ENU, 5.5 m long -- the flight the recorded episodes climbed and
descended.

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



