# Multi-story ZSON: persistent floor graphs and observed stair connections

**Measured status:** [The complete 15-episode campaign](MULTISTORY_RESULTS.md)
ran without runtime errors and produced all five recordings, but scored 0/15
annotated-target successes. Multi-story search quality is not solved.

## Architecture

The building is a **hierarchical graph**: each discovered floor retains the
existing occupancy map, room/door scene graph, LLM beliefs, landmarks, target
verification history, exploration progress and rejection memory. Inter-floor
edges contain actually traversed 3D stair paths. IDs are qualified by floor;
rooms or objects at the same XY coordinates on different floors cannot merge.
Floor IDs are discovery IDs, not surveyed floor numbers.

The detector, revisable room labels, RPT* objective, frontier/FALCON local search,
weighted A*, route commitment and multi-view target confirmation are preserved.
An observed-only building coordinator schedules stair portals after bounded
floor search or local exhaustion. It uses the existing RPT* solver and observed
travel-cost builder. Unknown floors retain explicit prior probability; a local
room ranking does not assert the whole building has been searched.

Floor changes require a translated, stable-height plateau. Turning on a stair
or crossing successive treads does not allocate a floor. New floors also require
at least 3 m² of connected, observed near-level support, excluding small stair
landings. A return to an existing
height restores the saved context and revalidates routes. Intermediate stair
observations do not contaminate floor semantics. Room clocks pause off-floor;
the episode action clock never pauses or refills.

Stairs are proposed from RGB-D support surfaces, with organized-depth normals,
step-height-limited connectivity and body-height obstacle clearance. Traversal
uses the same MOVE_FORWARD and TURN actions, optionally inspecting below the
horizon with existing LOOK actions (up to 60 degrees down). Near-field support
can reuse previously observed free space on the same floor, never unknown cells.
The underfoot height is measured from depth rather than assuming a fixed offset
between a smoothed navmesh base and a scanned tread. Steering checks legal
discrete headings, and retreat commits to a fixed observed reverse trail.
Failed proposals have bounded attempts and
cooldowns. The policy never receives scene geometry, GT heights, goal maps,
navmesh queries, evaluator distances or success-region entry signals.

Configuration is under `RPTSettings.multifloor` (`enabled: true` by default).
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
  active-floor scene graph, route, executed floor trail, floor heights/links);
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
Actual simulator smoke tests remain necessary: synthetic tests alone cannot
establish scan-level stair detection or navigation success.

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



