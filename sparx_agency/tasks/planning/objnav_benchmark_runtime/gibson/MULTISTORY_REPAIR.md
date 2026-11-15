# Multi-story ObjectNav repair — 2026-09-17

## Scope and evidence boundary

The previous complete campaign scored **0/15 annotated-target successes**, SPL 0 and mean final DTG 12.1099 m. Tests and video decoding did not establish navigation success. Its artifacts remain unchanged in `~/objnav_benchmark/multistory-20260917/campaign-v2/`.

This repair preserves observed RGB-D mapping, semantic room graphs, room-language reasoning, RPT* ordering, frontier/FALCON local exploration and path planning. It does not give navigation surveyed heights, scene topology, goal positions, distance fields or navmesh queries. All new attempts, including failed preflights and failed stair runs, are retained in `~/objnav_benchmark/multistory-repair-20260917/`.

The workspace initially had no tracked diff: the earlier implementation was already present in commit `829c154f`; only `.run/` was untracked. No user work was discarded and no commit/push was performed. Baseline and repaired source archives were saved externally. Several editor file views contained stale malformed fragments despite valid on-disk sources; executed-file compilation and tests were used to detect and repair those discrepancies.

## Exact diagnosed failures

| Evidence | Root cause | Repair |
|---|---|---|
| Prior Ranchester steps 88–101 and Hanson 83–96 repeatedly alternate downward inspection and level-view restoration without translating | Inspection identity includes pitch/yaw; ordinary RPT search restores zero pitch between requests | One camera owner; bounded location/floor-keyed inspection with cooldown and explicit restoration |
| Prior Coffeen has 164 stair frames equal to their preceding detection output; adjacent stair-to-next-frame audit finds 165 reused overlay pairs while RGB changes substantially | Stair handling returns before perception; the recorder reuses the previous boxes | Raw predictions run once on every RGB-D/pose observation, including traversal; projection/fusion/STOP are separate |
| Prior Pomaria abandons at step 109, still 0.206 m above the source floor | Retreat completion tests `not atlas.in_transition`, although departure hysteresis is 0.45 m | Commit before the first tread; require actual source-height/platform/entry return, not the departure flag |
| New ascent attempts 1–4 stall near the first tread and sometimes circulate on the source floor | Connector geometry is frozen during approach; near-waypoint steering loses flight alignment; borrowed occupancy follows the moving base height | Refine selected proposal geometry without changing identity/direction; keep measured edges immutable; plan toward connected flight support; anchor borrowed occupancy to its source floor |
| Replay of the failed first tread at step 172 changes from no upward continuation to an observed connected upward route after the source-slab fix | The source-floor 2D slab was incorrectly reapplied to a newly reached tread as base height changed | `stair_grid.py` retains the original floor-height datum; real measured body-height obstacles and the 0.24 m step limit remain enforced |
| New failed retreat revisits the same few XY positions repeatedly | Nearest-pose selection jumps backward around loops in the reverse trail; the trail originally ends too near the first tread | Include the already executed source-platform approach; use a monotonic, height-aware retreat cursor |
| Native source-floor target STOP at action 20 initially raises a command-contract error | Camera arbitration attaches pitch to STOP | STOP carries no LOOK/waypoint/heading command; regression covers both target and safety stops |

The previous videos were fully decoded and analyzed numerically for frame motion and stale overlays. This tooling did not provide visual image inspection to the assistant; no unverified visual classification is presented as ground truth.

## Camera and perception contracts

Positive pitch is down in REP-103. The Habitat bridge checks RGB/depth rotations and positions against each other and the declared camera mount on every synchronous observation. The recorded round trip had matching sensor pitches and a final mount-position residual of about 2e-8 m. The shared optical-to-ENU transform is retained; no compensating sign flip was added.

The camera controller is the only policy pitch writer. Inspection is bounded, deduplicated by observed location/floor, and cannot interrupt a committed transition. Stair pitch is latched; ordinary zero-pitch viewing resumes only after a safe lifecycle boundary. Exact requested/observed pitch and ownership are recorded.

Raw inference is continuous. Coherent depth support is checked at full pixel resolution before sparse projection, preventing stride aliasing and the old percentile-depth/box-centre-ray mismatch. Room/object fusion and target confirmation are quarantined during committed transitions and non-normal views. Geometric support, 3D association consistency, frame deduplication and a floor-qualified target latch precede STOP. Real beds/sofas are not blacklisted near stairs.

`stairs` and `staircase` are scene-independent context prompts, not goal labels or navigation oracles. A semantic hint may request inspection; a portal still requires step-connected geometry. The dedicated CPU detector uses the exact 26-prompt vocabulary. Health/inference provenance pins classes, checkpoint, configuration, packages and prompt features. The checkpoint and confidence thresholds were not changed.

## Transition lifecycle and persistent display

`SEARCH → APPROACH_STAIRS → TRAVERSE → CONFIRM_DESTINATION → SEARCH` preserves connector, direction, progress, source history and destination hypothesis. Small landings and stationary turns do not complete transitions. Retreat is bounded and explicit; failure to verify a safe return is a non-target `SAFE_HALT`, not resumed room search on stairs.

Each observed floor keeps independent occupancy, rooms, landmarks, 3D association anchors, target evidence and search history. Floor clocks pause during commitment; the global allowance never refills.

There is **one map per storey**, not K extra layers inside each storey. The recorder reserves K independent UNKNOWN grids using only a display count. Unvisited slots remain gray and have no inferred height or topology. Discovery IDs bind those slots; all panels use the same fixed observed-origin viewport and scale. Full grids and metadata are retained in `floor_maps.npz` and `floor_panels.json`; display counts never enter policy settings or observations.

## Validation before freezing

**1,722 tests passed** on the final candidate, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python`. Tests cover camera/STOP ownership, native frame conversion, continuous raw perception, mixed depth, wrong-floor support, 3D association, landing rejection, approach updates, source-slab height, committed recovery, loop-free retreat, context persistence and gray unknown panels. They are contracts, not navigation scores.

Native Ranchester component results are deliberately separate from scored episodes:

- Original toilet-query ascent/start: ordinary policy attempts a target STOP on the source floor at action 20. It does **not** demonstrate ascent. Its result and earlier preflight errors are retained.
- Bed-query ascent/return attempts 1–4 failed; none is replaced or relabelled successful.
- **Bed-query attempt 5 completed ascent and return without any spent-budget override**: portal selected at 140; traversal starts at 167; upper platform confirmed at 246; return traversal starts at 278; source platform restored at 333. There were **334 emitted actions and 13 native collisions**. Heights were approximately 0.039 m → 2.639 m → 0.039 m. Floor 0 has two visits; one observed edge has two traversals.
- The round-trip audit confirms fresh perception for every decision and unchanged inactive/transition maps, with zero detected grid-integrity errors. The departure observation is legitimately integrated before the next action commits to traversal.
- Separate ordinary-budget descent runs also completed the platform exit in 50 decisions, with two native collisions; the final round trip independently covers descent on the frozen candidate.

These are pose-controlled development component tests, not a claim of reliable discovery from arbitrary starts, independently verified ObjectNav success, or low-collision navigation.

## Frozen campaign and limitations

The next campaign uses the unchanged original `episodes-v2.json`: Ranchester, Pomaria, Hanson, Coffeen and Woodbine, three episodes each. It selects frontier explicitly and the first episode per building for recording before outcomes. Code, models, prompts, runtime, configuration, starts and scoring are frozen before all jobs. No tuning is allowed during the batch.

The protocol remains **`sparx-gibson-multistory-development/2`**. Native vertical validation stays **0.60 m**, based on the earlier independent audit; observed traversable tread height stays **0.24 m**. Neither bound is loosened or conflated.

Gibson annotations cover only a reference floor. A real target on another floor may be unannotated; the policy detector cannot certify it as ground truth. HM3D episode definitions are installed, but fully annotated scene geometry is not. Therefore this campaign remains a development stress test, not a complete multi-story ObjectNav benchmark. Its complete outcomes and recording checks belong in the campaign artifacts; until they finish, no navigation-improvement claim is justified.

## OSG Navigator research conclusion

**Autonomous cross-floor ObjectNav is not established by the auditable evidence, but it is also not proven absent.**

- arXiv 2508.04678v1 **§7.1.1 p.11** describes multi-floor benchmark scenes and aggregate episode counts. That is not an episode-level cross-floor result.
- **§7.7 pp.14–15, Table 7** evaluates graph construction. **Figure 8 p.16 explicitly uses teleoperated trajectories.**
- **p.12** says simulation variants use a standardized FMM local planner; **§7.2 p.12 and §7.10 p.18** describe autonomous real-robot tasks, without a reported stair-completion breakdown.
- The primary PDF, abstract and arXiv source/appendices did not identify an OSG implementation or its exact 400 selected HM3D episode IDs. An unused video macro has uncertain provenance and its Drive link returned HTTP 401; unseen supplementary footage is not evidence.
- Independent height checks of the public reference data found all 1,000 Gibson starts within 0.49571 m of their annotated goal floor. In contrast, **341 of 2,000 HM3D v1 validation episodes** have all category-goal viewpoint heights at least 1.5 m from the start. These are this audit's population measurements, not OSG's reported results; the selected 400 IDs remain unknown.

Detailed page/file citations and missing-evidence qualifications are in `~/papers/paper-2508-04678-detectors/SUMMARY.md` and `INTEGRATION.md`. The public reference-code and height audits cannot be substituted for OSG's unreleased or unlocated implementation/episode list.

