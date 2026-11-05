# MP3D protocol audit, and what a comparison does and does not establish

This file records where each protocol number comes from, what was measured on
the installed release, what the comparison papers do and do not publish, and
the rules for producing a full-split number. It is the place to look before
quoting an MP3D figure anywhere.

## Sources for every protocol field

| Field | Value | Source |
| --- | --- | --- |
| Episodes / scenes / categories | 2,195 / 11 / 21 | the installed `objectnav_mp3d_v1` val split (counted 2026-11-04); SG-Nav arXiv:2410.08189 p.7; ApexNav arXiv:2504.14478 Sec. V-A |
| Step budget | 500 | `ENVIRONMENT.MAX_EPISODE_STEPS`, habitat-lab `objectnav_mp3d.yaml`; challenge 2021 |
| Forward / turn / tilt | 0.25 m / 30° / 30° | habitat-lab `FORWARD_STEP_SIZE` default; `TURN_ANGLE`, `TILT_ANGLE` in the MP3D config |
| Body height / radius | 0.88 m / 0.18 m | `SIMULATOR.AGENT_0.HEIGHT` / `RADIUS` |
| RGB-D | 640×480, HFOV 79°, at `[0, 0.88, 0]` | `RGB_SENSOR` / `DEPTH_SENSOR` in the MP3D config |
| Depth range | 0.5 – 5.0 m | `DEPTH_SENSOR.MIN_DEPTH` / `MAX_DEPTH` |
| Sliding | off | `HABITAT_SIM_V0.ALLOW_SLIDING: False` |
| Success distance | 0.1 m | `TASK.SUCCESS.SUCCESS_DISTANCE` |
| Distance measured to | view points | `TASK.DISTANCE_TO_GOAL.DISTANCE_TO: VIEW_POINTS` |
| Action space | 6 actions | `TASK.POSSIBLE_ACTIONS` |
| Navmesh | recomputed for the agent (`navmesh="agent"`) | habitat-sim `Simulator._config_pathfinder`; habitat-lab `create_sim_config` copies HEIGHT/RADIUS; **measured below** |
| SR / SPL / SoftSPL / DTG | see below | `habitat/tasks/nav/nav.py` classes `Success`, `SPL`, `SoftSPL`, `DistanceToGoal` |

The three config files agree with each other; v0.2.4 restates the same values
in Hydra form. They were read from the habitat-lab and habitat-challenge
repositories at their `v0.2.2`, `v0.2.4` and `challenge-2021` tags, not from
memory. The whole `MP3DProtocol` dataclass is serialised into every run's
`run.json` and into the configuration lock; changing a field is a different
experiment with a different `protocol_id`.

### The metric definitions, as habitat-lab implements them

* `l = d0 = DistanceToGoal` at reset: `sim.geodesic_distance(start,
  [view_point.agent_state.position for goal in goals for view_point in
  goal.view_points])` — a single `MultiGoalShortestPath` over every view point
  of every goal of the (scene, category), reused for the episode
  (`hm3d/geodesic.py`, shared with HM3D because it is habitat-lab's measure).
* `p`: the sum of 3-D Euclidean chords between successive agent positions,
  from the reset position, accumulated on **every** action — turns, tilts, a
  blocked MOVE_FORWARD and STOP all add whatever the agent actually moved,
  which for most of them is zero.
* `Success = is_stop_called and dT < success_distance` (strict `<`).
* `SPL = Success · l / max(l, p)`; `SoftSPL = max(0, 1 − dT/d0) · l / max(l, p)`.

`DistanceToGoal` recomputes only when the agent has moved more than `1e-4`;
that is a cache, not a different value.

## The navmesh, measured on the installed release (2026-11-04)

habitat-sim auto-loads `<scene>.navmesh` and **recomputes** it whenever its
recorded agent settings differ from the agent's; habitat-lab's ObjectNav agent
is 0.18 m / 0.88 m and the shipped Matterport meshes are not. Both were
measured here with habitat-sim 0.2.4 against the publisher's own
`info.geodesic_distance`, which is the same quantity as `l`:

| Navmesh | Agent settings recorded in it | Navigable area, 2azQ1b91cZZ | Starts reachable | \|l − published\| |
| --- | --- | --- | --- | --- |
| shipped `<scene>.navmesh` | radius 0.10 m, height 1.50 m | 317.8 m² | 431 of 440 sampled | median 0.176 m, p90 0.91 m, max 18.1 m; 4 within 1 mm |
| recomputed at 0.18 m / 0.88 m (`navmesh="agent"`) | radius 0.18 m, height 0.88 m, max climb 0.20 m | 257.3 m² | **2,195 of 2,195** | **median 3 µm; 2,191 of 2,195 within 1 mm (99.8 %)**; max 0.08 m |

So the publisher generated the episodes on the recomputed mesh, and only that
mesh reproduces its numbers; the shipped file is a different, more permissive
world (more area, more connections, and nine starts whose goals it cannot
reach at all). `navmesh="agent"` is therefore the protocol, not a preference,
and `run_benchmark.sh` re-checks the agreement scene by scene before any
episode runs (`VALIDATE_STARTS=1`; `run.py` refuses a scene below 99 % within
1 mm, with the small-scene allowance HM3D's gate carries). The four starts off
by up to 8 cm sit on a view point, where float32 snapping puts the two numbers
a hair apart — the same effect HM3D's sweeps showed.

## Known deviations from the reference implementation

Stated here rather than buried in code comments. None of them changes the
success rule or the distances; each is a deliberate, recorded choice.

1. **No habitat-lab.** The task framework is not installed; the evaluator is
   reimplemented against the definitions above and cross-checked against the
   harness's independent recomputation on every episode (the `native_metrics`
   block). A disagreement fails the episode instead of being scored.
2. **Depth clipping and the depth-hole fill.** habitat-lab clamps depth into
   `[min, max]`. We mark below-min as `NaN` (no reading) and beyond-max as
   `+inf` (no surface in range), which is what our mapping needs and is
   strictly more conservative than clamping a far reading to 5 m of solid
   evidence. Enclosed no-return pixels (a mirror, a window, a black screen —
   Matterport scans have many) are filled from their rim under the same rule
   as on Gibson and HM3D (`depth_hole_*`); open voids are dropped as every
   baseline drops them. Both change only what the policy sees, never what the
   evaluator measures.
3. **Simulator version.** The comparison papers do not all pin a habitat-sim
   version. This workstation runs 0.2.4, which
   `MP3DProtocol.reference_habitat_sim_version` records; every run writes
   `reference_habitat_sim_version_match` into its configuration, and the audit
   repeats it.
4. **Semantic sensor.** The published config lists a `SEMANTIC_SENSOR`. A
   service run does not instantiate it: the method must not see ground-truth
   semantics, and no baseline in the comparison tables uses it either. The
   `--perception ground-truth` upper bound does (640x480, HFOV 79, at 0.88 m,
   as published), through the mpcat40 goal table
   `../ground_truth/mp3d_categories.py`; such a run is non-publishable.
5. **Method parameters are not protocol.** Body height and radius come from the
   protocol; every other knob (clearance, stop distance, closing, peeks, the
   explorer) is `RPTSettings`' default, exactly as on Gibson and HM3D. The
   method's room-type priors know the six HM3D/Gibson categories; MP3D's other
   fifteen run without a type prior (the node oracle still answers), which is a
   method matter and is recorded in each run's `method` block.
6. **Kinematic self-check.** `climb_m` 0.85 m and `reproject_m` 0.15 m are
   HM3D's bounds, derived from habitat-sim's step filter and the navmesh it
   recomputes at this agent (cell height 0.20 m, max climb 0.20 m, 45° slope
   over a 0.25 m stride), not from either dataset. Matterport's buildings have
   more open stairwells and split levels than HM3D's, so the bound is exercised
   at least as much here; a storey or a teleport is still refused.

## What the comparison does and does not establish

**Does.** Both papers evaluate the same published split with the same
habitat-lab evaluator, and report SR and SPL. Our run uses that split, that
evaluator's rules and that action geometry, on the navmesh that reproduces the
publisher's own geodesics, so the numbers are comparable in the way any two
papers' numbers are comparable.

**Does not.**

* Neither paper publishes per-episode results, its evaluator's hashes, or its
  exact habitat-sim build, so exact reproduction of *their* rows is not
  established and no direct SOTA claim is asserted by this package.
* ApexNav's released configs set a success radius of 0.2 m where the published
  config says 0.1 m. We follow the published config and **re-score the same
  rows at 0.2 m** (`report.py`), so a like-for-like row exists under their
  table; the re-scoring can only add successes.
* The two papers disagree on VLFM's MP3D row (36.2/15.9 vs 36.4/17.5). Both
  are carried with their sources (`sota.py`); neither is corrected against the
  other.
* SG-Nav's baseline rows are quotes from the originating papers; the stacks
  and radii behind them are those authors'.

## Freeze, then evaluate

A number intended for publication is produced under `frozen_eval.py`: the lock
captures the sources, the built policy and its converter, the dataset manifest,
the runtime versions, the protocol and the ordered selection, and `run` refuses
any drift and any selection but the complete 2,195-episode split. The
mandatory `--note` records what the configuration had already been developed
against; a lock is a reproducibility commitment, not evidence that the split
was unseen.

## Tuning discipline

Nothing in this package is tuned on MP3D. The method arrives with the settings
it has on Gibson and HM3D; the protocol is habitat-lab's; the only MP3D-specific
inputs are the published data, the 21-category label table and its context
vocabulary (which must not contain a goal word — `cabinet`, `desk`, `shower`,
`bathtub`, `stool` are context on HM3D and goals or goal prompts here). A
development run uses `--split train` or `val_mini` with `--episodes-per-scene`,
is labelled a subset everywhere, and is never averaged into a val number.
