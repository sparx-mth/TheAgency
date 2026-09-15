# MP3D protocol audit, and what a comparison does and does not establish

This file records where each protocol number comes from, what the comparison
papers do and do not publish, and the rules for producing a full-split number.
It is the place to look before quoting an MP3D figure anywhere.

## Sources for every protocol field

| Field | Value | Source |
| --- | --- | --- |
| Episodes / scenes / categories | 2,195 / 11 / 21 | SG-Nav arXiv:2410.08189 p.7; ApexNav arXiv:2504.14478 Sec. V-A |
| Step budget | 500 | `ENVIRONMENT.MAX_EPISODE_STEPS`, habitat-lab `objectnav_mp3d.yaml`; challenge 2021 |
| Forward / turn / tilt | 0.25 m / 30 deg / 30 deg | habitat-lab `FORWARD_STEP_SIZE` default; `TURN_ANGLE`, `TILT_ANGLE` in the MP3D config |
| Body height / radius | 0.88 m / 0.18 m | `SIMULATOR.AGENT_0.HEIGHT` / `RADIUS` |
| RGB-D | 640x480, HFOV 79, at `[0, 0.88, 0]` | `RGB_SENSOR` / `DEPTH_SENSOR` in the MP3D config |
| Depth range | 0.5 - 5.0 m | `DEPTH_SENSOR.MIN_DEPTH` / `MAX_DEPTH` |
| Sliding | off | `HABITAT_SIM_V0.ALLOW_SLIDING: False` |
| Success distance | 0.1 m | `TASK.SUCCESS.SUCCESS_DISTANCE` |
| Distance measured to | view points | `TASK.DISTANCE_TO_GOAL.DISTANCE_TO: VIEW_POINTS` |
| Action space | 6 actions | `TASK.POSSIBLE_ACTIONS` |
| Navmesh | recomputed for the agent | habitat-sim `Simulator._config_pathfinder`; habitat-lab `create_sim_config` copies HEIGHT/RADIUS |
| SR / SPL / SoftSPL / DTG | see below | `habitat/tasks/nav/nav.py` classes `Success`, `SPL`, `SoftSPL`, `DistanceToGoal` |

The three config files agree with each other; v0.2.4 restates the same values
in Hydra form. They were read from the habitat-lab and habitat-challenge
repositories at their `v0.2.2`, `v0.2.4` and `challenge-2021` tags, not from
memory.

### The metric definitions, as habitat-lab implements them

* `l = d0 = DistanceToGoal` at reset: `sim.geodesic_distance(start,
  [view_point.agent_state.position for goal in goals for view_point in
  goal.view_points])`.
* `p`: the sum of 3-D Euclidean chords between successive agent positions,
  from the reset position, accumulated on **every** action - turns, tilts, a
  blocked MOVE_FORWARD and STOP all add whatever the agent actually moved,
  which for most of them is zero.
* `Success = is_stop_called and dT < success_distance` (strict `<`).
* `SPL = Success * l / max(l, p)`; `SoftSPL = max(0, 1 - dT/d0) * l / max(l, p)`.

`DistanceToGoal` recomputes only when the agent has moved more than `1e-4`;
that is a cache, not a different value.

## Known deviations from the reference implementation

Stated here rather than buried in code comments. None of them changes the
success rule or the distances; each is a deliberate, recorded choice.

1. **No habitat-lab.** The task framework is not installed; the evaluator is
   reimplemented against the definitions above and cross-checked against the
   harness's independent recomputation on every episode. A disagreement fails
   the episode instead of being scored.
2. **Depth clipping.** habitat-lab clamps depth into `[min, max]`. We mark
   below-min as `NaN` (no reading) and beyond-max as `+inf` (no surface in
   range), which is what our mapping needs and is strictly more conservative
   than clamping a far reading to 5 m of solid evidence. The camera the policy
   sees is otherwise identical.
3. **Simulator version.** The comparison papers do not all pin a habitat-sim
   version. This workstation runs 0.2.4, which `PROTOCOL.reference_sim_version`
   records; preflight refuses any other version unless
   `--allow-sim-version-mismatch` is passed, and the run configuration keeps
   `reference_sim_version_match` either way.
4. **Semantic sensor.** The published config lists a `SEMANTIC_SENSOR`. We do
   not instantiate it: the method must not see ground-truth semantics, and no
   baseline in the comparison tables uses it either.
5. **`stop_distance_m = 1.0` and `map_size_m = 100`** are method parameters,
   not protocol. See the README's "Method settings".
6. **Navmesh.** We follow habitat-lab: the navmesh habitat-sim derives for the
   0.18 m / 0.88 m agent, not the shipped 0.1 m / 1.5 m file. This is recorded
   as `navmesh_source` and was not the adapter's first behaviour - the shipped
   file gives 58.006 m² of navigable area on Collierville against the
   recomputed 44.145 m², with 14 of 39 sampled point pairs reachable only on
   it, so the two produce different SR and different SPL. See the README's
   "Which navmesh the distances are measured on".

## What the comparison does and does not establish

**Does.** Both papers evaluate the same published split with the same
habitat-lab evaluator, and report SR and SPL. Our run uses that split, that
evaluator's rules and that action geometry, so the numbers are comparable in
the way any two papers' numbers are comparable.

**Does not.**

* Neither paper publishes per-episode results, its evaluator's hashes, or its
  exact habitat-sim build, so exact reproduction of *their* rows is not
  established and no direct SOTA claim is asserted by this package.
* ApexNav states "a success distance of 0.2 m" (Sec. V-A) where the published
  config says 0.1 m. We follow the published config. If that sentence means
  what it says rather than being a description of their stopping policy, their
  MP3D SR is measured under a laxer rule than ours, in their favour.
* The two papers disagree on VLFM's MP3D row (36.2/15.9 vs 36.4/17.5). Both
  are carried with their sources; neither is corrected against the other.
* DTG, TF and NM come from *Open Scene Graphs* (arXiv:2508.04678), which
  reports them for HM3D and Gibson only. There is no published MP3D DTG
  baseline. Ours is our own supplementary number.
* A configuration lock is a reproducibility commitment. It is **not** evidence
  that validation never influenced development, and it never creates a
  held-out claim.

## Freeze, then evaluate

A full-split number is produced by freezing the exact configuration before the
run and checking it again at execution, not by running and reporting.

```bash
M=sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d

python -m $M.frozen_eval freeze --lock ~/objnav_benchmark/mp3d/val.lock \
  --note "Development used <describe every MP3D episode ever run>." -- \
  --output ~/objnav_benchmark/mp3d/val

python -m $M.frozen_eval run --lock ~/objnav_benchmark/mp3d/val.lock -- \
  --output ~/objnav_benchmark/mp3d/val
```

The lock pins the actual Python sources, the policy and converter settings, the
method and model identity, the runtime versions, the seed, the data hashes, the
motion tolerances and the exact ordered list of all 2,195 episodes. `run`
re-checks that configuration against the prepared execution before the output
directory is created or the environment is reset, and supplies the publisher's
ordered episode identities as `expected_episode_ids`, so a subset cannot be
written into a frozen run. `--note` is mandatory and is stored in the lock;
`held_out_claim` stays false.

Freezing refuses `--scene`, `--limit` and sharding, and refuses a release whose
episode count is not the published 2,195.

## Tuning discipline

Do not tune against repeated full-validation feedback. MP3D ships a 61-scene
training split for that; synthetic scenarios and the existing corridor fixtures
cover the mechanics. Every parameter that was chosen with any MP3D validation
episode in view must be named in the freeze `--note`, in plain words, including
episodes run "just to see if it works".
