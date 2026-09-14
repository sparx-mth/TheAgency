# Shared ObjectNav runtime infrastructure

This is the common base for simulator feature branches. It composes the existing
scene-graph/LLM/RPT* search with the [shared benchmark harness](../objnav_benchmark/README.md).
It contains **no published dataset loader, episode list, category table, scoring
map, benchmark-specific launch UI, or built-in simulator benchmark profile**.
The search algorithm has not been replaced, and FALCON is not launched here.

## What is shared

| Component | Responsibility |
| --- | --- |
| `methods/rpt_policy.py` | Observed-map exploration, LLM room probabilities, RPT* room ordering, weighted A* and committed route execution |
| `methods/observed_map.py` | RGB-D to robot-height local 2.5D occupancy, conservative free-space integration and floor-change reset |
| `methods/route_memory.py` | Safe route retention, executor-consistent snapped arrival and stagnation tracking across safety replans |
| `methods/object_evidence.py` | Alias/box deduplication, distinct-frame/multi-view evidence and landmark-specific verification/cooldown |
| `methods/doors.py`, `room_labels.py`, `scene_graph.py` | Depth-backed doors, observed room segmentation and revisable evidence-based room labels |
| `methods/perception.py`, `services.py` | Existing detector/LLM clients with model/vocabulary provenance and service-drift checks |
| `habitat/simulator.py` | Optional RGB-D/action/frame bridge reusable by Habitat-backed adapters |
| `evaluation.py` | One shared prepared-environment execution path, optional recording, resume and pre-execution lock checks |
| `provenance.py` | Actual-source fingerprint, deterministic episode sharding and generic exact-configuration locks |
| `recording.py`, `visualization.py`, `dashboard.py` | Observations, trajectories, per-step decisions, videos and benchmark-neutral result dashboards |
| `replay.py` | Separate smooth video using an adapter-provided pose renderer; never policy/scoring input |

Shared core improvements include nearest/frame-aware object landmarks, optional
count-sensitive room classification and lazy OMPL imports. Ordinary A*/ObjectNav
must not load unused native sampling planners; explicit OMPL users retain the
existing public interfaces and missing-dependency errors.

## Adapter boundary — keep benchmark rules out of shared code

Each simulator branch supplies:

1. An `ObjNavEnv` implementation: published episode loading, reset/step, episode
   termination and evaluator-only measurements. `measure()` is not called early.
2. Exact scene/split identities and a `TableLabelMapper` with that dataset's
   category spellings, detector prompts and accepted aliases. The infrastructure
   registry does not claim any dataset is implemented.
3. Explicit camera intrinsics, registered metric depth, body dimensions, action
   lengths/angles, sliding policy, step budget and protocol/version metadata.
4. Its published success/STOP/distance/path-accounting rules. Shared defaults
   remain **STOP required, 3D observed path length, zero initial accumulator**.
   Override `EvaluationSettings` only when the adapter's protocol requires it.
5. Data/model provisioning and checks, including GPU ownership. No weights,
   licensed scenes, containers or remote services are downloaded/launched by the
   shared evaluation entrypoint.
6. A manifest identifying the actual dataset/assets, model checkpoint/revision,
   package versions, seed and protocol. Keep credentials out of this metadata;
   the shared service URL validator rejects embedded credentials/query strings.

MP3D, HM3Dv1 and HM3Dv2 can reuse the Habitat bridge. The HM3D versions should
share an adapter implementation where their schemas permit it, with separate
versioned data/protocol configurations. RoboTHOR needs an AI2-THOR environment
bridge but reuses the policy, action conversion, provenance and recording.
Neither its adapter nor the new Habitat dataset adapters are implemented here.

### Explicit embodiment and method settings

`RPTSettings` requires `body_height_m`, `body_radius_m`,
`preferred_clearance_m` and `stop_distance_m`. They are not inferred from a
benchmark name or silently inherited from the first simulator. Physical
inflation is never relaxed below the supplied body radius. The stop distance
is a method choice, not an override of the evaluator's success definition.

`RPTSearchPolicy` accepts optional converter, route, target-evidence, door and
room-label settings and records them in `configuration()`. `run_evaluation`
uses the same converter settings in the actual headless agent, so route
arrival/progress checks agree with the executor.

`HabitatRGBDSimulator` separately requires body height and radius; camera mount
height is not body height. It loads the supplied navmesh, never recomputes a
published navmesh or snaps a published start. Only the explicitly synthetic
renderer smoke constructs its own fixture navmesh.

The current search commands and map remain planar/2.5D. Floor changes reset
local state; this is not complete multi-floor exploration or a claim that all
3D simulator behavior has been solved.

## Shared execution and locks

Construct the environment, policy, label mapper and adapter metadata, then call
`run_evaluation` with an explicit output directory. It uses the existing
`run_benchmark`/`MetricsLogger`, not a second scoring loop, and closes the
environment even on recording failures. Extra `evaluation_diagnostics()` is an
optional recorder-only hook; environments without it remain recordable.

For frozen evaluation:

1. Use `evaluation_configuration(config, episode_ids, settings, policy=policy)`
   to capture actual Python sources, the policy, converter, selected episode
   order and execution settings alongside adapter metadata.
2. Call `freeze_configuration` with a new lock path and an explicit
   `development_note` describing prior tuning/exposure. Existing locks are not
   overwritten.
3. Pass that lock to `run_evaluation(frozen_lock=...)`. It checks the actual
   execution configuration before output creation or reset, not merely an
   earlier preflight. To claim a complete published split, also supply the
   publisher's exact ordered `expected_episode_ids`.

There is no hardcoded scene count or episode count. A lock is a reproducibility
commitment, **not proof that validation was unseen**. It never creates a
held-out or SOTA claim. Full benchmark equivalence, tuning disclosure and
reference-version caveats remain the adapter's responsibility. Dashboards only
summarize recorded rows and do not label arbitrary subsets as complete splits.

## Recording and visual-only replay

Recording is optional and loads plotting/video dependencies only when requested.
It keeps images/maps out of per-step JSON; the JSON records compact observed
state, actions, predictions, paths and reasoning. Model service metadata pins
checkpoint bytes and vocabulary atomically with inference. Missing or changed
services raise rather than silently changing the evaluated method.

`render_replay(recording_dir, render_pose, ...)` takes a callback receiving an
`AgentPose` and returning uint8 RGB. The adapter owns scene loading, GPU safety
and camera-pitch handling. Replay interpolates translation/pitch and the shortest
yaw arc, includes the final pose, preserves aspect ratio, and writes only a
separate `smooth_rgb.mp4`. It does not alter original scores or observations.

## Dependencies and validation

`requirements.txt` lists the shared CPU/scientific extras. Core contracts and the
lightweight harness remain free of simulator/model imports. Install simulator
bindings in their dedicated environments; do not install torch into the
lightweight test environment. Recording requires a working FFmpeg/libx264;
`IMAGEIO_FFMPEG_EXE` may select an already installed executable.

From the checkout root with the test interpreter selected:

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/mapping/objects/tests sparx_agency/core/mapping/topology/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark \
  sparx_agency/core/planning/planners/common/tests \
  sparx_agency/core/planning/planners/astar/tests -q
```

Tests use explicit synthetic geometry and the existing fake/corridor fixtures,
not a real dataset's constants. They cover route lifecycle, physical clearance,
object/room evidence, optional imports, explicit protocol accounting, locks,
resumption, metadata drift, recording without GT telemetry and replay integrity.

For a provisioned Habitat environment and an idle rendering GPU:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.smoke
```

This creates a temporary license-free room and checks real RGB-D, ENU frame
conversion, left/right turns, forward movement and STOP. Its result is explicitly
synthetic, not a published benchmark score.

## Branch use

This shared work was extracted from `feat/objnav-habitat-gibson-nadav` at
`cc57809f` into `feat/objnav-benchmark-infra-nadav`, without merging the dataset
adapter or its experiment artifacts. Existing simulator feature branches should
merge this infrastructure branch before implementation; new ones should start
from it (or the baseline once it is integrated there). Keep benchmark-specific
logic in adapter packages instead of forking these shared modules.

