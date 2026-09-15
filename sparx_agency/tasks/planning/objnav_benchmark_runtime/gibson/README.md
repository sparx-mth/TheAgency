# Gibson ObjectNav runtime

Habitat supplies RGB-D and pose. `--explorer frontier` retains the existing
scene-graph/LLM/RPT*/weighted-A* frontier baseline. `--explorer falcon` runs:
**bounded local FALCON -> room LLM -> RPT* room order -> A*/WA* transit**, with
immediate bounded target verification and committed routes. This is a sourced
**FALCON 2D/2.5D ObjectNav adaptation**, not the unchanged ROS aerial binary.
Neither selection requires an opening 360-degree scan. See the
[component mapping and runbook](BOUNDED_FALCON.md) and
[measured results and limitations](FALCON_RESULTS.md).

This optional runtime uses the [shared benchmark harness](../../objnav_benchmark/README.md).
GT goals, semantic floor maps and distance telemetry reach only the scorer and
recorder, never the policy. No navigation-specific training is performed.

## Refinements without replacing the algorithm

- **Committed routes:** keep a safe path through turns, minor goal drift and
  unrelated map changes. Replan for obstruction, blocked movement, excessive
  cross-track error, a changed goal or bounded lack of progress, not a timer.
  `replan_steps` remains accepted for old configurations but no longer replaces
  paths periodically. Positional stagnation survives safety-triggered replans.
- **Correct arrival:** reuse the discrete converter's arrival rule on the actual
  path, including a snapped endpoint. Merely standing near the endpoint of an
  unwalked return leg does not complete the route. Frontier completion and
  target verification agree with the executor.
- **Ground-robot map:** back-project depth in 3D and integrate a robot-height
  2.5D slab. Visible floor cells provide free evidence without clearing occupied
  columns or inventing free rays through furniture. A substantial floor-height
  change resets local state instead of superimposing storeys. This is not full
  multi-floor FALCON exploration.
- **Physical clearance:** retain preferred 0.30 m clearance and a physical
  0.18 m floor. Room/frontier eligibility uses the physical-radius field, so a
  narrow doorway is not rejected before A* can try its relaxation ladder.
  Execution checks use the accepted route's actual inflation radius. Genuine
  obstacles are not removed to make a path feasible.
- **Detection duplicates:** canonicalize couch/sofa and tv/television aliases,
  suppress overlapping duplicate boxes, associate the nearest eligible
  landmark, and count at most one observation per landmark per frame.
- **Target evidence:** STOP requires fresh repeated support from separated
  views of one hypothesis. Verification has a bounded turn budget attached to
  the selected landmark, not another visible object. Rejected/stalled targets
  are cooled down and require fresh evidence. This is not infallible visual
  recognition and never uses GT success as a STOP oracle.
- **Doors and rooms:** retain depth-backed door confirmation, door-aware
  watershed, evidence-gated room typing, count-sensitive refresh and label
  invalidation after partition changes. Geometric route commitment is not
  discarded merely because room IDs change.

Records include route adoptions/keeps/invalidations, planning calls, blocked
motion, duplicates removed, target support, floor revisions and room/door
reasoning. More landmarks or rooms are not themselves accuracy improvements.

## Detector choice and measured limitations

Checkpoint identity, SHA-256, configuration and package versions are recorded.
A service with a different checkpoint cannot silently reuse the same run.
Both checkpoints are available on this workstation:

- Current completed-integration default: `~/models/objnav/yolov8x-worldv2.pt`.
- Explicit earlier rollback choices: `yolov8s-worldv2.pt` and `yolov8l-worldv2.pt`.
- Selectable alternative: pretrained LLMDet Swin-L, via its dedicated backend.

The detector upgrade from `feat/objnav-open-vocabulary-detector-nadav` is now
preserved here. See its [setup, backend selection and provenance](../../../mapping/scene_graph/serve/README.md).
New demo configurations use X-v2; previously saved explicit checkpoint choices
are not overwritten. CLI clients verify and record the selected service, not
just the checkpoint filename.

The **earlier, S-versus-L** five-scene development comparison finished with **2/5 successes
for both models**: mean SPL **0.1811 (S)** versus **0.1574 (L)**, DTG **2.5693 m
(S)** versus **4.2834 m (L)**. These are the complete inherited batches, not
best episodes mixed across attempts. On 12 previously inspected development
views, median CPU inference was approximately 37 ms (S) versus 152 ms (L).
These measurements do not establish detection AP or statistical superiority.
Those historical results are not a comparison of the completed X-v2 integration
and do not override the user's later X-v2 selection.

False-positive target stops remain a known limitation. The completed selectable
detector upgrade does not replace the existing multi-view target-evidence logic
or establish that a larger model alone cures false STOPs.
See [recovery results and remaining work](RESUME_STATUS.md).

Detector services emit proposals at 0.05 for low-scoring doorway/frame prompts.
Navigation objects still require 0.35; door candidates need geometric and
multi-view checks. The default detector runs on CPU, not Habitat's rendering GPU.

## Published split and fair interpretation

The comparison set is the released **SemExp Gibson ObjectNav v1.1 validation
split: 1,000 episodes, 200 each in Collierville, Corozal, Darden, Markleeville
and Wiconisco**. All starts are retained, including the reference finite FMM
sentinel case `Markleeville/000188`. Bounds/corruption checks remain.

The first episode per scene has already been used for development. These five
examples are not an untouched holdout and cannot support a SOTA claim. Further
systematic tuning should use the 25 training scenes or synthetic scenarios,
not repeated full-validation feedback. Training archive dummy PointNav rows
must not be mistaken for released ObjectNav evaluation episodes.

[Protocol and tuning audit](PROTOCOL_AND_TUNING.md) cites OSG Navigator's Gibson
validation protocol, distinguishes train/validation/test, and records what the
papers do not establish about parameter selection. ApexNav and SG-Nav do not
report Gibson results. There is insufficient evidence to accuse their authors
of overfitting or to guarantee absence of validation/pretraining contamination.

The freeze-then-run entrypoint locks source, method/model configuration, runtime,
seed, data and motion tolerances. It validates the actual execution configuration
again after preflight. This is a reproducibility guard, not proof of unseen
validation; role metadata retains the development-contamination disclosure.

## Data and running

Installed data:

```text
~/datasets/gibson/scenes/<scene>.glb
~/datasets/gibson/scenes/<scene>.navmesh
~/datasets/objectnav/gibson/objectnav/gibson/v1.1/val/val_info.pbz2
~/datasets/objectnav/gibson/objectnav/gibson/v1.1/val/content/<scene>_episodes.json.gz
```

On another workstation, complete the [Gibson licence form](https://forms.gle/36TW9uVpjrE1Mkf9A)
and obtain the publisher's assets. The launcher's **Import scene ZIP...** extracts
only the selected scene pair without overwriting assets. A `.glb.json.gz` file
is metadata, not a mesh. Episodes are linked in [SemExp's dataset instructions](https://github.com/devendrachaplot/Object-Goal-Navigation#downloading-episode-dataset).
Legacy floor maps use a restricted numpy-only unpickler.

In PyCharm choose **Gibson One Scene Demo**, or launch the UI from the repo root:

```bash
.venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.demo
```

Configure paths once and acknowledge the simulator-version caveat. UI settings
are in `~/.config/sparx/gibson-demo.json`. The UI runs 1-10 episodes in one scene
and can reuse/start dedicated CPU detector/Ollama services without reconfiguring
unrelated services.

With the Habitat conda environment active:

```bash
export GIBSON_SCENES_DIR="$HOME/datasets/gibson/scenes"
export GIBSON_EPISODES_DIR="$HOME/datasets/objectnav/gibson/objectnav/gibson/v1.1/val"
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run \
  --scene Collierville --limit 1 --explorer falcon --record \
  --detector-backend yolo_world --detector-url http://127.0.0.1:18095 \
  --allow-sim-version-mismatch --output "$HOME/objnav_benchmark/gibson/smoke"

python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.five_scene \
  --episodes-dir "$GIBSON_EPISODES_DIR" --scenes-dir "$GIBSON_SCENES_DIR" \
  --detector-url http://127.0.0.1:18095 --allow-sim-version-mismatch \
  --output "$HOME/objnav_benchmark/gibson/five-scene"
```

`--preflight` checks inputs/services without rendering. The five-scene entrypoint
above retains its existing baseline configuration; use `compare_explorers` from
the FALCON runbook for a matched, explicitly selected comparison. Five-scene smoke runs
one previously inspected start per scene sequentially and refuses source drift.
Use the same output with `--resume` only for unchanged configuration/data/source.
Never mix attempts. Full validation uses the separate
[freeze workflow](PROTOCOL_AND_TUNING.md#freeze-then-evaluate).

For the completed default CPU detector, run in the existing detector environment:

```bash
VOCAB=$(python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run --print-vocabulary)
python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --backend yolo_world --model "$HOME/models/objnav/yolov8x-worldv2.pt" --device cpu \
  --host 127.0.0.1 --port 18095 --conf 0.05 --torch-threads 4 --classes "$VOCAB"
```

Configure `LLM_BACKEND`, `LLM_BASE_URL`, `LLM_MODEL` as for the shared LLM client.
API keys stay in environment variables, not result artifacts. Preflight verifies
model identity and vocabulary, not just a live port. GPU ownership is checked
before rendering. No models or licensed scenes are downloaded by the CLI.

Runtime: habitat-sim, numpy-quaternion, numpy, scipy, scikit-image, scikit-fmm,
OpenCV, networkx and requests; no habitat-lab or ROS. `requirements.txt` lists
extra CPU packages. Recording needs matplotlib and FFmpeg/libx264;
`IMAGEIO_FFMPEG_EXE` can select the detector environment's working encoder.

## Simulation realism and recordings

Habitat renders a **3D scanned mesh**, not stitched input frames. Scan texture
seams, missing geometry and quality limitations can affect perception. Apparent
jumps also come from the published instantaneous 0.25 m/30-degree actions and
six-decisions-per-second playback. Changing those observations/actions would
change the benchmark, so realism improvements are confined to a separate replay:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.smooth_replay \
  "$HOME/objnav_benchmark/gibson/smoke"
```

Run replay only after evaluation releases the GPU. `smooth_rgb.mp4` is explicitly
visualization-only: interpolated frames never enter policy/scoring, and original
trajectories/metrics/videos remain intact. It smooths motion, not scan defects.
The observed library/process failures are not evidence that texture seams crash
Habitat.

Recorded runs include live/HTML dashboards, H.264 video, trajectories, metrics,
per-step predictions and reasoning. GT distance curves remain evaluator-only.

## Scoring caveats and tests

The SemExp profile uses registered 640x480 RGB-D, 79-degree HFOV, 0.88 m camera,
0.18 m radius, depth 0.5-5 m, 500 actions and sliding. GT floor scoring uses 5 cm
cells, two-cell free dilation and 20-cell goal-category dilation. Success is
zero final FMM distance to the 1 m success region; **this evaluator does not
require STOP**. SPL uses initial boundary distance plus 1 m and actual planar
travel with the reference 1e-5 m initial accumulator. SoftSPL is supplementary.

The reference used Habitat 0.1.5; this workstation runs 0.2.4. That mismatch is
explicit. Native sliding may include a bounded 5 cm navmesh correction; motion
and path-length cross-checks remain. Exact OSG implementation equivalence is not
established. Our method is metric, not non-metric.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/mapping/objects/tests sparx_agency/core/mapping/topology/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark \
  sparx_agency/core/planning/planners/common/tests \
  sparx_agency/core/planning/planners/astar/tests -q
```

A*/ObjectNav now avoid unnecessary native OMPL imports, eliminating the previous
post-pytest allocator abort on this path. Explicit OMPL users still depend on
the installed binding's health; this change does not repair that library.
