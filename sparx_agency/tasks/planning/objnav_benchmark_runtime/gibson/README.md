# Gibson ObjectNav runtime

Habitat supplies registered RGB-D and measured pose. `--explorer frontier`
retains the observed-frontier baseline; `--explorer falcon` selects the bounded
FALCON planar adaptation. Both retain semantic scene graphs, room LLM reasoning,
RPT* room/portal ordering, local exploration and A*/WA* path planning. The ROS
aerial FALCON controller is unchanged. Neither explorer requires an opening
panorama.

See [runtime architecture](../README.md), [FALCON mapping](BOUNDED_FALCON.md),
[historical FALCON results](FALCON_RESULTS.md),
[distinct-building experiments](DISTINCT_BUILDINGS.md),
[protocol/tuning audit](PROTOCOL_AND_TUNING.md),
[multi-story protocol](MULTISTORY.md), and
[original multi-story outcomes](MULTISTORY_RESULTS.md).

## Refinements without replacing the algorithm

- **Routes:** preserve paths through turns, minor goal drift and unrelated map
  changes. Replan for obstruction, excessive cross-track error, changed goals
  or bounded lack of progress, not a periodic timer. Arrival uses the converter's
  rule on the actual walked path, including snapped endpoints.
- **Maps:** persistent floor grids share only an observed XY origin. Visible
  floor cells supply free evidence at their own locations; occupied columns are
  not erased and future destinations are never carved free. Transient stair
  observations do not enter semantic floor maps.
- **Transitions:** commit through approach, traversal and destination confirmation.
  Completion requires translated stable support and exit from the stair region.
  Retreat follows observed history back to a source platform; a height threshold
  alone cannot finish it. Exhausted recovery is an explicit non-target safety halt.
- **Camera:** one controller owns pitch. Positive pitch looks down. Inspections
  are bounded and keyed by floor/location, not changing yaw/pitch. Stair pitch
  stays latched until a safe transition completes. STOP never carries a LOOK.
- **Perception:** raw predictions run continuously, including stairs. Projection,
  semantic fusion and STOP confirmation are distinct. Coherent depth pixels use
  their own rays; mixed depth, unsupported floors and inconsistent 3D association
  are quarantined. No bed/sofa blacklist is used. Fresh separated-view evidence is
  still required for target STOP; recognition is not infallible.
- **Room reasoning:** observed doors, watershed rooms, revisable accumulated room
  labels, RPT* order and both local explorers remain. Room clocks pause during
  committed floor transitions; global action allowance never refills.

## Data and running

Use licensed local Gibson meshes/navmeshes and the released ObjectNav episode
annotations. `.glb.json.gz` rows in scene archives are dummy PointNav metadata,
not the 1,000 ObjectNav validation episodes. On another workstation obtain assets
through the publisher's [Gibson licence form](https://forms.gle/36TW9uVpjrE1Mkf9A)
and [episode instructions](https://github.com/devendrachaplot/Object-Goal-Navigation#downloading-episode-dataset).

The existing demo is available with `.venv/bin/python -m
sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.demo`; saved settings
are in `~/.config/sparx/gibson-demo.json`. It does not download licensed scenes.

Run simulator jobs with the existing Habitat conda interpreter; use `.venv`
only for lightweight tests/tools. Before rendering, check GPU ownership. Detector
and room-language services run on CPU on this workstation.

### Detector setup

Optional **hybrid** (YOLO + Grounding DINO + BLIP-2) and **grounded_vlm**
(DINO + BLIP-2, no YOLO) modes are available through the same detector HTTP
contract. Use the [configuration and provisioning runbook](../../../mapping/scene_graph/serve/GROUNDED_VLM.md)
and change only `backend` in its JSON example; `yolo_world` restores YOLO only.
The example uses a dedicated localhost:18100 service and verifies all labels by
default. Match `--detector-backend` to the selected service, and set
`--detector-timeout-s` explicitly for slower CPU verification. These options are
forwarded through generated runs, frozen campaigns and stair diagnostics.
Raw proposals, verification answers and rejection reasons are recorded under
`method.perception.detector_evidence` in `steps.jsonl`. They do not create
geometric portals, count as additional views or replace existing STOP safeguards.

YOLO-World X-v2 remains the completed-integration default. LLMDet Swin-L remains
selectable; S/L are explicit checkpoint overrides. Do not change another mission's
service or silently substitute a smaller model. See the
[detector setup and provenance](../../../mapping/scene_graph/serve/README.md).

The scene-independent Gibson context vocabulary now includes `stairs` and
`staircase`. They cannot match a navigation target or create a portal without
observed geometry. Restart the dedicated service with the exact ordered output
of `gibson.run --print-vocabulary`; a former 24-prompt service must fail validation.
Checkpoint, classes, prompt features, configuration and packages are pinned on
health/inference responses and in the frozen run configuration.

```bash
VOCAB=$(.venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run --print-vocabulary)
CUDA_VISIBLE_DEVICES="" "$HOME/.venvs/objnav-detector/bin/python" -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --backend yolo_world --model "$HOME/models/objnav/yolov8x-worldv2.pt" --device cpu \
  --host 127.0.0.1 --port 18095 --conf 0.05 --torch-threads 4 --classes "$VOCAB"
```

LLM settings use `LLM_BACKEND`, `LLM_BASE_URL`, `LLM_MODEL`; keys stay outside
artifacts. Use the existing CPU-only Ollama service and verify its model digest.
No inference models load inside the evaluator.

### Recordings and frozen multi-story runs

`distinct_buildings --manifest <frozen-episodes.json> --output <new-directory>
--explorers frontier --record-first --seed 17 --detector-url http://127.0.0.1:18095
--detector-backend yolo_world --allow-sim-version-mismatch` preflights and freezes
all jobs before rendering. See [MULTISTORY.md](MULTISTORY.md) for the full
generation/run commands. Use a new output directory after any source/model/data
change; never mix failed attempts or protocol versions into a resume.

An operator-authorized shared GPU can be selected with `--allow-shared-gpu` in
`run_development` and `distinct_buildings`, as in `run`. It is off by default and
is recorded in each frozen job; inspect owners and VRAM before using it. The
detector service needs its own explicit authorization, not an implicit bypass.
For three buildings with three episodes each, generate with `--multistory
--buildings 3 --episodes-per-building 3` and use one explorer. **Omit
`--record-first` to record all nine episodes.** Navigation, STOP, action limits
and scoring are unchanged by these deployment/recording choices.

Recording needs a working FFmpeg/libx264. `IMAGEIO_FFMPEG_EXE` can select the
detector environment's installed static encoder; the Habitat environment's
FFmpeg may fail to load its shared libraries. Verify the encoder before a run.

There is **one persistent panel per configured storey**, not K layers per floor.
The evaluator passes only a display count to the recorder. Unknown slots start
gray with no elevation/topology; policy knowledge remains observed-only. Full
independent grids are saved as numeric `floor_maps.npz`, with slot bindings,
objects and checksums in `floor_panels.json`. `steps.jsonl` separates raw
predictions, projection/fusion decisions, commands, observed pitch, transition
state and display metadata. All failures are retained. Smooth replay is optional
and evaluator-only; interpolated frames never enter navigation.

`stair_diagnostic` uses ordinary search unless its explicitly labelled historical
`--spent-floor-budget` override is requested. Do not use that override as autonomy
evidence. `--transitions 2` requires a genuine completed second traversal; an early
target STOP or safety halt remains a failed transition test.

## Scoring and interpretation

Published SemExp Gibson validation has 1,000 episodes, 200 each in Collierville,
Corozal, Darden, Markleeville and Wiconisco. The reference-floor planar score uses
5 cm cells, two-cell free dilation, a 1 m category dilation, final zero FMM
distance and no mandatory STOP. Preserve its sentinel start and protocol.
Reference Habitat is 0.1.5; this installation is 0.2.4. The mismatch is explicit.

The separate **multi-story development/2** protocol uses 3D travel and
height-qualified reference-floor goals. Its audited native vertical bound remains
**0.60 m**, not the policy's **0.24 m** observed tread limit. Cameras remain
640×480 RGB-D, 79° HFOV, 0.88 m height, 0.18 m radius, 0.5–5 m depth; actions are
0.25 m forward, 30° yaw, bounded 30° LOOK and a 500-action cap.

Installed Gibson semantics are incomplete across storeys. A legitimate target
on an unannotated floor cannot be verified using the policy detector as ground
truth. Goal coordinates, distance fields, topology and navmesh queries remain
evaluator-only. Report annotated success and STOP correctness separately from
transition completion, collisions, stalls, integrity and runtime errors. These
development runs are not an untouched holdout, a complete multi-floor ObjectNav
benchmark, or a SOTA claim.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/mapping/objects/tests sparx_agency/core/mapping/topology/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark \
  sparx_agency/core/planning/planners/common/tests \
  sparx_agency/core/planning/planners/astar/tests -q
```

Native staircase episodes remain necessary. Green tests, decoded recordings or
completed episodes alone do not establish successful navigation.
