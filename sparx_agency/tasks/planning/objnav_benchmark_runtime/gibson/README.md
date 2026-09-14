indoor profile uses a 0.75 m door cut, 1 m seed separation, 0.3 m minimum
clearance, 50-cell minimum room and 0.2 m basin-merge dynamics at the method's
This runtime composes the existing scene graph, LLM room reasoning, RPT*,
weighted A* and discrete action converter. It uses simulated RGB-D and GT pose,
not flight control, learned depth or estimated localization. The method sees
only observations and the requested category: GT semantic maps, goal positions
and distance telemetry stay on the evaluator/recorder side.

The optional Habitat/LLM/scientific dependencies live in the sibling
`objnav_benchmark_runtime` package. Scoring, logging, statistics and the run
loop remain in [`objnav_benchmark`](../../objnav_benchmark/README.md), without a
per-benchmark fork. Host frontier sweeps replace FALCON's local exploration;
this is an adaptation, not an unchanged flight-stack evaluation.

## Final five-scene smoke: 2026-09-14

Run: `~/objnav_benchmark/gibson/demos/door-room-fixes-final-20260914-124814-UTC`.
Exactly the first published episode (`<scene>/000000`) was run in each scene,
sequentially, under one frozen source fingerprint and configuration. All five
processes exited 0; all five have complete videos and no agent errors.

| Scene | Target | SR | SPL | DTG m | SoftSPL | Actions | STOP | Confirmed doors | Maximum/final regions |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Collierville | toilet | 1 | 0.4976 | 0.0000 | 0.4976 | 55 | yes | 1 | 4 / 4 |
| Corozal | bed | 1 | 0.8295 | 0.0000 | 0.8295 | 76 | yes | 1 | 4 / 4 |
| Darden | bed | 0 | 0.0000 | 7.0068 | 0.0000 | 4 | yes | 0 | 1 / 1 |
| Markleeville | bed | 0 | 0.0000 | 3.6638 | 0.1947 | 9 | yes | 0 | 1 / 1 |
| Wiconisco | toilet | 0 | 0.0000 | 11.3278 | 0.0000 | 500 | no | 1 | 8 / 5 |
| **Mean** | | **0.4000** | **0.2654** | **4.3997** | **0.3044** | | | | |

**Interpretation:** both successful episodes called STOP. Darden and
Markleeville stopped outside the success region; Wiconisco exhausted its
budget. Target verification/stopping and exploration efficiency remain open
problems. These five development episodes are NOT a full Gibson benchmark,
a SOTA result, a room-segmentation accuracy measurement, or an ablation isolating
the benefit of doors from the changed geometric room settings.

There were 20, 23, 0, 3 and 142 door candidates respectively, but only the
three confirmed landmarks listed above could affect segmentation. Room-label
history recorded 6, 6, 0, 0 and 18 changes; these include unknown-to-known
classification and resets to unknown after a partition changes. No direct
known-type-to-different-known-type revision occurred in this small run. That
revision path is covered by regression tests, not claimed as observed here.

The first attempt remains at
`~/objnav_benchmark/gibson/demos/door-room-fixes-20260914-121507-UTC` and is not
mixed into the final results. It exposed an unselected invalid start, a native
sliding tolerance issue and JSON-incompatible infinite solver diagnostics.
The original single-scene recording also remains at
`~/objnav_benchmark/gibson/demos/Collierville-20260914-110342-UTC`: SR 1.0,
SPL/SoftSPL 0.1131, DTG 0, but **500 actions with no STOP**. SemExp credited its
final position, not an explicit target-found declaration.

0.1 m grid. These are recorded as method settings, not benchmark changes.
Confirmed doors are never merged across by the core segmenter, and displayed

Doorway position is estimated from jamb/lintel depth bands instead of the far
wall visible through the centre of an open doorway. A plausible width and
height, three observations, and separated viewpoints are required before a
landmark can force a boundary. Overlapping aliases in one frame count once;
a strong competing object detection prevents a low-score door proposal from
becoming a boundary. No surveyed doors or GT semantic detections are used.

The small checkpoint produced doorway scores below 0.07 on raw development
views. The dedicated service therefore emits candidates at **0.05**, but
**navigation targets still require 0.35**. The lower value is a proposal gate,
not sufficient evidence of a door. These choices were adjusted during smoke
debugging, not a held-out evaluation. The actual quality of the predicted
boundaries must be inspected rather than inferred from their count.
even then they remain revisable. New room partitions also retire stale search
goals. The goal detector's STOP logic is unchanged by these two fixes.
clearance, 50-cell minimum room and 0.2 m basin-merge dynamics on the method's
0.1 m grid. The core watershed preserves confirmed door barriers during
merging. Door-to-room links are vetted against actual region adjacency.
These are method settings recorded in `run.json`, not changes to scoring.

Room labels wait for three confirmed landmarks spanning two distinct classes
and two graph updates. They are reconsidered when class counts/categories
change, refreshed every 50 actions, and invalidated after room splits/merges.
Labels remain provisional until repeated confident agreement, and can still
change afterwards. A changed partition retires stale search goals.
  curve, with reset and terminal observations included.
The shared `RoomTypeClassifier` retains its legacy class-set cache by default.
Online callers opt into class-diversity gating, count-sensitive signatures and
`classify(..., refresh=True)`. Refresh failures propagate rather than replacing
an old answer with a fabricated one. The recordings include provisional labels,
door nodes/links and the label history with step, old/new label and trigger.
- `metrics.json`, `episode.json`, `run.log`: per-episode results and provenance.
## Data and licence
  --allow-sim-version-mismatch --output "$HOME/objnav_benchmark/gibson/one-scene"
Obtain **Gibson for Habitat-sim**, normally `gibson_habitat_trainval.zip`, after
completing the [publisher's licence form](https://forms.gle/36TW9uVpjrE1Mkf9A).
The launcher never accepts the licence or generates a substitute scene.
The user-supplied archive on this workstation has now provided all five
validation scene pairs in `~/datasets/gibson/scenes`.
`IMAGEIO_FFMPEG_EXE` overrides that choice. Existing environments are reused;
The public SemExp ObjectNav **v1.1** release comes from
[the publisher's instructions](https://github.com/devendrachaplot/Object-Goal-Navigation#downloading-episode-dataset)
(Google Drive id `1tslnZAkH8m3V5nP8pbtBmaR2XEfr8Rau`). It is already installed
here at `~/datasets/objectnav/gibson/objectnav/gibson/v1.1/val`.
checks categories, floors, origins, starts, quaternions, map contents, meshes
and navmeshes. It qualifies ids as `<scene>/<six-digit row index>` and never
$GIBSON_SCENES_DIR/Collierville.glb
The legacy map pickle uses a numpy-only restricted unpickler; still obtain
data only from the publisher, not arbitrary pickle files.

## Runtime and services

`GIBSON_EPISODES_DIR` means the **v1.1/val directory**. The five canonical
scenes are Collierville, Corozal, Darden, Markleeville and Wiconisco, with 200
published episodes each. `Collierville.glb.json.gz` is episode metadata, NOT
a mesh. PointNav episodes, raw OBJ archives and PONI multi-goal episode files
are not substitutes for this release.
```
**Import scene ZIP…** extracts only the selected scene's GLB/navmesh pair.
It refuses missing/ambiguous members, traversal paths, symlinks, invalid GLB
headers and overwriting existing files. It does not regenerate navmeshes.
The legacy map pickle uses a numpy-only restricted unpickler; use only the
publisher's trusted archive. All inputs and meshes are fingerprinted.
The existing scene-graph detection HTTP service is reused. Give this run a
## One-button demo and recordings

In PyCharm select **Gibson One Scene Demo** and press Run. The configuration
is `.run/Gibson One Scene Demo.run.xml`. Alternatively:

```bash
.venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.demo
```

Select the data directories once, acknowledge the simulator-version caveat,
and click **Run one-scene demo**. Defaults are Collierville and one episode;
the UI permits only 1–10 episodes in a named scene, not the entire dataset.
Settings are saved outside git in `~/.config/sparx/gibson-demo.json`.

The launcher can reuse/start the existing CPU-only Ollama container and a
dedicated CPU YOLO-World service on port **18092**. Habitat alone owns the
rendering GPU. It will not reconfigure an unrelated model service. Stop
interrupts the child evaluation and closes its recorder; only processes
started by the launcher are cleaned up. Auxiliary model weights must be
provisioned with operator authorization.

Every recorded run includes:

- `live.html`, `latest.jpg`, `live.json`: live RGB/depth, detections, observed
  rooms, executed trail and actual planned path.
- `index.html`, `metrics.png`, `metrics.csv`, `demo_metrics.json`: the four
  metrics, plots and exports, with STOP/termination status made explicit.
- `recordings/<episode-key>/video.mp4`: browser-playable 1600×900 H.264 video,
  normally six decisions per playback second, NOT wall-clock speed.
- Per-recording `trajectory.png`/`.csv`, `steps.jsonl`, `metrics.json`,
  `episode.json` and `final.jpg`: path, evaluator-only distance curve, actual
  detections, policy decisions, room reasoning and label revisions.
- `run.json`, `episodes.jsonl`, `summary.json`, `audit.json`, `comparison.md`
  and logs: configuration, provenance, scores and qualified paper comparisons.

Maps in the video are **observed maps**, not GT floorplans. Low-confidence
proposal boxes are not confirmed landmarks. GT distance telemetry never reaches
the policy. No metrics are invented for an incomplete episode.

## Runtime and commands

Use the existing Habitat conda Python for simulation. The runtime requires
habitat-sim, numpy-quaternion, numpy, scipy, scikit-image, scikit-fmm, OpenCV,
networkx and requests. Direct habitat-sim is used; habitat-lab is not needed.
Recording additionally needs matplotlib and a working FFmpeg with libx264.
`IMAGEIO_FFMPEG_EXE` selects an encoder; the launcher uses the detector
environment's encoder when the Habitat environment's executable is broken.

```bash
VOCAB=$(python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run --print-vocabulary)
python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
export GIBSON_SCENES_DIR="$HOME/datasets/gibson/scenes"
export GIBSON_EPISODES_DIR="$HOME/datasets/objectnav/gibson/objectnav/gibson/v1.1/val"
  --model "$PWD/yolov8s-worldv2.pt" --device cpu --host 127.0.0.1 \
  --port 8092 --conf 0.25 --classes "$VOCAB"
If starting the detector manually, use its existing model environment and
checkpoint; never let it compete with Habitat on a nearly-full GPU:
containers, paid services, or licensed scenes are started/downloaded by the CLI.
GPU occupancy is checked immediately before rendering; the expert-only
`--allow-shared-gpu` override requires deliberate memory budgeting.

## Preflight → smoke → full run  --port 8092 --conf 0.05 --classes "$VOCAB"
  --port 18092 --conf 0.05 --classes "$VOCAB"
invalid selected episode. Full runs still check all starts. The published
Markleeville episode 188 is currently rejected as unreachable by the GT-map
Configure `LLM_BACKEND`, `LLM_BASE_URL`, `LLM_MODEL` as for the existing
LLMClient (default Ollama/qwen2.5:3b-instruct). Keys remain in `LLM_API_KEY`,
not `run.json`. Preflight checks the actual model and detector configuration;
Ollama model digests and detector checkpoint/configuration identities are
verified, not merely a reachable port. A service threshold that suppresses
door candidates is refused. The CLI does not download/start model services.
# One named scene and one episode, with recording; preflight adds --preflight.
  --scene Collierville --limit 1 --record --detector-url http://127.0.0.1:18092 \
  --allow-sim-version-mismatch --output "$HOME/objnav_benchmark/gibson/one-scene"

# Exactly the first episode in each of the five scenes, sequentially.
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.five_scene \
  --episodes-dir "$GIBSON_EPISODES_DIR" --scenes-dir "$GIBSON_SCENES_DIR" \
  --allow-sim-version-mismatch --output "$HOME/objnav_benchmark/gibson/five-scene"
The five-scene entrypoint calls the same runner in isolated child processes;
it is not a second evaluator. `campaign.json` tracks completed/failed scenes,
and the parent HTML/CSV/JSON summarize actual records with links to each video.
It refuses source changes mid-campaign. Do not mix attempts or select the best
run per scene. The first failed attempt was retained separately here.

For a full run omit `--scene` and `--limit`. For distributed runs use
`--shards N --shard-index I`, the same seed/configuration, no limit and separate
outputs. Merge complete shards with `gibson.report <run dirs> --merge-output
<new dir>`. Missing/overlapping/limited or incompatible shards are rejected.
Resume the exact original command/output with `--resume`; input, source,
settings, runtime or selection changes refuse resume. `--policy-config` changes
method settings, not the benchmark protocol.

**Full-run caveat:** Markleeville episode 188 currently fails the GT-map start
reachability check. Subset preflight now checks exactly the selected starts;
it never discards an invalid selected episode. Full validation still checks
all starts. This release/evaluator issue must be resolved before claiming a
complete 1,000-episode evaluation, not silently excluded.
## Scoring protocol and paper comparisons
The implemented profile is **SemExp Gibson ObjectNav v1.1**, not Habitat's
HM3D/MP3D goal-viewpoint evaluator. References inspected:
- [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation), revision
  `5d76902`: `arguments.py`, `constants.py`, `envs/habitat/objectgoal_env.py`,
  `envs/habitat/__init__.py`, its task YAML and `envs/utils/fmm_planner.py`.
- [PONI](https://github.com/srama2512/PONI), revision `30682c2`: its current
  multi-goal schema, goal-map validation and STOP-gated progress differ; they
  are not silently substituted for SemExp v1.1.
- [OSG Navigator](https://arxiv.org/abs/2508.04678), v1 p.11 §7.1 and Table 2
  p.12: five Gibson validation scenes and 1,000 episodes; SR, SPL and DTG.
  Its exact evaluator revision/episode hashes were not identified in the
  supplied paper. Exact OSG equivalence remains **unverified**.
- [ApexNav](https://arxiv.org/abs/2504.14478), v3 p.6 §V-A/Table I, and
  [SG-Nav](https://arxiv.org/abs/2410.08189), p.7 §4.1/Table 1: **no Gibson
  results**. Do not transplant their other-dataset values into this comparison.
Paper-reported rows in `report.py` are reference values, not reproduced results
or a claim about all current SOTA. They were transcribed from extracted text;
check the original table visually before publication.

| Quantity | This profile |
|---|---|
| Camera | registered 640×480 RGB-D, 79° horizontal FOV, 0.88 m height |
| Depth / radius | 0.5–5 m optical depth; 0.18 m agent radius |
| Actions | 0.25 m forward, 30° turns, STOP; 500-action limit; sliding enabled |
| Episode quaternion | publisher WXYZ, not assumed XYZW |
| GT scoring map | 5 cm floor map; free-space dilation two cells, category dilation 20 cells |
| Success | final FMM distance to the 1 m success region is exactly zero; STOP not required |
| SPL reference | initial boundary distance + 1 m |
| Travel | planar displacement, initial accumulator 1e-5 m |
| DTG | SemExp distance-to-success boundary, not Euclidean object-centre distance |
| Unreachable field cells | reference finite sentinel: maximum valid FMM distance + one cell |

**SR/SPL/DTG are all-episode means, failures included.** SoftSPL is supplementary
progress/efficiency, with boundary distance used consistently at both ends.
TF means training-free and NM means non-metric: these are method labels, not
numeric metrics. Our metric map/RPT method is TF, not NM. OSG's `-GT` variants
mean semantic annotations, not merely the perfect geometry used here.

The reference requests habitat-sim 0.1.5; this workstation uses 0.2.4. That
mismatch requires explicit acknowledgment and is recorded. STOP omits the
upstream dummy right turn, which does not change position-based SemExp scoring.
The four-action profile adds no extra LOOK sweeps. The Gibson motion guard
allows one 5 cm navmesh cell of sliding correction: a native 0.285 m step was
observed between valid Wiconisco points. The requested step remains 0.25 m;
all motion tolerances and actual travel are retained. Undefined solver bounds
are JSON null instead of erasing the episode's semantic diagnostics.

## Validation
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/mapping/topology/tests \
  sparx_agency/tasks/planning/objnav_benchmark -q
The final affected suite reported **1,482 passed**. The known .venv OMPL
allocator abort can occur after the pytest summary; it did so in that run
(exit 134). The actual final five-scene runtime exited 0 in every scene.
All five recorded videos were verified as H.264 at 1600×900, and the GPU
returned to its baseline occupancy after evaluation.
`habitat.smoke` provides an additional temporary synthetic-room renderer check,
explicitly marked as not a benchmark result. Unit tests use synthetic fixtures
and stub services; the recorded scene runs above used real Gibson assets and
the real detector/LLM. Do not confuse those evidence sources.
