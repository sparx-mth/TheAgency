# HM3D ObjectNav — execution guide (v1 and v2)

**HM3DTask** runs the shared navigation core on the HM3D ObjectNav benchmarks:
observed RGB-D mapping → YOLO-World detections → room scene graph → room LLM
(P(target) per room) → RPT\* room order → **room-search loop** with A\* routes →
discrete actions. Every failed plan, model call or decision lands in the
**exploration fallback** (a move, never an idle spin); stairs come from the
simulator's navmesh (ground truth, declared as such in the run configuration).
Habitat supplies ground-truth RGB-D and a perfect pose, so what is measured is
the search logic and nothing else. No navigation-specific training is performed.

The algorithm lives in the [shared core](../README.md) (`methods/`, `habitat/`,
`evaluation.py`, `recording.py`) and is byte-identical to what
[GibsonTask](../gibson/README.md) runs. This package only owns what is HM3D:
the episodes, the scenes, habitat-lab's view-point metrics, the two versioned
protocols, and the CLIs below. Goal positions, view points and distance
telemetry reach the scorer and the recorder only — never the policy.

> "ZSON" here means the zero-shot ObjectNav *task*. It is not a reproduction of
> the separately named ZSON paper, which appears in the comparison tables as a
> baseline.

Three processes cooperate, each in its own environment:

| process | interpreter | device | why separate |
|---|---|---|---|
| simulator + policy (`hm3d.run` / `hm3d.campaign`) | conda env `objnav-habitat` (Python 3.9, habitat-sim 0.2.4) | GPU 0 renders | habitat-sim pins its own Python/numpy |
| detector HTTP service (YOLO-World) | `~/.venvs/objnav-detector` (Python ≥ 3.10, torch) | CPU | torch never goes in the Habitat env |
| room LLM (Ollama, `qwen2.5:3b-instruct`) | system `ollama` | CPU | keeps the one GPU for the renderer |

All commands run **from the repository root**. Set once per shell:

```bash
cd ~/GIT/TheAgency
HAB_PY="$HOME/miniconda3/envs/objnav-habitat/bin/python"
DETECT_PY="$HOME/.venvs/objnav-detector/bin/python"
DATA="$HOME/datasets"            # datasets root (section 1.2)
MODELS="$HOME/models/objnav"     # weights root  (section 1.3)
```

---

## 1. Setup & downloads

### 1.1 Environments

**Simulator + policy** — habitat-sim 0.2.4 headless (the protocol's reference
version, see `HM3DProtocol.reference_habitat_sim_version`), then the CPU-side
runtime dependencies. habitat-lab is **not** needed: this adapter reads the
published episode files itself and reimplements `DistanceToGoal` on habitat-sim.

```bash
conda create -n objnav-habitat python=3.9 cmake=3.14.0 -y
conda install -n objnav-habitat habitat-sim=0.2.4 headless -c conda-forge -c aihabitat -y
"$HAB_PY" -m pip install -r sparx_agency/tasks/planning/objnav_benchmark_runtime/requirements.txt \
    -r sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/requirements.txt \
    "numpy==1.26.4" PyYAML
```

**Detector service** — a dedicated venv with a device-matched torch/torchvision
first (CPU wheels are fine), then the pinned detector stack (Ultralytics
YOLO-World + CLIP):

```bash
python3.10 -m venv "$HOME/.venvs/objnav-detector"
"$DETECT_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$DETECT_PY" -m pip install -r sparx_agency/tasks/mapping/scene_graph/serve/requirements-grounded-vlm.txt
```

**Room LLM** — install [Ollama](https://ollama.com/download); the model is pulled in step 2.1.

**Recordings** need FFmpeg with libx264 on the `PATH` (`sudo apt install ffmpeg`),
or point `IMAGEIO_FFMPEG_EXE` at one.

Optional, for the unit tests (lightweight venv, no simulator, no scenes):

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark -q
```

The simulator and the distance provider are injected, so the success rule, the
path accounting and the measurement arithmetic are tested on geometry the test
picks. `tests/test_hm3d_campaign.py` and one dataset test additionally read the
real installed episode split and are skipped or fail when it is absent.

### 1.2 Datasets

Episodes are public; scenes are licensed. Neither is downloaded by this repo.
Get the exact commands for your machine, and afterwards check the installation
(counts, meshes, navmeshes, and the release-name trap) with:

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.assets \
    --version v2 --data-root "$DATA" --instructions
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.assets \
    --version v2 --data-root "$DATA" --check
```

1. **Episodes** (public, no credentials) — habitat-lab's ObjectNav datasets:
   - v1: `objectnav_hm3d_v1.zip` → `$DATA/objectnav/hm3d/v1/` (2000 val episodes / 20 scenes)
   - v2: `objectnav_hm3d_v2.zip` → `$DATA/objectnav/hm3d/v2/` (1000 val episodes / 36 scenes)

   ```bash
   mkdir -p "$DATA/objectnav/hm3d" && cd "$DATA/objectnav/hm3d"
   wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
   wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
   unzip -q objectnav_hm3d_v1.zip && unzip -q objectnav_hm3d_v2.zip && cd -
   ```

2. **Scenes** (Matterport academic licence + API token, ~3.8 GB v0.1 / ~5.3 GB v0.2).
   Request access at <https://matterport.com/habitat-matterport-3d-research-dataset>,
   then, in the Habitat env, with the token **in your shell only**:

   ```bash
   # HM3D-Semantics v0.1 (for v1 episodes)
   "$HAB_PY" -m habitat_sim.utils.datasets_download --username "$MP_API_ID" --password "$MP_API_SECRET" \
       --uids hm3d_val_v0.1 --data-path "$DATA"
   # HM3D-Semantics v0.2 (for v2 episodes)
   "$HAB_PY" -m habitat_sim.utils.datasets_download --username "$MP_API_ID" --password "$MP_API_SECRET" \
       --uids hm3d_val_v0.2 --data-path "$DATA"
   ```

   habitat-sim's downloader only ever creates the name `hm3d` and repoints it at
   whichever release was fetched last — so make the `hm3d_v0.2` name by hand
   (`--instructions` spells it out; `--check` catches it if missed).

Expected layout:

```
$DATA/
├── objectnav/hm3d/v1/val/{val.json.gz, content/<stem>.json.gz}
├── objectnav/hm3d/v2/val/{val.json.gz, content/<stem>.json.gz}
└── scene_datasets/
    ├── hm3d/val/00NNN-<STEM>/<STEM>.basis.{glb,navmesh}          # v0.1  -> --version v1
    └── hm3d_v0.2/val/00NNN-<STEM>/<STEM>.basis.{glb,navmesh}     # v0.2  -> --version v2
```

The loader refuses v2 episodes on v0.1 scenes and vice versa: every episode's
`scene_id` prefix is checked against the protocol (four validation scenes had
debris cleared between the releases, so the geometry is not identical).

### 1.3 Model weights

**YOLO-World X-v2 + its CLIP ViT-B/32 text encoder** (SHA-256 verified, ~0.5 GB) —
the default and only detector the pipeline needs:

```bash
CUDA_VISIBLE_DEVICES="" "$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.provision_grounded_vlm \
    --output "$MODELS" --model none --include-yolo
# -> $MODELS/yolov8x-worldv2.pt and $MODELS/clip/ViT-B-32.pt
```

**Room LLM**: `qwen2.5:3b-instruct` via Ollama — pulled in step 2.1 into `$MODELS/ollama`.

*Optional, off by default:* Grounding DINO + BLIP-2 (`--detector-backend hybrid|grounded_vlm`,
~17 GB) — see [`scene_graph/serve/GROUNDED_VLM.md`](../../../mapping/scene_graph/serve/GROUNDED_VLM.md).

---

## 2. Single run & video recording

Steps 2.1–2.2 are long-running services: give each its own terminal.

### 2.1 Start the room LLM (terminal 1, CPU only, its own port)

```bash
CUDA_VISIBLE_DEVICES=-1 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS="$MODELS/ollama" \
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=4096 \
OLLAMA_KEEP_ALIVE=30m OLLAMA_NO_CLOUD=1 ollama serve
```

First time only, in another shell: `OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5:3b-instruct`.
Check: `curl -s http://127.0.0.1:11435/api/tags` lists the model.

### 2.2 Start the YOLO detector service (terminal 2, CPU only)

The service must be started with the **exact ordered HM3D vocabulary**; a run
refuses any other, and refuses an emission threshold high enough to hide door
candidates (`--conf 0.05` is that floor, not a STOP threshold).

```bash
VOCAB=$("$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run --print-vocabulary)
CUDA_VISIBLE_DEVICES="" "$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
    --backend yolo_world --model "$MODELS/yolov8x-worldv2.pt" --device cpu \
    --host 127.0.0.1 --port 18092 --conf 0.05 --torch-threads 4 --classes "$VOCAB"
```

Check: `curl -s http://127.0.0.1:18092/health` reports `"backend": "yolo_world"`
and the 29 classes. (Gibson's vocabulary differs; a service started for Gibson
is rejected here, by design.)

### 2.3 Export the run environment (terminal 3, the one that runs the simulator)

```bash
export LLM_BACKEND=ollama LLM_BASE_URL=http://127.0.0.1:11435 LLM_MODEL=qwen2.5:3b-instruct LLM_TIMEOUT_S=120
export HM3D_EPISODES_DIR="$DATA/objectnav/hm3d/v2/val"
export HM3D_SCENES_DIR="$DATA/scene_datasets"
RUN_FLAGS="--version v2 --detector-url http://127.0.0.1:18092 --seed 17 --explorer frontier"
```

`LLM_TIMEOUT_S=120`: the CPU model needs ~2 s per room in the prompt. Add
`--allow-shared-gpu` to `RUN_FLAGS` when GPU 0 also drives a desktop (the run
otherwise refuses a card with > 512 MiB already in use). For v1, point
`HM3D_EPISODES_DIR` at `.../hm3d/v1/val` and use `--version v1`.

### 2.4 Preflight (no rendering, no actions, no GPU)

Verifies the data, both services (model identities, vocabulary, thresholds),
package versions and the GPU gate, and prints the exact configuration the run
would carry:

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run \
    $RUN_FLAGS --scenes 00877-4ok3usBNeis --limit 1 --preflight
```

It exits 0 with `{"ready": true, ...}` — check `method.detector.metadata.backend`
reads `yolo_world`, `method.ground_truth_stairs` is `true`, and
`method.adaptation.local_exploration` is `frontier`. Any refusal names the missing
piece. Add `--validate-starts` to also load the scene and measure every `l`
against the publisher's own geodesic (slow; the only way to find an unreachable
start before the run stops at it).

### 2.5 One recorded episode

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run \
    $RUN_FLAGS --scenes 00877-4ok3usBNeis --limit 1 --record --video-fps 6 \
    --output "$HOME/objnav_benchmark/hm3d_v2/smoke"
```

One line per finished episode is printed
(`1/1 <episode> <target> SR=… SPL=… DTG=… SoftSPL=…`); a 500-action episode
takes ~10 minutes with CPU services. **Watch it live** by opening
`$HOME/objnav_benchmark/hm3d_v2/smoke/live.html` in a browser — it polls
`latest.jpg` (the dashboard frame: RGB with boxes, metric depth, the persistent
per-floor observed maps with rooms, trail and committed route, the decision and
the LLM's room reasons) and `live.json`.

What the run leaves behind:

| where | what |
|---|---|
| `<output>/index.html` | final dashboard: per-episode SR/SPL/DTG/SoftSPL cards, videos, trajectory plots, LLM reasons |
| `<output>/episodes.jsonl` | one row per episode incl. `agent_info` (`room_search_loop` events, `exploration_fallback.failures`, `building` transitions, route stats) |
| `<output>/run.json` | the frozen configuration: protocol, method settings, detector/LLM identities, dataset manifest, navmesh provenance |
| `<output>/recordings/<key>/video.mp4` | the dashboard video, one frame per decision (accelerated, not wall time) |
| `<output>/recordings/<key>/steps.jsonl` | every command, action, pose, detection and policy snapshot |
| `<output>/recordings/<key>/trajectory.csv` | pose, camera pitch, traversal state and evaluator-only DTG per action |
| `<output>/recordings/<key>/floor_maps.npz`, `floor_panels.json` | the persistent per-storey occupancy panels |
| `<output>/comparison.md` | the comparison table against the published numbers, incl. the 0.2 m re-score (see `report.py`) |

To inspect a recording as a collapsed phase/room/command timeline:

```bash
.venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.timeline_room_search_loop \
    "$HOME/objnav_benchmark/hm3d_v2/smoke/recordings/<key>"
```

### 2.6 Smooth first-person replay (visualization only)

Re-renders the recorded poses with interpolation into a separate
`smooth_rgb.mp4` beside each recording. It never touches scores or observations,
and it needs the rendering GPU:

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.smooth_replay \
    "$HOME/objnav_benchmark/hm3d_v2/smoke" --subframes 4
```

### 2.7 Choosing the explorer and tuning the method

`--explorer frontier` (default) runs the room-search loop with the observed
frontier sweep; `--explorer falcon` swaps in the bounded planar FALCON
adaptation. Anything else in `RPTSettings` (`methods/rpt_settings.py`) is
overridden with a JSON file:

```bash
cat > /tmp/policy.json <<'EOF'
{"detection_confidence": 0.35, "stop_distance_m": 1.0,
 "multifloor": {"stair_source": "ground_truth"}}
EOF
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run \
    $RUN_FLAGS --policy-config /tmp/policy.json --scenes 00877-4ok3usBNeis --limit 1 --preflight
```

Embodiment (`body_height_m` 0.88, `body_radius_m` 0.18) is always taken from
the HM3D agent and cannot be tuned away; `preferred_clearance_m` (0.30) and
`stop_distance_m` (1.0) are method choices and are recorded. Anything tuned on
`val` must be disclosed when the number is reported — see
[PROTOCOL_AND_TUNING.md](PROTOCOL_AND_TUNING.md).

---

## 3. Large-scale multi-environment benchmarking

A full split is 20 (v1) or 36 (v2) buildings, and habitat-sim holds a scene's
meshes and rendering context for the life of the process — so the campaign
runner executes **one child process per scene**, each writing a complete,
self-contained results directory (its own `run.json`, `episodes.jsonl`, lock),
then aggregates exactly the scenes that produced rows and names the ones that
did not. Every argument after `--` goes to every child's `hm3d.run` unchanged.

### 3.1 Smoke across the whole split (one episode per scene)

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign \
    "$HOME/objnav_benchmark/hm3d_v2/smoke_all" --version v2 \
    --episodes-dir "$HM3D_EPISODES_DIR" --scenes-dir "$HM3D_SCENES_DIR" \
    --limit 1 -- --detector-url http://127.0.0.1:18092 --seed 17 --explorer frontier --record
```

36 scenes × 1 episode ≈ 6 h with CPU services; the summary in
`smoke_all/campaign.json` and each scene's `comparison.md` is a diagnostic subset and
is labelled as such.

### 3.2 The full published validation split

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign \
    "$HOME/objnav_benchmark/hm3d_v2/campaign" --version v2 \
    --episodes-dir "$HM3D_EPISODES_DIR" --scenes-dir "$HM3D_SCENES_DIR" \
    --resume -- --detector-url http://127.0.0.1:18092 --seed 17 --explorer frontier
```

- `--resume` restarts a scene that started but did not finish; an unchanged
  configuration and unchanged data are required — the logger refuses anything
  else. Never resume after changing source, models or settings: use a new root.
- `--scenes 00877-4ok3usBNeis 00880-Nfvxx8J5NCo …` restricts the sweep to named
  buildings (a subset, labelled as such).
- `--aggregate-only` rebuilds the aggregate from the scenes that finished.
- Add `--record` after `--` to keep videos for every episode (≈ 20 MB each), or
  leave it off and record only the smoke runs.
- Run v1 the same way with `--version v1` and `HM3D_EPISODES_DIR=.../hm3d/v1/val`.
- The campaign exits non-zero when any scene crashed or produced no rows, so it
  can gate a scripted sweep.

Budget: 1000 episodes × ~8–10 min ≈ 6–7 days on one GPU with CPU services; v1's
2000 episodes double it. Split across machines with `hm3d.run`'s
`--shards N --shard-index I` (deterministic, stable selection) and merge with
`report.py --merge-output`, which refuses shards that disagree on data, protocol,
method or runtime, or that do not cover every episode exactly once:

```bash
# machine i of N, each with its own services:
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run \
    $RUN_FLAGS --shards 4 --shard-index $i --output "$HOME/objnav_benchmark/hm3d_v2/shard_$i"
# afterwards, on one machine:
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.report \
    "$HOME/objnav_benchmark/hm3d_v2/shard_"{0,1,2,3} --merge-output "$HOME/objnav_benchmark/hm3d_v2/merged"
```

### 3.3 Freeze, then evaluate (for a number you intend to publish)

A frozen evaluation is the complete published split, with no `--scenes`,
`--limit`, sharding or exclusion. The lock captures the actual Python sources,
the policy and converter settings, the selected episode order and the execution
settings; the run is checked against it before the first output is written and
before the first reset, not merely at an earlier preflight.

```bash
LOCK="$HOME/objnav_benchmark/hm3d_v2/locks/frontier_seed17.json"
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.frozen_eval freeze \
    --lock "$LOCK" --note "developed on Gibson multi-story development starts; HM3D val untouched" \
    -- $RUN_FLAGS
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.frozen_eval run \
    --lock "$LOCK" -- $RUN_FLAGS --output "$HOME/objnav_benchmark/hm3d_v2/frozen_frontier_seed17"
```

A lock is a reproducibility commitment, **not proof that validation was
unseen** — `--note` is where the tuning history is disclosed, and it is copied
into the results. `evaluation_role.json` in the output marks the run as a frozen
published-split validation.

### 3.4 Reading the numbers

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.report \
    "$HOME/objnav_benchmark/hm3d_v2/campaign"/*/          # or a single run / merged directory
```

SR, SPL and DTG are the three the comparison papers use; SoftSPL is
supplementary. All four are means over **every scored episode, failures
included**, as the leaderboards compare them. The report scores at habitat-lab's
0.1 m success distance and **re-scores the same rows at ApexNav's 0.2 m**, so
both exist from one run and cannot drift apart. It also lists, per scene, how
many episodes were scored, so a subset can never masquerade as the split.

Reference numbers live in [`sota.py`](sota.py) with the table each was printed
in; they are transcribed, never measured here, and the protocols behind them
differ. OSG's HM3D row (a 400-episode self-sampled subset with no published ids)
is excluded unless `--include-osg` is passed. Read
[PROTOCOL_AND_TUNING.md](PROTOCOL_AND_TUNING.md) before putting any of our
numbers next to theirs.

Per-episode diagnostics of the algorithm itself — how many rooms were visited,
in what order, which fell to the fallback and why, how many actions each phase
took — are in `agent_info` of every `episodes.jsonl` row
(`room_search_loop.events`, `exploration_fallback.failures`, `building.events`,
`route_commitment`).

### Development split

`--split train --episodes-per-scene N` runs the HM3D training split for
development. It renames the protocol, files its rows separately and can never
be called complete; `hm3d.campaign --split train --limit N` sweeps it the same
way. Tune there; touch `val` only under a lock.

---

## If something refuses to start

| message | fix |
|---|---|
| `Rendering GPU is occupied` | stop the other GPU process, or add `--allow-shared-gpu` (recorded in the run identity) |
| `Set --episodes-dir … and --scenes-dir …` | export `HM3D_EPISODES_DIR` / `HM3D_SCENES_DIR` (section 2.3) |
| `Missing …/val.json.gz` | `--episodes-dir` must be the *split* directory holding `val.json.gz` and `content/` |
| scene prefix / release mismatch from the loader | v1 episodes need `scene_datasets/hm3d` (v0.1), v2 need `hm3d_v0.2`; run `hm3d.assets --check` |
| `Only N of M checked starts reproduce the publisher's own geodesic` | wrong navmesh in force; check `HM3DProtocol.navmesh` (`agent` is the reference) — `--allow-geodesic-mismatch` only records a deliberate deviation |
| `N selected episodes have no reachable goal view point` | `--exclude-unreachable` with `--validate-starts`; the dropped ids are recorded in the manifest |
| `Requested LLM model … is not provisioned` | `OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5:3b-instruct`; check `LLM_BASE_URL` |
| `Detector emission threshold hides door candidates` | restart the service with `--conf 0.05` |
| vocabulary / backend mismatch on `/health` | restart the service with `--classes "$VOCAB"` from `hm3d.run --print-vocabulary` and `--backend yolo_world` |
| `Room oracle failed` in an episode | raise `LLM_TIMEOUT_S`; the run keeps exploring through it (see `exploration_fallback` in the record) |
| `Recording: … ffmpeg` | install FFmpeg with libx264, or set `IMAGEIO_FFMPEG_EXE` |

---

## Protocol reference

### The two published datasets

One adapter, two versioned protocols, because only the episodes and the scenes
differ — the agent, the actions and the six categories are identical.

| | **HM3D-v1** | **HM3D-v2** |
|---|---|---|
| Episode dataset | `objectnav/hm3d/v1` | `objectnav/hm3d/v2` |
| Scene release | HM3D-Semantics **v0.1** | HM3D-Semantics **v0.2** |
| Challenge | Habitat 2022 | Habitat 2023 episodes, 2022 agent |
| Validation split | **2000 episodes / 20 scenes** | **1000 episodes / 36 scenes** |
| Scenes named | `hm3d/val/...` | `hm3d_v0.2/val/...` |
| Category balance | `plant` is only 84/2000 | balanced, 135–195 each |

The six goal categories are the dataset's own spellings, in its category-index
order: `chair`, `bed`, `plant`, `toilet`, `tv_monitor`, `sofa`.

The 20 v1 validation scenes are a subset of the 36 v2 ones, but the **geometry
is not identical** — HM3DSem v0.2 manually cleared debris in four validation
scenes, one of which (`00878-XB4GS9ShBRE`) is in the v1 set. Pointing both names
at one download, as ApexNav's README suggests, therefore runs one of the two
versions against scenes its episodes were not generated on. The loader refuses
it: every episode's `scene_id` prefix is checked against the protocol.

The **2023 challenge itself is not what the papers ran** — it moved to a
HelloRobot Stretch with velocity control. "HM3D-v2" throughout this package
means the v2 *episodes* under the v1 *agent*, which is what ApexNav's released
configs do.

### The protocol, and where it comes from

Every value is read off habitat-lab's own `objectnav_hm3d` benchmark
configuration, not inferred from a benchmark name:

| | |
|---|---|
| Steps | 500, STOP included |
| Actions | STOP, MOVE_FORWARD 0.25 m, TURN_LEFT/RIGHT 30°, LOOK_UP/DOWN 30° (pitch unclamped) |
| Agent | height 0.88 m, radius 0.18 m, sliding **off** |
| Camera | 640×480 RGB-D, HFOV 79°, 0.88 m above the floor, depth 0.5–5.0 m |
| Success | STOP **and** geodesic distance to the nearest goal **view point** < **0.1 m** |
| SPL | `S · l / max(l, p)`, `l` = that same distance at reset, `p` = 3-D chords |

`l` is the distance *habitat-lab measures at reset*, not the row's published
`info.geodesic_distance` — and it is measured to view points, not to the object.
The 1 m "within a metre of the object" of the task description is baked into
where the generator placed the view points; 0.1 m is only a snap tolerance on
top of them.

habitat-lab is **not installed here**; only habitat-sim is. So this adapter
reads the published episode files itself and reimplements `DistanceToGoal`
against habitat-sim's `MultiGoalShortestPath` — one number, measured once per
step, from which success, SPL, SoftSPL and DTG all follow. See
[`geodesic.py`](geodesic.py).

#### The navmesh is not a detail

habitat-sim's `Simulator._config_pathfinder` loads the shipped
`<scene>.basis.navmesh` **and then recomputes it** whenever the mesh's recorded
settings differ from the agent's radius and height — which they always do here,
because HM3D's meshes are built at the library defaults (0.10 m / 1.50 m) and
ObjectNav's agent is 0.18 m / 0.88 m. So the reference pipeline runs on a
*recomputed* navmesh.

That changes what is reachable, and therefore `l`, and therefore every SPL.
Measured on an ObjectNav episode carrying a published `info.geodesic_distance`:
the mesh recomputed at 0.18/0.88 reproduced the publisher's number **bit-exactly**,
while the shipped mesh was 5.5% short. On one scene the ObjectNav settings cut
navigable area by 15% and made half of a sample of connected point pairs
unreachable.

`HM3DProtocol.navmesh` therefore defaults to `"agent"` — the reference
behaviour — and `"published"` exists only for a deliberate comparison. Whichever
is used, `navmesh_provenance()` records the settings, the navigable area and the
island count that were actually in force.

Because this lives in habitat-sim rather than habitat-lab, it is **not** a
difference between the papers we compare against: both run a habitat-sim that
recomputes, and so do we. What it does give us is a free check.
`info.geodesic_distance` is the publisher's geodesic to the nearest goal view
point — the same quantity as `l` — so `--validate-starts` reports how closely
our `l` reproduces it, and a systematic disagreement is a defect rather than a
difference of definition. `run.py` turns a bad agreement into a preflight
issue.

#### The two deviations we name rather than apply

- **Success radius.** ApexNav's released configs use **0.2 m** where
  habitat-lab uses 0.1 m. Loosening it can only add successes. We score at 0.1 m
  and **re-score the same episodes** at 0.2 m from the rows on disk, so both
  rows exist from one run and cannot drift apart.
- **Depth near clip.** ApexNav opens the depth sensor to `min_depth 0.0`;
  SG-Nav's HM3D config raises `max_depth` to 10 m. We keep habitat-lab's
  0.5–5.0 m. This changes what the *method* can see, not how it is scored.

## Files

| file | what it holds |
|---|---|
| `protocol.py` | `HM3DProtocol`, `HM3D_V1`, `HM3D_V2`, the navmesh choice, the named ApexNav radius |
| `dataset.py` | the published episodes and goals, scene-qualified ids, XYZW→WXYZ, the manifest |
| `geodesic.py` | habitat-lab's `DistanceToGoal` to view points, on habitat-sim's pathfinder |
| `env.py` | `HM3DEnv`: the episode (with navmesh storeys/stairs as its only metadata), the measurement, start validation, navmesh provenance |
| `assets.py` | what is installed, what is missing, and the commands that would fix it |
| `run.py` | preflight and the run, on the shared `run_evaluation`; builds the shared `RPTSearchPolicy` from this protocol's agent |
| `campaign.py` | one child process per scene, then an aggregate over what finished |
| `report.py` | the comparison table, the 0.2 m re-score, the audit, shard merging |
| `sota.py` | the published numbers, with the table each came from |
| `frozen_eval.py` | freeze an exact configuration, then run the full split under it |
| `smooth_replay.py` | visualization-only smooth video, after the fact, never scored |

Deeper reading: [shared core architecture](../README.md) (the room-search loop,
the exploration fallback, the frontier sweep, camera control),
[protocol and tuning](PROTOCOL_AND_TUNING.md),
[multi-story protocol and ground-truth stairs](../gibson/MULTISTORY.md),
[detector service](../../../mapping/scene_graph/serve/README.md), and
`LESSONS.md` at the repo root.

