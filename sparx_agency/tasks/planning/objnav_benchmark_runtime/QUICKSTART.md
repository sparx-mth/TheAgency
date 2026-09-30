# Gibson ObjectNav — end-to-end quick start

Run the full ObjectNav pipeline in the Habitat simulator on licensed Gibson
buildings: observed RGB-D mapping → YOLO-World detections → room scene graph →
room LLM (P(target) per room) → RPT\* room order → room-search loop with A\*
routes → discrete actions. Stairs come from the simulator's navmesh (ground
truth); floor transitions are decided explicitly. Everything else is observed.

Three processes cooperate, each in its own environment:

| process | interpreter | device | why separate |
|---|---|---|---|
| simulator + policy (`gibson.run_development` / `distinct_buildings`) | conda env `objnav-habitat` (Python 3.9, habitat-sim 0.2.4) | GPU 0 renders | habitat-sim pins its own Python/numpy |
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

**Simulator + policy** — habitat-sim 0.2.4 headless, then the CPU-side runtime deps:

```bash
conda create -n objnav-habitat python=3.9 cmake=3.14.0 -y
conda install -n objnav-habitat habitat-sim=0.2.4 headless -c conda-forge -c aihabitat -y
"$HAB_PY" -m pip install -r sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/requirements.txt \
    "numpy==1.26.4" scipy opencv-python networkx requests numpy-quaternion
```

**Detector service** — a dedicated venv with a device-matched torch/torchvision
first (CPU wheels are fine), then the pinned detector stack (Ultralytics YOLO-World + CLIP):

```bash
python3.10 -m venv "$HOME/.venvs/objnav-detector"
"$DETECT_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$DETECT_PY" -m pip install -r sparx_agency/tasks/mapping/scene_graph/serve/requirements-grounded-vlm.txt
```

**Room LLM** — install [Ollama](https://ollama.com/download); the model is pulled in step 2.1.

**Recordings** need FFmpeg on the `PATH` (`sudo apt install ffmpeg`).

Optional, for the unit tests (lightweight venv, no simulator):
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/tasks/planning/objnav_benchmark_runtime/tests`.

### 1.2 Datasets

Both are licensed / third-party; nothing here downloads them for you.

1. **Gibson scenes for Habitat** (`.glb` meshes + `.navmesh`): request access through the
   publisher's [Gibson licence form](https://forms.gle/36TW9uVpjrE1Mkf9A) and download
   `gibson_habitat_trainval.zip`. Keep the zip; step 2.3 extracts only the buildings it selects.
2. **SemExp ObjectNav episodes and semantic maps** (`objectnav/gibson/v1.1`): follow the
   [Object-Goal-Navigation episode instructions](https://github.com/devendrachaplot/Object-Goal-Navigation#downloading-episode-dataset).
   `train/train_info.pbz2` holds the per-floor semantic maps used to generate starts and to
   score; `val/` is the published 1,000-episode validation split.

Expected layout:

```
$DATA/
├── gibson_habitat_trainval.zip                       # from (1); or any path you pass as --archive
├── gibson/<campaign>/scenes/<Scene>.glb + .navmesh   # written by step 2.3
└── objectnav/objectnav/gibson/v1.1/
    ├── train/train_info.pbz2  (+ content/, train.json.gz)
    └── val/val_info.pbz2      (+ content/, val.json.gz)
```

### 1.3 Model weights

**YOLO-World X-v2 + its CLIP ViT-B/32 text encoder** (SHA-256 verified, ~0.5 GB), the default
and only detector the pipeline needs:

```bash
CUDA_VISIBLE_DEVICES="" "$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.provision_grounded_vlm \
    --output "$MODELS" --model none --include-yolo
# -> $MODELS/yolov8x-worldv2.pt and $MODELS/clip/ViT-B-32.pt
```

**Room LLM**: `qwen2.5:3b-instruct` via Ollama — pulled in step 2.1 into `$MODELS/ollama`.

*Optional, off by default:* Grounding DINO + BLIP-2 (`--detector-backend hybrid|grounded_vlm`,
~17 GB) — see [`scene_graph/serve/GROUNDED_VLM.md`](../../mapping/scene_graph/serve/GROUNDED_VLM.md).

---

## 2. Step-by-step execution

Run the steps in this order. Steps 2.1–2.2 are long-running services: give each its own terminal.

### 2.1 Start the room LLM (terminal 1, CPU only, its own port)

```bash
CUDA_VISIBLE_DEVICES=-1 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS="$MODELS/ollama" \
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_CONTEXT_LENGTH=8192 \
OLLAMA_KEEP_ALIVE=30m OLLAMA_NO_CLOUD=1 ollama serve
```
Two models serve the run -- `qwen2.5:3b-instruct` for the cheap, frequent room-type
classification and `qwen2.5:14b-instruct` (9 GB, Q4_K_M) for the one node-oracle judgement
per loop point -- so `OLLAMA_MAX_LOADED_MODELS=2` keeps both resident (~11 GB of RAM
together) instead of reloading the 14B from disk at every loop point. First time only, in
another shell:
```bash
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5:3b-instruct
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5:14b-instruct
```
Check: `curl -s http://127.0.0.1:11435/api/tags` lists both. On a box where Ollama runs as
the scene-graph stack's Docker container instead (`ollama-scene-graph`, CPU-only, port
11434, models in a named volume -- the development laptop), the equivalents are
`docker start ollama-scene-graph`, `docker exec ollama-scene-graph ollama pull
qwen2.5:14b-instruct`, and `LLM_BASE_URL=http://127.0.0.1:11434` below.

### 2.2 Start the YOLO detector service (terminal 2, CPU only)

The service must be started with the **exact ordered Gibson vocabulary**; a run refuses any other.

```bash
VOCAB=$("$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run --print-vocabulary)
CUDA_VISIBLE_DEVICES="" "$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
    --backend yolo_world --model "$MODELS/yolov8x-worldv2.pt" --device cpu \
    --host 127.0.0.1 --port 18095 --conf 0.05 --torch-threads 4 --classes "$VOCAB"
```

Check: `curl -s http://127.0.0.1:18095/health` reports `"backend": "yolo_world"`.
(`--conf 0.05` is the emission floor the door detector needs, not a STOP threshold.)

### 2.3 Generate the frozen episodes (once per campaign)

Deterministic, policy-independent starts in multi-story training buildings; extracts the
selected scenes from the zip into `--scenes-dir` and writes one immutable manifest:

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.generate_development \
    --train-info "$DATA/objectnav/objectnav/gibson/v1.1/train/train_info.pbz2" \
    --archive "$DATA/gibson_habitat_trainval.zip" \
    --scenes-dir "$DATA/gibson/multistory/scenes" \
    --output "$HOME/objnav_benchmark/multistory/episodes.json" \
    --multistory --buildings 3 --episodes-per-building 3 --seed 17
```

Omit `--multistory` for single-floor development starts (one episode per building).

### 2.4 Export the run environment (terminal 3, the one that runs the simulator)

```bash
export LLM_BACKEND=ollama LLM_BASE_URL=http://127.0.0.1:11435 LLM_MODEL=qwen2.5:3b-instruct LLM_TIMEOUT_S=120
export LLM_REASONING_MODEL=qwen2.5:14b-instruct LLM_REASONING_TIMEOUT_S=600
MANIFEST="$HOME/objnav_benchmark/multistory/episodes.json"
RUN_FLAGS="--detector-url http://127.0.0.1:18095 --seed 17 --allow-sim-version-mismatch"
```

`LLM_REASONING_MODEL`: the model the search's ONE judgement per loop point goes to
-- the node oracle that values every room and staircase from the whole map (P(going
there next finds the target) per node). A 3B model is not enough for it: it
double-counts search effort into semantics and cannot weigh a staircase against an
unknown room. `qwen2.5:14b-instruct` is the default and the one measured here: it
follows the prompt's rules and answers a four-node prompt in about a minute on the
laptop's CPU (32 threads, no GPU). `qwen2.5:32b-instruct` (~20 GB at Q4, several
minutes per call) is better still but does not fit beside Habitat in 30 GB of RAM; use
it on a box with more, or a hosted model via `LLM_BACKEND=openai`. Give the call a long
`LLM_REASONING_TIMEOUT_S`: the search waits for the answer rather than running on a
guess. Both models must be provisioned -- the health check looks for each and refuses
to run otherwise; nothing falls back to the small model unannounced.
`LLM_TIMEOUT_S=120`: the CPU model needs ~2 s per room in the prompt, and a building can have
14 rooms. `--allow-sim-version-mismatch` acknowledges habitat-sim 0.2.4 against the protocol's
reference 0.1.5. Add `--allow-shared-gpu` to `RUN_FLAGS` when GPU 0 also drives a desktop
(the run otherwise refuses a card with > 512 MiB already in use). `--detector-backend`
defaults to `yolo_world`.

### 2.5 Preflight (no rendering, no actions)

Verifies the manifest, scene assets, both services (model identities, vocabulary, thresholds),
package versions and the GPU gate, and writes the frozen configuration:

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run_development \
    --manifest "$MANIFEST" --scene Ranchester --output /tmp/objnav-preflight-out \
    --explorer frontier $RUN_FLAGS --preflight-output /tmp/objnav-preflight.json
```

It exits 0 after writing the JSON (`method.detector.metadata.backend` must read `yolo_world`,
`method.ground_truth_stairs` `true`); any refusal names the missing piece.

### 2.6 Smoke run: one recorded episode

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run_development \
    --manifest "$MANIFEST" --scene Ranchester --output "$HOME/objnav_benchmark/smoke" \
    --explorer frontier --limit 1 --record-first $RUN_FLAGS
```

One line per episode is printed (`success=`, `SPL=`, `steps=`); a 500-action episode takes
~10 minutes with CPU services. Watch it live in `$HOME/objnav_benchmark/smoke/live.html`.

### 2.7 Full campaign: every building, every episode, frozen before the first frame

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distinct_buildings \
    --manifest "$MANIFEST" --output "$HOME/objnav_benchmark/campaign" \
    --explorers frontier --record-first $RUN_FLAGS
```

Every job is preflighted and locked (`locks/`, `campaign.json`) before any rendering, then run
sequentially. Never resume after changing source, data, models or settings — use a new
`--output` directory. `--summarize-only` rebuilds the report from finished runs.

### 2.8 Read the results

| where | what |
|---|---|
| `<output>/index.html`, `RESULTS.md`, `statistics.json` | videos, per-episode outcomes, SPL, action counts, vertical range |
| `<output>/<explorer>/<Scene>/episodes.jsonl` | one row per episode incl. `agent_info` (`room_search_loop`, `building`, `exploration_fallback` diagnostics) |
| `<output>/<explorer>/<Scene>/recordings/<key>/` | `video.mp4`, `steps.jsonl` (every command, action, pose and detection), `trajectory.csv` |
| `.venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.timeline_room_search_loop <recordings/<key>>` | the recording as a collapsed phase/room/command timeline |

### Published Gibson validation instead of generated starts

Same services and flags; `gibson.run` reads the `val/` split and the original scorer:

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run \
    --episodes-dir "$DATA/objectnav/objectnav/gibson/v1.1/val" --scenes-dir "$DATA/gibson/val/scenes" \
    --explorer frontier --output "$HOME/objnav_benchmark/val" $RUN_FLAGS --preflight   # drop --preflight to run
```

---

## If something refuses to start

| message | fix |
|---|---|
| `Rendering GPU is occupied` | stop the other GPU process, or add `--allow-shared-gpu` (recorded in the run identity) |
| `Reference habitat-sim is 0.1.5; installed '0.2.4'` | `--allow-sim-version-mismatch` |
| `Requested LLM model ... is not provisioned` | `OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5:3b-instruct`; check `LLM_BASE_URL` |
| `Detector emission threshold hides door candidates` | restart the service with `--conf 0.05` |
| `No working FFmpeg: set IMAGEIO_FFMPEG_EXE` (`--record`) | the resolver tries `PATH`, then `ffmpeg` beside the interpreter; a conda `ffmpeg` can be present yet unloadable (`libiconv.so.2` missing in `objnav-habitat`). Point `IMAGEIO_FFMPEG_EXE` at any ffmpeg with `libx264` (e.g. another conda env's) or `apt install ffmpeg` |
| vocabulary / backend mismatch on `/health` | restart the service with `--classes "$VOCAB"` from `--print-vocabulary` and `--backend yolo_world` |
| `Room oracle failed` in an episode | raise `LLM_TIMEOUT_S`; the run now keeps exploring through it (see `exploration_fallback` in the record) |

Deeper reading: [runtime architecture](README.md), [Gibson protocol and data](gibson/README.md),
[multi-story protocol and ground-truth stairs](gibson/MULTISTORY.md),
[detector service](../../mapping/scene_graph/serve/README.md), and `LESSONS.md` at the repo root.

