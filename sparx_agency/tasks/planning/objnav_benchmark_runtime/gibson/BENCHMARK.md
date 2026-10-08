# Gibson ObjectNav benchmark -- install, run one episode, run all 1,000

This is the operator's guide for the published **Gibson ObjectNav validation benchmark**
(SemExp v1.1 `val`: five buildings -- Collierville, Corozal, Darden, Markleeville,
Wiconisco -- 200 episodes each, six target categories, 500 actions per episode, scored by
**SR**, **SPL**, **DTG** and SoftSPL). One script does everything end to end:

```bash
sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/run_benchmark.sh
```

It starts the two services the agent needs (the room LLM and the object detector), checks
the data and the configuration, runs the episodes, prints a progress line after each one,
keeps a live `progress.json`, writes the final metrics, and stops the services it started.
Interrupt it at any time; the same command with `--output <same dir>` continues where it
stopped.

The agent itself -- observed RGB-D mapping, room segmentation, an LLM that values every
room from the big picture, RPT* ordering, A* routes -- is described in [README.md](../README.md).
This file is only about getting the benchmark to run.

---

## 1. Install

Tested on Ubuntu 22.04/24.04, x86-64, one NVIDIA GPU (it can be shared with a desktop),
32 GB RAM. The two models and the simulator together take about 20 GB of RAM. **The GPU is
handed out by `gpu_plan.py` before the services start** (section 3.1): the LLM first, the
detector second, the renderer third, and whatever does not fit in the free VRAM runs on the
CPU -- so a 24 GB card runs all three on the GPU, an 8 GB card runs the renderer and the
detector there and the LLM on the CPU, and the plan is printed and recorded either way.

### 1.1 Clone and the Habitat environment (simulator + the policy)

```bash
git clone <this repository> ~/GIT/TheAgency && cd ~/GIT/TheAgency
conda create -n habitat python=3.9 cmake=3.14.0 -y
conda install -n habitat habitat-sim=0.2.4 headless -c conda-forge -c aihabitat -y
~/miniconda3/envs/habitat/bin/python -m pip install \
    -r sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/requirements.txt \
    "numpy==1.26.4" scipy opencv-python networkx requests numpy-quaternion
```

The script looks for this interpreter at `~/miniconda3/envs/habitat/bin/python`; set
`HAB_PY=/path/to/python` if it lives elsewhere.

### 1.2 The detector environment (YOLO-World, CPU)

A separate virtual environment, because torch never goes into the Habitat one:

```bash
python3.10 -m venv ~/.venvs/objnav-detector
~/.venvs/objnav-detector/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~/.venvs/objnav-detector/bin/python -m pip install \
    -r sparx_agency/tasks/mapping/scene_graph/serve/requirements-grounded-vlm.txt
```

Set `DETECT_PY` if it is not at `~/.venvs/objnav-detector/bin/python`.

### 1.3 The room LLM (Ollama)

Two models serve a run: `qwen2.5:3b-instruct` labels rooms from the objects seen in them
(cheap, frequent) and `qwen2.5:14b-instruct` is the **node oracle** -- the one judgement
per loop point that values every room of the home (about 30 s per call on a 32-thread
CPU; 9 GB of RAM). Either:

- **Docker (the default, `OLLAMA_MODE=docker`)** -- a CPU-only container named
  `ollama-scene-graph` listening on `127.0.0.1:11434`:
  ```bash
  docker run -d --name ollama-scene-graph -p 127.0.0.1:11434:11434 -v ollama-scene-graph:/root/.ollama \
      -e OLLAMA_NUM_PARALLEL=1 -e OLLAMA_MAX_LOADED_MODELS=2 -e OLLAMA_KEEP_ALIVE=30m ollama/ollama
  docker exec ollama-scene-graph ollama pull qwen2.5:3b-instruct
  docker exec ollama-scene-graph ollama pull qwen2.5:14b-instruct
  docker stop ollama-scene-graph      # the script starts and stops it itself
  ```
- **Native (`OLLAMA_MODE=native`)** -- [install Ollama](https://ollama.com/download); the
  script runs `ollama serve` on the CPU and pulls the two models the first time.
- **External (`OLLAMA_MODE=external`)** -- anything OpenAI/Ollama-compatible you run
  yourself; point `LLM_BASE_URL` at it (and `LLM_BACKEND=openai LLM_API_KEY=...` for a
  hosted model). The script only checks it answers.

`OLLAMA_MAX_LOADED_MODELS=2` matters: both models stay resident instead of the 14B being
reloaded from disk at every loop point.

### 1.4 Data (licensed; nothing here downloads it for you)

1. **Gibson scenes for Habitat** (`.glb` + `.navmesh` for the five val buildings): request
   access through the publisher's [Gibson licence form](https://forms.gle/36TW9uVpjrE1Mkf9A),
   download `gibson_habitat_trainval.zip`, and extract the five scenes:
   ```bash
   unzip -j gibson_habitat_trainval.zip 'gibson/Collierville.*' 'gibson/Corozal.*' 'gibson/Darden.*' \
         'gibson/Markleeville.*' 'gibson/Wiconisco.*' -d ~/datasets/gibson/scenes
   ```
2. **SemExp ObjectNav episodes** (`objectnav/gibson/v1.1`): follow the
   [Object-Goal-Navigation instructions](https://github.com/devendrachaplot/Object-Goal-Navigation#downloading-episode-dataset).
   The `val/` directory must hold `val_info.pbz2` (the per-scene semantic floor maps the
   scorer uses), `val.json.gz` and `content/`.

Expected layout (override with `GIBSON_SCENES_DIR` / `GIBSON_EPISODES_DIR`):

```
~/datasets/
├── gibson/scenes/{Collierville,Corozal,Darden,Markleeville,Wiconisco}.{glb,navmesh}
└── objectnav/gibson/objectnav/gibson/v1.1/val/{val_info.pbz2, val.json.gz, content/}
```

### 1.5 Detector weights (~0.5 GB, SHA-256 verified)

```bash
CUDA_VISIBLE_DEVICES="" ~/.venvs/objnav-detector/bin/python \
    -m sparx_agency.tasks.mapping.scene_graph.serve.provision_grounded_vlm \
    --output ~/models/objnav --model none --include-yolo
# -> ~/models/objnav/yolov8x-worldv2.pt (+ clip/ViT-B-32.pt)
```

Override with `MODELS=/path` or `YOLO_WEIGHTS=/path/to/yolov8x-worldv2.pt`.

---

## 2. Run

All commands from the repository root. The script is
`sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/run_benchmark.sh`; below it is
written as `run_benchmark.sh`.

### 2.1 Check everything without running an episode

```bash
run_benchmark.sh --preflight
```

Starts the services (if they are not up), verifies both LLM models are provisioned, the
detector serves the exact Gibson vocabulary, the dataset holds the 1,000 published
episodes and the five scenes, the GPU is free enough to render, and prints the frozen
configuration (`preflight.json` in the output directory). Any refusal names the missing piece.

### 2.2 One episode

```bash
run_benchmark.sh --scene Darden --limit 1              # the first published Darden episode, lean record
run_benchmark.sh --scene Corozal --limit 3 --record    # three Corozal episodes with video + HUD + per-step log
```

`--limit N` takes the first N published episodes of the scene (`--shards 200 --shard-index k`
after the script's own options picks episode k alone). An episode takes 10-20 minutes with
CPU services (most of it the 14B oracle's calls, ~1 min each on a 32-thread CPU); `--record`
adds a video (`recordings/<key>/video.mp4`), the per-step `steps.jsonl` and a live HTML page
(`live.html`) and costs some speed. Recording needs an FFmpeg with libx264: `apt install
ffmpeg`, or the script picks up the `imageio-ffmpeg` binary of any conda environment on the
machine (set `IMAGEIO_FFMPEG_EXE` to choose one).

### 2.3 The full benchmark

```bash
run_benchmark.sh
```

Runs all 1,000 episodes in order into `runs/gibson_val_<UTC stamp>/`, with **lean records**
(the scores and the counters that explain each outcome -- a few KB per episode -- instead
of the ~100 KB of per-action series a recorded episode keeps). Budget roughly 7-10 minutes
per episode with CPU services on a 32-thread machine, i.e. **several days** for the full
split. Three ways to make that manageable:

- **It resumes.** Interrupt with Ctrl-C (the services are stopped cleanly) and continue
  later with `run_benchmark.sh --output runs/gibson_val_<stamp>`: the completed episodes
  are read back from `episodes.jsonl` and skipped. The configuration is frozen in
  `run.json`; a resume after changing the source, the models, the data or the settings
  is refused on purpose -- start a new directory for a new experiment.
- **Shard across machines.** `run_benchmark.sh --shards 4 --shard-index 0` (… 1, 2, 3)
  on four machines, then merge the four complete directories into one result with
  `python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.report <dir0> <dir1> <dir2> <dir3> --merge-output <merged>`.
- **Run detached.** Name the directory up front, let the launcher record its own PID, and put
  the monitor (2.4) beside it; `setsid` gives the run its own session, so a closed terminal or a
  dropped SSH/IDE session cannot reach it:
  ```bash
  OUT=runs/gibson_val_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
  setsid nohup bash -c 'echo $$ > "$1/launcher.pid"; exec "$2" --output "$1"' _ "$OUT" \
      sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/run_benchmark.sh \
      > "$OUT/launcher.log" 2>&1 < /dev/null &
  setsid nohup "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.monitor \
      "$OUT" --interval 300 > /dev/null 2>&1 < /dev/null &
  ```
  (`bash -c` writes the PID itself because `setsid` forks when it is a group leader, so `$!`
  would name a process that has already exited.) To stop it, `kill -TERM -- -$(cat
  "$OUT/launcher.pid")` ends the whole group -- the run, the detector and the Ollama it
  started; the run lock is an advisory `flock`, so the same command with `--output "$OUT"`
  resumes afterwards. The console output is also in `<output>/run.log`.

### 2.4 Watching progress

Four views of the same thing:

- **The console.** One line per finished episode:
  `[######------------------------] 212/1000  21.2% | SR 0.642 SPL 0.331 DTG 1.52 | ETA 3d02h | last Corozal/000011 chair SR=1 SPL=0.712 steps=143 (412s)`
- **`<output>/progress.json`**, rewritten atomically after every episode: done/total,
  percent, elapsed and ETA (from this session's mean episode time), the episode in
  progress, the last episode's outcome, and the running SR / SPL / SoftSPL / DTG /
  mean steps / STOP rate / agent errors **overall, per scene and per target category**.
  Any tool can poll it (`watch -n 60 cat progress.json`).
- **`run_benchmark.sh --status <output>`** prints that file as a readable table, from any
  shell, while the run is going.
- **The monitor** (`gibson.monitor <output> --interval 300`, the detached companion of 2.3)
  reads for a run nobody is watching: every tick it appends one status line (UTC, done/total,
  SR/SPL/DTG, ETA, the episode in progress, the last outcome) to `<output>/monitor.log` and
  keeps `<output>/episodes.csv` current -- one row per finished episode with the scalar fields
  of `episodes.jsonl` only (id, scene, target, success, SPL, SoftSPL, path length, shortest
  path, start and final distance to goal, steps, STOP, termination, wall time, agent error).
  When the launcher's process is gone it writes the finished summary (or the progress view,
  if the run died before `summary.json`) to the log and to `<output>/FINAL_SUMMARY.txt`, and
  exits with the launcher's exit code. `--once` prints one line and exits, for a cron job.

### 2.5 Reading the results

| file in `<output>/` | what it holds |
|---|---|
| `summary.json` | the finished run: SR, SPL (both with 95 % intervals), SoftSPL, DTG, mean steps -- overall, per scene, per category; terminations; agent errors |
| `comparison.md`, `audit.json` | the same numbers beside the published SemExp / PONI / LFG / OSG-Nav rows, and whether this was the complete split (an incomplete or single-scene run is marked `NOT a full benchmark result`) |
| `progress.json` | the live progress (also the per-scene / per-category running table) |
| `episodes.jsonl` | one row per episode: id, scene, target, success, SPL, SoftSPL, DTG, path lengths, steps, STOP, termination, wall time, and the lean diagnostics (`agent_info`) |
| `episodes.csv`, `monitor.log`, `FINAL_SUMMARY.txt` | the monitor's (2.4): the scalar columns of `episodes.jsonl` alone, one status line per tick, and the summary it wrote when the run ended |
| `launcher.pid`, `launcher.log` | a detached launch (2.3): the launcher's PID (its process group, for `kill -TERM -- -PID`) and everything it printed |
| `evaluation_diagnostics.jsonl` | the evaluator's own per-episode line (final DTG, path, collisions, first success action) |
| `run.json`, `preflight.json` | the frozen configuration; `source-commit.txt`/`source-changes.diff` the code that ran |
| `run.log`, `detector.log` | the console output and the detector service's log |

`run_benchmark.sh --status <output>` after the run prints the headline numbers from
`summary.json`. For a single recorded episode, `recordings/<key>/index.html` has the video
beside the map, the LLM's readings and the metrics.

**Comparability caveat.** The reference evaluation used habitat-sim 0.1.5; this runs 0.2.4
(`--allow-sim-version-mismatch` is passed and recorded). The scorer is the SemExp one
(5 cm floor maps, 1 m goal dilation, FMM distance, success without a mandatory STOP), the
episodes are the published 1,000, starts are never snapped. The report says so.

---

## 3. Configuration (environment variables)

Everything has a default for the development laptop; set what differs on your machine.

### 3.1 The GPU: who gets the card

Three processes compete for one GPU, and measured on the CPU they are not equally slow:
the LLM is the bottleneck by an order of magnitude (seconds per room label, minutes per
node-oracle call on the 14B model), the detector next (hundreds of milliseconds per frame
on four cores), the renderer last. `gpu_plan.py` therefore hands the free VRAM out in that
order -- **1. LLM, 2. YOLO-World, 3. Habitat** -- and offloads what does not fit to the
CPU, least bottlenecking first. One constraint comes before the priorities: the conda
`habitat-sim` headless build cannot render without a GPU device, so the renderer's small
footprint is reserved first (software rendering is an opt-in experiment,
`HABITAT_CPU_RENDER=1`). An LLM whose estimate fits at least half-way is loaded partially
(Ollama splits the layers itself); below that it runs on the CPU.

The plan is printed at start-up, written to `gpu_plan.json` beside the services' logs and
recorded in every run's frozen configuration (`run.json` → `config.gpu_plan`). The
footprints are explicit, dated estimates: Ollama weights from the pulled manifests x 1.35
(KV cache and buffers), 2 GB for YOLO-World X + CLIP, 1.5 GB for the renderer; override
any of them when your measurement differs.

```bash
# the plan alone, for this machine
~/miniconda3/envs/habitat/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.gpu_plan
```

| variable | default | meaning |
|---|---|---|
| `GPU_PLAN` | `auto` | `off` restores CPU services with the renderer alone on the GPU |
| `LLM_VRAM_MIB`, `YOLO_VRAM_MIB`, `HABITAT_VRAM_MIB` | manifests x 1.35, `2048`, `1536` | the footprint estimates |
| `GPU_RESERVE_MIB` | `512` | headroom left for the driver / a desktop |
| `HABITAT_CPU_RENDER` | `0` | `1` lets the plan offload the renderer to software EGL (experimental) |
| `CLIP_WEIGHTS` | `$MODELS/clip/ViT-B-32.pt` | the detector's local text encoder, passed when present |
| `OLLAMA_MODELS_DIR` | `$MODELS/ollama` | where the manifests the LLM estimate reads live (native mode's model store) |

In `docker` Ollama mode the container's GPU access was fixed when it was created, and an
Ollama that is already running keeps its own device: the plan's LLM device is then
advisory, and the script says so. The per-request `LLM_NUM_GPU` the plan exports (`-1`
every layer, `0` CPU, empty = the server decides) is sent as Ollama's `num_gpu` option.

### 3.2 Everything else

| variable | default | meaning |
|---|---|---|
| `HAB_PY` | `~/miniconda3/envs/habitat/bin/python` | the Habitat interpreter |
| `DETECT_PY` | `~/.venvs/objnav-detector/bin/python` | the detector interpreter |
| `GIBSON_EPISODES_DIR` | `~/datasets/objectnav/gibson/objectnav/gibson/v1.1/val` | the SemExp val split |
| `GIBSON_SCENES_DIR` | `~/datasets/gibson/scenes` | the five `.glb`/`.navmesh` |
| `YOLO_WEIGHTS` | `$MODELS/yolov8x-worldv2.pt` (`MODELS=~/models/objnav`) | detector checkpoint |
| `OLLAMA_MODE` | `docker` | `docker` / `native` / `external` (see 1.3) |
| `OLLAMA_CONTAINER` | `ollama-scene-graph` | the container name in docker mode |
| `LLM_BASE_URL` | `http://127.0.0.1:11434` | where the LLM answers |
| `LLM_MODEL` / `LLM_REASONING_MODEL` | `qwen2.5:3b-instruct` / `qwen2.5:14b-instruct` | room labels / the node oracle |
| `LLM_REASONING_TIMEOUT_S` | `900` | how long one oracle call may take (the search waits rather than guessing) |
| `DETECTOR_PORT`, `DETECTOR_THREADS` | `18095`, `4` | the detector service |
| `SEED` | `17` | simulator, policy and LLM seed (per episode, so order does not matter) |
| `GPU_DEVICE`, `ALLOW_SHARED_GPU` | `0`, `0` | the rendering GPU; `1` accepts a card already in use (the plan sets it whenever more than one process holds the card) |
| `RUNS_ROOT` | `<repo>/runs` | where a run directory is created when `--output` is not given |
| `OMP_NUM_THREADS` | `4` | numpy/torch threads in the simulator process |

Anything the script does not recognise is forwarded to `gibson.run` verbatim
(`--shards`, `--shard-index`, `--policy-config file.json`, `--full-records` is the
script's own switch for the full diagnostics, `--video-fps`, …).

### 3.3 The five-scene campaign (`--five-scene N`)

`run_benchmark.sh --five-scene 3` brings the services up under the same GPU plan and runs
`gibson.five_scene`: the first 3 published `val` episodes of each of the five scenes (15),
one recorded `gibson.run` job per scene, stair traversal explicitly forbidden
(`policy_config.json`, checked against every job's `run.json`). Each episode prints
`EPISODE COMPLETE` (scene, goal, SR, SPL, DTG, runtime, steps, termination, video) and a
`RUNNING MEAN` line the moment it is scored; `benchmark_results.json` / `.csv` are rewritten
after every episode under `runs/benchmark_15episodes_<UTC stamp>/`, the services' logs and
`gpu_plan.json` beside it in `<output>.services/`. The campaign never resumes: a new
output directory per attempt. See [../QUICKSTART.md](../QUICKSTART.md) for the flags.

---

## 4. If something refuses to start

| message | fix |
|---|---|
| `Habitat interpreter not found` / `detector interpreter not found` | section 1.1 / 1.2, or set `HAB_PY` / `DETECT_PY` |
| `no val_info.pbz2 under …` | section 1.4, or set `GIBSON_EPISODES_DIR` |
| `Rendering GPU is occupied` | stop the other GPU process, or `ALLOW_SHARED_GPU=1` (recorded in the run) |
| `Requested LLM model … is not provisioned` | the pull failed or `LLM_BASE_URL` points at another server; `curl $LLM_BASE_URL/api/tags` |
| `Detector emission threshold hides door candidates` | a detector started by hand with `--conf` above 0.05 is running on the port; stop it and let the script start its own |
| `the detector did not answer … within 300s` | first start loads the weights; see `<output>/detector.log` (a torch/CUDA import error means the venv of 1.2 is incomplete) |
| `cannot resume … with a different config` | the source, settings, data or models changed since the run started: new `--output` directory |
| `Room oracle failed` inside episodes | the LLM is too slow for `LLM_REASONING_TIMEOUT_S` or fell over; the run keeps exploring and backs off (`exploration_fallback` in the row); check the container's memory |
| `--record needs FFmpeg` / `No working FFmpeg` | `apt install ffmpeg`, or `IMAGEIO_FFMPEG_EXE=/path/to/ffmpeg` (any build with libx264; a conda `ffmpeg` can be present yet unloadable) |

Deeper reading: [the runtime architecture](../README.md), [the protocol and data](README.md),
[QUICKSTART.md](../QUICKSTART.md) for the development (multi-storey) campaigns, and the
repo-root `LESSONS.md`.
