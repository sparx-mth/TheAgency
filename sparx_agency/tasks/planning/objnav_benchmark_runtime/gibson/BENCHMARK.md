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

**Segment Anything (only for `SEGMENTATION=1`, section 3.4):** MobileSAM (39 MB) and/or SAM 2.1-tiny
(76 MB) from the ultralytics assets release, into the same directory -- the service never downloads:

```bash
"$DETECT_PY" -c "from ultralytics.utils.downloads import attempt_download_asset as d; import os; \
  [d(os.path.expanduser('~/models/objnav/' + n)) for n in ('mobile_sam.pt', 'sam2.1_t.pt')]"
```

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
run_benchmark.sh --scene Wiconisco --limit 3 --record --category toilet --shuffle   # three random published toilet episodes
```

`--limit N` takes the first N published episodes of the scene (`--shards 200 --shard-index k`
after the script's own options picks episode k alone). `--category <goal>` keeps only the
episodes whose published goal is that category (`chair`, `couch`, `potted plant`, `bed`,
`toilet`, `tv`; about a sixth of a scene's 200) -- the scoring and the goal are unchanged, the
run is marked a subset -- and `--shuffle` makes `--limit N` a random sample of the pool, seeded
with `SEED`, instead of the first N in split order: the same seed selects the same episodes, and
`run.json` / `preflight.json` list them (`selected_episode_ids`). An episode takes 10-20 minutes with
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

- **It resumes -- from the same launch, not from the script.** Interrupt with Ctrl-C (the
  services are stopped cleanly) and continue later: the completed episodes are read back from
  `episodes.jsonl` and skipped. The configuration is frozen in `run.json`, and a resume whose
  configuration differs in *any* byte is refused on purpose -- the source hash (every `.py`
  under `sparx_agency/`), the models, the data, the settings, **and the GPU plan**. That last one
  means `run_benchmark.sh --output <dir>` under `GPU_PLAN=auto` is refused almost every time:
  the launcher re-plans the GPU and the plan's free/used MiB are not what they were. The
  resume that reproduces the configuration is the supervisor's (2.6), which replays the
  original processes; by hand, replay `run.json`'s `argv` with `--resume` under the original
  environment, with `gpu_plan.json` as the run recorded it. And while a run you may want to
  resume is going, **do not edit any `.py` under `sparx_agency/`** -- that changes the source
  hash -- and do not edit `run_benchmark.sh` itself: bash reads a running script incrementally.
  **When the code had to change** -- the run died on a bug in a self-check, as the full run did at
  episode 919 on 2026-10-10 -- the configuration *will* differ, and `gibson.run --resume
  --resume-accepting-changes kinematics source_sha256` is the explicit way through: it accepts
  a difference in the named keys only (`kinematics`, `source_sha256`, `gpu_plan` -- each
  provenance or a self-check that only refuses an episode, never an input to a decision or a
  score; anything else is refused as before), keeps the recorded configuration as the run's,
  writes both values into the resume's entry in `run.json` (`resumes[].config_changes`), and
  `comparison.md` / `audit.json` title and disclose it. The supervisor (2.6) passes it as
  `RESUME_ARGS`.
- **Shard across machines.** `run_benchmark.sh --shards 4 --shard-index 0` (… 1, 2, 3)
  on four machines, then merge the four complete directories into one result with
  `python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.report <dir0> <dir1> <dir2> <dir3> --merge-output <merged>`.
- **Run detached.** One command does what the recipe below spells out -- the launcher in its own session with its
  PID recorded, the monitor (2.4) beside it, and the supervisor (2.6) once the run is past its preflight -- and the
  *same* command brings the run back after the machine restarted (2.6):
  ```bash
  PROGRESS_ENTRY=docs/progress/entries/029-gibson-val-3b-laptop-run.md \
      sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/launch_detached.sh runs/gibson_val_$(date -u +%Y%m%dT%H%M%SZ)
  ```
  `PROGRESS_ENTRY` is required for a new run (the supervisor closes that entry when it publishes); every variable of
  section 3 applies; `SUPERVISOR_ARGS=--dry-run` makes the supervisor build the result commit at the end and push
  nothing. The script makes the output path absolute -- the supervisor finds the run's processes by the `--output`
  in their command line -- and writes the supervisor's settings to `<output>/launch.env` for the resume.
  `launch_detached.sh <output> --stop` ends a run for good.

  When the card is unusable until the machine restarts -- the NVIDIA userspace upgraded under the running kernel
  module, `nvidia-smi` reporting a driver/library mismatch (LESSONS 2026-10-12) -- do **not** launch on the CPU
  path and hope to move later: the renderer, the detector's device and the LLM's `num_gpu` are part of the frozen
  configuration, and a resume cannot change them. `--after-reboot` launches the same command at the next boot
  instead: the variables of the shell that `run_benchmark.sh`, `gpu_plan.py` and the supervisor read are written to
  `<output>/launch.env`, and a one-shot `@reboot` crontab entry (tagged `# gibson-launch:<output>`) runs the launch once
  `nvidia-smi` and `docker` answer (it gives up after 15 minutes rather than launch on the CPU; its transcript is
  `<output>/boot.log`). The entry removes itself when it fires; `--stop` cancels it beforehand; `--after-reboot`
  again replaces it. A later restart of the machine is the plain `launch_detached.sh <output>` resume as above.
  ```bash
  PROGRESS_ENTRY=docs/progress/entries/029-gibson-val-3b-laptop-run.md LLM_REASONING_MODEL=qwen2.5:3b-instruct \
      LLM_VRAM_MIB=2800 OLLAMA_CONTAINER=ollama-scene-graph-gpu \
      sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson/launch_detached.sh runs/gibson_val_X --after-reboot
  ```
  By hand, the launch itself is:
  ```bash
  OUT=$PWD/runs/gibson_val_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
  G=sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson
  setsid nohup bash -c 'echo $$ > "$1/launcher.pid"; exec "$2" --output "$1"' _ "$OUT" \
      "$G/run_benchmark.sh" > "$OUT/launcher.log" 2>&1 < /dev/null &
  setsid nohup "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.monitor \
      "$OUT" --interval 300 > /dev/null 2>&1 < /dev/null &
  sleep 60   # the run must be past its preflight before the supervisor snapshots it
  setsid nohup "$G/supervise_benchmark.sh" "$OUT" > /dev/null 2>&1 < /dev/null &
  ```
  (`bash -c` writes the PID itself because `setsid` forks when it is a group leader, so `$!` would name a process
  that has already exited; `$OUT` is absolute for the reason above.) To stop the run for good, `kill -TERM --
  -$(cat "$OUT/launcher.pid")` ends the whole group -- the run, the detector and the Ollama it
  started -- and the supervisor sees the exit code 130 and does not resume it (or `touch
  "$OUT/STOP"` first). The console output is also in `<output>/run.log`.

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
| `supervisor.pid`, `supervisor.log`, `replay/`, `resumes.txt`, `publish.*`, `published_commit.txt` | the supervisor's (2.6): its PID and log, the snapshot a resume replays, how many resumes there were, the commit message and file list it built, and the commit it pushed |
| `evaluation_diagnostics.jsonl` | the evaluator's own per-episode line (final DTG, path, collisions, first success action) |
| `run.json`, `preflight.json` | the frozen configuration; `source-commit.txt`/`source-changes.diff` the code that ran |
| `run.log`, `detector.log` | the console output and the detector service's log |

`run_benchmark.sh --status <output>` after the run prints the headline numbers from
`summary.json`. For a single recorded episode, `recordings/<key>/index.html` has the video
beside the map, the LLM's readings and the metrics.

### 2.6 Unattended: supervise, resume, publish

A full split takes a day; nobody watches it end. `supervise_benchmark.sh <output>` is the
detached companion that does the three things an operator would:

1. **Waits** for the launcher's process (`launcher.pid`), polling every `POLL_S` (60 s), and
   writes what it sees to `<output>/supervisor.log`.
2. **Resumes a run that died** -- the launcher gone, no `summary.json` -- at most `MAX_RESUMES`
   (2) times. Not by calling `run_benchmark.sh` again (see 2.3: that is refused), but by
   **replaying** the live processes it snapshotted into `<output>/replay/` when it started --
   `gibson.run`, the detector and the Ollama, each as the argv, environment and working
   directory read from `/proc` -- with `--resume` appended and `gpu_plan.json` restored to
   what the run recorded, behind a wrapper that writes `launcher.pid`, `run.log` and
   `exit_code.txt` as the launcher would; then it restarts the monitor. Services that still
   answer are reused, dead ones are replayed first. A run that died on the same episode twice
   is deterministic and is left dead; exit 130 (Ctrl-C / TERM) or a `<output>/STOP` file means
   the operator stopped it. `--snapshot` retakes the snapshot while the run is alive; `--no-resume`
   only waits and publishes. `RESUME_ARGS` is appended to every replayed resume -- after a code
   fix, `RESUME_ARGS="--resume-accepting-changes kinematics source_sha256"` (2.3) is how a
   supervisor started over a dead run gets it going again: it sees the dead launcher, resumes once
   with those arguments, and publishes at the end.
3. **Publishes the result when `summary.json` appears**, so the numbers reach the repository
   without anyone at the keyboard. In a temporary worktree on the remote's branch tip -- the
   checkout you work in is never touched, and a commit of yours in the meantime is simply the
   new base -- it writes `BENCHMARK_RESULTS.md` (one section per run: SR and SPL with 95 %
   intervals, DTG, per scene, per category, the published rows from `comparison.md`, the
   provenance), `results/<run>/{summary.json,episodes.csv,run.json}`, closes the progress entry
   (`PROGRESS_ENTRY`) and adds the CHANGELOG line; commits as the repository's configured git
   user; pushes `HEAD:<branch>` (`BRANCH` defaults to the branch checked out when it started,
   `REMOTE` to `origin`); and fast-forwards your local branch if it is checked out and the
   commit touches nothing you have modified -- otherwise `git pull`. Nothing is force-pushed; a
   failed push is rebuilt on a fresh fetch and retried `PUSH_RETRIES` (12) times every
   `PUSH_RETRY_S` (300 s), and if it never lands the commit is kept on a local branch
   `results/<run>` with the push command in the log. The pushed SHA ends up in
   `<output>/published_commit.txt`.

`--dry-run` builds the commit and shows it without pushing; `--publish-only` publishes a
finished run now; `--render` prints the markdown section for a finished run and exits.

### 2.7 Watching a failure: re-record named episodes of a run

A lean run keeps no frames. To *see* why an episode failed, re-run it with `--record` under
the run's own frozen settings -- `record_episodes.sh` does that for a list of episodes, one
`gibson.run` job each:

```bash
gibson/record_episodes.sh runs/gibson_val_<stamp> runs/gibson_val_<stamp>.inspection --after-run \
    "toilet-timeout=Darden/000018,Darden/000004" "couch-wrong-stop=Wiconisco/000123" ...
```

Each argument is a **slot**: a label and one or more candidate episodes. The job replays the
run's `run.json` argv (minus its selection, `--lean` and `--resume`) with `--scene S --shards 200
--shard-index k --limit 1 --record` -- which addresses exactly `S/00000k` -- under the environment
the supervisor snapshotted, reusing the detector and the Ollama when they answer and replaying
them when they do not. A re-run is seeded like the original, but a GPU LLM is not bit-exact, so
the first candidate whose re-run **reproduces the original outcome** is kept (`--expect same`:
the success flag; for a failure also its termination; for a success an SPL of at most 0.5 -- a
slot about a circuitous path) and the next is tried otherwise; `--expect any` keeps the first.
`--after-run` waits until the run's own `gibson.run` is gone and `summary.json` is there (or the
supervisor has given up) before touching the card. `--preflight` checks every slot's first
candidate -- data, services, ffmpeg, and that the shard arithmetic selects the episode -- and
records nothing. Pick the episodes from `episodes.csv` (2.4) with whatever filter the question
needs (termination, final DTG, SPL, scene, category).

The result, under `<output-dir>/`: `videos/<NN>_<label>_<Scene>_<k>.mp4` (one frame per
decision at 6 fps; decision k is at k/6 s), `_final.jpg` (the last frame with the FINAL banner)
and, for a wrong STOP, `_verify.jpg` / `_inspect.jpg` / `_stop.jpg` -- the frames at the
closing's first VERIFY (the sighting that took the search over), first INSPECT and STOP
decisions; `index.md` / `index.json` with the original beside the re-run for every attempt and
whether it reproduced; and under `jobs/<Scene>_<k>/` the full records: `recordings/<key>/`
holds `steps.jsonl` (every decision with its detections, planned path, room labels and the LLM's
readings), `trajectory.csv`, `metrics.json`, and the job's `index.html` shows the video beside
the map. The log is `record.log`. Re-issuing the command after an interruption reuses the jobs
already recorded.

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

An operator may **pin** a process instead of letting the budget decide: `GPU_PLAN_LLM`,
`GPU_PLAN_YOLO`, `GPU_PLAN_HABITAT` = `auto` (default), `gpu` or `cpu`. A `cpu` pin takes
nothing off the budget -- for Habitat it *is* the software-rendering opt-in, and the script
sets `HABITAT_RENDER=software` for it (below) -- and a `gpu` pin is served before the auto
components and placed with a warning, like a GPU-only component, when its estimate does not
fit. The pins are recorded in the plan (`pins`) and in every placement's reason. The 8 GB
laptop's "LLM on the card, renderer on the CPU, YOLO if the VRAM left allows" is

```bash
GPU_PLAN_LLM=gpu GPU_PLAN_HABITAT=cpu LLM_VRAM_MIB=2800 run_benchmark.sh --five-scene 3 --category toilet --shuffle
```

```bash
# the plan alone, for this machine
~/miniconda3/envs/habitat/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.gpu_plan
```

| variable | default | meaning |
|---|---|---|
| `GPU_PLAN` | `auto` | `off` restores CPU services with the renderer alone on the GPU |
| `GPU_PLAN_LLM`, `GPU_PLAN_YOLO`, `GPU_PLAN_HABITAT` | `auto` | `gpu` / `cpu` pins that process regardless of the budget (a Habitat `cpu` pin is software rendering) |
| `LLM_VRAM_MIB`, `YOLO_VRAM_MIB`, `HABITAT_VRAM_MIB` | manifests x 1.35, `2048`, `1536` | the footprint estimates (`LLM_VRAM_MIB` is required in docker mode: the manifests live in the container's volume) |
| `GPU_RESERVE_MIB` | `512` | headroom left for the driver / a desktop |
| `HABITAT_CPU_RENDER` | `0` | `1` lets the plan offload the renderer to software EGL when the card is short (experimental) |
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
| `SEGMENTATION` | `0` | `1` starts the box-prompted Segment Anything service beside the detector and runs the policy with `segmentation=true`: the target's depth is read on its SAM mask, not its box (section 3.4) |
| `SAM_WEIGHTS`, `SEGMENTER_PORT`, `SEGMENTER_DEVICE`, `SEGMENTER_THREADS` | `$MODELS/mobile_sam.pt`, `18096`, `cpu`, `4` | the segmentation service (`sam2.1_t.pt` is the stronger small checkpoint) |
| `SEED` | `17` | simulator, policy and LLM seed (per episode, so order does not matter) |
| `GPU_DEVICE`, `ALLOW_SHARED_GPU` | `0`, `0` | the rendering GPU; `1` accepts a card already in use (the plan sets it whenever more than one process holds the card) |
| `HABITAT_RENDER` | `gpu` | `software` renders on Mesa's llvmpipe with `/dev/dri` hidden from the run (an unprivileged user namespace, `unshare -rm`), so Mesa's software device is habitat's EGL device 0; passes `--software-render` to `gibson.run`, which skips the GPU gate and records the renderer in `run.json`. About 30 ms a frame and ~1.3 GB of RAM per simulator. Set by the plan itself when Habitat lands on the CPU (`GPU_PLAN_HABITAT=cpu`, or `HABITAT_CPU_RENDER=1` on a short card); set it by hand with `GPU_PLAN=off` when the NVIDIA EGL is unusable and `nvidia-smi` with it (a driver upgrade until the reboot that loads the matching kernel module) |
| `RUNS_ROOT` | `<repo>/runs` | where a run directory is created when `--output` is not given |
| `OMP_NUM_THREADS` | `4` | numpy/torch threads in the simulator process |

Anything the script does not recognise is forwarded to `gibson.run` verbatim
(`--shards`, `--shard-index`, `--policy-config file.json`, `--full-records` is the
script's own switch for the full diagnostics, `--video-fps`, …).

### 3.3 The five-scene campaign (`--five-scene N`)

`run_benchmark.sh --five-scene 3` brings the services up under the same GPU plan and runs
`gibson.five_scene`: the first 3 published `val` episodes of each of the five scenes (15),
one recorded `gibson.run` job per scene, stair traversal explicitly forbidden
(`policy_config.json`, checked against every job's `run.json`). With `--category toilet` the
pool of each scene is its published toilet episodes alone, and with `--shuffle` the 3 are a
random sample of that pool seeded with `SEED` rather than the first 3 -- the toilet scenario
in every building, three starts each, the same three on every launch with the same seed; the
campaign's `campaign.json` and `index.html` say which selection was made (`selection`). Each
episode prints `EPISODE COMPLETE` (scene, goal, SR, SPL, DTG, runtime, steps, termination,
video) and a `RUNNING MEAN` line the moment it is scored; `benchmark_results.json` / `.csv`
are rewritten after every episode under `runs/benchmark_[<category>_]15episodes_<UTC stamp>/`,
the services' logs and `gpu_plan.json` beside it in `<output>.services/`. The campaign never
resumes: a new output directory per attempt. See [../QUICKSTART.md](../QUICKSTART.md) for the flags.

### 3.4 The segmentation experiment (`SEGMENTATION=1`)

Off by default. With `SEGMENTATION=1` the script starts
`scene_graph/serve/segmentation_server.py` (MobileSAM, `SAM_WEIGHTS`, on `SEGMENTER_DEVICE`,
port `SEGMENTER_PORT`) beside the detector and passes `--segmentation --segmenter-url
--segmenter-model` to the run or to every five-scene job, which sets the policy's
`RPTSettings.segmentation`. Every frame's target-class boxes are then segmented and their
depth read on the object's own pixels instead of the box's -- the fix for the STOP a metre
short of a sofa with a chair in front of it (the box pipeline reads the chair's range). Nothing
else changes, and a target box the service could not mask is *not* read the old way, so a run
with the switch and one without differ in exactly that. `run.json` records the segmenter's
checkpoint hash and URL under `method.segmenter`; the HUD tints each target mask magenta;
`segmenter.log` sits beside `detector.log`. Provision the checkpoint first (section 1.5):
the service never downloads. The per-frame cost is one SAM call on frames with a target box
(about 0.3 s on four CPU threads). Compare the two pipelines with two campaigns, one per
setting, into separate output directories:

```bash
gibson/run_benchmark.sh --five-scene 1                    # the box pipeline
SEGMENTATION=1 gibson/run_benchmark.sh --five-scene 1     # the masked pipeline
```

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
| `unable to find CUDA device 0 among N EGL devices` (habitat) / `Failed to initialize NVML: Driver/library version mismatch` (`nvidia-smi`) | the NVIDIA userspace was upgraded under the running kernel module: reboot (DKMS has the new module), or run on the CPU renderer meanwhile with `GPU_PLAN=off HABITAT_RENDER=software` (section 3.2); the Intel iGPU is no fallback -- the Gibson 16384² texture does not fit it and renders black |
| `the segmenter did not answer … within 300s` / `SEGMENTATION=1 but no SAM weights` | `<output>/segmenter.log`; provision `mobile_sam.pt` (section 1.5) |
| `cannot resume … with a different config` | the source, settings, data or models changed since the run started: new `--output` directory |
| `Room oracle failed` inside episodes | the LLM is too slow for `LLM_REASONING_TIMEOUT_S` or fell over; the run keeps exploring and backs off (`exploration_fallback` in the row); check the container's memory |
| `--record needs FFmpeg` / `No working FFmpeg` | `apt install ffmpeg`, or `IMAGEIO_FFMPEG_EXE=/path/to/ffmpeg` (any build with libx264; a conda `ffmpeg` can be present yet unloadable) |

Deeper reading: [the runtime architecture](../README.md), [the protocol and data](README.md),
[QUICKSTART.md](../QUICKSTART.md) for the development (multi-storey) campaigns, and the
repo-root `LESSONS.md`.
