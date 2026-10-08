#!/usr/bin/env bash
# One-click Gibson ObjectNav benchmark: services up, preflight, the run, the summary, services down.
#
#   gibson/run_benchmark.sh                       # the full published val split (1,000 episodes), lean records
#   gibson/run_benchmark.sh --scene Darden --limit 1          # one episode of one scene
#   gibson/run_benchmark.sh --scene Corozal --limit 3 --record # three episodes with video + HUD recordings
#   gibson/run_benchmark.sh --output runs/gibson_val_X        # continue an interrupted run in that directory
#   gibson/run_benchmark.sh --preflight                       # services + data + config check, no episode
#   gibson/run_benchmark.sh --status runs/gibson_val_X        # print the live progress of a run and exit
#   gibson/run_benchmark.sh --five-scene 3                    # 3 recorded episodes per val scene (15), live dashboard
#   gibson/run_benchmark.sh --five-scene 3 --scenes Darden Markleeville Wiconisco   # ... for a subset of the scenes
#
# The GPU is handed out by gibson/gpu_plan.py before the services start -- LLM first, YOLO second,
# Habitat third; whatever does not fit runs on the CPU (GPU_PLAN=off restores CPU services).
# Everything else is read from the environment, with defaults for the development laptop;
# see BENCHMARK.md beside this file for the full list and what each one is for.
set -euo pipefail

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# -- where things are ---------------------------------------------------------------------
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OBJNAV_REPO:-$(cd -- "$HERE/../../../../.." && pwd)}"
HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/habitat/bin/python}"
DETECT_PY="${DETECT_PY:-$HOME/.venvs/objnav-detector/bin/python}"
DATA="${DATA:-$HOME/datasets}"
MODELS="${MODELS:-$HOME/models/objnav}"
GIBSON_EPISODES_DIR="${GIBSON_EPISODES_DIR:-$DATA/objectnav/gibson/objectnav/gibson/v1.1/val}"
GIBSON_SCENES_DIR="${GIBSON_SCENES_DIR:-$DATA/gibson/scenes}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-$MODELS/yolov8x-worldv2.pt}"
DETECTOR_HOST="${DETECTOR_HOST:-127.0.0.1}"
DETECTOR_PORT="${DETECTOR_PORT:-18095}"
DETECTOR_THREADS="${DETECTOR_THREADS:-4}"
# Ollama: "docker" starts/stops the named container, "native" runs `ollama serve`, "external" only checks it.
OLLAMA_MODE="${OLLAMA_MODE:-docker}"
OLLAMA_CONTAINER="${OLLAMA_CONTAINER:-ollama-scene-graph}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$MODELS/ollama}"
export LLM_BACKEND="${LLM_BACKEND:-ollama}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:3b-instruct}"
export LLM_REASONING_MODEL="${LLM_REASONING_MODEL:-qwen2.5:14b-instruct}"
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-120}" LLM_SEED="${LLM_SEED:-17}" LLM_TEMPERATURE="${LLM_TEMPERATURE:-0}"
export LLM_KEEP_ALIVE="${LLM_KEEP_ALIVE:-30m}" LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-768}"
export LLM_REASONING_TIMEOUT_S="${LLM_REASONING_TIMEOUT_S:-900}" LLM_REASONING_MAX_TOKENS="${LLM_REASONING_MAX_TOKENS:-1536}"
export LLM_REASONING_NUM_CTX="${LLM_REASONING_NUM_CTX:-8192}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}" MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
SEED="${SEED:-17}"
GPU_DEVICE="${GPU_DEVICE:-0}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
# auto: gpu_plan.py decides each process's device from the free VRAM; off: CPU services, Habitat on the GPU.
GPU_PLAN="${GPU_PLAN:-auto}"
CLIP_WEIGHTS="${CLIP_WEIGHTS:-$MODELS/clip/ViT-B-32.pt}"
RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"

# -- the command line -----------------------------------------------------------------------
OUTPUT="" SCENE="" LIMIT="" RECORD=0 PREFLIGHT=0 STATUS="" LEAN=1 FIVE_SCENE="" FIVE_SCENES=() EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --five-scene) FIVE_SCENE="$2"; RECORD=1; shift 2 ;;
    --scenes) shift; while [[ $# -gt 0 && "$1" != --* ]]; do FIVE_SCENES+=("$1"); shift; done ;;   # a subset, five-scene mode
    --record) RECORD=1; shift ;;
    --full-records) LEAN=0; shift ;;
    --preflight) PREFLIGHT=1; shift ;;
    --status) STATUS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;            # forwarded to gibson.run verbatim (e.g. --shards 2 --shard-index 0)
  esac
done

if [[ -n "$STATUS" ]]; then
  exec "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.progress "$STATUS"
fi

cd "$REPO"
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

[[ -x "$HAB_PY" ]] || die "Habitat interpreter not found at $HAB_PY (set HAB_PY; see BENCHMARK.md section 1)"
[[ -x "$DETECT_PY" ]] || die "detector interpreter not found at $DETECT_PY (set DETECT_PY; see BENCHMARK.md section 1)"
[[ -f "$GIBSON_EPISODES_DIR/val_info.pbz2" ]] || die "no val_info.pbz2 under $GIBSON_EPISODES_DIR (set GIBSON_EPISODES_DIR)"
[[ -d "$GIBSON_SCENES_DIR" ]] || die "no scene directory at $GIBSON_SCENES_DIR (set GIBSON_SCENES_DIR)"
[[ -f "$YOLO_WEIGHTS" ]] || die "no YOLO-World weights at $YOLO_WEIGHTS (set YOLO_WEIGHTS or MODELS)"
if [[ -z "$OUTPUT" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ -n "$FIVE_SCENE" ]]; then
    SCENE_COUNT=${#FIVE_SCENES[@]}; [[ $SCENE_COUNT -gt 0 ]] || SCENE_COUNT=5
    OUTPUT="$RUNS_ROOT/benchmark_$((FIVE_SCENE * SCENE_COUNT))episodes_${STAMP}"
  elif [[ -n "$SCENE" ]]; then OUTPUT="$RUNS_ROOT/gibson_${SCENE}_${STAMP}"; else OUTPUT="$RUNS_ROOT/gibson_val_${STAMP}"; fi
fi
if [[ -n "$FIVE_SCENE" ]]; then
  [[ -e "$OUTPUT" ]] && die "--five-scene needs a new output directory; $OUTPUT exists (the campaign never resumes)"
  SERVICE_DIR="$OUTPUT.services"      # the campaign owns $OUTPUT itself and refuses an existing directory
else
  SERVICE_DIR="$OUTPUT"
fi
mkdir -p "$SERVICE_DIR"
LOG="$SERVICE_DIR/run.log"
say "repo $REPO" | tee -a "$LOG"
say "output $OUTPUT" | tee -a "$LOG"
git --no-pager rev-parse HEAD > "$SERVICE_DIR/source-commit.txt" 2>/dev/null || true
git --no-pager diff --no-ext-diff > "$SERVICE_DIR/source-changes.diff" 2>/dev/null || true

# -- the GPU: who gets the card (LLM, then YOLO, then Habitat; the rest on the CPU) -------------------------
if [[ "$GPU_PLAN" == "auto" ]]; then
  export OLLAMA_MODELS_DIR
  PLAN_ENV="$("$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.gpu_plan \
               --gpu-device "$GPU_DEVICE" --format env --write "$SERVICE_DIR/gpu_plan.json")" \
    || die "gpu_plan.py could not place the processes (nvidia-smi? Ollama manifests under $OLLAMA_MODELS_DIR? set LLM_VRAM_MIB)"
  OPERATOR_SHARED_GPU="$ALLOW_SHARED_GPU"
  eval "$PLAN_ENV"                      # sets ALLOW_SHARED_GPU=1 when more than one process (or a desktop) holds the card
  [[ "$OPERATOR_SHARED_GPU" == 1 ]] && ALLOW_SHARED_GPU=1
  export OBJNAV_GPU_PLAN_JSON="$SERVICE_DIR/gpu_plan.json"
  say "GPU plan: LLM=$OBJNAV_LLM_DEVICE  YOLO=$OBJNAV_YOLO_DEVICE  Habitat=$OBJNAV_HABITAT_DEVICE  (gpu_plan.json; GPU_PLAN=off for CPU services)" | tee -a "$LOG"
  "$HAB_PY" - "$SERVICE_DIR/gpu_plan.json" <<'PY' | tee -a "$LOG"
import json, sys
plan = json.load(open(sys.argv[1]))
print("  %s: %d MiB total, %d used, %d free after a %d MiB reserve" % (plan["gpu_name"], plan["total_mib"], plan["used_mib"], plan["free_mib"], plan["reserve_mib"]))
for item in plan["placements"]:
    print("  %-8s %-11s %s" % (item["component"], item["device"], item["reason"]))
for warning in plan["warnings"]:
    print("  WARNING: " + warning)
PY
  GPU_DEVICE="$HABITAT_GPU_DEVICE"
  if [[ "$HABITAT_SOFTWARE_RENDER" == 1 ]]; then export LIBGL_ALWAYS_SOFTWARE=1; fi
else
  OBJNAV_LLM_DEVICE=cpu OBJNAV_YOLO_DEVICE=cpu OBJNAV_HABITAT_DEVICE=gpu
  OLLAMA_CUDA_VISIBLE_DEVICES=-1 DETECTOR_DEVICE=cpu DETECTOR_CUDA_VISIBLE_DEVICES=""
  export LLM_NUM_GPU=""
  say "GPU plan off: CPU services, Habitat on GPU $GPU_DEVICE" | tee -a "$LOG"
fi

# -- services ----------------------------------------------------------------------------------
STARTED_OLLAMA=0 DETECTOR_PID="" OLLAMA_PID=""
cleanup() {
  code=$?
  trap - EXIT
  if [[ -n "$DETECTOR_PID" ]]; then kill -TERM "$DETECTOR_PID" 2>/dev/null || true; wait "$DETECTOR_PID" 2>/dev/null || true; fi
  if [[ "$STARTED_OLLAMA" == 1 ]]; then
    case "$OLLAMA_MODE" in
      docker) docker stop --timeout 15 "$OLLAMA_CONTAINER" > /dev/null 2>&1 || true ;;
      native) if [[ -n "$OLLAMA_PID" ]]; then kill -TERM "$OLLAMA_PID" 2>/dev/null || true; wait "$OLLAMA_PID" 2>/dev/null || true; fi ;;
    esac
  fi
  date -u +%FT%TZ > "$SERVICE_DIR/finished_utc.txt"
  printf '%s\n' "$code" > "$SERVICE_DIR/exit_code.txt"
  say "exit $code (log: $LOG)"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_url() {   # url, seconds, label
  local url="$1" seconds="$2" label="$3" waited=0
  until curl -fsS --max-time 5 "$url" > /dev/null 2>&1; do
    sleep 2; waited=$((waited + 2))
    [[ $waited -lt $seconds ]] || die "$label did not answer at $url within ${seconds}s"
  done
}

ollama_has() { curl -fsS --max-time 10 "$LLM_BASE_URL/api/tags" | "$HAB_PY" -c "import json,sys; sys.exit(0 if '$1' in [m['name'] for m in json.load(sys.stdin)['models']] else 1)"; }

say "LLM: $OLLAMA_MODE at $LLM_BASE_URL ($LLM_MODEL for room labels, $LLM_REASONING_MODEL for the node oracle)" | tee -a "$LOG"
if ! curl -fsS --max-time 5 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; then
  case "$OLLAMA_MODE" in
    docker) docker start "$OLLAMA_CONTAINER" > /dev/null || die "docker start $OLLAMA_CONTAINER failed"; STARTED_OLLAMA=1 ;;
    native) command -v ollama > /dev/null || die "ollama not on PATH (OLLAMA_MODE=native)"
            mkdir -p "$OLLAMA_MODELS_DIR"
            if [[ "$OLLAMA_CUDA_VISIBLE_DEVICES" == -1 ]]; then say "starting ollama on the CPU" | tee -a "$LOG"
            else say "starting ollama on GPU $OLLAMA_CUDA_VISIBLE_DEVICES" | tee -a "$LOG"; fi
            CUDA_VISIBLE_DEVICES="$OLLAMA_CUDA_VISIBLE_DEVICES" OLLAMA_HOST="${LLM_BASE_URL#http://}" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" \
              OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=30m ollama serve > "$SERVICE_DIR/ollama.log" 2>&1 &
            OLLAMA_PID=$!; STARTED_OLLAMA=1 ;;
    external) die "no LLM service answers at $LLM_BASE_URL (OLLAMA_MODE=external: start it yourself)" ;;
    *) die "OLLAMA_MODE must be docker, native or external" ;;
  esac
  wait_for_url "$LLM_BASE_URL/api/tags" 120 "Ollama"
elif [[ "$OBJNAV_LLM_DEVICE" != cpu ]]; then
  say "NOTE: an Ollama already answers at $LLM_BASE_URL; the plan's LLM device ($OBJNAV_LLM_DEVICE) only holds if that server sees the card" | tee -a "$LOG"
fi
if [[ "$OLLAMA_MODE" == docker && "$STARTED_OLLAMA" == 1 && "$OBJNAV_LLM_DEVICE" != cpu ]]; then
  say "NOTE: the $OLLAMA_CONTAINER container's GPU access was fixed when it was created; the plan's LLM device is advisory in docker mode" | tee -a "$LOG"
fi
for model in "$LLM_MODEL" "$LLM_REASONING_MODEL"; do
  if ! ollama_has "$model"; then
    say "pulling $model (first time only; several GB)" | tee -a "$LOG"
    case "$OLLAMA_MODE" in
      docker) docker exec "$OLLAMA_CONTAINER" ollama pull "$model" ;;
      native) OLLAMA_HOST="${LLM_BASE_URL#http://}" ollama pull "$model" ;;
      *) die "$model is not provisioned on $LLM_BASE_URL" ;;
    esac
  fi
done

DETECTOR_URL="http://$DETECTOR_HOST:$DETECTOR_PORT"
if curl -fsS --max-time 5 "$DETECTOR_URL/health" > /dev/null 2>&1; then
  say "detector already up at $DETECTOR_URL (left running)" | tee -a "$LOG"
else
  VOCAB="$(OMP_NUM_THREADS=2 "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run --print-vocabulary 2>/dev/null | tail -1)"
  printf '%s\n' "$VOCAB" > "$SERVICE_DIR/vocabulary.txt"
  DETECTOR_ARGS=(--backend yolo_world --model "$YOLO_WEIGHTS" --device "$DETECTOR_DEVICE" --host "$DETECTOR_HOST" --port "$DETECTOR_PORT"
                 --conf 0.05 --torch-threads "$DETECTOR_THREADS" --classes "$VOCAB")
  [[ -f "$CLIP_WEIGHTS" ]] && DETECTOR_ARGS+=(--clip-model "$CLIP_WEIGHTS")     # the local text encoder: no download on the card
  # The service keeps the same >512 MiB occupancy gate as gibson.run; the plan has already accounted for who holds the card.
  [[ "$DETECTOR_DEVICE" == cuda* && "$ALLOW_SHARED_GPU" == 1 ]] && DETECTOR_ARGS+=(--allow-shared-gpu)
  say "starting the YOLO-World detector on $DETECTOR_DEVICE at $DETECTOR_URL" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$DETECTOR_CUDA_VISIBLE_DEVICES" "$DETECT_PY" -u -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
    "${DETECTOR_ARGS[@]}" > "$SERVICE_DIR/detector.log" 2>&1 &
  DETECTOR_PID=$!
  wait_for_url "$DETECTOR_URL/health" 300 "the detector"
fi

# -- FFmpeg for --record: PATH, then any imageio-ffmpeg binary an environment on this machine ships ------
if [[ "$RECORD" == 1 && -z "${IMAGEIO_FFMPEG_EXE:-}" ]] && ! command -v ffmpeg > /dev/null 2>&1; then
  FOUND="$("$DETECT_PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  if [[ -z "$FOUND" || ! -x "$FOUND" ]]; then
    FOUND="$(ls -1 "$HOME"/miniconda3/envs/*/lib/python3.*/site-packages/imageio_ffmpeg/binaries/ffmpeg-* 2>/dev/null | head -1 || true)"
  fi
  if [[ -n "$FOUND" && -x "$FOUND" ]]; then
    export IMAGEIO_FFMPEG_EXE="$FOUND"
    say "ffmpeg for the recordings: $FOUND" | tee -a "$LOG"
  else
    die "--record needs FFmpeg: apt install ffmpeg, or set IMAGEIO_FFMPEG_EXE to an encoder with libx264"
  fi
fi

# -- the five-scene campaign: N recorded episodes per val scene, each a gibson.run job, live dashboard --------
if [[ -n "$FIVE_SCENE" ]]; then
  FIVE_ARGS=(--episodes-dir "$GIBSON_EPISODES_DIR" --scenes-dir "$GIBSON_SCENES_DIR" --output "$OUTPUT"
             --episodes-per-scene "$FIVE_SCENE" --explorer frontier --detector-url "$DETECTOR_URL" --detector-backend yolo_world
             --seed "$SEED" --gpu-device "$GPU_DEVICE" --allow-sim-version-mismatch)
  [[ "$ALLOW_SHARED_GPU" == 1 ]] && FIVE_ARGS+=(--allow-shared-gpu)
  [[ ${#FIVE_SCENES[@]} -gt 0 ]] && FIVE_ARGS+=(--scenes "${FIVE_SCENES[@]}")
  FIVE_ARGS+=("${EXTRA[@]}")
  date -u +%FT%TZ > "$SERVICE_DIR/started_utc.txt"
  say "five-scene campaign: $FIVE_SCENE episode(s) per scene, stairs forbidden; results stream to $OUTPUT/benchmark_results.{json,csv}" | tee -a "$LOG"
  set +e
  "$HAB_PY" -u -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.five_scene "${FIVE_ARGS[@]}" 2>&1 | tee -a "$LOG"
  RUN_CODE=${PIPESTATUS[0]}
  set -e
  [[ "$RUN_CODE" == 0 ]] || die "five_scene exited with $RUN_CODE (see $LOG and $OUTPUT/<Scene>/run.log)"
  say "results: $OUTPUT/benchmark_results.json, benchmark_results.csv, summary.txt, index.html; videos under $OUTPUT/<Scene>/recordings/"
  exit 0
fi

# -- the run ------------------------------------------------------------------------------------
ARGS=(--episodes-dir "$GIBSON_EPISODES_DIR" --scenes-dir "$GIBSON_SCENES_DIR" --output "$OUTPUT"
      --explorer frontier --detector-url "$DETECTOR_URL" --detector-backend yolo_world
      --seed "$SEED" --gpu-device "$GPU_DEVICE" --allow-sim-version-mismatch)
[[ -n "$SCENE" ]] && ARGS+=(--scene "$SCENE")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
[[ "$RECORD" == 1 ]] && ARGS+=(--record)
[[ "$LEAN" == 1 ]] && ARGS+=(--lean)
[[ "$ALLOW_SHARED_GPU" == 1 ]] && ARGS+=(--allow-shared-gpu)
[[ -s "$OUTPUT/episodes.jsonl" ]] && { say "resuming: $OUTPUT already holds episodes" | tee -a "$LOG"; ARGS+=(--resume); }
ARGS+=("${EXTRA[@]}")

say "preflight" | tee -a "$LOG"
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run "${ARGS[@]}" --preflight > "$OUTPUT/preflight.json" \
  || { cat "$OUTPUT/preflight.json"; die "preflight refused the run (see $OUTPUT/preflight.json)"; }
"$HAB_PY" - "$OUTPUT/preflight.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))["configuration"]
print("  episodes selected: %d%s" % (len(config["selected_episode_ids"]), "  (FULL published split)" if config["full_split"] else ""))
print("  method: %s | lean records: %s | publishable: %s" % (config["method"]["method"], config["lean_records"], config["publishable"]))
print("  llm: %s (labels) + %s (node oracle) at %s" % (config["method"]["llm"]["model"], config["method"]["llm"]["reasoning_model"], config["method"]["llm"]["base_url"]))
print("  detector: %s" % config["method"]["detector"]["metadata"].get("backend"))
PY
if [[ "$PREFLIGHT" == 1 ]]; then say "preflight only: done"; exit 0; fi

date -u +%FT%TZ > "$OUTPUT/started_utc.txt"
say "running; progress: $OUTPUT/progress.json  (watch: $0 --status $OUTPUT)" | tee -a "$LOG"
set +e
"$HAB_PY" -u -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run "${ARGS[@]}" 2>&1 | tee -a "$LOG"
RUN_CODE=${PIPESTATUS[0]}
set -e
if [[ "$RUN_CODE" != 0 ]]; then die "gibson.run exited with $RUN_CODE (see $LOG); rerun the same command with --output $OUTPUT to resume"; fi

# -- the summary -------------------------------------------------------------------------------
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.progress "$OUTPUT" --summary | tee -a "$LOG"
say "results: $OUTPUT/summary.json (metrics), comparison.md (vs published), progress.json (per scene/category), episodes.jsonl (one row per episode)"
