#!/usr/bin/env bash
# One-click Gibson ObjectNav benchmark: services up, preflight, the run, the summary, services down.
#
#   gibson/run_benchmark.sh                       # the full published val split (1,000 episodes), lean records
#   gibson/run_benchmark.sh --scene Darden --limit 1          # one episode of one scene
#   gibson/run_benchmark.sh --scene Corozal --limit 3 --record # three episodes with video + HUD recordings
#   gibson/run_benchmark.sh --output runs/gibson_val_X        # continue an interrupted run in that directory
#   gibson/run_benchmark.sh --preflight                       # services + data + config check, no episode
#   gibson/run_benchmark.sh --status runs/gibson_val_X        # print the live progress of a run and exit
#
# Everything else is read from the environment, with defaults for the development laptop;
# see BENCHMARK.md beside this file for the full list and what each one is for.
set -euo pipefail

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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
RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"

# -- the command line -----------------------------------------------------------------------
OUTPUT="" SCENE="" LIMIT="" RECORD=0 PREFLIGHT=0 STATUS="" LEAN=1 EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
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
  if [[ -n "$SCENE" ]]; then OUTPUT="$RUNS_ROOT/gibson_${SCENE}_${STAMP}"; else OUTPUT="$RUNS_ROOT/gibson_val_${STAMP}"; fi
fi
mkdir -p "$OUTPUT"
LOG="$OUTPUT/run.log"
say "repo $REPO" | tee -a "$LOG"
say "output $OUTPUT" | tee -a "$LOG"
git --no-pager rev-parse HEAD > "$OUTPUT/source-commit.txt" 2>/dev/null || true
git --no-pager diff --no-ext-diff > "$OUTPUT/source-changes.diff" 2>/dev/null || true

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
  date -u +%FT%TZ > "$OUTPUT/finished_utc.txt"
  printf '%s\n' "$code" > "$OUTPUT/exit_code.txt"
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
            CUDA_VISIBLE_DEVICES=-1 OLLAMA_HOST="${LLM_BASE_URL#http://}" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" OLLAMA_NUM_PARALLEL=1 \
              OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=30m ollama serve > "$OUTPUT/ollama.log" 2>&1 &
            OLLAMA_PID=$!; STARTED_OLLAMA=1 ;;
    external) die "no LLM service answers at $LLM_BASE_URL (OLLAMA_MODE=external: start it yourself)" ;;
    *) die "OLLAMA_MODE must be docker, native or external" ;;
  esac
  wait_for_url "$LLM_BASE_URL/api/tags" 120 "Ollama"
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
  printf '%s\n' "$VOCAB" > "$OUTPUT/vocabulary.txt"
  say "starting the YOLO-World detector on the CPU at $DETECTOR_URL" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES='' "$DETECT_PY" -u -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
    --backend yolo_world --model "$YOLO_WEIGHTS" --device cpu --host "$DETECTOR_HOST" --port "$DETECTOR_PORT" \
    --conf 0.05 --torch-threads "$DETECTOR_THREADS" --classes "$VOCAB" > "$OUTPUT/detector.log" 2>&1 &
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
