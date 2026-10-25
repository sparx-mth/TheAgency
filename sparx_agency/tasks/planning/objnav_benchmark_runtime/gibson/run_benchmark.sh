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
#   gibson/run_benchmark.sh --five-scene 3 --category toilet --shuffle   # the toilet scenario in every scene: 3 seeded-
#                                                           #   random published toilet episodes per scene (15), recorded
#   SEGMENTATION=1 gibson/run_benchmark.sh --five-scene 1   # + the Segment Anything service: the target's depth is
#                                                           #   read on the object's own pixels (off by default)
#   gibson/run_benchmark.sh --agent falcon-yolo --five-scene 3   # the FALCON + YOLO + closing baseline (no LLM): 15
#                                                           #   recorded episodes; the same flag before a full-split run
#
# The GPU is handed out by gibson/gpu_plan.py before the services start -- LLM first, YOLO second,
# Habitat third; whatever does not fit runs on the CPU (GPU_PLAN=off restores CPU services). An operator
# pin overrides the budget per process: GPU_PLAN_LLM / GPU_PLAN_YOLO / GPU_PLAN_HABITAT = auto | gpu | cpu
# (GPU_PLAN_LLM=gpu GPU_PLAN_HABITAT=cpu: the LLM on the card, software rendering, YOLO where the VRAM allows).
# Everything else is read from the environment, with defaults for the development laptop;
# see BENCHMARK.md beside this file for the full list and what each one is for.
set -euo pipefail

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# -- where things are ---------------------------------------------------------------------
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OBJNAV_REPO:-$(cd -- "$HERE/../../../../.." && pwd)}"
# The services, their ports, the models, the LLM and the GPU plan are the shared launcher library's defaults;
# only what is Gibson -- its episodes and scenes -- is set here (after DATA, which the library defines).
source "$HERE/../benchmark_services.sh"
objnav_services_defaults
GIBSON_EPISODES_DIR="${GIBSON_EPISODES_DIR:-$DATA/objectnav/gibson/objectnav/gibson/v1.1/val}"
GIBSON_SCENES_DIR="${GIBSON_SCENES_DIR:-$DATA/gibson/scenes}"

# -- the command line -----------------------------------------------------------------------
OUTPUT="" SCENE="" LIMIT="" RECORD=0 PREFLIGHT=0 STATUS="" LEAN=1 FIVE_SCENE="" FIVE_SCENES=() CATEGORY="" AGENT="rpt" EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --agent) AGENT="$2"; EXTRA+=(--agent "$2"); shift 2 ;;   # rpt (default) or falcon-yolo: the baseline needs no LLM
    --scene) SCENE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --five-scene) FIVE_SCENE="$2"; RECORD=1; shift 2 ;;
    --scenes) shift; while [[ $# -gt 0 && "$1" != --* ]]; do FIVE_SCENES+=("$1"); shift; done ;;   # a subset, five-scene mode
    --category) CATEGORY="$2"; EXTRA+=(--category "$2"); shift 2 ;;   # forwarded; also names the default output directory
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
[[ "$SEGMENTATION" == 0 || -f "$SAM_WEIGHTS" ]] || die "SEGMENTATION=1 but no SAM weights at $SAM_WEIGHTS (set SAM_WEIGHTS; mobile_sam.pt or sam2.1_t.pt from the ultralytics assets release)"
if [[ -z "$OUTPUT" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  TAG="${CATEGORY:+${CATEGORY}_}"
  [[ "$AGENT" != rpt ]] && TAG="${AGENT}_${TAG}"
  if [[ -n "$FIVE_SCENE" ]]; then
    SCENE_COUNT=${#FIVE_SCENES[@]}; [[ $SCENE_COUNT -gt 0 ]] || SCENE_COUNT=5
    OUTPUT="$RUNS_ROOT/benchmark_${TAG}$((FIVE_SCENE * SCENE_COUNT))episodes_${STAMP}"
  elif [[ -n "$SCENE" ]]; then OUTPUT="$RUNS_ROOT/gibson_${SCENE}_${TAG}${STAMP}"; else OUTPUT="$RUNS_ROOT/gibson_val_${TAG}${STAMP}"; fi
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

# -- the services: GPU plan, renderer, Ollama, the detector with the Gibson vocabulary, segmenter, FFmpeg ----------
# (benchmark_services.sh: the same code hm3d/run_benchmark.sh runs; the detector takes its classes from the run
# module's --print-vocabulary, in the order the Gibson label table pins)
RUN_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run
objnav_services_up

# -- the five-scene campaign: N recorded episodes per val scene, each a gibson.run job, live dashboard --------
if [[ -n "$FIVE_SCENE" ]]; then
  FIVE_ARGS=(--episodes-dir "$GIBSON_EPISODES_DIR" --scenes-dir "$GIBSON_SCENES_DIR" --output "$OUTPUT"
             --episodes-per-scene "$FIVE_SCENE" --detector-url "$DETECTOR_URL" --detector-backend yolo_world
             --seed "$SEED" --gpu-device "$GPU_DEVICE" --allow-sim-version-mismatch)
  [[ "$AGENT" == rpt ]] && FIVE_ARGS+=(--explorer frontier)       # the baseline explores with FALCON only
  [[ "$ALLOW_SHARED_GPU" == 1 ]] && FIVE_ARGS+=(--allow-shared-gpu)
  [[ ${#FIVE_SCENES[@]} -gt 0 ]] && FIVE_ARGS+=(--scenes "${FIVE_SCENES[@]}")
  [[ ${#SEGMENTER_FLAGS[@]} -gt 0 ]] && FIVE_ARGS+=("${SEGMENTER_FLAGS[@]}")
  [[ ${#RENDER_FLAGS[@]} -gt 0 ]] && FIVE_ARGS+=("${RENDER_FLAGS[@]}")
  FIVE_ARGS+=("${EXTRA[@]}")
  date -u +%FT%TZ > "$SERVICE_DIR/started_utc.txt"
  say "five-scene campaign ($AGENT): $FIVE_SCENE ${CATEGORY:+$CATEGORY }episode(s) per scene, stairs forbidden; results stream to $OUTPUT/benchmark_results.{json,csv}" | tee -a "$LOG"
  set +e
  "${RENDER_WRAP[@]}" "$HAB_PY" -u -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.five_scene "${FIVE_ARGS[@]}" 2>&1 | tee -a "$LOG"
  RUN_CODE=${PIPESTATUS[0]}
  set -e
  [[ "$RUN_CODE" == 0 ]] || die "five_scene exited with $RUN_CODE (see $LOG and $OUTPUT/<Scene>/run.log)"
  say "results: $OUTPUT/benchmark_results.json, benchmark_results.csv, summary.txt, index.html; videos under $OUTPUT/<Scene>/recordings/"
  exit 0
fi

# -- the run ------------------------------------------------------------------------------------
ARGS=(--episodes-dir "$GIBSON_EPISODES_DIR" --scenes-dir "$GIBSON_SCENES_DIR" --output "$OUTPUT"
      --detector-url "$DETECTOR_URL" --detector-backend yolo_world
      --seed "$SEED" --gpu-device "$GPU_DEVICE" --allow-sim-version-mismatch)
[[ "$AGENT" == rpt ]] && ARGS+=(--explorer frontier)              # the baseline explores with FALCON only
[[ -n "$SCENE" ]] && ARGS+=(--scene "$SCENE")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
[[ "$RECORD" == 1 ]] && ARGS+=(--record)
[[ "$LEAN" == 1 ]] && ARGS+=(--lean)
[[ "$ALLOW_SHARED_GPU" == 1 ]] && ARGS+=(--allow-shared-gpu)
[[ -s "$OUTPUT/episodes.jsonl" ]] && { say "resuming: $OUTPUT already holds episodes" | tee -a "$LOG"; ARGS+=(--resume); }
[[ ${#SEGMENTER_FLAGS[@]} -gt 0 ]] && ARGS+=("${SEGMENTER_FLAGS[@]}")
[[ ${#RENDER_FLAGS[@]} -gt 0 ]] && ARGS+=("${RENDER_FLAGS[@]}")
ARGS+=("${EXTRA[@]}")

say "preflight" | tee -a "$LOG"
"${RENDER_WRAP[@]}" "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run "${ARGS[@]}" --preflight > "$OUTPUT/preflight.json" \
  || { cat "$OUTPUT/preflight.json"; die "preflight refused the run (see $OUTPUT/preflight.json)"; }
"$HAB_PY" - "$OUTPUT/preflight.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))["configuration"]
print("  episodes selected: %d%s" % (len(config["selected_episode_ids"]), "  (FULL published split)" if config["full_split"] else ""))
print("  method: %s | lean records: %s | publishable: %s" % (config["method"]["method"], config["lean_records"], config["publishable"]))
llm = config["method"].get("llm")
print("  llm: %s" % ("none (the FALCON + YOLO + closing baseline)" if not llm else
                      "%s (labels) + %s (node oracle) at %s" % (llm["model"], llm["reasoning_model"], llm["base_url"])))
print("  detector: %s" % config["method"]["detector"]["metadata"].get("backend"))
print("  renderer: %s" % config.get("renderer", "cuda gpu"))
segmenter = config["method"].get("segmenter")
print("  segmentation: %s" % ("ON -- %s on %s at %s" % (segmenter["model"], segmenter["device"], config["method"].get("segmenter_url"))
                              if segmenter else "off (box depth)"))
PY
if [[ "$PREFLIGHT" == 1 ]]; then say "preflight only: done"; exit 0; fi

date -u +%FT%TZ > "$OUTPUT/started_utc.txt"
say "running; progress: $OUTPUT/progress.json  (watch: $0 --status $OUTPUT)" | tee -a "$LOG"
set +e
"${RENDER_WRAP[@]}" "$HAB_PY" -u -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run "${ARGS[@]}" 2>&1 | tee -a "$LOG"
RUN_CODE=${PIPESTATUS[0]}
set -e
if [[ "$RUN_CODE" != 0 ]]; then die "gibson.run exited with $RUN_CODE (see $LOG); rerun the same command with --output $OUTPUT to resume"; fi

# -- the summary -------------------------------------------------------------------------------
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.progress "$OUTPUT" --summary | tee -a "$LOG"
say "results: $OUTPUT/summary.json (metrics), comparison.md (vs published), progress.json (per scene/category), episodes.jsonl (one row per episode)"
