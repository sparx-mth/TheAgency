#!/usr/bin/env bash
# One-click HM3D ObjectNav benchmark: services up, preflight, one process per scene over the split, the aggregate, services down.
#
#   hm3d/run_benchmark.sh                                   # the full published v2 val split (1,000 episodes / 36 scenes), lean records
#   hm3d/run_benchmark.sh --version v1                      # the v1 val split (2,000 episodes / 20 scenes)
#   hm3d/run_benchmark.sh --scene 00877-4ok3usBNeis --limit 1           # one episode of one scene
#   hm3d/run_benchmark.sh --scene 00877-4ok3usBNeis --limit 3 --record  # three recorded episodes (video + HUD) of one scene
#   hm3d/run_benchmark.sh --smoke                           # the minimal cover: one published episode per scene, every category at least once
#   hm3d/run_benchmark.sh --limit 2 --record                # the first two published episodes of every scene, recorded
#   hm3d/run_benchmark.sh --category toilet --limit 3       # the published toilet episodes, three per scene
#   hm3d/run_benchmark.sh --output runs/hm3d_v2_val_X       # continue an interrupted campaign there (finished scenes are kept)
#   hm3d/run_benchmark.sh --preflight                       # services + data + configuration check, no episode
#   hm3d/run_benchmark.sh --status runs/hm3d_v2_val_X       # the live progress of a campaign (or of one scene's directory)
#   hm3d/run_benchmark.sh --agent falcon-yolo --smoke       # the FALCON + YOLO + closing baseline (no LLM)
#
# The whole split runs as hm3d/campaign.py: one hm3d.run process per scene, each writing its own complete results
# directory under the output root, then one aggregate (campaign_summary.json) over the scenes that finished. A
# scene that dies costs one scene, and --output <root> resumes the rest. --scene runs hm3d.run directly.
#
# The services (GPU plan, Ollama, the YOLO-World detector started with the HM3D label table's vocabulary, the optional
# segmenter, FFmpeg) are benchmark_services.sh, shared with gibson/run_benchmark.sh, so the method's services are
# provisioned the same way on both benchmarks. The detector port defaults to 18092 here, 18095 for Gibson: the two
# tables share their words but not their order, and a detector is pinned to its order.
#
# Environment (defaults for the development laptop; the rest is in benchmark_services.sh and hm3d/README.md):
#   HM3D_EPISODES_DIR    the split directory holding <split>.json.gz and content/      default $DATA/objectnav/hm3d/<version>/<split>
#   HM3D_SCENES_DIR      habitat-lab's data/scene_datasets/, holding hm3d/ (v1) or hm3d_v0.2/ (v2)   default $DATA/scene_datasets
#   VALIDATE_STARTS=1    measure every start's shortest path before its scene runs and check it against the publisher's own
#                        geodesic (99% within 1 mm, or the scene refuses to run: that is the navmesh check)   default 1
#   EXCLUDE_UNREACHABLE  1: drop, and record, an episode whose goal view points are unreachable instead of stopping   default 0
#   SPLIT                val (the benchmark); train / val_mini are development splits and are labelled subsets   default val
set -euo pipefail

usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# -- where things are ---------------------------------------------------------------------
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OBJNAV_REPO:-$(cd -- "$HERE/../../../../.." && pwd)}"
DETECTOR_PORT="${DETECTOR_PORT:-18092}"            # before the shared defaults: HM3D's own port, see above
source "$HERE/../benchmark_services.sh"
objnav_services_defaults
HM3D_SCENES_DIR="${HM3D_SCENES_DIR:-$DATA/scene_datasets}"
SPLIT="${SPLIT:-val}"
VALIDATE_STARTS="${VALIDATE_STARTS:-1}"
EXCLUDE_UNREACHABLE="${EXCLUDE_UNREACHABLE:-0}"
RUN_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run
CAMPAIGN_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign
PROGRESS_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.progress

# -- the command line -----------------------------------------------------------------------
VERSION="v2" OUTPUT="" SCENE="" LIMIT="" SMOKE=0 RECORD=0 PREFLIGHT=0 STATUS="" LEAN=1 CATEGORY="" AGENT="rpt" SCENES=() EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --agent) AGENT="$2"; EXTRA+=(--agent "$2"); shift 2 ;;   # rpt (default) or falcon-yolo: the baseline needs no LLM
    --scene) SCENE="$2"; shift 2 ;;                            # one scene, run directly by hm3d.run
    --scenes) shift; while [[ $# -gt 0 && "$1" != --* ]]; do SCENES+=("$1"); shift; done ;;   # a subset of the campaign's scenes
    --limit) LIMIT="$2"; shift 2 ;;                            # episodes (per scene, in the campaign)
    --smoke) SMOKE=1; shift ;;
    --category) CATEGORY="$2"; EXTRA+=(--category "$2"); shift 2 ;;   # forwarded; also names the default output directory
    --record) RECORD=1; shift ;;
    --full-records) LEAN=0; shift ;;
    --preflight) PREFLIGHT=1; shift ;;
    --status) STATUS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;            # forwarded to hm3d.run verbatim (e.g. --shuffle, --episode-ids ..., --video-fps 10)
  esac
done

if [[ -n "$STATUS" ]]; then
  # A campaign root holds one directory per scene; a scene (or a --scene run) holds progress.json itself.
  if [[ -f "$STATUS/progress.json" ]]; then exec "$HAB_PY" -m "$PROGRESS_MODULE" "$STATUS"; fi
  FOUND=0
  for scene_dir in "$STATUS"/*/; do
    [[ -f "$scene_dir/progress.json" ]] || continue
    FOUND=1; printf '%s: ' "$(basename "$scene_dir")"; "$HAB_PY" -m "$PROGRESS_MODULE" "$scene_dir" || true
  done
  [[ "$FOUND" == 1 ]] || echo "no progress.json under $STATUS yet"
  [[ -f "$STATUS/campaign_summary.json" ]] && "$HAB_PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); o=r["overall"]; print("campaign: %d episodes in %d scene(s)%s | SR %.3f SPL %.3f SoftSPL %.3f | complete split: %s" % (r["episodes"], r["scenes_with_rows"], " (%d without rows)" % len(r["scenes_without_rows"]) if r["scenes_without_rows"] else "", o["success_rate"], o["spl"], o["soft_spl"], r["complete"]))' "$STATUS/campaign_summary.json"
  exit 0
fi

cd "$REPO"
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

case "$VERSION" in v1) SCENE_ROOT=hm3d ;; v2) SCENE_ROOT=hm3d_v0.2 ;; *) die "--version must be v1 or v2" ;; esac
HM3D_EPISODES_DIR="${HM3D_EPISODES_DIR:-$DATA/objectnav/hm3d/$VERSION/$SPLIT}"
[[ -x "$HAB_PY" ]] || die "Habitat interpreter not found at $HAB_PY (set HAB_PY; see hm3d/README.md section 1)"
[[ -x "$DETECT_PY" ]] || die "detector interpreter not found at $DETECT_PY (set DETECT_PY; see hm3d/README.md section 1)"
[[ -f "$HM3D_EPISODES_DIR/$SPLIT.json.gz" ]] || die "no $SPLIT.json.gz under $HM3D_EPISODES_DIR (set HM3D_EPISODES_DIR to the split directory; hm3d/README.md section 1.2)"
[[ -d "$HM3D_SCENES_DIR/$SCENE_ROOT/$SPLIT" ]] || die "no HM3D $VERSION scenes at $HM3D_SCENES_DIR/$SCENE_ROOT/$SPLIT (set HM3D_SCENES_DIR to habitat-lab's scene_datasets/; the scenes are licensed -- hm3d/README.md section 1.2)"
[[ -f "$YOLO_WEIGHTS" ]] || die "no YOLO-World weights at $YOLO_WEIGHTS (set YOLO_WEIGHTS or MODELS)"
[[ "$SEGMENTATION" == 0 || -f "$SAM_WEIGHTS" ]] || die "SEGMENTATION=1 but no SAM weights at $SAM_WEIGHTS (set SAM_WEIGHTS)"
[[ "$SMOKE" == 0 || -z "$LIMIT" ]] || die "--smoke chooses one episode per scene itself; it takes no --limit"
[[ "$SMOKE" == 0 || -z "$SCENE" ]] || die "--smoke is a campaign over scenes; use --scene X --limit 1 for one scene"
if [[ -z "$OUTPUT" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  TAG="${CATEGORY:+${CATEGORY}_}"
  [[ "$AGENT" != rpt ]] && TAG="${AGENT}_${TAG}"
  if [[ -n "$SCENE" ]]; then OUTPUT="$RUNS_ROOT/hm3d_${VERSION}_${SCENE}_${TAG}${STAMP}"
  elif [[ "$SMOKE" == 1 ]]; then OUTPUT="$RUNS_ROOT/hm3d_${VERSION}_${SPLIT}_smoke_${TAG}${STAMP}"
  elif [[ -n "$LIMIT" ]]; then OUTPUT="$RUNS_ROOT/hm3d_${VERSION}_${SPLIT}_${TAG}${LIMIT}per-scene_${STAMP}"
  else OUTPUT="$RUNS_ROOT/hm3d_${VERSION}_${SPLIT}_${TAG}${STAMP}"; fi
fi
# The campaign root holds one directory per scene and the service files beside them (files only: a stray
# subdirectory would read as a scene without rows); a --scene run owns its directory like gibson.run does.
SERVICE_DIR="$OUTPUT"
mkdir -p "$SERVICE_DIR"
LOG="$SERVICE_DIR/run.log"
say "repo $REPO" | tee -a "$LOG"
say "output $OUTPUT" | tee -a "$LOG"
say "HM3D $VERSION $SPLIT: episodes $HM3D_EPISODES_DIR, scenes $HM3D_SCENES_DIR/$SCENE_ROOT/$SPLIT" | tee -a "$LOG"
git --no-pager rev-parse HEAD > "$SERVICE_DIR/source-commit.txt" 2>/dev/null || true
git --no-pager diff --no-ext-diff > "$SERVICE_DIR/source-changes.diff" 2>/dev/null || true

# -- the services: GPU plan, renderer, Ollama, the detector with the HM3D vocabulary, segmenter, FFmpeg ------------
objnav_services_up

# -- the arguments every hm3d.run shares, campaign child or single scene ------------------------------------------
# DATA_ARGS name the split (the campaign passes them to its children itself); CHILD_ARGS are the method, the services
# and the records; VALIDATE_ARGS load a scene, so they join the run but never the mesh-free preflight.
DATA_ARGS=(--version "$VERSION" --split "$SPLIT" --episodes-dir "$HM3D_EPISODES_DIR" --scenes-dir "$HM3D_SCENES_DIR")
CHILD_ARGS=(--detector-url "$DETECTOR_URL" --detector-backend yolo_world --seed "$SEED" --gpu-device "$GPU_DEVICE")
[[ "$AGENT" == rpt ]] && CHILD_ARGS+=(--explorer frontier)        # as gibson/run_benchmark.sh: the baseline explores with FALCON only
[[ "$RECORD" == 1 ]] && CHILD_ARGS+=(--record)
[[ "$LEAN" == 1 ]] && CHILD_ARGS+=(--lean)
[[ "$ALLOW_SHARED_GPU" == 1 ]] && CHILD_ARGS+=(--allow-shared-gpu)
[[ ${#SEGMENTER_FLAGS[@]} -gt 0 ]] && CHILD_ARGS+=("${SEGMENTER_FLAGS[@]}")
[[ ${#RENDER_FLAGS[@]} -gt 0 ]] && CHILD_ARGS+=("${RENDER_FLAGS[@]}")
CHILD_ARGS+=("${EXTRA[@]}")
VALIDATE_ARGS=()
[[ "$VALIDATE_STARTS" == 1 ]] && VALIDATE_ARGS+=(--validate-starts)
[[ "$EXCLUDE_UNREACHABLE" == 1 ]] && VALIDATE_ARGS+=(--exclude-unreachable)

# -- preflight: the services, the episode files, the label table and the frozen configuration, without a mesh ------
# (--skip-scene-hashes: the children hash their own scene; hashing the whole split here would be minutes for nothing)
PREFLIGHT_ARGS=("${DATA_ARGS[@]}" "${CHILD_ARGS[@]}" --preflight --skip-scene-hashes)
if [[ -n "$SCENE" ]]; then
  PREFLIGHT_ARGS+=(--scenes "$SCENE")
  [[ -n "$LIMIT" ]] && PREFLIGHT_ARGS+=(--limit "$LIMIT")
elif [[ ${#SCENES[@]} -gt 0 ]]; then PREFLIGHT_ARGS+=(--scenes "${SCENES[@]}" --limit 1)
else PREFLIGHT_ARGS+=(--limit 1); fi
say "preflight" | tee -a "$LOG"
"${RENDER_WRAP[@]}" "$HAB_PY" -m "$RUN_MODULE" "${PREFLIGHT_ARGS[@]}" > "$SERVICE_DIR/preflight.json" \
  || { cat "$SERVICE_DIR/preflight.json"; die "preflight refused the run (see $SERVICE_DIR/preflight.json)"; }
"$HAB_PY" - "$SERVICE_DIR/preflight.json" "$SMOKE" "$SCENE" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))["configuration"]
print("  protocol: %s | habitat-sim %s (reference match: %s) | navmesh: %s" % (
    config["protocol_id"], config["runtime"].get("habitat-sim"), config["reference_habitat_sim_version_match"], config["protocol"]["navmesh"]))
if sys.argv[3]:
    print("  episodes selected in %s: %d" % (sys.argv[3], len(config["selected_episode_ids"])))
else:
    print("  the preflight selected %d episode(s) to check the configuration; the campaign runs %s" % (
        len(config["selected_episode_ids"]), "the smoke cover" if sys.argv[2] == "1" else "every scene of the split"))
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
date -u +%FT%TZ > "$SERVICE_DIR/started_utc.txt"

# -- one scene: hm3d.run directly, resumable in place ---------------------------------------------------------------
if [[ -n "$SCENE" ]]; then
  RUN_ARGS=("${DATA_ARGS[@]}" "${CHILD_ARGS[@]}" "${VALIDATE_ARGS[@]}" --scenes "$SCENE" --output "$OUTPUT")
  [[ -n "$LIMIT" ]] && RUN_ARGS+=(--limit "$LIMIT")
  [[ -s "$OUTPUT/episodes.jsonl" ]] && { say "resuming: $OUTPUT already holds episodes" | tee -a "$LOG"; RUN_ARGS+=(--resume); }
  say "running $SCENE; progress: $OUTPUT/progress.json  (watch: $0 --status $OUTPUT)" | tee -a "$LOG"
  set +e
  "${RENDER_WRAP[@]}" "$HAB_PY" -u -m "$RUN_MODULE" "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG"
  RUN_CODE=${PIPESTATUS[0]}
  set -e
  [[ "$RUN_CODE" == 0 ]] || die "hm3d.run exited with $RUN_CODE (see $LOG); rerun the same command with --output $OUTPUT to resume"
  "$HAB_PY" -m "$PROGRESS_MODULE" "$OUTPUT" --summary | tee -a "$LOG"
  say "results: $OUTPUT/summary.json (metrics), comparison.md + audit.json (vs published, both success radii), episodes.jsonl${RECORD:+, recordings/}"
  exit 0
fi

# -- the campaign: every scene of the split (or --scenes), one process each, then the aggregate ---------------------
CAMPAIGN_ARGS=("$OUTPUT" --version "$VERSION" --split "$SPLIT" --episodes-dir "$HM3D_EPISODES_DIR" --scenes-dir "$HM3D_SCENES_DIR")
[[ -n "$LIMIT" ]] && CAMPAIGN_ARGS+=(--limit "$LIMIT")
[[ "$SMOKE" == 1 ]] && CAMPAIGN_ARGS+=(--smoke)
[[ ${#SCENES[@]} -gt 0 ]] && CAMPAIGN_ARGS+=(--scenes "${SCENES[@]}")
if [[ -f "$OUTPUT/campaign.json" ]] || ls -d "$OUTPUT"/*/ > /dev/null 2>&1; then
  say "resuming: $OUTPUT already holds scenes (finished ones are kept, started ones continue)" | tee -a "$LOG"
  CAMPAIGN_ARGS+=(--resume)
fi
# hm3d.run's --version/--split/--episodes-dir/--scenes-dir come from the campaign itself; the rest is passed through after "--".
say "campaign ($AGENT): $(if [[ "$SMOKE" == 1 ]]; then echo "the smoke cover, one episode per scene"; elif [[ -n "$LIMIT" ]]; then echo "$LIMIT ${CATEGORY:+$CATEGORY }episode(s) per scene"; else echo "every ${CATEGORY:+$CATEGORY }episode of the split"; fi), one process per scene; per-scene progress under $OUTPUT/<scene>/progress.json  (watch: $0 --status $OUTPUT)" | tee -a "$LOG"
set +e
"${RENDER_WRAP[@]}" "$HAB_PY" -u -m "$CAMPAIGN_MODULE" "${CAMPAIGN_ARGS[@]}" -- "${CHILD_ARGS[@]}" "${VALIDATE_ARGS[@]}" 2>&1 | tee -a "$LOG"
RUN_CODE=${PIPESTATUS[0]}
set -e
[[ "$RUN_CODE" == 0 ]] || die "the campaign exited with $RUN_CODE (see $LOG and $OUTPUT/<scene>/run.json); rerun the same command with --output $OUTPUT to resume the unfinished scenes"
say "results: $OUTPUT/campaign_summary.json (the split's SR/SPL/SoftSPL and whether it is complete), campaign_metrics.csv (per scene), <scene>/comparison.md (vs published)${RECORD:+, <scene>/recordings/}"
