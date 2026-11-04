#!/usr/bin/env bash
# One-click MP3D ObjectNav benchmark: data check (the public episodes fetched if absent), services up, preflight,
# one process per scene over the published split, the aggregate, services down.
#
#   mp3d/run_benchmark.sh                                   # the full published val split (2,195 episodes / 11 scenes / 21 categories), lean records
#   mp3d/run_benchmark.sh --jobs 6                          # ... six scenes at once (give Ollama OLLAMA_NUM_PARALLEL slots)
#   mp3d/run_benchmark.sh --jobs max                        # ... as many scenes at once as the card and the cores hold
#   mp3d/run_benchmark.sh --scene 2azQ1b91cZZ --limit 1     # one episode of one scene
#   mp3d/run_benchmark.sh --scene 2azQ1b91cZZ --limit 3 --record  # three recorded episodes (video + HUD) of one scene
#   mp3d/run_benchmark.sh --smoke                           # the minimal cover: one published episode per scene, categories balanced
#   mp3d/run_benchmark.sh --limit 2 --record                # the first two published episodes of every scene, recorded
#   mp3d/run_benchmark.sh --limit 2 --record --save-frames  # ... plus every dashboard frame as a JPEG still (frames/)
#   mp3d/run_benchmark.sh --category toilet --limit 3       # the published toilet episodes, three per scene
#   mp3d/run_benchmark.sh --episode-ids-file cases.json --record --jobs 2
#                                                           # exactly the JSON list's <scene>/<index> episodes (a re-run of chosen cases)
#   mp3d/run_benchmark.sh --output runs/mp3d_v1_val_X       # continue an interrupted campaign there (finished scenes are kept)
#   mp3d/run_benchmark.sh --output runs/X --resume-accepting-changes kinematics evaluation source_sha256 gpu_plan
#                                                           # ... after a harness-only fix: the scenes that continue may differ in the self-check and provenance
#   mp3d/run_benchmark.sh --preflight                       # data + services + configuration check, no episode
#   mp3d/run_benchmark.sh --status runs/mp3d_v1_val_X       # the live progress of a campaign (or of one scene's directory)
#   mp3d/run_benchmark.sh --agent falcon-yolo --smoke       # the FALCON + YOLO + closing baseline (no LLM)
#
# The whole split runs as mp3d/campaign.py: one mp3d.run process per scene, each writing its own complete results
# directory under the output root, then one aggregate (campaign_summary.json) over the scenes that finished. A scene
# that dies costs one scene, and --output <root> resumes the rest. --scene runs mp3d.run directly.
#
# The protocol is habitat-lab's objectnav_mp3d (mp3d/protocol.py, every number transcribed): 500 actions, 0.25 m /
# 30 degree actions, 640x480 RGB-D at HFOV 79 degrees and 0.88 m, depth 0.5-5.0 m, agent 0.18 m / 0.88 m, no sliding,
# success = STOP within 0.1 m geodesic of a published goal view point, measured on the navmesh habitat-sim recomputes
# for this agent (the one habitat-lab's evaluator runs on; VALIDATE_STARTS checks it against the publisher's own
# geodesics). The services (GPU plan, Ollama, the YOLO-World detector started with the MP3D label table's 21-category
# vocabulary, the optional segmenter, FFmpeg) are benchmark_services.sh, shared with gibson/ and hm3d/, so the method's
# services are provisioned the same way on every benchmark. The detector port defaults to 18093 here (18092 HM3D,
# 18095 Gibson): the tables differ, and a detector is pinned to the vocabulary it was started with.
#
# Data: the episodes (objectnav_mp3d_v1, public) are fetched by mp3d/assets.py when absent; the scenes (mp3d_habitat.zip,
# licensed by Matterport) must be installed by the licence holder -- the launcher checks them and says what is missing.
#
# Environment (defaults for the development laptop; the rest is in benchmark_services.sh and mp3d/README.md):
#   MP3D_EPISODES_DIR    the split directory holding <split>.json.gz and content/      default $DATA/objectnav/mp3d/v1/<split>
#   MP3D_SCENES_DIR      habitat-lab's data/scene_datasets/, holding mp3d/<scene>/      default $DATA/scene_datasets
#   HAB_PY               the habitat-sim interpreter; falls back to ~/miniconda3/envs/{habitat,objnav-habitat}/bin/python
#   VALIDATE_STARTS=1    measure every start's shortest path before its scene runs and check it against the publisher's
#                        own geodesic (99% within 1 mm, or the scene refuses to run: the navmesh check)   default 1
#   EXCLUDE_UNREACHABLE  1: drop, and record, an episode whose goal view points are unreachable instead of stopping   default 0
#   SPLIT                val (the benchmark); train / val_mini are development splits and are labelled subsets   default val
#   OLLAMA_NUM_PARALLEL  requests a native Ollama serves at once; --jobs N defaults it to ceil(N/2) (and budgets N
#                        renderers in the GPU plan) unless set here; each slot is its own KV cache          default 1
#   HABITAT_VRAM_MIB     one habitat-sim renderer's VRAM, for the GPU plan and --jobs max (gpu_plan.py's default is
#                        1536; --jobs max sizes at 1000 unless set). Measure it on an MP3D scene with nvidia-smi.
set -euo pipefail

usage() { sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# -- where things are ---------------------------------------------------------------------
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OBJNAV_REPO:-$(cd -- "$HERE/../../../../.." && pwd)}"
DETECTOR_PORT="${DETECTOR_PORT:-18093}"            # before the shared defaults: MP3D's own port, see above
OPERATOR_OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-}"      # the operator's word, before the library's default of 1
source "$HERE/../benchmark_services.sh"
objnav_services_defaults
# The shared default names ~/miniconda3/envs/habitat; this workstation's habitat-sim 0.2.4 environment is objnav-habitat.
if [[ ! -x "$HAB_PY" ]]; then
  for candidate in "$HOME/miniconda3/envs/objnav-habitat/bin/python" "$HOME/miniconda3/envs/habitat/bin/python"; do
    [[ -x "$candidate" ]] && { HAB_PY="$candidate"; break; }
  done
fi
MP3D_SCENES_DIR="${MP3D_SCENES_DIR:-$DATA/scene_datasets}"
SPLIT="${SPLIT:-val}"
VALIDATE_STARTS="${VALIDATE_STARTS:-1}"
EXCLUDE_UNREACHABLE="${EXCLUDE_UNREACHABLE:-0}"
RUN_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.run
CAMPAIGN_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.campaign
ASSETS_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets
PROGRESS_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.progress
POOL_MODULE=sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.scene_pool   # the pool is dataset-agnostic

# -- the command line -----------------------------------------------------------------------
EPISODE_FILE="" VERSION="v1" OUTPUT="" SCENE="" LIMIT="" SMOKE=0 RECORD=0 PREFLIGHT=0 STATUS="" LEAN=1 CATEGORY="" AGENT="rpt" JOBS=1 SCENES=() EXTRA=() ACCEPT=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --agent) AGENT="$2"; EXTRA+=(--agent "$2"); shift 2 ;;   # rpt (default) or falcon-yolo: the baseline needs no LLM
    --perception) PERCEPTION="$2"; shift 2 ;;                  # service only on MP3D (the semantic-sensor oracle has no mpcat40 table yet)
    --scene) SCENE="$2"; shift 2 ;;                            # one scene, run directly by mp3d.run
    --scenes) shift; while [[ $# -gt 0 && "$1" != --* ]]; do SCENES+=("$1"); shift; done ;;   # a subset of the campaign's scenes
    --resume-accepting-changes) ACCEPT=("$1"); shift; while [[ $# -gt 0 && "$1" != --* ]]; do ACCEPT+=("$1"); shift; done ;;
                                                           # a resume's alone (mp3d.run refuses it otherwise): never the preflight's
    --limit) LIMIT="$2"; shift 2 ;;                            # episodes (per scene, in the campaign)
    --jobs) JOBS="$2"; shift 2 ;;                              # scenes at once (campaign only); "max" = what the card and the cores hold
    --smoke) SMOKE=1; shift ;;
    --episode-ids-file) EPISODE_FILE="$2"; shift 2 ;;          # campaign: a JSON list of <scene>/<index> ids, one process per scene
    --category) CATEGORY="$2"; EXTRA+=(--category "$2"); shift 2 ;;   # forwarded; also names the default output directory
    --record) RECORD=1; shift ;;
    --full-records) LEAN=0; shift ;;
    --preflight) PREFLIGHT=1; shift ;;
    --status) STATUS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;            # forwarded to mp3d.run verbatim (e.g. --shuffle, --episode-ids ..., --video-fps 10)
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

case "$VERSION" in v1) SCENE_ROOT=mp3d; PUBLISHED_SCENES=11; PUBLISHED_EPISODES=2195 ;; *) die "--version must be v1 (the one published MP3D ObjectNav dataset)" ;; esac
MP3D_EPISODES_DIR="${MP3D_EPISODES_DIR:-$DATA/objectnav/mp3d/$VERSION/$SPLIT}"
[[ -x "$HAB_PY" ]] || die "Habitat interpreter not found at $HAB_PY (set HAB_PY; see mp3d/README.md section 1)"
[[ "$PERCEPTION" == service ]] || die "--perception must be service on MP3D: the semantic-sensor oracle has no mpcat40 -> goal table here yet"
[[ -x "$DETECT_PY" ]] || die "detector interpreter not found at $DETECT_PY (set DETECT_PY; see mp3d/README.md section 1)"
[[ -f "$YOLO_WEIGHTS" ]] || die "no YOLO-World weights at $YOLO_WEIGHTS (set YOLO_WEIGHTS or MODELS)"
[[ "$SEGMENTATION" == 0 || -f "$SAM_WEIGHTS" ]] || die "SEGMENTATION=1 but no SAM weights at $SAM_WEIGHTS (set SAM_WEIGHTS)"
# -- the data: the public episodes are fetched when absent; the licensed scenes are checked and named when absent ------
if [[ ! -f "$MP3D_EPISODES_DIR/$SPLIT.json.gz" ]]; then
  if [[ "$MP3D_EPISODES_DIR" == "$DATA/objectnav/mp3d/$VERSION/$SPLIT" ]]; then
    say "no $SPLIT.json.gz under $MP3D_EPISODES_DIR: fetching the public objectnav_mp3d_$VERSION episodes (173 MB, no credentials)"
    "$HAB_PY" -m "$ASSETS_MODULE" --version "$VERSION" --split "$SPLIT" --data-root "$DATA" --fetch-episodes \
      || die "could not fetch the episodes; see mp3d/README.md section 1.2"
  else
    die "no $SPLIT.json.gz under $MP3D_EPISODES_DIR (set MP3D_EPISODES_DIR to the split directory, or unset it and the launcher fetches the public release under $DATA; mp3d/README.md section 1.2)"
  fi
fi
[[ -d "$MP3D_SCENES_DIR/$SCENE_ROOT" ]] || die "no Matterport3D scenes at $MP3D_SCENES_DIR/$SCENE_ROOT (set MP3D_SCENES_DIR to habitat-lab's scene_datasets/; the scenes are licensed -- $HAB_PY -m $ASSETS_MODULE --instructions)"
if [[ "$SPLIT" == val ]]; then
  "$HAB_PY" -m "$ASSETS_MODULE" --version "$VERSION" --episodes-dir "$MP3D_EPISODES_DIR" --scenes-dir "$MP3D_SCENES_DIR" --check > /tmp/mp3d_assets_check.$$ \
    || { "$HAB_PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); [print("  " + p) for p in r["problems"]]' /tmp/mp3d_assets_check.$$; rm -f /tmp/mp3d_assets_check.$$; die "the MP3D $SPLIT installation is incomplete (above); $HAB_PY -m $ASSETS_MODULE --instructions"; }
  rm -f /tmp/mp3d_assets_check.$$
fi
[[ "$SMOKE" == 0 || -z "$LIMIT" ]] || die "--smoke chooses one episode per scene itself; it takes no --limit"
[[ "$SMOKE" == 0 || -z "$SCENE" ]] || die "--smoke is a campaign over scenes; use --scene X --limit 1 for one scene"
if [[ "$JOBS" == max ]]; then
  # What the machine holds: one renderer's VRAM per child after the reserve, one core per child, never more than scenes.
  [[ -z "$SCENE" ]] || die "--jobs max runs scenes of a campaign at once; a --scene run is one process"
  CAMPAIGN_SCENES=$PUBLISHED_SCENES; [[ ${#SCENES[@]} -gt 0 ]] && CAMPAIGN_SCENES=${#SCENES[@]}
  # HM3D sizes this at 400 MiB per renderer (measured there); Matterport meshes are two to three times larger
  # (18-117 MB of basis-compressed glb against HM3D val's 14-76 MB), so until an MP3D renderer is measured the
  # sizing stays conservative. Set HABITAT_VRAM_MIB to what nvidia-smi shows for one mp3d.run process.
  JOBS="$(HABITAT_VRAM_MIB="${HABITAT_VRAM_MIB:-1000}" "$HAB_PY" -m "$POOL_MODULE" --scenes "$CAMPAIGN_SCENES" --gpu-device "$GPU_DEVICE" --explain)" \
    || die "could not size --jobs max (nvidia-smi?); pass a number"
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer or max"
[[ "$JOBS" == 1 || -z "$SCENE" ]] || die "--jobs runs scenes of a campaign at once; a --scene run is one process"
if [[ "$JOBS" -gt 1 ]]; then
  # As hm3d/run_benchmark.sh: the GPU plan budgets one renderer per child, and the native Ollama gets ceil(N/2)
  # request slots unless the operator named a number (the node oracle is about a third of a child's wall time).
  HABITAT_RENDERERS="$JOBS"
  [[ -n "$OPERATOR_OLLAMA_NUM_PARALLEL" ]] || OLLAMA_NUM_PARALLEL=$(( (JOBS + 1) / 2 ))
fi
if [[ -z "$OUTPUT" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  TAG="${CATEGORY:+${CATEGORY}_}"
  [[ "$AGENT" != rpt ]] && TAG="${AGENT}_${TAG}"
  if [[ -n "$SCENE" ]]; then OUTPUT="$RUNS_ROOT/mp3d_${VERSION}_${SCENE}_${TAG}${STAMP}"
  elif [[ "$SMOKE" == 1 ]]; then OUTPUT="$RUNS_ROOT/mp3d_${VERSION}_${SPLIT}_smoke_${TAG}${STAMP}"
  elif [[ -n "$LIMIT" ]]; then OUTPUT="$RUNS_ROOT/mp3d_${VERSION}_${SPLIT}_${TAG}${LIMIT}per-scene_${STAMP}"
  else OUTPUT="$RUNS_ROOT/mp3d_${VERSION}_${SPLIT}_${TAG}${STAMP}"; fi
fi
# The campaign root holds one directory per scene and the service files beside them (files only: a stray
# subdirectory would read as a scene without rows); a --scene run owns its directory like hm3d.run does.
SERVICE_DIR="$OUTPUT"
mkdir -p "$SERVICE_DIR"
LOG="$SERVICE_DIR/run.log"
say "repo $REPO" | tee -a "$LOG"
say "output $OUTPUT" | tee -a "$LOG"
say "MP3D $VERSION $SPLIT: episodes $MP3D_EPISODES_DIR, scenes $MP3D_SCENES_DIR/$SCENE_ROOT" | tee -a "$LOG"
git --no-pager rev-parse HEAD > "$SERVICE_DIR/source-commit.txt" 2>/dev/null || true
git --no-pager diff --no-ext-diff > "$SERVICE_DIR/source-changes.diff" 2>/dev/null || true

# -- the services: GPU plan, renderer, Ollama, the detector with the MP3D vocabulary, segmenter, FFmpeg -------------
objnav_services_up

# -- the arguments every mp3d.run shares, campaign child or single scene ------------------------------------------
# DATA_ARGS name the split (the campaign passes them to its children itself); CHILD_ARGS are the method, the services
# and the records; VALIDATE_ARGS load a scene, so they join the run but never the mesh-free preflight.
DATA_ARGS=(--version "$VERSION" --split "$SPLIT" --episodes-dir "$MP3D_EPISODES_DIR" --scenes-dir "$MP3D_SCENES_DIR")
CHILD_ARGS=(--seed "$SEED" --gpu-device "$GPU_DEVICE" --detector-url "$DETECTOR_URL" --detector-backend yolo_world)
[[ "$AGENT" == rpt ]] && CHILD_ARGS+=(--explorer frontier)        # as gibson/ and hm3d/: the baseline explores with FALCON only
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

# -- one scene: mp3d.run directly, resumable in place ---------------------------------------------------------------
if [[ -n "$SCENE" ]]; then
  RUN_ARGS=("${DATA_ARGS[@]}" "${CHILD_ARGS[@]}" "${VALIDATE_ARGS[@]}" --scenes "$SCENE" --output "$OUTPUT")
  [[ -n "$LIMIT" ]] && RUN_ARGS+=(--limit "$LIMIT")
  [[ -s "$OUTPUT/episodes.jsonl" ]] && { say "resuming: $OUTPUT already holds episodes" | tee -a "$LOG"; RUN_ARGS+=(--resume "${ACCEPT[@]}"); }
  say "running $SCENE; progress: $OUTPUT/progress.json  (watch: $0 --status $OUTPUT)" | tee -a "$LOG"
  set +e
  "${RENDER_WRAP[@]}" "$HAB_PY" -u -m "$RUN_MODULE" "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG"
  RUN_CODE=${PIPESTATUS[0]}
  set -e
  [[ "$RUN_CODE" == 0 ]] || die "mp3d.run exited with $RUN_CODE (see $LOG); rerun the same command with --output $OUTPUT to resume"
  "$HAB_PY" -m "$PROGRESS_MODULE" "$OUTPUT" --summary | tee -a "$LOG"
  say "results: $OUTPUT/summary.json (metrics), comparison.md + audit.json (vs published, both success radii), episodes.jsonl${RECORD:+, recordings/}"
  exit 0
fi

# -- the campaign: every scene of the split (or --scenes), one process each, then the aggregate ---------------------
CAMPAIGN_ARGS=("$OUTPUT" --version "$VERSION" --split "$SPLIT" --episodes-dir "$MP3D_EPISODES_DIR" --scenes-dir "$MP3D_SCENES_DIR")
[[ -n "$LIMIT" ]] && CAMPAIGN_ARGS+=(--limit "$LIMIT")
CAMPAIGN_ARGS+=(--jobs "$JOBS")
[[ "$SMOKE" == 1 ]] && CAMPAIGN_ARGS+=(--smoke)
[[ -n "$EPISODE_FILE" ]] && CAMPAIGN_ARGS+=(--episode-ids-file "$EPISODE_FILE")
[[ ${#SCENES[@]} -gt 0 ]] && CAMPAIGN_ARGS+=(--scenes "${SCENES[@]}")
if [[ -f "$OUTPUT/campaign.json" ]] || ls -d "$OUTPUT"/*/ > /dev/null 2>&1; then
  say "resuming: $OUTPUT already holds scenes (finished ones are kept, started ones continue)" | tee -a "$LOG"
  CAMPAIGN_ARGS+=(--resume)
fi
# mp3d.run's --version/--split/--episodes-dir/--scenes-dir come from the campaign itself; the rest is passed through after "--".
say "campaign ($AGENT): $(if [[ "$SMOKE" == 1 ]]; then echo "the smoke cover, one episode per scene"; elif [[ -n "$LIMIT" ]]; then echo "$LIMIT ${CATEGORY:+$CATEGORY }episode(s) per scene"; else echo "every ${CATEGORY:+$CATEGORY }episode of the split ($PUBLISHED_EPISODES over $PUBLISHED_SCENES scenes)"; fi), one process per scene, $JOBS at once; per-scene progress under $OUTPUT/<scene>/progress.json  (watch: $0 --status $OUTPUT)" | tee -a "$LOG"
set +e
"${RENDER_WRAP[@]}" "$HAB_PY" -u -m "$CAMPAIGN_MODULE" "${CAMPAIGN_ARGS[@]}" -- "${CHILD_ARGS[@]}" "${VALIDATE_ARGS[@]}" "${ACCEPT[@]}" 2>&1 | tee -a "$LOG"
RUN_CODE=${PIPESTATUS[0]}
set -e
[[ "$RUN_CODE" == 0 ]] || die "the campaign exited with $RUN_CODE (see $LOG and $OUTPUT/<scene>/run.json); rerun the same command with --output $OUTPUT to resume the unfinished scenes"
say "results: $OUTPUT/campaign_summary.json (the split's SR/SPL/SoftSPL and whether it is complete), campaign_metrics.csv (per scene), <scene>/comparison.md (vs published)${RECORD:+, <scene>/recordings/}"
