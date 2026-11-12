#!/usr/bin/env bash
# The look studies (2026-11-11): the warm-up and the local-exploration comparisons, one arm per command.
#
#   mp3d/look_study.sh <study> <arm> [--dry-run] [--jobs N] [--resume] [-- extra launcher args]
#   mp3d/look_study.sh <study> report [--explain]
#
#   study: warmup   -- the episode's first look: `without` (the ten-turn spin) against the vantage warm-up arms
#          explore  -- the room visit:          `without` (the frontier walk) against the viewpoint explore visit
#   arm:   one of the study's arms (see `look_study.py`, STUDIES: `without` first, then the variants by priority),
#          or `report` for the table over every arm that has run.
#
# Every arm runs the SAME fixed episode subset (sets/warmup_study_v1.json: 39 single-storey MP3D val episodes in 8
# scenes) under ground-truth perception with the same LLM at temperature 0 and seed 17. The `without` arm RECORDS
# every LLM answer (runs/look_study/llm_answers/without); every other arm REPLAYS that store -- and every other
# arm's store already present -- and records only what it asks anew, so two arms differ only where the change
# under test acts (an answer is a function of its question in its episode, whichever arm asked first). `without`
# is shared by both studies: run it once. Each arm runs from a frozen snapshot of the tree
# (~/GIT/TheAgency_snapshots/ls_<arm>), so the checkout may be edited while an arm runs; --resume continues an
# interrupted arm from its snapshot.
#
# --dry-run prints the plan (arm, config, episodes, model, stores, renderer, workers, a time estimate), checks the
# interpreter, the data, the LLM server, the GPU and the disk, and starts nothing; a problem is one line that says
# what is missing and what to do (look_study.py).
#
# Environment (laptop defaults; the main machine's campaign used the 14B at :11435):
#   HAB_PY             the habitat-sim interpreter            default ~/miniconda3/envs/habitat/bin/python (or objnav-habitat)
#   DATA               datasets root                          default ~/datasets
#   MP3D_EPISODES_DIR  the val split                          default $DATA/objectnav/mp3d/v1/val
#   MP3D_SCENES_DIR    habitat-lab's scene_datasets/          default $DATA/scene_datasets  (holds mp3d/<scene>/...)
#   LLM_BASE_URL       the Ollama server                      default http://127.0.0.1:11434 (started if absent, OLLAMA_MODE=native)
#   LLM_MODEL, LLM_REASONING_MODEL                            default qwen2.5:3b-instruct for both roles
#   HABITAT_RENDER     gpu | software                         default gpu
#   JOBS               scene processes at once                default 2 (--jobs)
#   SECONDS_PER_ACTION the dry run's time estimate            default 1.2
#   RUNS_ROOT, SNAPSHOT_ROOT, STALL_MINUTES, HABITAT_VRAM_MIB as in run_single_storey.sh
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../../../.." && pwd)"
usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
STUDY="${1:-}"; ARM="${2:-}"
[[ -n "$STUDY" && -n "$ARM" ]] || { usage; exit 2; }
[[ "$STUDY" == warmup || "$STUDY" == explore ]] || { echo "study must be warmup or explore" >&2; exit 2; }
shift 2
JOBS="${JOBS:-2}" DRY=0 RESUME=0 EXPLAIN=0 EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --explain) EXPLAIN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

export HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/habitat/bin/python}"
if [[ ! -x "$HAB_PY" ]]; then
  for candidate in "$HOME/miniconda3/envs/objnav-habitat/bin/python" "$HOME/miniconda3/envs/habitat/bin/python"; do
    [[ -x "$candidate" ]] && { HAB_PY="$candidate"; break; }
  done
fi
DATA="${DATA:-$HOME/datasets}"
export MP3D_EPISODES_DIR="${MP3D_EPISODES_DIR:-$DATA/objectnav/mp3d/v1/val}"
export MP3D_SCENES_DIR="${MP3D_SCENES_DIR:-$DATA/scene_datasets}"
export OLLAMA_MODE="${OLLAMA_MODE:-native}" LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:3b-instruct}" LLM_REASONING_MODEL="${LLM_REASONING_MODEL:-qwen2.5:3b-instruct}"
export HABITAT_RENDER="${HABITAT_RENDER:-gpu}"
export GPU_PLAN_LLM="${GPU_PLAN_LLM:-gpu}" HABITAT_VRAM_MIB="${HABITAT_VRAM_MIB:-1100}" ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-1}"
RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$HOME/GIT/TheAgency_snapshots}"
STALL_MINUTES="${STALL_MINUTES:-30}"
SETS="$HERE/sets"
SET="${SET:-$SETS/warmup_study_v1.json}"
STORE_ROOT="$RUNS_ROOT/look_study/llm_answers"
OUTPUT="$RUNS_ROOT/look_study/$ARM"
PY_STUDY=sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.look_study
PY_REPORT=sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.warmup_study

# -- the report ------------------------------------------------------------------------------------------------
if [[ "$ARM" == report ]]; then
  ARMS=$("$HAB_PY" -c "from $PY_STUDY import STUDIES; print(' '.join(STUDIES['$STUDY']))")
  SPECS=()
  for arm in $ARMS; do
    if ls "$RUNS_ROOT/look_study/$arm"/*/episodes.jsonl > /dev/null 2>&1; then SPECS+=("$arm=$RUNS_ROOT/look_study/$arm"); fi
  done
  [[ ${#SPECS[@]} -ge 1 ]] || { echo "no arm of the $STUDY study has run yet under $RUNS_ROOT/look_study" >&2; exit 1; }
  [[ "${SPECS[0]}" == without=* ]] || { echo "the \`without\` arm has not run: the table is paired against it" >&2; exit 1; }
  mkdir -p "$RUNS_ROOT/look_study"
  FLAGS=(--set "$SET" --markdown "$RUNS_ROOT/look_study/${STUDY}_report.md" --json "$RUNS_ROOT/look_study/${STUDY}_report.json")
  [[ "$EXPLAIN" == 1 ]] && FLAGS+=(--explain)
  cd "$REPO" && exec "$HAB_PY" -m "$PY_REPORT" report "${FLAGS[@]}" "${SPECS[@]}"
fi

# -- the plan, and the dry run -----------------------------------------------------------------------------------
cd "$REPO"
"$HAB_PY" -m "$PY_STUDY" --study "$STUDY" --arm "$ARM" --set "$SET" --sets-dir "$SETS" --runs-root "$RUNS_ROOT" \
    --hab-py "$HAB_PY" --episodes-dir "$MP3D_EPISODES_DIR" --scenes-dir "$MP3D_SCENES_DIR" \
    --llm-base-url "$LLM_BASE_URL" --llm-model "$LLM_MODEL" --llm-reasoning-model "$LLM_REASONING_MODEL" \
    --jobs "$JOBS" --renderer "$HABITAT_RENDER" --seconds-per-action "${SECONDS_PER_ACTION:-1.2}" $([[ "$DRY" == 1 ]] && echo --all-arms) \
  || { echo; echo "fix the lines above, then run again" >&2; exit 1; }
[[ "$DRY" == 1 ]] && exit 0

# -- the arm ------------------------------------------------------------------------------------------------------
CONFIG=$("$HAB_PY" -c "from $PY_STUDY import ARM_CONFIGS; print(ARM_CONFIGS.get('$ARM') or '')")
SNAPSHOT="$SNAPSHOT_ROOT/ls_$ARM"
if [[ -d "$SNAPSHOT" ]]; then
  echo "snapshot $SNAPSHOT exists: kept (a resume runs the code the arm started with)"
else
  mkdir -p "$SNAPSHOT"
  ( cd "$REPO" && { git ls-files; git ls-files --others --exclude-standard; } | rsync -a --files-from=- "$REPO/" "$SNAPSHOT/" )
  ( cd "$REPO" && git --no-pager rev-parse HEAD > "$SNAPSHOT/SNAPSHOT_COMMIT.txt" \
      && git --no-pager diff --no-ext-diff > "$SNAPSHOT/SNAPSHOT_CHANGES.diff" \
      && git ls-files --others --exclude-standard > "$SNAPSHOT/SNAPSHOT_UNTRACKED.txt" )
  echo "snapshot of $REPO at $(cat "$SNAPSHOT/SNAPSHOT_COMMIT.txt") -> $SNAPSHOT"
fi
mkdir -p "$STORE_ROOT/$ARM"
export LLM_ANSWERS_RECORD="$STORE_ROOT/$ARM"
if [[ "$ARM" != without ]]; then
  # Replay every arm's store but this one's: two arms that reach the same map state ask the same question and
  # must get the same answer, whichever ran first -- the server at temperature 0 does not guarantee it
  # (LESSONS.md 2026-10-31; on the main machine two arms with IDENTICAL warm-ups differed by 3 success flips).
  REPLAY_DIRS=()
  for dir in "$STORE_ROOT"/*/; do
    dir="${dir%/}"
    [[ "$(basename "$dir")" == "$ARM" ]] && continue
    ls "$dir"/answers-*.jsonl > /dev/null 2>&1 && REPLAY_DIRS+=("$dir")
  done
  export LLM_ANSWERS_REPLAY="$(IFS=:; echo "${REPLAY_DIRS[*]}")"
fi
[[ -n "$CONFIG" ]] && EXTRA+=(--policy-config "$SETS/$CONFIG")
[[ "$RESUME" == 1 ]] && EXTRA+=(--resume-accepting-changes gpu_plan source_sha256)
export OBJNAV_REPO="$SNAPSHOT" RUNS_ROOT
cd "$SNAPSHOT"
echo "arm $ARM of the $STUDY study: set $SET, record $LLM_ANSWERS_RECORD, replay ${LLM_ANSWERS_REPLAY:-none}, output $OUTPUT, jobs $JOBS"
exec "$SNAPSHOT/sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/run_benchmark.sh" \
  --perception ground-truth --episode-ids-file "$SET" --record-log-only --full-records --jobs "$JOBS" \
  --output "$OUTPUT" --stall-minutes "$STALL_MINUTES" "${EXTRA[@]}"
