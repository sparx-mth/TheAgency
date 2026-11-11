#!/usr/bin/env bash
# A/B of the room size and coverage mechanism on the fixed 200-episode set: the code before it against the code with it.
#
#   hm3d/ab_room_extents.sh [--output runs/ab_room_extents_<stamp>] [--jobs 2] [--without-ref 95aae785] [--dry-run]
#                           [--arm without|with|both] [--report-only <root>] [-- <extra run.py args>]
#
# Arm "without" runs the repository as it stood BEFORE the mechanism (a detached worktree at --without-ref, created
# beside this checkout as <main>.worktrees/ab-without-room-extents if missing); arm "with" runs THIS checkout -- the
# branch feat/objnav-room-extents-nadav, whose name and commit are written into the output as provenance. Both arms:
#   * the 200 episodes of sets/hm3d_v1_val_rep200.json (--episode-ids-file), HM3D v1 val, one process per scene;
#   * ground-truth perception (the search alone is under test; no detector or segmenter is started);
#   * qwen2.5:3b-instruct in BOTH LLM roles, on an Ollama of their own (LLM_BASE_URL port 11436 by default) so the
#     main run's server is never asked for anything and never evicts a model;
#   * deterministic answers: temperature 0, seed 17, LLM_EXCLUSIVE=1 (one request at a time on a freshly loaded
#     model, the only mode measured to make Ollama's answer depend on the request alone; the lock is per server,
#     so it serialises OUR requests only), and the answer store -- arm "without" records every answer it used,
#     arm "with" REPLAYS that store and records its own, so the two arms get the same answer to every question
#     they both ask (the room labels, every oracle prompt until the room lines differ) and part only where the
#     mechanism acts;
#   * --record, so the report can show the oracle's lines and numbers at the action the arms parted.
# The arms run one after the other (the replay needs the first arm's store), 2 scene workers each by default.
#
# The script REFUSES to start when the card does not hold a 3B model and the renderers beside what is running:
# AB_MIN_FREE_MIB (5500 by default: ~3.5 GB for the 3B at 2 slots, 2 x 750 MiB renderers, a margin) must be free,
# and the free RAM at least AB_MIN_FREE_RAM_GIB (8). --dry-run prints the plan and the resource check and exits.
#
# Afterwards: python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.ab_room_extents_report <root>/without <root>/with
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OBJNAV_REPO:-$(cd -- "$HERE/../../../../.." && pwd)}"
# The main checkout (this may be a worktree of it): its runs/ holds every run, and the "without" worktree is made
# beside this one under <main>.worktrees/.
MAIN_REPO="$(dirname "$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)")"
WITHOUT_REF="${AB_WITHOUT_REF:-95aae785}"
WITHOUT_TREE="${AB_WITHOUT_TREE:-${MAIN_REPO}.worktrees/ab-without-room-extents}"
OUTPUT="" JOBS="${AB_JOBS:-2}" DRY_RUN=0 ARM=both REPORT_ONLY="" EXTRA=()
SET_FILE="sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/sets/hm3d_v1_val_rep200.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --without-ref) WITHOUT_REF="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --report-only) REPORT_ONLY="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/objnav-habitat/bin/python}"
export HAB_PY
export OLLAMA_MODE="${OLLAMA_MODE:-native}"
export LLM_BASE_URL="${AB_LLM_BASE_URL:-http://127.0.0.1:11436}"
export LLM_MODEL="${AB_LLM_MODEL:-qwen2.5:3b-instruct}"
export LLM_REASONING_MODEL="$LLM_MODEL"
export LLM_TEMPERATURE=0 LLM_SEED="${LLM_SEED:-17}"
export LLM_EXCLUSIVE="${LLM_EXCLUSIVE:-1}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}" OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"
export LLM_NUM_CTX="${LLM_NUM_CTX:-4096}" LLM_REASONING_NUM_CTX="${LLM_REASONING_NUM_CTX:-4096}"
export GPU_PLAN_LLM="${GPU_PLAN_LLM:-gpu}" LLM_VRAM_MIB="${LLM_VRAM_MIB:-3584}" HABITAT_VRAM_MIB="${HABITAT_VRAM_MIB:-750}"
export GPU_PLAN_YOLO=cpu
AB_MIN_FREE_MIB="${AB_MIN_FREE_MIB:-5500}"
AB_MIN_FREE_RAM_GIB="${AB_MIN_FREE_RAM_GIB:-8}"

if [[ -n "$REPORT_ONLY" ]]; then
  exec "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.ab_room_extents_report "$REPORT_ONLY/without" "$REPORT_ONLY/with" \
    --markdown "$REPORT_ONLY/ab_report.md" --json "$REPORT_ONLY/ab_report.json"
fi

cd "$REPO"
[[ -f "$SET_FILE" ]] || die "the fixed set is missing: $SET_FILE"
[[ -x "$HAB_PY" ]] || die "Habitat interpreter not found at $HAB_PY"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${OUTPUT:-$MAIN_REPO/runs/ab_room_extents_$STAMP}"
[[ "$OUTPUT" = /* ]] || OUTPUT="$PWD/$OUTPUT"

# -- the resource check: never at the main run's expense ----------------------------------------------------------
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
FREE_RAM_GIB="$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)"
IDLE_CPU="$(top -bn1 | awk -F'[, ]+' '/%Cpu/ {for (i=1;i<=NF;i++) if ($i=="id") print $(i-1)}' | head -1)"
say "free: GPU ${FREE_MIB:-?} MiB (need $AB_MIN_FREE_MIB), RAM ${FREE_RAM_GIB} GiB (need $AB_MIN_FREE_RAM_GIB), CPU idle ${IDLE_CPU:-?}%"
if ! curl -fsS --max-time 3 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; then
  say "no Ollama at $LLM_BASE_URL yet: the launcher will start one there for the 3B (the main run's server is untouched)"
fi
ENOUGH=1
[[ -n "$FREE_MIB" && "$FREE_MIB" -ge "$AB_MIN_FREE_MIB" ]] || ENOUGH=0
[[ "$FREE_RAM_GIB" -ge "$AB_MIN_FREE_RAM_GIB" ]] || ENOUGH=0

WITHOUT_LAUNCHER="$WITHOUT_TREE/sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh"
WITH_LAUNCHER="$REPO/sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh"
say "plan: set $SET_FILE (200 episodes), jobs $JOBS, perception ground-truth, model $LLM_MODEL on $LLM_BASE_URL, exclusive=$LLM_EXCLUSIVE"
say "  without: $WITHOUT_TREE @ $WITHOUT_REF  -> $OUTPUT/without  (records answers to $OUTPUT/without_answers)"
say "  with:    $REPO @ $(git -C "$REPO" rev-parse --short HEAD)  -> $OUTPUT/with  (replays without_answers, records $OUTPUT/with_answers)"
if [[ "$DRY_RUN" == 1 ]]; then
  say "dry run: $([[ "$ENOUGH" == 1 ]] && echo 'the resources are there' || echo 'NOT enough free resources; nothing would start')"
  exit 0
fi
[[ "$ENOUGH" == 1 ]] || die "not enough free resources beside the running work (GPU ${FREE_MIB:-?} MiB free, need $AB_MIN_FREE_MIB; RAM ${FREE_RAM_GIB} GiB, need $AB_MIN_FREE_RAM_GIB): nothing started"

# -- the "without" worktree --------------------------------------------------------------------------------------
if [[ "$ARM" == both || "$ARM" == without ]]; then
  if [[ ! -d "$WITHOUT_TREE" ]]; then
    say "creating the worktree of the code before the mechanism: $WITHOUT_TREE @ $WITHOUT_REF"
    git -C "$REPO" worktree add --detach "$WITHOUT_TREE" "$WITHOUT_REF" > /dev/null
  fi
  [[ -x "$WITHOUT_LAUNCHER" ]] || die "no launcher in the without worktree: $WITHOUT_LAUNCHER"
  AT="$(git -C "$WITHOUT_TREE" rev-parse --short HEAD)"
  [[ "$AT" == "$(git -C "$REPO" rev-parse --short "$WITHOUT_REF")" ]] || die "the without worktree is at $AT, not $WITHOUT_REF"
fi
mkdir -p "$OUTPUT"
git -C "$REPO" rev-parse HEAD > "$OUTPUT/with_commit.txt"
git -C "$REPO" branch --show-current > "$OUTPUT/with_branch.txt" || true
echo "$WITHOUT_REF" > "$OUTPUT/without_ref.txt"
cp "$SET_FILE" "$OUTPUT/episode_set.json"

run_arm() {   # name launcher repo [env...]
  local name="$1" launcher="$2" repo="$3"; shift 3
  say "=== arm $name: $launcher"
  ( cd "$repo" && env "$@" OBJNAV_REPO="$repo" \
      "$launcher" --version v1 --perception ground-truth --episode-ids-file "$REPO/$SET_FILE" \
      --jobs "$JOBS" --record --output "$OUTPUT/$name" "${EXTRA[@]}" ) 2>&1 | tee -a "$OUTPUT/$name.console.log"
}

if [[ "$ARM" == both || "$ARM" == without ]]; then
  run_arm without "$WITHOUT_LAUNCHER" "$WITHOUT_TREE" LLM_ANSWERS_RECORD="$OUTPUT/without_answers"
fi
if [[ "$ARM" == both || "$ARM" == with ]]; then
  [[ -d "$OUTPUT/without_answers" ]] || say "WARNING: no without_answers store to replay; arm with runs on fresh answers"
  run_arm with "$WITH_LAUNCHER" "$REPO" LLM_ANSWERS_REPLAY="$OUTPUT/without_answers" LLM_ANSWERS_RECORD="$OUTPUT/with_answers"
fi
if [[ "$ARM" == both ]]; then
  say "=== report"
  "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.ab_room_extents_report "$OUTPUT/without" "$OUTPUT/with" \
    --markdown "$OUTPUT/ab_report.md" --json "$OUTPUT/ab_report.json" | tee "$OUTPUT/ab_report.txt"
fi
