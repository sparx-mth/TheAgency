#!/usr/bin/env bash
# Run one arm of the warm-up study (2026-11-11) on the study's episode subset, from a FROZEN copy of the source.
#
#   mp3d/run_warmup_arm.sh <arm> [--policy-config cfg.json] [--jobs N] [--resume] [-- extra launcher args]
#
# The study compares the vantage warm-up (RPTSettings.warmup.mode == "vantage") against the ten-turn spin under
# ground-truth perception on sets/warmup_study_v1.json: single-storey episodes the running MP3D ground-truth
# campaign had already finished, whose rows ARE the spin arm. Every arm here:
#
# * runs from a snapshot of the tree (rsync of the tracked and untracked-but-not-ignored files, as
#   run_single_storey.sh does), so the tree can be edited while the arm runs;
# * talks to the campaign's own resident Ollama (the 14B in both roles at LLM_BASE_URL, default :11435), REPLAYS
#   the campaign's LLM answer store (REPLAY) so a question the campaign already asked in an episode gets the same
#   answer, and records its own new answers in a store of its own (STORE: two runs never write one directory);
# * renders on the CPU (HABITAT_RENDER=software: Mesa llvmpipe) so it takes no VRAM from the campaign's renderers
#   and leaves the campaign's pool gate alone -- about a second an action; GPU rendering is HABITAT_RENDER=gpu;
# * writes --record-log-only --full-records (steps.jsonl, trajectory.csv, the policy's whole report per episode),
#   which warmup_study.py reads for the first sight of the target and the area the warm-up uncovered.
#
# Environment: HAB_PY, LLM_BASE_URL, LLM_MODEL / LLM_REASONING_MODEL, SET (the subset), REPLAY (os.pathsep-separated
# stores to answer from), STORE (this arm's record directory), SNAPSHOT_ROOT, RUNS_ROOT, STALL_MINUTES,
# MP3D_SCENES_DIR / MP3D_EPISODES_DIR.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../../../.." && pwd)"
NAME="${1:?usage: run_warmup_arm.sh <arm> [--policy-config cfg] [--jobs N] [--resume] [-- launcher args]}"; shift
JOBS=3 POLICY_CONFIG="" RESUME=0 EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --policy-config) POLICY_CONFIG="$(realpath "$2")"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$HOME/GIT/TheAgency_snapshots}"
SNAPSHOT="$SNAPSHOT_ROOT/wu_$NAME"
RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"
OUTPUT="$RUNS_ROOT/wu_$NAME"
SET="${SET:-$REPO/sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/sets/warmup_study_v1.json}"
STORE="${STORE:-$RUNS_ROOT/wu_${NAME}_llm_answers}"
STALL_MINUTES="${STALL_MINUTES:-30}"
[[ -f "$SET" ]] || { echo "no episode subset at $SET (warmup_study.py select)" >&2; exit 2; }

if [[ -d "$SNAPSHOT" ]]; then
  echo "snapshot $SNAPSHOT exists: kept (a resume runs the code the arm started with)"
else
  mkdir -p "$SNAPSHOT"
  ( cd "$REPO" && { git ls-files; git ls-files --others --exclude-standard; } \
      | rsync -a --files-from=- "$REPO/" "$SNAPSHOT/" )
  ( cd "$REPO" && git --no-pager rev-parse HEAD > "$SNAPSHOT/SNAPSHOT_COMMIT.txt" \
      && git --no-pager diff --no-ext-diff > "$SNAPSHOT/SNAPSHOT_CHANGES.diff" \
      && git ls-files --others --exclude-standard > "$SNAPSHOT/SNAPSHOT_UNTRACKED.txt" )
  echo "snapshot of $REPO at $(cat "$SNAPSHOT/SNAPSHOT_COMMIT.txt") -> $SNAPSHOT"
fi
mkdir -p "$STORE"
[[ -n "$POLICY_CONFIG" ]] && EXTRA+=(--policy-config "$POLICY_CONFIG")
[[ "$RESUME" == 1 ]] && EXTRA+=(--resume-accepting-changes gpu_plan source_sha256)

cd "$SNAPSHOT"
export OBJNAV_REPO="$SNAPSHOT" RUNS_ROOT
export HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/objnav-habitat/bin/python}"
export MP3D_SCENES_DIR="${MP3D_SCENES_DIR:-$HOME/datasets/mp3d/scene_datasets}"
export MP3D_EPISODES_DIR="${MP3D_EPISODES_DIR:-$HOME/datasets/objectnav/mp3d/v1/val}"
export OLLAMA_MODE="${OLLAMA_MODE:-native}" LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11435}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:14b-instruct}" LLM_REASONING_MODEL="${LLM_REASONING_MODEL:-qwen2.5:14b-instruct}"
export LLM_ANSWERS_RECORD="$STORE"
[[ -n "${REPLAY:-}" ]] && export LLM_ANSWERS_REPLAY="$REPLAY"
export GPU_PLAN_LLM="${GPU_PLAN_LLM:-gpu}" LLM_VRAM_MIB="${LLM_VRAM_MIB:-0}" ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-1}"
export HABITAT_RENDER="${HABITAT_RENDER:-software}"
echo "arm $NAME: set $SET, store $STORE, replay ${REPLAY:-none}, output $OUTPUT, jobs $JOBS, renderer $HABITAT_RENDER, snapshot $SNAPSHOT"
exec "$SNAPSHOT/sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/run_benchmark.sh" \
  --perception ground-truth --episode-ids-file "$SET" --record-log-only --full-records --jobs "$JOBS" \
  --output "$OUTPUT" --stall-minutes "$STALL_MINUTES" "${EXTRA[@]}"
