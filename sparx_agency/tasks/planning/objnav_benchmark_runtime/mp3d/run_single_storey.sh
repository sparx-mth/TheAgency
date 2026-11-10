#!/usr/bin/env bash
# Run an MP3D ground-truth campaign on the fixed single-storey set from a FROZEN copy of the source tree.
#
#   mp3d/run_single_storey.sh <name> [--policy-config cfg.json] [--jobs N] [--resume] [-- extra launcher args]
#
# Why a frozen copy: every scene child imports the repository at its own start, so a campaign that runs for
# hours while the working tree is being edited would mix two versions of the method across its scenes. The
# tracked files and the untracked-but-not-ignored ones are rsynced into $SNAPSHOT_ROOT/<name>/ once (an
# existing snapshot is kept: a --resume runs what the first launch ran), and the launcher runs from there with
# its output under the MAIN repository's runs/ directory.
#
# Every run of the study shares ONE LLM answer store ($STORE, LLM_ANSWERS_RECORD): a question already answered
# in an episode gets the same answer in every later run (scoped by episode, llm_answer_cache.py), so two runs
# part only where a change under test makes them ask something new. The store grows with each run.
#
# Environment (defaults are this laptop's): HAB_PY, LLM_BASE_URL (the resident Ollama), SET (the episode list),
# STORE (this run's record directory), REPLAY (earlier stores to answer from, os.pathsep-separated), SNAPSHOT_ROOT,
# HABITAT_VRAM_MIB (one renderer's VRAM for the GPU plan), STALL_MINUTES.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../../../.." && pwd)"
NAME="${1:?usage: run_single_storey.sh <name> [--policy-config cfg] [--jobs N] [--resume] [-- launcher args]}"; shift
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
SNAPSHOT="$SNAPSHOT_ROOT/$NAME"
RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"
OUTPUT="$RUNS_ROOT/ss_$NAME"
SET="${SET:-$REPO/sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/sets/single_storey_flat_v1.json}"
STORE="${STORE:-$RUNS_ROOT/single_storey_v1_llm_answers}"
STALL_MINUTES="${STALL_MINUTES:-30}"

if [[ -d "$SNAPSHOT" ]]; then
  echo "snapshot $SNAPSHOT exists: kept (a resume runs the code the campaign started with)"
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
export HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/habitat/bin/python}"
export OLLAMA_MODE="${OLLAMA_MODE:-native}" LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:3b-instruct}" LLM_REASONING_MODEL="${LLM_REASONING_MODEL:-qwen2.5:3b-instruct}"
export LLM_ANSWERS_RECORD="$STORE"
# REPLAY (os.pathsep-separated directories) serves earlier runs' answers when this run records elsewhere: two runs that
# may hold the same episode at the same time must not record into one directory (LESSONS.md 2026-11-10).
[[ -n "${REPLAY:-}" ]] && export LLM_ANSWERS_REPLAY="$REPLAY"
export GPU_PLAN_LLM="${GPU_PLAN_LLM:-gpu}" LLM_VRAM_MIB="${LLM_VRAM_MIB:-0}" HABITAT_VRAM_MIB="${HABITAT_VRAM_MIB:-1100}"
export ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-1}"
echo "campaign $NAME: set $SET, store $STORE, output $OUTPUT, jobs $JOBS, snapshot $SNAPSHOT"
exec "$SNAPSHOT/sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/run_benchmark.sh" \
  --perception ground-truth --episode-ids-file "$SET" --record --jobs "$JOBS" --output "$OUTPUT" \
  --stall-minutes "$STALL_MINUTES" "${EXTRA[@]}"
