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
#   * qwen2.5:3b-instruct in BOTH LLM roles, on an Ollama of their own: a native `ollama serve` on port 11436 where
#     the binary exists (the GPU machine), so the main run's server is never asked for anything and never evicts a
#     model; where Ollama is only the Docker container publishing 11434 (the laptop), that container -- nothing else
#     runs beside the A/B there. Both routes at the benchmark's 8192-token window (4096 cut the oracle prompt);
#   * deterministic answers: temperature 0, seed 17, LLM_EXCLUSIVE=1 (one request at a time on a freshly loaded
#     model, the only mode measured to make Ollama's answer depend on the request alone; the lock is per server,
#     so it serialises OUR requests only), and the answer store -- arm "without" records every answer it used,
#     arm "with" REPLAYS that store and records its own, so the two arms get the same answer to every question
#     they both ask (the room labels, every oracle prompt until the room lines differ) and part only where the
#     mechanism acts;
#   * --record, so the report can show the oracle's lines and numbers at the action the arms parted.
# The arms run one after the other (the replay needs the first arm's store), 2 scene workers each by default. An arm
# is launched again (up to AB_MAX_ATTEMPTS times, the frozen GPU plan replayed) until every episode of the set has a
# row and every scene its summary -- a scene the stall watchdog killed, or a crash, resumes where it stopped; and
# --output <root> of an earlier run resumes it the same way.
#
# The script REFUSES to start when the card does not hold a 3B model and the renderers beside what is running:
# AB_MIN_FREE_MIB (5500 by default: ~3.5 GB for the 3B at 2 slots of 8192, 2 x 750 MiB renderers, a margin) must be
# free -- what the A/B's own server already holds counts as free -- and the free RAM at least AB_MIN_FREE_RAM_GIB (8).
# --dry-run prints the plan and the resource check and exits; `-- --preflight` runs both arms' preflights, no episode.
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
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

# The Habitat interpreter: the GPU machine's objnav-habitat env, else the laptop's habitat env (benchmark_services.sh's default).
if [[ -z "${HAB_PY:-}" ]]; then
  for candidate in "$HOME/miniconda3/envs/objnav-habitat/bin/python" "$HOME/miniconda3/envs/habitat/bin/python"; do
    [[ -x "$candidate" ]] && { HAB_PY="$candidate"; break; }
  done
  HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/objnav-habitat/bin/python}"
fi
export HAB_PY
# The LLM server. Where `ollama` is on PATH (the GPU machine) the launcher starts a native server of the A/B's own on
# port 11436, beside whatever a resident server is doing. Where Ollama runs only as a Docker container (the laptop:
# no binary, the container publishing 11434), that container is the server: nothing else runs beside the A/B there,
# which is what LLM_EXCLUSIVE needs anyway. AB_LLM_BASE_URL / OLLAMA_MODE / OLLAMA_CONTAINER override either way.
if [[ -z "${OLLAMA_MODE:-}" && -z "${AB_LLM_BASE_URL:-}" ]] && ! command -v ollama > /dev/null 2>&1 && command -v docker > /dev/null 2>&1; then
  DOCKER_OLLAMA="$(docker ps --filter publish=11434 --format '{{.Names}}' 2>/dev/null | head -1)"
  if [[ -n "$DOCKER_OLLAMA" ]]; then
    OLLAMA_MODE=docker AB_LLM_BASE_URL="http://127.0.0.1:11434"
    export OLLAMA_CONTAINER="${OLLAMA_CONTAINER:-$DOCKER_OLLAMA}"
  fi
fi
export OLLAMA_MODE="${OLLAMA_MODE:-native}"
export LLM_BASE_URL="${AB_LLM_BASE_URL:-http://127.0.0.1:11436}"
export LLM_MODEL="${AB_LLM_MODEL:-qwen2.5:3b-instruct}"
export LLM_REASONING_MODEL="$LLM_MODEL"
export LLM_TEMPERATURE=0 LLM_SEED="${LLM_SEED:-17}"
# The exclusive lock lives at a FIXED path, not under the temp dir: "1" would put it under $TMPDIR, which a CLI
# session sets per session, and the lock's path is part of every scene's frozen configuration -- a resume from another
# session (after the 2026-11-12 host freeze) was refused for it ("method differ") until the old TMPDIR was recreated.
# One lock per server port serialises every A/B process that talks to that server, which is all the lock is for.
export LLM_EXCLUSIVE="${LLM_EXCLUSIVE:-/tmp/sparx-llm-ab-room-extents-${LLM_BASE_URL##*:}.lock}"
# The window is the benchmark's default, 8192, on both routes and in both arms: the oracle's system prompt alone is
# ~3,100 tokens before the mechanism and ~3,900 with it, so 4096 cut the "with" probe (15.8k characters evaluated as
# 2,050 tokens) and would have filled on any large house in either arm. The store key carries num_ctx: the two arms
# must agree on it or nothing replays. The 3B at 2 slots x 8192 is ~2.8 GB; LLM_VRAM_MIB budgets it.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}" OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export LLM_NUM_CTX="${LLM_NUM_CTX:-8192}" LLM_REASONING_NUM_CTX="${LLM_REASONING_NUM_CTX:-8192}"
export GPU_PLAN_LLM="${GPU_PLAN_LLM:-gpu}" HABITAT_VRAM_MIB="${HABITAT_VRAM_MIB:-750}"
AB_LLM_VRAM_MIB="${LLM_VRAM_MIB:-3584}"     # the 3B's budget; what our server already holds is taken off it below
export GPU_PLAN_YOLO=cpu
AB_MIN_FREE_MIB="${AB_MIN_FREE_MIB:-5500}"
AB_MIN_FREE_RAM_GIB="${AB_MIN_FREE_RAM_GIB:-8}"
AB_MAX_ATTEMPTS="${AB_MAX_ATTEMPTS:-4}"     # launches of one arm, the first included, before the script gives up on it

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
# What the A/B's OWN server already holds on the card (a model still resident from an earlier arm or a resume) is
# not another run's: it counts as free for the check, and the GPU plan -- which reads the card's "used" as taken --
# must not budget it a second time (that left the renderers "expecting allocation failure" on an 8 GB card), so
# each arm budgets only what is not yet resident when it starts.
ours_mib() {
  curl -fsS --max-time 3 "$LLM_BASE_URL/api/ps" 2>/dev/null \
    | "$HAB_PY" -c 'import json,sys; print(sum(int(m.get("size_vram") or 0) for m in json.load(sys.stdin).get("models", [])) // 1048576)' 2>/dev/null || echo 0
}
llm_budget_mib() { local left=$(( AB_LLM_VRAM_MIB - $(ours_mib) )); [[ "$left" -gt 0 ]] && echo "$left" || echo 0; }
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
FREE_RAM_GIB="$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)"
IDLE_CPU="$(top -bn1 | awk -F'[, ]+' '/%Cpu/ {for (i=1;i<=NF;i++) if ($i=="id") print $(i-1)}' | head -1)"
OURS_MIB=0
if curl -fsS --max-time 3 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; then OURS_MIB="$(ours_mib)"
else say "no Ollama at $LLM_BASE_URL yet: the launcher will start one there for the 3B (the main run's server is untouched)"; fi
say "free: GPU ${FREE_MIB:-?} MiB + ${OURS_MIB} MiB held by our own server at $LLM_BASE_URL (need $AB_MIN_FREE_MIB), RAM ${FREE_RAM_GIB} GiB (need $AB_MIN_FREE_RAM_GIB), CPU idle ${IDLE_CPU:-?}%"
ENOUGH=1
[[ -n "$FREE_MIB" && $(( FREE_MIB + OURS_MIB )) -ge "$AB_MIN_FREE_MIB" ]] || ENOUGH=0
[[ "$FREE_RAM_GIB" -ge "$AB_MIN_FREE_RAM_GIB" ]] || ENOUGH=0

WITHOUT_LAUNCHER="$WITHOUT_TREE/sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh"
WITH_LAUNCHER="$REPO/sparx_agency/tasks/planning/objnav_benchmark_runtime/hm3d/run_benchmark.sh"
say "plan: set $SET_FILE (200 episodes), jobs $JOBS, perception ground-truth, model $LLM_MODEL on $LLM_BASE_URL ($OLLAMA_MODE${OLLAMA_CONTAINER:+ $OLLAMA_CONTAINER}), exclusive=$LLM_EXCLUSIVE, interpreter $HAB_PY"
say "  without: $WITHOUT_TREE @ $WITHOUT_REF  -> $OUTPUT/without  (records answers to $OUTPUT/without_answers)"
say "  with:    $REPO @ $(git -C "$REPO" rev-parse --short HEAD)  -> $OUTPUT/with  (replays without_answers, records $OUTPUT/with_answers)"
if [[ "$DRY_RUN" == 1 ]]; then
  say "dry run: $([[ "$ENOUGH" == 1 ]] && echo 'the resources are there' || echo 'NOT enough free resources; nothing would start')"
  exit 0
fi
[[ "$ENOUGH" == 1 ]] || die "not enough free resources beside the running work (GPU ${FREE_MIB:-?} MiB free + ${OURS_MIB} MiB ours, need $AB_MIN_FREE_MIB; RAM ${FREE_RAM_GIB} GiB, need $AB_MIN_FREE_RAM_GIB): nothing started"

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
printf 'HAB_PY=%s\nOLLAMA_MODE=%s\nOLLAMA_CONTAINER=%s\nLLM_BASE_URL=%s\nLLM_MODEL=%s\nLLM_EXCLUSIVE=%s\nJOBS=%s\nhost=%s\n' \
  "$HAB_PY" "$OLLAMA_MODE" "${OLLAMA_CONTAINER:-}" "$LLM_BASE_URL" "$LLM_MODEL" "$LLM_EXCLUSIVE" "$JOBS" "$(hostname)" > "$OUTPUT/ab_env.txt"

run_arm() {   # name launcher repo [env...]
  local name="$1" launcher="$2" repo="$3"; shift 3
  local budget; budget="$(llm_budget_mib)"
  say "=== arm $name: $launcher  (LLM budget $budget MiB beyond what our server holds)"
  ( cd "$repo" && env "$@" OBJNAV_REPO="$repo" LLM_VRAM_MIB="$budget" \
      "$launcher" --version v1 --perception ground-truth --episode-ids-file "$REPO/$SET_FILE" \
      --jobs "$JOBS" --record --output "$OUTPUT/$name" "${EXTRA[@]}" ) 2>&1 | tee -a "$OUTPUT/$name.console.log"
}

arm_complete() {   # root: 0 when every episode of the set has a row and every scene wrote its summary
  "$HAB_PY" - "$1" "$REPO/$SET_FILE" <<'PY'
import json, sys
from pathlib import Path
root, wanted = Path(sys.argv[1]), json.load(open(sys.argv[2]))
summary = root / "campaign_summary.json"
if not summary.is_file():
    sys.exit(1)
report = json.load(summary.open())
scenes = {episode.split("/")[0] for episode in wanted}
done = (report["episodes"] == len(wanted) and report["scenes_with_rows"] == len(scenes)
        and not report["scenes_unfinished"] and not report["scenes_without_rows"])
print("%d of %d episodes, %d of %d scenes, unfinished %s, without rows %s" % (
    report["episodes"], len(wanted), report["scenes_with_rows"], len(scenes),
    report["scenes_unfinished"] or "none", report["scenes_without_rows"] or "none"))
sys.exit(0 if done else 1)
PY
}

run_arm_to_completion() {   # name launcher repo [env...]: the arm until every row is in, resuming what a stall or a crash left
  local name="$1" attempt=1 replay=()
  if [[ " ${EXTRA[*]} " == *" --preflight "* ]]; then run_arm "$@"; return; fi     # a preflight writes no rows
  while :; do
    # A resume keeps the plan the run was frozen with; planning again would be refused for its changed MiB.
    replay=(); [[ -f "$OUTPUT/$name/gpu_plan.json" ]] && replay=(GPU_PLAN_REPLAY="$OUTPUT/$name/gpu_plan.json")
    run_arm "$@" "${replay[@]}" || true          # the rows decide, not the launcher's exit code
    if arm_complete "$OUTPUT/$name"; then say "arm $name: complete"; return 0; fi
    [[ "$attempt" -lt "$AB_MAX_ATTEMPTS" ]] || die "arm $name is still incomplete after $attempt attempts; resume with --arm $name --output $OUTPUT"
    attempt=$(( attempt + 1 ))
    say "arm $name: incomplete; resuming it (attempt $attempt of $AB_MAX_ATTEMPTS)"
    sleep 15
  done
}

if [[ "$ARM" == both || "$ARM" == without ]]; then
  run_arm_to_completion without "$WITHOUT_LAUNCHER" "$WITHOUT_TREE" LLM_ANSWERS_RECORD="$OUTPUT/without_answers"
fi
if [[ "$ARM" == both || "$ARM" == with ]]; then
  [[ -d "$OUTPUT/without_answers" ]] || say "WARNING: no without_answers store to replay; arm with runs on fresh answers"
  run_arm_to_completion with "$WITH_LAUNCHER" "$REPO" LLM_ANSWERS_REPLAY="$OUTPUT/without_answers" LLM_ANSWERS_RECORD="$OUTPUT/with_answers"
fi
# The report, whenever both arms are in -- a resume of one arm (--arm with --output <root>) ends the A/B as much as a
# run of both does. A preflight pass-through writes no rows and gets no report.
if [[ " ${EXTRA[*]} " != *" --preflight "* ]] && arm_complete "$OUTPUT/without" > /dev/null && arm_complete "$OUTPUT/with" > /dev/null; then
  say "=== report"
  "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.ab_room_extents_report "$OUTPUT/without" "$OUTPUT/with" \
    --markdown "$OUTPUT/ab_report.md" --json "$OUTPUT/ab_report.json" | tee "$OUTPUT/ab_report.txt"
elif [[ " ${EXTRA[*]} " != *" --preflight "* ]]; then
  say "no report: an arm is incomplete (resume it with --arm without|with --output $OUTPUT)"
fi
