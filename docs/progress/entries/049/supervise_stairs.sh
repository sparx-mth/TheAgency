#!/usr/bin/env bash
# RUN 2 of the MP3D 14B study: the same split, perception, model, seed and settings as run 1 with ONE change,
# allow_stair_traversal TRUE (policy_stairs_on.json), replaying run 1's recorded LLM answers (LLM_ANSWERS_REPLAY) so the
# arms part only where stair traversal changes what the agent asks; its own answers go to its own store. Otherwise
# the supervised, resumable sharded campaign of supervise.sh:

#   * the launcher is invoked with a fixed --output root; whenever it exits before campaign_summary.json says
#     complete (a stalled shard killed by the watchdog, a crashed worker, a lost service), it is invoked again on the
#     same root, which resumes finished-but-unmerged scenes, continues started shards and starts the rest;
#   * every LLM answer is recorded, keyed by episode, into one store OUTSIDE the root (LLM_ANSWERS_RECORD), so any
#     episode can be replayed exactly later;
#   * the resident Ollama is confirmed (or started) before each attempt; the resource monitor samples the machine.
# Usage: supervise.sh [jobs]         (the jobs file under the support directory overrides the cap live; 0 drains)
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
REPO="$HOME/GIT/TheAgency.worktrees/copilot-worktree-2026-11-11T10-35-31-03da"
NAME_FILE="$HERE/run_name_stairs"
if [[ -s "$NAME_FILE" ]]; then NAME="$(cat "$NAME_FILE")"; else NAME="gt_oracle_mp3d_2195ep_14b_stairs_$(date -u +%Y%m%dT%H%M%SZ)"; echo "$NAME" > "$NAME_FILE"; fi
ROOT="$RUNS_ROOT/$NAME"
SUPPORT="$RUNS_ROOT/${NAME}_support"
mkdir -p "$SUPPORT"
[[ -s "$SUPPORT/jobs" ]] || echo "${1:-8}" > "$SUPPORT/jobs"
export LLM_ANSWERS_RECORD="$SUPPORT/llm_answers"
RUN1="$RUNS_ROOT/$(cat "$HERE/run_name")_support/llm_answers"
[[ -d "$RUN1" ]] || { echo "run 1 answer store $RUN1 missing"; exit 1; }
export LLM_ANSWERS_REPLAY="$RUN1"
LOG="$SUPPORT/supervisor.log"
say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

complete() {
  [[ -f "$ROOT/campaign_summary.json" ]] && "$HAB_PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["complete"] else 1)' "$ROOT/campaign_summary.json"
}

# The monitor samples for the whole run (one per supervisor process).
if [[ ! -s "$SUPPORT/monitor.pid" ]] || ! kill -0 "$(cat "$SUPPORT/monitor.pid")" 2> /dev/null; then
  (cd "$REPO" && setsid nohup "$HOME/GIT/TheAgency/.venv/bin/python" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.resource_monitor \
     record "$SUPPORT/resources.csv" --interval 15 > "$SUPPORT/monitor.log" 2>&1 < /dev/null & echo $! > "$SUPPORT/monitor.pid")
fi

say "run $NAME: root $ROOT, support $SUPPORT, jobs $(cat "$SUPPORT/jobs"), answer store $LLM_ANSWERS_RECORD, replaying $LLM_ANSWERS_REPLAY, policy $(cat "$HERE/policy_stairs_on.json")"
for attempt in $(seq 1 12); do
  if complete; then say "complete after $((attempt - 1)) attempt(s)"; break; fi
  "$HERE/start_ollama.sh" 2>&1 | tee -a "$LOG"
  ACCEPT=()
  if [[ -f "$ROOT/gpu_plan.json" ]]; then
    # A resume keeps the GPU plan the run was frozen with; a re-planned one differs in what the card held at the time.
    export GPU_PLAN_REPLAY="$ROOT/gpu_plan.json"
    ACCEPT=(--resume-accepting-changes gpu_plan)
  fi
  say "attempt $attempt: launching (resume: $([[ -d "$ROOT" ]] && echo yes || echo no))"
  (cd "$REPO" && sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/run_benchmark.sh \
      --perception ground-truth --record-log-only --full-records \
      --shard-episodes 10 --jobs "$(cat "$SUPPORT/jobs")" --jobs-file "$SUPPORT/jobs" --stall-minutes 30 \
      --policy-config "$HERE/policy_stairs_on.json" --output "$ROOT" "${ACCEPT[@]}") >> "$SUPPORT/launcher_attempt_$attempt.log" 2>&1
  code=$?
  say "attempt $attempt: launcher exit $code (log: $SUPPORT/launcher_attempt_$attempt.log)"
  if complete; then say "complete after $attempt attempt(s)"; break; fi
  if [[ "$(cat "$SUPPORT/jobs")" == 0 ]]; then say "jobs file is 0: drained on purpose, not resuming"; break; fi
  say "not complete; resuming in 60 s"
  sleep 60
done
say "supervisor done"
