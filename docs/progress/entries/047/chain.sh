#!/usr/bin/env bash
# Chain the two arms: wait for run 1's campaign_summary.json to say complete, repair its answer store of any key two
# processes answered differently (a watchdog-resumed shard re-asks; the replay loader refuses such a store), then start
# run 2 (supervise_stairs.sh) at once so the card is never idle between the arms. The report and the commit of run 1
# are done by hand while run 2 runs. Polls every 5 minutes.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
REPO="$HOME/GIT/TheAgency.worktrees/copilot-worktree-2026-11-11T10-35-31-03da"
N1="$(cat "$HERE/run_name")"; R1="$RUNS_ROOT/$N1"; S1="${R1}_support"
LOG="$HERE/chain.log"
say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
say "chain: waiting for $N1 to complete"
while true; do
  if [[ -f "$R1/campaign_summary.json" ]] && "$HAB_PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["complete"] else 1)' "$R1/campaign_summary.json"; then
    break
  fi
  # the run-1 supervisor must still be alive, or the chain waits for nothing
  if [[ -s "$HERE/supervisor.pid" ]] && ! kill -0 "$(cat "$HERE/supervisor.pid")" 2> /dev/null; then
    say "run 1's supervisor is gone without a complete summary; not chaining"; exit 1
  fi
  sleep 300
done
say "run 1 complete: $("$HAB_PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("%d episodes SR %.3f SPL %.3f" % (r["episodes"], r["overall"]["success_rate"], r["overall"]["spl"]))' "$R1/campaign_summary.json" 2>/dev/null || echo "(summary unreadable; see file)")"
say "checking run 1's answer store for conflicting keys"
(cd "$REPO" && "$HAB_PY" -m sparx_agency.core.mapping.topology.llm_answer_cache --resolve-conflicts "$S1/llm_answers" --keep earliest) 2>&1 | tee -a "$LOG"
say "starting run 2 (stairs on)"
(cd "$HERE" && setsid nohup ./supervise_stairs.sh "$(cat "$S1/jobs")" > supervisor_stairs_console.log 2>&1 < /dev/null & echo $! > supervisor_stairs.pid)
say "run 2 supervisor pid $(cat "$HERE/supervisor_stairs.pid")"
