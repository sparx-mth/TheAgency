#!/usr/bin/env bash
# One screen of the full run's health: campaign status, pool events, machine, services, intruders on the card.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
ARM="${1:-}"; N="$(cat "$HERE/run_name${ARM:+_$ARM}")"; R="$RUNS_ROOT/$N"; S="${R}_support"
REPO="$HOME/GIT/TheAgency.worktrees/copilot-worktree-2026-11-11T10-35-31-03da"
cd "$REPO"
sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/run_benchmark.sh --status "$R" 2>&1 | head -${LINES_:-3}
A="$(ls "$S"/launcher_attempt_*.log 2>/dev/null | tail -1)"; [[ -n "$A" ]] || { echo "no launcher attempt yet for $N"; exit 0; }
echo "attempt $(basename "$A" .log | sed 's/launcher_attempt_//'): shards started $(grep -c '\[pool\] start' "$A"), ended $(grep -c '\[pool\] end' "$A"), stalled $(grep -c STALLED "$A"), non-zero exits $(grep '\[pool\] end' "$A" | grep -vc 'exit 0'), merged $(grep -c '\[campaign\] merged' "$A")"
grep '\[pool\] end' "$A" | grep -v 'exit 0' | tail -3
echo "card: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)  | other GPU users: $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | grep -v 'ollama\|objnav-habitat' | tr '\n' ';')"
echo "sample: $(tail -1 "$S/resources.csv")"
echo "supervisor: $(tail -1 "$S/supervisor.log")"
echo "ollama: $(curl -fsS --max-time 5 "$LLM_BASE_URL/api/ps" | "$HAB_PY" -c 'import json,sys; d=json.load(sys.stdin); print(", ".join("%s %.1f GB" % (m["name"], m["size_vram"]/1e9) for m in d["models"]) or "no model loaded")' 2>/dev/null || echo DOWN)  keeper restarts: $(grep -c restarting "$HERE/ollama_keeper.log")"
echo "docker: $(docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | tr '\n' ';')  sphera starts in last 2 h: $(docker events --since 2h --until 1s --filter container=sphera --filter event=start --format '{{.Time}}' 2>/dev/null | wc -l)"
echo "disk: $(df -h "$RUNS_ROOT" | awk 'NR==2 {print $4" free"}')  root size: $(du -sh "$R" 2>/dev/null | cut -f1)  store: $(du -sh "$S/llm_answers" 2>/dev/null | cut -f1)"
"$HAB_PY" - "$R" <<'PY'
import json, sys
beats=[json.loads(l) for l in open(sys.argv[1]+"/campaign_heartbeat.jsonl") if l.strip()]
if len(beats) > 12:
    a, b = beats[-13], beats[-1]; print("last hour: %.0f ep/h, %d done, running %d" % ((b["done"]-a["done"]) / ((b["unix"]-a["unix"])/3600), b["done"], b["running"]))
PY
