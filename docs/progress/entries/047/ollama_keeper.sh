#!/usr/bin/env bash
# Keep the resident Ollama answering for the whole run: every 60 s, if 127.0.0.1:11435 stops answering, start it again
# (start_ollama.sh is idempotent). Logs every restart. Stop it with its pid file when the run is over.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
LOG="$HERE/ollama_keeper.log"
echo "[$(date -u +%FT%TZ)] keeper started" >> "$LOG"
while true; do
  if ! curl -fsS --max-time 10 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] ollama not answering; restarting" >> "$LOG"
    "$HERE/start_ollama.sh" >> "$LOG" 2>&1
  fi
  sleep 60
done
