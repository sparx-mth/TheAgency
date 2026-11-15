#!/usr/bin/env bash
# Start (or confirm) the resident native Ollama the MP3D 14B run uses: 4 request slots, 8k default context, Ollama's default f16 KV,
# on GPU 0, serving ~/models/objnav/ollama on 127.0.0.1:11435. Idempotent: a server already answering is left alone.
# Usage: start_ollama.sh [NUM_PARALLEL]   (default 4; a different slot count needs the old server stopped first)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
SLOTS="${1:-$OLLAMA_NUM_PARALLEL}"
if curl -fsS --max-time 5 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; then
  echo "ollama already answers at $LLM_BASE_URL (left running)"; exit 0
fi
LOG="$HERE/ollama_11435_$(date -u +%Y%m%dT%H%M%SZ).log"
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST="${LLM_BASE_URL#http://}" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" \
  OLLAMA_NUM_PARALLEL="$SLOTS" OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" \
  OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=30m setsid nohup ollama serve > "$LOG" 2>&1 &
echo "$!" > "$HERE/ollama_11435.pid"
for _ in $(seq 1 60); do curl -fsS --max-time 5 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1 && break; sleep 2; done
curl -fsS --max-time 300 "$LLM_BASE_URL/api/chat" -d "{\"model\": \"$LLM_MODEL\", \"stream\": false, \"keep_alive\": \"30m\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply OK.\"}], \"options\": {\"num_ctx\": 8192, \"num_predict\": 2}}" > /dev/null
echo "ollama up at $LLM_BASE_URL with $SLOTS slot(s) (pid $(cat "$HERE/ollama_11435.pid"), log $LOG); $LLM_MODEL warm at 8192"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
