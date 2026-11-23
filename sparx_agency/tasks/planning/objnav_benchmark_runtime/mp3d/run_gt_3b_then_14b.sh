#!/usr/bin/env bash
# One command, two full MP3D v1 val runs (2,195 episodes each) in sequence under ground-truth perception, the same
# node-oracle prompt version for both, so that only the model differs:
#
#   stage 1  qwen2.5:3b-instruct  in both LLM roles (room labels and node oracle)
#   stage 2  qwen2.5:14b-instruct in both LLM roles
#
# Each stage is: checks -> a 3-episode smoke (one short episode in each of 3 scenes) -> the full sharded campaign ->
# a results summary committed and pushed to BRANCH (mp3d/results/<campaign>_stage<N>_<model>/). The run directories,
# step logs, answer stores and utilisation logs stay on this machine under runs/<campaign>/.
#
#   mp3d/run_gt_3b_then_14b.sh --detach      # start -- or RESUME -- both stages in their own session (survives the shell)
#   mp3d/run_gt_3b_then_14b.sh --status      # one block per stage: smoke, rows, live shard progress, governor, last log lines
#   mp3d/run_gt_3b_then_14b.sh --stop        # stop for good: STOP file, TERM to the orchestrator (the launcher stops its shards)
#   mp3d/run_gt_3b_then_14b.sh               # the same as --detach, in the foreground
#   mp3d/run_gt_3b_then_14b.sh --stage 2     # only the given stage(s), e.g. --stage "2"
#
# RESUMING is the same command again (--detach): a finished stage is skipped (its summary committed and pushed if that
# had not happened), a passed smoke is not repeated, an interrupted campaign continues from the episodes it had written
# (run_benchmark.sh --output resume: finished shards are kept, started ones continue, the GPU plan is replayed).
#
# What it checks before each stage (and aborts on): the checkout is on BRANCH and contains REQUIRED_COMMIT; the
# registered prompt version PROMPT_VERSION has a digest starting PROMPT_DIGEST_PREFIX; both models are in the native
# Ollama store (pulled if not); nvidia-smi answers; no stale launcher of the stage is alive. After each launch: every
# Ollama server of the stage answers, workers are alive, and the first run.json's node_oracle.prompt carries the digest
# prefix and the stage's model. The smoke must give 3 rows without an agent error, the prompt digest and the model in
# run.json, and recorded LLM answers from that model. A model / prompt-version mismatch (the 14B asked under the 3B's
# prompt) is NOT an error: the harness records node_oracle.model.prompt_tuned_on_it=false; this script logs it.
#
# A failed commit or push never stops the run: the commit stays local, the failure is logged and the next stage starts
# (a later invocation retries the push).
#
# Parallelism (RTX 4090 24 GB, 24 threads, 64 GB; target <= 90 % of GPU, VRAM, CPU and RAM), from the MP3D 14B run of
# 2026-11-11 (runs/gt_oracle_mp3d_2195ep_14b_20261111T095333Z: ~820 MiB per GT renderer, 8 workers = VRAM 86 % median,
# CPU ~20 %, RAM ~20 GB; one Ollama server caps at ~0.5 requests/s whatever its slots, its JSON sampling being serial
# on the CPU) and the HM3D 3B smoke of 2026-11-17 (the 3B runner 5.5 GB at 10 slots):
#   3B : 2 Ollama servers x 5 slots (~3.9 GB each), at most 16 workers   (plan: 16 x 850 + 7,800 MiB)
#   14B: 1 Ollama server  x 4 slots (15.1 GB),      at most 7 workers    (plan:  7 x 850 + 15,200 MiB)
# The shard pool starts a shard only while VRAM stays under VRAM_CAP (0.88) with a renderer to spare, and
# mp3d/jobs_governor.py lowers the live jobs file when a 2-minute mean of CPU, RAM, VRAM or GPU passes 90 % (raises it
# back towards the ceiling after 30 calm minutes). --jobs max is not used: it sizes itself before the LLM is loaded.
#
# Logs, all under runs/<campaign>/ (CAMPAIGN):
#   support/orchestrator.log                 this script, every check and decision
#   support/stage<N>_launcher.out            the launcher's console, every launch
#   support/stage<N>_utilization.csv         GPU/VRAM/CPU/RAM/disk and per-process VRAM every 15 s (../utilization_log.py)
#   support/stage<N>_governor.csv            the governor's samples and decisions (30 s)
#   support/stage<N>_{started,finished}_utc.txt, stage<N>_smoke.passed, stage<N>.committed, stage<N>.pushed
#   stage<N>_<model>/                        the campaign root: run.log, preflight.json, gpu_plan.json, source-commit.txt,
#                                            source-changes.diff, ollama*.log, shard_plan.json, campaign_summary.json, and
#                                            per scene run.json (model, prompt version and digest, every setting),
#                                            episodes.jsonl (full records), shard_*/recordings/*/steps.jsonl +
#                                            trajectory.csv + map journal (every pose, action and decision), console logs
#   stage<N>_<model>_llm_answers/            every LLM request (system + user prompt, model, episode) and its reply
#   stage<N>_<model>_smoke/, ..._smoke_llm_answers/   the same for the smoke
set -uo pipefail

# -- the plan ---------------------------------------------------------------------------------------------------------
BRANCH="${BRANCH:-feat/objnav-habitat-mp3d-nadav}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-a13214e2}"
PROMPT_VERSION="${PROMPT_VERSION:-facts-qwen2.5-3b-v1}"
PROMPT_DIGEST_PREFIX="${PROMPT_DIGEST_PREFIX:-2d025f01}"
MODEL_1="${MODEL_1:-qwen2.5:3b-instruct}"   JOBS_1="${JOBS_1:-16}"  SLOTS_1="${SLOTS_1:-5}"  SERVERS_1="${SERVERS_1:-2}"  LLM_VRAM_1="${LLM_VRAM_1:-7800}"
MODEL_2="${MODEL_2:-qwen2.5:14b-instruct}"  JOBS_2="${JOBS_2:-7}"   SLOTS_2="${SLOTS_2:-4}"  SERVERS_2="${SERVERS_2:-1}"  LLM_VRAM_2="${LLM_VRAM_2:-15200}"
SHARD_EPISODES="${SHARD_EPISODES:-10}"
HABITAT_VRAM_MIB="${HABITAT_VRAM_MIB:-850}"
VRAM_CAP="${VRAM_CAP:-0.88}"
LIMIT_PCT="${LIMIT_PCT:-90}"
STALL_MINUTES="${STALL_MINUTES:-30}"
MAX_RELAUNCHES="${MAX_RELAUNCHES:-6}"           # launches per stage; each relaunch resumes and must have made progress
SMOKE_SCENES="${SMOKE_SCENES:-pLe4wQe7qrG zsNo4HB9uLZ oLBMNvg9in8}"   # their first published episodes are short
CAMPAIGN="${CAMPAIGN:-mp3d_v1_gt_3b_vs_14b}"
LLM_PORT="${LLM_PORT:-11440}"                   # this campaign's own servers: 11434 (system) and 11435 are left alone

# -- where things are (this PC) -----------------------------------------------------------------------------------------
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
REPO="$(cd -- "$HERE/../../../../.." && pwd)"
export HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/objnav-habitat/bin/python}"
export MP3D_SCENES_DIR="${MP3D_SCENES_DIR:-$HOME/datasets/mp3d/scene_datasets}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$HOME/models/objnav/ollama}"
LLM_BASE_URL="http://127.0.0.1:$LLM_PORT"
LAUNCHER="$HERE/run_benchmark.sh"
RESULTS_ROOT="$HERE/results"
PY="${PY:-$REPO/venv/bin/python}"
ROOT="$REPO/runs/$CAMPAIGN"
SUPPORT="$ROOT/support"
LOG="$SUPPORT/orchestrator.log"
PIDFILE="$SUPPORT/orchestrator.pid"

# -- the command line -----------------------------------------------------------------------------------------------------
MODE=run STAGES="1 2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) MODE=detach; shift ;;
    --status) MODE=status; shift ;;
    --stop) MODE=stop; shift ;;
    --stage) STAGES="$2"; shift 2 ;;
    -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$SUPPORT"

say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
die() { say "ABORT: $*"; exit 1; }
alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -qF "run_gt_3b_then_14b.sh"; }

stage_model() { local v="MODEL_$1"; echo "${!v}"; }
stage_jobs() { local v="JOBS_$1"; echo "${!v}"; }
stage_slots() { local v="SLOTS_$1"; echo "${!v}"; }
stage_servers() { local v="SERVERS_$1"; echo "${!v}"; }
stage_llm_vram() { local v="LLM_VRAM_$1"; echo "${!v}"; }
stage_tag() { echo "stage$1_$(stage_model "$1" | tr ':.' '__')"; }
stage_root() { echo "$ROOT/$(stage_tag "$1")"; }
rows_in() { cat "$1"/*/episodes.jsonl "$1"/*/shard_*/episodes.jsonl 2>/dev/null | wc -l; }
stage_complete() {
  [[ -f "$1/campaign_summary.json" ]] && "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["complete"] else 1)' "$1/campaign_summary.json" 2>/dev/null
}

# -- --status ---------------------------------------------------------------------------------------------------------------
if [[ "$MODE" == status ]]; then
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if alive "$pid"; then echo "orchestrator: RUNNING (pid $pid)"; else echo "orchestrator: not running$([[ -e "$SUPPORT/STOP" ]] && echo " (STOP file present)")"; fi
  for s in 1 2; do
    r="$(stage_root "$s")"
    printf '\n== stage %s: %s (jobs <= %s, %s server(s) x %s slots)\n' "$s" "$(stage_model "$s")" "$(stage_jobs "$s")" "$(stage_servers "$s")" "$(stage_slots "$s")"
    [[ -f "$SUPPORT/stage${s}_smoke.passed" ]] && echo "smoke: passed -- $(cat "$SUPPORT/stage${s}_smoke.passed")"
    if [[ ! -d "$r" ]]; then echo "full run: not started"; continue; fi
    echo "started $(cat "$SUPPORT/stage${s}_started_utc.txt" 2>/dev/null || echo ?)$([[ -f "$SUPPORT/stage${s}_finished_utc.txt" ]] && echo ", finished $(cat "$SUPPORT/stage${s}_finished_utc.txt")")"
    echo "rows written: $(rows_in "$r") of 2195; workers alive: $(pgrep -fc "mp3d\.run .*$r/" || true); jobs file: $(cat "$SUPPORT/stage${s}_jobs" 2>/dev/null || echo -)"
    [[ -f "$r/shard_plan.json" ]] && "$LAUNCHER" --status "$r" 2>/dev/null | tail -15
    [[ -f "$SUPPORT/stage${s}_governor.csv" ]] && echo "governor (last): $(tail -1 "$SUPPORT/stage${s}_governor.csv")"
    [[ -f "$SUPPORT/stage${s}.committed" ]] && echo "results committed: $(cat "$SUPPORT/stage${s}.committed")$([[ -f "$SUPPORT/stage${s}.pushed" ]] && echo ", pushed" || echo ", NOT pushed yet")"
  done
  printf '\norchestrator log (%s), last lines:\n' "$LOG"; tail -6 "$LOG" 2>/dev/null
  exit 0
fi

# -- --stop -------------------------------------------------------------------------------------------------------------------
if [[ "$MODE" == stop ]]; then
  touch "$SUPPORT/STOP"
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if alive "$pid"; then kill -TERM "$pid"; echo "STOP file written; TERM sent to orchestrator $pid (it stops the launcher, which stops its shards and servers)"
  else echo "STOP file written; no orchestrator is running"; fi
  exit 0
fi

# -- --detach: the same command in its own session, immune to this shell's end --------------------------------------------
if [[ "$MODE" == detach ]]; then
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  alive "$pid" && { echo "already running (pid $pid); follow it with: $SELF --status"; exit 0; }
  rm -f "$SUPPORT/STOP"
  args=(); [[ "$STAGES" != "1 2" ]] && args=(--stage "$STAGES")
  setsid nohup "$SELF" "${args[@]}" >> "$SUPPORT/orchestrator.console.log" 2>&1 < /dev/null &
  sleep 3
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  alive "$pid" || { echo "the orchestrator did not stay up; see $SUPPORT/orchestrator.console.log and $LOG"; tail -5 "$LOG"; exit 1; }
  echo "running detached (pid $pid). Follow: $SELF --status | tail -f $LOG ; stop: $SELF --stop ; resume: $SELF --detach"
  exit 0
fi

# -- the run itself: one instance at a time ---------------------------------------------------------------------------------
exec 9> "$SUPPORT/orchestrator.lock"
flock -n 9 || { echo "another orchestrator holds $SUPPORT/orchestrator.lock" >&2; exit 1; }
echo $$ > "$PIDFILE"
rm -f "$SUPPORT/STOP"
LAUNCHER_PID="" HELPER_PIDS=()
stop_helpers() { for p in "${HELPER_PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; HELPER_PIDS=(); }
on_signal() {
  say "signal received: stopping the launcher ${LAUNCHER_PID:-(none)} and the samplers"
  if [[ -n "$LAUNCHER_PID" ]] && kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    kill -TERM "$LAUNCHER_PID" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 "$LAUNCHER_PID" 2>/dev/null || break; sleep 1; done
  fi
  stop_helpers
  say "stopped; resume with: $SELF --detach"
  exit 143
}
trap on_signal INT TERM HUP
trap stop_helpers EXIT

cd "$REPO" || die "cannot cd to $REPO"
say "campaign $CAMPAIGN -> $ROOT (stages: $STAGES; orchestrator pid $$)"

# -- checks ---------------------------------------------------------------------------------------------------------------------
ollama_up() { curl -fsS --max-time 5 "$1/api/tags" > /dev/null 2>&1; }
ollama_has() { curl -fsS --max-time 10 "$LLM_BASE_URL/api/tags" | "$PY" -c "import json,sys; sys.exit(0 if '$1' in [m['name'] for m in json.load(sys.stdin)['models']] else 1)"; }
server_urls() { local k; for (( k = 0; k < $1; k++ )); do echo "http://127.0.0.1:$((LLM_PORT + k))"; done; }

check_source() {
  local branch; branch="$(git rev-parse --abbrev-ref HEAD)"
  [[ "$branch" == "$BRANCH" ]] || die "the checkout at $REPO is on $branch, not $BRANCH"
  git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD || die "HEAD $(git rev-parse --short HEAD) does not contain $REQUIRED_COMMIT"
  say "check: $BRANCH at $(git rev-parse HEAD) contains $REQUIRED_COMMIT; tracked tree $(if [[ -z "$(git status --porcelain --untracked-files=no)" ]]; then echo clean; else echo "MODIFIED: $(git status --porcelain --untracked-files=no | tr '\n' ' ') (recorded in each run's source-changes.diff)"; fi)"
  local digest
  digest="$("$HAB_PY" -c '
import sys
from sparx_agency.core.mapping.topology.search_node_prompts import DEFAULT_PROMPT, PROMPTS
p = PROMPTS[sys.argv[1]]
assert DEFAULT_PROMPT == p.name, "the default prompt version is %s, not %s" % (DEFAULT_PROMPT, p.name)
assert p.sha256.startswith(sys.argv[2]), "prompt %s has digest %s, not %s..." % (p.name, p.sha256, sys.argv[2])
print(p.sha256)' "$PROMPT_VERSION" "$PROMPT_DIGEST_PREFIX" 2>> "$LOG")" || die "the registered prompt $PROMPT_VERSION is not the default or its digest does not start $PROMPT_DIGEST_PREFIX (see $LOG)"
  echo "$digest" > "$SUPPORT/prompt_digest.txt"
  say "check: prompt $PROMPT_VERSION is the default, digest $digest"
}

kill_stale_ollama() {   # an `ollama serve` of ours left on the stage's ports by a launcher that was killed hard
  local k port pid
  for (( k = 0; k < 4; k++ )); do
    port=$((LLM_PORT + k))
    for pid in $(ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
      if [[ "$(cat "/proc/$pid/comm" 2>/dev/null)" == ollama && "$(stat -c %U "/proc/$pid")" == "$USER" ]]; then
        say "stale ollama $pid on port $port (a launcher that did not clean up): stopping it"; kill -TERM "$pid"; sleep 3
      else
        die "port $port is held by pid $pid ($(cat "/proc/$pid/comm" 2>/dev/null)), not this campaign's ollama; set LLM_PORT"
      fi
    done
  done
}

ensure_models() {   # both models in the native store, pulled if not, through a temporary server on our port
  kill_stale_ollama
  mkdir -p "$OLLAMA_MODELS_DIR"
  OLLAMA_HOST="${LLM_BASE_URL#http://}" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" ollama serve > "$SUPPORT/ollama_check.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 60); do ollama_up "$LLM_BASE_URL" && break; sleep 1; done
  ollama_up "$LLM_BASE_URL" || { kill "$pid"; die "a temporary Ollama did not answer at $LLM_BASE_URL (see $SUPPORT/ollama_check.log)"; }
  local model ok=1
  for model in "$MODEL_1" "$MODEL_2"; do
    if ollama_has "$model"; then say "check: model present in $OLLAMA_MODELS_DIR: $model"
    else
      say "pulling $model into $OLLAMA_MODELS_DIR"
      OLLAMA_HOST="${LLM_BASE_URL#http://}" ollama pull "$model" >> "$LOG" 2>&1 && ollama_has "$model" || ok=0
    fi
  done
  kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 2
  [[ "$ok" == 1 ]] || die "a model could not be pulled (see $LOG)"
}

check_gpu() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$SUPPORT/gpu_check.txt" 2>&1 \
    || die "nvidia-smi does not answer: $(cat "$SUPPORT/gpu_check.txt")"
  say "check: GPU $(cat "$SUPPORT/gpu_check.txt"); RAM available $(awk '/MemAvailable/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo); disk free $(df -h --output=avail "$ROOT" | tail -1 | tr -d ' ')"
}

check_no_stale_launcher() {   # $1: run directory
  local pids; pids="$(pgrep -f "mp3d/run_benchmark.sh .*--output $1( |$)" || true)$(pgrep -f "mp3d\.run .*$1/" || true)"
  [[ -z "$pids" ]] || die "processes of $1 are still alive ($(echo $pids)) without an orchestrator: stop them (kill -TERM), then resume"
}

# The configuration a run froze: node_oracle.prompt digest and version, the model in both roles, GT perception.
check_run_json() {   # $1 run.json, $2 model; prints one line
  "$PY" - "$1" "$PROMPT_VERSION" "$PROMPT_DIGEST_PREFIX" "$2" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))["config"]
method = config["method"]
oracle = method["node_oracle"]
prompt, model = oracle["prompt"], oracle.get("model") or {}
assert prompt["version"] == sys.argv[2], "prompt version %r, expected %r" % (prompt["version"], sys.argv[2])
assert prompt["sha256"].startswith(sys.argv[3]), "node_oracle.prompt digest %s, expected %s..." % (prompt["sha256"], sys.argv[3])
llm = method["llm"]
assert llm["model"] == llm["reasoning_model"] == sys.argv[4], "LLM roles %s / %s, expected %s" % (llm["model"], llm["reasoning_model"], sys.argv[4])
assert config.get("perception") == "ground-truth", "perception is %r" % config.get("perception")
print("prompt %s digest %s, oracle model %s (Ollama digest %s), prompt tuned on it: %s%s" % (
    prompt["version"], prompt["sha256"], model.get("name"), str(model.get("digest"))[:12], model.get("prompt_tuned_on_it"),
    "" if model.get("prompt_tuned_on_it") else " -- MISMATCH RECORDED (intended: the prompt is the 3B's)"))
PY
}

# -- one launch of run_benchmark.sh, with its samplers and post-launch checks ------------------------------------------------
launch() {   # stage, run dir, answers dir, jobs, servers, sharded(0/1), extra launcher args...
  local s="$1" r="$2" answers="$3" jobs="$4" servers="$5" sharded="$6"; shift 6
  local model; model="$(stage_model "$s")"
  local env_args=(OLLAMA_MODE=native LLM_BASE_URL="$LLM_BASE_URL" OLLAMA_MODELS_DIR="$OLLAMA_MODELS_DIR" OLLAMA_CONTEXT_LENGTH=8192
                  LLM_MODEL="$model" LLM_REASONING_MODEL="$model" GPU_PLAN_LLM=gpu LLM_VRAM_MIB="$(stage_llm_vram "$s")"
                  HABITAT_VRAM_MIB="$HABITAT_VRAM_MIB" VRAM_CAP="$VRAM_CAP" OLLAMA_NUM_PARALLEL="$(stage_slots "$s")"
                  LLM_ANSWERS_RECORD="$answers" HAB_PY="$HAB_PY" MP3D_SCENES_DIR="$MP3D_SCENES_DIR")
  local args=(--agent rpt_room_llm --version v1 --perception ground-truth --full-records --record-log-only --stall-minutes "$STALL_MINUTES" --output "$r" "$@")
  if [[ -f "$r/gpu_plan.json" ]] && ls -d "$r"/*/ > /dev/null 2>&1; then
    # A resume keeps the plan the run was frozen with; source_sha256 is accepted only while HEAD is the recorded commit.
    env_args+=(GPU_PLAN_REPLAY="$r/gpu_plan.json")
    local accept=(gpu_plan) recorded
    recorded="$(cat "$SUPPORT/$(basename "$r").commit" 2>/dev/null)"
    if [[ -n "$recorded" ]] && git diff --quiet "$recorded" HEAD -- '*.py'; then accept+=(source_sha256)
    else die "the Python sources changed between the commit $(basename "$r") started from (${recorded:0:8}) and HEAD $(git rev-parse --short HEAD); resolve by hand"; fi
    args+=(--resume-accepting-changes "${accept[@]}")
  fi
  [[ -s "$SUPPORT/$(basename "$r").commit" ]] || git rev-parse HEAD > "$SUPPORT/$(basename "$r").commit"
  say "launch: $(basename "$r") -- $model, jobs $jobs, $servers server(s) x $(stage_slots "$s") slots at $LLM_BASE_URL.., answers -> $answers"
  echo "$(date -u +%FT%TZ) env ${env_args[*]} $LAUNCHER ${args[*]}" >> "$SUPPORT/stage${s}_commands.txt"
  env "${env_args[@]}" "$LAUNCHER" "${args[@]}" >> "$SUPPORT/stage${s}_launcher.out" 2>&1 < /dev/null &
  LAUNCHER_PID=$!
  # The services and the workers, once the run is up (the preflight and the warm-up take a minute or two).
  local url up first line
  for _ in $(seq 1 90); do sleep 10; up=1; for url in $(server_urls "$servers"); do ollama_up "$url" || up=0; done; [[ "$up" == 1 ]] && break; kill -0 "$LAUNCHER_PID" 2>/dev/null || break; done
  for url in $(server_urls "$servers"); do
    if ollama_up "$url"; then say "check: Ollama answers at $url"; else say "WARNING: no Ollama at $url yet (launcher alive: $(kill -0 "$LAUNCHER_PID" 2>/dev/null && echo yes || echo no))"; fi
  done
  for _ in $(seq 1 90); do
    first="$(ls "$r"/*/run.json "$r"/*/shard_*/run.json 2>/dev/null | head -1)"; [[ -n "$first" ]] && break
    kill -0 "$LAUNCHER_PID" 2>/dev/null || break; sleep 10
  done
  if [[ -n "$first" ]]; then
    if line="$(check_run_json "$first" "$model" 2>&1)"; then
      say "check: $first: $line"
      say "check: workers alive: $(pgrep -fc "mp3d\.run .*$r/" || true); Ollama: $(curl -fsS --max-time 10 "$LLM_BASE_URL/api/ps" | "$PY" -c 'import json,sys; print(", ".join("%s %.1f GiB VRAM" % (m["name"], m["size_vram"]/2**30) for m in json.load(sys.stdin)["models"]) or "no model loaded")' 2>/dev/null)"
    else
      say "check FAILED on $first: $line -- stopping this launch"; kill -TERM "$LAUNCHER_PID" 2>/dev/null
    fi
  elif kill -0 "$LAUNCHER_PID" 2>/dev/null; then say "WARNING: no run.json after 15 minutes; the launch goes on"
  fi
  wait "$LAUNCHER_PID"; local code=$?
  LAUNCHER_PID=""
  return "$code"
}

start_samplers() {   # stage; utilisation for the whole stage, the governor for a sharded launch ($2=1)
  local s="$1"
  "$PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.utilization_log --out "$SUPPORT/stage${s}_utilization.csv" \
    --interval 15 --disk-path "$ROOT" >> "$SUPPORT/stage${s}_utilization.stderr" 2>&1 &
  HELPER_PIDS+=($!)
  if [[ "${2:-0}" == 1 ]]; then
    "$PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.jobs_governor --jobs-file "$SUPPORT/stage${s}_jobs" \
      --ceiling "$(stage_jobs "$s")" --limit-pct "$LIMIT_PCT" --log "$SUPPORT/stage${s}_governor.csv" \
      >> "$SUPPORT/stage${s}_governor.stderr" 2>&1 &
    HELPER_PIDS+=($!)
  fi
}

# -- the smoke: 3 episodes, one per scene, the stage's model and settings ------------------------------------------------------
run_smoke() {   # stage
  local s="$1" r answers model n
  r="$(stage_root "$s")_smoke"; answers="${r}_llm_answers"; model="$(stage_model "$s")"
  if [[ -f "$SUPPORT/stage${s}_smoke.passed" ]]; then say "stage $s smoke: passed earlier ($(cat "$SUPPORT/stage${s}_smoke.passed"))"; return 0; fi
  n="$(echo $SMOKE_SCENES | wc -w)"
  check_no_stale_launcher "$r"
  say "stage $s smoke: $n episodes (first of $SMOKE_SCENES) with $model"
  start_samplers "$s" 0
  # shellcheck disable=SC2086
  launch "$s" "$r" "$answers" "$n" 1 0 --scenes $SMOKE_SCENES --limit 1 --jobs "$n"
  local code=$?
  stop_helpers
  [[ -e "$SUPPORT/STOP" ]] && die "STOP file present"
  [[ "$code" == 0 ]] || die "stage $s smoke: the launcher exited $code (see $SUPPORT/stage${s}_launcher.out, $r/run.log, $r/*.console.log)"
  local verdict
  verdict="$("$PY" - "$r" "$answers" "$model" "$n" <<'PY'
import glob, json, os, sys
root, answers, model, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(root, "*", "episodes.jsonl"))) for l in open(f) if l.strip()]
assert len(rows) == n, "%d rows, expected %d" % (len(rows), n)
bad = [r["episode_id"] for r in rows if r.get("agent_error") or r.get("termination") == "error"]
assert not bad, "agent errors in %s" % bad
calls = [json.loads(l) for f in glob.glob(os.path.join(answers, "answers-*.jsonl")) for l in open(f) if l.strip()]
assert calls, "no LLM answer was recorded under %s" % answers
models = sorted({c.get("model") for c in calls})
assert models == [model], "recorded answers came from %s, expected %s" % (models, model)
print("%d episodes (%s), SR %d/%d, %d LLM answers recorded from %s" % (
    len(rows), ", ".join("%s:%s %s" % (r["scene_id"], r["target_category"], "OK" if r["success"] else r["termination"]) for r in rows),
    sum(bool(r["success"]) for r in rows), len(rows), len(calls), model))
PY
)" || die "stage $s smoke failed its checks: $verdict"
  local line
  line="$(check_run_json "$(ls "$r"/*/run.json | head -1)" "$model" 2>&1)" || die "stage $s smoke run.json: $line"
  say "stage $s smoke run.json: $line"
  echo "$(date -u +%FT%TZ) $verdict" > "$SUPPORT/stage${s}_smoke.passed"
  say "stage $s smoke PASSED: $verdict"
}

# -- the full stage --------------------------------------------------------------------------------------------------------------
run_full() {   # stage
  local s="$1" r answers model attempt=0 before after code
  r="$(stage_root "$s")"; answers="${r}_llm_answers"; model="$(stage_model "$s")"
  if stage_complete "$r"; then
    [[ -f "$SUPPORT/stage${s}_finished_utc.txt" ]] || date -u -r "$r/campaign_summary.json" +%FT%TZ > "$SUPPORT/stage${s}_finished_utc.txt"
    say "stage $s ($model): complete already"; return 0
  fi
  check_no_stale_launcher "$r"
  [[ -f "$SUPPORT/stage${s}_started_utc.txt" ]] || date -u +%FT%TZ > "$SUPPORT/stage${s}_started_utc.txt"
  [[ -s "$SUPPORT/stage${s}_jobs" ]] || echo "$(stage_jobs "$s")" > "$SUPPORT/stage${s}_jobs"
  while :; do
    attempt=$((attempt + 1)); before="$(rows_in "$r")"
    say "stage $s ($model): launch $attempt, $before rows so far; jobs file $(cat "$SUPPORT/stage${s}_jobs")"
    start_samplers "$s" 1
    launch "$s" "$r" "$answers" "$(stage_jobs "$s")" "$(stage_servers "$s")" 1 \
      --shard-episodes "$SHARD_EPISODES" --jobs "$(stage_jobs "$s")" --jobs-file "$SUPPORT/stage${s}_jobs" --llm-servers "$(stage_servers "$s")"
    code=$?
    stop_helpers
    after="$(rows_in "$r")"
    say "stage $s: launcher exited $code with $after rows ($((after - before)) this launch)"
    if stage_complete "$r"; then
      date -u +%FT%TZ > "$SUPPORT/stage${s}_finished_utc.txt"
      say "stage $s ($model): COMPLETE -- $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); o=r["overall"]; print("%d episodes, SR %.3f, SPL %.3f" % (r["episodes"], o["success_rate"], o["spl"]))' "$r/campaign_summary.json")"
      return 0
    fi
    [[ -e "$SUPPORT/STOP" ]] && die "STOP file present: not relaunching"
    [[ $attempt -lt $MAX_RELAUNCHES ]] || die "stage $s is not complete after $attempt launches; see $SUPPORT/stage${s}_launcher.out and $r/run.log, then resume: $SELF --detach"
    [[ $after -gt $before ]] || die "stage $s made no progress in launch $attempt; see $SUPPORT/stage${s}_launcher.out and $r/run.log, then resume: $SELF --detach"
    say "stage $s: relaunching in 60 s to resume the unfinished shards"; sleep 60
  done
}

# -- the results summary, committed and pushed (never forced; a failure is logged and the run goes on) ------------------------------
commit_results() {   # stage
  local s="$1" r name out model overall
  r="$(stage_root "$s")"; model="$(stage_model "$s")"
  name="${CAMPAIGN}_$(stage_tag "$s")"; out="$RESULTS_ROOT/$name"
  if [[ ! -f "$SUPPORT/stage${s}.committed" ]]; then
    "$PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.results_summary "$r" --out "$out" \
      --title "MP3D v1 val, ground-truth perception: $model in both LLM roles, prompt $PROMPT_VERSION" \
      --started-utc "$(cat "$SUPPORT/stage${s}_started_utc.txt")" --finished-utc "$(cat "$SUPPORT/stage${s}_finished_utc.txt")" >> "$LOG" 2>&1 \
      || { say "results summary FAILED for stage $s (see $LOG); the run stays at $r; continuing"; return 0; }
    say "results summary written: $out"
    [[ "$(git rev-parse --abbrev-ref HEAD)" == "$BRANCH" ]] || { say "COMMIT SKIPPED: the checkout is no longer on $BRANCH; the summary stays at $out"; return 0; }
    overall="$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); o=d["summary"]["overall"]; p=d["provenance"]; print("%d episodes, SR %.3f, SPL %.3f; %s h" % (o["n_episodes"], o["success_rate"], o["spl"], p["duration_h"]))' "$out/summary.json")"
    git add -- "$out" >> "$LOG" 2>&1
    if git diff --cached --quiet -- "$out"; then
      say "stage $s results: nothing new to commit"
    elif git commit -q -F - -- "$out" >> "$LOG" 2>&1 <<EOF
Record MP3D v1 val GT run: $model in both LLM roles, prompt $PROMPT_VERSION

- Add the results summary of $(basename "$r") ($overall): SR and SPL overall and per category, the episode count, the model, the prompt version and digest, the source commit and the start and end times (mp3d/results/$name).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
EOF
    then say "committed $(git rev-parse --short HEAD): results of stage $s"
    else say "COMMIT FAILED for stage $s (see $LOG); the summary stays at $out; continuing"; return 0; fi
    git rev-parse HEAD > "$SUPPORT/stage${s}.committed"
  fi
  [[ -f "$SUPPORT/stage${s}.pushed" ]] && return 0
  local k
  for k in 1 2 3; do
    if git push origin "$BRANCH" >> "$LOG" 2>&1; then say "pushed $BRANCH (results of stage $s)"; touch "$SUPPORT/stage${s}.pushed"; return 0; fi
    say "push attempt $k failed (see $LOG)"; sleep 60
  done
  say "PUSH FAILED for stage $s: the commit $(cut -c1-8 "$SUPPORT/stage${s}.committed") stays local (a later run of this command retries; or: git push origin $BRANCH); continuing"
  return 0
}

# -- the stages -------------------------------------------------------------------------------------------------------------------
for s in $STAGES; do
  [[ -e "$SUPPORT/STOP" ]] && die "STOP file present"
  if [[ -f "$SUPPORT/stage${s}_finished_utc.txt" ]] && stage_complete "$(stage_root "$s")"; then
    say "stage $s ($(stage_model "$s")): complete already"
  else
    check_source
    check_gpu
    ensure_models
    run_smoke "$s"
    check_source
    run_full "$s"
  fi
  commit_results "$s"
done
say "all requested stages done ($STAGES). Results under $RESULTS_ROOT/, runs under $ROOT/"
rm -f "$PIDFILE"
