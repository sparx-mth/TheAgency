#!/usr/bin/env bash
# One command, two full HM3D v1 val runs in sequence under ground-truth perception, the same prompt version for both:
#
#   stage 1  qwen2.5:3b-instruct  in both LLM roles (room labels and node oracle)
#   stage 2  qwen2.5:14b-instruct in both LLM roles
#
# and after each stage a short results summary committed and pushed to the branch (hm3d/results/<run name>/; the run
# directories, step logs and answer stores stay on this machine). Re-running the same command RESUMES: a finished stage
# is skipped, an interrupted one continues from the episodes it had written (the launcher's --output resume), a
# summary already committed is not committed twice.
#
#   hm3d/run_gt_3b_then_14b.sh                    # the two full runs, 2,000 episodes each
#   hm3d/run_gt_3b_then_14b.sh --smoke            # one episode per scene per stage (20 x 2), the same checks and commits
#                                                 # (into runs/<campaign>_smoke/, results under hm3d/results/<campaign>_smoke_*)
#   hm3d/run_gt_3b_then_14b.sh --stage 2          # only the 14B stage (e.g. the 3B stage was run elsewhere)
#   hm3d/run_gt_3b_then_14b.sh --status           # one line per stage, and the live per-scene progress
#
# What it checks before a stage starts (and aborts on):
#   - the checkout is on BRANCH and contains REQUIRED_COMMIT (a13214e2, the prompt the comparison is about);
#   - a native Ollama holds both models (pulled if not, into OLLAMA_MODELS_DIR);
#   - the launcher's preflight passes, and the frozen configuration names prompt version PROMPT_VERSION with a digest
#     starting PROMPT_DIGEST_PREFIX (checked again in the first scene's run.json once the run is up);
#   - the services answer after the launch (Ollama at LLM_BASE_URL) and the workers are alive.
# A model/prompt-version mismatch (the 14B asked under the 3B's prompt) is NOT an error: the harness says so once on
# stderr and records node_oracle.model.prompt_tuned_on_it=false (policy_factory.oracle_model_record); this script only
# logs it.
#
# Parallelism is chosen here, not by --jobs max (which sizes itself on the VRAM free before the LLM is loaded and can
# over-commit the card): JOBS_3B / SLOTS_3B and JOBS_14B / SLOTS_14B below, from the measurements of 2026-11-11..17 on
# this RTX 4090 (24 GB): a ground-truth renderer takes ~700 MiB of VRAM; the 3B runner 5.5 GB at 10 slots; the 14B runner
# 15.1 / 16.6 / 18.2 GB at 4 / 5 / 6 slots. The target is at most 90 % of VRAM, GPU, CPU and RAM: the smoke of 2026-11-17
# measured stage 1 at VRAM 81 %, CPU <= 63 %, RAM 37 % (20 workers is one per scene, the campaign's ceiling) and stage 2 at
# VRAM 91 % with 9 workers -- hence 8 (~88 %). GPU compute stays low under both models because Ollama's runner samples
# the JSON reply on one CPU thread per request (the 3B's oracle call takes ~25 s with 20 workers waiting on 10 slots).
#
# Logs, all under runs/<campaign>/: support/orchestrator.log (this script), stage*/run.log + ollama.log + gpu_plan.json +
# preflight.json + source-commit.txt + source-changes.diff (the launcher), stage*/<scene>/run.json (the frozen
# configuration: model, prompt version and digest, settings), stage*/<scene>/episodes.jsonl (full per-episode records),
# stage*/<scene>/recordings/<key>/steps.jsonl + trajectory.csv + map_journal.npz (every decision and pose, no video),
# stage*/<scene>.console.log, stage*_llm_answers/ (every LLM request with its prompts and reply), support/stage*_utilization.csv
# (GPU/VRAM/CPU/RAM every 15 s).
set -uo pipefail

# -- the plan -------------------------------------------------------------------------------------------------------
BRANCH="${BRANCH:-feat/objnav-habitat-hm3d-nadav}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-a13214e2}"
PROMPT_VERSION="${PROMPT_VERSION:-facts-qwen2.5-3b-v1}"
PROMPT_DIGEST_PREFIX="${PROMPT_DIGEST_PREFIX:-2d025f01}"
MODEL_3B="${MODEL_3B:-qwen2.5:3b-instruct}"
MODEL_14B="${MODEL_14B:-qwen2.5:14b-instruct}"
JOBS_3B="${JOBS_3B:-20}"      SLOTS_3B="${SLOTS_3B:-10}"    LLM_VRAM_3B="${LLM_VRAM_3B:-6000}"
JOBS_14B="${JOBS_14B:-8}"     SLOTS_14B="${SLOTS_14B:-4}"   LLM_VRAM_14B="${LLM_VRAM_14B:-15200}"
HABITAT_VRAM_MIB="${HABITAT_VRAM_MIB:-750}"
STALL_MINUTES="${STALL_MINUTES:-30}"
MAX_RELAUNCHES="${MAX_RELAUNCHES:-3}"           # a stage that exits unclean is relaunched (resumed) this many times
CAMPAIGN="${CAMPAIGN:-hm3d_v1_gt_3b_vs_14b}"
# Longest scenes first (the Nov-4 and Nov-11 runs' durations), so the tail of a stage is short.
SCENE_ORDER="00878-XB4GS9ShBRE 00820-mL8ThkuaVTM 00839-zt1RVoi7PcG 00835-q3zU7Yy5E5s 00890-6s7QHgap2fW 00832-qyAac8rV8Zk 00802-wcojb4TFT35 00814-p53SfW6mjZe 00873-bxsVRursffK 00800-TEEsavR23oF 00891-cvZr5TUy5C5 00848-ziup5kvtCCR 00853-5cdEh9F2hJL 00843-DYehNKdT76V 00880-Nfvxx8J5NCo 00824-Dd4bFSTQ8gi 00876-mv2HUxq3B53 00877-4ok3usBNeis 00813-svBbv1Pavdk 00829-QaLdnwvtxbs"

# -- where things are (this PC) --------------------------------------------------------------------------------------
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../../../../.." && pwd)"
export HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/objnav-habitat/bin/python}"
export OLLAMA_MODE="${OLLAMA_MODE:-native}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11435}"
export OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$HOME/models/objnav/ollama}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
LAUNCHER="$HERE/run_benchmark.sh"
RESULTS_ROOT="$HERE/results"
PY="${PY:-$REPO/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

# -- the command line ------------------------------------------------------------------------------------------------
SMOKE=0 STAGES="1 2" STATUS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --stage) STAGES="$2"; shift 2 ;;
    --status) STATUS=1; shift ;;
    -h|--help) sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$SMOKE" == 1 ]] && CAMPAIGN="${CAMPAIGN}_smoke"
ROOT="$REPO/runs/$CAMPAIGN"
SUPPORT="$ROOT/support"
mkdir -p "$SUPPORT"
LOG="$SUPPORT/orchestrator.log"

say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
die() { say "ABORT: $*"; exit 1; }

stage_root() { case "$1" in 1) echo "$ROOT/stage1_3b" ;; 2) echo "$ROOT/stage2_14b" ;; esac; }
stage_model() { case "$1" in 1) echo "$MODEL_3B" ;; 2) echo "$MODEL_14B" ;; esac; }
stage_jobs() { case "$1" in 1) echo "$JOBS_3B" ;; 2) echo "$JOBS_14B" ;; esac; }
stage_slots() { case "$1" in 1) echo "$SLOTS_3B" ;; 2) echo "$SLOTS_14B" ;; esac; }
stage_llm_vram() { case "$1" in 1) echo "$LLM_VRAM_3B" ;; 2) echo "$LLM_VRAM_14B" ;; esac; }
# A full stage is complete when the aggregate says the whole split is (campaign_summary.json "complete"); a smoke stage
# -- a subset, never "complete" -- when every scene child of the campaign exited 0 (campaign.json).
stage_complete() {
  [[ -f "$1/campaign_summary.json" && -f "$1/campaign.json" ]] || return 1
  "$PY" -c 'import json,sys
summary = json.load(open(sys.argv[1] + "/campaign_summary.json")); campaign = json.load(open(sys.argv[1] + "/campaign.json"))
if sys.argv[2] == "1": sys.exit(0 if campaign["runs"] and all(r["returncode"] == 0 for r in campaign["runs"]) and not summary["scenes_without_rows"] else 1)
sys.exit(0 if summary["complete"] else 1)' "$1" "$SMOKE" 2>/dev/null
}
rows_in() { cat "$1"/*/episodes.jsonl 2>/dev/null | wc -l; }

# -- --status --------------------------------------------------------------------------------------------------------
if [[ "$STATUS" == 1 ]]; then
  for s in 1 2; do
    r="$(stage_root $s)"
    if [[ ! -d "$r" ]]; then echo "stage $s ($(stage_model $s)): not started"; continue; fi
    if stage_complete "$r"; then
      echo "stage $s ($(stage_model $s)): COMPLETE -- $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); o=r["overall"]; print("%d episodes, SR %.3f SPL %.3f DTG %.2f" % (r["episodes"], o["success_rate"], o["spl"], o["distance_to_goal_m"]))' "$r/campaign_summary.json")"
    else
      echo "stage $s ($(stage_model $s)): $(rows_in "$r") rows written; workers alive: $(pgrep -fc "hm3d.run --version v1 .*$r" || true); last launcher line: $(tail -1 "$r/run.log" 2>/dev/null | cut -c1-120)"
      for p in "$r"/*/progress.json; do [[ -f "$p" ]] && "$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print("    %-18s %3d/%-3d SR %.2f SPL %.2f" % (sys.argv[1].split("/")[-2], p["done"], p["total"], p["overall"]["SR"], p["overall"]["SPL"]))' "$p"; done
    fi
  done
  echo "orchestrator log: $LOG"; tail -3 "$LOG" 2>/dev/null
  exit 0
fi

# -- pre-checks that hold for both stages -------------------------------------------------------------------------------
cd "$REPO" || die "cannot cd to $REPO"
say "campaign $CAMPAIGN -> $ROOT (smoke=$SMOKE, stages: $STAGES)"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$CURRENT_BRANCH" == "$BRANCH" ]] || die "on branch $CURRENT_BRANCH, not $BRANCH (the results are committed to the branch the run was made on)"
git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD || die "HEAD $(git rev-parse --short HEAD) does not contain $REQUIRED_COMMIT"
say "branch $BRANCH at $(git rev-parse HEAD) contains $REQUIRED_COMMIT; tree: $(if [[ -z "$(git status --porcelain --untracked-files=no)" ]]; then echo clean; else echo "MODIFIED ($(git status --porcelain --untracked-files=no | wc -l) files; recorded in each stage's source-changes.diff)"; fi)"
[[ -x "$HAB_PY" ]] || die "no Habitat interpreter at $HAB_PY"
command -v ollama > /dev/null || die "ollama is not on PATH"
command -v nvidia-smi > /dev/null || die "nvidia-smi is not on PATH"
"$PY" -c '
import sys
from sparx_agency.core.mapping.topology.search_node_prompts import PROMPTS, DEFAULT_PROMPT
p = PROMPTS[sys.argv[1]]
assert DEFAULT_PROMPT == sys.argv[1], "the default prompt version is %s, not %s" % (DEFAULT_PROMPT, sys.argv[1])
assert p.sha256.startswith(sys.argv[2]), "prompt %s has digest %s, not %s..." % (p.name, p.sha256, sys.argv[2])
print(p.sha256)' "$PROMPT_VERSION" "$PROMPT_DIGEST_PREFIX" 2>> "$LOG" | tail -1 > "$SUPPORT/prompt_digest.txt" \
  && [[ -s "$SUPPORT/prompt_digest.txt" ]] || die "the registered prompt version $PROMPT_VERSION does not have digest $PROMPT_DIGEST_PREFIX... (see $LOG)"
say "prompt version $PROMPT_VERSION registered with digest $(cut -c1-16 "$SUPPORT/prompt_digest.txt")... (the default; the 14B stage is asked under it on purpose)"

# -- Ollama: both models present, pulled if not (a temporary native server when none answers) --------------------------
ollama_up() { curl -fsS --max-time 5 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; }
ollama_has() { curl -fsS --max-time 10 "$LLM_BASE_URL/api/tags" | "$PY" -c "import json,sys; sys.exit(0 if '$1' in [m['name'] for m in json.load(sys.stdin)['models']] else 1)"; }
ensure_models() {
  local started=0 pid=""
  if ! ollama_up; then
    say "starting a temporary native Ollama at $LLM_BASE_URL to check the models"
    mkdir -p "$OLLAMA_MODELS_DIR"
    OLLAMA_HOST="${LLM_BASE_URL#http://}" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" ollama serve > "$SUPPORT/ollama_check.log" 2>&1 &
    pid=$!; started=1
    for _ in $(seq 1 60); do ollama_up && break; sleep 1; done
    ollama_up || die "the temporary Ollama did not answer at $LLM_BASE_URL (see $SUPPORT/ollama_check.log)"
  fi
  for model in "$MODEL_3B" "$MODEL_14B"; do
    if ollama_has "$model"; then say "model present: $model"
    else say "pulling $model into $OLLAMA_MODELS_DIR (several GB)"; OLLAMA_HOST="${LLM_BASE_URL#http://}" ollama pull "$model" >> "$LOG" 2>&1 || die "could not pull $model"
         ollama_has "$model" || die "$model is still missing after the pull"; fi
  done
  if [[ "$started" == 1 ]]; then kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 2; say "temporary Ollama stopped (each stage's launcher starts its own, with the stage's slot count)"; fi
}
ensure_models
if ollama_up; then
  die "an Ollama already answers at $LLM_BASE_URL: the launcher would budget it as placed and keep its slot count; stop it first (this script starts one per stage)"
fi

# -- the results summary, committed and pushed (never forced) ------------------------------------------------------------
commit_results() {   # stage, stage root
  local s="$1" r="$2" name out title
  name="$(basename "$ROOT")_stage${s}_$(stage_model "$s" | tr ':.' '__')"
  out="$RESULTS_ROOT/$name"
  title="HM3D v1 val, ground-truth perception, $(stage_model "$s") in both LLM roles, prompt $PROMPT_VERSION$( [[ "$SMOKE" == 1 ]] && echo " (SMOKE: one episode per scene)")"
  "$PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.results_summary "$r" --out "$out" --title "$title" >> "$LOG" 2>&1 \
    || { say "results summary FAILED for stage $s (see $LOG); the run stays at $r"; return 1; }
  say "results summary written: $out"
  git add -- "$out" >> "$LOG" 2>&1
  if git diff --cached --quiet -- "$out"; then say "results of stage $s already committed; nothing to commit"; return 0; fi
  local overall
  overall="$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); o=d["summary"]["overall"]; p=d["provenance"]; print("%d episodes, SR %.3f, SPL %.3f, DTG %.2f m; %s h" % (o["n_episodes"], o["success_rate"], o["spl"], o["distance_to_goal_m"] or float("nan"), p["duration_h"]))' "$out/summary.json")"
  git commit -q -F - -- "$out" <<EOF
Record HM3D v1 GT run: $(stage_model "$s") in both roles, prompt $PROMPT_VERSION$( [[ "$SMOKE" == 1 ]] && echo " (smoke)")

- Add the results summary of $(basename "$r") ($overall): SR and SPL overall and per category, the model, the prompt version and digest, the source commit and the start and end times (hm3d/results/$name).
EOF
  say "committed $(git rev-parse --short HEAD): results of stage $s"
  if git push >> "$LOG" 2>&1; then say "pushed to $(git rev-parse --abbrev-ref --symbolic-full-name '@{u}')"
  else say "PUSH FAILED (the commit stays local; push it by hand later: git push); continuing"; fi
}

# -- one stage -------------------------------------------------------------------------------------------------------
run_stage() {   # stage number
  local s="$1" r model jobs slots llm_vram attempt=0 before after code
  r="$(stage_root "$s")"; model="$(stage_model "$s")"; jobs="$(stage_jobs "$s")"; slots="$(stage_slots "$s")"; llm_vram="$(stage_llm_vram "$s")"
  if stage_complete "$r"; then say "stage $s ($model): already complete at $r"; commit_results "$s" "$r"; return 0; fi
  mkdir -p "$r"
  local common=(--version v1 --perception ground-truth --jobs "$jobs" --full-records --record-log-only
                --stall-minutes "$STALL_MINUTES" --output "$r" --scenes $SCENE_ORDER)
  [[ "$SMOKE" == 1 ]] && common+=(--limit 1)
  local stage_env=(HAB_PY="$HAB_PY" OLLAMA_MODE="$OLLAMA_MODE" LLM_BASE_URL="$LLM_BASE_URL" OLLAMA_MODELS_DIR="$OLLAMA_MODELS_DIR"
             OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" LLM_MODEL="$model" LLM_REASONING_MODEL="$model"
             GPU_PLAN_LLM=gpu LLM_VRAM_MIB="$llm_vram" HABITAT_VRAM_MIB="$HABITAT_VRAM_MIB" OLLAMA_NUM_PARALLEL="$slots"
             LLM_ANSWERS_RECORD="${r}_llm_answers")
  # the preflight: services up, the frozen configuration printed, services down; the prompt digest is read off it
  if [[ ! -f "$r/campaign.json" ]] && ! ls -d "$r"/*/ > /dev/null 2>&1; then
    say "stage $s ($model): preflight (jobs $jobs, Ollama slots $slots, LLM budget $llm_vram MiB, renderer $HABITAT_VRAM_MIB MiB)"
    env "${stage_env[@]}" "$LAUNCHER" "${common[@]}" --preflight > "$SUPPORT/stage${s}_preflight.out" 2>&1 \
      || die "the preflight of stage $s refused the run (see $SUPPORT/stage${s}_preflight.out and $r/preflight.json)"
    "$PY" - "$r/preflight.json" "$PROMPT_VERSION" "$PROMPT_DIGEST_PREFIX" "$model" <<'PYCHECK' >> "$LOG" 2>&1 || die "the frozen configuration of stage $s is not the one asked for (see $LOG)"
import json, sys
config = json.load(open(sys.argv[1]))["configuration"]
oracle = config["method"]["node_oracle"]
prompt, model = oracle["prompt"], oracle.get("model") or {}
assert prompt["version"] == sys.argv[2], "prompt version %r, expected %r" % (prompt["version"], sys.argv[2])
assert prompt["sha256"].startswith(sys.argv[3]), "prompt digest %s, expected %s..." % (prompt["sha256"], sys.argv[3])
llm = config["method"]["llm"]
assert llm["model"] == llm["reasoning_model"] == sys.argv[4], "LLM roles %s / %s, expected %s" % (llm["model"], llm["reasoning_model"], sys.argv[4])
assert config["method"].get("perception") == "ground-truth", "perception is not ground-truth"
assert config["recording"]["enabled"] and not config["recording"].get("video", True), "the log-only recording is not on"
assert not config["lean_records"], "lean records: the full per-episode diagnostics are wanted"
print("preflight ok: model %s (prompt tuned on it: %s), prompt %s %s, perception %s, gpu plan %s" % (
    model.get("name"), model.get("prompt_tuned_on_it"), prompt["version"], prompt["sha256"][:16], config["method"]["perception"],
    {p["component"]: p["device"] for p in (config.get("gpu_plan") or {}).get("placements", [])}))
PYCHECK
    say "stage $s: $(grep '^preflight ok' "$LOG" | tail -1)"
  fi
  # the run, relaunched (resumed in place) when it exits unclean and had made progress
  while :; do
    attempt=$((attempt + 1)); before="$(rows_in "$r")"
    say "stage $s ($model): launch $attempt (rows so far $before); progress: $0 --status  |  $LAUNCHER --status $r"
    "$PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.utilization_log --out "$SUPPORT/stage${s}_utilization.csv" --interval 15 \
      >> "$SUPPORT/stage${s}_utilization.stderr" 2>&1 &
    local sampler=$!
    local replay=()
    [[ -f "$r/gpu_plan.json" ]] && ls -d "$r"/*/ > /dev/null 2>&1 && replay=(GPU_PLAN_REPLAY="$r/gpu_plan.json")   # a resume keeps its frozen plan
    env "${stage_env[@]}" "${replay[@]}" "$LAUNCHER" "${common[@]}" >> "$SUPPORT/stage${s}_launcher.out" 2>&1 &
    local launcher=$!
    echo "$launcher" > "$SUPPORT/stage${s}_launcher.pid"
    # the services and the first worker's record, checked once the run is up
    for _ in $(seq 1 90); do sleep 10; ollama_up && break; done
    ollama_up && say "stage $s: Ollama answers at $LLM_BASE_URL ($(curl -fsS --max-time 10 "$LLM_BASE_URL/api/ps" | "$PY" -c 'import json,sys; print(", ".join("%s %.1f GiB in VRAM" % (m["name"], m["size_vram"]/2**30) for m in json.load(sys.stdin)["models"]) or "no model loaded yet")'))" \
               || say "stage $s: WARNING Ollama does not answer at $LLM_BASE_URL yet"
    for _ in $(seq 1 60); do sleep 10; ls "$r"/*/run.json > /dev/null 2>&1 && break; kill -0 "$launcher" 2>/dev/null || break; done
    local first; first="$(ls "$r"/*/run.json 2>/dev/null | head -1)"
    if [[ -n "$first" ]]; then
      "$PY" -c '
import json, sys
o = json.load(open(sys.argv[1]))["config"]["method"]["node_oracle"]
assert o["prompt"]["sha256"].startswith(sys.argv[2]), "run.json prompt digest %s" % o["prompt"]["sha256"]
print("run.json: prompt %s %s..., oracle model %s, prompt tuned on it: %s" % (o["prompt"]["version"], o["prompt"]["sha256"][:16], o["model"]["name"], o["model"]["prompt_tuned_on_it"]))' "$first" "$PROMPT_DIGEST_PREFIX" >> "$LOG" 2>&1 \
        && say "stage $s: $(grep '^run.json:' "$LOG" | tail -1)  workers alive: $(pgrep -fc "hm3d.run --version v1" || true)" \
        || { say "stage $s: the first run.json does not carry prompt digest $PROMPT_DIGEST_PREFIX; stopping the stage"; kill -TERM "$launcher" 2>/dev/null; }
    fi
    wait "$launcher"; code=$?
    kill -TERM "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    after="$(rows_in "$r")"
    say "stage $s: launcher exited $code with $after rows ($((after - before)) this launch)"
    if stage_complete "$r"; then say "stage $s ($model): COMPLETE"; break; fi
    if [[ $attempt -ge $MAX_RELAUNCHES ]]; then die "stage $s is not complete after $attempt launches; fix what $SUPPORT/stage${s}_launcher.out and $r/<scene>.console.log say, then re-run this command to resume"; fi
    [[ $after -gt $before ]] || die "stage $s made no progress in launch $attempt; see $SUPPORT/stage${s}_launcher.out and $r/<scene>.console.log, then re-run this command to resume"
    say "stage $s: relaunching to resume the unfinished scenes"; sleep 10
  done
  commit_results "$s" "$r"
}

for s in $STAGES; do run_stage "$s"; done
say "all requested stages done: $STAGES. Results under $RESULTS_ROOT/, runs under $ROOT/"
