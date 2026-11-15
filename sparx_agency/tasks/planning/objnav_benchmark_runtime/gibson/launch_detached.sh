#!/usr/bin/env bash
# The detached full-split run of BENCHMARK.md 2.3 as one command -- and the one command that brings the same
# run back after the machine restarted.
#
#   gibson/launch_detached.sh <output>          # a NEW run: run_benchmark.sh --output <output> in its own session,
#                                               #   the monitor beside it, the supervisor once the run is past its preflight
#   gibson/launch_detached.sh <output>          # <output> already holds a run: nothing is relaunched by hand -- the
#                                               #   Ollama container is started (the supervisor cannot replay a container)
#                                               #   and a fresh supervisor resumes the run from its replay snapshot
#   gibson/launch_detached.sh <output> --stop   # end the run for good: STOP file, TERM to the launcher's process group
#                                               #   (and cancel a launch scheduled for the next boot)
#   gibson/launch_detached.sh <output> --after-reboot   # the GPU is unusable until the machine restarts (a driver
#                                               #   upgraded under the running kernel module): the NEW run is launched
#                                               #   at the next boot instead of now -- the launch's environment goes to
#                                               #   <output>/launch.env and a one-shot @reboot crontab entry runs the
#                                               #   launch once nvidia-smi and docker answer; <output>/boot.log says how it went
#
# Everything run_benchmark.sh reads from the environment applies to a new run (LLM_REASONING_MODEL, GPU_PLAN,
# LLM_VRAM_MIB, OLLAMA_CONTAINER, SEGMENTATION, ...); a resume replays the processes with the environment they had.
# RUN_ARGS is a command-line fragment appended to run_benchmark.sh for a new run -- RUN_ARGS="--agent falcon-yolo"
# launches the FALCON + YOLO + closing baseline (no LLM is started, checked or ensured on a resume).
# The supervisor's knobs (PROGRESS_ENTRY -- required for a new run, so that publishing closes THIS run's entry --
# MAX_RESUMES, RESUME_ARGS, SUPERVISOR_ARGS such as --dry-run) and the container name are written to <output>/launch.env
# at the first launch and read back on a resume, so after a reboot the command needs no environment at all.
# <output> is made absolute: the supervisor finds the run's processes by the --output in their command line.
set -euo pipefail

usage() { sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
REPO="${OBJNAV_REPO:-$(cd -- "$HERE/../../../../.." && pwd)}"
HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/habitat/bin/python}"
MONITOR_MODULE="sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.monitor"
PERSISTED=(PROGRESS_ENTRY SUPERVISOR_ARGS MAX_RESUMES RESUME_ARGS RUN_ARGS MONITOR_INTERVAL_S OLLAMA_MODE OLLAMA_CONTAINER)
# Every variable run_benchmark.sh, gpu_plan.py and the supervisor read from the environment, by name family: what a
# launch scheduled for the next boot carries over, so that it is the launch that would have been made now.
RUN_ENV_PATTERN='^(HAB_PY|DETECT_PY|DATA|MODELS|SEED|SEGMENTATION|ALLOW_SHARED_GPU|CLIP_WEIGHTS|SAM_WEIGHTS|RUNS_ROOT|REMOTE|BRANCH|POLL_S|RESUME_DELAY_S|PUSH_RETRIES|PUSH_RETRY_S|(GIBSON|YOLO|DETECTOR|SEGMENTER|OLLAMA|LLM|OMP|HABITAT|MAGNUM|GPU|OBJNAV)_[A-Z0-9_]*)$'
BOOT_WAIT_S="${BOOT_WAIT_S:-900}"          # how long the boot launch waits for nvidia-smi and docker before it gives up

OUT="" STOP=0 SCHEDULE=0 BOOT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop) STOP=1 ;;
    --after-reboot) SCHEDULE=1 ;;
    --boot) BOOT=1 ;;                     # internal: what the @reboot crontab entry runs
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *) OUT="$1" ;;
  esac
  shift
done
[[ -n "$OUT" ]] || { usage >&2; echo "ERROR: the run directory is required" >&2; exit 2; }
mkdir -p "$OUT"; OUT="$(cd -- "$OUT" && pwd)"; RUN_NAME="$(basename -- "$OUT")"

say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }
# A pid is this run's only if its command line names the run: after a reboot the number may belong to anything.
ours() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null && [[ -r "/proc/$1/cmdline" ]] && tr '\0' ' ' 2>/dev/null < "/proc/$1/cmdline" | grep -qF -- "$RUN_NAME"; }
launcher_pid() { cat "$OUT/launcher.pid" 2>/dev/null || true; }
supervisor_pid() { cat "$OUT/supervisor.pid" 2>/dev/null || true; }

# -- the boot launch's crontab entry: one line, tagged with the run directory, removed the moment it fires ----------
CRON_TAG="# gibson-launch:$OUT"
scheduled() { crontab -l 2>/dev/null | grep -qF -- "$CRON_TAG"; }
unschedule() {
  local current remaining
  current="$(crontab -l 2>/dev/null || true)"
  grep -qF -- "$CRON_TAG" <<< "$current" || return 0
  remaining="$(grep -vF -- "$CRON_TAG" <<< "$current" || true)"
  if [[ -n "$remaining" ]]; then printf '%s\n' "$remaining" | crontab -; else crontab -r 2>/dev/null || true; fi
}
schedule() {   # cron runs the line through /bin/sh with PATH=/usr/bin:/bin and HOME set -- all the script needs
  [[ "$OUT" != *%* && "$SELF" != *%* ]] || die "a crontab line cannot carry a '%' (in $OUT or $SELF)"
  { crontab -l 2>/dev/null | grep -vF -- "$CRON_TAG" || true
    printf '@reboot %q %q --boot >> %q 2>&1 %s\n' "$SELF" "$OUT" "$OUT/boot.log" "$CRON_TAG"; } | crontab -
}
persist_env() {   # the named variables that are set, once each, as KEY=VALUE lines in <output>/launch.env
  local key; local -A seen=()
  : > "$OUT/launch.env"
  for key in "$@"; do
    [[ -n "${!key:-}" && -z "${seen[$key]:-}" ]] || continue
    seen[$key]=1; printf '%s=%s\n' "$key" "${!key}" >> "$OUT/launch.env"
  done
}

cd "$REPO"
[[ -x "$HAB_PY" ]] || die "Habitat interpreter not found at $HAB_PY (set HAB_PY)"

# -- stop -------------------------------------------------------------------------------------------------
if [[ "$STOP" == 1 ]]; then
  if scheduled; then unschedule; say "the boot launch of $RUN_NAME is cancelled (crontab entry removed)"; fi
  if [[ ! -e "$OUT/run.json" && ! -s "$OUT/episodes.jsonl" && ! -f "$OUT/replay/run.cmdline" ]] && ! ours "$(launcher_pid)"; then
    say "nothing of $RUN_NAME has run; no STOP file needed"; exit 0
  fi
  touch "$OUT/STOP"
  pid="$(launcher_pid)"
  if ours "$pid"; then kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; say "TERM sent to launcher $pid and its group; the supervisor sees the STOP file and will not resume"
  else say "no live launcher for $RUN_NAME; STOP file written so no supervisor resumes it"; fi
  exit 0
fi

# -- the boot launch: the @reboot entry fired; one shot, then wait for the card and the container runtime -------------
if [[ "$BOOT" == 1 ]]; then
  unschedule                            # whatever happens next, the entry does not fire at the boot after this one
  say "boot launch of $RUN_NAME (pid $$): waiting up to ${BOOT_WAIT_S}s for nvidia-smi and docker"
  waited=0
  until nvidia-smi -L > /dev/null 2>&1 && docker info > /dev/null 2>&1; do
    sleep 10; waited=$((waited + 10))
    [[ $waited -lt $BOOT_WAIT_S ]] || die "no working nvidia-smi and docker after ${waited}s: NOT launching -- a run launched on the CPU could never move to the GPU (BENCHMARK.md 2.3); launch by hand once the card is back"
  done
  say "after ${waited}s: $(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader | head -1); docker answers"
fi

# -- the persisted settings: written at the first launch (or when it is scheduled), read back (environment wins) ------
if [[ -f "$OUT/launch.env" && "$SCHEDULE" != 1 ]]; then       # a new schedule starts from this shell's environment alone
  while IFS='=' read -r key value; do
    [[ -n "$key" && -z "${!key:-}" ]] && export "$key=$value"
  done < "$OUT/launch.env"
fi
OLLAMA_MODE="${OLLAMA_MODE:-docker}"; OLLAMA_CONTAINER="${OLLAMA_CONTAINER:-ollama-scene-graph}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-300}"
export OLLAMA_MODE OLLAMA_CONTAINER MONITOR_INTERVAL_S
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
BASELINE=0; [[ " ${RUN_ARGS:-} " == *" --agent falcon-yolo "* ]] && BASELINE=1     # the baseline needs no LLM

start_supervisor() {
  # shellcheck disable=SC2206   # word-splitting SUPERVISOR_ARGS is the point: it is a command-line fragment
  local extra=(${SUPERVISOR_ARGS:-}) pid="" waited=0
  setsid nohup "$HERE/supervise_benchmark.sh" "$OUT" "${extra[@]}" > /dev/null 2>&1 < /dev/null &
  until pid="$(supervisor_pid)" && ours "$pid"; do     # a stale supervisor.pid from an earlier boot is not ours
    sleep 1; waited=$((waited + 1))
    [[ $waited -lt 20 ]] || die "the supervisor did not start within ${waited}s (see $OUT/supervisor.log)"
  done
  say "supervisor $pid: resumes <= ${MAX_RESUMES:-2}, publishes at the end${SUPERVISOR_ARGS:+ ($SUPERVISOR_ARGS)}, entry ${PROGRESS_ENTRY:-<the default of the supervisor>}"
}

# -- resume: the run exists; the supervisor replays it from its snapshot once the container answers -------------------
if [[ -s "$OUT/episodes.jsonl" || -f "$OUT/replay/run.cmdline" ]]; then
  [[ "$SCHEDULE" != 1 ]] || die "$RUN_NAME exists: --after-reboot is for a run's first launch; after the boot, '$0 $OUT' resumes it"
  [[ -s "$OUT/summary.json" ]] && die "$RUN_NAME is finished (summary.json); nothing to resume"
  [[ -e "$OUT/STOP" ]] && die "$RUN_NAME carries a STOP file: remove it first if the run should go on"
  ours "$(launcher_pid)" && die "$RUN_NAME is running (launcher $(launcher_pid)); use run_benchmark.sh --status $OUT"
  ours "$(supervisor_pid)" && die "a supervisor ($(supervisor_pid)) already watches $RUN_NAME; it resumes the run itself"
  [[ -f "$OUT/replay/run.cmdline" ]] || die "no replay snapshot under $OUT/replay: this run cannot be resumed by replay (BENCHMARK.md 2.3 for the by-hand resume)"
  if [[ "$BASELINE" != 1 ]] && ! curl -fsS --max-time 5 "$LLM_BASE_URL/api/tags" > /dev/null 2>&1; then
    case "$OLLAMA_MODE" in
      docker) say "starting the Ollama container $OLLAMA_CONTAINER (a container has no /proc snapshot the supervisor could replay)"
              docker start "$OLLAMA_CONTAINER" > /dev/null || die "docker start $OLLAMA_CONTAINER failed" ;;
      native) say "no Ollama at $LLM_BASE_URL; the supervisor replays the snapshotted 'ollama serve'" ;;
      external) die "no LLM service answers at $LLM_BASE_URL (OLLAMA_MODE=external: start it yourself)" ;;
    esac
  fi
  say "resuming $RUN_NAME: $(wc -l < "$OUT/episodes.jsonl" 2>/dev/null || echo 0) episodes on disk; the supervisor relaunches the run in ~2 minutes"
  start_supervisor
  exit 0
fi

# -- a new run -------------------------------------------------------------------------------------------------
[[ -e "$OUT/run.json" ]] && die "$OUT holds a run.json but no episodes: remove the directory or pick another"
if [[ "$BOOT" != 1 && "$SCHEDULE" != 1 ]] && scheduled; then       # (--after-reboot again replaces the schedule)
  die "$RUN_NAME is scheduled for the next boot (crontab); $0 $OUT --stop cancels that first"
fi
[[ -n "${PROGRESS_ENTRY:-}" ]] || die "set PROGRESS_ENTRY to this run's progress entry (docs/progress/entries/NNN-*.md): the supervisor closes it when it publishes, and its default is an earlier run's"
[[ -f "$REPO/$PROGRESS_ENTRY" ]] || die "PROGRESS_ENTRY $PROGRESS_ENTRY does not exist under $REPO"

if [[ "$SCHEDULE" == 1 ]]; then
  [[ "$BOOT" != 1 ]] || die "--after-reboot and --boot exclude each other"
  ours "$(launcher_pid)" && die "$RUN_NAME is running (launcher $(launcher_pid)); nothing to schedule"
  mapfile -t run_env < <(compgen -e | grep -E "$RUN_ENV_PATTERN" || true)
  persist_env "${PERSISTED[@]}" "${run_env[@]}"
  schedule
  say "scheduled: $RUN_NAME launches at the next boot, once nvidia-smi and docker answer (crontab @reboot; log: $OUT/boot.log)"
  say "with $(wc -l < "$OUT/launch.env") variables from this shell in $OUT/launch.env:"
  sed 's/^/    /' "$OUT/launch.env"
  say "cancel: $0 $OUT --stop   |   after the boot: $HERE/run_benchmark.sh --status $OUT"
  exit 0
fi
[[ "$BOOT" == 1 ]] || persist_env "${PERSISTED[@]}"      # the boot launch keeps the file --after-reboot wrote

say "launching $RUN_NAME detached (log: $OUT/launcher.log)${RUN_ARGS:+ with $RUN_ARGS}"
# shellcheck disable=SC2206   # word-splitting RUN_ARGS is the point: it is a command-line fragment
run_args=(${RUN_ARGS:-})
setsid nohup bash -c 'echo $$ > "$1/launcher.pid"; out="$1"; run="$2"; shift 2; exec "$run" --output "$out" "$@"' \
  _ "$OUT" "$HERE/run_benchmark.sh" "${run_args[@]}" > "$OUT/launcher.log" 2>&1 < /dev/null &
setsid nohup "$HAB_PY" -m "$MONITOR_MODULE" "$OUT" --interval "$MONITOR_INTERVAL_S" > /dev/null 2>&1 < /dev/null &

# The supervisor snapshots the live processes, so it starts once run.json exists (the run is past its preflight and
# has frozen its configuration) and gibson.run is up; the first detector start can take a minute.
waited=0
until [[ -s "$OUT/run.json" ]] && pgrep -f -- "gibson\.run .*--output ${OUT}( |$)" > /dev/null; do
  sleep 5; waited=$((waited + 5))
  if ! ours "$(launcher_pid)"; then
    say "the launcher ended before the run started:"; tail -20 "$OUT/launcher.log" >&2; exit 1
  fi
  [[ $waited -lt 1200 ]] || die "no run.json after ${waited}s; see $OUT/launcher.log"
done
say "run started (launcher $(launcher_pid)); frozen configuration in $OUT/run.json"
start_supervisor
say "watch: $HERE/run_benchmark.sh --status $OUT   |   tail -f $OUT/run.log   |   stop: $0 $OUT --stop"


