#!/usr/bin/env bash
# Unattended companion of run_benchmark.sh for a detached run: waits for it, resumes it if it dies,
# and when it completes commits the result (SR, SPL, DTG) and pushes it to the branch.
#
#   gibson/supervise_benchmark.sh <output>                  # wait -> resume (up to MAX_RESUMES) -> publish
#   gibson/supervise_benchmark.sh <output> --dry-run        # ... build the commit in a temporary worktree, push nothing
#   gibson/supervise_benchmark.sh <output> --publish-only   # the run is finished: publish now, no waiting
#   gibson/supervise_benchmark.sh <output> --render         # print the results markdown of a finished run and exit
#   gibson/supervise_benchmark.sh <output> --snapshot       # (re)capture the live processes for a later resume, and exit
#   gibson/supervise_benchmark.sh <output> --no-resume      # supervise, but never relaunch a dead run
#
# Resume. A resume must reproduce the frozen configuration byte for byte, and run_benchmark.sh cannot:
# it re-plans the GPU (gpu_plan.json's free/used MiB differ between launches) and that plan is part of
# the configuration, as is the hash of every .py under sparx_agency/. So the supervisor snapshots the live
# Ollama, detector, segmenter (when the run has one) and gibson.run (argv, environment, cwd, from /proc) into <output>/replay/ and, when the
# launcher dies without a summary.json, relaunches *those* with --resume behind a wrapper that plays the
# launcher's part (launcher.pid, run.log, exit_code.txt) and restarts the monitor. Any .py edit under
# sparx_agency/ during the run still makes a resume impossible -- refused, by design. A run that died on
# the same episode twice is deterministic and is not relaunched again; exit 130 (Ctrl-C / TERM to the
# launcher) or a <output>/STOP file means the operator stopped it, and it stays stopped.
#
# Publish. In a temporary worktree on the remote's branch tip -- the checkout you work in is never
# touched -- it writes gibson/BENCHMARK_RESULTS.md (one section per run), gibson/results/<run>/
# {summary.json,episodes.csv,run.json}, closes the progress entry, adds the CHANGELOG line, commits as the
# repository's configured user, pushes HEAD:<branch>, then fast-forwards the local branch if it is checked
# out and clean. Nothing is ever force-pushed; a rejected or failed push is rebuilt on a fresh fetch and
# retried; if it never lands, the commit is kept on a local branch results/<run>.
#
# Env: HAB_PY (the Habitat interpreter), REMOTE (origin), BRANCH (the branch checked out when the supervisor
#      starts), MAX_RESUMES (2), POLL_S (60), RESUME_DELAY_S (120), PUSH_RETRIES (12), PUSH_RETRY_S (300),
#      MONITOR_INTERVAL_S (300), PROGRESS_ENTRY (docs/progress/entries/026-...md),
#      RESUME_ARGS (extra gibson.run arguments appended to every replayed resume, e.g.
#      "--resume-accepting-changes kinematics source_sha256" after a self-check fix; see BENCHMARK.md 2.6).
# Logs: <output>/supervisor.log.
set -euo pipefail

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
REPO="${OBJNAV_REPO:-$(cd -- "$HERE/../../../../.." && pwd)}"
HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/habitat/bin/python}"
GIBSON_REL="sparx_agency/tasks/planning/objnav_benchmark_runtime/gibson"
MONITOR_MODULE="sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.monitor"
REPORT_MODULE="sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.report"
REMOTE="${REMOTE:-origin}"
MAX_RESUMES="${MAX_RESUMES:-2}"
POLL_S="${POLL_S:-60}"
RESUME_DELAY_S="${RESUME_DELAY_S:-120}"
PUSH_RETRIES="${PUSH_RETRIES:-12}"
PUSH_RETRY_S="${PUSH_RETRY_S:-300}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-300}"
PROGRESS_ENTRY="${PROGRESS_ENTRY:-docs/progress/entries/026-full-gibson-val-run-detached-monitor.md}"
export RESUME_ARGS="${RESUME_ARGS:-}"       # exported: the relaunch wrapper's --replay run reads it

# -- the command line -----------------------------------------------------------------------------
MODE=supervise DRY_RUN="" OUT="" REPLAY_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --publish-only) MODE=publish ;;
    --render) MODE=render ;;
    --snapshot) MODE=snapshot ;;
    --no-resume) MAX_RESUMES=0 ;;
    --replay) MODE=replay; REPLAY_NAME="$2"; shift ;;      # internal: exec a snapshotted process
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *) OUT="$1" ;;
  esac
  shift
done
[[ -n "$OUT" ]] || { usage >&2; echo "ERROR: the run directory is required" >&2; exit 2; }
[[ -d "$OUT" ]] || { echo "ERROR: no run directory at $OUT" >&2; exit 2; }
OUT="$(cd -- "$OUT" && pwd)"
RUN_NAME="$(basename -- "$OUT")"
LOG="$OUT/supervisor.log"
REPLAY="$OUT/replay"

say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
die() { say "ERROR: $*" >&2; exit 1; }
nul_lines() { tr '\0' '\n' < "$1"; }
argv_value() { nul_lines "$REPLAY/$1.cmdline" | grep -A1 -x -- "$2" | tail -1; }   # the value after a flag
env_value() { nul_lines "$REPLAY/$1.environ" | sed -n "s/^$2=//p" | head -1; }

# -- internal: become a snapshotted process (argv + environment + cwd as captured; --resume [RESUME_ARGS] for the run) --
if [[ "$MODE" == replay ]]; then
  EXTRA=()
  if [[ "$REPLAY_NAME" == run ]]; then
    EXTRA=(--resume)
    # shellcheck disable=SC2206   # word-splitting RESUME_ARGS is the point: it is a command-line fragment
    [[ -n "$RESUME_ARGS" ]] && EXTRA+=($RESUME_ARGS)
  fi
  exec "$HAB_PY" - "$REPLAY" "$REPLAY_NAME" "${EXTRA[@]}" <<'PY'
import os, sys
replay, name, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
def nul(path):
    return [item.decode() for item in open(path, "rb").read().split(b"\0") if item]
argv = nul("%s/%s.cmdline" % (replay, name)) + extra
env = dict(item.split("=", 1) for item in nul("%s/%s.environ" % (replay, name)) if "=" in item)
os.chdir(open("%s/%s.cwd" % (replay, name)).read().strip())
os.execvpe(argv[0], argv, env)          # the saved PATH resolves a bare name such as "ollama"
PY
fi

[[ -x "$HAB_PY" ]] || die "Habitat interpreter not found at $HAB_PY (set HAB_PY)"

# -- the run's processes -----------------------------------------------------------------------------
launcher_pid() { cat "$OUT/launcher.pid" 2>/dev/null || true; }
alive() {   # up, and still this run's launcher: after a reboot the number in launcher.pid may belong to anything
  [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null && [[ -r "/proc/$1/cmdline" ]] \
    && tr '\0' ' ' 2>/dev/null < "/proc/$1/cmdline" | grep -qF -- "$RUN_NAME"
}
launcher_exit_code() {   # the current attempt's code: exit_code.txt newer than launcher.pid; else nothing
  if [[ -f "$OUT/exit_code.txt" && -f "$OUT/launcher.pid" && "$OUT/exit_code.txt" -nt "$OUT/launcher.pid" ]]; then
    cat "$OUT/exit_code.txt"
  fi
}
run_completed() { [[ -s "$OUT/summary.json" ]]; }
episode_in_progress() {   # what progress.json says was running (the episode a dead run died on)
  "$HAB_PY" - "$OUT/progress.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    current = json.load(open(sys.argv[1])).get("current") or {}
    print(current.get("episode_id") or "")
except Exception:
    print("")
PY
}

snap() {   # name pid -> replay/<name>.{cmdline,environ,cwd,pid}
  local name="$1" pid="$2"
  [[ -r "/proc/$pid/environ" ]] || die "cannot read /proc/$pid/environ for $name"
  rm -f "$REPLAY/$name".{cmdline,environ,cwd,pid}          # a plain cp of a /proc file keeps its read-only mode
  cp --no-preserve=mode "/proc/$pid/cmdline" "$REPLAY/$name.cmdline"
  cp --no-preserve=mode "/proc/$pid/environ" "$REPLAY/$name.environ"
  chmod 600 "$REPLAY/$name.environ"                          # the environment may hold keys
  readlink "/proc/$pid/cwd" > "$REPLAY/$name.cwd"
  printf '%s\n' "$pid" > "$REPLAY/$name.pid"
  say "snapshot $name: pid $pid, $(nul_lines "$REPLAY/$name.cmdline" | head -1), cwd $(cat "$REPLAY/$name.cwd")"
}

snapshot() {   # the live gibson.run, its detector and its Ollama, plus the GPU plan the configuration recorded
  mkdir -p "$REPLAY"
  local run_pid det_pid seg_pid oll_pid detector_url segmenter_url llm_url port p
  run_pid="$(pgrep -f -- "gibson\.run .*--output ${OUT}( |$)" | head -1 || true)"
  [[ -n "$run_pid" ]] || die "no gibson.run for $OUT is running; a snapshot needs the live run"
  snap run "$run_pid"
  detector_url="$(argv_value run --detector-url)"; port="${detector_url##*:}"
  det_pid="$(pgrep -f -- "detection_server .*--port ${port}( |$)" | head -1 || true)"
  if [[ -n "$det_pid" ]]; then snap detector "$det_pid"; else say "WARNING: no detector process on port $port (external?); it cannot be restarted by a resume"; fi
  segmenter_url="$(argv_value run --segmenter-url || true)"           # the Segment Anything service, when the run has one
  if [[ -n "$segmenter_url" ]]; then
    port="${segmenter_url##*:}"
    seg_pid="$(pgrep -f -- "segmentation_server .*--port ${port}( |$)" | head -1 || true)"
    if [[ -n "$seg_pid" ]]; then snap segmenter "$seg_pid"; else say "WARNING: no segmenter process on port $port (external?); it cannot be restarted by a resume"; fi
  fi
  llm_url="$(env_value run LLM_BASE_URL)"; oll_pid=""
  for p in $(pgrep -x ollama || true); do
    [[ -r "/proc/$p/environ" ]] || continue              # another user's Ollama (the system one) is not ours
    if tr '\0' '\n' < "/proc/$p/environ" | grep -qx -- "OLLAMA_HOST=${llm_url#http://}"; then oll_pid="$p"; break; fi
  done
  if [[ -n "$oll_pid" ]]; then snap ollama "$oll_pid"; else say "WARNING: no readable 'ollama serve' with OLLAMA_HOST=${llm_url#http://} (docker/external?); it cannot be restarted by a resume"; fi
  cp "$OUT/gpu_plan.json" "$REPLAY/gpu_plan.json"
  "$HAB_PY" - "$OUT/run.json" "$REPLAY/gpu_plan.json" <<'PY' | tee -a "$LOG"
import json, sys
recorded = json.load(open(sys.argv[1]))["config"]["gpu_plan"]
plan = json.load(open(sys.argv[2]))
same = {k: v for k, v in recorded.items() if k != "source"} == plan
print("snapshot gpu_plan.json %s the plan in run.json" % ("matches" if same else "DOES NOT MATCH -- a resume would be refused"))
PY
}

ensure_service() {   # name url timeout_s logfile: answering, or replayed from the snapshot
  local name="$1" url="$2" timeout="$3" logfile="$4" waited=0
  if curl -fsS --max-time 5 "$url" > /dev/null 2>&1; then say "$name answers at $url"; return 0; fi
  [[ -f "$REPLAY/$name.cmdline" ]] || die "$name is down at $url and there is no snapshot to restart it from"
  say "$name is down: restarting it from the snapshot"
  setsid nohup "$SELF" "$OUT" --replay "$name" >> "$logfile" 2>&1 < /dev/null &
  until curl -fsS --max-time 5 "$url" > /dev/null 2>&1; do
    sleep 2; waited=$((waited + 2))
    [[ $waited -lt $timeout ]] || die "$name did not answer at $url within ${timeout}s (see $logfile)"
  done
  say "$name is up"
}

relaunch() {   # the run again, --resume, from the snapshot, behind a wrapper that plays the launcher's part
  local old new waited=0
  old="$(launcher_pid)"
  [[ -f "$REPLAY/run.cmdline" ]] || die "no replay snapshot in $REPLAY (run with --snapshot while the run is alive)"
  pkill -TERM -f -- "gibson\.run .*--output ${OUT}( |$)" 2>/dev/null || true
  sleep 3
  pkill -KILL -f -- "gibson\.run .*--output ${OUT}( |$)" 2>/dev/null || true
  pkill -TERM -f -- "gibson\.monitor ${OUT}( |$)" 2>/dev/null || true
  cp "$REPLAY/gpu_plan.json" "$OUT/gpu_plan.json"            # the plan the frozen configuration recorded
  ensure_service ollama "$(env_value run LLM_BASE_URL)/api/tags" 180 "$OUT/ollama.log"
  ensure_service detector "$(argv_value run --detector-url)/health" 300 "$OUT/detector.log"
  if [[ -n "$(argv_value run --segmenter-url || true)" ]]; then
    ensure_service segmenter "$(argv_value run --segmenter-url)/health" 300 "$OUT/segmenter.log"
  fi
  say "relaunching gibson.run --resume${RESUME_ARGS:+ $RESUME_ARGS} (attempt $((ATTEMPT + 1)))"
  setsid nohup bash -c '
    printf "%s\n" "$$" > "$1/launcher.pid"
    "$2" "$1" --replay run
    code=$?
    date -u +%FT%TZ > "$1/finished_utc.txt"
    printf "%s\n" "$code" > "$1/exit_code.txt"
    exit "$code"' _ "$OUT" "$SELF" >> "$OUT/run.log" 2>&1 < /dev/null &
  while [[ "$(launcher_pid)" == "$old" ]]; do
    sleep 1; waited=$((waited + 1))
    [[ $waited -lt 30 ]] || die "the relaunched run did not write launcher.pid within 30s"
  done
  new="$(launcher_pid)"
  printf '%s\n' "$ATTEMPT" > "$OUT/resumes.txt"
  setsid nohup "$HAB_PY" -m "$MONITOR_MODULE" "$OUT" --interval "$MONITOR_INTERVAL_S" > /dev/null 2>&1 < /dev/null &
  say "launcher $new is running; monitor restarted (every ${MONITOR_INTERVAL_S}s)"
}

# -- the result ----------------------------------------------------------------------------------------
ensure_report() {   # comparison.md and audit.json beside summary.json, recomputed if the launcher died before writing them
  if [[ ! -s "$OUT/comparison.md" || ! -s "$OUT/audit.json" ]]; then
    say "recomputing comparison.md / audit.json"
    (cd "$REPO" && "$HAB_PY" -m "$REPORT_MODULE" "$OUT" > /dev/null) || die "gibson.report failed on $OUT"
  fi
}

write_results() {   # <tree|-> : write the result files into a worktree (or print the markdown for '-'); lists the paths
  local tree="$1" commit diff_bytes
  commit="$(cat "$OUT/source-commit.txt" 2>/dev/null || echo unknown)"
  diff_bytes="$(wc -c < "$OUT/source-changes.diff" 2>/dev/null || echo 0)"
  (cd "$REPO" && exec "$HAB_PY" - "$OUT" "$tree" "$GIBSON_REL" "$PROGRESS_ENTRY" "$commit" "$diff_bytes" "$(hostname)") <<'PY'
import csv, json, pathlib, shutil, sys, time
out, tree, gibson_rel, entry_rel, commit, diff_bytes, host = sys.argv[1:8]
out = pathlib.Path(out); run_name = out.name
sys.path.insert(0, ".")
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.monitor import write_episodes_csv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.progress import human_duration

def read_json(name, default=None):
    path = out / name
    return json.loads(path.read_text()) if path.is_file() else default
def read_text(name, default=""):
    path = out / name
    return path.read_text().strip() if path.is_file() else default

summary, run, audit, progress = read_json("summary.json"), read_json("run.json"), read_json("audit.json", {}), read_json("progress.json", {})
config = run["config"]; method = config["method"]; overall = summary["overall"]
f3 = lambda x: "%.3f" % x
f2 = lambda x: "n/a" if x is None else "%.2f" % x
ci = lambda c: "%s-%s" % (f3(c[0]), f3(c[1]))
sr, spl, dtg, soft = overall["success_rate"], overall["spl"], overall.get("distance_to_goal_m"), overall["soft_spl"]
n, selected = overall["n_episodes"], len(config["selected_episode_ids"])
full = bool(audit.get("full_split_completed"))
started = run.get("started_utc") or read_text("started_utc.txt")
finished = run.get("finished_utc") or read_text("finished_utc.txt") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
resumes = len(run.get("resumes") or [])
wall_total = float(audit.get("total_wall_s") or 0.0)
changed_resumes = [entry for entry in (run.get("resumes") or []) if entry.get("config_changes")]
def describe_change(key, change):
    recorded, resumed = change.get("recorded"), change.get("resumed")
    if key == "kinematics" and isinstance(recorded, dict) and isinstance(resumed, dict):
        fields = sorted(set(recorded) | set(resumed))
        return "`kinematics` " + ", ".join("%s %s -> %s" % (f, recorded.get(f), resumed.get(f)) for f in fields
                                           if recorded.get(f) != resumed.get(f))
    if key == "source_sha256":
        return "`source_sha256` %s... -> %s..." % (str(recorded)[:12], str(resumed)[:12])
    return "`%s` changed" % key
resume_note = ""
if changed_resumes:
    resume_note = "\n- **Resumed under a changed frozen configuration** (`gibson.run --resume-accepting-changes`, recorded in `run.json` " \
                  "under `resumes[].config_changes`): " + "; ".join(
        "after %d episodes at %s, %s" % (entry.get("episodes_done", 0), entry.get("utc"),
                                         "; ".join(describe_change(k, v) for k, v in sorted(entry["config_changes"].items())))
        for entry in changed_resumes) + \
        ". Each is provenance or a self-check that only refuses an episode -- never an input to a decision or a score; " \
        "the scorer and the protocol were the same before and after."
llm, det = method["llm"], method["detector"]["metadata"]
plan = config.get("gpu_plan") or {}
placements = ", ".join("%s=%s" % (p["component"], p["device"]) for p in plan.get("placements", ())) or "not recorded"
runtime = config.get("runtime", {})
headline = "SR %s, SPL %s, DTG %s m" % (f3(sr), f3(spl), f2(dtg))
scope = ("Full Gibson validation: %d of %d published episodes" % (n, selected) if full
         else "NOT a full benchmark result: %d of %d selected episodes (full_split_completed: %s)" % (n, selected, full))

def group_table(groups, label):
    lines = ["| %s | n | SR | SPL | SoftSPL | DTG (m) | steps |" % label, "|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append("| %s | %d | %s | %s | %s | %s | %.0f |" % (g["key"], g["n_episodes"], f3(g["success_rate"]), f3(g["spl"]),
                                                            f3(g["soft_spl"]), f2(g.get("distance_to_goal_m")), g["mean_steps"]))
    return "\n".join(lines)

comparison = read_text("comparison.md")
comparison = "\n".join(("##" + line) if line.startswith("#") else line for line in comparison.splitlines())
terminations = ", ".join("%s %d" % item for item in sorted((summary.get("terminations") or {}).items()))
section = """## `%(run)s` -- finished %(finished)s

**%(scope)s.** Protocol `%(protocol)s`: the published SemExp v1.1 `val` episodes, 500 actions, SR / SPL / DTG as in
arXiv:2508.04678 Table 2 (DTG = FMM distance to the 1 m success region; SoftSPL supplementary).

| | SR | SPL | DTG (m) | SoftSPL | mean steps | episodes |
|---|---:|---:|---:|---:|---:|---:|
| **this run** | **%(sr)s** (95%% CI %(sr_ci)s) | **%(spl)s** (95%% CI %(spl_ci)s) | **%(dtg)s** | %(soft)s | %(steps).0f | %(n)d |

Terminations: %(terminations)s. Agent errors: %(errors)d. Episodes with no reachable goal at the end: %(unreachable)d.

### Per scene

%(by_scene)s

### Per category

%(by_category)s

### Provenance

- Source commit `%(commit)s`%(dirty)s; source sha256 `%(sha)s` (every `.py` under `sparx_agency/`).
- Method `%(method)s`: local exploration `%(explorer)s`; room labels `%(llm_model)s`, node oracle `%(llm_reasoning)s`
  (%(llm_backend)s at `%(llm_url)s`, seed %(llm_seed)s, temperature %(llm_temp)s); detector `%(det_backend)s`.
- habitat-sim %(habitat)s (reference 0.1.5; `reference_sim_version_match: %(sim_match)s`, acknowledged), numpy %(numpy)s,
  scikit-fmm %(skfmm)s; seed %(seed)d; lean records %(lean)s; recording %(recording)s.
- GPU plan %(placements)s on %(gpu)s (%(host)s). Started %(started)s, finished %(finished)s; %(resumes)d resume(s);
  episode wall time in total %(wall)s (mean %(mean_wall)s per episode).%(resume_note)s
- Data: `results/%(run)s/summary.json` (every metric with its interval, per scene and per category), `episodes.csv`
  (the per-episode scalars: id, scene, target, success, SPL, SoftSPL, path length, shortest path, start and final
  distance to goal, steps, STOP, termination, wall time, agent error) and `run.json` (the frozen configuration, the
  dataset file hashes and the command line). The run directory was `runs/%(run)s` on %(host)s (gitignored).

### Beside the published rows (`comparison.md` of the run)

%(comparison)s
""" % dict(
    run=run_name, finished=finished, scope=scope, protocol=config["protocol"]["protocol_id"],
    sr=f3(sr), sr_ci=ci(overall["success_rate_ci"]), spl=f3(spl), spl_ci=ci(overall["spl_ci"]), dtg=f2(dtg), soft=f3(soft),
    steps=overall["mean_steps"], n=n, terminations=terminations or "none recorded", errors=int(summary.get("agent_errors", 0)),
    unreachable=int(overall.get("n_unreachable_end", 0)),
    by_scene=group_table(summary.get("by_scene", ()), "scene"), by_category=group_table(summary.get("by_category", ()), "category"),
    commit=commit[:12], dirty=" with uncommitted changes at launch (`source-changes.diff` in the run directory)" if int(diff_bytes) else " (clean tree)",
    sha=config["source_sha256"][:16] + "...", method=method["method"], explorer=method.get("local_exploration", "?"),
    llm_model=llm["model"], llm_reasoning=llm["reasoning_model"], llm_backend=llm.get("backend", "?"), llm_url=llm["base_url"],
    llm_seed=llm.get("seed", "?"), llm_temp=llm.get("temperature", "?"), det_backend=det.get("backend", "?"),
    habitat=runtime.get("habitat-sim", "?"), sim_match=str(config.get("reference_sim_version_match")).lower(), numpy=runtime.get("numpy", "?"),
    skfmm=runtime.get("scikit-fmm", "?"), seed=config["seed"], lean=str(config.get("lean_records")).lower(),
    recording="on" if (config.get("recording") or {}).get("enabled") else "off (no video, no frames)",
    placements=placements, gpu=plan.get("gpu_name", "?"), host=host, started=started, resumes=resumes,
    wall=human_duration(wall_total), mean_wall=human_duration(wall_total / n) if n else "n/a", comparison=comparison,
    resume_note=resume_note)

if tree == "-":
    print(section); sys.exit(0)

tree = pathlib.Path(tree)
written = []
# 1. the ledger: one section per run, newest at the bottom
ledger = tree / gibson_rel / "BENCHMARK_RESULTS.md"
if not ledger.is_file():
    ledger.write_text("# Gibson ObjectNav benchmark -- results\n\nOne section per completed run of "
                      "[`run_benchmark.sh`](BENCHMARK.md), written by `supervise_benchmark.sh` when the run ends; "
                      "the data behind each section is under `results/<run>/`. Numbers are not comparable to the "
                      "published rows beyond the caveats each section repeats.\n")
text = ledger.read_text()
marker = "## `%s`" % run_name
if marker in text:
    head, _, rest = text.partition(marker)
    tail = rest.split("\n## `", 1)
    text = head + section + ("\n## `" + tail[1] if len(tail) > 1 else "")
else:
    text = text.rstrip("\n") + "\n\n" + section
ledger.write_text(text); written.append(ledger)
# 2. the data
data = tree / gibson_rel / "results" / run_name
data.mkdir(parents=True, exist_ok=True)
shutil.copyfile(out / "summary.json", data / "summary.json"); written.append(data / "summary.json")
shutil.copyfile(out / "run.json", data / "run.json"); written.append(data / "run.json")
write_episodes_csv(out / "episodes.jsonl", data / "episodes.csv"); written.append(data / "episodes.csv")
# 3. the progress entry
entry = tree / entry_rel
if entry.is_file():
    result = ("Finished %s after %d resume(s): **%s** over %d/%d published episodes (full split: %s; agent errors %d). "
              "Per-scene and per-category tables, the published-row comparison and the provenance: "
              "`%s/BENCHMARK_RESULTS.md` (section `%s`); the data under `%s/results/%s/`. Published from the detached "
              "run by `supervise_benchmark.sh`." % (finished, resumes, headline, n, selected, str(full).lower(),
                                                   int(summary.get("agent_errors", 0)), gibson_rel, run_name, gibson_rel, run_name))
    body = entry.read_text()
    body = body.replace("**Status:** in-progress --", "**Status:** done --", 1)
    body = body.replace("- [ ] When the run ends:", "- [x] When the run ends:", 1)
    if "## Result\n(when the run ends)" in body:
        body = body.replace("## Result\n(when the run ends)", "## Result\n" + result, 1)
    else:
        body = body.rstrip("\n") + "\n\n## Result (%s)\n%s\n" % (finished[:10], result)
    entry.write_text(body); written.append(entry)
# 4. the changelog
changelog = tree / "CHANGELOG.md"
if changelog.is_file():
    line = ("- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/%s/`, %s): "
            "%s over %d of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200)%s, lean records, habitat-sim %s; per scene "
            "and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.\n"
            % (run_name, finished[:10], headline, n, "" if full else " -- NOT the full split", runtime.get("habitat-sim", "?")))
    body = changelog.read_text()
    anchor = "## [Unreleased]\n### Added\n"
    body = body.replace(anchor, anchor + line, 1) if anchor in body else body.replace("## [Unreleased]\n", "## [Unreleased]\n### Added\n" + line, 1)
    changelog.write_text(body); written.append(changelog)
# the commit message, and the list of paths for the shell
subject = "Record the Gibson validation result: %s" % headline if full else "Record a partial Gibson validation run: %s" % headline
message = subject + "\n\n" + "\n".join([
    "- %s/BENCHMARK_RESULTS.md: the %d-episode SemExp v1.1 val run %s -- SR and SPL with 95%% intervals, DTG, per scene and per "
    "category, beside the published Table 2 rows, with the frozen configuration and provenance." % (gibson_rel, n, run_name),
    "- %s/results/%s/: summary.json, the per-episode scalars as episodes.csv, and run.json, so every table can be recomputed." % (gibson_rel, run_name),
    "- %s: the progress entry closed with the result." % entry_rel,
    "- CHANGELOG.md: the result under Unreleased."]) + "\n\nProgress: 026-full-gibson-val-run-detached-monitor\n"
(out / "publish.commit-msg.txt").write_text(message)
for path in written:
    print(path.relative_to(tree))
PY
}

worktree_cleanup() { if [[ -n "${WORKTREE:-}" && -d "$WORKTREE" ]]; then git -C "$REPO" worktree remove --force "$WORKTREE" > /dev/null 2>&1 || rm -rf "$WORKTREE"; git -C "$REPO" worktree prune > /dev/null 2>&1 || true; fi; WORKTREE=""; }

build_commit() {   # a fresh worktree on the remote's branch tip (or the local branch when the fetch fails), the files, the commit
  local base paths
  if git -C "$REPO" fetch --quiet "$REMOTE" "$BRANCH" 2>>"$LOG"; then base=FETCH_HEAD; else say "fetch from $REMOTE failed; basing on the local $BRANCH"; base="$BRANCH"; fi
  WORKTREE="$(mktemp -d /tmp/gibson-publish.XXXXXX)"; rmdir "$WORKTREE"
  git -C "$REPO" worktree add --detach "$WORKTREE" "$base" >>"$LOG" 2>&1 || die "could not create a worktree on $base"
  write_results "$WORKTREE" > "$OUT/publish.paths.txt" || die "writing the result files failed"
  mapfile -t paths < "$OUT/publish.paths.txt"
  [[ ${#paths[@]} -gt 0 && -s "$OUT/publish.commit-msg.txt" ]] || die "write_results produced no files or no message"
  git -C "$WORKTREE" add -- "${paths[@]}"
  git -C "$WORKTREE" commit --quiet -F "$OUT/publish.commit-msg.txt" || die "git commit failed in $WORKTREE"
  COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
  say "built $(git -C "$WORKTREE" rev-parse --short HEAD) on $base ($(git -C "$REPO" rev-parse --short "$base")): $(head -1 "$OUT/publish.commit-msg.txt")"
}

fast_forward_local() {
  local head
  head="$(git -C "$REPO" symbolic-ref --quiet --short HEAD || echo detached)"
  if [[ "$head" != "$BRANCH" ]]; then say "the checkout is on '$head', not $BRANCH: not fast-forwarding it (git pull when you are back on $BRANCH)"; return 0; fi
  if git -C "$REPO" fetch --quiet "$REMOTE" "$BRANCH" 2>>"$LOG" && git -C "$REPO" merge --ff-only --quiet FETCH_HEAD 2>>"$LOG"; then
    say "local $BRANCH fast-forwarded to $(git -C "$REPO" rev-parse --short HEAD)"
  else
    say "could not fast-forward the local $BRANCH (local commits, or uncommitted changes in the files the commit touches): git pull"
  fi
}

publish() {
  local attempt
  ensure_report
  for attempt in $(seq 1 "$PUSH_RETRIES"); do
    build_commit
    if [[ "$DRY_RUN" == 1 ]]; then
      say "dry run: nothing pushed; the commit would be"
      git -C "$WORKTREE" --no-pager show --stat --format='%h %s%n%n%b' HEAD | tee -a "$LOG"
      worktree_cleanup; return 0
    fi
    if git -C "$WORKTREE" push --quiet "$REMOTE" "HEAD:refs/heads/$BRANCH" 2>>"$LOG"; then
      say "pushed $(git -C "$WORKTREE" rev-parse --short HEAD) to $REMOTE $BRANCH"
      printf '%s\n' "$COMMIT" > "$OUT/published_commit.txt"
      worktree_cleanup; fast_forward_local; return 0
    fi
    say "push to $REMOTE $BRANCH failed (attempt $attempt of $PUSH_RETRIES); rebuilding on a fresh fetch in ${PUSH_RETRY_S}s"
    worktree_cleanup; sleep "$PUSH_RETRY_S"
  done
  build_commit
  git -C "$REPO" branch -f "results/$RUN_NAME" "$COMMIT"
  worktree_cleanup
  die "the push never landed; the commit is on the local branch results/$RUN_NAME -- push it with: git push $REMOTE results/$RUN_NAME:$BRANCH"
}

# -- modes ---------------------------------------------------------------------------------------------
BRANCH="${BRANCH:-$(git -C "$REPO" symbolic-ref --quiet --short HEAD || true)}"
WORKTREE=""
trap worktree_cleanup EXIT
case "$MODE" in
  render) ensure_report; write_results -; exit 0 ;;
  snapshot) snapshot; exit 0 ;;
esac
[[ -n "$BRANCH" ]] || die "the checkout is detached; set BRANCH to the branch the result should be pushed to"
if [[ "$MODE" == publish ]]; then
  run_completed || die "no summary.json in $OUT: the run has not completed"
  say "publish only: $RUN_NAME -> $REMOTE $BRANCH${DRY_RUN:+ (dry run)}"
  publish; exit 0
fi

printf '%s\n' "$$" > "$OUT/supervisor.pid"
ATTEMPT=0 LAST_DIED_ON=""
say "supervising $RUN_NAME (pid $$): resumes <= $MAX_RESUMES, publish to $REMOTE $BRANCH${DRY_RUN:+ (dry run)}, poll ${POLL_S}s"
if [[ ! -f "$REPLAY/run.cmdline" && "$MAX_RESUMES" -gt 0 ]] && alive "$(launcher_pid)"; then
  ( snapshot ) || say "WARNING: no replay snapshot could be taken; a dead run will not be resumable"
fi
while :; do
  pid="$(launcher_pid)"; [[ -n "$pid" ]] || die "no launcher.pid in $OUT (launch as BENCHMARK.md 2.3 describes)"
  say "attempt $((ATTEMPT + 1)): waiting for launcher $pid"
  while alive "$pid"; do sleep "$POLL_S"; done
  code="$(launcher_exit_code)"
  say "launcher $pid ended (exit code ${code:-not recorded})"
  if run_completed; then
    say "the run completed: summary.json is there${code:+ (launcher exit $code)}"
    publish; exit 0
  fi
  died_on="$(episode_in_progress)"
  stop_reason=""
  [[ "$code" == 130 || "$code" == 143 ]] && stop_reason="exit $code"
  [[ -e "$OUT/STOP" ]] && stop_reason="${stop_reason:+$stop_reason, }STOP file"
  if [[ -n "$stop_reason" ]]; then say "stopped by the operator ($stop_reason); not resuming"; exit 2; fi
  if [[ -n "$died_on" && "$died_on" == "$LAST_DIED_ON" ]]; then say "died on $died_on twice: deterministic, not resuming -- see $OUT/run.log"; exit 1; fi
  if [[ "$ATTEMPT" -ge "$MAX_RESUMES" ]]; then say "no resumes left (MAX_RESUMES=$MAX_RESUMES); giving up -- see $OUT/run.log"; exit 1; fi
  LAST_DIED_ON="$died_on"; ATTEMPT=$((ATTEMPT + 1))
  say "the run died${died_on:+ on $died_on} without a summary; resuming in ${RESUME_DELAY_S}s (resume $ATTEMPT of $MAX_RESUMES)"
  sleep "$RESUME_DELAY_S"
  relaunch
done







