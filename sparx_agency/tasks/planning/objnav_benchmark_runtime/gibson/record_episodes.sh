#!/usr/bin/env bash
# Re-run named episodes of a finished (or finishing) Gibson run with --record, under that run's own frozen
# settings, so a failure can be watched: one MP4 per slot, the final frame, and for a wrong STOP the frames
# that led to it.
#
#   gibson/record_episodes.sh <run-dir> <output-dir> [options] <slot>...
#
#   <slot>   label=Scene/NNNNNN[,Scene/NNNNNN...]   one video per slot. The first candidate whose re-run
#            reproduces the original outcome (--expect) is kept; otherwise the next is tried. A re-run is
#            seeded like the original, but a GPU LLM is not bit-exact, so a failure may not come back.
#   --after-run       wait until <run-dir>'s own gibson.run has ended (no such process for three polls, and
#                     summary.json there or its supervisor gone): the recordings then have the card and the
#                     services to themselves instead of slowing a benchmark that is still scoring.
#   --expect same|any same (default): the re-run must match the original's success flag -- and, for a
#                     failure, its termination; for a success, keep SPL <= 0.5, the slot being about a
#                     circuitous path. any: keep the first re-run whatever it did.
#   --preflight       gibson.run --preflight for every slot's first candidate (data, services, ffmpeg);
#                     record nothing.
#   --poll S          seconds between --after-run polls (60).
#
# Each candidate is one gibson.run job: the recorded run's argv (run.json) with its selection, --lean and
# --resume removed and --scene S --shards 200 --shard-index k --limit 1 --record added, under the environment
# the supervisor snapshotted (<run-dir>/replay/run.environ; the current environment otherwise), into
# <output-dir>/jobs/<Scene>_<k>/ -- full records, not lean, so steps.jsonl carries every decision, detection
# and LLM reading. The detector and the Ollama are reused when they answer and replayed from the snapshot
# (supervise_benchmark.sh --replay) when they do not. A job directory that already holds its episode is
# reused, so the command can be re-issued after an interruption.
#
# Output: <output-dir>/videos/<NN>_<label>_<Scene>_<k>.mp4 and _final.jpg; for a wrong STOP also
# _verify.jpg / _inspect.jpg / _stop.jpg (the frames at the first VERIFY, the first INSPECT and the STOP
# decision of the closing); index.md and index.json with the original beside the re-run for every attempt.
# Env: HAB_PY (default: the snapshotted interpreter, else ~/miniconda3/envs/habitat/bin/python).
# Log: <output-dir>/record.log.
set -euo pipefail

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OBJNAV_REPO:-$(cd -- "$HERE/../../../../.." && pwd)}"
[[ $# -ge 2 && "$1" != -h && "$1" != --help ]] || { usage; exit 2; }
RUN="$1"
[[ -d "$RUN" && -f "$RUN/run.json" ]] || { echo "ERROR: $RUN is not a run directory (no run.json)" >&2; exit 2; }
HAB_PY="${HAB_PY:-}"
if [[ -z "$HAB_PY" && -f "$RUN/replay/run.cmdline" ]]; then HAB_PY="$(tr '\0' '\n' < "$RUN/replay/run.cmdline" | head -1)"; fi
HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/habitat/bin/python}"
[[ -x "$HAB_PY" ]] || { echo "ERROR: Habitat interpreter not found at $HAB_PY (set HAB_PY)" >&2; exit 2; }

exec "$HAB_PY" - "$REPO" "$HERE" "$HAB_PY" "$@" <<'PY'
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

MODULE = "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run"
STRIP_FLAGS = {"--lean", "--resume", "--record", "--preflight", "--print-vocabulary"}
STRIP_VALUED = {"--output", "--limit", "--scene", "--shards", "--shard-index", "--progress-file", "--video-fps"}
STRIP_MANY = {"--resume-accepting-changes"}
CLOSING_FRAMES = (("verify", "VERIFY"), ("inspect", "INSPECT"), ("stop", "STOP"))

repo, here, hab_py = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
parser = argparse.ArgumentParser(prog="record_episodes.sh", add_help=False)
parser.add_argument("run_dir", type=pathlib.Path)
parser.add_argument("output_dir", type=pathlib.Path)
parser.add_argument("slots", nargs="*")
parser.add_argument("--after-run", action="store_true")
parser.add_argument("--expect", choices=("same", "any"), default="same")
parser.add_argument("--preflight", action="store_true")
parser.add_argument("--poll", type=float, default=60.0)
args = parser.parse_intermixed_args(sys.argv[4:])      # options may sit between the directories and the slots
run_dir, out = args.run_dir.resolve(), args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
LOG = out / "record.log"


def utc(ts=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def log(message):
    line = "[%s] %s" % (utc(), message)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def nul(path):
    return [item.decode() for item in pathlib.Path(path).read_bytes().split(b"\0") if item]


def read_rows(path):
    rows = {}
    if not pathlib.Path(path).is_file():
        return rows
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[row["episode_id"]] = row
    return rows


# -- the run being inspected -----------------------------------------------------------------------
info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
originals = read_rows(run_dir / "episodes.jsonl")
replay = run_dir / "replay"


def base_argv():
    """The recorded command line without its selection, resume and record switches."""
    argv, kept, skip, many = list(info["argv"])[1:], [], 0, False
    for token in argv:
        if skip:
            skip -= 1
            continue
        if many:
            if token.startswith("--"):
                many = False
            else:
                continue
        if token in STRIP_FLAGS:
            continue
        if token in STRIP_VALUED:
            skip = 1
            continue
        if token in STRIP_MANY:
            many = True
            continue
        kept.append(token)
    if "--allow-shared-gpu" not in kept:
        kept.append("--allow-shared-gpu")          # the detector and the Ollama hold the card already
    return kept


BASE = base_argv()
ENV = dict(item.split("=", 1) for item in nul(replay / "run.environ") if "=" in item) if (replay / "run.environ").is_file() else dict(os.environ)
CWD = (replay / "run.cwd").read_text().strip() if (replay / "run.cwd").is_file() else str(repo)
PYTHON = nul(replay / "run.cmdline")[0] if (replay / "run.cmdline").is_file() else hab_py


def option(flag, default=None):
    return BASE[BASE.index(flag) + 1] if flag in BASE else default


DETECTOR_URL = option("--detector-url", "http://127.0.0.1:8092")
LLM_URL = ENV.get("LLM_BASE_URL", "http://127.0.0.1:11434")

# -- the slots -------------------------------------------------------------------------------------------
slots = []
for spec in args.slots:
    if "=" not in spec:
        sys.exit("slot %r must be label=Scene/NNNNNN[,Scene/NNNNNN...]" % spec)
    label, candidates = spec.split("=", 1)
    candidates = [c.strip() for c in candidates.split(",") if c.strip()]
    for candidate in candidates:
        if candidate not in originals:
            sys.exit("%s is not an episode of %s (%d episodes on disk)" % (candidate, run_dir, len(originals)))
        scene, index = candidate.split("/")
        if not index.isdigit() or not 0 <= int(index) < 200:
            sys.exit("%s: the index must be 000000..000199" % candidate)
    slots.append((label.replace(" ", "-"), candidates))
if not slots:
    sys.exit("no slots given")


def job_argv(candidate, job_dir, preflight=False):
    scene, index = candidate.split("/")
    argv = [PYTHON, "-u", "-m", MODULE] + BASE + ["--scene", scene, "--shards", "200", "--shard-index", str(int(index)),
                                                   "--limit", "1", "--record", "--output", str(job_dir)]
    return argv + ["--preflight"] if preflight else argv


# -- services --------------------------------------------------------------------------------------------
def answers(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def ensure_services():
    checks = (("detector", DETECTOR_URL.rstrip("/") + "/health"), ("ollama", LLM_URL.rstrip("/") + "/api/tags"))
    for name, url in checks:
        if answers(url):
            continue
        if not (replay / ("%s.cmdline" % name)).is_file():
            sys.exit("%s does not answer at %s and there is no snapshot in %s to restart it from" % (name, url, replay))
        log("%s is down at %s: replaying it from the snapshot" % (name, url))
        with (out / ("%s.log" % name)).open("a") as stream:
            subprocess.Popen([str(here / "supervise_benchmark.sh"), str(run_dir), "--replay", name],
                             env=dict(os.environ, HAB_PY=hab_py), start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        waited = 0
        while not answers(url):
            time.sleep(2)
            waited += 2
            if waited >= 300:
                sys.exit("%s did not answer at %s within 300 s (see %s)" % (name, url, out / ("%s.log" % name)))
        log("%s is up" % name)


# -- waiting for the run --------------------------------------------------------------------------------
def run_alive():
    pattern = r"gibson\.run .*--output %s( |$)" % str(run_dir).replace(".", r"\.")
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def supervisor_alive():
    try:
        pid = int((run_dir / "supervisor.pid").read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def wait_for_run():
    absent = 0
    announced = False
    while True:
        if run_alive():
            absent = 0
        else:
            absent += 1
        if absent >= 3 and ((run_dir / "summary.json").is_file() or not supervisor_alive()):
            log("%s has ended (%s); starting the recordings" % (
                run_dir.name, "summary.json is there" if (run_dir / "summary.json").is_file() else "no summary, supervisor gone"))
            return
        if not announced:
            log("waiting for %s to end (polling every %.0f s)" % (run_dir.name, args.poll))
            announced = True
        time.sleep(args.poll)


# -- one recording job -------------------------------------------------------------------------------------
def key_of(candidate):
    return hashlib.sha256(candidate.encode()).hexdigest()[:12]


def record(candidate):
    scene, index = candidate.split("/")
    job_dir = out / "jobs" / ("%s_%s" % (scene, index))
    row = read_rows(job_dir / "episodes.jsonl").get(candidate)
    recording = job_dir / "recordings" / key_of(candidate)
    if row is not None and (recording / "video.mp4").is_file():
        log("%s: already recorded in %s (reused)" % (candidate, job_dir))
        return 0, row, recording
    if job_dir.exists():
        shutil.rmtree(job_dir)                      # a half-finished job: start it clean
    job_dir.mkdir(parents=True)
    ensure_services()
    started = time.time()
    log("%s: recording into %s" % (candidate, job_dir))
    with (job_dir / "job.log").open("w") as stream:
        code = subprocess.call(job_argv(candidate, job_dir), env=ENV, cwd=CWD, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT)
    row = read_rows(job_dir / "episodes.jsonl").get(candidate)
    log("%s: exit %d after %s%s" % (candidate, code, human(time.time() - started),
                                    "" if row else " -- NO EPISODE ROW (see %s)" % (job_dir / "job.log")))
    return code, row, recording


def human(seconds):
    seconds = int(round(seconds))
    return "%dm%02ds" % divmod(seconds, 60) if seconds >= 60 else "%ds" % seconds


def reproduced(original, row):
    if row is None:
        return False
    if args.expect == "any":
        return True
    if bool(row["success"]) != bool(original["success"]):
        return False
    if not original["success"]:
        return row["termination"] == original["termination"]
    return float(row["spl"]) <= 0.5


# -- frames ------------------------------------------------------------------------------------------------
def closing_frames(recording):
    """Frame indices of the first VERIFY, the first INSPECT and the STOP decision (one video frame per decision)."""
    found = {}
    steps = recording / "steps.jsonl"
    if not steps.is_file():
        return found
    for line in steps.read_text(encoding="utf-8").splitlines():
        try:
            step = json.loads(line)
        except ValueError:
            continue
        phase = ((step.get("method") or {}).get("target_closing") or {}).get("phase")
        for name, wanted in CLOSING_FRAMES:
            if phase == wanted and name not in found:
                found[name] = int(step["step"])
    return found


def extract_frame(video, index, dest):
    if not shutil.which("ffmpeg"):
        return False
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", "select=eq(n\\,%d)" % index,
                             "-vsync", "vfr", "-frames:v", "1", str(dest)], capture_output=True)
    return result.returncode == 0 and dest.is_file()


def publish(number, label, candidate, original, row, recording):
    videos = out / "videos"
    videos.mkdir(exist_ok=True)
    scene, index = candidate.split("/")
    stem = "%02d_%s_%s_%s" % (number, label, scene, index)
    files = {}
    for name, source in (("video", recording / "video.mp4"), ("final", recording / "final.jpg")):
        if source.is_file():
            dest = videos / (stem + (".mp4" if name == "video" else "_final.jpg"))
            shutil.copyfile(source, dest)
            files[name] = str(dest.relative_to(out))
    if row is not None and not row["success"] and row["termination"] == "stop" and "video" in files:
        for name, frame in closing_frames(recording).items():
            dest = videos / ("%s_%s.jpg" % (stem, name))
            if extract_frame(recording / "video.mp4", frame, dest):
                files[name] = "%s (decision %d)" % (dest.relative_to(out), frame)
    return files


# -- the index ---------------------------------------------------------------------------------------------
def metrics(row):
    if row is None:
        return "no episode row (the job failed)"
    dtg = row.get("distance_to_goal_m")
    return "SR=%d %s steps=%d SPL=%.3f start DTG=%.2f final DTG=%s" % (
        int(bool(row["success"])), row["termination"], int(row["steps"]), float(row["spl"]),
        float(row["start_distance_to_goal_m"]), "n/a" if dtg is None else "%.2f m" % dtg)


def write_index(results, finished):
    lines = ["# Recorded episodes of `%s`" % run_dir.name, "",
             "Re-runs of the published episodes below with `--record`, under the run's frozen settings (`run.json` argv, "
             "the snapshotted environment), %s. A re-run is seeded like the original but a GPU LLM is not bit-exact, so "
             "each attempt is listed with whether it reproduced the original outcome (`--expect %s`). Videos: one frame per "
             "decision at 6 fps -- decision k is at k/6 s; `_final.jpg` is the last frame with the FINAL banner; for a wrong "
             "STOP, `_verify/_inspect/_stop.jpg` are the frames at the closing's first VERIFY, first INSPECT and STOP "
             "decisions. Each job directory under `jobs/` holds the full records: `recordings/<key>/steps.jsonl` (every "
             "decision with the detections, the planned path and the LLM's readings), `trajectory.csv`, `index.html`."
             % ("finished %s" % utc() if finished else "IN PROGRESS", args.expect), ""]
    lines += ["| # | slot | episode | attempt | original | re-run | reproduced | files |", "|---|---|---|---|---|---|---|---|"]
    for item in results:
        lines.append("| %d | %s | %s | %d | %s | %s | %s | %s |" % (
            item["slot_number"], item["label"], item["episode"], item["attempt"], metrics(item["original"]),
            metrics(item["rerun"]), "yes" if item["reproduced"] else ("no" if item["rerun"] is not None else "job failed"),
            ", ".join("%s: `%s`" % (k, v) for k, v in sorted(item["files"].items())) or "-"))
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "index.json").write_text(json.dumps({"run": str(run_dir), "finished": finished, "expect": args.expect,
                                                 "results": results}, indent=1), encoding="utf-8")


# -- main ------------------------------------------------------------------------------------------------------
if args.preflight:
    bad = 0
    for label, candidates in slots:
        candidate = candidates[0]
        job_dir = out / "preflight" / candidate.replace("/", "_")
        job_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(job_argv(candidate, job_dir, preflight=True), env=ENV, cwd=CWD, capture_output=True, text=True)
        try:
            verdict = json.loads(proc.stdout)
        except ValueError:
            verdict = {"ready": False, "issues": ["no JSON from gibson.run: %s" % (proc.stderr.strip()[-300:] or proc.stdout[-300:])]}
        selected = (verdict.get("configuration") or {}).get("selected_episode_ids")
        ok = verdict.get("ready") and selected == [candidate]
        bad += 0 if ok else 1
        print("%-28s %-22s %s" % (label, candidate, "ready, selects exactly %s, recording %s" % (
            candidate, (verdict["configuration"]["recording"])) if ok else "NOT READY: %s" % (verdict.get("issues") or "selected %s" % selected)))
    shutil.rmtree(out / "preflight", ignore_errors=True)
    sys.exit(1 if bad else 0)

log("recording %d slot(s) of %s into %s (expect %s)" % (len(slots), run_dir.name, out, args.expect))
if args.after_run:
    wait_for_run()
results = []
for number, (label, candidates) in enumerate(slots, 1):
    kept = False
    for attempt, candidate in enumerate(candidates, 1):
        original = originals[candidate]
        code, row, recording = record(candidate)
        ok = reproduced(original, row)
        files = publish(number, label, candidate, original, row, recording) if row is not None else {}
        results.append({"slot_number": number, "label": label, "episode": candidate, "attempt": attempt, "exit_code": code,
                        "original": original and {k: original.get(k) for k in ("success", "termination", "steps", "spl", "soft_spl",
                                                                               "start_distance_to_goal_m", "distance_to_goal_m", "path_length_m")},
                        "rerun": row and {k: row.get(k) for k in ("success", "termination", "steps", "spl", "soft_spl",
                                                                   "start_distance_to_goal_m", "distance_to_goal_m", "path_length_m", "wall_s")},
                        "reproduced": ok, "files": files, "job_dir": str((out / "jobs" / candidate.replace("/", "_")).relative_to(out))})
        write_index(results, finished=False)
        log("slot %d %s: %s %s -> %s" % (number, label, candidate, "REPRODUCED" if ok else "diverged", metrics(row)))
        if ok:
            kept = True
            break
    if not kept:
        log("slot %d %s: no candidate reproduced the original outcome; the attempts are kept under videos/ anyway" % (number, label))
write_index(results, finished=True)
videos = sorted((out / "videos").glob("*.mp4")) if (out / "videos").is_dir() else []
log("done: %d video(s) under %s; index: %s" % (len(videos), out / "videos", out / "index.md"))
PY


