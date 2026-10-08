"""Watch a detached Gibson run: a status line per tick, a flat CSV of the scalars, the summary when it ends.

``run_benchmark.sh --status <dir>`` prints the progress once and
``progress.json`` is rewritten after every episode, but a run launched with
``nohup`` has nobody reading either. This module does, on a schedule, and is
meant to be launched detached beside the run::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.monitor <output> --interval 300

Every tick it

* appends one line to ``<output>/monitor.log`` -- UTC time, done/total,
  running SR / SPL / DTG, ETA, the episode in progress and the last outcome;
* regenerates ``<output>/episodes.csv`` whenever ``episodes.jsonl`` has grown:
  one row per finished episode holding only the scalar fields of the record
  (:data:`CSV_COLUMNS` -- id, scene, target, success, SPL, SoftSPL, path
  length, shortest path, start and final distance to goal, steps, STOP,
  termination, wall time, agent error) and none of the diagnostics;
* notices the run has ended -- the launcher's process is gone (``--pid``, or
  ``<output>/launcher.pid``), or, without a PID, the launcher's
  ``exit_code.txt`` appeared after the monitor started -- then writes the
  finished summary (:func:`progress.describe_summary`: ``summary.json``, or
  the progress file when the run died before writing one) to the log, to
  ``<output>/FINAL_SUMMARY.txt`` and to stdout, and exits with the
  launcher's exit code.

Runs under the Habitat interpreter like ``--status`` does; Python 3.8 syntax.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.results_io import write_atomically
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.progress import (
    UTC, describe_summary, human_duration)

#: The scalar fields of an ``episodes.jsonl`` row, in CSV column order. Nothing nested.
CSV_COLUMNS = ("episode_id", "scene_id", "target_category", "success", "spl", "soft_spl",
               "path_length_m", "shortest_path_m", "start_distance_to_goal_m", "distance_to_goal_m",
               "steps", "stop_called", "termination", "wall_s", "agent_error")
LOG_FILE = "monitor.log"
CSV_FILE = "episodes.csv"
SUMMARY_FILE = "FINAL_SUMMARY.txt"
PID_FILE = "launcher.pid"
EXIT_CODE_FILE = "exit_code.txt"
#: The exit code reported when the launcher vanished without writing ``exit_code.txt`` (killed).
EXIT_CODE_UNKNOWN = -1


def _utc(ts: Optional[float] = None) -> str:
    return time.strftime(UTC, time.gmtime(time.time() if ts is None else ts))


# -- the CSV ------------------------------------------------------------------------
def _cell(value: Any) -> Any:
    """A CSV cell: booleans as 0/1, None as empty, everything else as is."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def read_episode_scalars(episodes_jsonl: pathlib.Path) -> List[Dict[str, Any]]:
    """The :data:`CSV_COLUMNS` of every complete row of ``episodes.jsonl``.

    A run killed mid-write leaves a truncated last line; it is skipped here
    (the harness itself rejects it on resume). A missing file is no rows.
    """
    if not episodes_jsonl.is_file():
        return []
    rows = []
    with episodes_jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            rows.append({key: record.get(key) for key in CSV_COLUMNS})
    return rows


def episodes_csv_text(rows: Sequence[Dict[str, Any]]) -> str:
    """The rows as CSV text with a :data:`CSV_COLUMNS` header, no trailing newline."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(row.get(key)) for key in CSV_COLUMNS})
    return buffer.getvalue().rstrip("\n")


def write_episodes_csv(episodes_jsonl: pathlib.Path, csv_path: pathlib.Path) -> int:
    """Rewrite ``csv_path`` from ``episodes_jsonl`` atomically; returns the number of rows."""
    rows = read_episode_scalars(episodes_jsonl)
    write_atomically(csv_path, episodes_csv_text(rows))
    return len(rows)


# -- the status line ----------------------------------------------------------------
def status_line(snapshot: Optional[Dict[str, Any]], now: Optional[float] = None) -> str:
    """One log line from a ``progress.json`` snapshot (or none yet)."""
    stamp = _utc(now)
    if snapshot is None:
        return "%s | no progress.json yet (services starting, or the first episode has not finished)" % stamp
    overall = snapshot.get("overall") or {}
    dtg = overall.get("DTG_m")
    text = "%s | %d/%d %5.1f%% | SR %.3f SPL %.3f DTG %s | ETA %s" % (
        stamp, snapshot.get("done", 0), snapshot.get("total", 0), snapshot.get("percent", 0.0),
        overall.get("SR", 0.0), overall.get("SPL", 0.0), "n/a" if dtg is None else "%.2f" % dtg,
        snapshot.get("eta_human") or "n/a")
    current = snapshot.get("current")
    if current:
        text += " | running %s since %s" % (current.get("episode_id"), current.get("started_utc"))
    last = snapshot.get("last")
    if last:
        text += " | last %s %s SR=%d SPL=%.3f DTG=%s steps=%d (%ds)" % (
            last.get("episode_id"), last.get("target"), int(last.get("SR", 0)), float(last.get("SPL", 0.0)),
            "n/a" if last.get("DTG_m") is None else "%.2f" % last["DTG_m"], int(last.get("steps", 0)),
            int(last.get("wall_s", 0)))
        if last.get("agent_error"):
            text += " AGENT ERROR"
    if snapshot.get("finished_utc"):
        text += " | FINISHED %s" % snapshot["finished_utc"]
    return text


def read_snapshot(output: pathlib.Path) -> Optional[Dict[str, Any]]:
    """``progress.json`` parsed, or None when it is not there (or is being replaced this instant)."""
    path = output / "progress.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# -- has the run ended? ---------------------------------------------------------------
def pid_alive(pid: int) -> bool:
    """Whether ``pid`` names a live, non-zombie process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open("/proc/%d/status" % pid, "r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("State:"):
                    return not line.split()[1].startswith("Z")
    except OSError:
        pass
    return True


def read_pid(output: pathlib.Path) -> Optional[int]:
    """The launcher's PID from ``launcher.pid``, or None."""
    try:
        return int((output / PID_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def launcher_started(output: pathlib.Path) -> Optional[float]:
    """When the launcher wrote ``launcher.pid`` -- the run's start, after any preflight into the directory -- or None."""
    try:
        return (output / PID_FILE).stat().st_mtime
    except OSError:
        return None


def exit_code(output: pathlib.Path, since: float) -> Optional[int]:
    """The launcher's recorded exit code, if ``exit_code.txt`` was written after ``since``."""
    path = output / EXIT_CODE_FILE
    try:
        if path.stat().st_mtime < since:
            return None
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def run_finished(output: pathlib.Path, pid: Optional[int], since: float) -> Optional[int]:
    """The exit code once the run has ended, else None.

    With a PID the process going away is the signal and ``exit_code.txt`` only
    supplies the code (:data:`EXIT_CODE_UNKNOWN` when it never appeared: the
    launcher was killed before its trap ran). Without one, the code file
    appearing after ``since`` is the signal. Either way a code file older than
    ``since`` is ignored -- a preflight into the same directory leaves one
    behind -- which is why ``since`` should be the launcher's start
    (:func:`launcher_started`) rather than the monitor's.
    """
    code = exit_code(output, since)
    if pid is not None:
        if pid_alive(pid):
            return None
        return EXIT_CODE_UNKNOWN if code is None else code
    return code


# -- the loop -------------------------------------------------------------------------
class Monitor:
    """One run directory watched on a schedule; see the module docstring."""

    def __init__(self, output: pathlib.Path, pid: Optional[int] = None, log: Optional[pathlib.Path] = None,
                 csv_path: Optional[pathlib.Path] = None, since: Optional[float] = None) -> None:
        self.output = pathlib.Path(output)
        self.pid = pid
        self.log = log or self.output / LOG_FILE
        self.csv_path = csv_path or self.output / CSV_FILE
        self.since = time.time() if since is None else since
        self._csv_signature = None  # type: Optional[tuple]

    def append(self, text: str) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def refresh_csv(self) -> Optional[int]:
        """Rewrite the CSV when ``episodes.jsonl`` changed; the row count when it did, else None."""
        episodes = self.output / "episodes.jsonl"
        try:
            stat = episodes.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            return None
        if signature == self._csv_signature:
            return None
        rows = write_episodes_csv(episodes, self.csv_path)
        self._csv_signature = signature
        return rows

    def tick(self) -> str:
        """One pass: refresh the CSV, append and return the status line."""
        self.refresh_csv()
        line = status_line(read_snapshot(self.output))
        self.append(line)
        return line

    def finished(self) -> Optional[int]:
        return run_finished(self.output, self.pid, self.since)

    def conclude(self, code: int) -> str:
        """The run ended: the final CSV, the summary into the log and ``FINAL_SUMMARY.txt``; returns the text."""
        self.refresh_csv()
        verdict = "completed" if code == 0 else ("killed (no exit code recorded)" if code == EXIT_CODE_UNKNOWN
                                                  else "FAILED with exit code %d" % code)
        text = "%s | run %s\n%s\nepisodes.csv: %s" % (_utc(), verdict, describe_summary(self.output), self.csv_path)
        self.append(text)
        write_atomically(self.output / SUMMARY_FILE, text)
        return text

    def watch(self, interval_s: float, once: bool = False) -> int:
        """Tick until the run ends (or once); returns the exit code to report (0 while a run is still going)."""
        self.append("%s | monitor started (pid %s, every %s)" % (
            _utc(self.since), "none" if self.pid is None else self.pid, human_duration(interval_s)))
        while True:
            print(self.tick(), flush=True)
            code = self.finished()
            if code is not None:
                print(self.conclude(code), flush=True)
                return 0 if code == 0 else 1
            if once:
                return 0
            time.sleep(interval_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Follow a Gibson run directory: a status line per tick into monitor.log, "
                                                 "episodes.csv kept current, the summary when the run ends.")
    parser.add_argument("output", type=pathlib.Path, help="the run directory (run_benchmark.sh --output)")
    parser.add_argument("--interval", type=float, default=300.0, help="seconds between ticks (default 300)")
    parser.add_argument("--pid", type=int, help="the launcher's PID (default: <output>/launcher.pid when present)")
    parser.add_argument("--once", action="store_true", help="one tick and exit")
    parser.add_argument("--log", type=pathlib.Path, help="where the status lines go (default <output>/monitor.log)")
    parser.add_argument("--csv", type=pathlib.Path, help="the scalar CSV (default <output>/episodes.csv)")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    pid = args.pid if args.pid is not None else read_pid(args.output)
    monitor = Monitor(args.output, pid=pid, log=args.log, csv_path=args.csv, since=launcher_started(args.output))
    return monitor.watch(args.interval, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())

