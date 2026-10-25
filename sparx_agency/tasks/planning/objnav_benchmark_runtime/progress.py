"""Live progress of a benchmark run, and the lean per-episode record a 1,000-episode run can afford.

Two things a long benchmark needs that the harness does not provide on its own:

* **Progress you can watch.** :class:`ProgressTracker` keeps the running
  SR / SPL / SoftSPL / DTG -- overall, per scene and per target category --
  and rewrites one small ``progress.json`` after every episode (atomically,
  so a reader never sees half a file), with the percentage done, the
  elapsed and estimated remaining time, the episode in progress and the last
  episode's outcome. It also returns one console line per episode with a
  text bar. A resumed run is seeded with the rows already on disk, so the
  numbers describe the whole run, not the session.
* **Lean records.** ``episodes.jsonl`` holds the agent's full diagnostics by
  default -- about 100 KB per episode, most of it the per-action coverage
  curve and the loop's event log -- which is what you want for one recorded
  episode and not for a thousand. :func:`lean_episode_info` keeps the
  counters that explain an outcome (rooms, LLM calls, closing, loop stats,
  fallback stats, latency summary) and drops the per-action series; the
  score fields of the row are untouched.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord

UTC = "%Y-%m-%dT%H:%M:%SZ"

#: The policy diagnostics a lean record keeps whole (all small, all counters or verdicts).
LEAN_POLICY_KEYS = ("method", "rooms", "max_rooms", "landmarks", "warmup", "llm_queries", "oracle_reuses",
                    "room_label_queries", "plan_calls", "duplicates_removed", "blocked", "floor_revisions",
                    "oracle_repair_attempts", "oracle_repair_successes", "target_closing", "spawn_floor_guard",
                    "target_evidence", "room_scans", "openings", "glances", "sight", "doors", "doorway_peek",
                    "exploration")
#: Per-action series and event logs a lean record drops, wherever they appear.
LEAN_DROP_KEYS = frozenset(("coverage_curve", "events", "estimate_events", "estimates", "records", "history",
                            "label_history", "solver_records", "last_reasoning", "failures", "last_demoted",
                            "regrown", "floor_maps", "camera", "perception", "hierarchy", "building",
                            "frontier_sweep", "route_commitment", "supervisor", "settings"))


def _utc(ts: Optional[float] = None) -> str:
    return time.strftime(UTC, time.gmtime(time.time() if ts is None else ts))


def _write_atomically(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class _Group:
    """Running sums for one group of episodes (overall, a scene, a category)."""

    def __init__(self) -> None:
        self.n = 0
        self.successes = 0
        self.spl = 0.0
        self.soft_spl = 0.0
        self.dtg = 0.0
        self.dtg_n = 0
        self.steps = 0
        self.wall_s = 0.0
        self.stops = 0
        self.errors = 0

    def add(self, record: EpisodeRecord) -> None:
        self.n += 1
        self.successes += int(bool(record.success))
        self.spl += float(record.spl)
        self.soft_spl += float(record.soft_spl)
        if record.distance_to_goal_m is not None and math.isfinite(float(record.distance_to_goal_m)):
            self.dtg += float(record.distance_to_goal_m)
            self.dtg_n += 1
        self.steps += int(record.steps)
        self.wall_s += float(record.wall_s)
        self.stops += int(bool(record.stop_called))
        self.errors += int(record.agent_error is not None)

    def report(self) -> Dict[str, Any]:
        n = max(1, self.n)
        return {"episodes": self.n, "SR": round(self.successes / n, 4), "SPL": round(self.spl / n, 4),
                "SoftSPL": round(self.soft_spl / n, 4),
                "DTG_m": round(self.dtg / self.dtg_n, 3) if self.dtg_n else None,
                "unreachable_end": self.n - self.dtg_n, "mean_steps": round(self.steps / n, 1),
                "stop_rate": round(self.stops / n, 4), "agent_errors": self.errors,
                "mean_wall_s": round(self.wall_s / n, 1)}


class ProgressTracker:
    """The run's progress: running metrics, ETA, the episode in progress; one JSON file and one console line.

    Args:
        path: Where ``progress.json`` is written (rewritten after every episode).
        total: How many episodes the run selects, resumed ones included.
        completed: Rows already on disk when the run (re)started; counted as
            done and folded into the running metrics, not into the ETA.
        label: A name for the run, written into the file.
        bar_width: Characters in the console bar.
    """

    def __init__(self, path: pathlib.Path, total: int, completed: Sequence[EpisodeRecord] = (),
                 label: str = "gibson val", bar_width: int = 30) -> None:
        self.path = pathlib.Path(path)
        self.total = int(total)
        self.label = label
        self.bar_width = max(10, int(bar_width))
        self.started = time.time()
        self.session_wall_s = 0.0
        self.session_done = 0
        self.overall = _Group()
        self.by_scene = {}      # type: Dict[str, _Group]
        self.by_category = {}   # type: Dict[str, _Group]
        self.terminations = {}  # type: Dict[str, int]
        self.current = None     # type: Optional[Dict[str, Any]]
        self.last = None        # type: Optional[Dict[str, Any]]
        self.finished_utc = None  # type: Optional[str]
        self.resumed_from = len(completed)
        for record in completed:
            self._fold(record)
        self.write()

    # -- the episode in progress ------------------------------------------------
    def begin(self, episode_id: str) -> None:
        """An episode started: remembered for the file (and for a reader of a killed run)."""
        scene = episode_id.split("/")[0] if "/" in episode_id else None
        self.current = {"episode_id": episode_id, "scene": scene, "started_utc": _utc(), "started_at": time.time()}
        self.write()

    # -- after an episode ---------------------------------------------------------
    def update(self, position: int, total: int, record: EpisodeRecord) -> str:
        """Fold one finished episode in, rewrite the file, and return the console line."""
        self.total = int(total)
        self._fold(record)
        self.session_done += 1
        self.session_wall_s += float(record.wall_s)
        if self.current is not None and self.current.get("episode_id") == record.episode_id:
            self.current = None
        self.last = {"episode_id": record.episode_id, "scene": record.scene_id, "target": record.target_category,
                     "SR": int(bool(record.success)), "SPL": round(float(record.spl), 4),
                     "SoftSPL": round(float(record.soft_spl), 4),
                     "DTG_m": None if record.distance_to_goal_m is None else round(float(record.distance_to_goal_m), 3),
                     "steps": int(record.steps), "stop_called": bool(record.stop_called),
                     "termination": record.termination, "wall_s": round(float(record.wall_s), 1),
                     "agent_error": record.agent_error}
        self.write()
        return self.console_line()

    def finish(self) -> None:
        self.finished_utc = _utc()
        self.current = None
        self.write()

    # -- the numbers --------------------------------------------------------------
    def _fold(self, record: EpisodeRecord) -> None:
        self.overall.add(record)
        self.by_scene.setdefault(record.scene_id, _Group()).add(record)
        self.by_category.setdefault(record.target_category, _Group()).add(record)
        self.terminations[record.termination] = self.terminations.get(record.termination, 0) + 1

    @property
    def done(self) -> int:
        return self.overall.n

    @property
    def percent(self) -> float:
        return 100.0 * self.done / self.total if self.total else 100.0

    def eta_s(self) -> Optional[float]:
        """Seconds left, from this session's mean episode wall time; None before the first episode."""
        if not self.session_done:
            return None
        remaining = max(0, self.total - self.done)
        return remaining * (self.session_wall_s / self.session_done)

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        eta = self.eta_s()
        return {"label": self.label, "started_utc": _utc(self.started), "updated_utc": _utc(now),
                "finished_utc": self.finished_utc, "total": self.total, "done": self.done,
                "remaining": max(0, self.total - self.done), "percent": round(self.percent, 2),
                "resumed_from": self.resumed_from, "session_episodes": self.session_done,
                "elapsed_s": round(now - self.started, 1), "eta_s": None if eta is None else round(eta, 1),
                "eta_human": None if eta is None else human_duration(eta),
                "mean_episode_s": round(self.session_wall_s / self.session_done, 1) if self.session_done else None,
                "current": None if self.current is None else {k: v for k, v in self.current.items() if k != "started_at"},
                "last": self.last,
                "overall": self.overall.report(),
                "by_scene": {key: group.report() for key, group in sorted(self.by_scene.items())},
                "by_category": {key: group.report() for key, group in sorted(self.by_category.items())},
                "terminations": dict(sorted(self.terminations.items()))}

    def write(self) -> None:
        _write_atomically(self.path, json.dumps(self.snapshot(), indent=1, allow_nan=False) + "\n")

    def console_line(self) -> str:
        filled = int(round(self.bar_width * self.done / self.total)) if self.total else self.bar_width
        bar = "#" * filled + "-" * (self.bar_width - filled)
        overall = self.overall.report()
        eta = self.eta_s()
        text = "[%s] %d/%d %5.1f%% | SR %.3f SPL %.3f DTG %s | ETA %s" % (
            bar, self.done, self.total, self.percent, overall["SR"], overall["SPL"],
            "n/a" if overall["DTG_m"] is None else "%.2f" % overall["DTG_m"],
            "n/a" if eta is None else human_duration(eta))
        if self.last:
            last = self.last
            text += " | last %s %s SR=%d SPL=%.3f steps=%d (%ds)" % (
                last["episode_id"], last["target"], last["SR"], last["SPL"], last["steps"], int(last["wall_s"]))
            if last.get("agent_error"):
                text += " AGENT ERROR"
        return text


def human_duration(seconds: float) -> str:
    """``3d02h``, ``3h12m``, ``47m05s``, ``55s``."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours >= 24:
        return "%dd%02dh" % divmod(hours, 24)
    if hours:
        return "%dh%02dm" % (hours, minutes)
    if minutes:
        return "%dm%02ds" % (minutes, secs)
    return "%ds" % secs


class ProgressEnv(ObjNavEnv):
    """An environment wrapper that tells the tracker when an episode begins; everything else passes through."""

    def __init__(self, env: ObjNavEnv, tracker: ProgressTracker) -> None:
        self.env, self.tracker = env, tracker

    @property
    def name(self):
        return self.env.name

    def episode_ids(self):
        return self.env.episode_ids()

    def reset(self, episode_id):
        self.tracker.begin(episode_id)
        return self.env.reset(episode_id)

    def step(self, action):
        return self.env.step(action)

    @property
    def episode_over(self):
        return self.env.episode_over

    def measure(self):
        return self.env.measure()

    def close(self):
        self.env.close()

    def evaluation_diagnostics(self):
        return self.env.evaluation_diagnostics()

    def validate_starts(self, ids):
        return self.env.validate_starts(ids)

    def __getattr__(self, name):
        return getattr(self.env, name)


# -- lean records -------------------------------------------------------------
def _prune(value: Any, depth: int = 0) -> Any:
    """Drop :data:`LEAN_DROP_KEYS` at every level of a diagnostics tree."""
    if isinstance(value, dict):
        return {key: _prune(item, depth + 1) for key, item in value.items() if key not in LEAN_DROP_KEYS}
    if isinstance(value, list):
        return [_prune(item, depth + 1) for item in value] if depth < 3 else "[%d items]" % len(value)
    return value


def lean_episode_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """The counters that explain an outcome, without the per-action series.

    The headless agent's own log (statuses, steps, blocked counts) stays
    whole. Of the policy's report, :data:`LEAN_POLICY_KEYS` are kept
    (pruned of :data:`LEAN_DROP_KEYS` inside), the room-search loop and the
    exploration fallback are reduced to their ``stats``, and the exploration
    metrics to their scalar summary and latency table.
    """
    out = {key: value for key, value in info.items() if key != "policy"}
    policy = info.get("policy")
    if not isinstance(policy, dict):
        return out
    lean = {}  # type: Dict[str, Any]
    for key in LEAN_POLICY_KEYS:
        if key in policy:
            lean[key] = _prune(policy[key])
    loop = policy.get("room_search_loop")
    if isinstance(loop, dict):
        lean["room_search_loop"] = {"stats": loop.get("stats"), "second_pass": loop.get("second_pass"),
                                    "excluded": loop.get("excluded"), "finished": loop.get("finished"),
                                    "order": loop.get("order")}
    fallback = policy.get("exploration_fallback")
    if isinstance(fallback, dict):
        lean["exploration_fallback"] = {"stats": fallback.get("stats"),
                                        "failures": len(fallback.get("failures") or ())}
    metrics = policy.get("exploration_metrics")
    if isinstance(metrics, dict):
        lean["exploration_metrics"] = {key: value for key, value in metrics.items()
                                       if key not in ("coverage_curve", "coverage_definition")}
    reasoning = policy.get("last_reasoning")
    if isinstance(reasoning, dict):
        oracle = reasoning.get("oracle") or {}
        lean["last_oracle"] = {"source": oracle.get("source"), "reading": oracle.get("reading"),
                               "pass_verdict": oracle.get("pass_verdict"), "nodes": len(oracle.get("probs") or {})}
    out["policy"] = lean
    out["lean"] = True
    return out


class LeanHeadlessObjNavAgent(HeadlessObjNavAgent):
    """The headless agent whose per-episode diagnostics are :func:`lean_episode_info` of the full report."""

    def episode_info(self) -> Dict[str, Any]:
        return lean_episode_info(super().episode_info())


# -- the command line: read a run's progress or its finished summary ----------------------------
def _group_line(name: str, group: Dict[str, Any]) -> str:
    dtg = group.get("DTG_m")
    return "  %-14s n=%4d  SR %.3f  SPL %.3f  SoftSPL %.3f  DTG %s  steps %.0f  stop %.2f  errors %d" % (
        name, group["episodes"], group["SR"], group["SPL"], group["SoftSPL"],
        "  n/a" if dtg is None else "%5.2f" % dtg, group["mean_steps"], group["stop_rate"], group["agent_errors"])


def describe_progress(directory: pathlib.Path) -> str:
    """The progress file of a run directory as a few readable lines; a parallel run's shards are folded into one view."""
    path = pathlib.Path(directory) / "progress.json"
    if not path.is_file():
        shards = shard_progress_files(directory)
        if shards:
            return describe_shards(directory, shards)
        return "no progress.json in %s (the run has not started an episode yet)" % directory
    snap = json.loads(path.read_text(encoding="utf-8"))
    return _describe_snapshot(snap)


def _describe_snapshot(snap: Dict[str, Any]) -> str:
    lines = ["%s: %d/%d episodes (%.1f%%)%s" % (snap.get("label", "run"), snap["done"], snap["total"], snap["percent"],
                                                "  FINISHED %s" % snap["finished_utc"] if snap.get("finished_utc") else "")]
    lines.append("  started %s  updated %s  elapsed %s  ETA %s" % (
        snap["started_utc"], snap["updated_utc"], human_duration(snap["elapsed_s"]), snap.get("eta_human") or "n/a"))
    if snap.get("current"):
        lines.append("  running: %s (since %s)" % (snap["current"]["episode_id"], snap["current"]["started_utc"]))
    if snap.get("last"):
        last = snap["last"]
        lines.append("  last: %s %s SR=%d SPL=%.3f DTG=%s steps=%d %s%s" % (
            last["episode_id"], last["target"], last["SR"], last["SPL"],
            "n/a" if last["DTG_m"] is None else "%.2f" % last["DTG_m"], last["steps"], last["termination"],
            " AGENT ERROR: %s" % last["agent_error"] if last.get("agent_error") else ""))
    lines.append(_group_line("overall", snap["overall"]))
    for name, group in snap.get("by_scene", {}).items():
        lines.append(_group_line(name, group))
    for name, group in snap.get("by_category", {}).items():
        lines.append(_group_line(name, group))
    if snap.get("terminations"):
        lines.append("  terminations: " + ", ".join("%s=%d" % item for item in snap["terminations"].items()))
    return "\n".join(lines)


# -- a parallel run (run_benchmark.sh --parallel N): one progress file per shard until the merge ----------------
def shard_progress_files(directory: pathlib.Path) -> List[pathlib.Path]:
    """The ``shard_<k>/progress.json`` files of a parallel run, in shard order (empty for a single-process run)."""
    directory = pathlib.Path(directory)
    found = []
    for path in directory.glob("shard_*/progress.json"):
        suffix = path.parent.name[len("shard_"):]
        if suffix.isdigit() and path.is_file():
            found.append((int(suffix), path))
    return [path for _, path in sorted(found)]


def _weighted(groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One ``_Group.report()``-shaped row over several, weighted by their episode counts."""
    n = sum(int(group["episodes"]) for group in groups)
    dtg_n = sum(int(group["episodes"]) - int(group.get("unreachable_end", 0)) for group in groups)

    def mean(key: str, weights: Sequence[int], total: int) -> float:
        return round(sum(float(group[key]) * weight for group, weight in zip(groups, weights)) / total, 4) if total else 0.0

    counts = [int(group["episodes"]) for group in groups]
    reached = [int(group["episodes"]) - int(group.get("unreachable_end", 0)) for group in groups]
    dtg = [float(group["DTG_m"]) if group.get("DTG_m") is not None else 0.0 for group in groups]
    return {"episodes": n, "SR": mean("SR", counts, n), "SPL": mean("SPL", counts, n), "SoftSPL": mean("SoftSPL", counts, n),
            "DTG_m": round(sum(d * w for d, w in zip(dtg, reached)) / dtg_n, 3) if dtg_n else None,
            "unreachable_end": n - dtg_n, "mean_steps": round(mean("mean_steps", counts, n), 1),
            "stop_rate": mean("stop_rate", counts, n), "agent_errors": sum(int(group["agent_errors"]) for group in groups),
            "mean_wall_s": round(mean("mean_wall_s", counts, n), 1)}


def describe_shards(directory: pathlib.Path, files: Sequence[pathlib.Path]) -> str:
    """The shards of a parallel run as one view: the totals, the slowest shard's ETA, then one line per shard."""
    snaps = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    done = sum(int(snap["done"]) for snap in snaps)
    total = sum(int(snap["total"]) for snap in snaps)
    etas = [snap.get("eta_s") for snap in snaps if not snap.get("finished_utc")]
    eta = None if not etas or any(value is None for value in etas) else max(float(value) for value in etas)
    finished = all(snap.get("finished_utc") for snap in snaps)
    lines = ["%s: %d/%d episodes (%.1f%%) in %d shards%s" % (
        snaps[0].get("label", "run"), done, total, 100.0 * done / total if total else 0.0, len(snaps),
        "  ALL SHARDS FINISHED (merge pending or done)" if finished else "")]
    lines.append("  started %s  updated %s  ETA (slowest shard) %s" % (
        min(snap["started_utc"] for snap in snaps), max(snap["updated_utc"] for snap in snaps),
        "n/a" if eta is None else human_duration(eta)))
    lines.append(_group_line("overall", _weighted([snap["overall"] for snap in snaps])))
    scenes = sorted({name for snap in snaps for name in snap.get("by_scene", {})})
    for name in scenes:
        lines.append(_group_line(name, _weighted([snap["by_scene"][name] for snap in snaps if name in snap.get("by_scene", {})])))
    terminations = {}  # type: Dict[str, int]
    for snap in snaps:
        for key, value in (snap.get("terminations") or {}).items():
            terminations[key] = terminations.get(key, 0) + int(value)
    if terminations:
        lines.append("  terminations: " + ", ".join("%s=%d" % item for item in sorted(terminations.items())))
    for path, snap in zip(files, snaps):
        current = snap.get("current") or {}
        lines.append("  %-8s %3d/%3d  SR %.3f  SPL %.3f  elapsed %s  ETA %s%s" % (
            path.parent.name, snap["done"], snap["total"], snap["overall"]["SR"], snap["overall"]["SPL"],
            human_duration(snap["elapsed_s"]), snap.get("eta_human") or "n/a",
            "  running %s" % current["episode_id"] if current.get("episode_id") else
            ("  FINISHED" if snap.get("finished_utc") else "")))
    return "\n".join(lines)


def describe_summary(directory: pathlib.Path) -> str:
    """The finished run's ``summary.json`` as the headline numbers, or the progress when it has none yet."""
    path = pathlib.Path(directory) / "summary.json"
    if not path.is_file():
        return describe_progress(directory)
    summary = json.loads(path.read_text(encoding="utf-8"))
    overall = summary["overall"]
    lines = ["RESULT %s/%s %s: %d episodes" % (summary["benchmark"], summary["split"], summary["agent"], overall["n_episodes"]),
             "  SR %.3f (95%% CI %.3f-%.3f)  SPL %.3f (95%% CI %.3f-%.3f)  SoftSPL %.3f  DTG %s m  mean steps %.0f" % (
                 overall["success_rate"], overall["success_rate_ci"][0], overall["success_rate_ci"][1],
                 overall["spl"], overall["spl_ci"][0], overall["spl_ci"][1], overall["soft_spl"],
                 "n/a" if overall.get("distance_to_goal_m") is None else "%.3f" % overall["distance_to_goal_m"],
                 overall["mean_steps"])]
    for block in ("by_scene", "by_category"):
        for group in summary.get(block, ()):
            lines.append("  %-14s n=%4d  SR %.3f  SPL %.3f  SoftSPL %.3f  DTG %s" % (
                group["key"], group["n_episodes"], group["success_rate"], group["spl"], group["soft_spl"],
                "n/a" if group.get("distance_to_goal_m") is None else "%.2f" % group["distance_to_goal_m"]))
    lines.append("  terminations: %s  agent errors: %d" % (
        ", ".join("%s=%d" % item for item in summary.get("terminations", {}).items()), summary.get("agent_errors", 0)))
    audit = pathlib.Path(directory) / "audit.json"
    if audit.is_file():
        data = json.loads(audit.read_text(encoding="utf-8"))
        lines.append("  full published split completed: %s  (comparison.md beside this file)" % data.get("full_split_completed"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Print the live progress (default) or the finished summary of a benchmark run directory.")
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--summary", action="store_true", help="the finished summary.json instead of progress.json")
    args = parser.parse_args(argv)
    print(describe_summary(args.directory) if args.summary else describe_progress(args.directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
