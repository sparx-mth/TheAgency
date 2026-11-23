"""The short, committable record of a finished MP3D campaign: what it measured, under what, and when.

A campaign root is hundreds of megabytes of rows, step logs, answer stores and
console logs that stay on the machine that ran them. What travels with the
branch is this directory (``mp3d/results/<run name>/``, the convention of
``gibson/results/``):

* ``summary.md`` -- SR and SPL overall and per goal category (with the harness's
  intervals), SoftSPL and DTG, the episode count and whether the split is
  complete, the LLM in each role, the node oracle's prompt version and digest
  and whether that version was tuned on the model that answered, the source
  commit (and whether the tree was clean), perception, the start and end
  times and the duration;
* ``summary.json`` -- the same as data (``aggregate.summarise`` of the rows,
  plus the provenance), for whoever compares runs by script;
* ``episodes.csv`` -- one row per episode (the ``gibson/results`` columns);
* ``campaign_summary.json``, ``campaign_metrics.csv`` and one scene's
  ``run.json`` (the frozen configuration), copied unchanged.

::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.results_summary runs/X \\
        [--out sparx_agency/.../mp3d/results/X] [--title "..."]

Standard library plus the harness's aggregate; no simulator, no model.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.results_io import read_episode_rows
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    model_provenance, model_report_rows, model_report_title)

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
EPISODE_COLUMNS = ("episode_id", "scene_id", "target_category", "success", "spl", "soft_spl", "path_length_m",
                   "shortest_path_m", "start_distance_to_goal_m", "distance_to_goal_m", "steps", "stop_called",
                   "termination", "wall_s", "agent_error")
UTC = "%Y-%m-%dT%H:%M:%SZ"


def _read(path: Path) -> Optional[str]:
    return path.read_text().strip() if path.is_file() else None


def _json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.is_file() else None


def _duration(start: Optional[str], end: Optional[str]) -> Optional[float]:
    if not start or not end:
        return None
    return round((time.mktime(time.strptime(end, UTC)) - time.mktime(time.strptime(start, UTC))) / 3600.0, 2)


def provenance(root: Path, started_utc: Optional[str] = None, finished_utc: Optional[str] = None) -> Dict[str, object]:
    """Commit, tree state, LLM roles, prompt version and digest, perception, timing: from the files the launcher and the children wrote."""
    scenes = sorted(p for p in root.iterdir() if p.is_dir() and (p / "run.json").is_file())
    run = _json(scenes[0] / "run.json") if scenes else None
    config = (run or {}).get("config") or {}
    method = config.get("method") or {}
    diff = root / "source-changes.diff"
    # The launcher rewrites started_utc.txt at every (resumed) launch: a caller that knows the first start says so.
    started = started_utc or _read(root / "started_utc.txt")
    finished = finished_utc or _read(root / "finished_utc.txt")
    return {
        "run_root": str(root),
        "source_commit": _read(root / "source-commit.txt"),
        "source_tree_clean": diff.is_file() and diff.stat().st_size == 0,
        **model_provenance(method),
        "perception": method.get("perception", "service"),
        "ground_truth": {k: v for k, v in (method.get("ground_truth") or {}).items() if k in ("on_object_m", "min_pixels")},
        "allow_stair_traversal": method.get("allow_stair_traversal"),
        "protocol_id": config.get("protocol_id"),
        "seed": config.get("seed"),
        "lean_records": config.get("lean_records"),
        "recording": config.get("recording"),
        "runtime": config.get("runtime"),
        "started_utc": started, "finished_utc": finished, "duration_h": _duration(started, finished),
        "exit_code": _read(root / "exit_code.txt"),
        "scenes_with_run_json": len(scenes),
    }


def load_records(root: Path) -> list:
    records = []
    for source in sorted(root.glob("*/episodes.jsonl")):
        records.extend(record for _, record in read_episode_rows(source))
    return records


def _group_row(name: str, g) -> str:
    dtg = "–" if g.distance_to_goal_m is None else "%.2f" % g.distance_to_goal_m
    return "| %s | %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f | %s | %.0f |" % (
        name, g.n_episodes, g.success_rate, g.success_rate_ci[0], g.success_rate_ci[1],
        g.spl, g.spl_ci[0], g.spl_ci[1], g.soft_spl, dtg, g.mean_steps)


def markdown(title: str, prov: Dict[str, object], summary, campaign: Optional[dict]) -> str:
    complete = campaign.get("complete") if campaign else None
    lines = ["# %s" % title, "",
             "| | |", "|---|---|",
             "| episodes | %d of %s published (%s) |" % (summary.overall.n_episodes, (campaign or {}).get("episodes_published", "?"),
                                                           "complete split" if complete else "INCOMPLETE"),
             "| scenes | %d |" % len(summary.by_scene),
             *model_report_rows(prov),
             "| perception | %s%s |" % (prov["perception"], (" (on-object %.2f m, min %s px)" % (prov["ground_truth"].get("on_object_m", 0),
                                                                                       prov["ground_truth"].get("min_pixels"))) if prov["ground_truth"] else ""),
             "| stair traversal | %s |" % prov["allow_stair_traversal"],
             "| source commit | `%s`%s |" % (prov["source_commit"], "" if prov["source_tree_clean"] else " (tree NOT clean: see source-changes.diff in the run root)"),
             "| protocol | %s, seed %s |" % (prov["protocol_id"], prov["seed"]),
             "| started / finished (UTC) | %s / %s (%s h) |" % (prov["started_utc"], prov["finished_utc"], prov["duration_h"]),
             "| run root (on the machine that ran it) | `%s` |" % prov["run_root"],
             "", "## Metrics (means over every scored episode, failures included; 95 % intervals)", "",
             "| group | n | SR [CI] | SPL [CI] | SoftSPL | DTG m | steps |", "|---|---|---|---|---|---|---|",
             _group_row("**overall**", summary.overall)]
    for g in summary.by_category:
        lines.append(_group_row(g.key, g))
    lines += ["", "Terminations: %s; agent errors: %d." % (json.dumps(summary.terminations), summary.agent_errors),
              "", "## Per scene", "", "| scene | n | SR | SPL | DTG m |", "|---|---|---|---|---|"]
    for g in summary.by_scene:
        lines.append("| %s | %d | %.3f | %.3f | %s |" % (g.key, g.n_episodes, g.success_rate, g.spl,
                                                       "–" if g.distance_to_goal_m is None else "%.2f" % g.distance_to_goal_m))
    lines += ["", "Written by `mp3d/results_summary.py` from the run root's `episodes.jsonl` rows; the rows, step logs, "
              "LLM answer store and console logs stay with the run root."]
    return "\n".join(lines) + "\n"


def write(root: Path, out: Path, title: Optional[str] = None, started_utc: Optional[str] = None,
          finished_utc: Optional[str] = None) -> Dict[str, object]:
    root = root.resolve()
    records = load_records(root)
    if not records:
        raise ValueError("no episode rows under %s" % root)
    summary = summarise(records)
    prov = provenance(root, started_utc, finished_utc)
    campaign = _json(root / "campaign_summary.json")
    title = title or model_report_title("MP3D", prov)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(markdown(title, prov, summary, campaign))
    data = {"title": title, "provenance": prov, "campaign": {k: v for k, v in (campaign or {}).items() if k != "by_scene"},
            "summary": asdict(summary)}
    (out / "summary.json").write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    with (out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in sorted(records, key=lambda r: r.episode_id):
            row = record.to_row()
            writer.writerow({k: ("" if row.get(k) is None else (int(row[k]) if isinstance(row[k], bool) else row[k]))
                             for k in EPISODE_COLUMNS})
    for name in ("campaign_summary.json", "campaign_metrics.csv"):
        if (root / name).is_file():
            shutil.copy2(root / name, out / name)
    scenes = sorted(p for p in root.iterdir() if p.is_dir() and (p / "run.json").is_file())
    if scenes:
        shutil.copy2(scenes[0] / "run.json", out / "run.json")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="The campaign root")
    parser.add_argument("--out", type=Path, help="Where to write (default: mp3d/results/<root's name>)")
    parser.add_argument("--title")
    parser.add_argument("--started-utc", help="The campaign's first start (%s); default: the root's started_utc.txt" % UTC.replace("%", "%%"))
    parser.add_argument("--finished-utc", help="Its end; default: the root's finished_utc.txt")
    args = parser.parse_args(argv)
    out = args.out or DEFAULT_RESULTS_ROOT / args.root.name
    data = write(args.root, out, args.title, args.started_utc, args.finished_utc)
    overall = data["summary"]["overall"]
    print("%s: %d episodes, SR %.3f SPL %.3f -> %s" % (data["title"], overall["n_episodes"], overall["success_rate"],
                                                        overall["spl"], out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
