"""Gather one-mission-per-process flight results into an arm's ``summary.json``.

    python -m ...world_goal.aggregate_flights --flights ~/navdp_world_goal/flights

``fly_navdp.py`` flies a single mission per session, because a mission starts
wherever the aircraft already is and nothing can reposition it between two. Each
session leaves a ``results_NN.json`` behind. This joins them into the one
``summary.json`` per arm that ``report.py`` reads, with the same schema
``fly_navdp._write_summary`` writes for a single-session run.

It touches no simulator and no GPU, so a comparison interrupted halfway can be
summarised from whatever flew — which is the usual case, since a session that
never got PX4 up leaves no file at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_results(arm_dir: Path) -> List[Dict]:
    """Every mission result in one arm's directory, ordered by mission index.

    Reads both the ``results_NN.json`` of one-mission sessions and the
    ``results.json`` of an all-in-one session, so a directory containing either
    kind — or both — summarises correctly. A mission flown twice keeps the last
    result read, which is the re-flight.
    """
    by_mission: Dict[int, Dict] = {}
    for path in sorted(arm_dir.glob("results*.json")):
        try:
            entries = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for entry in entries:
            by_mission[int(entry.get("mission", len(by_mission)))] = entry
    return [by_mission[key] for key in sorted(by_mission)]


def summarise(arm: str, results: List[Dict]) -> Dict:
    """The arm's aggregate, in the schema ``report.py`` expects."""
    mean = lambda key: (float(np.nanmean([r.get(key, float("nan")) for r in results]))
                        if results else float("nan"))
    return {
        "arm": arm,
        "missions": len(results),
        "reached": sum(1 for r in results if r.get("reached")),
        "collisions": sum(1 for r in results if r.get("collided")),
        "min_clear_m": mean("min_clear_m"),
        "path_len_m": mean("path_len_m"),
        "duration_s": mean("duration_s"),
        "goal_error_m": mean("goal_error_m"),
        "results": results,
    }


def aggregate(flights_dir: Path) -> Dict[str, Dict]:
    """Write a ``summary.json`` into every arm directory. Returns them by arm."""
    summaries = {}
    for arm_dir in sorted(p for p in flights_dir.iterdir() if p.is_dir()):
        results = load_results(arm_dir)
        if not results:
            continue
        summary = summarise(arm_dir.name, results)
        (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries[arm_dir.name] = summary
    return summaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flights", required=True,
                        help="directory holding one subdirectory per arm")
    args = parser.parse_args(argv)

    summaries = aggregate(Path(args.flights).expanduser())
    if not summaries:
        print("[aggregate] no flight results found")
        return 1
    for arm, summary in summaries.items():
        print(f"[aggregate] {arm}: {summary['reached']}/{summary['missions']} reached, "
              f"{summary['collisions']} collisions, "
              f"mean min clearance {summary['min_clear_m']:.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
