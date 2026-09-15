"""Auditable MP3D reports and strict shard merging; no simulator/model imports.

The published rows below are transcribed from the comparison papers, not
measured here, and each carries the table it was read from. Where two papers
print different numbers for the same method (VLFM on MP3D), both rows are kept
under their own sources rather than silently averaged or picked.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.comparison import comparison_table
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    read_episode_rows, strict_json, write_atomically)
from sparx_agency.tasks.planning.objnav_benchmark.run_info import read_run_info
from sparx_agency.tasks.planning.objnav_benchmark.summaries import ReportedResult
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    PROTOCOL, PUBLISHED_VAL_EPISODES, PUBLISHED_VAL_SCENES)

SG_NAV = "arXiv:2410.08189v1, Table 1, p.7"
APEXNAV = "arXiv:2504.14478v1, Table I, p.6"

#: (method, SR %, SPL %, source, note). Both papers state the same MP3D
#: validation split: 11 scenes, 21 goal categories, 2,195 episodes.
REPORTED = (
    ("SemExp", 36.0, 14.4, SG_NAV, "trained ObjectNav policy; not zero-shot"),
    ("PONI", 31.8, 12.1, SG_NAV, "trained ObjectNav policy; not zero-shot"),
    ("ZSON", 15.3, 4.8, SG_NAV, "unsupervised pretraining; not zero-shot"),
    ("CoW", 7.4, 3.7, SG_NAV, "zero-shot"),
    ("ESC", 28.7, 14.2, SG_NAV, "zero-shot"),
    ("L3MVN", 34.9, 14.5, SG_NAV, "zero-shot"),
    ("OpenFMNav", 37.2, 15.7, SG_NAV, "zero-shot"),
    ("VLFM", 36.2, 15.9, SG_NAV, "zero-shot; ApexNav prints 36.4/17.5 for VLFM"),
    ("VLFM (ApexNav re-evaluation)", 36.4, 17.5, APEXNAV,
     "zero-shot; SG-Nav prints 36.2/15.9 for VLFM"),
    ("VLFM* (shortest-path planner)", 32.5, 15.9, APEXNAV,
     "PointNav module replaced by a shortest-path planner"),
    ("SG-Nav-LLaMA", 40.1, 16.0, SG_NAV, "zero-shot"),
    ("SG-Nav-GPT", 40.2, 16.0, SG_NAV, "zero-shot"),
    ("ApexNav", 39.2, 17.8, APEXNAV, "zero-shot"),
)


def baselines():
    """The transcribed rows as :class:`ReportedResult`, percentages divided by 100."""
    return [ReportedResult(method, PROTOCOL.benchmark, PROTOCOL.split, sr / 100.0,
                           spl / 100.0, source,
                           notes="%s. Paper-reported; protocol equivalence not "
                                 "established here." % note)
            for method, sr, spl, source, note in REPORTED]


def _read_run(directory):
    directory = Path(directory).expanduser()
    config = read_run_info(directory / "run.json")["config"]
    if config.get("protocol") != asdict(PROTOCOL):
        raise ValueError("Not an MP3D ObjectNav v1 run: %s" % directory)
    records = [row for _, row in read_episode_rows(directory / "episodes.jsonl")]
    ids = [row.episode_id for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode records")
    if not set(ids).issubset(config["selected_episode_ids"]):
        raise ValueError("Records do not belong to the recorded selection")
    if any((row.benchmark, row.split, row.agent)
           != (PROTOCOL.benchmark, PROTOCOL.split, config["method"]["method"])
           for row in records):
        raise ValueError("Record identity disagrees with the run configuration")
    return config, records


def write_report(directory):
    """Recompute from episode rows; mark unfinished, subset or diagnostic results."""
    directory = Path(directory).expanduser()
    config, records = _read_run(directory)
    summary = summarise(records)
    selected = config["selected_episode_ids"]
    complete = {row.episode_id for row in records} == set(selected)
    full = (config.get("full_split") and complete
            and len(selected) == PUBLISHED_VAL_EPISODES
            and config["method"]["method"] != "diagnostic-stop")
    title = ("Full MP3D ObjectNav v1 validation" if full else
             "NOT a full benchmark result (subset, incomplete, or diagnostic)")
    dataset = config.get("dataset") or {}
    starts = config.get("start_validation")
    audit = {
        "full_split_completed": bool(full),
        "episodes": len(records),
        "selected_episodes": len(selected),
        "scenes": len(dataset.get("scene_counts") or {}),
        "published_scenes": PUBLISHED_VAL_SCENES,
        "published_episodes": PUBLISHED_VAL_EPISODES,
        "agent_errors": summary.agent_errors,
        "reference_sim_version_match": config["reference_sim_version_match"],
        # None, not 0, when the run skipped the preflight: "not checked" and
        # "checked and clean" must not read the same in an audit.
        "unreachable_starts": (None if starts is None
                               else len(starts.get("unreachable_starts") or [])),
        "published_geodesic_gaps": (None if starts is None
                                    else len(starts.get("published_geodesic_gaps") or {})),
        "starts_preflighted": starts is not None,
        "primary_metrics": ["SR", "SPL"],
        "supplementary_metrics": ["DTG (final geodesic distance to a view point, m)",
                                  "SoftSPL", "steps", "wall_s"],
        "total_wall_s": sum(row.wall_s for row in records),
        "notes": "Success is STOP within %.2f m of a published goal view point, "
                 "as habitat-lab scores ObjectNav. Neither comparison paper "
                 "publishes per-episode results or its evaluator's hashes, so "
                 "no direct SOTA claim is asserted." % PROTOCOL.success_distance_m,
    }
    text = "# %s\n\n" % title
    text += ("SR and SPL are episode means including failures, and are the two "
             "metrics every MP3D paper below reports. DTG here is the final "
             "geodesic distance to the nearest published view point; SoftSPL is "
             "supplementary. Neither is a published MP3D baseline: the DTG/TF/NM "
             "columns of arXiv:2508.04678 cover HM3D and Gibson, not MP3D.\n\n")
    text += comparison_table(summary, baselines())
    text += ("\n\n**Comparison caveat:** the published rows are reference values "
             "transcribed from their papers, not paired re-evaluations. TF = "
             "training-free, NM = non-metric (the columns OSG-Nav uses): our "
             "method is training-free and metric. Ground-truth pose and depth "
             "do not mean ground-truth semantics -- detections stay predicted.\n\n")
    text += "## Audit\n\n" + "\n".join("- **%s:** %s" % item for item in audit.items()) + "\n"
    write_atomically(directory / "comparison.md", text)
    write_atomically(directory / "audit.json", strict_json(audit, "MP3D audit", indent=2))
    return audit


def merge_runs(directories, output):
    """Merge only complete, disjoint shards of exactly the same experiment."""
    runs = [_read_run(directory) for directory in directories]
    if not runs:
        raise ValueError("No shard directories supplied")
    base = runs[0][0]
    comparable = ("protocol", "runtime", "source_sha256", "method", "seed",
                  "reference_sim_version_match", "dataset", "shards", "limit", "scene")
    records, shard_ids = [], set()
    for config, rows in runs:
        if any(config.get(key) != base.get(key) for key in comparable):
            raise ValueError("Shards disagree on data, protocol, method, source, or runtime")
        if config["limit"] is not None or config["shard_index"] in shard_ids:
            raise ValueError("Limited or duplicate shards cannot form a full evaluation")
        if {r.episode_id for r in rows} != set(config["selected_episode_ids"]):
            raise ValueError("A shard is incomplete; resume it before merging")
        shard_ids.add(config["shard_index"])
        records.extend(rows)
    if shard_ids != set(range(base["shards"])):
        raise ValueError("Missing shards")
    ids = [r.episode_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Shards overlap; every episode must appear exactly once")
    config = dict(base, selected_episode_ids=sorted(ids), shards=base["shards"],
                  shard_index=0,
                  full_split=len(ids) == PUBLISHED_VAL_EPISODES and base.get("full_split"),
                  merged_from=[str(Path(p).resolve()) for p in directories])
    with MetricsLogger(Path(output), config) as logger:
        for row in sorted(records, key=lambda r: r.episode_id):
            logger.log(row)
        logger.finish(summarise(records))
    return write_report(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--merge-output", type=Path)
    args = parser.parse_args(argv)
    if args.merge_output:
        audit = merge_runs(args.runs, args.merge_output)
    elif len(args.runs) == 1:
        audit = write_report(args.runs[0])
    else:
        parser.error("Multiple runs require --merge-output")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
