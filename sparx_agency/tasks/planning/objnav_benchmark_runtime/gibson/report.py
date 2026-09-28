"""Auditable Gibson reports and strict shard merging; no simulator/model imports."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.comparison import comparison_table
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.results_io import read_episode_rows, strict_json, write_atomically
from sparx_agency.tasks.planning.objnav_benchmark.run_info import read_run_info
from sparx_agency.tasks.planning.objnav_benchmark.summaries import ReportedResult
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, SCENES

SOURCE = "arXiv:2508.04678v1, Table 2, p.12"
# Transcribed from the paper, not measured here. See README for protocol gaps.
# TF/NM are labels, not quantities to average, and GT in OSG means semantics.
REPORTED = (
    ("SemExp", 0.657, 0.339, 1.474, False, False),
    ("PONI", 0.736, 0.410, 1.250, False, False),
    ("LGX", 0.310, 0.052, 4.775, True, False),
    ("LFG", 0.645, 0.406, 1.812, True, False),
    ("FBE", 0.641, 0.283, 1.780, True, False),
    ("SemUtil", 0.693, 0.405, 1.488, True, False),
    ("OSG-Nav", 0.734, 0.386, 1.722, True, True),
)


def _read_run(directory):
    directory = Path(directory).expanduser()
    config = read_run_info(directory / "run.json")["config"]
    if config.get("protocol") != asdict(PROTOCOL):
        raise ValueError("Not a Gibson SemExp v1.1 run: %s" % directory)
    records = [row for _, row in read_episode_rows(directory / "episodes.jsonl")]
    ids = [row.episode_id for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode records")
    if not set(ids).issubset(config["selected_episode_ids"]):
        raise ValueError("Records do not belong to the recorded selection")
    if any((row.benchmark, row.split, row.agent) != (
            "gibson", "val", config["method"]["method"]) for row in records):
        raise ValueError("Record identity disagrees with the run configuration")
    return config, records


def write_report(directory):
    """Recompute from episode rows; mark unfinished/subset/diagnostic results."""
    directory = Path(directory).expanduser()
    config, records = _read_run(directory)
    summary = summarise(records)
    full_ids = {"%s/%06d" % (scene, index) for scene in SCENES for index in range(200)}
    full = (config["full_split"] and {r.episode_id for r in records} == full_ids
            and config["method"]["method"] != "diagnostic-stop")
    title = "Full Gibson validation" if full else "NOT a full benchmark result (subset, incomplete, or diagnostic)"
    baselines = [ReportedResult(
        method, "gibson", "val", sr, spl, SOURCE, distance_to_goal_m=dtg,
        notes="TF=%s; NM=%s. Paper-reported, protocol equivalence not established."
              % (tf, nm)) for method, sr, spl, dtg, tf, nm in REPORTED]
    audit = {"full_split_completed": full, "episodes": len(records),
             "selected_episodes": len(config["selected_episode_ids"]),
             "agent_errors": summary.agent_errors,
             "reference_sim_version_match": config["reference_sim_version_match"],
             "exact_osg_protocol_reproduction": False,
             "primary_metrics": ["SR", "SPL", "DTG (SemExp distance-to-success, m)"],
             "supplementary_metrics": ["SoftSPL", "steps", "wall_s"],
             "total_wall_s": sum(row.wall_s for row in records),
             "notes": "OSG's exact evaluator/episode hashes are not published in the supplied paper. "
                      "SemExp and current PONI code differ; no direct SOTA win is asserted."}
    text = "# %s\n\n" % title
    text += ("SR/SPL are episode means including failures; DTG here is FMM distance "
             "to the 1 m success region. SoftSPL is supplementary.\n\n")
    text += comparison_table(summary, baselines)
    text += ("\n\n**Comparison caveat:** the published rows are reference values, not "
             "paired re-evaluations. ApexNav and SG-Nav do not report Gibson. "
             "TF = training-free; NM = non-metric. Our RPT adaptation is TF, not NM. "
             "GT pose/depth do not mean GT semantic detections.\n\n")
    text += "## Audit\n\n" + "\n".join("- **%s:** %s" % item for item in audit.items()) + "\n"
    write_atomically(directory / "comparison.md", text)
    write_atomically(directory / "audit.json", strict_json(audit, "Gibson audit", indent=2))
    return audit


def merge_runs(directories, output):
    """Merge only complete, disjoint shards of exactly the same experiment."""
    runs = [_read_run(directory) for directory in directories]
    if not runs:
        raise ValueError("No shard directories supplied")
    base = runs[0][0]
    comparable_keys = ("protocol", "runtime", "source_sha256", "method", "seed",
                       "reference_sim_version_match", "dataset", "shards", "limit",
                       "on_agent_error")
    records, shard_ids = [], set()
    for config, rows in runs:
        if any(config[key] != base[key] for key in comparable_keys):
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
    expected = {"%s/%06d" % (scene, index) for scene in SCENES for index in range(200)}
    if len(ids) != 1000 or set(ids) != expected:
        raise ValueError("Merged runs must cover each of the 1,000 episodes exactly once")
    config = dict(base, selected_episode_ids=sorted(ids), full_split=True,
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

