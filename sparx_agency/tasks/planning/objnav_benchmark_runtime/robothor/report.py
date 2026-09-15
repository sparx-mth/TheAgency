"""Auditable RoboTHOR reports and strict shard merging; no simulator/model imports.

Recomputes every figure from the episode rows on disk rather than trusting a
summary file, and refuses to call a partial run a benchmark result. The report
states, in the artefact itself, the three things that decide whether a number
means anything:

* whether the complete published split actually finished;
* which reading of the visibility rule produced the success column, and what
  the other reading would have given -- both are recorded per episode, so the
  difference is reported rather than assumed;
* what fraction of episodes ship ``shortest_path_length == 0``, where upstream
  SPL arithmetic scores even a successful episode 0 unless the agent's very
  first action was STOP.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import asdict
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.comparison import comparison_table
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    read_episode_rows, strict_json, write_atomically,
)
from sparx_agency.tasks.planning.objnav_benchmark.run_info import read_run_info
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.protocol import (
    PROTOCOL, VAL_EPISODES,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.reported import (
    OUR_PROPERTIES, VALIDATION_RESULTS, properties_row,
)


def _read_run(directory):
    """The run configuration and its episode rows, checked against each other."""
    directory = Path(directory).expanduser()
    config = read_run_info(directory / "run.json")["config"]
    if config.get("protocol") != asdict(PROTOCOL):
        raise ValueError(
            "Not a RoboTHOR challenge-2021 run, or the protocol changed: %s"
            % directory)
    records = [row for _, row in read_episode_rows(directory / "episodes.jsonl")]
    ids = [row.episode_id for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode records")
    if not set(ids).issubset(config["selected_episode_ids"]):
        raise ValueError("Records do not belong to the recorded selection")
    method = config["method"]["method"]
    if any((row.benchmark, row.split, row.agent)
           != (PROTOCOL.benchmark, PROTOCOL.split, method) for row in records):
        raise ValueError("Record identity disagrees with the run configuration")
    return config, records


def _with_properties(result):
    """A reported row whose notes carry OSG Navigator's TF / NM labels."""
    tf, nm = properties_row(result.method)
    label = "TF=%s; NM=%s" % (tf, nm)
    notes = "%s. %s" % (result.notes, label) if result.notes else label
    return dataclasses.replace(result, notes=notes)


def _adapter_audit(directory):
    """The adapter's own telemetry, written beside the results by the CLI.

    Absent for a run produced some other way, or for merged shards; the report
    then says so rather than inventing zeros, because "the two readings never
    disagreed" and "nobody looked" are different claims.
    """
    path = Path(directory) / "adapter_audit.json"
    if not path.is_file():
        return {"recorded": False}
    payload = json.loads(path.read_text())
    payload["recorded"] = True
    return payload


def write_report(directory, *, reported=VALIDATION_RESULTS):
    """Recompute from episode rows; mark unfinished/subset/diagnostic results."""
    directory = Path(directory).expanduser()
    config, records = _read_run(directory)
    summary = summarise(records)
    method = config["method"]["method"]
    complete = (bool(config.get("full_split"))
                and len(records) == VAL_EPISODES
                and method != "diagnostic-stop"
                and summary.agent_errors == 0)
    title = ("Full RoboTHOR ObjectNav validation (1,800 episodes)" if complete
             else "NOT a full benchmark result (subset, incomplete, "
                  "diagnostic, or with agent errors)")
    tf, nm = properties_row(method, OUR_PROPERTIES)
    baselines = [_with_properties(r) for r in reported]
    audit = {
        "complete_published_split": complete,
        "episodes": len(records),
        "selected_episodes": len(config["selected_episode_ids"]),
        "published_split_episodes": VAL_EPISODES,
        "agent_errors": summary.agent_errors,
        "success_rule": config.get("success_rule"),
        "adapter_audit": _adapter_audit(directory),
        "zero_shortest_path_episodes":
            (config.get("dataset") or {}).get("zero_shortest_path_episodes"),
        "reference_build_match": config.get("reference_build_match"),
        "pose_source": config.get("pose_source"),
        "primary_metrics": ["SR", "SPL"],
        "supplementary_metrics": ["SoftSPL", "DTG (geodesic, m)", "steps",
                                  "wall_s"],
        "method_properties": {"training_free": tf, "non_metric": nm},
        "total_wall_s": sum(row.wall_s for row in records),
        "notes": (
            "SR and SPL are the only metrics any paper reports on RoboTHOR. "
            "SoftSPL and DTG are ours and have no published counterpart here. "
            "TF/NM are method labels from OSG Navigator's Table 2, not "
            "quantities. Comparison rows are validation-split only: ProcTHOR's "
            "65.2/28.8 and ProcTHOR-ZS's 55.0/23.7, which ESC and SG-Nav print "
            "in 'RoboTHOR' columns, are test-split numbers and are excluded."),
    }
    table = comparison_table(summary, baselines, our_label=method)
    payload = {"title": title, "audit": audit,
               "summary": json.loads(strict_json(asdict(summary), "summary")),
               "comparison_markdown": table}
    write_atomically(directory / "report.json",
                     strict_json(payload, "report", indent=2) + "\n")
    write_atomically(directory / "report.md", _markdown(title, audit, table))
    return payload


def _markdown(title, audit, table):
    """A short human-readable report; the JSON stays authoritative."""
    lines = ["# %s" % title, ""]
    if not audit["complete_published_split"]:
        lines += ["> This is **not** a published-split benchmark result. "
                  "%d of %d episodes; %d agent errors."
                  % (audit["episodes"], audit["published_split_episodes"],
                     audit["agent_errors"]), ""]
    lines += [table, "", "## Audit", "",
              "```json", json.dumps(audit, indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--zero-shot-only", action="store_true",
                        help="Drop the trained EmbCLIP reference row")
    args = parser.parse_args(argv)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.reported import (
        validation_results)
    payload = write_report(
        args.directory,
        reported=validation_results(include_trained=not args.zero_shot_only))
    print(payload["comparison_markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
