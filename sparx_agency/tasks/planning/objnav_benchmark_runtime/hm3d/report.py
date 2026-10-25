"""Auditable HM3D reports and strict shard merging; no simulator or model imports.

Two tables come out of one run, because two published protocols disagree about
one number. Ours is scored under habitat-lab's own success radius (0.1 m to the
nearest goal view point). ApexNav's released configs use 0.2 m, so the same
episodes are **re-scored** at 0.2 m from the rows already on disk -- no second
run, no second simulation, and the two numbers cannot drift apart. Re-scoring
upward can only add successes, never remove them, so the 0.2 m row is an upper
bound on the 0.1 m row by construction.

Nothing here recomputes a distance or re-reads a dataset: it reads
``episodes.jsonl``, applies the published arithmetic, and says what it did.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.comparison import comparison_table
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    read_episode_rows, strict_json, write_atomically)
from sparx_agency.tasks.planning.objnav_benchmark.run_info import read_run_info
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import (
    PUBLISHED_VAL, published_counts)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import (
    APEXNAV_SUCCESS_DISTANCE_M, protocol_for)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.sota import reported_results


def _read_run(directory):
    """The run's configuration and its episode rows, checked to belong together."""
    directory = Path(directory).expanduser()
    config = read_run_info(directory / "run.json")["config"]
    protocol = config.get("protocol") or {}
    version = protocol.get("dataset_version")
    if version not in PUBLISHED_VAL:
        raise ValueError("Not an HM3D ObjectNav run: %s" % directory)
    expected = protocol_for(version).with_split(protocol.get("split", "val"))
    if protocol != asdict(expected):
        raise ValueError(
            "%s ran a different HM3D protocol than this module defines; compare "
            "run.json's protocol block before trusting any number" % directory)
    records = [row for _, row in read_episode_rows(directory / "episodes.jsonl")]
    if not records:
        raise ValueError("No episode rows in %s" % directory)
    ids = [row.episode_id for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode records")
    selected = config.get("selected_episode_ids") or []
    if not set(ids).issubset(selected):
        raise ValueError("Records do not belong to the recorded selection")
    agent = config["method"]["method"]
    if any((row.benchmark, row.split) != (expected.benchmark, expected.split)
           or row.agent != agent for row in records):
        raise ValueError("Record identity disagrees with the run configuration")
    return config, expected, records


def rescore(records, success_distance_m):
    """The same episodes under another success radius, from the rows alone.

    Args:
        records: The run's :class:`EpisodeRecord` rows.
        success_distance_m: The radius to apply, metres. Habitat's rule is
            ``STOP and dT < radius``, strictly less.

    Returns:
        New records with ``success`` and ``spl`` recomputed. ``soft_spl`` is
        untouched: habitat-lab gates it on neither success nor STOP, so the
        radius does not enter it. An agent-error episode stays a failure --
        a crash is not made a success by a looser radius.
    """
    rescored = []
    for row in records:
        distance = row.distance_to_goal_m  # None is habitat's inf
        success = bool(row.agent_error is None and row.stop_called
                       and distance is not None
                       and distance < success_distance_m)
        l, p = row.shortest_path_m, row.path_length_m
        ratio = 1.0 if max(l, p) <= 0 else l / max(l, p)
        rescored.append(replace(row, success=success,
                                spl=float(success) * ratio))
    return rescored


def _audit(config, protocol, records, summary):
    """What a reader must know before quoting the number above it."""
    expected = published_counts(protocol)
    dataset = config.get("dataset") or {}
    complete = (bool(expected)
                and bool(config.get("full_split"))
                and len(records) == expected["episodes"]
                and dataset.get("scene_count") == expected["scenes"]
                and config["method"]["method"] != "diagnostic-stop"
                and not summary.agent_errors)
    unreachable = config.get("excluded_unreachable_episode_ids") or []
    return {
        "full_split_completed": complete,
        "split": protocol.split,
        "published_benchmark_split": bool(expected),
        "episodes_scored": len(records),
        "episodes_selected": len(config.get("selected_episode_ids") or []),
        "episodes_published": expected.get("episodes"),
        "scenes_published": expected.get("scenes"),
        "scenes_loaded": dataset.get("scene_count"),
        "agent_errors": summary.agent_errors,
        "starts_validated": bool(config.get("starts_validated")),
        "excluded_unreachable_episodes": len(unreachable),
        "navmesh": protocol.navmesh,
        "success_distance_m": protocol.success_distance_m,
        "scene_assets_hashed": dataset.get("scene_assets_hashed"),
        "reference_habitat_sim_version_match":
            config.get("reference_habitat_sim_version_match"),
        "primary_metrics": ["SR", "SPL", "DTG (geodesic to the nearest goal "
                            "view point, m)"],
        "supplementary_metrics": ["SoftSPL", "steps", "wall_s"],
        "total_wall_s": sum(row.wall_s for row in records),
        "notes": "SR/SPL/SoftSPL/DTG are means over every scored episode, "
                 "failures included. DTG is habitat-lab's distance_to_goal, so "
                 "it is not OSG's under-specified DTG. No claim of bit-exact "
                 "reproduction of any paper's pipeline is made.",
    }


def write_report(directory, *, include_osg=False):
    """Write ``comparison.md`` and ``audit.json`` beside a finished run."""
    directory = Path(directory).expanduser()
    config, protocol, records = _read_run(directory)
    summary = summarise(records)
    audit = _audit(config, protocol, records, summary)
    loose = rescore(records, APEXNAV_SUCCESS_DISTANCE_M)
    loose_summary = summarise(loose)
    gained = sum(a.success != b.success for a, b in zip(records, loose))
    audit["apexnav_radius_extra_successes"] = gained
    audit["apexnav_radius_success_rate"] = loose_summary.overall.success_rate

    title = ("Full HM3D-%s %s validation" % (protocol.dataset_version, protocol.split)
             if audit["full_split_completed"] else
             "NOT a full benchmark result (subset, incomplete, or diagnostic)")
    text = "# %s\n\n" % title
    text += ("Scored under habitat-lab's own ObjectNav rule: STOP, with the "
             "geodesic distance to the nearest goal **view point** below "
             "%.2f m. SR, SPL, SoftSPL and DTG are means over every scored "
             "episode, failures included.\n\n" % protocol.success_distance_m)
    if audit["published_benchmark_split"]:
        text += comparison_table(summary, reported_results(
            protocol.benchmark, protocol.split, include_osg=include_osg))
    else:
        # The papers report validation. Printing our development rows under
        # their table would invite exactly the comparison it is not.
        text += ("This run is on the **%s** split, which no compared paper "
                 "reports, so there is no table to put it under. Development "
                 "numbers belong in an experiment log, not beside a benchmark."
                 "\n\n" % protocol.split)
        text += comparison_table(summary, ())
    text += ("\n\n## The same episodes at ApexNav's 0.2 m success radius\n\n"
             "ApexNav's released configs loosen the success radius from "
             "habitat-lab's 0.1 m to 0.2 m, which can only add successes. These "
             "are the **same %d episodes**, re-scored from the rows on disk -- "
             "not a second run. %d episode(s) change from failure to success.\n\n"
             % (len(records), gained))
    text += comparison_table(
        loose_summary,
        reported_results(protocol.benchmark, protocol.split, include_osg=False)
        if audit["published_benchmark_split"] else (),
        our_label=records[0].agent + " @0.2 m")
    text += ("\n\n**Comparison caveats.** The published rows are transcribed "
             "reference values, not paired re-evaluations, and the papers do "
             "not share one protocol: ApexNav's own results are at a 0.2 m "
             "success radius against habitat-lab's 0.1 m, most of its HM3D-v1 "
             "baseline rows are quotes from other papers rather than its own "
             "measurements, and two papers print different numbers for the same "
             "baseline. Ground-truth pose and depth do not mean ground-truth "
             "semantics: our detections are predicted. Each row's own note says "
             "which of these applies to it.\n\n")
    text += "## Audit\n\n" + "\n".join(
        "- **%s:** %s" % item for item in audit.items()) + "\n"
    write_atomically(directory / "comparison.md", text)
    write_atomically(directory / "audit.json",
                     strict_json(audit, "HM3D audit", indent=2))
    return audit


def merge_runs(directories, output):
    """Merge only complete, disjoint shards of one identical experiment."""
    runs = [_read_run(directory) for directory in directories]
    if not runs:
        raise ValueError("No shard directories supplied")
    base, protocol, _ = runs[0]
    comparable = ("protocol", "runtime", "method", "seed", "dataset", "shards",
                  "limit", "scenes", "reference_habitat_sim_version_match",
                  "excluded_unreachable_episode_ids")
    records, shard_ids = [], set()
    for config, _, rows in runs:
        if any(config.get(key) != base.get(key) for key in comparable):
            raise ValueError("Shards disagree on data, protocol, method, or runtime")
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
        raise ValueError("Shards overlap; each episode must appear exactly once")
    # Each shard carries full_split=False, because on its own it is a subset.
    # Their union is the published split exactly when it covers every episode
    # with nothing limited, filtered or excluded -- so decide it here rather
    # than inheriting a flag that was never about the merged result.
    published = PUBLISHED_VAL[protocol.dataset_version]["episodes"]
    merged_full = (len(ids) == published and not base.get("limit")
                   and not base.get("scenes")
                   and not base.get("excluded_unreachable_episode_ids"))
    config = dict(base, selected_episode_ids=sorted(ids), shards=1, shard_index=0,
                  full_split=merged_full,
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
    parser.add_argument("--include-osg", action="store_true",
                        help="Also print OSG's 400-episode rows, which are on a "
                             "different protocol and are not comparable")
    args = parser.parse_args(argv)
    if args.merge_output:
        audit = merge_runs(args.runs, args.merge_output)
    elif len(args.runs) == 1:
        audit = write_report(args.runs[0], include_osg=args.include_osg)
    else:
        parser.error("Multiple runs require --merge-output")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
