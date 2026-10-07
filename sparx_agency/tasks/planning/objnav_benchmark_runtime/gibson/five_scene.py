"""Five-scene Gibson smoke: first published episode per scene, sequential recordings."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import html
import json
from pathlib import Path
import subprocess
import sys
import time

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.results_io import read_episode_rows, strict_json, write_atomically
from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import STYLE
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.detector_options import add_detector_options, detector_flags
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.five_scene_console import (
    failure_line, recording_video, status_line, summary_table)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import source_fingerprint


def scene_record(root, scene):
    """The one scored episode of a scene directory; None for an attempt that aborted before scoring."""
    source = Path(root) / scene / "episodes.jsonl"
    if not source.exists():
        return None
    found = [row for _, row in read_episode_rows(source)]
    if not found:
        return None  # aborted infrastructure attempt, not a scored failure
    if len(found) != 1 or found[0].scene_id != scene:
        raise ValueError("Expected exactly the selected episode in %s" % source)
    return found[0]


def report(root):
    """Summarise actual completed scene rows, with links to every recording."""
    root = Path(root)
    records, rows = [], []
    for scene in SCENES:
        record = scene_record(root, scene)
        if record is None:
            continue
        records.append(record)
        policy = record.agent_info.get("policy", {})
        doors = policy.get("doors", {})
        changes = sum(event.get("previous") is not None and event["previous"] != event["label"]
                      for event in policy.get("room_label_history", []))
        rows.append({"scene": scene, "target": record.target_category,
                     "SR": int(record.success), "SPL": record.spl, "DTG_m": record.distance_to_goal_m,
                     "SoftSPL": record.soft_spl, "runtime_s": record.wall_s, "steps": record.steps,
                     "STOP": record.stop_called,
                     "termination": record.termination, "doors_confirmed": doors.get("confirmed", 0),
                     "door_detections": doors.get("detections", 0), "max_rooms": policy.get("max_rooms", 0),
                     "final_rooms": policy.get("rooms", 0), "label_changes": changes})
    if not records:
        return None
    summary = summarise(records)
    write_atomically(root / "summary.json", strict_json(asdict(summary), "five-scene summary", indent=2))
    with (root / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "<tr>" + "".join("<th>%s</th>" % html.escape(key) for key in rows[0]) + "<th>Recording/report</th></tr>"
    for row in rows:
        table += "<tr>" + "".join("<td>%s</td>" % html.escape("%.4f" % v if isinstance(v, float) else str(v))
                                    for v in row.values())
        table += '<td><a href="%s/index.html">Video, map, reasoning and graphs</a></td></tr>' % row["scene"]
    overall = summary.overall
    text = ('<!doctype html><meta charset="utf-8"><title>Gibson five-scene smoke</title><style>%s</style>'
            '<h1>Gibson: door-aware, revisable-room smoke</h1><p class="warning">%d/5 scenes completed; '
            'one first published episode per scene, NOT the 1,000-episode benchmark. '
            'SR may be credited at the step limit without STOP. Label changes include resets to unknown '
            'after room partition changes; they are not a semantic-accuracy score.</p>'
            '<p>Mean SR %.3f · SPL %.4f · DTG %s m · SoftSPL %.4f</p>'
            '<p><a href="metrics.csv">All metrics CSV</a> · <a href="summary.json">Summary JSON</a></p>'
            '<table>%s</table>') % (STYLE, len(records), overall.success_rate, overall.spl,
                                   overall.distance_to_goal_m, overall.soft_spl, table)
    write_atomically(root / "index.html", text)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_detector_options(parser, url_default="http://127.0.0.1:18092")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--allow-sim-version-mismatch", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Forwarded to every scene run; needed when GPU 0 also drives a desktop")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    fingerprint = source_fingerprint()
    status = {"source_sha256": fingerprint, "episodes_per_scene": 1, "completed": [], "failed": []}
    write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
    records = []
    for scene in SCENES:
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Source changed mid-campaign; refusing to mix algorithm versions")
        directory = args.output / scene
        directory.mkdir()
        command = [sys.executable, "-u", "-m", "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run",
                   "--scene", scene, "--limit", "1", "--record", "--seed", str(args.seed),
                   "--gpu-device", str(args.gpu_device),
                   "--episodes-dir", args.episodes_dir, "--scenes-dir", args.scenes_dir,
                   "--output", str(directory)] + detector_flags(args)
        if args.allow_sim_version_mismatch:
            command.append("--allow-sim-version-mismatch")
        if args.allow_shared_gpu:
            command.append("--allow-shared-gpu")
        status["running"] = scene
        write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
        print("Starting", scene, flush=True)
        started = time.monotonic()
        log = directory / "run.log"
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        job_s = time.monotonic() - started
        status["completed" if result.returncode == 0 else "failed"].append(scene)
        status["running"] = None
        write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
        report(args.output)
        record = scene_record(args.output, scene)
        if record is None:
            print(failure_line(scene, result.returncode, log), flush=True)
        else:
            records.append(record)
            print(status_line(record, recording_video(directory, record), job_s), flush=True)
    if records:
        table = summary_table(records)
        write_atomically(args.output / "summary.txt", table)
        print("\nFIVE-SCENE SUMMARY (%d/%d scenes scored; one first published episode each)\n%s"
              % (len(records), len(SCENES), table), flush=True)
    else:
        print("\nFIVE-SCENE SUMMARY: no scene produced a scored episode", flush=True)
    return int(bool(status["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())

