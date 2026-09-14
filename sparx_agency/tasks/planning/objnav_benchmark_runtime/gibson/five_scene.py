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

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.results_io import read_episode_rows, strict_json, write_atomically
from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import STYLE
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import source_fingerprint


def report(root):
    """Summarise actual completed scene rows, with links to every recording."""
    root = Path(root)
    records, rows = [], []
    for scene in SCENES:
        source = root / scene / "episodes.jsonl"
        if not source.exists():
            continue
        found = [row for _, row in read_episode_rows(source)]
        if len(found) != 1 or found[0].scene_id != scene:
            raise ValueError("Expected exactly the selected episode in %s" % source)
        record = found[0]
        records.append(record)
        policy = record.agent_info.get("policy", {})
        doors = policy.get("doors", {})
        changes = sum(event.get("previous") is not None and event["previous"] != event["label"]
                      for event in policy.get("room_label_history", []))
        rows.append({"scene": scene, "target": record.target_category,
                     "SR": int(record.success), "SPL": record.spl, "DTG_m": record.distance_to_goal_m,
                     "SoftSPL": record.soft_spl, "steps": record.steps, "STOP": record.stop_called,
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
    parser.add_argument("--detector-url", default="http://127.0.0.1:18092")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-sim-version-mismatch", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    fingerprint = source_fingerprint()
    status = {"source_sha256": fingerprint, "episodes_per_scene": 1, "completed": [], "failed": []}
    write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
    for scene in SCENES:
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Source changed mid-campaign; refusing to mix algorithm versions")
        directory = args.output / scene
        directory.mkdir()
        command = [sys.executable, "-u", "-m", "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run",
                   "--scene", scene, "--limit", "1", "--record", "--seed", str(args.seed),
                   "--episodes-dir", args.episodes_dir, "--scenes-dir", args.scenes_dir,
                   "--detector-url", args.detector_url, "--output", str(directory)]
        if args.allow_sim_version_mismatch:
            command.append("--allow-sim-version-mismatch")
        status["running"] = scene
        write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
        print("Starting", scene, flush=True)
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        status["completed" if result.returncode == 0 else "failed"].append(scene)
        status["running"] = None
        write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
        report(args.output)
        print("Finished", scene, "exit", result.returncode, flush=True)
    return int(bool(status["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())

