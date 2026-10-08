"""Five-scene Gibson campaign: the first N published val episodes per scene, sequential recordings.

Each scene is one recorded ``gibson.run`` job over the first ``--episodes-per-scene``
episodes of that scene in official ``val`` split order. Episodes are reported the
moment they are scored (``EPISODE COMPLETE`` + ``RUNNING MEAN``), and
``benchmark_results.json`` / ``benchmark_results.csv`` are rewritten after each one.
Stair traversal is explicitly forbidden: every job receives a written
``policy_config.json`` with ``allow_stair_traversal: false`` and the frozen run
configuration is checked to say so.
"""
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
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.five_scene_stream import (
    EpisodeTail, result_row, running_mean_line, write_benchmark_results)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import source_fingerprint


def scene_records(root, scene, expected):
    """The scored episodes of a scene directory, in finishing order; empty for an attempt that aborted before scoring.

    Raises:
        ValueError: A row from another scene, or more rows than the job was asked for.
    """
    source = Path(root) / scene / "episodes.jsonl"
    if not source.exists():
        return []
    found = [row for _, row in read_episode_rows(source)]
    if len(found) > expected or any(row.scene_id != scene for row in found):
        raise ValueError("Expected at most the %d selected %s episodes in %s" % (expected, scene, source))
    return found


def scene_record(root, scene):
    """The one scored episode of a single-episode scene directory; None when none was scored."""
    found = scene_records(root, scene, 1)
    return found[0] if found else None


def assert_stairs_forbidden(directory):
    """Refuse to count a scene job whose frozen configuration allows stair traversal."""
    run = json.loads((Path(directory) / "run.json").read_text())
    allowed = run.get("config", {}).get("method", {}).get("allow_stair_traversal")
    if allowed is not False:
        raise RuntimeError("%s/run.json records allow_stair_traversal=%r; the campaign requires False"
                           % (directory, allowed))


def report(root, episodes_per_scene=1):
    """Summarise actual completed scene rows, with links to every recording."""
    root = Path(root)
    records, rows = [], []
    for scene in SCENES:
        for record in scene_records(root, scene, episodes_per_scene):
            records.append(record)
            _append_report_row(rows, scene, record)
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
    text = ('<!doctype html><meta charset="utf-8"><title>Gibson five-scene campaign</title><style>%s</style>'
            '<h1>Gibson: door-aware, revisable-room campaign</h1><p class="warning">%d episodes completed; '
            'the first %d published episode(s) per scene, NOT the 1,000-episode benchmark. '
            'SR may be credited at the step limit without STOP. Label changes include resets to unknown '
            'after room partition changes; they are not a semantic-accuracy score.</p>'
            '<p>Mean SR %.3f · SPL %.4f · DTG %s m · SoftSPL %.4f</p>'
            '<p><a href="metrics.csv">All metrics CSV</a> · <a href="summary.json">Summary JSON</a></p>'
            '<table>%s</table>') % (STYLE, len(records), episodes_per_scene, overall.success_rate, overall.spl,
                                   overall.distance_to_goal_m, overall.soft_spl, table)
    write_atomically(root / "index.html", text)
    return rows


def _append_report_row(rows, scene, record):
    policy = record.agent_info.get("policy", {})
    doors = policy.get("doors", {})
    changes = sum(event.get("previous") is not None and event["previous"] != event["label"]
                  for event in policy.get("room_label_history", []))
    rows.append({"scene": scene, "episode_id": record.episode_id, "target": record.target_category,
                 "SR": int(record.success), "SPL": record.spl, "DTG_m": record.distance_to_goal_m,
                 "SoftSPL": record.soft_spl, "runtime_s": record.wall_s, "steps": record.steps,
                 "STOP": record.stop_called,
                 "termination": record.termination, "doors_confirmed": doors.get("confirmed", 0),
                 "door_detections": doors.get("detections", 0), "max_rooms": policy.get("max_rooms", 0),
                 "final_rooms": policy.get("rooms", 0), "label_changes": changes})


def run_scene_job(args, scene, directory, policy_config, on_episode):
    """Run one recorded ``gibson.run`` job for ``scene`` and report each episode as it is scored.

    Returns:
        ``(exit code, job wall seconds)``.
    """
    command = [sys.executable, "-u", "-m", "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run",
               "--scene", scene, "--limit", str(args.episodes_per_scene), "--record", "--seed", str(args.seed),
               "--gpu-device", str(args.gpu_device), "--video-fps", str(args.video_fps),
               "--policy-config", str(policy_config),
               "--episodes-dir", args.episodes_dir, "--scenes-dir", args.scenes_dir,
               "--output", str(directory)] + detector_flags(args)
    if args.explorer is not None:
        command += ["--explorer", args.explorer]
    if args.allow_sim_version_mismatch:
        command.append("--allow-sim-version-mismatch")
    if args.allow_shared_gpu:
        command.append("--allow-shared-gpu")
    started = time.monotonic()
    tail = EpisodeTail(directory)
    with (directory / "run.log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            finished = process.poll() is not None
            for record in tail.poll():
                on_episode(record)
            if finished:
                break
            time.sleep(args.poll_s)
    return process.returncode, time.monotonic() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_detector_options(parser, url_default="http://127.0.0.1:18092")
    parser.add_argument("--episodes-per-scene", type=int, default=1,
                        help="The first N published val episodes of every scene, in split order")
    parser.add_argument("--explorer", choices=("frontier", "falcon"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--poll-s", type=float, default=5.0, help="How often a running job's episodes.jsonl is read")
    parser.add_argument("--allow-sim-version-mismatch", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Forwarded to every scene run; needed when GPU 0 also drives a desktop")
    args = parser.parse_args(argv)
    if not 1 <= args.episodes_per_scene <= 200:
        parser.error("--episodes-per-scene must be between 1 and 200")
    args.output.mkdir(parents=True, exist_ok=False)
    per_scene, total = args.episodes_per_scene, args.episodes_per_scene * len(SCENES)
    policy_config = args.output / "policy_config.json"
    write_atomically(policy_config, strict_json({"allow_stair_traversal": False}, "policy config", indent=2))
    fingerprint = source_fingerprint()
    status = {"source_sha256": fingerprint, "episodes_per_scene": per_scene, "episodes_total": total,
              "allow_stair_traversal": False, "split": "val", "scenes": list(SCENES), "seed": args.seed,
              "explorer": args.explorer, "completed": [], "failed": []}
    write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
    records, rows = [], []

    def on_episode(record, directory):
        records.append(record)
        video = recording_video(directory, record)
        rows.append(result_row(len(records), record, video))
        write_benchmark_results(args.output, rows, records, total, status)
        print(status_line(record, video), flush=True)
        print(running_mean_line(records, total), flush=True)

    for scene in SCENES:
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Source changed mid-campaign; refusing to mix algorithm versions")
        directory = args.output / scene
        directory.mkdir()
        status["running"] = scene
        write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
        print("Starting %s (%d episodes)" % (scene, per_scene), flush=True)
        before = len(records)
        code, job_s = run_scene_job(args, scene, directory, policy_config,
                                    lambda record, directory=directory: on_episode(record, directory))
        scored = len(records) - before
        if scored:
            assert_stairs_forbidden(directory)
        status["completed" if code == 0 else "failed"].append(scene)
        status["running"] = None
        write_atomically(args.output / "campaign.json", strict_json(status, "campaign", indent=2))
        report(args.output, per_scene)
        if not scored:
            print(failure_line(scene, code, directory / "run.log"), flush=True)
        else:
            print("SCENE DONE        scene=%s  scored=%d/%d  exit=%d  job=%.0f s" % (scene, scored, per_scene, code, job_s),
                  flush=True)
    if records:
        table = summary_table(records)
        write_atomically(args.output / "summary.txt", table)
        print("\nFIVE-SCENE SUMMARY (%d/%d episodes scored; the first %d published episode(s) per scene)\n%s"
              % (len(records), total, per_scene, table), flush=True)
    else:
        print("\nFIVE-SCENE SUMMARY: no scene produced a scored episode", flush=True)
    return int(bool(status["failed"]) or len(records) != total)


if __name__ == "__main__":
    raise SystemExit(main())
