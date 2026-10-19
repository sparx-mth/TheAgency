"""Frozen paired campaign on distinct generated Gibson training buildings.

Preflights and locks every execution configuration before the first simulator
reset. Runs sequentially, alternates explorer order by building, records both,
and retains failures. No policy tuning or replacement building selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import paired_comparison, summarise
from sparx_agency.tasks.planning.objnav_benchmark.results_io import read_episode_rows
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.compare_explorers import summarize
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.development_dataset import SCHEMA, SPLIT
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.detector_options import add_detector_options, detector_flags
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_SCHEMA
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import source_fingerprint
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import run_development


def jobs_for(scenes, explorers=("frontier", "falcon")):
    """Predeclared counterbalanced order, unrelated to any navigation outcome."""
    if not explorers or len(explorers) != len(set(explorers)) or not set(explorers) <= {"frontier", "falcon"}:
        raise ValueError("Need distinct, supported explorers")
    jobs = []
    for index, scene in enumerate(scenes):
        order = tuple(explorers) if index % 2 == 0 else tuple(reversed(explorers))
        jobs.extend((backend, scene) for backend in order)
    return jobs


def command_for(args, backend, scene):
    command = ["--manifest", str(args.manifest.resolve()), "--scene", scene,
               "--output", str(args.output / backend / scene), "--explorer", backend,
               "--seed", str(args.seed), "--record", "--video-fps", str(args.video_fps)]
    command += detector_flags(args)
    if getattr(args, "record_first", False):
        command.append("--record-first")
    if args.allow_sim_version_mismatch:
        command.append("--allow-sim-version-mismatch")
    if getattr(args, "allow_shared_gpu", False):
        command.append("--allow-shared-gpu")
    if args.policy_config:
        command += ["--policy-config", str(args.policy_config)]
    return command


def freeze(args, data):
    locks = args.output / "locks"
    locks.mkdir()
    source = source_fingerprint()
    identities = {}
    jobs = jobs_for(data["scenes"], getattr(args, "explorers", ("frontier", "falcon")))
    for backend, scene in jobs:
        arguments = run_development.parser().parse_args(command_for(args, backend, scene))
        env, _, config = run_development.prepare(arguments)
        env.close()
        if config["source_sha256"] != source:
            raise RuntimeError("Source changed during campaign preflight")
        comparable = dict(config)
        method = dict(config["method"])
        method.pop("method", None)
        method.pop("local_exploration", None)
        method.pop("falcon_running", None)
        method.pop("falcon_source", None)
        method["adaptation"] = dict(method["adaptation"])
        method["adaptation"].pop("local_exploration", None)
        comparable["method"] = method
        if scene in identities and identities[scene] != comparable:
            raise RuntimeError("Explorer pair differs in models, observations, protocol or data")
        identities[scene] = comparable
        with (locks / (backend + "-" + scene + ".json")).open("x") as stream:
            json.dump(config, stream, indent=2, allow_nan=False)
            stream.write("\n")
    recordings = len(jobs) if getattr(args, "record_first", False) else len(data["episodes"]) * len({job[0] for job in jobs})
    manifest = {"role": "frozen_generated_training_development", "split": data.get("split", SPLIT),
                "held_out_claim": False, "policy_tuning": False, "source_sha256": source,
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "jobs": jobs, "buildings": data["scenes"],
                "recordings_expected": recordings, "video_fps": args.video_fps,
                "seed": args.seed, "detector_url": args.detector_url}
    with (args.output / "campaign.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def finish(root, data):
    """Reuse metric aggregation and write an accessible paired video gallery."""
    if data.get("schema") == MULTIFLOOR_SCHEMA:
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_report import write_multifloor_report
        return write_multifloor_report(root, data)
    report = summarize(root, role="frozen_generated_training_development", require_falcon_activity=False)
    expected = {scene + "/000000" for scene in data["scenes"]}
    records = {}
    for backend in ("frontier", "falcon"):
        records[backend] = [row for path in sorted((root / backend).glob("*/episodes.jsonl"))
                            for _, row in read_episode_rows(path)]
        if {r.episode_id for r in records[backend]} != expected or len(records[backend]) != len(expected):
            raise RuntimeError("Missing or duplicate distinct-building results for " + backend)
    stats = {backend: asdict(summarise(rows)) for backend, rows in records.items()}
    stats["paired"] = asdict(paired_comparison(records["falcon"], records["frontier"]))
    (root / "statistics.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    videos = []
    lines = ["<!doctype html><meta charset='utf-8'><title>15-building ObjectNav comparison</title>",
             "<style>body{font:16px sans-serif;background:#181818;color:#eee;margin:24px}a{color:#8cf}.pair{display:flex;gap:16px;flex-wrap:wrap}video{max-width:100%;width:720px}section{margin-bottom:40px}</style>",
             "<h1>Gibson distinct-building ObjectNav comparison</h1>",
             "<p>Generated training-development starts, fixed before evaluation. Not published validation or a held-out claim. Both explorers use the same detector, sensors, actions and 500-action cap.</p>",
             "<p><a href='COMPARISON.md'>Per-episode results</a> | <a href='statistics.json'>Confidence intervals and paired statistics</a> | <a href='coverage.csv'>Observed-area curves</a></p>"]
    for scene in data["scenes"]:
        lines += ["<section><h2>" + html.escape(scene) + "</h2><div class='pair'>"]
        for backend in ("frontier", "falcon"):
            record = next(r for r in records[backend] if r.scene_id == scene)
            key = hashlib.sha256(record.episode_id.encode()).hexdigest()[:12]
            path = Path(backend) / scene / "recordings" / key / "video.mp4"
            if not (root / path).is_file() or (root / path).stat().st_size == 0:
                raise RuntimeError("Missing recording: " + str(path))
            videos.append({"explorer": backend, "scene": scene, "episode_id": record.episode_id, "path": str(path)})
            label = "%s | target: %s | success: %s | SPL %.3f | actions %d" % (backend, record.target_category, record.success, record.spl, record.steps)
            lines.append("<div><h3>%s</h3><video controls preload='none' src='%s'></video></div>" % (html.escape(label), path.as_posix()))
        lines.append("</div></section>")
    (root / "recordings.json").write_text(json.dumps(videos, indent=2) + "\n")
    (root / "index.html").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_detector_options(parser, required=True)
    parser.add_argument("--job-timeout-s", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--explorers", nargs="+", choices=("frontier", "falcon"), default=["frontier", "falcon"])
    parser.add_argument("--record-first", action="store_true")
    parser.add_argument("--allow-sim-version-mismatch", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Explicit operator authorization; forwarded into every preflight and frozen job")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.job_timeout_s <= 0:
        parser.error("--job-timeout-s must be positive")
    data = json.loads(args.manifest.read_text())
    if data.get("schema") not in (SCHEMA, MULTIFLOOR_SCHEMA) or len(data["scenes"]) != len(set(data["scenes"])):
        raise ValueError("Need a distinct-building generated training manifest")
    if data["schema"] == SCHEMA and set(args.explorers) != {"frontier", "falcon"}:
        raise ValueError("The legacy comparison requires both explorers; single-explorer campaigns use --multistory data")
    if args.summarize_only:
        finish(args.output, data)
        return
    if args.resume:
        campaign = json.loads((args.output / "campaign.json").read_text())
        if campaign["source_sha256"] != source_fingerprint() or campaign["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise RuntimeError("Source or generated episodes changed; cannot resume this comparison")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        campaign = freeze(args, data)
    jobs = jobs_for(data["scenes"], args.explorers)
    if [list(job) for job in jobs] != [list(job) for job in campaign["jobs"]]:
        raise RuntimeError("Requested job selection differs from frozen campaign")
    for number, (backend, scene) in enumerate(jobs, 1):
        if campaign["source_sha256"] != source_fingerprint():
            raise RuntimeError("Source changed during the frozen comparison")
        destination = args.output / backend / scene
        destination.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-m", "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run_development"]
        command += command_for(args, backend, scene)
        command += ["--expect-config", str(args.output / "locks" / (backend + "-" + scene + ".json"))]
        if args.resume and (destination / "run.json").exists():
            command.append("--resume")
        print("%d/%d %s %s" % (number, len(jobs), backend, scene), flush=True)
        (args.output / "progress.json").write_text(json.dumps({"job": number, "total": len(jobs), "explorer": backend, "scene": scene}))
        with (destination / "console.log").open("a") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=args.job_timeout_s)
    finish(args.output, data)
    print("Recordings and results:", args.output / "index.html", flush=True)


if __name__ == "__main__":
    main()

