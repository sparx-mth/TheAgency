"""Run HM3D scene by scene, each in its own process, then aggregate the rows.

Why a subprocess per scene rather than one long loop: habitat-sim holds a
rendering context and a scene's meshes for as long as the process lives, and
HM3D validation is 20 or 36 buildings. Loading them all in one process is how a
run dies at episode 700 with an allocator failure and takes the whole sweep with
it. One process per scene bounds that, and a scene that dies costs one scene.

Each child writes a complete, self-contained results directory -- its own
``run.json``, ``episodes.jsonl`` and lock -- so a scene can be resumed or rerun
on its own, and :func:`aggregate` only ever summarises rows that actually
finished. It never fabricates a missing scene and never averages a partial one
into a total without saying so.

This is an orchestrator, not a second evaluator: the scoring, the logging and
the protocol all stay in ``run.py`` and the shared harness.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    read_episode_rows, strict_json, write_atomically)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import (
    HM3DDataset, published_counts)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import (
    PROTOCOLS, protocol_for)

RUN_MODULE = "sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run"


def scene_keys(episodes_dir, protocol):
    """Every scene of the split, in publisher order, without loading a mesh.

    One row per shard is enough to learn a scene's folder name, and it is all
    this may ask for: the train split is over seven million rows, which is a
    hundred seconds and several gigabytes to parse just to list 145 names. It
    must also not demand a complete split -- a development split has no
    published count, and requiring one would make ``--split train`` fail here
    before a single scene ran.
    """
    dataset = HM3DDataset(episodes_dir, None, protocol, full=False,
                          require_scenes=False, episodes_per_scene=1)
    keys, seen = [], set()
    for episode in dataset.episodes.values():
        if episode.scene_key not in seen:
            seen.add(episode.scene_key)
            keys.append(episode.scene_key)
    return tuple(keys)


def scene_command(scene, output, *, version, episodes_dir, scenes_dir, limit=None,
                  split="val", extra=()):
    """The exact child command for one scene. A subset run, and labelled as one."""
    command = [sys.executable, "-m", RUN_MODULE, "--version", version,
               "--split", split, "--episodes-dir", str(episodes_dir),
               "--scenes-dir", str(scenes_dir), "--scenes", scene,
               "--output", str(output)]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command + list(extra)


def run_campaign(root, *, version, episodes_dir, scenes_dir, limit=None, scenes=None,
                 split="val", extra=(), resume=False, runner=subprocess.run):
    """Run each scene in its own process and return what each one did.

    Args:
        root: Directory to hold one subdirectory per scene.
        version: ``"v1"`` or ``"v2"``.
        limit: Episodes per scene, or ``None`` for all of them.
        split: Which split the children run. It is passed explicitly rather
            than left to ``extra`` because the scene list is read from the
            split's own episode files.
        scenes: Restrict to these scenes; all of the split by default.
        extra: Further ``run.py`` arguments, passed to every child unchanged.
        resume: Skip a scene that already holds a finished ``summary.json``,
            and pass ``--resume`` to one that started but did not finish.
        runner: Injected for tests; must behave like ``subprocess.run``.

    Returns:
        One record per scene: its command, its exit status and its directory.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_for(version).with_split(split)
    selected = list(scenes) if scenes else list(scene_keys(episodes_dir, protocol))
    results = []
    for scene in selected:
        output = root / scene
        finished = (output / "summary.json").is_file()
        if resume and finished:
            results.append({"scene": scene, "status": "already finished",
                            "returncode": 0, "output": str(output)})
            continue
        arguments = list(extra)
        if resume and output.exists():
            arguments.append("--resume")
        command = scene_command(scene, output, version=version, split=split,
                                episodes_dir=episodes_dir, scenes_dir=scenes_dir,
                                limit=limit, extra=arguments)
        completed = runner(command)
        results.append({"scene": scene, "status": "ran",
                        "returncode": int(getattr(completed, "returncode", 1)),
                        "output": str(output), "command": command})
    write_atomically(root / "campaign.json",
                     strict_json({"version": version, "split": split, "limit": limit,
                                  "scenes": len(selected), "runs": results},
                                 "HM3D campaign", indent=2))
    return results


def aggregate(root, *, episodes_dir=None, version=None, split="val"):
    """Summarise every scene that produced rows, and say which ones did not.

    A scene whose process died leaves no rows and is reported as incomplete
    rather than being quietly left out of the denominator. The returned summary
    covers exactly the episodes that were scored, and ``complete`` says whether
    that was all of them.
    """
    root = Path(root).expanduser()
    records, per_scene, missing = [], [], []
    # A child that died before writing anything leaves no directory at all --
    # the worst case, and the one a directory scan cannot see. The campaign
    # manifest is the only record of what was asked for.
    requested = []
    manifest = root / "campaign.json"
    if manifest.is_file():
        requested = [entry["scene"]
                     for entry in json.loads(manifest.read_text()).get("runs", [])]
    missing.extend(scene for scene in requested if not (root / scene).is_dir())
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        source = directory / "episodes.jsonl"
        if not source.is_file():
            missing.append(directory.name)
            continue
        rows = [row for _, row in read_episode_rows(source)]
        if not rows:
            missing.append(directory.name)
            continue
        records.extend(rows)
        scene_summary = summarise(rows).overall
        per_scene.append({"scene": directory.name, "episodes": len(rows),
                          "success_rate": scene_summary.success_rate,
                          "spl": scene_summary.spl, "soft_spl": scene_summary.soft_spl,
                          "finished": (directory / "summary.json").is_file()})
    if not records:
        return None
    summary = summarise(records)
    protocol = (protocol_for(version).with_split(split) if version is not None
                else None)
    expected_scenes = None
    if episodes_dir is not None and protocol is not None:
        expected_scenes = len(scene_keys(episodes_dir, protocol))
    published = published_counts(protocol) if protocol is not None else {}
    # Every scene producing a row is not the same as every episode being run:
    # a smoke sweep, or a campaign where each scene died after one episode,
    # would otherwise be written out as a complete benchmark result with a
    # headline SR.
    complete = bool(published
                    and expected_scenes is not None
                    and not missing
                    and len(per_scene) == expected_scenes == published["scenes"]
                    and len(records) == published["episodes"]
                    and all(scene["finished"] for scene in per_scene))
    report = {"episodes": len(records), "scenes_with_rows": len(per_scene),
              "scenes_expected": expected_scenes,
              "episodes_published": published.get("episodes"),
              "scenes_published": published.get("scenes"),
              "split": split, "dataset_version": version,
              "agent": records[0].agent,
              "scenes_without_rows": sorted(set(missing)),
              "scenes_unfinished": [s["scene"] for s in per_scene
                                    if not s["finished"]],
              "complete": complete,
              "overall": asdict(summary.overall), "by_scene": per_scene}
    write_atomically(root / "campaign_summary.json",
                     strict_json(report, "HM3D campaign summary", indent=2))
    with (root / "campaign_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_scene[0]))
        writer.writeheader()
        writer.writerows(per_scene)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--version", choices=sorted(PROTOCOLS), default="v2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes-dir", required=True, type=Path)
    parser.add_argument("--scenes-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Episodes per scene")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    # Anything after a bare "--" is passed through to every child run.py.
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    failed = []
    if not args.aggregate_only:
        results = run_campaign(args.root, version=args.version, split=args.split,
                               episodes_dir=args.episodes_dir,
                               scenes_dir=args.scenes_dir, limit=args.limit,
                               scenes=args.scenes, extra=extra, resume=args.resume)
        failed = [r["scene"] for r in results if r["returncode"] != 0]
        if failed:
            print("Scenes that did not exit cleanly: %s" % ", ".join(failed))
    report = aggregate(args.root, episodes_dir=args.episodes_dir,
                       version=args.version, split=args.split)
    if report is None:
        print("No scene produced any episode rows")
        return 1
    print(json.dumps({k: v for k, v in report.items() if k != "by_scene"}, indent=2))
    # Exit on what actually happened, not on whether the split is complete: a
    # deliberate subset succeeds, and a sweep where every scene crashed fails,
    # however it was invoked.
    return 0 if not failed and not report["scenes_without_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
