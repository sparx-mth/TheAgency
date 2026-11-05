"""Run MP3D scene by scene, each in its own process, then aggregate the rows.

Why a subprocess per scene rather than one long loop: habitat-sim holds a
rendering context and a scene's meshes for as long as the process lives, and
Matterport3D's buildings are large (the validation split's eleven average some
200 episodes each). One process per scene bounds that, and a scene that dies
costs one scene.

Each child writes a complete, self-contained results directory -- its own
``run.json``, ``episodes.jsonl`` and lock -- so a scene can be resumed or rerun
on its own, and :func:`aggregate` only ever summarises rows that actually
finished. It never fabricates a missing scene and never averages a partial one
into a total without saying so.

This is an orchestrator, not a second evaluator: the scoring, the logging and
the protocol all stay in ``run.py`` and the shared harness. The concurrency
(``scene_pool``) and the smoke cover (``smoke_plan``) are the HM3D campaign's,
reused verbatim: neither knows which dataset it is planning.

``--shard-episodes N`` cuts every scene into shards of at most N episodes, each
its own process, so no scene becomes the campaign's tail; the shards run on a
VRAM-gated pool whose size can be changed while it runs, over one or several
LLM servers, and each scene is merged as soon as its shards finish
(``sharded_campaign.py``). The aggregate is the same either way.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    read_episode_rows, strict_json, write_atomically)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign import (
    RESUME_ACCEPTANCE_FLAG, episode_plan, split_resume_acceptance)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.scene_pool import (
    run_concurrently)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.smoke_plan import (
    cover_plan, describe)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import (
    CATEGORIES, MP3DDataset, published_counts)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    APEXNAV_SUCCESS_DISTANCE_M, PROTOCOLS, protocol_for)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report import rescore

RUN_MODULE = "sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.run"


def scene_keys(episodes_dir, protocol):
    """Every scene of the split, in publisher order, without loading a mesh.

    One row per shard is enough to learn a scene's name, and it must not
    demand a complete split -- a development split has no published count.
    """
    dataset = MP3DDataset(episodes_dir, None, protocol, full=False,
                          require_scenes=False, episodes_per_scene=1)
    keys, seen = [], set()
    for episode in dataset.episodes.values():
        if episode.scene_key not in seen:
            seen.add(episode.scene_key)
            keys.append(episode.scene_key)
    return tuple(keys)


def smoke_cover(episodes_dir, protocol, scenes=None):
    """The minimal cover (``smoke_plan.cover_plan``) over the split's published episodes.

    One published episode per scene, the categories balanced so as many of the
    21 as the eleven scenes allow appear; the rest are reported, not dropped.

    Returns:
        ``(plan, account)``: ``{scene: [episode_id]}`` and its JSON-ready description.
    """
    dataset = MP3DDataset(episodes_dir, None, protocol, full=False, require_scenes=False)
    plan = cover_plan(dataset.episodes, CATEGORIES, scenes=scenes)
    return plan, describe(plan, dataset.episodes, CATEGORIES)


def scene_command(scene, output, *, version, episodes_dir, scenes_dir, limit=None,
                  split="val", extra=(), episode_ids=None):
    """The exact child command for one scene. A subset run, and labelled as one."""
    command = [sys.executable, "-m", RUN_MODULE, "--version", version,
               "--split", split, "--episodes-dir", str(episodes_dir),
               "--scenes-dir", str(scenes_dir), "--scenes", scene,
               "--output", str(output)]
    if episode_ids:
        command += ["--episode-ids"] + [str(i) for i in episode_ids]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command + list(extra)


def run_campaign(root, *, version, episodes_dir, scenes_dir, limit=None, scenes=None,
                 split="val", extra=(), resume=False, runner=subprocess.run, smoke=False,
                 jobs=1, episode_ids=None):
    """Run each scene in its own process and return what each one did.

    Args:
        root: Directory to hold one subdirectory per scene.
        version: ``"v1"``.
        limit: Episodes per scene, or ``None`` for all of them.
        split: Which split the children run.
        scenes: Restrict to these scenes; all of the split by default.
        extra: Further ``run.py`` arguments, passed to every child unchanged.
        resume: Skip a scene that already holds a finished ``summary.json``,
            and pass ``--resume`` to one that started but did not finish --
            with the ``--resume-accepting-changes KEY...`` of ``extra``, if
            any, which a scene that starts afresh never receives.
        runner: Injected for tests; must behave like ``subprocess.run``.
        smoke: Run the minimal cover instead (``smoke_plan``); the plan is
            written to ``<root>/smoke_plan.json``. Exclusive with ``limit``.
        episode_ids: Run exactly these published episodes (``<scene>/<index>``).
            Exclusive with ``smoke``, ``limit`` and ``scenes``.
        jobs: Scenes run at once. 1 runs them in order with the child's output
            on this console; more runs them through ``scene_pool``, each
            child's output in ``<root>/<scene>.console.log``.

    Returns:
        One record per scene: its command, its exit status and its directory.
    """
    if smoke and limit is not None:
        raise ValueError("--smoke chooses one episode per scene itself; it takes no --limit")
    if episode_ids and (smoke or limit is not None or scenes):
        raise ValueError("an explicit episode list names its own scenes; it takes no --smoke, --limit or --scenes")
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_for(version).with_split(split)
    selected = list(scenes) if scenes else list(scene_keys(episodes_dir, protocol))
    plan = {}
    if episode_ids:
        plan = episode_plan(episode_ids, selected)
        selected = list(plan)
    if smoke:
        plan, account = smoke_cover(episodes_dir, protocol, scenes=selected)
        write_atomically(root / "smoke_plan.json",
                         strict_json(dict(account, version=version, split=split),
                                     "MP3D smoke plan", indent=2))
    if int(jobs) < 1:
        raise ValueError("jobs must be at least 1, got %r" % (jobs,))
    shared, acceptance = split_resume_acceptance(extra)
    if acceptance and not resume:
        raise ValueError("%s is a resume's: it needs a campaign root that already holds scenes" % RESUME_ACCEPTANCE_FLAG)
    results, pending = [], []
    for scene in selected:
        output = root / scene
        finished = (output / "summary.json").is_file()
        if resume and finished:
            results.append({"scene": scene, "status": "already finished",
                            "returncode": 0, "output": str(output)})
            continue
        arguments = list(shared)
        if resume and output.exists():
            arguments += ["--resume"] + acceptance
        command = scene_command(scene, output, version=version, split=split,
                                episodes_dir=episodes_dir, scenes_dir=scenes_dir,
                                limit=limit, extra=arguments, episode_ids=plan.get(scene))
        if int(jobs) == 1:
            completed = runner(command)
            results.append({"scene": scene, "status": "ran",
                            "returncode": int(getattr(completed, "returncode", 1)),
                            "output": str(output), "command": command})
        else:
            pending.append((scene, command))
    if pending:
        returncodes = run_concurrently(pending, root, int(jobs), runner=runner)
        results.extend({"scene": scene, "status": "ran", "returncode": returncodes[scene],
                        "output": str(root / scene), "command": command}
                       for scene, command in pending)
        order = {scene: index for index, scene in enumerate(selected)}
        results.sort(key=lambda entry: order[entry["scene"]])
    write_atomically(root / "campaign.json",
                     strict_json({"version": version, "split": split, "limit": limit,
                                  "smoke": bool(smoke), "scenes": len(selected), "jobs": int(jobs),
                                  "runs": results},
                                 "MP3D campaign", indent=2))
    return results


def aggregate(root, *, episodes_dir=None, version=None, split="val"):
    """Summarise every scene that produced rows, and say which ones did not.

    A scene whose process died leaves no rows and is reported as incomplete
    rather than being quietly left out of the denominator. ``complete`` says
    whether every published episode of every published scene was scored.
    """
    root = Path(root).expanduser()
    records, per_scene, missing = [], [], []
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
    complete = bool(published
                    and expected_scenes is not None
                    and not missing
                    and len(per_scene) == expected_scenes == published["scenes"]
                    and len(records) == published["episodes"]
                    and all(scene["finished"] for scene in per_scene))
    # ApexNav's 0.2 m radius over the same rows (report.rescore: it can only add successes), and the
    # goal categories over every scene: the two breakdowns the papers' MP3D tables are read against.
    loose = rescore(records, APEXNAV_SUCCESS_DISTANCE_M)
    loose_overall = summarise(loose).overall
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
              "overall": asdict(summary.overall), "by_scene": per_scene,
              "by_category": [asdict(group) for group in summary.by_category],
              "apexnav_radius": {"success_distance_m": APEXNAV_SUCCESS_DISTANCE_M,
                                 "success_rate": loose_overall.success_rate, "spl": loose_overall.spl,
                                 "extra_successes": sum(a.success != b.success for a, b in zip(records, loose))}}
    write_atomically(root / "campaign_summary.json",
                     strict_json(report, "MP3D campaign summary", indent=2))
    with (root / "campaign_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_scene[0]))
        writer.writeheader()
        writer.writerows(per_scene)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--version", choices=sorted(PROTOCOLS), default="v1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes-dir", required=True, type=Path)
    parser.add_argument("--scenes-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Episodes per scene")
    parser.add_argument("--smoke", action="store_true",
                        help="The minimal cover: one explicit published episode per scene, categories "
                             "balanced (smoke_plan.py). A diagnostic subset, labelled as one; exclusive with --limit")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--episode-ids-file", type=Path,
                        help="A JSON list of published episode ids (<scene>/<index>): run exactly those, one "
                             "process per scene -- the same starts and goals for a re-run after a change")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Scenes run at once; above 1 each child logs to <root>/<scene>.console.log. "
                             "With --shard-episodes: shards at once, unless --jobs-file says otherwise")
    sharding = parser.add_argument_group("sharded campaign (--shard-episodes)")
    sharding.add_argument("--shard-episodes", type=int,
                          help="Cut every scene into shards of at most N episodes, one process each "
                               "(<root>/<scene>/shard_<k>/), merged per scene when its shards finish")
    sharding.add_argument("--jobs-file", type=Path,
                          help="A file whose integer overrides --jobs while the campaign runs; 0 drains it")
    sharding.add_argument("--llm-urls", nargs="+", default=(),
                          help="LLM servers to deal the shards over (a started shard keeps its own)")
    sharding.add_argument("--vram-cap", type=float, default=0.88,
                          help="Start a shard only while used VRAM + one renderer stays under this share of the card")
    sharding.add_argument("--renderer-mib", type=int, default=int(os.environ.get("HABITAT_VRAM_MIB") or 1400),
                          help="One renderer's VRAM budget for the start gate (default HABITAT_VRAM_MIB, else 1400)")
    # Anything after a bare "--" is passed through to every child run.py.
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    failed = []
    if not args.aggregate_only and args.shard_episodes:
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.shard_pool import VramGate
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.sharded_campaign import (
            option_value, run_sharded_campaign)
        if args.limit is not None or args.smoke or args.episode_ids_file:
            parser.error("--shard-episodes runs every episode of the selected scenes; it takes no "
                         "--limit, --smoke or --episode-ids-file")
        # A terminated campaign stops its shards first (the pool's finally), instead of leaving them
        # running against services the launcher is about to take down.
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        gate = VramGate(args.renderer_mib, args.vram_cap, gpu_device=int(option_value(extra, "--gpu-device") or 0))
        results = run_sharded_campaign(args.root, version=args.version, split=args.split,
                                       episodes_dir=args.episodes_dir, scenes_dir=args.scenes_dir,
                                       shard_episodes=args.shard_episodes, scenes=args.scenes, extra=extra,
                                       resume=args.resume, jobs=args.jobs, jobs_file=args.jobs_file,
                                       llm_urls=args.llm_urls, gate=gate)
        failed = [r["scene"] for r in results if r["returncode"] != 0]
    elif not args.aggregate_only:
        results = run_campaign(args.root, version=args.version, split=args.split,
                               episodes_dir=args.episodes_dir,
                               scenes_dir=args.scenes_dir, limit=args.limit,
                               scenes=args.scenes, extra=extra, resume=args.resume,
                               smoke=args.smoke, jobs=args.jobs,
                               episode_ids=json.loads(args.episode_ids_file.read_text())
                               if args.episode_ids_file else None)
        failed = [r["scene"] for r in results if r["returncode"] != 0]
    if failed:
        print("Scenes that did not exit cleanly%s: %s"
              % (" or are not merged" if args.shard_episodes else "", ", ".join(failed)))
    report = aggregate(args.root, episodes_dir=args.episodes_dir,
                       version=args.version, split=args.split)
    if report is None:
        print("No scene produced any episode rows")
        return 1
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_scene", "by_category")}, indent=2))
    return 0 if not failed and not report["scenes_without_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
