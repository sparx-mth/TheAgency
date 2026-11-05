"""Run the MP3D split as shards on a pool of workers, and merge each scene as soon as its shards finish.

``campaign.py --shard-episodes N`` comes here instead of starting one process
per scene. The shards are ``shard_plan.py``'s, the pool is ``shard_pool.py``'s,
and the children are the same ``mp3d.run`` command lines the unsharded campaign
starts, plus ``--shards K --shard-index k`` -- so the scoring, the protocol and
the records are untouched, and a scene whose shards are merged reads to the
aggregate exactly as an unsharded scene does.

Resuming a root (``resume=True``) skips merged scenes and finished shards,
passes ``--resume`` (with any ``--resume-accepting-changes``) to started ones,
starts the rest, and merges every scene whose shards are then all finished.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import time

from sparx_agency.tasks.planning.objnav_benchmark.results_io import strict_json, write_atomically
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign import (
    RESUME_ACCEPTANCE_FLAG, split_resume_acceptance)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import MP3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.shard_plan import (
    assign_llm_urls, merge_scene, plan_shards, recorded_llm_url, scene_merged, write_plan)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.shard_pool import (
    PoolJob, VramGate, read_cap, run_pool)

#: Child arguments that choose episodes themselves; a sharded campaign does the choosing.
SELECTION_FLAGS = ("--limit", "--shards", "--shard-index", "--episode-ids", "--category", "--shuffle")

#: The pool's periodic record of how far the campaign is, read by ``campaign_status.py``.
HEARTBEAT_FILE = "campaign_heartbeat.jsonl"


def option_value(arguments, flag):
    """The value after ``flag`` in ``arguments``, or ``None``."""
    arguments = list(arguments)
    return arguments[arguments.index(flag) + 1] if flag in arguments[:-1] else None


def scene_episode_counts(episodes_dir, protocol, scenes=None, episodes_per_scene=None):
    """``{scene: episodes}`` in the publisher's order: what each scene's children will select between them."""
    dataset = MP3DDataset(episodes_dir, None, protocol, full=False, require_scenes=False,
                          scenes=list(scenes) if scenes else None, episodes_per_scene=episodes_per_scene)
    counts = OrderedDict()
    for episode in dataset.episodes.values():
        counts[episode.scene_key] = counts.get(episode.scene_key, 0) + 1
    if scenes:
        missing = [scene for scene in scenes if scene not in counts]
        if missing:
            raise ValueError("the split holds no episodes of %s" % ", ".join(missing))
        counts = OrderedDict((scene, counts[scene]) for scene in scenes)
    return counts


def done_episodes(root, plan):
    """Rows finished so far over the planned shards: a merged scene's count, else each shard's progress."""
    root, done = Path(root), 0
    for scene in dict.fromkeys(job.scene for job in plan):
        summary = root / scene / "summary.json"
        if summary.is_file():
            done += int(json.loads(summary.read_text())["overall"]["n_episodes"])
            continue
        for job in (j for j in plan if j.scene == scene):
            progress = job.directory(root) / "progress.json"
            if progress.is_file():
                try:
                    done += int(json.loads(progress.read_text())["done"])
                except (ValueError, KeyError):
                    pass    # being rewritten this instant; the next beat counts it
    return done


def run_sharded_campaign(root, *, version, episodes_dir, scenes_dir, shard_episodes, split="val", scenes=None,
                         extra=(), resume=False, jobs=1, jobs_file=None, llm_urls=(), gate=None,
                         popen=subprocess.Popen, sleep=time.sleep, log=print):
    """Run every shard of the split (or of ``scenes``) and merge each finished scene.

    Args:
        root: The campaign root; ``<root>/<scene>/shard_<k>/`` per shard, ``<root>/<scene>/`` merged.
        shard_episodes: The most episodes one shard holds.
        extra: Further ``mp3d.run`` arguments for every child (never one that
            selects episodes: ``SELECTION_FLAGS``).
        resume: Continue a root that already holds shards.
        jobs: Shards at once, unless ``jobs_file`` holds another number (re-read live).
        llm_urls: LLM servers to deal the shards over; empty keeps the environment's.
        gate: The start gate; a :class:`VramGate` at 88 % with 1,400 MiB per renderer by default.

    Returns:
        One record per scene: ``status`` ("merged", "already merged", or why not),
        ``returncode`` (0 only when merged), and its shards' outcomes.

    Raises:
        ValueError: A selecting flag in ``extra``; a fresh campaign into a root
            that already holds shards; a resume acceptance without ``resume``.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    shared, acceptance = split_resume_acceptance(extra)
    clashing = [flag for flag in SELECTION_FLAGS if flag in shared]
    if clashing:
        raise ValueError("a sharded campaign chooses the episodes itself; it takes no %s" % ", ".join(clashing))
    if acceptance and not resume:
        raise ValueError("%s is a resume's: it needs a campaign root that already holds shards" % RESUME_ACCEPTANCE_FLAG)
    protocol = protocol_for(version).with_split(split)
    per_scene = option_value(shared, "--episodes-per-scene")
    counts = scene_episode_counts(episodes_dir, protocol, scenes, int(per_scene) if per_scene else None)
    plan = plan_shards(counts, shard_episodes)
    by_scene = OrderedDict()
    for job in plan:
        by_scene.setdefault(job.scene, []).append(job)
    started = [job for job in plan if (job.directory(root) / "run.json").is_file() or scene_merged(root, job.scene)]
    if started and not resume:
        raise ValueError("%s already holds shards (e.g. %s); resume it instead of starting it again"
                         % (root, started[0].key))
    todo = [job for job in plan if not scene_merged(root, job.scene)
            and not (job.directory(root) / "summary.json").is_file()]
    urls = assign_llm_urls(todo, llm_urls, root)
    write_plan(root, plan, {job.key: recorded_llm_url(job.directory(root)) or urls.get(job.key) for job in plan},
               shard_episodes)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.campaign import scene_command
    pool_jobs = []
    for job in todo:
        output = job.directory(root)
        arguments = list(shared) + ["--shards", str(job.shards), "--shard-index", str(job.index)]
        if (output / "run.json").is_file():
            arguments += ["--resume"] + acceptance
        command = scene_command(job.scene, output, version=version, split=split, episodes_dir=episodes_dir,
                                scenes_dir=scenes_dir, extra=arguments)
        environment = {"LLM_BASE_URL": urls[job.key]} if job.key in urls else {}
        pool_jobs.append(PoolJob(job.key, command, root / job.scene / ("shard_%d.console.log" % job.index),
                                 environment))
    merges = {}

    def merge(scene):
        if scene in merges or scene_merged(root, scene):
            return
        if not all((job.directory(root) / "summary.json").is_file() for job in by_scene[scene]):
            return
        try:
            audit = merge_scene(root, scene, by_scene[scene][0].shards)
            merges[scene] = "merged"
            log("[campaign] merged %s: %d shards, %d episodes" % (scene, len(by_scene[scene]), audit["episodes_scored"]))
        except (OSError, ValueError) as exc:
            merges[scene] = "merge failed: %s" % exc
            log("[campaign] could not merge %s: %s" % (scene, exc))

    def heartbeat(running, pending):
        beat = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "unix": time.time(),
                "done": done_episodes(root, plan), "total": sum(job.episodes for job in plan),
                "running": running, "pending": pending}
        with (root / HEARTBEAT_FILE).open("a") as stream:
            stream.write(json.dumps(beat) + "\n")

    log("[campaign] %d shards over %d scenes (%d episodes), %d to run; at most %d at once%s"
        % (len(plan), len(by_scene), sum(job.episodes for job in plan), len(todo), int(jobs),
           "; LLM servers " + ", ".join(dict.fromkeys(urls.values())) if urls else ""))
    codes, unstarted = run_pool(
        pool_jobs, cap=lambda: read_cap(jobs, jobs_file), gate=gate or VramGate(1400),
        on_exit=lambda job, code: merge(job.key.split("/")[0]) if code == 0 else None,
        heartbeat=heartbeat, popen=popen, sleep=sleep, log=log)
    for scene in by_scene:
        merge(scene)
    results = []
    for scene, scene_jobs in by_scene.items():
        failed = sorted(job.key for job in scene_jobs if codes.get(job.key, 0) != 0)
        waiting = sorted(job.key for job in scene_jobs if job.key in unstarted)
        if scene in merges:
            status = merges[scene]
        elif scene_merged(root, scene):
            status = "already merged"
        else:
            status = "unfinished"
        results.append({"scene": scene, "status": status, "output": str(root / scene),
                        "returncode": 0 if status in ("merged", "already merged") else 1,
                        "shards": len(scene_jobs), "failed_shards": failed, "unstarted_shards": waiting,
                        "shard_returncodes": {job.key: codes[job.key] for job in scene_jobs if job.key in codes}})
    write_atomically(root / "campaign.json", strict_json(
        {"version": version, "split": split, "limit": None, "smoke": False, "sharded": True,
         "shard_episodes": int(shard_episodes), "scenes": len(by_scene), "jobs": int(jobs),
         "llm_urls": [str(url).rstrip("/") for url in llm_urls], "runs": results}, "MP3D campaign", indent=2))
    return results
