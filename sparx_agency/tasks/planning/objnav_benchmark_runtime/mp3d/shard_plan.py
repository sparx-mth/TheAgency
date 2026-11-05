"""Cut an MP3D campaign's scenes into shards, so no scene becomes the campaign's tail.

One process per scene makes a campaign exactly as long as its slowest scene:
the validation split's eleven scenes hold about 200 episodes each, so the scene
that starts last, or holds the long episodes, runs alone for hours while the
card idles (the HM3D ground-truth sweep of 2026-11-04 spent its last hours on a
few scenes and was finished as hand-made "tail" runs). Here each scene is cut
into ``ceil(n / shard_episodes)`` shards, each its own ``mp3d.run`` process:

* a shard is ``mp3d.run --scenes S --shards K --shard-index k``: the scene's
  episodes in the publisher's order, strided by ``K`` from ``k`` -- exactly the
  selection ``provenance.select_episodes`` makes -- so no episode is dropped,
  filtered or repeated, and each shard keeps the publisher's order;
* its results directory is ``<root>/<scene>/shard_<k>/``, a complete,
  self-contained run with its own ``run.json``, rows, lock and progress;
* once every shard of a scene has finished, :func:`merge_scene` writes the
  scene's own directory ``<root>/<scene>/`` from them through
  ``report.merge_runs``, which refuses anything but complete, disjoint shards
  of one experiment, so the campaign aggregate reads a sharded scene exactly
  as it reads an unsharded one.

The start order interleaves the scenes (every scene's shard 0, then every
scene's shard 1, ...), so the workers alive at any moment hold different scenes.

A shard may also be given its own LLM server (:func:`assign_llm_urls`): a shard
that already started keeps the address it froze into its ``run.json``, because
a resume must reproduce its configuration, and fresh shards go to the least
loaded server.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil

from sparx_agency.tasks.planning.objnav_benchmark.results_io import strict_json, write_atomically
from sparx_agency.tasks.planning.objnav_benchmark.run_info import read_run_info

#: The plan a sharded campaign writes beside its scene directories. A file, never
#: a directory: the aggregate reads every subdirectory of the root as a scene.
PLAN_FILE = "shard_plan.json"


@dataclass(frozen=True)
class ShardJob:
    """One shard: ``mp3d.run --scenes <scene> --shards <shards> --shard-index <index>``.

    Attributes:
        scene: The scene key, e.g. ``2azQ1b91cZZ``.
        shards: How many shards the scene is cut into.
        index: This shard's index, ``0 <= index < shards``.
        episodes: How many episodes the shard selects.
    """

    scene: str
    shards: int
    index: int
    episodes: int

    @property
    def key(self):
        """``<scene>/shard_<index>``, the shard's directory relative to the campaign root."""
        return "%s/shard_%d" % (self.scene, self.index)

    def directory(self, root):
        """The shard's results directory under the campaign ``root``."""
        return Path(root) / self.scene / ("shard_%d" % self.index)


def shard_count(episodes, shard_episodes):
    """How many shards ``episodes`` are cut into so that none holds more than ``shard_episodes``.

    Raises:
        ValueError: ``shard_episodes`` or ``episodes`` below one.
    """
    if int(shard_episodes) < 1:
        raise ValueError("shard_episodes must be at least 1, got %r" % (shard_episodes,))
    if int(episodes) < 1:
        raise ValueError("a scene to shard needs at least one episode, got %r" % (episodes,))
    return int(math.ceil(int(episodes) / float(shard_episodes)))


def plan_shards(scene_episodes, shard_episodes):
    """Every shard of every scene, interleaved across scenes in start order.

    Args:
        scene_episodes: ``{scene: episode count}`` in the publisher's scene order.
        shard_episodes: The most episodes one shard may hold.

    Returns:
        ``[ShardJob]``: every scene's shard 0 in scene order, then every scene's
        shard 1, and so on. The episode counts add up to the scenes' totals.
    """
    per_scene = []
    for scene, episodes in scene_episodes.items():
        count = shard_count(episodes, shard_episodes)
        per_scene.append([ShardJob(scene, count, index, len(range(index, int(episodes), count)))
                          for index in range(count)])
    jobs = []
    for rank in range(max((len(shards) for shards in per_scene), default=0)):
        jobs.extend(shards[rank] for shards in per_scene if rank < len(shards))
    return jobs


def recorded_llm_url(directory):
    """The LLM server a started shard froze into its ``run.json``, or ``None`` before it starts."""
    path = Path(directory) / "run.json"
    if not path.is_file():
        return None
    url = ((read_run_info(path)["config"].get("method") or {}).get("llm") or {}).get("base_url")
    return None if url is None else str(url).rstrip("/")


def assign_llm_urls(jobs, urls, root):
    """Which LLM server each shard about to run talks to.

    Args:
        jobs: The shards that will run now (finished ones left out).
        urls: The servers, e.g. ``["http://127.0.0.1:11435", "http://127.0.0.1:11436"]``;
            empty leaves every shard on the environment's ``LLM_BASE_URL``.
        root: The campaign root, where started shards recorded theirs.

    Returns:
        ``{job.key: url}``. A started shard keeps the server it froze; each fresh
        one goes to the server with the fewest shards so far (ties: the first named).

    Raises:
        ValueError: A started shard froze a server that is not among ``urls``:
            its resume would be refused by its own configuration check, so it is
            named here instead.
    """
    urls = [str(url).rstrip("/") for url in urls]
    if not urls:
        return {}
    assigned, fresh = {}, []
    for job in jobs:
        frozen = recorded_llm_url(job.directory(root))
        if frozen is None:
            fresh.append(job)
        elif frozen in urls:
            assigned[job.key] = frozen
        else:
            raise ValueError("shard %s froze the LLM server %s, which is not among %s; name it again to resume "
                             "the shard against the configuration it started with" % (job.key, frozen, urls))
    load = {url: sum(1 for value in assigned.values() if value == url) for url in urls}
    for job in fresh:
        url = min(urls, key=lambda candidate: (load[candidate], urls.index(candidate)))
        assigned[job.key] = url
        load[url] += 1
    return assigned


def write_plan(root, jobs, llm_urls, shard_episodes):
    """Write ``<root>/shard_plan.json``: every shard, its episode count and its LLM server."""
    payload = {"shard_episodes": int(shard_episodes),
               "episodes": sum(job.episodes for job in jobs),
               "scenes": list(dict.fromkeys(job.scene for job in jobs)),
               "shards": [dict(asdict(job), key=job.key, llm_url=llm_urls.get(job.key)) for job in jobs]}
    write_atomically(Path(root) / PLAN_FILE, strict_json(payload, "MP3D shard plan", indent=2))


def read_plan(root):
    """The plan a sharded campaign wrote under ``root``, or ``None`` for an unsharded root."""
    path = Path(root) / PLAN_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def scene_merged(root, scene):
    """Whether the scene's own directory holds its merged, finished rows."""
    return (Path(root) / scene / "summary.json").is_file()


def merge_scene(root, scene, shards):
    """Write ``<root>/<scene>/`` from the scene's ``shards`` finished shard directories.

    ``report.merge_runs`` writes into a staging directory first; its files are
    then moved into the scene directory with ``summary.json`` last, so a crash
    part-way leaves a scene the aggregate still calls unfinished, and a second
    merge simply replaces the files.

    Returns:
        The merged scene's audit (``report.write_report``).

    Raises:
        ValueError: A shard is missing, incomplete or from another experiment
            (``report.merge_runs``).
    """
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report import merge_runs
    scene_dir = Path(root) / scene
    staging = scene_dir / ".merge"
    if staging.exists():
        shutil.rmtree(staging)
    audit = merge_runs([scene_dir / ("shard_%d" % index) for index in range(int(shards))], staging)
    files = sorted(path.name for path in staging.iterdir()
                   if path.is_file() and path.name not in ("summary.json", "run.lock"))
    for name in files + ["summary.json"]:
        os.replace(str(staging / name), str(scene_dir / name))
    shutil.rmtree(staging)
    return audit
