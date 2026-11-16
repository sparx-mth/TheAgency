"""The fixed single-storey set on the CPU with a stand-in for the language model: one arm of a paired check.

The check this repeats (``CHANGELOG.md`` 2026-11-15, 2026-11-16): the same 180
MP3D episodes (``sets/single_storey_flat_v1.json``) before and after a change
to the closing, the planner or the map, with nothing but the change differing
between the two arms. Three choices make that true and keep the GPU free:

* **the language model is a stand-in** (``stub_llm.py``: every room
  ``living_room`` at 0.8, the nodes of a prompt an equal share of 90 %), so
  no answer differs run to run and no model is loaded;
* **the renderer is Mesa's llvmpipe**: every child runs in a user namespace
  with ``/dev/dri`` hidden and GLVND told to load Mesa alone, and is given
  ``--software-render`` (``LESSONS.md`` 2026-10-12, 2026-10-31);
* **the scene processes are started against free RAM, not a job count**
  (``ram_gate.RamGate``, ``LESSONS.md`` 2026-11-15): under llvmpipe a scene's
  geometry, textures and semantic mesh live in its process -- 2 to 6 GB a
  scene (:data:`SCENE_GIB`) -- and six at once took a 30 GB machine into
  swap. The pool is the campaign's own (``shard_pool.run_pool``), with a
  gate that picks the first pending scene that fits.

Ground-truth perception, the frontier explorer, seed 17, lean rows, one
process per scene, first fit from the queue. These are the flags and the
environment of the check of 2026-11-15 (:data:`RUN_FLAGS`,
:data:`CHILD_ENV`), kept exactly: on 2026-11-16 this launcher, on the tree
that check's second arm ran, reproduced its stored trajectories -- so an arm
run then is an arm of a check run now. ``--record`` adds a log-only
recording of every episode (``steps.jsonl``), for a look at chosen episodes.

``--tree`` runs the code of ANOTHER copy of the repository -- a frozen
snapshot of the commit before the change -- with this file's own tree
supplying only the launcher and the stand-in::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.cpu_stub_campaign \\
        --output runs/cpu_ab/before --tree ~/GIT/TheAgency_snapshots/before
    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.cpu_stub_campaign \\
        --output runs/cpu_ab/after
    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.stuck_report \\
        runs/cpu_ab/before runs/cpu_ab/after

``--shard N`` cuts a scene's list into processes of at most N episodes
(``<output>/<scene>/shard_<k>/``), so the longest scene is not the arm's
tail; each shard holds the scene's RAM again, which the gate counts.
``--only`` runs a subset of the set (a control, a second look).

The numbers are not results of the method (the stand-in orders the rooms by
travel cost and exploration state alone); only the paired differences are.

Python 3.9 syntax, standard library only beyond this package.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sparx_agency.tasks.planning.objnav_benchmark_runtime.campaign_common import episode_plan
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.ram_gate import RamGate
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.shard_pool import PoolJob, run_pool
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_set import DEFAULT_SET

RUNTIME_DIR = Path(__file__).resolve().parents[1]
REPO = RUNTIME_DIR.parents[3]
STUB = RUNTIME_DIR / "stub_llm.py"
#: Hide every DRM device from the child, so Mesa's software rasteriser is the only EGL device there is.
HIDE_DRI = ["unshare", "-rm", "sh", "-c", 'mount -t tmpfs none /dev/dri && exec "$@"', "_"]
MESA_EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
#: What every child is run with, beside its scene and episodes: the check's own flags, never changed between arms.
RUN_FLAGS = ("--seed", "17", "--perception", "ground-truth", "--explorer", "frontier", "--lean", "--software-render",
             "--validate-starts")
#: ... and its environment: no CUDA device, the thread counts the check was first run with.
CHILD_ENV = {"CUDA_VISIBLE_DEVICES": "", "LP_NUM_THREADS": "6", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
             "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
#: What a scene's process holds once loaded under llvmpipe, GiB (measured 2026-11-15); an unlisted scene is
#: budgeted at :data:`DEFAULT_SCENE_GIB`.
SCENE_GIB = {"Z6MFQCViBuw": 6.2, "QUCTc6BB5sX": 5.7, "2azQ1b91cZZ": 5.4, "TbHJrupSAjP": 4.4, "oLBMNvg9in8": 4.2,
             "zsNo4HB9uLZ": 2.8, "8194nk5LbLH": 2.1, "pLe4wQe7qrG": 2.0}
DEFAULT_SCENE_GIB = 6.0


def shards(episode_ids, size):
    """``episode_ids`` in consecutive lists of at most ``size`` (``size`` under 1: one list)."""
    ids = list(episode_ids)
    if int(size) < 1 or len(ids) <= int(size):
        return [ids]
    return [ids[start:start + int(size)] for start in range(0, len(ids), int(size))]


def scene_jobs(plan, output, *, tree, python, episodes_dir, scenes_dir, extra=(), shard=0, record=False):
    """The :class:`PoolJob` s of ``plan``: one per scene, or one per shard of it, the longest lists first.

    Args:
        plan: ``{scene: [episode id, ...]}`` (``campaign_common.episode_plan``).
        output: The arm's root; a scene writes ``<output>/<scene>/``, a shard ``<output>/<scene>/shard_<k>/``.
        tree: The repository copy whose code runs (the child's cwd and ``PYTHONPATH``).
        python: The Habitat environment's interpreter.
        extra: Further ``mp3d.run`` arguments for every child (``--policy-config ...``).
        shard: Episodes per process at most; under 1, one process per scene.
        record: Add a log-only recording of every episode.

    Returns:
        ``([PoolJob] in queue order, {job key: scene})``.
    """
    env = dict(CHILD_ENV, PYTHONPATH=str(tree))
    if MESA_EGL_VENDOR.is_file():
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(MESA_EGL_VENDOR)     # GLVND loads Mesa alone: not NVIDIA's EGL
    flags = list(RUN_FLAGS) + (["--record-log-only"] if record else [])
    pieces = []
    for scene in plan:
        parts = shards(plan[scene], shard)
        for index, ids in enumerate(parts):
            sharded = len(parts) > 1
            pieces.append((("%s/shard_%d" % (scene, index)) if sharded else scene,
                           (Path(output) / scene / ("shard_%d" % index)) if sharded else Path(output) / scene,
                           scene, ids))
    jobs, scene_of = [], {}
    for key, directory, scene, ids in sorted(pieces, key=lambda piece: (-len(piece[3]), piece[0])):
        command = (HIDE_DRI + ["env", "--chdir=%s" % tree, str(python), "-u", str(STUB), "mp3d", "--version", "v1",
                               "--split", "val", "--episodes-dir", str(episodes_dir), "--scenes-dir", str(scenes_dir),
                               "--scenes", scene, "--output", str(directory), "--episode-ids"]
                   + list(ids) + flags + list(extra))
        jobs.append(PoolJob(key=key, command=command, env=env, directory=directory,
                            log_path=Path(output) / ("%s.console.log" % key.replace("/", "."))))
        scene_of[key] = scene
    return jobs, scene_of


def tree_revision(tree):
    """What code ``tree`` holds: a snapshot's recorded commit, else the checkout's HEAD and whether it is dirty."""
    stamp = Path(tree) / "SNAPSHOT_COMMIT.txt"
    if stamp.is_file():
        changes = Path(tree) / "SNAPSHOT_CHANGES.diff"
        return {"commit": stamp.read_text().strip(), "snapshot": True,
                "changes_bytes": changes.stat().st_size if changes.is_file() else None}
    try:
        commit = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(tree), "status", "--porcelain"], capture_output=True, text=True,
                                    check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "snapshot": False}
    return {"commit": commit, "snapshot": False, "dirty": dirty}


def missing_rows(plan, output):
    """``{scene: [episode id, ...]}`` -- the planned episodes with no row under their scene's directory."""
    missing = {}
    for scene, wanted in plan.items():
        done = set()
        for path in sorted((Path(output) / scene).glob("**/episodes.jsonl")):
            with open(str(path)) as stream:
                done |= {json.loads(line)["episode_id"] for line in stream if line.strip()}
        absent = [episode_id for episode_id in wanted if episode_id not in done]
        if absent:
            missing[scene] = absent
    return missing


def build_parser():
    """The launcher's command line; the paths default to the development laptop's, as ``benchmark_services.sh`` does."""
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="The arm's root: one directory per scene under it")
    parser.add_argument("--tree", type=Path, default=REPO,
                        help="The repository copy whose code runs (default: this one); a frozen snapshot for 'before'")
    parser.add_argument("--set", type=Path, default=DEFAULT_SET, dest="episode_set",
                        help="JSON list of <scene>/<index> ids (default: the fixed single-storey set)")
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="Run only these ids of the set, or every id of the scenes named (a control, a second look)")
    parser.add_argument("--python", type=Path,
                        default=Path(os.environ.get("HAB_PY", str(home / "miniconda3" / "envs" / "habitat" / "bin" / "python"))))
    parser.add_argument("--episodes-dir", type=Path,
                        default=Path(os.environ.get("MP3D_EPISODES_DIR", str(home / "datasets" / "objectnav" / "mp3d" / "v1" / "val"))))
    parser.add_argument("--scenes-dir", type=Path,
                        default=Path(os.environ.get("MP3D_SCENES_DIR", str(home / "datasets" / "scene_datasets"))))
    parser.add_argument("--shard", type=int, default=0,
                        help="Episodes per process at most (0: one process per scene)")
    parser.add_argument("--record", action="store_true", help="Also record every episode's log (steps.jsonl); no video")
    parser.add_argument("--jobs", type=int, default=6, help="Scene processes at once, at most (RAM permitting)")
    parser.add_argument("--reserve-gib", type=float, default=4.5, help="RAM left untouched")
    parser.add_argument("--stall-minutes", type=float, default=30.0, help="Kill a scene silent for this long; 0 disables")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="-- further mp3d.run arguments for every scene")
    return parser


def selected_ids(ids, only):
    """``ids`` cut down to ``only`` -- ids of the set, or scenes whose every id is meant; all of them without it.

    Raises:
        SystemExit: ``only`` names an id or a scene the set does not hold.
    """
    if not only:
        return list(ids)
    wanted, scenes = set(only), {str(i).split("/")[0] for i in ids}
    unknown = sorted(w for w in wanted if w not in set(ids) and w not in scenes)
    if unknown:
        raise SystemExit("--only names ids or scenes the set does not hold: %s" % ", ".join(unknown))
    return [i for i in ids if i in wanted or str(i).split("/")[0] in wanted]


def main(argv=None):
    args = build_parser().parse_args(argv)
    extra = [item for item in args.extra if item != "--"]
    ids = selected_ids(json.loads(args.episode_set.read_text()), args.only)
    plan = episode_plan(ids, {str(i).split("/")[0] for i in ids})
    tree = args.tree.expanduser().resolve()
    if not (tree / "sparx_agency").is_dir():
        raise SystemExit("--tree %s holds no sparx_agency package" % tree)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs, scene_of = scene_jobs(plan, output, tree=tree, python=args.python.expanduser(),
                                episodes_dir=args.episodes_dir, scenes_dir=args.scenes_dir, extra=extra,
                                shard=args.shard, record=args.record)
    record = {"tree": str(tree), "revision": tree_revision(tree), "episode_set": str(args.episode_set),
              "episodes": len(ids), "scenes": {scene: len(plan[scene]) for scene in plan}, "run_flags": list(RUN_FLAGS),
              "environment": dict(CHILD_ENV), "extra": extra, "shard": args.shard, "recorded": bool(args.record),
              "language_model": "stand-in (stub_llm.py)", "renderer": "Mesa llvmpipe, /dev/dri hidden",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commands": {job.key: job.command for job in jobs}}
    (output / "campaign.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print("[campaign] %d episodes in %d scenes (%d processes), tree %s (%s), output %s"
          % (len(ids), len(plan), len(jobs), tree, record["revision"].get("commit"), output), flush=True)
    gate = RamGate(lambda job: SCENE_GIB.get(scene_of[job.key], DEFAULT_SCENE_GIB), args.reserve_gib)
    codes, unstarted = run_pool(jobs, cap=lambda: args.jobs, gate=gate, stall_s=60.0 * args.stall_minutes,
                                log=lambda line: print(line, flush=True))
    missing = missing_rows(plan, output)
    record.update(returncodes=codes, unstarted=unstarted, missing_rows=missing,
                  finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (output / "campaign.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    # The verdict is the rows, not the exit codes: a child may die in an interpreter's teardown with every row written.
    print("[campaign] done: %d of %d episodes have a row%s"
          % (len(ids) - sum(len(v) for v in missing.values()), len(ids),
             "; missing in %s" % ", ".join("%s (%d)" % (scene, len(v)) for scene, v in sorted(missing.items()))
             if missing else ""), flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
