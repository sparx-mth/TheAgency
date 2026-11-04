"""Run several scene processes at once, each logging to its own file.

A scene's child process is single-threaded between its service calls, so one
child leaves the GPU and most CPU cores idle. Running ``jobs`` children side by
side is the campaign's throughput lever; it does not touch the policy, the
scoring or any child's arguments. Interleaved progress lines from concurrent
children would be unreadable, so each child's stdout and stderr go to
``<root>/<scene>.console.log`` -- a file beside the scene directories, never
inside one, because the aggregate treats every subdirectory of the root as a
scene.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import threading
import time


def max_workers(scene_count, *, cpu_count, free_vram_mib, renderer_mib, reserve_mib=512, cores_per_worker=1):
    """How many scene children the machine can hold at once: the campaign's ``--jobs max``.

    One child renders one scene (``renderer_mib`` of VRAM each, after a
    ``reserve_mib`` the driver and the display keep) and, between its service
    calls, keeps about one core busy -- the LLM and the detector are the
    services' cores, not the child's. There is never a reason to run more
    children than scenes.

    Args:
        scene_count: Scenes the campaign will run.
        cpu_count: Cores on the machine (``os.cpu_count()``).
        free_vram_mib: Free VRAM on the rendering card right now.
        renderer_mib: What one habitat-sim renderer takes (``HABITAT_VRAM_MIB``).
        reserve_mib: VRAM left untouched.
        cores_per_worker: Cores budgeted per child.

    Returns:
        At least 1, at most ``scene_count``.
    """
    if int(scene_count) < 1:
        raise ValueError("scene_count must be at least 1, got %r" % (scene_count,))
    by_vram = (int(free_vram_mib) - int(reserve_mib)) // max(1, int(renderer_mib))
    by_cpu = int(cpu_count) // max(1, int(cores_per_worker))
    return max(1, min(int(scene_count), by_vram, by_cpu))


def console_log_path(root, scene):
    """Where a concurrent child's console output goes."""
    return Path(root) / ("%s.console.log" % scene)


def run_concurrently(jobs, root, workers, *, runner=subprocess.run, announce=print):
    """Run ``(scene, command)`` jobs with at most ``workers`` children alive at once.

    Args:
        jobs: ``[(scene, command)]`` in the order they should start.
        root: The campaign root; each child's console log is written beside its
            scene directory.
        workers: The most children alive at once.
        runner: Must behave like ``subprocess.run`` and accept ``stdout`` and
            ``stderr`` keyword arguments.
        announce: Receives one line when a scene starts and one when it ends.

    Returns:
        ``{scene: returncode}`` for every job.
    """
    if int(workers) < 1:
        raise ValueError("workers must be at least 1, got %r" % (workers,))
    scenes = [scene for scene, _ in jobs]
    if len(set(scenes)) != len(scenes):
        raise ValueError("a scene may run only once per campaign: %s" % scenes)
    lock = threading.Lock()
    returncodes = {}

    def one(scene, command):
        log_path = console_log_path(root, scene)
        started = time.time()
        with lock:
            announce("[campaign] start %s (console: %s)" % (scene, log_path))
        with open(log_path, "a") as stream:
            completed = runner(command, stdout=stream, stderr=subprocess.STDOUT)
        code = int(getattr(completed, "returncode", 1))
        with lock:
            returncodes[scene] = code
            announce("[campaign] end   %s exit %d after %.1f h"
                     % (scene, code, (time.time() - started) / 3600.0))

    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(one, scene, command) for scene, command in jobs]
        for future in futures:
            future.result()
    return returncodes


def main(argv=None):
    """Print ``--jobs max`` for this machine: ``python -m ...hm3d.scene_pool --scenes 20 [--gpu-device 0]``."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gpu_plan import (
        DEFAULT_RESERVE_MIB, HABITAT_RENDER_MIB, nvidia_smi_memory)
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--scenes", type=int, required=True, help="Scenes the campaign will run")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--renderer-mib", type=int, default=int(os.environ.get("HABITAT_VRAM_MIB", HABITAT_RENDER_MIB)))
    parser.add_argument("--reserve-mib", type=int, default=int(os.environ.get("GPU_RESERVE_MIB", DEFAULT_RESERVE_MIB)))
    parser.add_argument("--explain", action="store_true", help="Print the budget beside the number, on stderr")
    args = parser.parse_args(argv)
    name, total, used = nvidia_smi_memory(args.gpu_device)
    workers = max_workers(args.scenes, cpu_count=os.cpu_count() or 1, free_vram_mib=total - used,
                          renderer_mib=args.renderer_mib, reserve_mib=args.reserve_mib)
    if args.explain:
        import sys
        print("%s: %d MiB free of %d; %d MiB per renderer after a %d MiB reserve; %d cores; %d scenes -> %d jobs"
              % (name, total - used, total, args.renderer_mib, args.reserve_mib, os.cpu_count() or 1, args.scenes, workers),
              file=sys.stderr)
    print(workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
