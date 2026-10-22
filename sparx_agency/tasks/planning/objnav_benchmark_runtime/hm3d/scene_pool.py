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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import threading
import time


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
