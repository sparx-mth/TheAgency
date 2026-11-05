"""Start a campaign's shard processes as the machine has room for them.

Two rules from the HM3D ground-truth sweep of 2026-11-04 shape this pool:

* **VRAM never stops running work; it only delays new work.** A paused
  renderer keeps its memory, so a guard that paused workers for VRAM froze them
  for hours. Here VRAM decides only whether the *next* shard may start
  (:class:`VramGate`): it starts when the card's used memory, plus one
  renderer's budget for it and for every shard started too recently to show its
  own, stays under the cap.
* **The worker count is tuned while the run goes.** The cap is re-read every
  poll (:func:`read_cap`: an integer written to a file overrides the default),
  so raising it starts more shards, lowering it lets the running ones finish
  without replacements, and 0 drains the pool: it returns once nothing runs,
  with the rest unstarted, for a resume under new settings.

Each shard's console goes to its own log file; its environment is this
process's plus the job's own overrides (its LLM server, say).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List

#: The share of the card the pool fills up to, as the campaign's resource guard caps it.
DEFAULT_VRAM_CAP = 0.88


@dataclass
class PoolJob:
    """One process to run.

    Attributes:
        key: Its name in logs and results, e.g. ``2azQ1b91cZZ/shard_3``.
        command: The argv.
        log_path: Where its stdout and stderr go (appended).
        env: Environment variables set for it on top of this process's.
    """

    key: str
    command: List[str]
    log_path: Path
    env: Dict[str, str] = field(default_factory=dict)


def read_cap(default, path=None):
    """The live worker cap: the non-negative integer in ``path`` when it holds one, else ``default``."""
    if path:
        try:
            text = Path(path).read_text().strip()
            if text and int(text) >= 0:
                return int(text)
        except (OSError, ValueError):
            pass
    return int(default)


class VramGate:
    """Whether one more renderer may start now: only ever delays a start, never touches running work.

    Args:
        renderer_mib: One renderer's VRAM budget.
        cap_fraction: The share of the card the used memory may reach.
        spacing_s: The least time between two starts, so each shows its memory
            before the next is judged.
        settle_s: How long a started renderer is budgeted at ``renderer_mib`` on
            top of what nvidia-smi shows (its scene may still be loading).
        gpu_device: The card's index.
        query: ``gpu_device -> (name, total MiB, used MiB)``; nvidia-smi by default.
        clock: Monotonic seconds; injected for tests.
    """

    def __init__(self, renderer_mib, cap_fraction=DEFAULT_VRAM_CAP, *, spacing_s=20.0, settle_s=120.0,
                 gpu_device=0, query=None, clock=time.monotonic):
        if not 0 < float(cap_fraction) <= 1:
            raise ValueError("cap_fraction must be in (0, 1], got %r" % (cap_fraction,))
        if int(renderer_mib) < 1:
            raise ValueError("renderer_mib must be positive, got %r" % (renderer_mib,))
        if query is None:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.gpu_plan import nvidia_smi_memory
            query = nvidia_smi_memory
        self.renderer_mib = int(renderer_mib)
        self.cap_fraction = float(cap_fraction)
        self.spacing_s = float(spacing_s)
        self.settle_s = float(settle_s)
        self.gpu_device = int(gpu_device)
        self._query = query
        self._clock = clock
        self._starts = []
        self.last_reason = None

    def ready(self):
        """True when one more renderer fits under the cap now; otherwise ``last_reason`` says why not."""
        now = self._clock()
        self._starts = [t for t in self._starts if now - t < self.settle_s]
        if self._starts and now - self._starts[-1] < self.spacing_s:
            self.last_reason = "spacing starts %.0f s apart" % self.spacing_s
            return False
        try:
            _, total, used = self._query(self.gpu_device)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            self.last_reason = "nvidia-smi failed (%s)" % exc
            return False
        need = used + self.renderer_mib * (1 + len(self._starts))
        if need > self.cap_fraction * total:
            self.last_reason = ("VRAM: %d MiB used + %d x %d MiB budgeted > %.0f%% of %d MiB"
                                % (used, 1 + len(self._starts), self.renderer_mib, 100 * self.cap_fraction, total))
            return False
        self.last_reason = None
        return True

    def started(self):
        """Record a start, so the next judgement budgets it until it has settled."""
        self._starts.append(self._clock())


def run_pool(jobs, *, cap, gate, on_exit=None, heartbeat=None, heartbeat_s=300.0, popen=subprocess.Popen,
             poll_s=5.0, sleep=time.sleep, clock=time.monotonic, log=print):
    """Run ``jobs`` in order, at most ``cap()`` at once, each started only when ``gate.ready()``.

    Args:
        jobs: ``[PoolJob]`` in start order.
        cap: ``() -> int``, re-read every poll; 0 with nothing running drains the pool.
        gate: Has ``ready() -> bool``, ``started()`` and ``last_reason`` (:class:`VramGate`).
        on_exit: ``(job, returncode)``, called as each process ends.
        heartbeat: ``(running, pending)``, called every ``heartbeat_s``.
        popen: Must behave like ``subprocess.Popen``.

    Returns:
        ``(returncodes, unstarted)``: ``{key: returncode}`` of every job that ran,
        and the keys of those a drain left unstarted.

    On any exception, KeyboardInterrupt and SystemExit included, the running
    processes are terminated (then killed after 60 s) before it propagates, so
    no shard outlives the services its campaign brought up.
    """
    pending, running, codes = list(jobs), {}, {}
    waiting_logged = last_beat = None
    try:
        while pending or running:
            for key in list(running):
                job, process, stream, started = running[key]
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del running[key]
                codes[key] = code
                log("[pool] end   %s exit %d after %.2f h (%d running, %d pending)"
                    % (key, code, (clock() - started) / 3600.0, len(running), len(pending)))
                if on_exit is not None:
                    on_exit(job, code)
            limit = cap()
            if pending and not running and limit <= 0:
                log("[pool] drained (cap 0): %d job(s) not started" % len(pending))
                break
            if pending and len(running) < limit:
                if gate.ready():
                    job = pending.pop(0)
                    job.log_path.parent.mkdir(parents=True, exist_ok=True)
                    stream = open(str(job.log_path), "a")
                    process = popen(job.command, stdout=stream, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, **job.env))
                    gate.started()
                    running[job.key] = (job, process, stream, clock())
                    log("[pool] start %s%s (%d running, %d pending, cap %d)"
                        % (job.key, "".join(" %s=%s" % item for item in sorted(job.env.items())),
                           len(running), len(pending), limit))
                    waiting_logged = None
                    continue
                if waiting_logged is None or clock() - waiting_logged >= 60.0:
                    log("[pool] waiting to start %s: %s" % (pending[0].key, gate.last_reason))
                    waiting_logged = clock()
            if heartbeat is not None and (last_beat is None or clock() - last_beat >= heartbeat_s):
                heartbeat(len(running), len(pending))
                last_beat = clock()
            sleep(poll_s)
    finally:
        _stop(running, log)
    return codes, [job.key for job in pending]


def _stop(running, log, grace_s=60.0):
    """Terminate every process still running, kill those that outlast ``grace_s``."""
    if not running:
        return
    log("[pool] stopping %d running job(s)" % len(running))
    for _, process, _, _ in running.values():
        process.terminate()
    deadline = time.monotonic() + grace_s
    for _, process, stream, _ in running.values():
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stream.close()
