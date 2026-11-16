"""A start gate on free RAM for ``shard_pool.run_pool``: which pending job fits in memory now.

``shard_pool.VramGate`` holds a pool of equal jobs back while the graphics
card is full. A pool whose jobs render on the CPU has the same problem in
main memory and jobs of unequal size: under Mesa's llvmpipe a scene's
geometry, textures and semantic mesh live in its process -- 2 to 6 GiB a
scene -- and six MP3D scenes at once took a 30 GB machine into swap
(``LESSONS.md`` 2026-11-15). :class:`RamGate` therefore *chooses*: the first
pending job whose need still leaves a reserve of the memory available
(``run_pool`` calls ``choose(pending)`` where a gate has it).

Standard library only.
"""
from __future__ import annotations

import time

GIB = 1024.0 ** 3


def mem_available_bytes(path="/proc/meminfo"):
    """``MemAvailable`` of this machine, in bytes."""
    with open(path) as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("%s has no MemAvailable line" % path)


class RamGate:
    """Which pending scene process fits in memory now: only ever delays a start, never touches running work.

    A gate for ``shard_pool.run_pool`` that chooses (``choose``): the first
    pending job whose need, on top of the needs of every job started too
    recently to show its own footprint, still leaves ``reserve_gib`` of the
    available memory.

    Args:
        need: ``PoolJob -> GiB`` its process holds once its scene is loaded.
        reserve_gib: Memory left untouched.
        spacing_s: The least time between two starts.
        settle_s: How long a started job is budgeted at its need on top of
            what the machine already shows (its scene is still loading).
        available: ``() -> bytes``; ``/proc/meminfo`` by default.
        clock: Monotonic seconds; injected for tests.
    """

    def __init__(self, need, reserve_gib=4.5, *, spacing_s=15.0, settle_s=80.0, available=mem_available_bytes,
                 clock=time.monotonic):
        if float(reserve_gib) < 0:
            raise ValueError("reserve_gib must be non-negative, got %r" % (reserve_gib,))
        self.need = need
        self.reserve_gib = float(reserve_gib)
        self.spacing_s, self.settle_s = float(spacing_s), float(settle_s)
        self._available, self._clock = available, clock
        self._starts = []                       # (time, need) of the starts still settling
        self._chosen = None                     # the need of the job ``choose`` last picked, for ``started``
        self.last_reason = None

    def choose(self, pending):
        """The index of the first job of ``pending`` that fits now, or None; ``last_reason`` says why none does."""
        now = self._clock()
        self._starts = [(at, need) for at, need in self._starts if now - at < self.settle_s]
        self._chosen = None
        if self._starts and now - self._starts[-1][0] < self.spacing_s:
            self.last_reason = "spacing starts %.0f s apart" % self.spacing_s
            return None
        free = self._available() / GIB
        settling = sum(need for _, need in self._starts)
        for index, job in enumerate(pending):
            need = float(self.need(job))
            if free - settling - need >= self.reserve_gib:
                self._chosen, self.last_reason = need, None
                return index
        self.last_reason = ("RAM: %.1f GiB available - %.1f settling leaves no pending job (%.1f GiB the smallest) its "
                            "%.1f GiB reserve" % (free, settling, min(float(self.need(job)) for job in pending),
                                                  self.reserve_gib))
        return None

    def ready(self):
        """Not this gate's question: it chooses among the pending jobs (:meth:`choose`)."""
        raise NotImplementedError("RamGate chooses a job; the pool calls choose(pending)")

    def started(self):
        """Record the start of the job last chosen, so the next judgement budgets it until it has settled."""
        if self._chosen is None:
            raise RuntimeError("started() without a job chosen")
        self._starts.append((self._clock(), self._chosen))
        self._chosen = None
