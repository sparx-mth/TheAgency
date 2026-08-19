"""Honest latency measurement: per-iteration timing, real percentiles, drift.

A control loop is not killed by the mean. It is killed by the tail: one frame in
a hundred that takes three times as long is a missed deadline, and a missed
deadline is a stale command on a flying aircraft. The benchmark this package
replaces (``tasks/planning/vlas/navdp/trt/benchmark/bench.py:fps``) divides a
single total by twenty iterations with no ``torch.cuda.synchronize`` -- so it
reports a number that cannot show a tail even in principle, and on the GPU path
it partly measures enqueue time rather than execution time. This module exists
to make that measurement honest, and every decision in it follows from that:

  * **Each iteration is timed individually** with :func:`time.perf_counter`.
    Percentiles over per-iteration samples are real; a divided total has none.
  * **The synchronize callable is inside the timed region.** CUDA work is
    asynchronous, so a timer stopped before the sync measures the launch, not
    the kernel. ``sync`` is also called once after warmup so the first measured
    iteration does not absorb the warmup's outstanding work.
  * **Percentiles are nearest-rank**, never interpolated. Every reported value
    is a sample that actually happened, and the method is deterministic, so two
    runs of the same numbers agree to the bit -- which matters when a build is
    accepted or rejected on a p99 threshold.
  * **The measurement environment is reported, not assumed.**
    :func:`clock_warnings` states in the report why the absolute numbers on this
    machine are not reproducible, and :func:`drift_check` tests the one failure
    that silently invalidates a laptop run: the GPU getting slower as it heats.

Standard library only at module scope -- no numpy, no torch. ``statistics``
covers everything needed, so this module imports on a bare Jetson interpreter
and inside the ROS1/Noetic container. ``torch`` is imported lazily and only by
:func:`cuda_sync`. Python-3.8-compatible syntax.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import List

#: Seam for deterministic tests. Production code must never rebind this; the
#: test suite scripts it so percentile assertions are exact instead of flaky.
_perf_counter = time.perf_counter


@dataclass
class LatencyStats:
    """The result of one timing run, in milliseconds.

    Percentiles are nearest-rank (see :func:`percentile`), so ``p99_ms`` is an
    observed iteration and not an interpolation between two of them.

    Args:
        mean_ms: arithmetic mean of the per-iteration samples.
        p50_ms: median (nearest-rank 50th percentile).
        p90_ms: nearest-rank 90th percentile.
        p99_ms: nearest-rank 99th percentile -- the control-loop number.
        min_ms: fastest iteration.
        max_ms: slowest iteration.
        std_ms: sample standard deviation (Bessel-corrected), 0.0 for n == 1.
        iters: number of measured iterations (warmup excluded).
        warmup: number of untimed warmup iterations that preceded them.
        samples_ms: the raw per-iteration samples, kept so
            :func:`drift_check` can be run on a result. Excluded from
            :meth:`as_dict` -- a report row holds the summary, not the run.
    """

    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    iters: int
    warmup: int
    samples_ms: List[float] = field(default_factory=list)

    @property
    def hz(self):
        """Throughput implied by the mean, in decisions per second.

        Returns:
            float: ``1000.0 / mean_ms``.

        Raises:
            ValueError: if ``mean_ms`` is not positive. A non-positive mean is a
                broken measurement, and returning ``inf`` would let it flow into
                a speedup ratio and be reported as a win.
        """
        if self.mean_ms <= 0.0:
            raise ValueError(
                "LatencyStats.hz undefined for mean_ms=%r; the timing run "
                "produced no measurable duration" % (self.mean_ms,))
        return 1000.0 / self.mean_ms

    def as_dict(self):
        """Summary as a flat JSON-safe dict (no raw samples).

        Returns:
            Dict[str, float]: every scalar field plus ``hz``, or ``hz`` omitted
            when the mean is non-positive and therefore has no throughput.
        """
        out = {
            "mean_ms": float(self.mean_ms), "p50_ms": float(self.p50_ms),
            "p90_ms": float(self.p90_ms), "p99_ms": float(self.p99_ms),
            "min_ms": float(self.min_ms), "max_ms": float(self.max_ms),
            "std_ms": float(self.std_ms), "iters": int(self.iters),
            "warmup": int(self.warmup),
        }
        if self.mean_ms > 0.0:
            out["hz"] = self.hz
        return out

    def __str__(self):
        """One compact human line, safe to log even for a degenerate run."""
        rate = "%.1f Hz" % self.hz if self.mean_ms > 0.0 else "-- Hz"
        return ("mean %.3f ms (%s) | p50 %.3f p90 %.3f p99 %.3f | "
                "min %.3f max %.3f sd %.3f | n=%d +%dw"
                % (self.mean_ms, rate, self.p50_ms, self.p90_ms, self.p99_ms,
                   self.min_ms, self.max_ms, self.std_ms, self.iters,
                   self.warmup))

    @classmethod
    def from_samples(cls, samples_ms, warmup=0):
        """Summarize per-iteration samples that were captured elsewhere.

        Args:
            samples_ms: per-iteration durations in milliseconds.
            warmup: warmup iterations that preceded them, for the record.

        Returns:
            LatencyStats: the summary.

        Raises:
            ValueError: if ``samples_ms`` is empty.
        """
        vals = [float(v) for v in samples_ms]
        if not vals:
            raise ValueError("cannot summarize an empty timing run")
        ordered = sorted(vals)
        return cls(
            mean_ms=statistics.fmean(vals),
            p50_ms=percentile(ordered, 50.0, presorted=True),
            p90_ms=percentile(ordered, 90.0, presorted=True),
            p99_ms=percentile(ordered, 99.0, presorted=True),
            min_ms=ordered[0], max_ms=ordered[-1],
            std_ms=statistics.stdev(vals) if len(vals) > 1 else 0.0,
            iters=len(vals), warmup=int(warmup), samples_ms=vals,
        )


def percentile(samples, q, presorted=False):
    """Nearest-rank percentile of ``samples``.

    The rank is ``ceil(q / 100 * n)`` clamped to ``[1, n]`` and the sample at
    that 1-based rank of the sorted run is returned verbatim. No interpolation:
    the value is one that was actually observed, and the result is bit-identical
    across runs and platforms -- required, because engines are accepted or
    rejected against a fixed p99 budget.

    Args:
        samples: durations; sorted ascending already if ``presorted``.
        q: percentile in ``[0, 100]``.
        presorted: skip the sort when the caller has already sorted.

    Returns:
        float: the sample at the nearest rank.

    Raises:
        ValueError: on an empty sample or a ``q`` outside ``[0, 100]``.
    """
    if not samples:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile q=%r outside [0, 100]" % (q,))
    ordered = samples if presorted else sorted(samples)
    n = len(ordered)
    rank = int(math.ceil(q / 100.0 * n))
    return float(ordered[min(max(rank, 1), n) - 1])


def measure(fn, warmup=5, iters=50, sync=None, min_seconds=0.0):
    """Time ``fn()`` and return honest statistics over the individual calls.

    Warmup iterations run first and are never timed; ``sync`` is invoked once
    after them so their outstanding asynchronous work cannot land inside the
    first measured sample. Each measured iteration is then bracketed by its own
    :func:`time.perf_counter` pair, with ``sync`` called *inside* the bracket --
    for CUDA work a timer stopped before the synchronize measures the kernel
    launch, not the kernel.

    Args:
        fn: zero-argument callable to time. Its return value is discarded.
        warmup: untimed iterations to run first.
        iters: minimum number of timed iterations.
        sync: optional zero-argument callable run after each iteration and once
            after warmup. Pass :func:`cuda_sync`'s result for GPU work.
        min_seconds: if positive, keep iterating past ``iters`` until this much
            wall time has elapsed inside the measured loop -- the way to get a
            meaningful p99 out of a kernel too fast for 50 samples to resolve.

    Returns:
        LatencyStats: the summary, carrying its raw samples.

    Raises:
        TypeError: if ``fn`` (or a given ``sync``) is not callable.
        ValueError: if ``iters < 1``, ``warmup < 0`` or ``min_seconds < 0``.
    """
    if not callable(fn):
        raise TypeError("measure() needs a zero-argument callable, got %r" % (fn,))
    if sync is not None and not callable(sync):
        raise TypeError("sync must be a zero-argument callable, got %r" % (sync,))
    if iters < 1:
        raise ValueError("iters must be >= 1, got %r" % (iters,))
    if warmup < 0:
        raise ValueError("warmup must be >= 0, got %r" % (warmup,))
    if min_seconds < 0.0:
        raise ValueError("min_seconds must be >= 0, got %r" % (min_seconds,))

    for _ in range(int(warmup)):
        fn()
    if sync is not None:
        sync()

    samples = []  # milliseconds, one entry per measured iteration
    started = None
    while True:
        t0 = _perf_counter()
        if started is None:
            started = t0
        fn()
        if sync is not None:
            sync()
        t1 = _perf_counter()
        samples.append((t1 - t0) * 1000.0)
        if len(samples) >= int(iters) and (t1 - started) >= min_seconds:
            break
    return LatencyStats.from_samples(samples, warmup=int(warmup))


def cuda_sync():
    """Return a synchronize callable pinned to the active GPU, or None.

    ``torch`` is imported lazily so this module stays importable in a bare
    environment. ``None`` is the *answer* ("this process has no CUDA device to
    synchronize"), not a swallowed failure: a missing torch and a CPU-only torch
    are the same fact to a caller of :func:`measure`, which simply times without
    a sync. The device index is resolved once and passed explicitly, so a later
    ``torch.cuda.set_device`` elsewhere in the process cannot silently move what
    is being synchronized mid-run.

    Returns:
        Optional[Callable[[], None]]: the synchronize callable, or None.
    """
    try:
        import torch  # noqa: PLC0415 (lazy on purpose: keeps this module bare)
    except Exception:  # noqa: BLE001 -- absence of torch is the answer, not an error
        return None
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return None
    index = int(torch.cuda.current_device())

    def _sync():
        """Block until every kernel queued on the captured device has finished."""
        torch.cuda.synchronize(index)

    return _sync


def speedup(before, after):
    """Throughput ratio ``after.hz / before.hz``.

    Args:
        before: baseline statistics.
        after: optimized statistics.

    Returns:
        float: how many times faster ``after`` is. 2.0 means twice the rate.

    Raises:
        ValueError: if the baseline rate is not positive -- a speedup against a
            zero baseline is infinite and meaningless, and must not be reported.
    """
    base_hz = before.hz
    if base_hz <= 0.0:
        raise ValueError("baseline rate %r is not positive; speedup undefined"
                         % (base_hz,))
    return after.hz / base_hz


def _desktop_clock_warnings():
    """Clock-stability warnings for an x86 dGPU / laptop GPU."""
    return [
        "GPU clocks are not pinned: 'nvidia-smi -lgc' is refused without root "
        "on this machine ('The current user does not have permission to change "
        "clocks') and Applications Clocks report N/A, so the SM clock floats "
        "freely between ~457 MHz idle and ~3090 MHz boost. Absolute Hz numbers "
        "are only comparable WITHIN one run.",
        "Because the clock floats, an A/B comparison must be INTERLEAVED "
        "(alternate baseline and candidate inside a single process) rather than "
        "run back to back; two sequential runs can differ by the boost state "
        "alone and show a 'speedup' that is pure clock.",
        "A laptop GPU thermally throttles under sustained load: check that the "
        "LAST iterations are not systematically slower than the first before "
        "trusting the mean (see drift_check()).",
    ]


def _jetson_clock_warnings(hardware):
    """Clock-stability warnings for a Jetson, including the 15 W case."""
    out = [
        "Jetson DVFS: apply BOTH 'nvpmodel -m <mode>' and 'jetson_clocks' "
        "before timing anything. Without them the GPU/EMC clock ramps during "
        "the run and the ramp shows up as latency variance, not as a slower "
        "mean -- it corrupts p90/p99 specifically.",
        "Report the nvpmodel mode next to every number and only compare runs "
        "taken in the same mode; a mode change silently rescales every result.",
    ]
    if hardware.is_15w:
        out.append(
            "This board is at a %s W budget: the SoC will cap clocks to stay "
            "inside it, so sustained numbers land below the first-iteration "
            "numbers. Interleave A/B runs and check drift_check() before "
            "quoting a mean." % (hardware.power_budget_w,))
    return out


def clock_warnings(hardware):
    """Reasons the numbers from this machine may not be reproducible.

    These belong in the report next to the measurement. The point is that a
    reader six months later can tell whether a 1.4x is a real engine win or the
    boost state of a laptop, and that a Jetson result taken without
    ``jetson_clocks`` is flagged rather than quietly believed.

    Args:
        hardware: a :class:`..hardware.detect.HardwareProfile`; the branch is on
            ``.is_jetson`` and ``.is_15w``.

    Returns:
        List[str]: one sentence per reason, ready to print or embed in JSON.

    Raises:
        ValueError: if ``hardware`` is None -- call ``detect()`` and pass the
            real profile rather than getting warnings for a machine nobody
            identified.
    """
    if hardware is None:
        raise ValueError("clock_warnings() needs a HardwareProfile; call "
                         "sparx_agency.tasks.common.hardware.detect.detect()")
    if hardware.is_jetson:
        return _jetson_clock_warnings(hardware)
    return _desktop_clock_warnings()


def drift_check(samples_ms, first_frac=0.25, last_frac=0.25, tol=0.10):
    """Compare the first and last slice of a run to catch a drifting machine.

    This is the check that makes a laptop measurement trustworthy. A rising
    trend is thermal throttle or a clock droop -- the run got slower as it went,
    so the mean flatters the steady state the drone will actually fly with. A
    falling trend means warmup never finished (a cold cache, a lazily built
    cuDNN plan, a TensorRT context still on its first shapes), so the mean is
    pessimistic. Both invalidate the run, so both fail; the message says which.

    Args:
        samples_ms: per-iteration durations, in the order they were measured.
        first_frac: fraction of the run forming the leading slice.
        last_frac: fraction forming the trailing slice.
        tol: allowed relative change, e.g. 0.10 for 10 %.

    Returns:
        Tuple[bool, str]: ``(ok, message)``. ``ok`` is True when the two slice
        means agree within ``tol``; the message always states the numbers.

    Raises:
        ValueError: on fewer than two samples, a fraction outside ``(0, 1]``, a
            negative ``tol``, or slices that would overlap (run more iterations
            rather than measuring the same samples twice).
    """
    vals = [float(v) for v in samples_ms]
    if len(vals) < 2:
        raise ValueError("drift_check needs at least 2 samples, got %d" % len(vals))
    for name, frac in (("first_frac", first_frac), ("last_frac", last_frac)):
        if not 0.0 < frac <= 1.0:
            raise ValueError("%s must be in (0, 1], got %r" % (name, frac))
    if tol < 0.0:
        raise ValueError("tol must be >= 0, got %r" % (tol,))

    n = len(vals)
    n_first = max(1, int(round(n * first_frac)))
    n_last = max(1, int(round(n * last_frac)))
    if n_first + n_last > n:
        raise ValueError(
            "first/last slices overlap (%d + %d > %d samples); run more "
            "iterations or shrink the fractions" % (n_first, n_last, n))

    head = statistics.fmean(vals[:n_first])
    tail = statistics.fmean(vals[-n_last:])
    if head <= 0.0:
        raise ValueError("leading slice mean %r is not positive; the timing run "
                         "is broken" % (head,))
    drift = (tail - head) / head
    detail = ("first %d iters %.3f ms, last %d iters %.3f ms (%+.1f%%, tol %.1f%%)"
              % (n_first, head, n_last, tail, drift * 100.0, tol * 100.0))
    if drift > tol:
        return False, ("SLOWING DOWN: " + detail + " -- thermal throttle or a "
                       "clock droop; the mean overstates steady-state rate")
    if drift < -tol:
        return False, ("SPEEDING UP: " + detail + " -- warmup was too short "
                       "(cold cache or first-shape build); the mean is pessimistic")
    return True, "stable: " + detail
