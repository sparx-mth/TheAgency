"""Tests for the honest latency harness.

Timing tests that actually sleep are slow and flaky, and a flaky timing test
gets muted, which is how a benchmark stops being trusted. So every assertion
here is exact: :func:`latency._perf_counter` is monkeypatched with a scripted
sequence of timestamps, which makes percentiles, warmup exclusion, sync-call
counts and the ``min_seconds`` extension deterministic facts rather than
tolerances. The scripted clock raises when exhausted, so an unexpected extra
clock read fails loudly instead of drifting into a real timer.
"""
from __future__ import annotations

import sys
import types

import pytest

from sparx_agency.tasks.common.hardware.detect import HardwareProfile
from sparx_agency.tasks.common.trt_optimizer.bench import latency
from sparx_agency.tasks.common.trt_optimizer.bench.latency import (
    LatencyStats, clock_warnings, cuda_sync, drift_check, measure, percentile,
    speedup,
)


class ScriptedClock(object):
    """A perf_counter stand-in returning a fixed sequence of seconds."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self):
        if self.calls >= len(self.values):
            raise AssertionError(
                "scripted clock exhausted after %d reads" % self.calls)
        value = self.values[self.calls]
        self.calls += 1
        return value


def pairs_for(durations_s, gap_s=0.0):
    """Timestamp sequence giving each iteration exactly ``durations_s[i]``."""
    out = []
    now = 0.0
    for d in durations_s:
        out.append(now)
        now += d
        out.append(now)
        now += gap_s
    return out


def install_clock(monkeypatch, values):
    """Patch the module's private clock seam and return the scripted clock."""
    clock = ScriptedClock(values)
    monkeypatch.setattr(latency, "_perf_counter", clock)
    return clock


def x86_profile():
    """A laptop dGPU profile (the machine this toolkit is developed on)."""
    return HardwareProfile(arch="x86_64", is_jetson=False,
                           gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
                           compute_capability=(12, 0))


def jetson_15w_profile():
    """An Orin profile pinned to a 15 W nvpmodel budget."""
    return HardwareProfile(arch="aarch64", is_jetson=True,
                           gpu_name="Jetson AGX Orin", jetson_model="Jetson AGX Orin",
                           nvpmodel_id=3, nvpmodel_name="MODE_15W",
                           power_budget_w=15, compute_capability=(8, 7))


# --------------------------------------------------------------------------
# percentiles
# --------------------------------------------------------------------------

def test_percentiles_are_exact_nearest_rank(monkeypatch):
    """1..10 ms scripted: nearest rank picks observed samples, never a blend."""
    install_clock(monkeypatch, pairs_for([i / 1000.0 for i in range(1, 11)]))
    stats = measure(lambda: None, warmup=0, iters=10)

    assert stats.iters == 10
    assert stats.mean_ms == pytest.approx(5.5)
    assert stats.p50_ms == pytest.approx(5.0)   # ceil(0.50*10) = 5 -> 5 ms
    assert stats.p90_ms == pytest.approx(9.0)   # ceil(0.90*10) = 9 -> 9 ms
    assert stats.p99_ms == pytest.approx(10.0)  # ceil(0.99*10) = 10 -> 10 ms
    assert stats.min_ms == pytest.approx(1.0)
    assert stats.max_ms == pytest.approx(10.0)


def test_percentile_is_not_interpolated():
    """Every returned value is a member of the sample."""
    samples = [1.0, 2.0, 4.0, 8.0]
    for q in (0.0, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0):
        assert percentile(samples, q) in samples
    assert percentile(samples, 50.0) == 2.0
    assert percentile(samples, 100.0) == 8.0
    assert percentile(samples, 0.0) == 1.0


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        percentile([], 50.0)
    with pytest.raises(ValueError):
        percentile([1.0], 101.0)


def test_percentile_ignores_input_order():
    assert percentile([9.0, 1.0, 5.0], 50.0) == 5.0


# --------------------------------------------------------------------------
# LatencyStats
# --------------------------------------------------------------------------

def test_hz_is_the_reciprocal_mean():
    stats = LatencyStats.from_samples([10.0, 10.0, 10.0])
    assert stats.hz == pytest.approx(100.0)


def test_hz_raises_on_a_degenerate_mean():
    stats = LatencyStats.from_samples([0.0, 0.0])
    with pytest.raises(ValueError):
        stats.hz


def test_as_dict_is_flat_and_carries_hz():
    stats = LatencyStats.from_samples([10.0, 20.0], warmup=3)
    d = stats.as_dict()
    assert d["hz"] == pytest.approx(1000.0 / 15.0)
    assert d["iters"] == 2 and d["warmup"] == 3
    assert "samples_ms" not in d
    assert all(isinstance(v, (int, float)) for v in d.values())


def test_str_is_one_line_and_survives_a_zero_mean():
    stats = LatencyStats.from_samples([12.0, 12.0])
    text = str(stats)
    assert "\n" not in text and "p99" in text and "Hz" in text
    assert "-- Hz" in str(LatencyStats.from_samples([0.0]))


def test_from_samples_rejects_an_empty_run():
    with pytest.raises(ValueError):
        LatencyStats.from_samples([])


def test_std_is_zero_for_a_single_sample():
    assert LatencyStats.from_samples([7.0]).std_ms == 0.0


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------

def test_warmup_iterations_run_but_are_not_timed(monkeypatch):
    """Warmup calls fn and consumes no clock reads, so stats see only the rest."""
    clock = install_clock(monkeypatch, pairs_for([0.002, 0.004, 0.006, 0.008]))
    calls = []
    stats = measure(lambda: calls.append(1), warmup=3, iters=4)

    assert len(calls) == 7          # 3 warmup + 4 measured
    assert stats.iters == 4         # warmup excluded from the statistics
    assert stats.warmup == 3
    assert clock.calls == 8         # two reads per measured iteration only
    assert stats.min_ms == pytest.approx(2.0)
    assert stats.max_ms == pytest.approx(8.0)
    assert stats.mean_ms == pytest.approx(5.0)


def test_sync_runs_once_after_warmup_and_once_per_iteration(monkeypatch):
    install_clock(monkeypatch, pairs_for([0.001] * 6))
    hits = []
    measure(lambda: None, warmup=4, iters=6, sync=lambda: hits.append(1))
    assert len(hits) == 7           # 1 after warmup + 1 per measured iteration


def test_sync_is_inside_the_timed_bracket(monkeypatch):
    """The sample must include the sync, or GPU time is measured as launch time."""
    install_clock(monkeypatch, pairs_for([0.005]))
    order = []
    measure(lambda: order.append("fn"), warmup=0, iters=1,
            sync=lambda: order.append("sync"))
    assert order == ["sync", "fn", "sync"]


def test_min_seconds_extends_the_run_past_iters(monkeypatch):
    """iters is a floor; min_seconds keeps going until the wall clock agrees."""
    install_clock(monkeypatch, pairs_for([0.1] * 8))
    stats = measure(lambda: None, warmup=0, iters=2, min_seconds=0.5)
    assert stats.iters == 5         # stops on the first iteration ending at >= 0.5 s


def test_min_seconds_zero_does_not_extend(monkeypatch):
    clock = install_clock(monkeypatch, pairs_for([0.1] * 4))
    stats = measure(lambda: None, warmup=0, iters=2)
    assert stats.iters == 2
    assert clock.calls == 4


def test_min_seconds_never_shortens_the_run(monkeypatch):
    install_clock(monkeypatch, pairs_for([1.0] * 3))
    assert measure(lambda: None, warmup=0, iters=3, min_seconds=0.001).iters == 3


def test_measure_keeps_the_raw_samples_for_drift_check(monkeypatch):
    install_clock(monkeypatch, pairs_for([0.001, 0.002, 0.003]))
    stats = measure(lambda: None, warmup=0, iters=3)
    assert stats.samples_ms == pytest.approx([1.0, 2.0, 3.0])


@pytest.mark.parametrize("kwargs", [
    {"iters": 0}, {"warmup": -1}, {"min_seconds": -0.5},
])
def test_measure_rejects_impossible_arguments(kwargs):
    with pytest.raises(ValueError):
        measure(lambda: None, **kwargs)


def test_measure_rejects_non_callables():
    with pytest.raises(TypeError):
        measure(object())
    with pytest.raises(TypeError):
        measure(lambda: None, iters=1, warmup=0, sync=object())


def test_measure_works_on_the_real_clock():
    """No monkeypatch: the real path produces positive, ordered statistics."""
    stats = measure(lambda: sum(range(200)), warmup=2, iters=8)
    assert stats.iters == 8
    assert 0.0 < stats.min_ms <= stats.p50_ms <= stats.p99_ms <= stats.max_ms
    assert stats.hz > 0.0


# --------------------------------------------------------------------------
# speedup
# --------------------------------------------------------------------------

def test_speedup_is_a_throughput_ratio():
    before = LatencyStats.from_samples([20.0])   # 50 Hz
    after = LatencyStats.from_samples([10.0])    # 100 Hz
    assert speedup(before, after) == pytest.approx(2.0)
    assert speedup(after, before) == pytest.approx(0.5)


def test_speedup_raises_on_a_non_positive_baseline():
    before = LatencyStats.from_samples([0.0])
    after = LatencyStats.from_samples([10.0])
    with pytest.raises(ValueError):
        speedup(before, after)


# --------------------------------------------------------------------------
# clock_warnings
# --------------------------------------------------------------------------

def test_clock_warnings_x86_names_the_floating_clock_and_interleaving():
    warns = clock_warnings(x86_profile())
    assert warns and all(isinstance(w, str) for w in warns)
    joined = " ".join(warns).lower()
    assert "nvidia-smi -lgc" in joined or "permission to change" in joined
    assert "interleaved" in joined
    assert "within one run" in joined
    assert "throttle" in joined and "last iterations" in joined


def test_clock_warnings_jetson_demands_nvpmodel_and_jetson_clocks():
    warns = clock_warnings(jetson_15w_profile())
    joined = " ".join(warns).lower()
    assert "nvpmodel" in joined and "jetson_clocks" in joined
    assert "dvfs" in joined and "variance" in joined


def test_clock_warnings_differ_between_x86_and_jetson():
    assert clock_warnings(x86_profile()) != clock_warnings(jetson_15w_profile())
    assert "nvidia-smi" not in " ".join(clock_warnings(jetson_15w_profile()))
    assert "jetson_clocks" not in " ".join(clock_warnings(x86_profile()))


def test_clock_warnings_15w_adds_a_budget_warning():
    hot = jetson_15w_profile()
    assert hot.is_15w
    full = HardwareProfile(arch="aarch64", is_jetson=True,
                           nvpmodel_name="MODE_50W", power_budget_w=50)
    assert not full.is_15w
    assert len(clock_warnings(hot)) > len(clock_warnings(full))
    assert "15 W budget" in " ".join(clock_warnings(hot))


def test_clock_warnings_requires_a_profile():
    with pytest.raises(ValueError):
        clock_warnings(None)


# --------------------------------------------------------------------------
# drift_check
# --------------------------------------------------------------------------

def test_drift_check_passes_a_flat_run():
    ok, msg = drift_check([10.0] * 20)
    assert ok is True
    assert "stable" in msg


def test_drift_check_catches_a_rising_trend():
    ok, msg = drift_check([10.0] * 10 + [15.0] * 10)
    assert ok is False
    assert "SLOWING DOWN" in msg and "throttle" in msg
    assert "+50.0%" in msg


def test_drift_check_catches_a_falling_trend():
    ok, msg = drift_check([20.0] * 10 + [10.0] * 10)
    assert ok is False
    assert "SPEEDING UP" in msg and "warmup" in msg


def test_drift_check_tolerance_is_relative_and_respected():
    samples = [10.0] * 10 + [10.5] * 10          # +5 %
    assert drift_check(samples, tol=0.10)[0] is True
    assert drift_check(samples, tol=0.01)[0] is False


def test_drift_check_slices_honour_the_fractions():
    """A wider trailing slice dilutes a single slow sample -- fractions matter."""
    samples = [1.0] * 7 + [1.2]                  # only the very last is slow
    # last slice = 1 sample -> +20 %, over the 10 % tolerance
    assert drift_check(samples, first_frac=0.25, last_frac=0.125)[0] is False
    # last slice = 4 samples -> +5 %, inside it
    assert drift_check(samples, first_frac=0.5, last_frac=0.5)[0] is True


def test_drift_check_reports_both_slice_means():
    _, msg = drift_check([10.0] * 4 + [12.0] * 4, first_frac=0.5, last_frac=0.5)
    assert "10.000 ms" in msg and "12.000 ms" in msg


@pytest.mark.parametrize("args,kwargs", [
    (([10.0],), {}),                                  # too few samples
    (([10.0] * 8,), {"first_frac": 0.0}),             # empty leading slice
    (([10.0] * 8,), {"last_frac": 1.5}),              # fraction out of range
    (([10.0] * 8,), {"tol": -0.1}),                   # negative tolerance
    (([10.0] * 8,), {"first_frac": 0.8, "last_frac": 0.8}),  # overlapping slices
    (([0.0] * 8,), {}),                               # zero baseline slice
])
def test_drift_check_rejects_impossible_input(args, kwargs):
    with pytest.raises(ValueError):
        drift_check(*args, **kwargs)


def test_drift_check_consumes_a_measure_result(monkeypatch):
    install_clock(monkeypatch, pairs_for([0.010] * 10 + [0.020] * 10))
    stats = measure(lambda: None, warmup=0, iters=20)
    ok, _ = drift_check(stats.samples_ms)
    assert ok is False


# --------------------------------------------------------------------------
# cuda_sync
# --------------------------------------------------------------------------

def _fake_torch(available, index=0, log=None):
    """A torch stand-in exposing only what cuda_sync touches."""
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        current_device=lambda: index,
        synchronize=lambda dev: (log if log is not None else []).append(dev),
    )
    return types.SimpleNamespace(cuda=cuda)


def test_cuda_sync_is_none_without_a_cuda_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    assert cuda_sync() is None


def test_cuda_sync_is_none_when_torch_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)   # import torch -> ImportError
    assert cuda_sync() is None


def test_cuda_sync_pins_the_device_index(monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(available=True, index=3, log=log))
    sync = cuda_sync()
    assert callable(sync)
    sync()
    sync()
    assert log == [3, 3]


def test_measure_accepts_a_cuda_sync_result(monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(available=True, index=0, log=log))
    install_clock(monkeypatch, pairs_for([0.001] * 3))
    measure(lambda: None, warmup=1, iters=3, sync=cuda_sync())
    assert len(log) == 4            # 1 after warmup + 1 per measured iteration
