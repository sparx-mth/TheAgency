"""Tests for the Stage-2 torch profiler.

Split in two on purpose. The pure-logic half -- overlap detection and writing
measurements back into an inventory -- runs in the repo's torch-free ``.venv``,
because that logic is what the rest of the pipeline reasons with and it must
never need a GPU to be verified. The second half needs a real
``torch.nn.Module`` (hooks, call counting, hook removal on a raising run) and is
skipped wherever torch is absent.

The load-bearing assertion in the whole file is that a block called three times
inside one forward is reported as three calls: the double-count and the
under-count are the two ways this profiler could quietly produce a wrong
Amdahl share.
"""
from __future__ import annotations

import sys
import types

import pytest

from sparx_agency.tasks.common.trt_optimizer import profile_torch
from sparx_agency.tasks.common.trt_optimizer.spec import Cadence, Component

try:  # the torch-free venv is a supported place to run this file
    import torch
    HAS_TORCH = True
    HAS_CUDA = bool(torch.cuda.is_available())
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")


# --------------------------------------------------------------------------
# A duck-typed module tree, so the reporting logic is testable without torch.
# --------------------------------------------------------------------------
class FakeModule(object):
    """Minimal stand-in exposing the two methods the profiler resolves with."""

    def __init__(self, children=None):
        self._children = dict(children or {})

    def named_modules(self, prefix=""):
        """Yield ``(dotted_name, module)`` for self and every descendant."""
        yield (prefix, self)
        for name, child in self._children.items():
            child_prefix = name if not prefix else prefix + "." + name
            for item in child.named_modules(child_prefix):
                yield item

    def get_submodule(self, target):
        """Resolve a dotted path, raising AttributeError like torch does."""
        if target == "":
            return self
        node = self
        for part in target.split("."):
            if part not in node._children:
                raise AttributeError("no child %r" % (part,))
            node = node._children[part]
        return node


def _tree():
    """backbone{block{linear}}, head, and 'shortcut' aliasing backbone.block."""
    linear = FakeModule()
    block = FakeModule({"linear": linear})
    backbone = FakeModule({"block": block})
    head = FakeModule()
    return FakeModule({"backbone": backbone, "head": head, "shortcut": block})


# --------------------------------------------------------------------------
# detect_overlap
# --------------------------------------------------------------------------
def test_detect_overlap_reports_nested_names():
    pairs = profile_torch.detect_overlap(
        _tree(), ["backbone", "backbone.block", "head"])
    assert pairs == [("backbone", "backbone.block")]


def test_detect_overlap_is_empty_for_a_clean_partition():
    assert profile_torch.detect_overlap(_tree(), ["backbone", "head"]) == []


def test_detect_overlap_finds_containment_the_names_hide():
    # 'shortcut' is bound to the same object as backbone.block, so summing
    # backbone and shortcut double-counts even though neither name says so.
    pairs = profile_torch.detect_overlap(_tree(), ["backbone", "shortcut"])
    assert pairs == [("backbone", "shortcut")]


def test_detect_overlap_reports_the_full_chain():
    pairs = profile_torch.detect_overlap(
        _tree(), ["backbone", "backbone.block", "backbone.block.linear"])
    assert pairs == [("backbone", "backbone.block"),
                     ("backbone", "backbone.block.linear"),
                     ("backbone.block", "backbone.block.linear")]


def test_detect_overlap_treats_the_root_as_an_ancestor():
    pairs = profile_torch.detect_overlap(_tree(), ["", "head"])
    assert pairs == [("", "head")]


def test_detect_overlap_collapses_duplicate_names():
    pairs = profile_torch.detect_overlap(
        _tree(), ["backbone", "backbone", "backbone.block"])
    assert pairs == [("backbone", "backbone.block")]


def test_detect_overlap_never_raises_on_an_unknown_name():
    pairs = profile_torch.detect_overlap(
        _tree(), ["ghost", "ghost.inner", "head"])
    assert pairs == [("ghost", "ghost.inner")]


def test_detect_overlap_does_not_call_two_names_for_one_module_nesting():
    block = FakeModule()
    model = FakeModule({"a": block, "b": block})
    assert profile_torch.detect_overlap(model, ["a", "b"]) == []


# --------------------------------------------------------------------------
# fill_latencies
# --------------------------------------------------------------------------
def _denoiser(calls_per_decision=1.0):
    return Component(name="head.denoiser", params=2_000_000,
                     cadence=Cadence.PER_STEP,
                     calls_per_decision=calls_per_decision)


def test_fill_latencies_overrides_a_wrong_declared_cadence():
    component = _denoiser(calls_per_decision=1.0)  # the adapter guessed
    measured = {"head.denoiser": {"ms_per_call": 4.0, "calls_per_run": 20.0}}

    components, notes = profile_torch.fill_latencies([component], measured)

    assert components[0] is component
    assert component.latency_ms == 4.0
    assert component.calls_per_decision == 20.0
    assert component.decision_ms == 80.0
    assert len(notes) == 1
    note = notes[0]
    assert "head.denoiser" in note and "1" in note and "20" in note
    assert "measured cadence wins" in note


def test_fill_latencies_is_silent_when_the_declaration_was_right():
    component = _denoiser(calls_per_decision=20.0)
    measured = {"head.denoiser": {"ms_per_call": 4.0, "calls_per_run": 20.0}}

    _, notes = profile_torch.fill_latencies([component], measured)

    assert notes == []
    assert component.latency_ms == 4.0
    assert component.calls_per_decision == 20.0


def test_fill_latencies_overrides_a_fractional_cadence():
    component = Component(name="system2", cadence=Cadence.PER_PLAN,
                          calls_per_decision=0.125)
    measured = {"system2": {"ms_per_call": 300.0, "calls_per_run": 0.25}}

    _, notes = profile_torch.fill_latencies([component], measured)

    assert component.calls_per_decision == 0.25
    assert len(notes) == 1


def test_fill_latencies_refuses_to_call_an_unseen_component_free():
    component = Component(name="rescue", cadence=Cadence.ON_DEMAND)
    measured = {"rescue": {"ms_per_call": 0.0, "calls_per_run": 0.0}}

    _, notes = profile_torch.fill_latencies([component], measured)

    assert component.latency_ms is None  # unmeasured, NOT zero
    assert component.calls_per_decision == 1.0
    assert len(notes) == 1
    assert "never called" in notes[0]


def test_fill_latencies_notes_a_component_with_no_measurement_at_all():
    component = _denoiser()
    _, notes = profile_torch.fill_latencies([component], {})

    assert component.latency_ms is None
    assert len(notes) == 1
    assert "was not measured" in notes[0]


def test_fill_latencies_accepts_a_bare_float_measurement():
    component = _denoiser(calls_per_decision=20.0)
    _, notes = profile_torch.fill_latencies([component], {"head.denoiser": 4.5})

    assert component.latency_ms == 4.5
    assert component.calls_per_decision == 20.0  # unknown counts change nothing
    assert notes == []


def test_fill_latencies_lets_an_explicit_call_count_win():
    component = _denoiser(calls_per_decision=1.0)
    measured = {"head.denoiser": {"ms_per_call": 4.0, "calls_per_run": 20.0}}

    profile_torch.fill_latencies([component], measured,
                                 calls_per_run={"head.denoiser": 8.0})

    assert component.calls_per_decision == 8.0


def test_fill_latencies_raises_on_a_malformed_measurement():
    with pytest.raises(KeyError):
        profile_torch.fill_latencies([_denoiser()],
                                     {"head.denoiser": {"mean": 4.0}})


def test_fill_latencies_leaves_the_decision_budget_undefined_when_incomplete():
    from sparx_agency.tasks.common.trt_optimizer.spec import Plan

    measured = _denoiser(calls_per_decision=20.0)
    unseen = Component(name="rescue", cadence=Cadence.ON_DEMAND)
    plan = Plan(model="fake", components=[measured, unseen])
    profile_torch.fill_latencies(
        plan.components,
        {"head.denoiser": {"ms_per_call": 4.0, "calls_per_run": 20.0},
         "rescue": {"ms_per_call": 0.0, "calls_per_run": 0.0}})

    assert plan.decision_ms() is None


# --------------------------------------------------------------------------
# profile_end_to_end -- delegation to the shared bench timer
# --------------------------------------------------------------------------
_LATENCY_MODULE = "sparx_agency.tasks.common.trt_optimizer.bench.latency"


def _fake_latency_module(record):
    module = types.ModuleType(_LATENCY_MODULE)

    def measure(fn, warmup=5, iters=50, sync=None, min_seconds=0.0):
        record["call"] = (fn, warmup, iters, sync)
        return "STATS"

    def cuda_sync():
        record["cuda_sync"] = record.get("cuda_sync", 0) + 1
        return "AUTO_SYNC"

    module.measure = measure
    module.cuda_sync = cuda_sync
    return module


def test_profile_end_to_end_forwards_to_bench_measure(monkeypatch):
    record = {}
    monkeypatch.setitem(sys.modules, _LATENCY_MODULE,
                        _fake_latency_module(record))

    def run():
        return None

    assert profile_torch.profile_end_to_end(run, warmup=1, iters=2) == "STATS"
    assert record["call"] == (run, 1, 2, "AUTO_SYNC")


def test_profile_end_to_end_keeps_an_explicit_sync(monkeypatch):
    record = {}
    monkeypatch.setitem(sys.modules, _LATENCY_MODULE,
                        _fake_latency_module(record))

    def sync():
        return None

    profile_torch.profile_end_to_end(lambda: None, sync=sync)

    assert record["call"][3] is sync
    assert "cuda_sync" not in record  # auto-detection not consulted


def test_profile_end_to_end_returns_real_latency_stats():
    counter = {"n": 0}

    def run():
        counter["n"] += 1

    stats = profile_torch.profile_end_to_end(run, warmup=2, iters=3)

    assert counter["n"] == 5
    assert stats.iters == 3
    assert stats.mean_ms >= 0.0


# --------------------------------------------------------------------------
# argument validation (no torch needed -- it is checked before the import)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"component_names": []},
    {"component_names": ["a"], "iters": 0},
    {"component_names": ["a"], "warmup": -1},
])
def test_profile_components_rejects_a_useless_run(kwargs):
    names = kwargs.pop("component_names")
    with pytest.raises(ValueError):
        profile_torch.profile_components(object(), lambda: None, names,
                                         **kwargs)


@pytest.mark.skipif(HAS_TORCH, reason="only meaningful without torch")
def test_profile_components_names_the_missing_dependency():
    with pytest.raises(ImportError) as excinfo:
        profile_torch.profile_components(object(), lambda: None, ["a"])
    assert "torch" in str(excinfo.value)


# --------------------------------------------------------------------------
# the real thing: hooks on a real nn.Module
# --------------------------------------------------------------------------
@pytest.fixture
def tiny_model():
    """A model whose inner block runs three times inside one forward."""
    torch_mod = pytest.importorskip("torch")
    nn = torch_mod.nn

    class Block(nn.Module):
        def __init__(self):
            super(Block, self).__init__()
            self.fc = nn.Linear(256, 256)

        def forward(self, x):
            return self.fc(x).relu()

    class Head(nn.Module):
        """Stands in for a K-step denoiser: one module, K calls."""

        def __init__(self, steps):
            super(Head, self).__init__()
            self.block = Block()
            self.steps = steps

        def forward(self, x):
            for _ in range(self.steps):
                x = self.block(x)
            return x

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.stem = nn.Linear(256, 256)
            self.head = Head(steps=3)

        def forward(self, x):
            return self.head(self.stem(x))

    model = Model().eval()
    return model


def _runner(model, device=None):
    torch_mod = pytest.importorskip("torch")
    x = torch_mod.randn(64, 256)
    if device is not None:
        model.to(device)
        x = x.to(device)

    def run():
        with torch_mod.no_grad():
            model(x)

    return run


def test_profile_components_counts_every_call_of_a_looped_block(tiny_model):
    run = _runner(tiny_model)
    out = profile_torch.profile_components(
        tiny_model, run, ["head", "head.block", "stem"], warmup=2, iters=5)

    assert out["head.block"]["calls_per_run"] == 3.0
    assert out["head"]["calls_per_run"] == 1.0
    assert out["stem"]["calls_per_run"] == 1.0
    for name in out:
        assert 0.0 < out[name]["ms_per_call"] < 1000.0
    # 'head' wraps three 'head.block' calls, so it must be the slower one.
    assert out["head"]["ms_per_call"] > 1.5 * out["head.block"]["ms_per_call"]


def test_profile_components_removes_every_hook(tiny_model):
    run = _runner(tiny_model)
    profile_torch.profile_components(tiny_model, run, ["head", "head.block"],
                                     warmup=1, iters=2)

    for module in tiny_model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_profile_components_removes_hooks_when_the_run_raises(tiny_model):
    def run():
        raise RuntimeError("out of memory, allegedly")

    with pytest.raises(RuntimeError):
        profile_torch.profile_components(tiny_model, run, ["head"], warmup=0,
                                         iters=1)

    for module in tiny_model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_profile_components_raises_on_an_unknown_submodule(tiny_model):
    with pytest.raises(KeyError):
        profile_torch.profile_components(tiny_model, _runner(tiny_model),
                                         ["head.ghost"], warmup=0, iters=1)


def test_profile_components_accepts_an_explicit_cpu_device(tiny_model):
    out = profile_torch.profile_components(
        tiny_model, _runner(tiny_model), ["head.block"], warmup=1, iters=3,
        device="cpu")

    assert out["head.block"]["calls_per_run"] == 3.0


def test_profile_components_rejects_cuda_it_cannot_reach(tiny_model,
                                                         monkeypatch):
    torch_mod = pytest.importorskip("torch")
    monkeypatch.setattr(torch_mod.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError) as excinfo:
        profile_torch.profile_components(tiny_model, _runner(tiny_model),
                                         ["head"], warmup=0, iters=1,
                                         device="cuda")
    assert "CUDA" in str(excinfo.value)


def test_detect_overlap_flags_the_real_nesting(tiny_model):
    pairs = profile_torch.detect_overlap(tiny_model,
                                         ["head", "head.block", "stem"])
    assert ("head", "head.block") in pairs
    assert len(pairs) == 1


def test_measured_cadence_corrects_a_wrong_inventory(tiny_model):
    """End to end: a declared 1.0 becomes the measured 3.0 with a note."""
    declared = Component(name="head.block", cadence=Cadence.PER_STEP,
                         calls_per_decision=1.0)
    measured = profile_torch.profile_components(
        tiny_model, _runner(tiny_model), ["head.block"], warmup=1, iters=3)

    components, notes = profile_torch.fill_latencies([declared], measured)

    assert components[0].calls_per_decision == 3.0
    assert components[0].latency_ms > 0.0
    assert len(notes) == 1 and "measured cadence wins" in notes[0]


def test_peak_memory_bytes_runs_the_callable_once(tiny_model):
    counter = {"n": 0}
    inner = _runner(tiny_model)

    def run():
        counter["n"] += 1
        inner()

    peak = profile_torch.peak_memory_bytes(run)

    assert counter["n"] == 1
    assert isinstance(peak, int) and peak >= 0


@needs_cuda
def test_profile_components_on_cuda(tiny_model):
    run = _runner(tiny_model, device="cuda")
    out = profile_torch.profile_components(
        tiny_model, run, ["head", "head.block"], warmup=3, iters=5)

    assert out["head.block"]["calls_per_run"] == 3.0
    assert out["head"]["ms_per_call"] > out["head.block"]["ms_per_call"]
    assert 0.0 < out["head"]["ms_per_call"] < 1000.0


@needs_cuda
def test_peak_memory_bytes_is_positive_on_cuda(tiny_model):
    assert profile_torch.peak_memory_bytes(
        _runner(tiny_model, device="cuda")) > 0


@needs_cuda
def test_the_cuda_sync_is_what_makes_the_number_real(monkeypatch):
    """Deleting the synchronize does not make the profiler faster, only wrong.

    A queue-deep GPU workload returns from ``forward`` in microseconds while
    the kernels run for milliseconds. Timed without a synchronize it is
    reported as effectively free -- the failure this module exists to prevent,
    and one that leaves no trace in the output except an implausibly small
    number. Measured on an RTX 5070 the gap is over 1000x.
    """
    nn = torch.nn

    class Big(nn.Module):
        def __init__(self):
            super(Big, self).__init__()
            self.w = nn.Parameter(torch.randn(2048, 2048))

        def forward(self, x):
            for _ in range(4):
                x = x @ self.w
            return x

    class Wrapper(nn.Module):
        def __init__(self):
            super(Wrapper, self).__init__()
            self.big = Big()

        def forward(self, x):
            return self.big(x)

    model = Wrapper().eval().cuda()
    x = torch.randn(2048, 2048, device="cuda")

    def run():
        with torch.no_grad():
            model(x)

    synced = profile_torch.profile_components(model, run, ["big"], warmup=2,
                                              iters=5)["big"]["ms_per_call"]
    monkeypatch.setattr(profile_torch, "_make_sync",
                        lambda *args, **kwargs: (lambda: None))
    unsynced = profile_torch.profile_components(model, run, ["big"], warmup=2,
                                                iters=5)["big"]["ms_per_call"]

    assert synced > 5.0 * unsynced
