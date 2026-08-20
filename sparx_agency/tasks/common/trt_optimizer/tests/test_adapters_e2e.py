"""The whole pipeline, driven by a reference adapter from another domain entirely.

This is the single most valuable test in the package. Everything else proves a
module; this proves the *claim* -- that a toolkit written beside a
vision-language-action policy optimizes an ImageNet classifier without one line
of it knowing what a classifier is.

Nothing here is a stub. ``plan`` dissects and profiles resnet18 on the GPU and
decides what is worth converting, ``export`` writes a real ONNX through the op
gate, ``build`` produces a real FP16 TensorRT engine and verifies its weight
bytes per parameter, the shared
:class:`~sparx_agency.core.planning.vlas.common.trt.engine_runner.TRTEngineRunner`
runs it, and ``bench`` compares the engine's *decisions* against the torch
reference through the adapter's own metrics and gates.

The weights are untrained on purpose -- ``ImageClassifierAdapter`` seeds its
construction, so the CPU model that was exported and the CUDA model that is the
reference are bit-identical, and no checkpoint is downloaded. That makes the
accuracy comparison meaningful (same weights, two runtimes) while saying nothing
about accuracy on real images, which is what
:meth:`ImageClassifierAdapter.scenarios` warns about at length.

Run it with the interpreter that owns TensorRT::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
      ~/miniconda3/envs/navdp/bin/python -m pytest \\
      sparx_agency/tasks/common/trt_optimizer/tests/test_adapters_e2e.py -q -s
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the export stage needs torch")
pytest.importorskip("torchvision", reason="the reference adapter is torchvision")
trt = pytest.importorskip("tensorrt", reason="no TensorRT in this interpreter")
pytest.importorskip("onnx", reason="the export stage needs onnx")
pytest.importorskip("pycuda", reason="TRTEngineRunner needs pycuda")

from sparx_agency.core.planning.vlas.common.trt.engine_runner import (  # noqa: E402
    TRTEngineRunner)
from sparx_agency.tasks.common.trt_optimizer import (  # noqa: E402
    adapter as adapter_mod, decide, dissect, pipeline, target as target_mod)
from sparx_agency.tasks.common.trt_optimizer.adapters import (  # noqa: E402
    ImageClassifierAdapter)
from sparx_agency.tasks.common.trt_optimizer.bench import (  # noqa: E402
    latency, report as report_mod)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="building and running an engine needs a CUDA device")

#: How many scenarios the accuracy comparison averages over. Each is one image,
#: so ``top1_agreement`` is the fraction of them whose prediction survived.
SCENARIOS = 8


class _Run(object):
    """Everything one end-to-end run produced, so each stage can be asserted."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Drive plan -> export -> build -> run -> bench once, for the whole module.

    The CUDA model is released at teardown rather than left to the end of the
    process. It matters: the next module in the suite profiles a tiny network
    whose per-component shares shift measurably while this one is still
    resident, and a timing-sensitive test should not fail because of what an
    earlier module forgot to hand back.
    """
    out = tmp_path_factory.mktemp("classifier_e2e")
    adapter = adapter_mod.check(ImageClassifierAdapter())
    target = target_mod.resolve()
    target_mod.require_buildable(target, "fp16")

    plan_obj, verdicts = pipeline.plan(adapter, checkpoint=None, device="cuda",
                                       scenarios=SCENARIOS, target=target,
                                       warmup=3, iters=10)
    manifest = pipeline.export(adapter, checkpoint=None, out_dir=out / "onnx",
                               device="cpu")

    model = adapter.patch(adapter.load(device="cuda"))
    params = dissect.total_params(model)
    started = time.perf_counter()
    engines = pipeline.build(out / "onnx", out / "engines", target=target,
                             precision="fp16",
                             param_counts={adapter.graph_key: params})
    build_seconds = time.perf_counter() - started

    runner = TRTEngineRunner(engines[adapter.graph_key])
    runtimes = {adapter.graph_key: runner}
    scenarios = adapter.scenarios(SCENARIOS, seed=0)

    def _reference(scenario):
        return adapter.run_reference(model, scenario)

    before = latency.measure(lambda: _reference(scenarios[0]), warmup=10,
                             iters=60, sync=latency.cuda_sync())
    report = pipeline.bench(adapter, runtimes, _reference, scenarios, before,
                            target=target, precision="fp16", plan_obj=plan_obj)

    stacked = adapter.decision_metrics(
        np.concatenate([_reference(s) for s in scenarios], axis=0),
        np.concatenate([adapter.run_engines(runtimes, s) for s in scenarios],
                       axis=0))
    result = _Run(adapter=adapter, target=target, plan=plan_obj,
                  verdicts=verdicts, manifest=manifest, params=params,
                  engine_path=engines[adapter.graph_key], runner=runner,
                  build_seconds=build_seconds, report=report, metrics=stacked,
                  out=out)
    yield result
    # Release the torch side and let the allocator hand the VRAM back. The
    # engine runner is deliberately NOT torn down: it owns a pycuda device
    # allocation and a retained primary context, and destroying those from a
    # fixture finalizer races the interpreter's own CUDA teardown and can fault
    # a process that has already reported every test as passed.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_the_plan_was_decided_from_measurements(run):
    """Stage 1-3: every component was profiled, and the trunk earns an engine."""
    assert run.plan.decision_ms() is not None
    assert run.plan.baseline_hz > 0.0
    assert sum(c.params for c in run.plan.components) == run.params
    converted = decide.convertible(run.verdicts)
    assert converted, [v.action for v in run.verdicts]
    # Nothing was converted for being easy: everything named is a measured
    # share of a profiled budget, and the cheap layers are left alone.
    assert any(v.action == "leave_in_torch" for v in run.verdicts)


def test_the_exported_graph_is_the_one_the_adapter_declared(run):
    """Stage 4: the manifest names the adapter's engine key, IO and shapes."""
    graphs = run.manifest["graphs"]
    assert list(graphs) == [run.adapter.graph_key]
    entry = graphs[run.adapter.graph_key]
    assert entry["inputs"] == ["image"]
    assert entry["outputs"] == ["logits"]
    assert entry["shapes"] == {"image": [1, 3, 224, 224]}
    assert entry["opset"] >= 17
    assert (run.out / "onnx" / entry["onnx"]).is_file()


def test_the_engine_is_real_fp16_and_locked_to_this_device(run):
    """Stage 5: a built engine proves nothing until its precision is verified.

    On a strongly-typed TensorRT the precision is whatever the ONNX carried, so
    the sidecar's measured bytes per parameter is the only evidence that FP16
    was not silently widened back to FP32.
    """
    sidecar = json.loads(Path(str(run.engine_path) + ".json").read_text())
    assert run.engine_path.is_file()
    assert sidecar["precision"] == "fp16"
    assert sidecar["sm"] == run.target.hardware.sm
    assert sidecar["trt_version"] == str(trt.__version__)
    assert sidecar["verification"]["verified"] is True
    assert sidecar["verification"]["bytes_per_param"] <= 2.05


def test_the_engine_makes_the_same_predictions_as_torch(run):
    """Stage 6: the gated metric. Not one prediction moved."""
    assert run.runner.input_names == ["image"]
    assert run.runner.output_names == ["logits"]
    assert run.metrics["top1_agreement"] == 1.0
    assert run.metrics["top5_agreement"] == 1.0
    # The diagnostics are reported, and are exactly what FP16 rounding costs.
    assert 0.0 < run.metrics["mean_abs_logit_error"] < 0.05
    assert run.metrics["max_softmax_delta"] < 0.05


def test_the_report_passes_its_gates_and_shows_a_speedup(run):
    """Stage 7: the deliverable, gated by the adapter's own decision metrics."""
    report = run.report
    assert report.passed is True
    assert report.speedup > 1.0
    gated = dict((row.metric, row) for row in report.quality)
    assert gated["top1_agreement"].measured == 1.0
    assert gated["top1_agreement"].passed is True
    assert gated["mean_abs_logit_error"].threshold == "(diagnostic)"
    markdown = report_mod.render_markdown(report)
    assert "top1_agreement" in markdown and "**PASS**" in markdown


def test_print_the_measured_numbers(run, capsys):
    """Not an assertion: put the real numbers in the run log."""
    with capsys.disabled():
        print("\n" + report_mod.summarize(run.report))
        print("torch  : %s" % run.report.before)
        print("TRT    : %s" % run.report.after)
        print("speedup: %.2fx (%.0f -> %.0f Hz)"
              % (run.report.speedup, run.report.before.hz, run.report.after.hz))
        print("metrics: %s" % run.metrics)
        print("engine : %s (%.1f MB, built in %.1f s for %d params)"
              % (run.engine_path.name,
                 run.engine_path.stat().st_size / 1e6, run.build_seconds,
                 run.params))
    assert True
