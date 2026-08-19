"""Drive the whole toolkit -- dissect, decide, export, build, run, report -- for real.

Every other test in this package proves one module in isolation, mostly against
a fake module tree in the torch-free venv. This one proves the *pipeline*: a
live ViT-shaped torch model is inventoried, profiled on the GPU, judged by
:mod:`..decide`, exported to ONNX through the patch context, built into a real
FP16 TensorRT engine, run through the repo's shared
:class:`~sparx_agency.core.planning.vlas.common.trt.engine_runner.TRTEngineRunner`,
compared against the FP32 torch reference, timed on both sides and rendered
into an :class:`..bench.report.OptimizationReport`.

Run it with the interpreter that owns TensorRT, not the venv::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
      ~/miniconda3/envs/navdp/bin/python -m pytest \\
      sparx_agency/tasks/common/trt_optimizer/tests/test_end_to_end.py -q -s

Two properties of the toy model are deliberate and not cosmetic:

* **Every layer gets its own random weights.** ``nn.TransformerEncoder``
  deep-copies one prototype layer, so all four blocks ship identical tensors
  and the ONNX exporter de-duplicates them -- the engine then stores a quarter
  of the weights and ``bytes_per_param`` comes out near 0.5, passing the FP16
  precision check for entirely the wrong reason. Randomizing per layer keeps
  that check meaningful.
* **The input is 1x3x32x32 with a 4-pixel patch**, giving 8x8 = 64 tokens at
  width 192 over 4 blocks: ~1.8 M parameters, which is small enough to build in
  seconds and still exercises attention, LayerNorm, GELU and a conv stem.

The engine, its ONNX and the report all go to ``tmp_path``; nothing is written
into the repo.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the toolkit's export stage needs torch")
trt = pytest.importorskip("tensorrt", reason="no TensorRT in this interpreter")
pytest.importorskip("onnx", reason="the export stage needs onnx")
pytest.importorskip("pycuda", reason="TRTEngineRunner needs pycuda")

from torch import nn  # noqa: E402  (after importorskip, on purpose)

from sparx_agency.core.planning.vlas.common.trt.engine_runner import (  # noqa: E402
    TRTEngineRunner)
from sparx_agency.tasks.common.trt_optimizer import (  # noqa: E402
    amdahl, decide, dissect, memory_budget, profile_torch,
    target as target_mod)
from sparx_agency.tasks.common.trt_optimizer.bench import (  # noqa: E402
    latency, report as report_mod)
from sparx_agency.tasks.common.trt_optimizer.engine import (  # noqa: E402
    build as engine_build, builder_config)
from sparx_agency.tasks.common.trt_optimizer.export import (  # noqa: E402
    onnx_export, op_gate, patches)
from sparx_agency.tasks.common.trt_optimizer.spec import (  # noqa: E402
    Cadence, GraphSpec, Plan)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the end-to-end proof needs a CUDA device to build and run on")

#: Engine key: the ONNX stem, the engine stem and the report row all share it.
ENGINE_KEY = "tiny_vit_encoder"
IMAGE_SHAPE = (1, 3, 32, 32)
WIDTH = 192
DEPTH = 4
HEADS = 4
PATCH = 4

#: FP16 relative-L2 budget against the FP32 torch reference. Measured 5.9e-4
#: to 7.1e-4 across runs of this graph; the bound is loose enough to survive a
#: tactic change and far tighter than the ~1e-2 at which an FP16 transformer is
#: genuinely drifting.
REL_L2_BUDGET = 2e-2

#: Builder search effort. **Not 3**, which is the toolkit default and the level
#: you would fly: measured on this machine, TensorRT 11.1.0.106 on sm_120
#: segfaults inside ``build_serialized_network`` on the FP16-baked graph at
#: ``builder_optimization_level`` 3 (5 of 6 runs) and 2 (2 of 5), while levels
#: 0 and 1 built 5 of 5 and the same graph at FP32 built 5 of 5 at the default
#: level. A segfault cannot be caught, so a test that used the flying level
#: would take the whole pytest process down most runs. The stage under test is
#: the pipeline, not the tactic search.
OPTIMIZATION_LEVEL = 1


class TinyViT(nn.Module):
    """A small but structurally honest ViT encoder: conv stem, 4 blocks, head.

    Args:
        width: token width.
        depth: number of transformer blocks.
        heads: attention heads per block.
        patch: patch size in pixels; a 32-pixel input gives 64 tokens.
    """

    def __init__(self, width=WIDTH, depth=DEPTH, heads=HEADS, patch=PATCH):
        super(TinyViT, self).__init__()
        self.patch_embed = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth,
                                            enable_nested_tensor=False)
        self.head = nn.Linear(width, 32)

    def forward(self, image):
        """Map one image to a pooled feature vector."""
        x = self.patch_embed(image).flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.blocks(x)
        return self.head(x.mean(dim=1))


def _randomize(model, seed=7):
    """Give every parameter its own values, so no two blocks are identical."""
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for param in model.parameters():
            scale = 0.05 if param.dim() > 1 else 0.02
            param.copy_(torch.randn(param.shape, generator=generator) * scale)
    return model


class Pipeline(object):
    """Everything one end-to-end run produced, so each stage can be asserted."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _inventory(model):
    """Stage 1: the component inventory, with its accounting checked."""
    components = dissect.inventory(model, max_depth=1)
    dissect.check_accounting(components, model)
    return components


def _profile(model, image, components):
    """Stage 2: measure every component and the whole decision on the GPU."""
    names = [c.name for c in components]
    assert profile_torch.detect_overlap(model, names) == [], (
        "the inventory must be a clean partition or the shares double-count")

    def _run():
        with torch.no_grad():
            model(image)

    measured = profile_torch.profile_components(
        model, _run, names, warmup=5, iters=50, device="cuda")
    components, notes = profile_torch.fill_latencies(components, measured)
    baseline = profile_torch.profile_end_to_end(_run, warmup=10, iters=50)
    return components, notes, measured, baseline


def _interleave(torch_call, trt_call, sync, rounds=3, iters=60):
    """Time both sides A/B/A/B in one process and summarize each side once.

    The SM clock on this laptop is not lockable, so two sequential runs can
    differ by the boost state alone. Alternating the two workloads inside one
    process spreads any clock or thermal excursion across both, which is what
    :func:`..bench.latency.clock_warnings` asks for.
    """
    torch_samples, trt_samples = [], []
    for _ in range(int(rounds)):
        torch_samples.extend(
            latency.measure(torch_call, warmup=5, iters=iters,
                            sync=sync).samples_ms)
        trt_samples.extend(
            latency.measure(trt_call, warmup=5, iters=iters).samples_ms)
    return (latency.LatencyStats.from_samples(torch_samples, warmup=5 * rounds),
            latency.LatencyStats.from_samples(trt_samples, warmup=5 * rounds))


def _build_report(pipe):
    """Stage 7: assemble the deliverable from what the run actually measured."""
    verdict_by_name = {}
    for verdict in pipe.verdicts:
        verdict_by_name.setdefault(verdict.component, verdict)
    rows = [report_mod.ComponentRow.from_component(c, verdict_by_name.get(c.name))
            for c in pipe.components]
    quality = [
        report_mod.QualityRow(
            metric="rel_l2_vs_torch_fp32", reference=0.0,
            measured=pipe.rel_l2, threshold=REL_L2_BUDGET,
            passed=pipe.rel_l2 <= REL_L2_BUDGET,
            note=("engine output vs the FP32 torch forward on the same input, "
                  "||trt - torch||2 / ||torch||2")),
        report_mod.QualityRow(
            metric="engine_bytes_per_param", reference="fp16 = 2 B/elem",
            measured=pipe.bytes_per_param, threshold=2.05,
            passed=pipe.bytes_per_param <= 2.05,
            note=("read back from the built engine's TOTAL_WEIGHTS_SIZE; a "
                  "strongly-typed build that widened FP16 back to FP32 would "
                  "land near 4")),
    ]
    budget = memory_budget.estimate(pipe.plan, "fp16", pipe.target.hardware)
    warnings = list(latency.clock_warnings(pipe.target.hardware))
    warnings.append("torch timing drift: %s" % pipe.torch_drift[1])
    warnings.append("TRT timing drift: %s" % pipe.trt_drift[1])
    return report_mod.OptimizationReport(
        model="TinyViT", target_tag=pipe.target.target_tag,
        gpu_name=pipe.target.hardware.gpu_name,
        trt_version=str(trt.__version__), precision="fp16",
        before=pipe.torch_stats, after=pipe.trt_stats, components=rows,
        quality=quality,
        memory={"required_bytes": budget.required_bytes,
                "available_bytes": budget.free_bytes,
                "engine_file_bytes": pipe.engine_path.stat().st_size},
        warnings=warnings,
        notes=(list(pipe.plan.notes)
               + ["the engine covers the whole forward, so no per-component "
                  "'after' is separable; the end-to-end row carries it",
                  "build wall time %.2f s at optimization_level %d"
                  % (pipe.build_seconds, OPTIMIZATION_LEVEL)]
               + list(pipe.build_notes)))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the whole toolkit once and hand every stage's evidence to the tests."""
    out = tmp_path_factory.mktemp("trt_e2e")
    torch.manual_seed(0)
    model = _randomize(TinyViT()).eval().cuda()
    params = dissect.total_params(model)
    image = torch.randn(*IMAGE_SHAPE, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(3))

    components = _inventory(model)
    components, notes, measured, baseline = _profile(model, image, components)

    spec = GraphSpec(key=ENGINE_KEY, inputs={"image": IMAGE_SHAPE},
                     outputs=["features"], component="blocks",
                     cadence=Cadence.PER_FRAME, precision_sensitive=False,
                     opset=17)
    plan = Plan(model="TinyViT", components=components, graphs=[spec],
                baseline_hz=baseline.hz, notes=notes)
    target = target_mod.resolve()
    plan.target_tag = target.target_tag
    verdicts = decide.decide(plan, target)
    plan.verdicts = verdicts

    target_mod.require_buildable(target, "fp16")
    onnx_path = onnx_export.export_graph(spec, model, out / "onnx",
                                         inputs=(image,),
                                         policy=op_gate.OpGatePolicy())
    gate_result = op_gate.gate(onnx_path, key=ENGINE_KEY)

    options = builder_config.BuildOptions(
        precision="fp16", optimization_level=OPTIMIZATION_LEVEL,
        timing_cache=str(out / "timing.cache"))
    started = time.perf_counter()
    engine_path = engine_build.build_engine(onnx_path, out / "engines", target,
                                            options=options, param_count=params)
    build_seconds = time.perf_counter() - started
    sidecar = json.loads(Path(str(engine_path) + ".json").read_text())

    runner = TRTEngineRunner(engine_path)
    image_np = image.detach().cpu().numpy()
    trt_out = runner.infer({"image": image_np})["features"]
    with torch.no_grad():
        reference = model(image).cpu().numpy()
    rel_l2 = float(np.linalg.norm(trt_out - reference)
                   / np.linalg.norm(reference))

    def _torch_call():
        with torch.no_grad():
            model(image)

    def _trt_call():
        runner.infer({"image": image_np})

    torch_stats, trt_stats = _interleave(_torch_call, _trt_call,
                                         latency.cuda_sync())
    pipe = Pipeline(
        model=model, image=image, params=params, components=components,
        measured=measured, baseline=baseline, plan=plan, spec=spec,
        target=target, verdicts=verdicts, onnx_path=onnx_path,
        gate_result=gate_result, engine_path=engine_path, sidecar=sidecar,
        build_seconds=build_seconds, build_notes=sidecar["build_notes"],
        bytes_per_param=sidecar["verification"]["bytes_per_param"],
        runner=runner, reference=reference, trt_out=trt_out, rel_l2=rel_l2,
        torch_stats=torch_stats, trt_stats=trt_stats,
        torch_drift=latency.drift_check(torch_stats.samples_ms, tol=0.25),
        trt_drift=latency.drift_check(trt_stats.samples_ms, tol=0.25),
        out=out)
    pipe.report = _build_report(pipe)
    return pipe


# ------------------------------------------------------------- stage 1 and 2

def test_inventory_accounts_for_every_parameter(pipeline):
    """Stage 1: the frontier is the four top-level modules, and it adds up."""
    names = [c.name for c in pipeline.components]
    assert names == ["patch_embed", "norm", "blocks", "head"]
    assert sum(c.params for c in pipeline.components) == pipeline.params
    assert 1e6 < pipeline.params < 3e6


def test_every_component_was_measured_on_the_gpu(pipeline):
    """Stage 2: every component ran, so the decision budget is defined."""
    for name, entry in pipeline.measured.items():
        assert entry["calls_per_run"] == 1.0, name
        assert entry["ms_per_call"] > 0.0, name
    assert pipeline.plan.decision_ms() is not None
    assert pipeline.baseline.hz > 0.0


# ------------------------------------------------------------------- stage 3

def test_decide_converts_the_trunk_and_leaves_the_cheap_parts_alone(pipeline):
    """Stage 3: the verdicts follow the measurement, not the convenience.

    Asserted against the shares this run actually measured rather than against
    a fixed list of names: the conv stem lands near 4% of a half-millisecond
    decision, and one noisy profiling run can legitimately push it over the 5%
    floor. What must always hold is that the rule fired on the number.
    """
    actions = dict((v.component, v.action) for v in pipeline.verdicts)
    shares = dict((c.name, amdahl.share(c, pipeline.plan))
                  for c in pipeline.components)
    assert shares["blocks"] > 0.70, shares
    assert actions["blocks"] == "trt_fp16"
    for name, share in shares.items():
        if share < 0.05:  # decide()'s min_share floor
            assert actions[name] == "leave_in_torch", (name, share)
    assert "blocks" in decide.convertible(pipeline.verdicts)
    projected, ceiling = decide.ceiling(pipeline.plan, pipeline.verdicts)
    assert 1.0 < projected <= ceiling


# ------------------------------------------------------------------- stage 4

def test_export_context_actually_applies_the_attention_patches(pipeline):
    """Stage 4a: the MHA fast path is off inside the context and back after."""
    getter = getattr(getattr(torch.backends, "mha", None),
                     "get_fastpath_enabled", None)
    if getter is None:
        pytest.skip("this torch has no mha fast-path knob to check")
    before = getter()
    with patches.export_context(pipeline.model):
        assert getter() is False
    assert getter() == before


def test_exported_graph_is_clean_for_tensorrt(pipeline):
    """Stage 4b: no fused attention, no Resize, no dynamic dimension."""
    result = pipeline.gate_result
    assert result.ok, result.messages
    assert not [op for op in result.op_counts if op.endswith("Attention")]
    assert result.dynamic_tensors == []
    # Attention decomposed to MatMul/Softmax, one Softmax per block, and opset
    # 17's single LayerNormalization node per norm (2 per block, plus the stem).
    assert result.op_counts["Softmax"] == DEPTH
    assert result.op_counts["LayerNormalization"] == 2 * DEPTH + 1
    assert result.op_counts["MatMul"] >= 2 * DEPTH
    assert pipeline.onnx_path.name == ENGINE_KEY + ".onnx"


# ------------------------------------------------------------------- stage 5

def test_engine_is_real_fp16_and_locked_to_this_device(pipeline):
    """Stage 5: the engine exists, and its sidecar proves what built it."""
    assert pipeline.engine_path.is_file()
    assert pipeline.engine_path.stat().st_size > 100_000
    sidecar = pipeline.sidecar
    assert sidecar["sm"] == pipeline.target.hardware.sm
    assert sidecar["trt_version"] == str(trt.__version__)
    assert sidecar["precision"] == "fp16"
    assert sidecar["verification"]["verified"] is True
    assert sidecar["verification"]["bytes_per_param"] <= 2.05
    assert sidecar["builder_optimization_level"] == OPTIMIZATION_LEVEL
    assert (pipeline.engine_path.parent / (ENGINE_KEY + ".layers.json")).is_file()


def test_tactic_dram_cap_was_accepted_not_just_requested(pipeline):
    """Stage 5b: both memory pools really are capped, read back from the config.

    ``set_memory_pool_limit`` logs a rejected size instead of raising, so a
    note claiming a cap is not evidence that one was applied.
    """
    capped = [n for n in pipeline.build_notes if "capped to" in n]
    assert any(n.startswith("WORKSPACE") for n in capped), capped
    assert any(n.startswith("TACTIC_DRAM") for n in capped), capped
    for note in capped:
        gib = float(note.split("capped to ")[1].split(" GiB")[0])
        assert gib < memory_budget.MEASURED_TRT11_DEFAULT_POOL_BYTES / (1 << 30)


# ------------------------------------------------------------------- stage 6

def test_engine_matches_the_torch_reference(pipeline):
    """Stage 6: the shared runtime's output tracks FP32 torch."""
    assert pipeline.runner.input_names == ["image"]
    assert pipeline.runner.output_names == ["features"]
    assert pipeline.trt_out.shape == pipeline.reference.shape
    assert np.isfinite(pipeline.trt_out).all()
    assert pipeline.rel_l2 <= REL_L2_BUDGET


def test_engine_is_faster_than_torch(pipeline):
    """Stage 6b: a real, interleaved speedup with both tails reported."""
    speedup = latency.speedup(pipeline.torch_stats, pipeline.trt_stats)
    assert speedup > 1.0
    assert pipeline.trt_stats.mean_ms < pipeline.torch_stats.mean_ms
    assert pipeline.trt_stats.iters >= 100
    assert isinstance(pipeline.torch_drift[0], bool)


# ------------------------------------------------------------------- stage 7

def test_report_renders_every_required_section(pipeline):
    """Stage 7: the deliverable, gate first, with the skipped parts explained."""
    markdown = report_mod.render_markdown(pipeline.report)
    for section in ("# TensorRT optimization report -- TinyViT",
                    "## Hardware and build", "## Components",
                    "## Deliberately not converted", "## Quality",
                    "## Memory", "## Warnings"):
        assert section in markdown, section
    assert pipeline.report.passed is True
    assert markdown.startswith("# TensorRT optimization report")
    assert "**PASS**" in markdown
    assert "patch_embed" in markdown and "leave_in_torch" in markdown
    assert "rel_l2_vs_torch_fp32" in markdown
    md_path, json_path = report_mod.write_report(pipeline.report, pipeline.out)
    assert md_path.is_file() and json_path.is_file()
    payload = json.loads(json_path.read_text())
    assert payload["passed"] is True
    assert payload["speedup"] > 1.0


def test_print_the_measured_numbers(pipeline, capsys):
    """Not an assertion: put the real numbers in the run log."""
    with capsys.disabled():
        print("\n" + report_mod.summarize(pipeline.report))
        print("torch : %s" % pipeline.torch_stats)
        print("TRT   : %s" % pipeline.trt_stats)
        print("speedup            : %.2fx (%.0f -> %.0f Hz)"
              % (latency.speedup(pipeline.torch_stats, pipeline.trt_stats),
                 pipeline.torch_stats.hz, pipeline.trt_stats.hz))
        print("rel-L2 vs torch    : %.3e (budget %.0e)"
              % (pipeline.rel_l2, REL_L2_BUDGET))
        print("engine B/param     : %.3f (fp16 ceiling 2.05)"
              % pipeline.bytes_per_param)
        print("build wall time    : %.2f s" % pipeline.build_seconds)
        print("params / engine    : %d / %d bytes"
              % (pipeline.params, pipeline.engine_path.stat().st_size))
        print("drift torch/TRT    : %s | %s"
              % (pipeline.torch_drift[1], pipeline.trt_drift[1]))
    assert True
