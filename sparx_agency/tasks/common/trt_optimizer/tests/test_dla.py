"""Tests for the NVDLA gate: op scanning, runtime detection and the verdict.

The primary path is deliberately ONNX-free. ``.venv`` (the interpreter that
runs this suite) has neither ``onnx`` nor ``tensorrt``, and that is the point:
:func:`scan_ops` accepts any object exposing ``.graph.node[i].op_type`` and
:func:`runtime_supports_dla` accepts an injected ``tensorrt`` module, so the
whole decision is testable on a laptop with no NVIDIA stack at all. The two
tests that do use real ``onnx`` are skipped when it is missing rather than
being the thing the suite is built on.

``_FakeTrt`` is not a mock of convenience: it encodes the exact trap the module
exists to avoid. It exposes ``DeviceType.DLA``, ``BuilderFlag.GPU_FALLBACK``
and ``MemoryPoolType.DLA_MANAGED_SRAM`` at *every* version, including 11.1
where DLA does not work -- so a ``hasattr``-based implementation would pass the
happy-path tests and fail :func:`test_trt11_rejected_despite_dla_enums`.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.hardware.detect import HardwareProfile
from sparx_agency.tasks.common.trt_optimizer.engine.dla import (
    DLA_SUPPORTED_OPS,
    DLA_UNSUPPORTED_OPS,
    DlaVerdict,
    evaluate,
    power_note,
    runtime_supports_dla,
    scan_ops,
)

try:  # the .venv has no onnx; the real-ONNX tests skip themselves
    import onnx  # noqa: F401
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------

class _FakeNode(object):
    """One ONNX node, reduced to the only field the scanner reads."""

    def __init__(self, op_type):
        self.op_type = op_type


class _FakeGraph(object):
    def __init__(self, op_types):
        self.node = [_FakeNode(op) for op in op_types]


class _FakeModel(object):
    """Duck-typed stand-in for ``onnx.ModelProto`` (``.graph.node``)."""

    def __init__(self, op_types):
        self.graph = _FakeGraph(op_types)


class _FakeRuntime(object):
    def __init__(self, cores):
        self.num_DLA_cores = cores


class _FakeSeverity(object):
    ERROR = 1


class _FakeLogger(object):
    ERROR = _FakeSeverity.ERROR

    def __init__(self, severity=None):
        self.severity = severity


class _FakeTrt(object):
    """A ``tensorrt`` module stand-in that keeps the vestigial DLA enums.

    Every DLA-looking attribute exists at every version, exactly as in the real
    TensorRT 11.1 -- so only a version + ``num_DLA_cores`` gate can tell the
    releases apart.
    """

    class DeviceType(object):
        DLA = "DLA"

    class BuilderFlag(object):
        GPU_FALLBACK = "GPU_FALLBACK"

    class MemoryPoolType(object):
        DLA_MANAGED_SRAM = "DLA_MANAGED_SRAM"

    class OnnxParserFlag(object):
        REPORT_CAPABILITY_DLA = "REPORT_CAPABILITY_DLA"

    Logger = _FakeLogger

    def __init__(self, version, cores, runtime_raises=False):
        self.__version__ = version
        self._cores = cores
        self._runtime_raises = runtime_raises

    def Runtime(self, logger):  # noqa: N802 -- mirrors the TensorRT API
        if self._runtime_raises:
            raise RuntimeError("no CUDA driver")
        return _FakeRuntime(self._cores)


TRT_10 = _FakeTrt("10.3.0.26", cores=2)
TRT_10_NO_CORES = _FakeTrt("10.3.0.26", cores=0)
TRT_11 = _FakeTrt("11.1.0.106", cores=2)

CONV_BACKBONE = ["Conv", "BatchNormalization", "Relu", "MaxPool", "Conv",
                 "BatchNormalization", "Relu", "Add", "GlobalAveragePool",
                 "Flatten", "Gemm"]
TRANSFORMER = ["LayerNormalization", "MatMul", "MatMul", "Softmax", "MatMul",
               "Add", "LayerNormalization", "MatMul", "Gelu", "MatMul", "Add"]
CONV_THEN_ATTENTION = CONV_BACKBONE[:4] + TRANSFORMER


def _jetson(model="NVIDIA Jetson AGX Orin Developer Kit", watts=None, cores=2):
    """A HardwareProfile as ``hardware.detect`` would fill it on an Orin."""
    return HardwareProfile(arch="aarch64", is_jetson=True, jetson_model=model,
                           gpu_name=model, power_budget_w=watts,
                           compute_capability=(8, 7), dla_cores=cores,
                           allow_dla=cores > 0, target_tag="orin_sm87")


def _x86():
    return HardwareProfile(arch="x86_64", is_jetson=False,
                           gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
                           compute_capability=(12, 0), dla_cores=0)


# --------------------------------------------------------------------------
# op tables
# --------------------------------------------------------------------------

def test_op_tables_are_disjoint():
    assert not (DLA_SUPPORTED_OPS & DLA_UNSUPPORTED_OPS)


@pytest.mark.parametrize("op", [
    "LayerNormalization", "RMSNormalization", "Softmax", "MatMul", "Einsum",
    "Gather", "TopK", "NonZero", "Erf", "Gelu", "ScatterND", "Range", "Shape",
    "NonMaxSuppression",
])
def test_known_blockers_are_listed_unsupported(op):
    assert op in DLA_UNSUPPORTED_OPS


@pytest.mark.parametrize("op", [
    "Conv", "Relu", "LeakyRelu", "MaxPool", "AveragePool", "Add", "Mul", "Sub",
    "Concat", "Slice", "BatchNormalization", "Sigmoid", "Tanh", "Resize",
    "Pad", "Flatten", "Reshape", "Gemm",
])
def test_cnn_vocabulary_is_listed_supported(op):
    assert op in DLA_SUPPORTED_OPS


# --------------------------------------------------------------------------
# scan_ops
# --------------------------------------------------------------------------

def test_scan_conv_only_graph_is_all_prefix():
    scan = scan_ops(_FakeModel(CONV_BACKBONE))
    assert scan["total"] == len(CONV_BACKBONE)
    assert scan["unsupported"] == {}
    assert scan["unknown"] == {}
    assert scan["supported"]["Conv"] == 2
    assert scan["supported"]["BatchNormalization"] == 2
    assert scan["first_unsupported_index"] is None
    assert scan["contiguous_supported_prefix"] == len(CONV_BACKBONE)


def test_scan_transformer_graph_has_no_prefix():
    scan = scan_ops(_FakeModel(TRANSFORMER))
    assert scan["contiguous_supported_prefix"] == 0
    assert scan["first_unsupported_index"] == 0
    assert scan["unsupported"]["MatMul"] == 5
    assert scan["unsupported"]["LayerNormalization"] == 2
    assert scan["unsupported"]["Softmax"] == 1
    assert scan["unsupported"]["Gelu"] == 1
    assert sum(scan["supported"].values()) == 2  # the two residual Adds


def test_scan_conv_prefix_then_attention_finds_the_boundary():
    scan = scan_ops(_FakeModel(CONV_THEN_ATTENTION))
    assert scan["contiguous_supported_prefix"] == 4
    assert scan["first_unsupported_index"] == 4
    assert scan["total"] == len(CONV_THEN_ATTENTION)


def test_unknown_ops_break_the_prefix_without_being_called_unsupported():
    scan = scan_ops(_FakeModel(["Conv", "Relu", "Bilateral", "Conv"]))
    assert scan["unknown"] == {"Bilateral": 1}
    assert scan["contiguous_supported_prefix"] == 2
    assert scan["first_unsupported_index"] is None
    assert sum(scan["supported"].values()) == 3


def test_scan_rejects_an_object_that_is_not_a_graph():
    with pytest.raises(TypeError):
        scan_ops(object())


@pytest.mark.skipif(HAS_ONNX, reason="onnx is installed on this interpreter")
def test_scan_path_without_onnx_raises_importerror(tmp_path):
    with pytest.raises(ImportError):
        scan_ops(tmp_path / "model.onnx")


@pytest.mark.skipif(not HAS_ONNX, reason="needs the onnx package")
def test_scan_real_onnx_file(tmp_path):
    from onnx import helper, TensorProto

    nodes = [helper.make_node("Relu", ["x"], ["y"]),
             helper.make_node("Softmax", ["y"], ["z"], axis=-1)]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 8])])
    path = tmp_path / "tiny.onnx"
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))
    scan = scan_ops(path)
    assert scan["total"] == 2
    assert scan["contiguous_supported_prefix"] == 1
    assert scan["first_unsupported_index"] == 1


# --------------------------------------------------------------------------
# runtime_supports_dla
# --------------------------------------------------------------------------

def test_trt11_rejected_despite_dla_enums():
    ok, why = runtime_supports_dla(trt_module=TRT_11)
    assert ok is False
    assert "10.7" in why and "11" in why
    # the enums a hasattr() probe would have believed are all present
    assert TRT_11.DeviceType.DLA and TRT_11.BuilderFlag.GPU_FALLBACK
    assert TRT_11.MemoryPoolType.DLA_MANAGED_SRAM
    assert TRT_11.OnnxParserFlag.REPORT_CAPABILITY_DLA


def test_trt10_with_zero_cores_rejected():
    ok, why = runtime_supports_dla(trt_module=TRT_10_NO_CORES)
    assert ok is False
    assert "num_DLA_cores=0" in why


def test_trt10_with_two_cores_accepted():
    ok, why = runtime_supports_dla(trt_module=TRT_10)
    assert ok is True
    assert "num_DLA_cores=2" in why


def test_runtime_probe_failure_is_a_refusal_not_a_crash():
    ok, why = runtime_supports_dla(
        trt_module=_FakeTrt("10.3.0.26", cores=2, runtime_raises=True))
    assert ok is False
    assert "num_DLA_cores" in why


def test_unparseable_version_is_refused():
    ok, why = runtime_supports_dla(trt_module=_FakeTrt("unknown", cores=2))
    assert ok is False
    assert "version" in why


def test_board_without_dla_short_circuits_before_importing_tensorrt():
    ok, why = runtime_supports_dla(trt_module=None, hardware=_x86())
    assert ok is False
    assert "dla_cores=0" in why


def test_missing_tensorrt_is_a_refusal_with_a_reason():
    try:
        import tensorrt  # noqa: F401
        pytest.skip("tensorrt is installed on this interpreter")
    except ImportError:
        pass
    ok, why = runtime_supports_dla()
    assert ok is False
    assert "tensorrt" in why


# --------------------------------------------------------------------------
# evaluate -- one test per rule
# --------------------------------------------------------------------------

def test_evaluate_refuses_when_the_runtime_has_no_dla():
    verdict = evaluate(_FakeModel(CONV_BACKBONE), _jetson(), trt_module=TRT_11)
    assert isinstance(verdict, DlaVerdict)
    assert verdict.use_dla is False
    assert "10.7" in verdict.why


def test_evaluate_refuses_when_the_board_has_no_dla_cores():
    verdict = evaluate(_FakeModel(CONV_BACKBONE), _jetson(cores=0),
                       trt_module=TRT_10)
    assert verdict.use_dla is False
    assert "dla_cores=0" in verdict.why


def test_evaluate_refuses_a_graph_below_the_eligible_fraction():
    verdict = evaluate(_FakeModel(TRANSFORMER), _jetson(), trt_module=TRT_10)
    assert verdict.use_dla is False
    assert verdict.eligible_fraction == pytest.approx(2 / 11.0)
    assert "MatMul x5" in verdict.why
    assert "MatMul x5" in verdict.unsupported_sample


def test_evaluate_refuses_a_split_eligible_region():
    # 8 of 11 nodes eligible (> 60%), but the eligible ones straddle a Softmax:
    # two DLA regions, therefore two reformats.
    ops = (["Conv", "Relu", "BatchNormalization", "Conv"] + ["Softmax"]
           + ["Conv", "Relu", "Add", "Conv", "Relu", "Gemm"])
    verdict = evaluate(_FakeModel(ops), _jetson(), trt_module=TRT_10)
    assert verdict.use_dla is False
    assert verdict.contiguous_prefix == 4
    assert verdict.eligible_fraction > 0.6
    assert "contiguous" in verdict.why and "reformat" in verdict.why


def test_evaluate_accepts_a_pure_cnn_backbone():
    verdict = evaluate(_FakeModel(CONV_BACKBONE), _jetson(), trt_module=TRT_10)
    assert verdict.use_dla is True
    assert verdict.eligible_fraction == 1.0
    assert verdict.contiguous_prefix == len(CONV_BACKBONE)
    assert "INT8" in verdict.why
    assert "ONE handoff" in verdict.why
    assert "backbone" in verdict.why and "transformer head" in verdict.why


def test_evaluate_flags_trailing_unknown_ops_on_an_accepted_graph():
    verdict = evaluate(_FakeModel(CONV_BACKBONE + ["Bilateral"]), _jetson(),
                       trt_module=TRT_10)
    assert verdict.use_dla is True
    assert "unknown DLA status" in verdict.why


def test_evaluate_honours_a_stricter_threshold():
    ops = ["Conv", "Relu", "Conv", "Relu", "MatMul"]  # 80% eligible
    assert evaluate(_FakeModel(ops), _jetson(), trt_module=TRT_10).use_dla
    strict = evaluate(_FakeModel(ops), _jetson(), trt_module=TRT_10,
                      min_eligible_fraction=0.9)
    assert strict.use_dla is False


def test_evaluate_raises_on_an_empty_graph():
    with pytest.raises(ValueError):
        evaluate(_FakeModel([]), _jetson(), trt_module=TRT_10)


def test_evaluate_raises_on_a_nonsense_threshold():
    with pytest.raises(ValueError):
        evaluate(_FakeModel(CONV_BACKBONE), _jetson(), trt_module=TRT_10,
                 min_eligible_fraction=1.5)


# --------------------------------------------------------------------------
# power_note
# --------------------------------------------------------------------------

def test_power_note_warns_about_one_core_on_an_orin_nx_at_15w():
    note = power_note(_jetson("NVIDIA Jetson Orin NX 16GB", watts=15))
    assert "ONE DLA core" in note
    assert "25 W" in note
    assert "15 W" in note


def test_power_note_quotes_the_compute_ratio_and_the_build_rule():
    note = power_note(_jetson(watts=15))
    assert "11.8%" in note and "38.4%" in note and "3.3x" in note
    assert "jetson_clocks" in note and "timing cache" in note
    assert "ONE DLA core" not in note  # AGX Orin keeps both cores at 15 W


def test_power_note_is_empty_off_jetson():
    assert power_note(_x86()) == ""
