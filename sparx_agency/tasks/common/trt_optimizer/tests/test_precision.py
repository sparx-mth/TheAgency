"""Tests for the precision baking/verification contract.

Split in two on purpose. The bulk is pure logic and runs in the repo ``.venv``,
which has neither tensorrt nor onnx: TensorRT is stood in for by tiny fake
modules, and the engine by an object with a scripted ``get_engine_stat``. That
fake is the only way to test the silent-widening case at all -- reproducing it
for real would mean owning a GPU without an INT4 kernel.

The ONNX classification tests need a real ``onnx`` and are skipped in ``.venv``;
run them with the navdp conda interpreter.
"""
from __future__ import annotations

import types

import pytest

from sparx_agency.tasks.common.trt_optimizer.engine import precision as P


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class _WeakFlags(object):
    """TensorRT <= 10 BuilderFlag: has the precision flags."""

    FP16 = 0
    INT8 = 1
    TF32 = 2


class _StrongFlags(object):
    """TensorRT 11 BuilderFlag: the measured member list, precision gone."""

    DEBUG = 0
    GPU_FALLBACK = 1
    REFIT = 2
    SPARSE_WEIGHTS = 3
    STRICT_NANS = 4
    TF32 = 5
    VERSION_COMPATIBLE = 6
    WEIGHT_STREAMING = 7


class _EngineStat(object):
    TOTAL_WEIGHTS_SIZE = 0
    STRIPPED_WEIGHTS_SIZE = 1


def _fake_trt(with_stat=True):
    """A stand-in tensorrt module for a strongly-typed build."""
    ns = types.SimpleNamespace(BuilderFlag=_StrongFlags)
    if with_stat:
        ns.EngineStat = _EngineStat
    return ns


class _FakeEngine(object):
    """Engine whose weight-bytes answer is scripted by the test."""

    def __init__(self, total, raises=False):
        self._total = total
        self._raises = raises
        self.asked = []

    def get_engine_stat(self, stat):
        self.asked.append(stat)
        if self._raises:
            raise RuntimeError("stat unsupported")
        return self._total


class _StatlessEngine(object):
    """An older engine with no get_engine_stat at all."""


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

def test_precisions_and_ceilings_cover_the_same_names():
    assert P.PRECISIONS == ("fp32", "fp16", "bf16", "int8", "fp8", "int4",
                            "nvfp4")
    assert set(P.BYTES_PER_ELEM_CEILING) == set(P.PRECISIONS)


def test_ceilings_are_the_measured_values():
    c = P.BYTES_PER_ELEM_CEILING
    assert (c["fp32"], c["fp16"], c["bf16"]) == (4.05, 2.05, 2.05)
    assert (c["int8"], c["fp8"]) == (1.05, 1.05)
    assert (c["int4"], c["nvfp4"]) == (0.60, 0.80)


def test_ceilings_leave_headroom_but_not_a_whole_format():
    """Each ceiling sits above its element width and below the next one up."""
    for name, width in (("fp16", 2.0), ("int8", 1.0), ("int4", 0.5),
                        ("nvfp4", 0.5)):
        assert P.BYTES_PER_ELEM_CEILING[name] > width
    assert P.BYTES_PER_ELEM_CEILING["int4"] < 1.0
    assert P.BYTES_PER_ELEM_CEILING["nvfp4"] < 1.0
    assert P.BYTES_PER_ELEM_CEILING["int8"] < 2.0


# --------------------------------------------------------------------------
# is_strongly_typed
# --------------------------------------------------------------------------

def test_is_strongly_typed_false_when_fp16_flag_exists():
    assert P.is_strongly_typed(types.SimpleNamespace(BuilderFlag=_WeakFlags)) \
        is False


def test_is_strongly_typed_true_when_fp16_flag_is_gone():
    assert P.is_strongly_typed(_fake_trt()) is True


def test_is_strongly_typed_raises_without_builder_flag():
    with pytest.raises(TypeError):
        P.is_strongly_typed(types.SimpleNamespace())


# --------------------------------------------------------------------------
# bake_precision
# --------------------------------------------------------------------------

def test_bake_precision_fp32_returns_the_source_untouched(tmp_path):
    src = tmp_path / "graph.onnx"
    src.write_bytes(b"not really onnx")
    out = tmp_path / "never_written.onnx"
    assert P.bake_precision(src, out, "fp32") == src
    assert not out.exists()


def test_bake_precision_rejects_an_unknown_precision(tmp_path):
    with pytest.raises(ValueError):
        P.bake_precision(tmp_path / "g.onnx", tmp_path / "o.onnx", "fp24")


def test_bake_precision_raises_for_a_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.bake_precision(tmp_path / "absent.onnx", tmp_path / "o.onnx", "fp32")


@pytest.mark.parametrize("prec", ["int8", "fp8", "int4", "nvfp4"])
def test_bake_precision_quantized_names_modelopt_quantization(tmp_path, prec):
    with pytest.raises(NotImplementedError) as exc:
        P.bake_precision(tmp_path / "g.onnx", tmp_path / "o.onnx", prec)
    msg = str(exc.value)
    assert "modelopt.onnx.quantization" in msg
    assert "nvidia-modelopt is NOT" in msg


def test_bake_precision_bf16_names_modelopt_autocast(tmp_path):
    with pytest.raises(NotImplementedError) as exc:
        P.bake_precision(tmp_path / "g.onnx", tmp_path / "o.onnx", "bf16")
    msg = str(exc.value)
    assert "modelopt.onnx.autocast" in msg
    assert "nvidia-modelopt is NOT" in msg


def test_bake_precision_never_offers_a_silent_fp16_fallback(tmp_path):
    """The refusal must say it is refusing, not hint at a workaround."""
    with pytest.raises(NotImplementedError) as exc:
        P.bake_precision(tmp_path / "g.onnx", tmp_path / "o.onnx", "int4")
    assert "Refusing to fall back" in str(exc.value)


# --------------------------------------------------------------------------
# verify_engine_precision
# --------------------------------------------------------------------------

def test_verify_uses_the_measured_fp16_anchor():
    """11,776 bytes over the same model's 5,760 params is honest FP16."""
    engine = _FakeEngine(11_776_000)
    ok, bpe, msg = P.verify_engine_precision(engine, "fp16", 5_760_000, _fake_trt())
    assert ok is True
    assert bpe == pytest.approx(11_776_000 / 5_760_000.0)
    assert engine.asked == [_EngineStat.TOTAL_WEIGHTS_SIZE]
    assert "verified" in msg


def test_verify_catches_an_fp32_engine_claiming_fp16():
    """The measured FP32 build of the same model: 23,040 bytes."""
    ok, bpe, msg = P.verify_engine_precision(_FakeEngine(23_040_000), "fp16", 5_760_000,
                                             _fake_trt())
    assert ok is False
    assert bpe == pytest.approx(4.0)
    assert "NOT honoured" in msg and "fp16" in msg


def test_verify_catches_int4_silently_widened_to_fp16():
    """The case nothing else in the pipeline can see: the build succeeded."""
    ok, bpe, msg = P.verify_engine_precision(_FakeEngine(2 * 5_760_000), "int4",
                                             5_760_000, _fake_trt())
    assert ok is False
    assert bpe == pytest.approx(2.0)
    assert "NOT honoured" in msg


def test_verify_flags_a_near_miss_as_probably_pinned_layers():
    """13,440 B over a 6,448-param FP16 toy: measured, and 2.084 B/elem."""
    ok, bpe, msg = P.verify_engine_precision(_FakeEngine(13_440_000), "fp16", 6_448_000,
                                             _fake_trt())
    assert ok is False
    assert bpe == pytest.approx(2.084, abs=1e-3)
    assert "only just over" in msg


def test_verify_does_not_excuse_a_real_widening():
    ok, _, msg = P.verify_engine_precision(_FakeEngine(4 * 6_448_000), "fp16", 6_448_000,
                                           _fake_trt())
    assert ok is False
    assert "only just over" not in msg


def test_verify_accepts_a_real_nvfp4_engine():
    ok, bpe, _ = P.verify_engine_precision(_FakeEngine(int(0.5625 * 4_096_000)),
                                           "nvfp4", 4_096_000, _fake_trt())
    assert ok is True
    assert bpe < P.BYTES_PER_ELEM_CEILING["nvfp4"]


def test_verify_accepts_an_fp32_engine_at_the_ceiling():
    ok, _, _ = P.verify_engine_precision(_FakeEngine(4 * 1_000_000), "fp32", 1_000_000,
                                         _fake_trt())
    assert ok is True


def test_verify_skips_when_the_build_has_no_engine_stat():
    ok, bpe, msg = P.verify_engine_precision(_FakeEngine(99), "fp16", 10,
                                             _fake_trt(with_stat=False))
    assert ok is True
    assert bpe is None
    assert "SKIPPED" in msg and "NOT verified" in msg


def test_verify_skips_when_the_engine_has_no_getter():
    ok, bpe, msg = P.verify_engine_precision(_StatlessEngine(), "fp16", 10,
                                             _fake_trt())
    assert (ok, bpe) == (True, None)
    assert "SKIPPED" in msg


def test_verify_skips_when_the_getter_raises():
    ok, bpe, msg = P.verify_engine_precision(_FakeEngine(0, raises=True),
                                             "fp16", 10, _fake_trt())
    assert (ok, bpe) == (True, None)
    assert "SKIPPED" in msg


def test_verify_skips_on_the_minus_one_sentinel():
    """get_engine_stat returns -1 for a stat it cannot answer."""
    ok, bpe, msg = P.verify_engine_precision(_FakeEngine(-1), "fp16", 10,
                                             _fake_trt())
    assert (ok, bpe) == (True, None)
    assert "SKIPPED" in msg


def test_verify_skipped_message_never_claims_a_pass():
    _, _, msg = P.verify_engine_precision(_StatlessEngine(), "int4", 10,
                                          _fake_trt())
    assert "verified:" not in msg


def test_verify_lazily_imports_tensorrt_when_none_is_given():
    """Works on a machine without tensorrt (skips) and with it (passes)."""
    ok, _, _ = P.verify_engine_precision(_FakeEngine(2 * 512_000), "fp16", 512_000)
    assert ok is True


def test_verify_rejects_an_unknown_precision():
    with pytest.raises(ValueError):
        P.verify_engine_precision(_FakeEngine(10), "fp24", 10, _fake_trt())


@pytest.mark.parametrize("count", [0, -5])
def test_verify_rejects_a_non_positive_param_count(count):
    with pytest.raises(ValueError):
        P.verify_engine_precision(_FakeEngine(10), "fp16", count, _fake_trt())


# --------------------------------------------------------------------------
# deny list
# --------------------------------------------------------------------------

EXPECTED_DENY = [
    "*vision_tower*", "*visual*", "*vision_model*", "*embed_vision*",
    "*multi_modal_projector*", "*lm_head*", "*output_layer*", "output.*",
    "*router*", "*mlp.gate*", "*block_sparse_moe.gate*", "mtp.*",
    "*proj_out*", "nn.Embedding", "nn.BatchNorm*", "nn.LeakyReLU",
]


def test_deny_list_is_the_modelopt_default_set_in_order():
    assert P.quantization_deny_list() == EXPECTED_DENY


def test_deny_list_is_a_fresh_list_each_call():
    first = P.quantization_deny_list()
    first.append("*mine*")
    assert "*mine*" not in P.quantization_deny_list()


def test_every_deny_entry_has_a_reason():
    for pattern in P.quantization_deny_list():
        reason = P.reason_for(pattern)
        assert isinstance(reason, str) and len(reason) > 30


def test_reason_for_an_unknown_pattern_raises():
    with pytest.raises(KeyError):
        P.reason_for("*conv1*")


def test_vision_reason_names_the_garbage_embedding_failure():
    assert "garbage" in P.reason_for("*vision_tower*")


def test_router_reason_explains_the_discontinuity():
    assert "top-k" in P.reason_for("*router*")


def test_linear_alignment_rule_is_documented_and_exposed():
    assert P.LINEAR_DIM_ALIGNMENT == 16
    doc = P.quantization_deny_list.__doc__
    assert "in_features" in doc and "out_features" in doc


# --------------------------------------------------------------------------
# report note
# --------------------------------------------------------------------------

def test_sensitive_op_note_names_the_three_ops_and_the_opset_floor():
    note = P.sensitive_op_note()
    for op in ("LayerNormalization", "Softmax", "ReduceMean"):
        assert op in note
    assert "17" in note
    assert "ReduceMean/Sub/Pow/Div" in note


def test_sensitive_ops_match_the_shared_fp16_converter():
    """The note must describe the op block list bake_precision actually uses."""
    fp16_onnx = pytest.importorskip(
        "sparx_agency.tasks.planning.vlas.common.engine.fp16_onnx")
    note = P.sensitive_op_note()
    for op in fp16_onnx.SENSITIVE_OPS:
        assert op in note


# --------------------------------------------------------------------------
# onnx classification (needs a real onnx; skipped in .venv)
# --------------------------------------------------------------------------

def _graph_with(initializers, nodes=None, name="g"):
    """Build a minimal ModelProto around some initializers."""
    import onnx
    from onnx import helper

    inp = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])
    out = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])
    if nodes is None:
        nodes = [helper.make_node("Identity", ["x"], ["y"])]
    graph = helper.make_graph(nodes, name, [inp], [out],
                              initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _init(name, elem_type, values):
    from onnx import helper
    return helper.make_tensor(name, elem_type, [len(values)], values)


def test_onnx_precision_fp32():
    onnx = pytest.importorskip("onnx")
    model = _graph_with([_init("w", onnx.TensorProto.FLOAT, [1.0, 2.0])])
    assert P.onnx_precision(model) == "fp32"


def test_onnx_precision_fp16():
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import numpy_helper

    w = numpy_helper.from_array(np.zeros(4, dtype=np.float16), "w")
    assert P.onnx_precision(_graph_with([w])) == "fp16"
    assert w.data_type == onnx.TensorProto.FLOAT16


def test_onnx_precision_mixed():
    pytest.importorskip("onnx")
    import numpy as np
    from onnx import numpy_helper

    inits = [numpy_helper.from_array(np.zeros(4, dtype=np.float16), "w16"),
             numpy_helper.from_array(np.zeros(4, dtype=np.float32), "w32")]
    assert P.onnx_precision(_graph_with(inits)) == "mixed"


def test_onnx_precision_ignores_integer_shape_constants():
    onnx = pytest.importorskip("onnx")
    inits = [_init("w", onnx.TensorProto.FLOAT, [1.0]),
             _init("shape", onnx.TensorProto.INT64, [1, 2])]
    assert P.onnx_precision(_graph_with(inits)) == "fp32"


def test_onnx_precision_qdq_beats_the_initializer_dtypes():
    onnx = pytest.importorskip("onnx")
    from onnx import helper

    nodes = [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q"]),
             helper.make_node("DequantizeLinear", ["q", "s", "z"], ["y"])]
    inits = [_init("s", onnx.TensorProto.FLOAT, [0.1]),
             _init("z", onnx.TensorProto.INT8, [0])]
    assert P.onnx_precision(_graph_with(inits, nodes)) == "qdq"


def test_onnx_precision_raises_without_a_float_initializer():
    onnx = pytest.importorskip("onnx")
    model = _graph_with([_init("shape", onnx.TensorProto.INT64, [1, 2])])
    with pytest.raises(ValueError):
        P.onnx_precision(model)


def test_onnx_precision_accepts_a_path(tmp_path):
    onnx = pytest.importorskip("onnx")
    model = _graph_with([_init("w", onnx.TensorProto.FLOAT, [1.0])])
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    assert P.onnx_precision(path) == "fp32"
    assert P.onnx_precision(str(path)) == "fp32"


def test_bake_precision_fp16_delegates_to_the_shared_converter(tmp_path):
    """End to end: the baked graph loads and is no longer pure FP32."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxconverter_common")
    import numpy as np
    from onnx import helper, numpy_helper

    w = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), "w")
    inp = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])
    out = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])],
                              "m", [inp], [out], initializer=[w])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 17)])
    src = tmp_path / "fp32.onnx"
    onnx.save(model, str(src))

    dst = P.bake_precision(src, tmp_path / "fp16.onnx", "fp16")
    assert dst.is_file()
    assert P.onnx_precision(src) == "fp32"
    assert P.onnx_precision(dst) in ("fp16", "mixed")


def test_verify_skips_below_the_alignment_floor():
    """Measured: a 650-param FP16 engine reports 3.15 B/elem because TensorRT's
    weight alignment padding dominates. Below the floor the ratio says nothing,
    so the check reports SKIPPED rather than failing a genuine FP16 engine."""
    ok, bpe, msg = P.verify_engine_precision(_FakeEngine(2048), "fp16", 650,
                                             _fake_trt())
    assert ok is True
    assert bpe is None
    assert "SKIPPED" in msg
    assert "alignment padding" in msg
    assert "NOT verified" in msg


def test_verify_still_runs_at_the_floor():
    ok, bpe, _ = P.verify_engine_precision(
        _FakeEngine(2 * P.MIN_PARAMS_FOR_PRECISION_CHECK), "fp16",
        P.MIN_PARAMS_FOR_PRECISION_CHECK, _fake_trt())
    assert ok is True
    assert bpe == pytest.approx(2.0)
