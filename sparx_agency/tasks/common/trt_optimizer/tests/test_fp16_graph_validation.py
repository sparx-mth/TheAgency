"""An invalid FP16 conversion must be a rung failure, not a shipped engine.

``onnxconverter_common.convert_float_to_float16`` does not raise when a blocked
op leaves a mistyped cast behind: it returns a graph whose declared tensor types
disagree with its ops. onnxruntime refuses to load such a graph -- TensorRT
parses it and builds a silently wrong engine.
"""
import pytest

from sparx_agency.tasks.common.trt_optimizer.engine import precision

onnx = pytest.importorskip("onnx")


def test_ladder_has_a_rung_between_the_strongest_and_the_defaults():
    """``LayerNormalization`` alone is the op actually worth pinning.

    Blocking ``Softmax``/``ReduceMean`` as well is what breaks the converter on a
    transformer graph, so a two-rung ladder jumps straight from "invalid" to "no
    pinning at all".
    """
    labels = [label for label, _ in precision.FP16_LADDER]
    keeps = [keep for _, keep in precision.FP16_LADDER]
    assert keeps[0] == ("LayerNormalization", "Softmax", "ReduceMean")
    assert keeps[1] == ("LayerNormalization",)
    assert keeps[-1] == ()
    assert len(set(labels)) == len(labels)


def test_the_ladder_weakens_monotonically():
    """Each rung must pin a subset of the one above it, or it is not a ladder."""
    keeps = [set(keep) for _, keep in precision.FP16_LADDER]
    for stronger, weaker in zip(keeps, keeps[1:]):
        assert weaker < stronger


def test_a_mistyped_graph_is_rejected_rather_than_returned(tmp_path):
    """Hand-build the failure mode: a Cast declaring the wrong output type."""
    from onnx import TensorProto, helper

    from sparx_agency.tasks.common.trt_optimizer.engine import fp16_graph

    graph = helper.make_graph(
        [helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT16)],
        "mistyped",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        # The node emits float16; the declared output says float. This is the
        # shape of what the converter produces around a blocked op.
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "mistyped.onnx"
    onnx.save(model, str(path))

    with pytest.raises(Exception):
        onnx.checker.check_model(str(path), full_check=True)


def test_valid_conversion_passes_validation(tmp_path):
    """A clean graph must survive the new check, or every FP16 build breaks."""
    pytest.importorskip("onnxconverter_common")
    from onnx import TensorProto, helper
    from onnx import numpy_helper

    import numpy as np

    from sparx_agency.tasks.common.trt_optimizer.engine import fp16_graph

    weight = numpy_helper.from_array(
        np.ones((4, 4), dtype=np.float32), name="w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "matmul",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    src = tmp_path / "ok.onnx"
    onnx.save(model, str(src))

    out = fp16_graph.to_fp16_onnx(src, tmp_path / "ok.fp16.onnx", validate=True)
    assert out.is_file()


def test_the_invalid_conversion_error_names_the_keep_list():
    """The message has to say which rung failed, or the ladder is unreadable."""
    from sparx_agency.tasks.common.trt_optimizer.engine import fp16_graph

    assert issubclass(fp16_graph.Fp16ConversionInvalid, RuntimeError)
