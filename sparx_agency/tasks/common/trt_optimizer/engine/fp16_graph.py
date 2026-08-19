"""Convert an FP32 ONNX graph to mixed FP16, for a strongly-typed TensorRT build.

TensorRT 11 removed weak typing: there is no ``BuilderFlag.FP16``, every network
is strongly typed, and the engine's precision is exactly what the ONNX carries.
So on that generation FP16 is produced *here*, at the graph level, or not at all.

Two choices are deliberate:

``keep_io_types=True``
    Graph inputs and outputs stay FP32 and the engine casts at its boundary, so
    a numpy runtime keeps feeding and reading FP32 and does not have to know
    which precision the engine was built at.

the keep-list
    Numerically sensitive ops stay FP32. This is the graph-level equivalent of
    the per-layer precision pins that TensorRT 10 offered and 11 removed.

The keep-list is a trade, not a free win, and the trade is documented in
:data:`SENSITIVE_OPS`: keeping those ops FP32 can leave a boundary the TensorRT
parser rejects. :mod:`..engine.precision` owns the ladder that walks down from
the strongest keep-list to one that parses; this module just does what it is
told.

Lives here rather than under any one model's package because nothing about it is
model-specific -- it is a property of ONNX, ``onnxconverter_common`` and the
TensorRT generation.
"""
from __future__ import annotations

from pathlib import Path

#: Ops kept in FP32 on top of the converter's own defaults.
#:
#: Sound numerically -- these are where FP16's 5-bit exponent actually bites --
#: but keeping them out of the conversion can leave a *mixed* boundary the
#: TensorRT parser refuses. Measured on a 4-layer ``nn.TransformerEncoder``: the
#: attention in-projection ``Gemm`` comes out Half while the bias TensorRT
#: broadcasts against it stays Float, and the parse dies with
#: "ElementWiseOperation SUM must have same input types. But they are of types
#: Half and Float". Callers that cannot afford to fail should walk
#: :data:`..engine.precision.FP16_LADDER` rather than passing this blindly.
SENSITIVE_OPS = ("LayerNormalization", "Softmax", "ReduceMean")


class Fp16ConversionInvalid(RuntimeError):
    """The converted FP16 graph is not a valid ONNX model.

    Raised instead of returning it, because this failure is otherwise silent all
    the way to production. ``convert_float_to_float16`` does not raise when a
    blocked op leaves a mistyped cast behind: it returns a graph whose declared
    tensor types disagree with its ops. ``onnxruntime`` refuses to load such a
    graph -- but **TensorRT's parser accepts it and builds an engine**, and that
    engine is quietly, badly wrong.

    Measured on InternVLA-N1's System-1 condition graph: with ``Softmax`` in the
    keep-list the converter emits ``Softmax_output_cast0`` declaring a float16
    output for a float input; TensorRT built it without complaint and the engine
    came out at **1.0e-1** relative L2 against FP32, against 3.0e-4 for the same
    graph converted one rung down. Nothing in the build log said so.
    """


def to_fp16_onnx(src_onnx, dst_onnx, keep_ops=SENSITIVE_OPS, validate=True):
    """Write an FP16 copy of ``src_onnx`` with FP32 IO and ``keep_ops`` in FP32.

    Args:
        src_onnx: path to the FP32 ONNX graph.
        dst_onnx: output path for the converted graph.
        keep_ops: op types to keep in FP32, added to the converter's defaults.
            Pass ``()`` for the converter defaults alone -- the next rung down
            when the stronger list produces a graph TensorRT will not parse.
        validate: run ``onnx.checker.check_model(full_check=True)`` on the
            result and refuse to return an invalid graph. Leave it on. The check
            costs a shape-inference pass and is the only thing standing between
            a mistyped cast and a silently wrong engine -- ``full_check`` is
            what runs type inference, so the default ``False`` would pass every
            graph described in :class:`Fp16ConversionInvalid`.

    Returns:
        pathlib.Path: ``dst_onnx``.

    Raises:
        ImportError: if ``onnx`` or ``onnxconverter_common`` is missing, which
            on a strongly-typed TensorRT means FP16 is unreachable and a build
            would silently produce FP32.
        Fp16ConversionInvalid: when ``validate`` is set and the converted graph
            does not type-check. Callers walking
            :data:`..engine.precision.FP16_LADDER` should treat this exactly as
            they treat a converter exception: try the next rung.
    """
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src_onnx))
    block = list(getattr(float16, "DEFAULT_OP_BLOCK_LIST", []))
    for op in keep_ops or ():
        if op not in block:
            block.append(op)
    converted = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=False, op_block_list=block)
    dst = Path(dst_onnx)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(dst))
    if validate:
        try:
            # Checked from the path, not the in-memory proto: that is the form
            # that copes with a model over the 2 GB protobuf limit and its
            # external data, and it is the artifact the builder will read.
            onnx.checker.check_model(str(dst), full_check=True)
        except Exception as exc:  # noqa: BLE001  (checker raises several types)
            raise Fp16ConversionInvalid(
                "FP16 conversion of %s with keep_ops=%r produced an invalid "
                "graph: %s. TensorRT would parse this and build a wrong engine "
                "without warning; try a weaker keep-list."
                % (Path(src_onnx).name, tuple(keep_ops or ()),
                   str(exc).splitlines()[0]))
    return dst
