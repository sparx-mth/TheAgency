"""Convert an FP32 ONNX graph to mixed FP16 for strongly-typed TensorRT builds.

TensorRT 11 removed weak typing (no ``BuilderFlag.FP16``): precision now comes
from the network's types via a STRONGLY_TYPED network. To get FP16 acceleration
there, the ONNX itself must carry FP16 types. This converts an FP32 graph to
FP16 while:

  * keeping graph **inputs/outputs FP32** (``keep_io_types=True``) so the numpy
    runtime keeps feeding/reading FP32 (the engine casts at the boundary), and
  * keeping the numerically sensitive ops in FP32 (LayerNormalization, Softmax,
    ReduceMean) -- the same "pin sensitive layers high" intent as the TRT-10
    per-layer precision pins, applied at the ONNX level.

Only used on TensorRT >= 11; the TRT-10 path on the Orin keeps the classic
weakly-typed ``BuilderFlag.FP16`` build from an unmodified FP32 ONNX.
"""
from __future__ import annotations

from pathlib import Path

# Sensitive ops to keep in FP32 (added to onnxconverter_common's defaults).
SENSITIVE_OPS = ("LayerNormalization", "Softmax", "ReduceMean")


def to_fp16_onnx(src_onnx, dst_onnx, keep_ops=SENSITIVE_OPS):
    """Write an FP16 copy of ``src_onnx`` (FP32 IO, sensitive ops kept FP32).

    Args:
        src_onnx: path to the FP32 ONNX graph.
        dst_onnx: output path for the FP16 graph.
        keep_ops: op types to keep in FP32 (added to the converter defaults).

    Returns:
        ``dst_onnx`` as a :class:`pathlib.Path`.
    """
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src_onnx))
    block = list(getattr(float16, "DEFAULT_OP_BLOCK_LIST", []))
    for op in keep_ops:
        if op not in block:
            block.append(op)
    model16 = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=False, op_block_list=block)
    dst = Path(dst_onnx)
    onnx.save(model16, str(dst))
    return dst
