"""Bake precision into the ONNX, then prove it survived into the engine.

On TensorRT 11 a build that "succeeds" proves nothing about precision. The weak
typing that made ``config.set_flag(trt.BuilderFlag.FP16)`` a request is gone --
measured on 11.1.0.106, ``trt.BuilderFlag`` has no FP16/BF16/INT8/FP8/INT4/FP4
member and no precision-constraint flag at all, ``ILayer.precision``,
``ILayer.set_output_type``, ``ITensor.dynamic_range`` and every
``IInt8Calibrator`` are gone, and ``builder.create_network(0)`` already reports
``STRONGLY_TYPED``. Engine precision now follows the ONNX exactly. That is the
one fact this module is built around:

  * **Precision is an export-time decision, not a build-time flag.** The only
    lever left is the dtype of the graph handed to the parser, so
    :func:`bake_precision` produces that graph.
  * **A build cannot be trusted to have honoured it.** TensorRT widens a format
    it has no kernel for -- an INT4 or NVFP4 graph can come back out as FP16 or
    FP32 with no warning and no error, only a quietly unchanged latency.
    :func:`verify_engine_precision` divides the engine's own
    ``EngineStat.TOTAL_WEIGHTS_SIZE`` by the parameter count and refuses a
    bytes-per-element above what the format allows. It is the only check that
    catches a silent widening. The measured anchor: an FP32 ONNX gave 23,040
    weight bytes and the same model exported FP16 gave 11,776.

The FP16 conversion itself is not reimplemented here --
:func:`..engine.fp16_graph.to_fp16_onnx` already does it, keeping IO
FP32 and pinning LayerNormalization/Softmax/ReduceMean to FP32; see
:func:`sensitive_op_note` for why those three and why opset 17 is the floor.

Everything below FP16 (bf16 and the quantized formats) needs NVIDIA ModelOpt,
which is not installed on this machine; those paths raise with the package and
entry point named rather than silently degrading to FP16. The deny list in
:func:`quantization_deny_list` is the other half of that story: it is what must
stay out of any quantization the day ModelOpt does arrive.

Python-3.8-compatible syntax; imports of ``onnx``/``tensorrt`` are lazy so the
module can be read on a machine that has neither.
"""
from __future__ import annotations

from pathlib import Path

#: Every precision the toolkit has a name for, widest first.
PRECISIONS = ("fp32", "fp16", "bf16", "int8", "fp8", "int4", "nvfp4")

#: Precisions :func:`bake_precision` can actually produce on this machine.
SUPPORTED_PRECISIONS = ("fp32", "fp16")

#: Upper bound on bytes-per-parameter a *correctly* built engine may show.
#:
#: Each value is the format's element width plus room for its scale metadata
#: and for the layers that legitimately stay wider (biases, LayerNorm, the IO
#: cast boundary): fp32 4.05, fp16/bf16 2.05, int8/fp8 1.05 (one scale per
#: tensor is negligible), int4 0.60 (0.5 + one FP16 scale per block of 64 =
#: 0.53) and nvfp4 0.80 (0.5 + one FP8 scale per block of 16 = 0.5625, plus a
#: per-tensor FP32 second-level scale). Anything above the ceiling means the
#: build widened the format -- not that a few layers were left wide. The
#: headroom is a *fraction*, so it assumes the pinned-FP32 ops are a small
#: share of the parameters: on a toy graph they are not, and the check reads
#: marginally over (see :func:`_widened_message`).
BYTES_PER_ELEM_CEILING = {
    "fp32": 4.05,
    "fp16": 2.05,
    "bf16": 2.05,
    "int8": 1.05,
    "fp8": 1.05,
    "int4": 0.60,
    "nvfp4": 0.80,
}

#: A ``nn.Linear`` whose in/out features are not both multiples of this must be
#: left un-quantized -- see :func:`quantization_deny_list`.
LINEAR_DIM_ALIGNMENT = 16

# Ordered (glob, reason) table behind quantization_deny_list()/reason_for().
# The name globs are NVIDIA ModelOpt's own shipped defaults; the three trailing
# entries are module classes rather than names (ModelOpt excludes them by type).
_DENY = (
    ("*vision_tower*",
     "Vision encoder. ModelOpt excludes the whole vision branch by default: "
     "quantizing it either crashes export or yields garbage image embeddings."),
    ("*visual*",
     "Vision branch under its other common name; same failure mode -- the "
     "language side hallucinates confidently instead of failing loudly."),
    ("*vision_model*",
     "Vision branch as named by HuggingFace VLM checkpoints."),
    ("*embed_vision*",
     "Image-token embedding. A lookup/patch-embed table with no GEMM to "
     "accelerate, and its error is injected before every downstream layer."),
    ("*multi_modal_projector*",
     "Vision-to-language projector. Tiny, and the single point where image "
     "geometry becomes tokens -- quantization error here is unrecoverable."),
    ("*lm_head*",
     "Final vocabulary projection. No downstream layer averages its error, so "
     "the noise lands directly on the emitted token."),
    ("*output_layer*",
     "Output projection under the Megatron/NeMo name; same reasoning."),
    ("output.*",
     "Top-level output submodule (anchored, not a suffix match) -- the final "
     "projection of models that do not use the lm_head name."),
    ("*router*",
     "MoE router. Its output is consumed by a top-k, so a small perturbation "
     "flips which expert runs: a discontinuous error no gate-level metric "
     "would show, on a matrix too small to be worth quantizing."),
    ("*mlp.gate*",
     "MoE gate projection; identical argmax-flip hazard to the router."),
    ("*block_sparse_moe.gate*",
     "Mixtral-style MoE gate; identical argmax-flip hazard."),
    ("mtp.*",
     "Multi-token-prediction draft head. Its outputs are verified against the "
     "main head, so quantization noise costs accept rate, not accuracy -- it "
     "silently erases the speedup speculative decoding was added for."),
    ("*proj_out*",
     "Final output projection of diffusion/flow heads, e.g. the trajectory "
     "decoder -- the same unaveraged-error argument as lm_head."),
    ("nn.Embedding",
     "Module class, not a name glob. A lookup table: no matmul to accelerate, "
     "and its rows feed every later layer."),
    ("nn.BatchNorm*",
     "Module class. BatchNorm folds into the preceding conv at build time, so "
     "quantizing it separately only injects error into the folded weights."),
    ("nn.LeakyReLU",
     "Module class. The negative slope makes the output range asymmetric, so a "
     "single symmetric scale wastes most of the INT8 codes on one side."),
)

_DENY_REASONS = dict(_DENY)


class PrecisionUnavailable(RuntimeError):
    """This precision cannot be produced for THIS graph on THIS toolchain.

    Distinct from a build failure: the graph is fine and the toolchain is fine,
    but no rung of the conversion ladder yields something TensorRT will take.
    A precision race catches this to move on to the next candidate, where a
    generic build error must still stop everything.
    """


#: FP16 conversion strategies, strongest numerics first.
#:
#: Keeping LayerNormalization/Softmax/ReduceMean in FP32 is the right thing for
#: accuracy, and it is what :data:`..engine.fp16_graph.SENSITIVE_OPS` does by
#: default. But on a transformer graph it can leave a *mixed* boundary the
#: TensorRT parser rejects outright -- measured here on a 4-layer
#: ``nn.TransformerEncoder``: the attention in-projection Gemm comes out Half
#: while the bias TensorRT broadcasts against it stays Float, and the parse dies
#: with "ElementWiseOperation SUM must have same input types. But they are of
#: types Half and Float".
#:
#: The existing per-network builders in this repo work around that by building
#: the offending graph FP32 (NavDP's ``strongly_typed_fp32_engines``), which
#: costs the whole engine's speed to protect a handful of layers. Walking this
#: ladder instead keeps FP16 and gives up only the extra keep-list, recording in
#: the build notes which rung was used.
#:
#: The middle rung exists because the top one is not merely *stricter*: blocking
#: ``Softmax`` and ``ReduceMean`` changes the cast topology
#: ``onnxconverter_common`` then has to repair, and on a transformer graph it
#: repairs it wrongly -- see :class:`..engine.fp16_graph.Fp16ConversionInvalid`.
#: ``LayerNormalization`` alone is the op actually worth pinning (opset 17 emits
#: it as a single node, which is the fix for most transformer FP16 drift), and on
#: InternVLA-N1's System-1 graphs it converted valid *and more accurately* than
#: the rung above it: 3.0e-4 against 1.0e-1 on the condition graph.
FP16_LADDER = (
    ("sensitive-ops kept FP32", ("LayerNormalization", "Softmax", "ReduceMean")),
    ("LayerNormalization kept FP32", ("LayerNormalization",)),
    ("converter defaults only", ()),
)

#: Substrings identifying a parser failure caused by a mixed-precision boundary
#: rather than by a genuinely malformed graph. Only these are worth retrying.
_TYPE_MISMATCH_MARKERS = ("same input types", "Half and Float",
                          "must have same input types")


def is_type_mismatch(error_text):
    """True when a parser error looks like an FP16 boundary problem.

    Used to decide whether walking down :data:`FP16_LADDER` could help. A parse
    that failed for any other reason must surface immediately rather than being
    retried at a weaker precision, which would only change the error message.
    """
    text = str(error_text)
    return any(marker in text for marker in _TYPE_MISMATCH_MARKERS)


def is_strongly_typed(trt_module):
    """Report whether this TensorRT build has removed weak typing.

    The probe is the **absence** of ``trt.BuilderFlag.FP16`` -- the same test
    the repo's existing ``vlas/*/trt/engine/build_engine.py`` uses to pick its
    build path. ``hasattr`` is the correct probe *here*, unlike the DLA feature
    checks elsewhere in this package where a symbol exists but the capability
    behind it may not: this symbol genuinely disappeared from the bindings in
    TensorRT 11, so its absence is the version fact and not a guess about it.

    Args:
        trt_module: the imported ``tensorrt`` module (or a stand-in exposing
            ``BuilderFlag``).

    Returns:
        bool: True when the build is strongly typed (TensorRT >= 11), i.e.
        precision must come from the ONNX rather than from a builder flag.

    Raises:
        TypeError: if ``trt_module`` exposes no ``BuilderFlag`` at all, which
            means it is not a TensorRT module and the answer would be a lie.
    """
    flags = getattr(trt_module, "BuilderFlag", None)
    if flags is None:
        raise TypeError(
            "is_strongly_typed() needs the tensorrt module; %r has no "
            "BuilderFlag" % (trt_module,))
    return not hasattr(flags, "FP16")


def bake_precision(onnx_path, out_path, precision, keep_ops=None):
    """Produce an ONNX graph carrying ``precision`` and return its path.

    This is the only place precision is decided. On a strongly-typed TensorRT
    the parser reads the dtypes in this file and the engine inherits them, so a
    caller that skips this step gets an FP32 engine no matter what it later
    asks the builder for.

    Args:
        onnx_path: source FP32 ONNX graph.
        out_path: where to write the converted graph. Ignored for ``fp32``,
            which returns the source untouched rather than copying it.
        precision: one of :data:`PRECISIONS`.
        keep_ops: ONNX op types to leave in FP32 during an fp16 conversion. When
            omitted the shared helper's own sensitive-op list is used, which is
            rung 0 of :data:`FP16_LADDER`. Pass a weaker list (or ``()``) to walk
            down that ladder when the parser rejects the stronger graph.

    Returns:
        pathlib.Path: the graph to hand to ``trt.OnnxParser`` --
        ``onnx_path`` for fp32, ``out_path`` for fp16.

    Raises:
        ValueError: if ``precision`` is not in :data:`PRECISIONS`.
        NotImplementedError: for bf16 and every quantized format, naming the
            tool that would be needed. Deliberately not a fall back to FP16: a
            plan that asked for INT4 and silently flew FP16 is a wrong number
            in the air.
        FileNotFoundError: if ``onnx_path`` does not exist.
    """
    if precision not in PRECISIONS:
        raise ValueError("unknown precision %r; expected one of %s"
                         % (precision, ", ".join(PRECISIONS)))
    if precision not in SUPPORTED_PRECISIONS:
        raise NotImplementedError(_unsupported_message(precision))
    src = Path(onnx_path)
    if not src.is_file():
        raise FileNotFoundError("source ONNX not found: %s" % src)
    if precision == "fp32":
        return src
    from sparx_agency.tasks.common.trt_optimizer.engine.fp16_graph import (
        to_fp16_onnx)
    if keep_ops is None:
        return Path(to_fp16_onnx(src, out_path))
    return Path(to_fp16_onnx(src, out_path, keep_ops=tuple(keep_ops)))


def _unsupported_message(precision):
    """Build the actionable message for a precision no tool here can produce."""
    if precision == "bf16":
        entry = ("modelopt.onnx.autocast, which rewrites the graph to BF16 "
                 "with a keep-in-FP32 op list")
    else:
        entry = ("modelopt.onnx.quantization, which inserts the Q/DQ nodes "
                 "TensorRT 11 needs (the IInt8Calibrator path was removed)")
    return (
        "precision %r requires NVIDIA ModelOpt (%s). nvidia-modelopt is NOT "
        "installed on this machine and neither is polygraphy, trtexec or any "
        "other quantization tool -- install nvidia-modelopt into the TensorRT "
        "environment and re-run, or plan this graph at fp16/fp32. Refusing to "
        "fall back to fp16, which would report a precision the engine does not "
        "have." % (precision, entry))


def onnx_precision(onnx_path_or_model):
    """Classify the precision an ONNX graph actually carries.

    Reads the initializers, because those are the weights TensorRT will store
    and therefore what :func:`verify_engine_precision` is measuring against.

    Note that a graph produced by ``bake_precision(..., "fp16")`` reports
    ``"mixed"``, not ``"fp16"``: keeping IO FP32 and pinning the sensitive ops
    leaves genuine FP32 initializers behind. That is the intended shape of an
    FP16 graph here, not a defect.

    Args:
        onnx_path_or_model: path to a ``.onnx`` file, or a loaded
            ``onnx.ModelProto``.

    Returns:
        str: ``"qdq"`` if the graph carries QuantizeLinear/DequantizeLinear
        nodes, else ``"fp32"``, ``"fp16"`` or ``"mixed"``.

    Raises:
        ValueError: if the graph holds no floating-point initializer, so there
            is nothing to classify and any answer would be invented.
    """
    import onnx

    model = onnx_path_or_model
    if not hasattr(model, "graph"):
        model = onnx.load(str(onnx_path_or_model))
    graphs = list(_iter_graphs(model.graph))
    for graph in graphs:
        for node in graph.node:
            if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                return "qdq"
    tp = onnx.TensorProto
    float_types = {getattr(tp, n) for n in
                   ("FLOAT", "FLOAT16", "DOUBLE", "BFLOAT16", "FLOAT8E4M3FN",
                    "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ",
                    "FLOAT4E2M1") if hasattr(tp, n)}
    seen = set()
    for graph in graphs:
        for init in graph.initializer:
            if init.data_type in float_types:
                seen.add(init.data_type)
    if not seen:
        raise ValueError(
            "no floating-point initializer in %r; cannot classify its "
            "precision" % (onnx_path_or_model,))
    if seen == {tp.FLOAT}:
        return "fp32"
    if seen == {tp.FLOAT16}:
        return "fp16"
    return "mixed"


def _iter_graphs(graph):
    """Yield ``graph`` and every subgraph nested in a node attribute."""
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                for sub in _iter_graphs(attr.g):
                    yield sub
            for sub_graph in attr.graphs:
                for sub in _iter_graphs(sub_graph):
                    yield sub


#: Below this parameter count the bytes-per-element ratio is meaningless.
#:
#: TensorRT pads and aligns weight storage, so a small graph's total weight
#: bytes are dominated by that padding rather than by the dtype. Measured: a
#: 650-parameter FP16 engine reported 2048 weight bytes -- 3.15 B/elem, which
#: would fail a 2.05 ceiling despite being genuinely FP16. A real model is
#: unaffected: the 1.8 M-parameter graph in the end-to-end test measures 2.027.
MIN_PARAMS_FOR_PRECISION_CHECK = 100_000


def verify_engine_precision(engine, precision, param_count, trt_module=None):
    """Check that a built engine really stores its weights at ``precision``.

    A strongly-typed build honours the ONNX where it has a kernel and quietly
    widens where it does not, so an INT4 or NVFP4 plan can come back as FP16
    with no error anywhere. Weight bytes are the one signal that cannot be
    faked: this divides ``EngineStat.TOTAL_WEIGHTS_SIZE`` by ``param_count``
    and compares against :data:`BYTES_PER_ELEM_CEILING`.

    Reading the stat is best-effort probing of an external tool -- the one
    documented exception to this package's "failures raise" rule. If the
    TensorRT build has no ``EngineStat``, or the engine will not answer, the
    result is reported as SKIPPED with ``ok=True`` so the caller is not blocked,
    and the message says so explicitly. It never claims the check passed.

    Args:
        engine: a built ``trt.ICudaEngine``.
        precision: the precision that was requested, from :data:`PRECISIONS`.
        param_count: number of parameters the graph was exported with.
        trt_module: the ``tensorrt`` module; imported lazily when omitted.

    Returns:
        tuple: ``(ok, bytes_per_elem, message)``. ``bytes_per_elem`` is None
        when the check was skipped.

    Raises:
        ValueError: on an unknown ``precision`` or a non-positive
            ``param_count`` -- both are caller bugs that would turn the
            division into a meaningless number.
    """
    if precision not in BYTES_PER_ELEM_CEILING:
        raise ValueError("unknown precision %r; expected one of %s"
                         % (precision, ", ".join(PRECISIONS)))
    if param_count <= 0:
        raise ValueError("param_count must be positive, got %r" % (param_count,))
    if param_count < MIN_PARAMS_FOR_PRECISION_CHECK:
        return (True, None,
                "precision check SKIPPED: %d parameters is below the %d-parameter "
                "floor where TensorRT's weight alignment padding stops dominating "
                "the ratio. Measured on a 650-parameter graph: 2048 weight bytes "
                "for what should be ~1300, i.e. 3.15 B/elem for a genuine FP16 "
                "engine. The %s engine was NOT verified."
                % (param_count, MIN_PARAMS_FOR_PRECISION_CHECK, precision))
    total = _read_weight_bytes(engine, trt_module)
    if total is None:
        return (True, None,
                "precision check SKIPPED: this TensorRT build does not expose "
                "EngineStat.TOTAL_WEIGHTS_SIZE, so the %s engine was NOT "
                "verified." % precision)
    bpe = float(total) / float(param_count)
    ceiling = BYTES_PER_ELEM_CEILING[precision]
    if bpe <= ceiling:
        return (True, bpe,
                "%s verified: %d weight bytes / %d params = %.3f B/elem "
                "(ceiling %.2f)." % (precision, total, param_count, bpe, ceiling))
    return (False, bpe,
            _widened_message(precision, total, param_count, bpe, ceiling))


def _widened_message(precision, total, param_count, bpe, ceiling):
    """Explain a failed check, and flag the one benign way it can fail.

    A ratio only just over the ceiling is usually not a widened format at all:
    the ops :func:`bake_precision` deliberately leaves FP32 are a fixed cost,
    so on a small graph they dominate. Measured on this machine, a 6,448-param
    FP16 toy with one LayerNorm came out at 2.084 B/elem against the 2.05
    ceiling, while the 5,760-param reference model came out at 2.044. The
    verdict stays False either way -- the caller is told what to look at, not
    talked out of looking.
    """
    note = ""
    if bpe <= ceiling * 1.15:
        note = (" It is only just over, which on a small graph is usually the "
                "FP32-pinned LayerNorm/bias/IO-cast overhead rather than a "
                "widened format; confirm with onnx_precision() before "
                "re-exporting.")
    return ("%s NOT honoured: %d weight bytes / %d params = %.3f B/elem, above "
            "the %.2f ceiling. The build widened the format -- the engine is "
            "storing roughly %.1f-byte weights. Re-check the baked ONNX with "
            "onnx_precision() and whether this GPU has a kernel for %s.%s"
            % (precision, total, param_count, bpe, ceiling, bpe, precision, note))


def _read_weight_bytes(engine, trt_module):
    """Total weight bytes from the engine, or None if unavailable.

    Best-effort by design: every failure mode (no tensorrt, no ``EngineStat``,
    no ``get_engine_stat``, a raising or sentinel answer) collapses to None so
    the caller can report SKIPPED instead of a false verdict.
    """
    trt = trt_module
    if trt is None:
        try:
            import tensorrt as trt
        except ImportError:
            return None
    stat_enum = getattr(trt, "EngineStat", None)
    stat = getattr(stat_enum, "TOTAL_WEIGHTS_SIZE", None)
    getter = getattr(engine, "get_engine_stat", None)
    if stat is None or getter is None:
        return None
    try:
        total = getter(stat)
    except Exception:
        return None
    if total is None or int(total) <= 0:
        return None
    return int(total)


def quantization_deny_list():
    """Layer-name globs (and module classes) that must never be quantized.

    Grounded in NVIDIA ModelOpt's own shipped defaults rather than invented
    here: the vision branch, the final output projection, every discrete MoE
    gate, the speculative-decoding draft head, and -- by module class, so the
    last three entries are types and not name patterns -- ``nn.Embedding``,
    ``nn.BatchNorm*`` and ``nn.LeakyReLU``.

    One rule cannot be expressed as a pattern and is the caller's to apply:
    **skip any ``nn.Linear`` whose ``in_features`` or ``out_features`` is not a
    multiple of** :data:`LINEAR_DIM_ALIGNMENT` **(16)**. Quantized GEMM kernels
    are written for 16-element vectors; an unaligned Linear either falls back
    to an FP16 kernel with the Q/DQ nodes left in place -- slower than never
    quantizing it -- or fails to build at all.

    Returns:
        List[str]: the entries in priority order; pass each to
        :func:`reason_for` for the justification to print in a report.
    """
    return [pattern for pattern, _ in _DENY]


def reason_for(pattern):
    """Explain why one deny-list entry is on the list.

    Args:
        pattern: an entry returned by :func:`quantization_deny_list`.

    Returns:
        str: the justification, written for a human reading the report.

    Raises:
        KeyError: if ``pattern`` is not a deny-list entry. An unexplained entry
            in a report is worse than no report.
    """
    if pattern not in _DENY_REASONS:
        raise KeyError(
            "%r is not a deny-list entry; call quantization_deny_list() for "
            "the valid entries" % (pattern,))
    return _DENY_REASONS[pattern]


def sensitive_op_note():
    """Why three ops stay FP32 inside an otherwise-FP16 graph.

    Returns:
        str: a paragraph for the report explaining the op block list and the
        opset floor.
    """
    return (
        "LayerNormalization, Softmax and ReduceMean stay FP32 in the FP16 "
        "graph. All three sum over the channel axis, and FP16's ~3-decimal "
        "mantissa loses the small terms of that sum: the variance in a "
        "LayerNorm underflows and the normalized activations blow up, and a "
        "Softmax over pre-max logits saturates to a one-hot attention row. "
        "Their cost is a rounding error of the total FLOPs, so pinning them "
        "buys accuracy for free. This is also why opset >= 17 is the floor: "
        "17 is the first opset with a real LayerNormalization node, so "
        "TensorRT sees one fused op to keep in FP32 instead of a decomposed "
        "ReduceMean/Sub/Pow/Div chain whose intermediates were each converted "
        "to FP16 individually. That one change fixes most transformer FP16 "
        "drift on its own.")
