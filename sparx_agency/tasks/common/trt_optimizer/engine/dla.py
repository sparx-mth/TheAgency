"""Answer "does any of this belong on the NVDLA?" with evidence, not optimism.

An idle NVDLA beside a saturated Orin GPU is permanently tempting, and the
temptation is usually wrong. This module exists to make the refusal cheap,
specific and quotable in the report -- and to name the one shape that does pay.

Three gates, in this order:

1. **The runtime must still have DLA at all.** NVIDIA states TensorRT 10.7 was
   the *last* release supporting DLA; TensorRT 11.0/11.1/11.2 do not support it
   in any form. The trap is detection, not the build: ``trt.DeviceType.DLA``,
   ``trt.BuilderFlag.GPU_FALLBACK``, ``trt.MemoryPoolType.DLA_MANAGED_SRAM``
   and ``trt.OnnxParserFlag.REPORT_CAPABILITY_DLA`` all still *exist* in
   TensorRT 11.1, where DLA does not work, so a ``hasattr`` probe cheerfully
   reports a DLA that will never produce an engine.
   :func:`runtime_supports_dla` gates on the parsed version *and* on
   ``trt.Runtime(logger).num_DLA_cores``, and never on attribute existence.
2. **The board must expose cores.** An x86 dGPU has none; a Jetson reports them
   through :class:`~sparx_agency.tasks.common.hardware.detect.HardwareProfile`.
3. **The eligible region must be one contiguous prefix.** A graph that
   ping-pongs between DLA and GPU is worse than one that simply stays on the
   GPU. Every DLA<->GPU boundary is a reformat between the GPU's linear NCHW
   layout and DLA's ``kDLA_LINEAR``/``kCHW16``, plus a subgraph launch; a
   handful of those and the partitioning has eaten the whole win. The payoff
   shape is exactly one handoff: a CNN visual backbone pinned entirely to DLA
   in INT8, feeding a GPU-side transformer head.

Two constraints never visible in an op table, and just as fatal:

* **No dynamic shapes.** A DLA engine needs an optimization profile with
  ``min == opt == max`` on every dimension. That costs nothing here because
  :class:`~sparx_agency.tasks.common.trt_optimizer.spec.GraphSpec` is static by
  contract, but a graph with a dynamic batch or image size is not a candidate.
* **INT8 or don't bother.** On Orin, DLA in FP16 is GPU-class at best; INT8 is
  the only precision where its throughput per watt clearly repays the layer
  restrictions, and INT8 means calibration scales have to exist before the
  build.

Pure standard library: ``tensorrt`` and ``onnx`` are imported lazily inside the
functions that need them, so this module imports on the dev laptop, on the Orin
and inside a Noetic container. Python-3.8-compatible syntax throughout.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

#: Last ``(major, minor)`` TensorRT release with DLA support. Anything newer is
#: refused outright: NVIDIA states 10.7 was the last release supporting DLA and
#: 11.x dropped it, keeping only the vestigial enums that fool ``hasattr``.
LAST_TRT_WITH_DLA = (10, 7)

DLA_UNSUPPORTED_OPS = frozenset([
    # Transformer normalization and attention: no DLA layer maps to these.
    "LayerNormalization", "RMSNormalization", "GroupNormalization",
    "InstanceNormalization",
    # Softmax runs on DLA only over a small reduced dimension; over a large one
    # (attention logits, a wide classifier) it is either rejected or so slow it
    # is a loss, so it is counted unsupported here.
    "Softmax", "LogSoftmax",
    # A MatMul/Einsum whose second input is a *constant* is really a fully
    # connected layer and DLA can run it; with a non-constant second input
    # (attention Q@K^T) it has no DLA mapping. Counted unsupported by name
    # because the scanner judges op types, not initializers -- these two are
    # the entries worth a human second look when they dominate a report.
    "MatMul", "Einsum",
    # Gather/scatter/sort/select: index-driven, not a DLA layer.
    "Gather", "GatherElements", "GatherND", "ScatterND", "ScatterElements",
    "TopK", "NonZero", "NonMaxSuppression", "Where",
    # Transcendental activations of the GELU family (Erf-based).
    "Erf", "Gelu",
    # Dynamic-shape and control-flow constructs. DLA needs min==opt==max on
    # every dimension, so anything that computes or consumes a shape at run
    # time disqualifies the region outright.
    "Shape", "Range", "ConstantOfShape", "Expand", "Loop", "If", "Scan",
])
"""ONNX op types the NVDLA cannot run (or cannot run at a profit).

Grounded in the TensorRT DLA layer-support list rather than guessed. Note the
two nuances documented inline: ``Softmax`` is listed because DLA only handles a
*small* reduced dimension, and ``MatMul``/``Einsum`` are listed because the
profitable case (constant second input == fully connected) cannot be told from
the hostile case (attention) by op type alone.
"""

DLA_SUPPORTED_OPS = frozenset([
    "Conv", "ConvTranspose", "Gemm",
    "Relu", "LeakyRelu", "PRelu", "Clip", "Sigmoid", "Tanh",
    "MaxPool", "AveragePool", "GlobalAveragePool",
    "Add", "Mul", "Sub",
    "Concat", "Slice", "Pad", "Resize",
    "BatchNormalization",
    "Flatten", "Reshape",
])
"""ONNX op types the NVDLA runs well -- the classic CNN vocabulary.

``Resize`` means nearest or bilinear only, ``Reshape`` means a static target
shape (a dynamic one lands in :data:`DLA_UNSUPPORTED_OPS` via ``Shape``), and
``BatchNormalization`` is expected to fold into the preceding convolution.
Anything in neither table is reported as *unknown* rather than assumed: an
optimistic default here is what produces an engine that silently runs on the
GPU after all.
"""


def _trt_version(trt_module):
    """Parse ``(major, minor)`` from a TensorRT module, None if unreadable."""
    raw = getattr(trt_module, "__version__", "")
    m = re.match(r"\s*(\d+)\.(\d+)", str(raw))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _num_dla_cores(trt_module):
    """Read ``trt.Runtime(logger).num_DLA_cores`` -- the only honest probe."""
    logger = trt_module.Logger(trt_module.Logger.ERROR)
    return int(trt_module.Runtime(logger).num_DLA_cores)


def runtime_supports_dla(trt_module=None, hardware=None):
    """Report whether this TensorRT runtime can build a DLA engine at all.

    Gates on both facts that matter, and on neither of the two that lie. The
    version gate is first because it is decisive: TensorRT 11.x removed DLA
    while keeping ``DeviceType.DLA``, ``BuilderFlag.GPU_FALLBACK``,
    ``MemoryPoolType.DLA_*`` and ``OnnxParserFlag.REPORT_CAPABILITY_DLA``, so
    every ``hasattr``-style probe answers "yes" on a runtime that cannot do it.

    Args:
        trt_module: an already-imported ``tensorrt`` module, or None to import
            it lazily here. Passing it in is what makes this testable without
            TensorRT installed.
        hardware: optional ``HardwareProfile``-like object with ``dla_cores``.
            When given and it reports no cores, the answer is settled before
            ``tensorrt`` is imported at all.

    Returns:
        Tuple of ``(supported, reason)``. ``reason`` is written to be quoted
        verbatim in the report, and is filled in the negative case too.

    Note:
        This is the documented exception to the repo's "failures raise" rule:
        probing an external toolchain is best-effort, so a missing module, an
        unparseable version or a runtime that refuses to instantiate all return
        ``(False, reason)``. The failure direction is safe -- refusing DLA
        yields an ordinary GPU engine, never a wrong number in flight.
    """
    if hardware is not None and hardware.dla_cores <= 0:
        return (False, "hardware profile reports dla_cores=%d (no DLA on this "
                       "board)" % hardware.dla_cores)
    if trt_module is None:
        try:
            import tensorrt as trt_module  # noqa: F811 (lazy, optional dep)
        except ImportError as exc:
            return (False, "tensorrt is not importable here (%s), so DLA "
                           "cannot be probed and is refused" % exc)
    version = _trt_version(trt_module)
    if version is None:
        return (False, "cannot parse a TensorRT version from %r; DLA refused"
                       % getattr(trt_module, "__version__", None))
    if version > LAST_TRT_WITH_DLA:
        return (False, "TensorRT %d.%d has no DLA support: %d.%d was the last "
                       "release supporting DLA, and 11.x removed it entirely "
                       "(its surviving DLA enums are vestigial)"
                       % (version[0], version[1],
                          LAST_TRT_WITH_DLA[0], LAST_TRT_WITH_DLA[1]))
    try:
        cores = _num_dla_cores(trt_module)
    except Exception as exc:  # noqa: BLE001 -- best-effort probe, see Note
        return (False, "trt.Runtime(logger).num_DLA_cores could not be read "
                       "(%s: %s); DLA refused" % (type(exc).__name__, exc))
    if cores <= 0:
        return (False, "TensorRT %d.%d reports num_DLA_cores=%d"
                       % (version[0], version[1], cores))
    return (True, "TensorRT %d.%d reports num_DLA_cores=%d"
                  % (version[0], version[1], cores))


def _graph_nodes(onnx_path_or_model):
    """Return the graph nodes of a path, a ModelProto or a duck-typed stand-in.

    Raises:
        ImportError: a path was given and the ``onnx`` package is absent.
        TypeError: the object exposes no ``.graph.node``.
    """
    if isinstance(onnx_path_or_model, (str, Path)):
        try:
            import onnx
        except ImportError as exc:
            raise ImportError(
                "scan_ops(%r) needs the onnx package to read the file; pass "
                "an already-loaded model instead, or run this on an "
                "interpreter with onnx installed"
                % str(onnx_path_or_model)) from exc
        model = onnx.load(str(onnx_path_or_model))
    else:
        model = onnx_path_or_model
    graph = getattr(model, "graph", None)
    nodes = getattr(graph, "node", None)
    if nodes is None:
        raise TypeError("scan_ops expected a path or an object with "
                        ".graph.node, got %r" % type(onnx_path_or_model))
    return nodes


def scan_ops(onnx_path_or_model):
    """Classify every node of an ONNX graph against the DLA op tables.

    Args:
        onnx_path_or_model: path to a ``.onnx`` file (``onnx`` is imported
            lazily to read it), or any object exposing ``.graph.node`` with an
            ``.op_type`` on each node -- a ``ModelProto`` or a stand-in.

    Returns:
        Dict with:
          * ``total``: node count.
          * ``supported`` / ``unsupported`` / ``unknown``: op type -> count.
            An op in neither table lands in ``unknown``; it is never assumed
            eligible.
          * ``first_unsupported_index``: index of the first *known-unsupported*
            node, or None. Unknown ops do not set it -- they need a human, not
            a verdict.
          * ``contiguous_supported_prefix``: how many leading nodes are
            known-supported. This stops at the first unknown node too, because
            a node nobody has vouched for breaks the DLA region just as
            effectively as a forbidden one.
    """
    supported = {}
    unsupported = {}
    unknown = {}
    first_unsupported_index = None
    prefix = 0
    prefix_open = True
    total = 0
    for index, node in enumerate(_graph_nodes(onnx_path_or_model)):
        op = node.op_type
        total += 1
        if op in DLA_UNSUPPORTED_OPS:
            unsupported[op] = unsupported.get(op, 0) + 1
            if first_unsupported_index is None:
                first_unsupported_index = index
        elif op in DLA_SUPPORTED_OPS:
            supported[op] = supported.get(op, 0) + 1
        else:
            unknown[op] = unknown.get(op, 0) + 1
        if prefix_open and op in DLA_SUPPORTED_OPS:
            prefix += 1
        else:
            prefix_open = False
    return {
        "total": total,
        "supported": supported,
        "unsupported": unsupported,
        "unknown": unknown,
        "first_unsupported_index": first_unsupported_index,
        "contiguous_supported_prefix": prefix,
    }


@dataclass
class DlaVerdict:
    """The DLA decision for one graph, with the evidence that produced it.

    Args:
        use_dla: True only when every gate passed.
        eligible_fraction: known-supported nodes / total nodes. 0.0 when the
            graph was never scanned: a runtime or board gate fired first.
        contiguous_prefix: leading known-supported node count.
        why: the rule that fired, phrased for a human reading the report.
        unsupported_sample: the dominant blockers, ``"Op xN"``, most frequent
            first -- the ops actually found, never a generic list.
    """

    use_dla: bool
    eligible_fraction: float = 0.0
    contiguous_prefix: int = 0
    why: str = ""
    unsupported_sample: List[str] = field(default_factory=list)


def _top_ops(counts, limit=5):
    """Format the ``limit`` most frequent ops as ``["MatMul x24", ...]``."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ["%s x%d" % (op, n) for op, n in ranked[:limit]]


_PAYOFF_WHY = (
    "%d of %d nodes are DLA-eligible (%.0f%%) and they are one contiguous "
    "prefix, which is the only shape that pays: a CNN visual backbone pinned "
    "entirely to DLA in INT8 feeding a GPU-side transformer head -- ONE "
    "handoff, not dozens. Build it INT8 with min==opt==max shapes."
)


def evaluate(onnx_path, hardware, trt_module=None, min_eligible_fraction=0.60):
    """Decide whether this graph should be built for DLA.

    Args:
        onnx_path: anything :func:`scan_ops` accepts -- a ``.onnx`` path or an
            already-loaded model. The runtime and board gates are checked
            first, so a hopeless target never pays to read the file.
        hardware: ``HardwareProfile``-like object exposing ``dla_cores``.
        trt_module: an imported ``tensorrt`` module, or None to import lazily.
        min_eligible_fraction: the share of nodes that must be known-supported.
            0.60 by default: below that, the GPU is doing most of the work
            anyway and DLA only adds boundaries.

    Returns:
        :class:`DlaVerdict`.

    Raises:
        ValueError: ``min_eligible_fraction`` outside ``[0, 1]``, or a graph
            with no nodes -- an empty graph is a broken export, not a verdict.

    Note:
        ``hardware`` is deliberately *not* forwarded to
        :func:`runtime_supports_dla`: the runtime rule and the board rule are
        reported separately so the report says which one actually fired.
    """
    if not 0.0 <= min_eligible_fraction <= 1.0:
        raise ValueError("min_eligible_fraction must be in [0, 1], got %r"
                         % (min_eligible_fraction,))
    ok, reason = runtime_supports_dla(trt_module=trt_module)
    if not ok:
        return DlaVerdict(False, why="No DLA on this runtime: %s" % reason)
    if hardware.dla_cores <= 0:
        return DlaVerdict(False, why="the target board reports dla_cores=%d, "
                                     "so there is nothing to offload to"
                                     % hardware.dla_cores)
    scan = scan_ops(onnx_path)
    total = scan["total"]
    if total == 0:
        raise ValueError("the ONNX graph has no nodes; nothing to evaluate")
    eligible = sum(scan["supported"].values())
    fraction = eligible / float(total)
    prefix = scan["contiguous_supported_prefix"]
    sample = _top_ops(scan["unsupported"])
    return _judge_graph(scan, fraction, prefix, sample, min_eligible_fraction)


def _judge_graph(scan, fraction, prefix, sample, min_eligible_fraction):
    """Apply the two graph-shape rules to a finished op scan."""
    total = scan["total"]
    eligible = sum(scan["supported"].values())
    unknown_n = sum(scan["unknown"].values())
    if fraction < min_eligible_fraction:
        blockers = ", ".join(sample + _top_ops(scan["unknown"], 2)) or "none"
        return DlaVerdict(False, fraction, prefix, unsupported_sample=sample,
                          why="only %.0f%% of %d nodes are DLA-eligible (need "
                              "%.0f%%); the blockers found are %s -- keep "
                              "this graph on the GPU"
                              % (100.0 * fraction, total,
                                 100.0 * min_eligible_fraction, blockers))
    if prefix < eligible:
        return DlaVerdict(False, fraction, prefix, unsupported_sample=sample,
                          why="the %d eligible nodes are not one contiguous "
                              "prefix (only the first %d are); every "
                              "DLA<->GPU boundary costs a "
                              "kDLA_LINEAR/kCHW16 reformat and the "
                              "partitioning eats the win"
                              % (eligible, prefix))
    why = _PAYOFF_WHY % (eligible, total, 100.0 * fraction)
    if unknown_n:
        why += (" %d trailing node(s) of unknown DLA status will fall back to "
                "the GPU; check them before trusting the split." % unknown_n)
    return DlaVerdict(True, fraction, prefix, why, sample)


_POWER_RATIO = (
    "At 15 W an AGX Orin's GPU drops to ~11.8% of its MAXN compute while DLA "
    "holds ~38.4% of its own, so DLA is worth roughly 3.3x more at 15 W than "
    "at MAXN -- the power cap is the strongest argument DLA has."
)
_NX_ONE_CORE = (
    "Careful on an Orin NX 16GB: at 15 W only ONE DLA core is enabled; both "
    "cores appear only at MAXN and 25 W, so a two-core plan silently "
    "becomes a one-core plan when the board is capped."
)
_BUILD_AT_MAXN = (
    "Do not BUILD engines at 15 W: build at MAXN with jetson_clocks, keep the "
    "timing cache, then switch the board to 15 W to fly. A capped build times "
    "its tactics against a starved clock and bakes the wrong kernels in."
)


def power_note(hardware):
    """Return the 15 W paragraph for the report, or ``''`` when irrelevant.

    Args:
        hardware: ``HardwareProfile``-like object exposing ``dla_cores``,
            ``is_jetson``, ``power_budget_w`` and ``jetson_model``.

    Returns:
        A report paragraph, or ``''`` on a board with no DLA (an x86 dGPU has
        nothing to say about power modes it does not have).
    """
    if not hardware.is_jetson or hardware.dla_cores <= 0:
        return ""
    parts = [_POWER_RATIO]
    watts = hardware.power_budget_w
    model = (hardware.jetson_model or "").lower()
    if "orin nx" in model:
        parts.append(_NX_ONE_CORE)
    if watts is None:
        parts.append("This board's active nvpmodel budget could not be read, "
                     "so assume the capped picture until it is.")
    elif watts <= 15:
        parts.append("This board is capped at %d W right now." % watts)
    parts.append(_BUILD_AT_MAXN)
    return " ".join(parts)
