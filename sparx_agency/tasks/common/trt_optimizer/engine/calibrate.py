"""Reach INT8 on either TensorRT generation -- or refuse, and say exactly how.

INT8 is the one quantized format that reliably pays on both target classes this
package aims at. Measured here on sm_120, a 2048-cube GEMM ran INT8 at 149.5
TFLOP/s against FP16's 75.5 -- 1.98x, where FP8 managed 1.10x and NVFP4 was
*slower* than FP16. On an Orin it is the only precision where the DLA's layer
restrictions repay themselves at all. So a toolkit that cannot reach INT8 is not
finished, and this module owns the two genuinely different ways of getting
there:

* **The entropy-calibrator route -- TensorRT <= 10**, which is the Jetson's
  JetPack stack. The builder is weakly typed, ``trt.IInt8EntropyCalibrator2``
  still exists, and INT8 comes from a builder flag plus a calibrator that
  streams real batches through the network while TensorRT fits one activation
  range per tensor. :func:`make_entropy_calibrator` builds that object for a
  graph with any number of inputs, in the order TensorRT asks for them.
* **The Q/DQ route -- TensorRT >= 11**, which is this machine. Every
  ``IInt8Calibrator`` class is gone along with every precision ``BuilderFlag``,
  so the only remaining way to ask for INT8 is to hand the parser an ONNX that
  already carries QuantizeLinear/DequantizeLinear nodes. That graph is produced
  by ``nvidia-modelopt``, which is not installed here: :func:`qdq_available`
  says so and :func:`qdq_instructions` prints the commands that fix it.

Both routes stand on the same foundation, which is that **an INT8 engine is only
as good as the data its ranges were fitted on**. That is why
:func:`collect_calibration_arrays` takes its samples from the adapter's own
scenarios and refuses a thin or mis-shaped set rather than padding one, and why
:data:`CALIBRATION_GUIDANCE` is written as five numbered rules with the numbers
in them.

What this module will never do is let INT8 quietly become FP32.
:func:`require_int8_buildable` raises with the missing tool named,
:func:`make_entropy_calibrator` refuses on a TensorRT that has no calibrator
class instead of handing back something inert, and
:func:`..engine.precision.bake_precision` already refuses to bake a precision it
cannot produce. A plan that asked for INT8 and silently flew FP32 is a wrong
number in the air.

``tensorrt``, ``pycuda`` and ``modelopt`` are imported lazily, so this module is
readable -- and every decision in it is testable -- on an interpreter that has
none of the three. Python-3.8-compatible syntax throughout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sparx_agency.tasks.common.trt_optimizer.engine import precision as prec

#: Fewest samples that produces a calibration worth flying. See rule 2.
MIN_SAMPLES = 128

#: Sample count at which calibration quality saturates. See rule 2.
MAX_USEFUL_SAMPLES = 512

#: Beyond this the measured delta is under 0.1%: pure wasted wall time.
SATURATION_SAMPLES = 1024

#: INT8 by builder flag + ``IInt8EntropyCalibrator2`` (TensorRT <= 10).
ROUTE_ENTROPY_CALIBRATOR = "entropy_calibrator"

#: INT8 by Q/DQ nodes baked into the ONNX (TensorRT >= 11).
ROUTE_QDQ = "qdq"

#: Precisions that can only be reached through the Q/DQ route.
QDQ_PRECISIONS = ("int8", "fp8", "int4", "nvfp4")

#: The one command that turns "cannot build INT8 here" into "can".
MODELOPT_PIP = 'pip install "nvidia-modelopt[onnx]"'

#: Per-precision Q/DQ settings: mode, calibration method, block size, note.
_QDQ_MODES = {
    "int8": ("int8", "entropy", None,
             "weights per-channel, activations per-tensor (rule 4)"),
    "fp8": ("fp8", "max", None,
            "per-tensor scales; benchmark it -- FP8 measured only 1.10x FP16 "
            "on sm_120, so it is rarely worth the export"),
    "int4": ("int4", "max", 128,
             "WEIGHT-ONLY: activations stay FP16, so it is a memory-bandwidth "
             "win and not a tensor-core one, at block size 64 or 128 -- never "
             "32, which fails to build (rule 5)"),
    "nvfp4": ("nvfp4", "max", 16,
              "block size 16 is fixed by the format; NVFP4 measured SLOWER "
              "than FP16 on sm_120 -- benchmark before believing it (rule 5)"),
}

CALIBRATION_GUIDANCE = """\
Five calibration rules. Each of them is a number, not a preference:

1. In-domain data ONLY. Uniform random input is out of distribution for any
   pretrained backbone: its activations occupy a different range from real
   input, so the fitted scales are wrong in a way that produces confident
   numbers on the calibration set and does not transfer to the deployed one.
   Real captures, or a simulator whose output the model was trained on.

2. 128-512 samples. Below 128 the histograms are too sparse for the entropy
   fit; quality saturates near 512; beyond 1024 the measured delta is under
   0.1% and every extra sample is pure build time.

3. For an ITERATIVE graph -- a denoise, flow or refinement loop -- capture
   across the WHOLE loop, not one step. Step 1 sees near-pure noise and the
   last step sees a nearly clean sample; calibrating on either alone fits a
   range the quantized graph will spend most of its calls outside of.

4. Per-channel weights (axis 0) + per-tensor activations for INT8. Never
   per-tensor weights: one scale across every output channel is set by the
   widest channel, and every narrower channel loses most of its 256 codes.

5. INT4 is weight-only, block size 64 or 128 -- block size 32 fails to build
   with "Autotuner: no tactics to implement operation". NVFP4 block size is 16.
"""


# --------------------------------------------------------------------------
# lazy imports -- each failure names what to install and where
# --------------------------------------------------------------------------

def _import_trt():
    """Import ``tensorrt``, or raise saying which interpreter is wrong."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "tensorrt is not importable from this interpreter (%s), so the "
            "INT8 route cannot be decided. Calibration and the build must run "
            "in the SAME interpreter that will serve the engine -- on this "
            "machine that is the 'navdp' conda env, not .venv. Pass "
            "trt_module= to reason about another machine's toolchain." % exc
        ) from exc
    return trt


def _import_pycuda():
    """Import ``pycuda.driver``, or raise saying what it was needed for."""
    try:
        import pycuda.driver as cuda
    except ImportError as exc:
        raise RuntimeError(
            "pycuda is not importable (%s). The entropy calibrator has to hand "
            "TensorRT DEVICE pointers, so it needs pycuda to allocate and fill "
            "them: `pip install pycuda` into the interpreter that builds "
            "engines. There is no host-memory fallback -- get_batch() may only "
            "return device addresses." % exc
        ) from exc
    return cuda


# --------------------------------------------------------------------------
# which route, and can this machine take it
# --------------------------------------------------------------------------

def int8_route(trt_module=None):
    """Report which INT8 route this TensorRT generation forces, and why.

    Args:
        trt_module: an imported ``tensorrt`` module, or None to import it here.
            Passing it in is what makes the decision testable on a machine with
            no TensorRT, and lets a workstation reason about the Orin's stack.

    Returns:
        ``(route, reason)`` -- :data:`ROUTE_ENTROPY_CALIBRATOR` or
        :data:`ROUTE_QDQ`, plus a sentence written to be quoted in a report.

    Raises:
        RuntimeError: if ``trt_module`` is None and TensorRT is not importable.
            The route is a property of the toolchain, so answering without one
            would be a guess.
    """
    trt = trt_module if trt_module is not None else _import_trt()
    version = getattr(trt, "__version__", "unknown")
    if prec.is_strongly_typed(trt):
        return (ROUTE_QDQ,
                "TensorRT %s is strongly typed: every IInt8Calibrator class and "
                "every precision BuilderFlag was removed, so INT8 can only come "
                "from QuantizeLinear/DequantizeLinear nodes already present in "
                "the ONNX." % version)
    return (ROUTE_ENTROPY_CALIBRATOR,
            "TensorRT %s is weakly typed: BuilderFlag.INT8 plus an "
            "IInt8EntropyCalibrator2 is the supported route, and no Q/DQ graph "
            "is needed." % version)


def qdq_available():
    """Report whether this machine can produce a Q/DQ graph at all.

    Probing an installed package is best-effort, so this returns a verdict
    rather than raising -- it is the question :func:`require_int8_buildable`
    asks before it refuses.

    Returns:
        ``(available, reason)``. ``reason`` names ``modelopt`` either way, so a
        report never has to explain the absence twice.
    """
    try:
        import modelopt.onnx.quantization as _quant  # noqa: F401
    except Exception as exc:  # noqa: BLE001  (probing: absence is the answer)
        return (False,
                "nvidia-modelopt is NOT installed here (%s), and it is the only "
                "producer of a Q/DQ ONNX in this environment -- polygraphy, "
                "trtexec and onnxruntime's quantizer are all absent too. "
                "Without modelopt there is no INT8 on a strongly-typed "
                "TensorRT." % exc)
    version = "unknown version"
    try:
        import modelopt
        version = str(getattr(modelopt, "__version__", version))
    except Exception:  # noqa: BLE001  (version is cosmetic, presence is not)
        pass
    return (True, "modelopt.onnx.quantization is importable (%s), so a Q/DQ "
                  "graph can be produced here." % version)


def require_int8_buildable(trt_module=None):
    """Raise unless this machine can genuinely produce an INT8 engine.

    This is the gate that makes "asked for INT8, silently got FP32" impossible.
    Call it before the build, not after: on the Q/DQ route the failure is
    otherwise invisible, because an un-quantized graph parses perfectly and
    builds a perfectly valid FP32 engine.

    Args:
        trt_module: an imported ``tensorrt`` module, or None to import it here.

    Returns:
        The route name that will be used -- :data:`ROUTE_ENTROPY_CALIBRATOR` or
        :data:`ROUTE_QDQ`.

    Raises:
        RuntimeError: naming the missing tool and, on the Q/DQ route, carrying
            the full :func:`qdq_instructions` text.
    """
    route, reason = int8_route(trt_module)
    if route == ROUTE_QDQ:
        ok, why = qdq_available()
        if not ok:
            raise RuntimeError("%s %s\n\n%s"
                               % (reason, why, qdq_instructions("int8")))
        return route
    _import_pycuda()
    return route


def qdq_instructions(precision):
    """The exact steps that turn "no INT8 here" into a quantized ONNX.

    Args:
        precision: one of :data:`QDQ_PRECISIONS`.

    Returns:
        A multi-line actionable string: the pip command, the
        ``modelopt.onnx.quantization`` call shape with this precision's mode and
        calibration method, the deny list to exclude, and how to check the
        result actually carries the precision.

    Raises:
        ValueError: for a precision with no Q/DQ route (fp32/fp16 need no
            quantization and bf16 is an autocast, not a quantization).
    """
    if precision not in _QDQ_MODES:
        raise ValueError(
            "%r has no Q/DQ route; quantizable precisions are %s. fp32/fp16 "
            "need no quantization (see precision.bake_precision) and bf16 is a "
            "cast, produced by modelopt.onnx.autocast instead."
            % (precision, ", ".join(QDQ_PRECISIONS)))
    mode, method, block, note = _QDQ_MODES[precision]
    block_arg = ("" if block is None
                 else "\n           block_size=%d," % block)
    deny = "\n".join("           %r," % pattern for pattern in deny_list())
    return _QDQ_TEMPLATE % {
        "precision": precision, "PRECISION": precision.upper(),
        "mode": mode, "method": method, "note": note,
        "block": block_arg, "deny": deny, "pip": MODELOPT_PIP,
        "min": MIN_SAMPLES, "max": MAX_USEFUL_SAMPLES,
        "align": prec.LINEAR_DIM_ALIGNMENT,
    }


_QDQ_TEMPLATE = """\
%(PRECISION)s on a strongly-typed TensorRT (>= 11) needs an ONNX that already
carries QuantizeLinear/DequantizeLinear nodes -- %(note)s.

1. Install the only producer, into the interpreter that BUILDS engines:

       %(pip)s

2. Collect %(min)d-%(max)d in-domain samples and save them as an .npz keyed by
   input tensor name (this is exactly what collect_calibration_arrays returns):

       np.savez("calib.npz", **collect_calibration_arrays(adapter, key,
                                                          scenarios, capture))

3. Quantize the exported FP32 ONNX:

       from modelopt.onnx.quantization import quantize
       quantize(
           onnx_path="engines/onnx/<key>.onnx",
           quantize_mode="%(mode)s",
           calibration_method="%(method)s",%(block)s
           calibration_data="calib.npz",
           nodes_to_exclude=DENY,
           output_path="engines/onnx/<key>.%(precision)s.onnx",
       )

   The same thing from a shell, if you would rather not write the script:

       python -m modelopt.onnx.quantization --onnx_path engines/onnx/<key>.onnx \\
           --quantize_mode %(mode)s --calibration_data calib.npz \\
           --output_path engines/onnx/<key>.%(precision)s.onnx

   Confirm the mode spelling against the installed version's --help before a
   long run; modelopt renames modes between releases and an unknown mode is
   rejected rather than ignored.

4. DENY -- pass these as nodes_to_exclude. precision.reason_for(<pattern>)
   gives the justification to print in the report:

       DENY = [
%(deny)s
       ]

   The last three entries are module CLASSES, not name globs, so on the ONNX
   side there are no node names to match them against: exclude them by op type
   instead, op_types_to_exclude=["Gather", "BatchNormalization", "LeakyRelu"].

   Plus the one rule no pattern can express: skip any nn.Linear whose
   in_features or out_features is not a multiple of %(align)d. An unaligned
   quantized GEMM either falls back to an FP16 kernel with the Q/DQ nodes still
   in the graph -- slower than never quantizing it -- or fails to build.

5. Build the QUANTIZED file directly (do not run it through bake_precision,
   which refuses %(precision)s on purpose). Two checks, both cheap:
   precision.onnx_precision(path) must return "qdq" BEFORE the build, and
   precision.verify_engine_precision(engine, "%(precision)s", n_params) must
   pass AFTER it. A build that succeeds proves nothing on its own.
"""


def deny_list():
    """Layer-name globs that must stay out of any quantization.

    Delegates to :func:`..engine.precision.quantization_deny_list` -- the list
    and its justifications live there, next to the rest of the precision
    policy, and duplicating it here is how the two copies would drift.

    Returns:
        List[str]: patterns in priority order. Pass each to
        :func:`..engine.precision.reason_for` for the report text.
    """
    return prec.quantization_deny_list()


def sample_count_advice(n):
    """Judge a calibration sample count against rule 2 (128-512).

    Args:
        n: how many samples were collected.

    Returns:
        ``(ok, message)``. ``ok`` answers "will this count produce a
        calibration worth flying", so it is True for everything at or above
        :data:`MIN_SAMPLES` -- an oversized set is wasted build time, not a bad
        engine, and the message says which of the two it is.
    """
    n = int(n)
    if n < MIN_SAMPLES:
        return (False,
                "%d calibration samples is below the %d minimum: the activation "
                "histograms are too sparse for the entropy fit, so the scales "
                "will be set by whichever few samples happened to be widest. "
                "Collect %d-%d in-domain samples (rule 2)."
                % (n, MIN_SAMPLES, MIN_SAMPLES, MAX_USEFUL_SAMPLES))
    if n <= MAX_USEFUL_SAMPLES:
        return (True, "%d calibration samples is inside the %d-%d band (rule 2)."
                      % (n, MIN_SAMPLES, MAX_USEFUL_SAMPLES))
    if n <= SATURATION_SAMPLES:
        return (True,
                "%d calibration samples is past the %d where quality saturates. "
                "Harmless, but the extra %d buy almost nothing (rule 2)."
                % (n, MAX_USEFUL_SAMPLES, n - MAX_USEFUL_SAMPLES))
    return (True,
            "%d calibration samples is far past the %d saturation point: beyond "
            "%d the measured delta is under 0.1%%, so roughly %d samples of "
            "build time are being spent for nothing. Cut it to %d and put the "
            "effort into making them more representative instead (rule 2)."
            % (n, SATURATION_SAMPLES, SATURATION_SAMPLES, n - MAX_USEFUL_SAMPLES,
               MAX_USEFUL_SAMPLES))


# --------------------------------------------------------------------------
# gathering the data -- the half that decides whether INT8 is any good
# --------------------------------------------------------------------------

def collect_calibration_arrays(adapter, graph_key, scenarios, capture_fn,
                               max_samples=MAX_USEFUL_SAMPLES,
                               min_samples=MIN_SAMPLES):
    """Gather per-input-tensor calibration stacks for one graph.

    The samples come from the adapter's own scenarios, which is the whole point:
    rule 1 says calibration data must be in-domain, and the adapter is the only
    thing that knows what this network's domain looks like.

    ``capture_fn`` returns the inputs of ONE call of the graph. For an iterative
    graph it may instead return a leading-axis stack of several calls -- shape
    ``(M, *declared)`` rather than ``declared`` -- which is how rule 3 is
    satisfied: hand back every step of the loop from one scenario rather than
    only the first.

    A graph with a dynamic axis (a :class:`..spec.GraphSpec` carrying a
    :class:`..spec.ShapeProfile`) may be captured at any size that axis allows,
    but every sample must be captured at the *same* size: TensorRT calibrates
    against one profile shape, and the shape worth calibrating at is the
    profile's ``opt``, which is what its tactics were tuned for.

    Args:
        adapter: a :class:`..adapter.ModelAdapter`; only ``graphs()`` is used.
        graph_key: which :class:`..spec.GraphSpec` to collect for.
        scenarios: iterable of the adapter's scenario objects. Iterated lazily
            and abandoned once ``max_samples`` is reached, so an expensive
            generator is not drained for samples that would be discarded.
        capture_fn: ``scenario -> {input_name: ndarray}`` for one call (or a
            stack of calls) of this graph.
        max_samples: stop here. Defaults to :data:`MAX_USEFUL_SAMPLES`, where
            quality saturates.
        min_samples: refuse to return fewer than this.

    Returns:
        Dict mapping input tensor name -> contiguous float32 ``(M, *shape)``
        array, with the same ``M`` for every input, ready for
        :func:`make_entropy_calibrator` or ``np.savez`` for modelopt.

    Raises:
        KeyError: no graph named ``graph_key``, listing the ones there are.
        ValueError: a capture missing a declared input, carrying an undeclared
            one, disagreeing with the GraphSpec's declared shape, ragged across
            inputs, or captured at two different sizes on a dynamic axis; or
            fewer than ``min_samples`` collected. None of these can be papered
            over: a wrong shape means the wrong tensor was captured, and a thin
            set means the scales are fitted on noise.
    """
    spec = _graph_spec(adapter, graph_key)
    if int(min_samples) > int(max_samples):
        raise ValueError("min_samples (%d) exceeds max_samples (%d)"
                         % (min_samples, max_samples))
    declared = dict((name, tuple(int(d) for d in shape))
                    for name, shape in spec.inputs.items())
    if not declared:
        raise ValueError("GraphSpec %r declares no inputs; nothing to "
                         "calibrate" % graph_key)

    buckets = dict((name, []) for name in declared)
    pinned = {}
    total, index = 0, -1
    for index, scenario in enumerate(scenarios):
        rows = _normalize_capture(capture_fn(scenario), declared, graph_key,
                                  index)
        for name, arr in rows.items():
            _pin_shape(pinned, name, arr, graph_key, index)
            buckets[name].append(arr)
        total += next(iter(rows.values())).shape[0]
        if total >= max_samples:
            break

    if total < int(min_samples):
        _, advice = sample_count_advice(total)
        raise ValueError(
            "collected only %d calibration sample(s) for graph %r from %d "
            "scenario(s): %s Widen the scenario set, or -- for an iterative "
            "graph -- return every step of the loop from capture_fn instead of "
            "only the first (rule 3)."
            % (total, graph_key, index + 1, advice))

    out = {}
    for name, chunks in buckets.items():
        stack = np.concatenate(chunks, axis=0)[:int(max_samples)]
        out[name] = np.ascontiguousarray(stack, np.float32)
    return out


def _graph_spec(adapter, graph_key):
    """The adapter's GraphSpec for ``graph_key``, or a KeyError listing keys.

    The spec is validated on the way out, because every shape comparison below
    reads it as ground truth: an axis left at -1 with no ShapeProfile would
    otherwise match any capture at all, and quietly accept the wrong tensor.
    """
    graphs = list(adapter.graphs())
    for spec in graphs:
        if spec.key == graph_key:
            spec.validate()
            return spec
    raise KeyError("adapter %r declares no graph %r; available: %s"
                   % (getattr(adapter, "name", adapter), graph_key,
                      ", ".join(sorted(g.key for g in graphs)) or "none"))


def _normalize_capture(captured, declared, graph_key, index):
    """Turn one capture into ``{name: (M, *shape)}``, or raise saying why not.

    A capture is checked against the GraphSpec rather than trusted, because the
    failure it prevents is silent: an array of the wrong shape still calibrates,
    it just fits the range of a tensor the engine never sees.
    """
    if not hasattr(captured, "items"):
        raise ValueError(
            "capture_fn returned %r for scenario %d of graph %r; it must return "
            "a mapping of input tensor name -> ndarray"
            % (type(captured).__name__, index, graph_key))
    missing = sorted(set(declared) - set(captured))
    if missing:
        raise ValueError(
            "capture_fn omitted input(s) %s for scenario %d of graph %r. Every "
            "declared input needs its own calibration data -- TensorRT fits a "
            "range per tensor and has none to fall back on."
            % (", ".join(repr(m) for m in missing), index, graph_key))
    unknown = sorted(set(captured) - set(declared))
    if unknown:
        raise ValueError(
            "capture_fn returned undeclared input(s) %s for scenario %d of "
            "graph %r; the GraphSpec declares %s"
            % (", ".join(repr(u) for u in unknown), index, graph_key,
               ", ".join(sorted(declared))))

    rows = {}
    for name, shape in sorted(declared.items()):
        arr = np.ascontiguousarray(captured[name], np.float32)
        if _fits(arr.shape, shape):
            rows[name] = arr.reshape((1,) + arr.shape)
        elif _fits(arr.shape[1:], shape):
            rows[name] = arr
        else:
            raise ValueError(
                "capture_fn gave input %r of graph %r shape %s for scenario "
                "%d, but the GraphSpec declares %s. A capture is either exactly "
                "that shape (one sample) or a leading-axis stack of them, "
                "(M, %s)%s."
                % (name, graph_key, arr.shape, index, shape,
                   ", ".join(str(d) for d in shape),
                   "" if all(int(d) > 0 for d in shape)
                   else "; an axis declared <= 0 is dynamic and accepts any "
                        "size its ShapeProfile allows"))
    counts = sorted(set(a.shape[0] for a in rows.values()))
    if len(counts) > 1:
        raise ValueError(
            "capture_fn returned a ragged capture for scenario %d of graph %r: "
            "%s. Row i of every input must belong to the same call, because "
            "get_batch() hands TensorRT one row of each per batch."
            % (index, graph_key,
               ", ".join("%s=%d" % (n, a.shape[0])
                         for n, a in sorted(rows.items()))))
    return rows


def _fits(shape, declared):
    """True when a concrete shape satisfies a declared one.

    A declared axis of <= 0 is dynamic -- the GraphSpec carries a ShapeProfile
    for it -- and accepts any concrete size; every other axis must match
    exactly. Rank always must.
    """
    if len(shape) != len(declared):
        return False
    return all(int(d) <= 0 or int(s) == int(d)
               for s, d in zip(shape, declared))


def _pin_shape(pinned, name, arr, graph_key, index):
    """Hold every sample of one input to the same concrete shape.

    Only reachable for a dynamic input, and it is a hard failure there rather
    than a resize: TensorRT calibrates against ONE optimization-profile shape,
    so a set captured at several sizes fits the ranges of a graph the engine
    will not be running.
    """
    shape = tuple(arr.shape[1:])
    first = pinned.setdefault(name, shape)
    if shape != first:
        raise ValueError(
            "input %r of graph %r was captured at shape %s in scenario %d but "
            "at %s earlier. A dynamic axis may be any size the ShapeProfile "
            "allows, but the calibration set must be captured at ONE of them -- "
            "TensorRT fits its ranges against a single profile shape. Capture "
            "at the profile's opt shape."
            % (name, graph_key, shape, index, first))


# --------------------------------------------------------------------------
# route A -- the entropy calibrator, TensorRT <= 10
# --------------------------------------------------------------------------

def make_entropy_calibrator(input_arrays, cache_path, batch_size=1,
                            trt_module=None, cuda_module=None,
                            allow_undersized=False):
    """Build an ``IInt8EntropyCalibrator2`` over a mapping of input stacks.

    TensorRT drives calibration by calling ``get_batch(names)`` repeatedly with
    the input names it wants, and the two things that make this hard are both
    handled here: the returned pointers must be **device** addresses **in the
    order asked** (not the order the mapping happens to be in), and the method
    must return None -- not an empty list -- once the data is exhausted.

    Nothing about this is network-specific. Any number of inputs, any shapes,
    any names; the graph's own :class:`..spec.GraphSpec` decides all three and
    :func:`collect_calibration_arrays` produces the stacks.

    Args:
        input_arrays: mapping input-tensor-name -> ``(M, *shape)`` float32
            stack, every input with the same ``M``.
        cache_path: calibration cache to read at the start and write at the end.
            Reading it is what makes a rebuild of the same graph instant.
        batch_size: calibration batch size; 1 for the static single-sample
            graphs this pipeline exports.
        trt_module: imported ``tensorrt``, or None to import lazily. Passing it
            is what lets the route check be tested where TensorRT 11 is all
            there is.
        cuda_module: imported ``pycuda.driver``, or None to import lazily.
        allow_undersized: proceed with fewer than :data:`MIN_SAMPLES` samples.
            Off by default -- an undersized calibration produces an engine that
            builds, runs, and is quietly wrong.

    Returns:
        A TensorRT calibrator instance. Hand it to
        :func:`..engine.build.build_engine` as ``calibrator=``, which assigns it
        to ``config.int8_calibrator`` and holds the CUDA primary context the
        device buffers below need.

    Raises:
        RuntimeError: on a strongly-typed TensorRT (>= 11), where no calibrator
            class exists and the Q/DQ route is the only one -- the message
            carries :func:`qdq_instructions`.
        ValueError: on an empty mapping, ragged sample counts, fewer samples
            than one batch, or an undersized set without ``allow_undersized``.
    """
    trt = trt_module if trt_module is not None else _import_trt()
    if prec.is_strongly_typed(trt) or not hasattr(trt, "IInt8EntropyCalibrator2"):
        raise RuntimeError(
            "TensorRT %s has no IInt8EntropyCalibrator2: every calibrator class "
            "was removed when weak typing was, so there is nothing for this "
            "factory to build. INT8 here goes through the Q/DQ route instead -- "
            "quantize the ONNX, then build the quantized file.\n\n%s"
            % (getattr(trt, "__version__", "(unknown version)"),
               qdq_instructions("int8")))

    arrays = _validated_arrays(input_arrays)
    counts = sorted(set(a.shape[0] for a in arrays.values()))
    if len(counts) > 1:
        raise ValueError(
            "calibration inputs have different sample counts (%s). Row i of "
            "every input must be the same sample; use "
            "collect_calibration_arrays, which guarantees that."
            % ", ".join("%s=%d" % (n, a.shape[0])
                        for n, a in sorted(arrays.items())))
    n = counts[0]
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1, got %r" % (batch_size,))
    if n < batch_size:
        raise ValueError("%d calibration sample(s) cannot fill one batch of %d"
                         % (n, batch_size))
    ok, advice = sample_count_advice(n)
    if not ok and not allow_undersized:
        raise ValueError(
            "%s Pass allow_undersized=True to calibrate on this set anyway -- "
            "which is defensible for a smoke test and never for an engine that "
            "flies." % advice)

    cuda = cuda_module if cuda_module is not None else _import_pycuda()
    cache = Path(cache_path)

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """Stream calibration batches to TensorRT, in the order it asks."""

        def __init__(self):
            super().__init__()
            self._pos = 0
            self._device = dict(
                (name, cuda.mem_alloc(arr[0].nbytes * batch_size))
                for name, arr in arrays.items())

        def get_batch_size(self):
            """The batch size every ``get_batch`` call will deliver."""
            return batch_size

        def get_batch(self, names, *_unused):
            """Device pointers for ``names``, or None once exhausted.

            ``names`` is TensorRT's order, not the mapping's, and the returned
            list is positional -- returning them in the mapping's order feeds
            each tensor another tensor's data and produces plausible, wrong
            scales.
            """
            if self._pos + batch_size > n:
                return None
            pointers = []
            for name in names:
                if name not in self._device:
                    raise KeyError(
                        "TensorRT asked for calibration data for input %r, "
                        "which is not in the calibration set (%s). The stacks "
                        "were collected for a different graph, or an input was "
                        "renamed after export."
                        % (name, ", ".join(sorted(self._device))))
                chunk = arrays[name][self._pos:self._pos + batch_size]
                cuda.memcpy_htod(self._device[name],
                                 np.ascontiguousarray(chunk, np.float32))
                pointers.append(int(self._device[name]))
            self._pos += batch_size
            return pointers

        def read_calibration_cache(self):
            """The cached scales, or None to calibrate from the data."""
            return cache.read_bytes() if cache.is_file() else None

        def write_calibration_cache(self, cache_bytes):
            """Persist the fitted scales so a rebuild skips calibration."""
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(cache_bytes)

    return _EntropyCalibrator()


def _validated_arrays(input_arrays):
    """Coerce the calibration mapping to contiguous float32 stacks."""
    if not hasattr(input_arrays, "items"):
        raise ValueError("input_arrays must be a mapping of input tensor name "
                         "-> (M, *shape) array, got %r"
                         % type(input_arrays).__name__)
    if not input_arrays:
        raise ValueError("input_arrays is empty; a calibrator with no data "
                         "would leave every activation range unset")
    arrays = {}
    for name, values in input_arrays.items():
        arr = np.ascontiguousarray(values, np.float32)
        if arr.ndim < 2:
            raise ValueError(
                "calibration input %r has shape %s; it must be a stack of "
                "samples, (M, *shape), even when the sample is 1-D"
                % (name, arr.shape))
        if arr.shape[0] < 1:
            raise ValueError("calibration input %r holds no samples" % name)
        arrays[name] = arr
    return arrays
