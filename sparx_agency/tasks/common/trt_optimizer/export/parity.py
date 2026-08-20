"""Tiers (a) and (b): prove the export before anything is built.

Two FP32 comparisons, both on CPU, both deterministic, both free:

**(a) wrapper vs original module.** Blesses the deliberate surgery -- a baked
positional embedding, a constant mask lifted out of the graph, an ``-inf`` mask
replaced by a large finite value, an int64 input narrowed to float32. Each of
those is a change to the computation, and each must be shown to be a change
without a difference.

**(b) ONNX vs wrapper.** Blesses the export itself -- op decomposition, constant
folding, opset behaviour, and whatever ``onnxslim`` removed.

Skipping them does not make the on-target precision gate prove more; it makes
its failures **unattributable**. When tier (c) fails and (a)/(b) never ran, an
export bug and an FP16 numerics problem look identical.

``onnxruntime`` here is CPU-only, and that is a feature rather than a limitation:
a CUDA or TensorRT execution provider would let the "reference" diverge on the
very silicon under test.

Tolerances are **relative L2** in float64, ``||a - b|| / (||b|| + eps)``, and are
compared with ``<=`` so a configured tolerance means what it says. ``max_abs`` is
reported and never gated -- one outlier element is not a decision.

Lifted and generalised from ``tasks/planning/vlas/navdp/trt/export/validate_parity.py``
and its FlowNav counterpart; FlowNav's ``<=`` comparison and config-read
tolerances are the ones copied, NavDP's hardcoded module constants are not.

Not importable on aarch64 in practice: ``onnxruntime``'s CPU-feature detection
SIGABRTs there and a native abort cannot be caught. Validate on x86; on a Jetson
tier (c) is the accuracy proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

#: Relative-L2 ceiling applied when a graph names none of its own. Generous
#: enough for a deep FP32 transformer's op-reordering noise, tight enough that a
#: real export bug (a dropped residual, a wrong mask) cannot hide under it.
DEFAULT_TOLERANCE = 2e-3


@dataclass
class ParityResult:
    """One comparison's outcome.

    Args:
        key: engine key.
        tier: ``"a"`` (wrapper vs module) or ``"b"`` (ONNX vs wrapper).
        output: output tensor name.
        rel_l2: relative L2 in float64.
        max_abs: largest absolute elementwise difference. Diagnostic only.
        tolerance: the ceiling ``rel_l2`` was compared against.
    """

    key: str
    tier: str
    output: str
    rel_l2: float
    max_abs: float
    tolerance: float

    @property
    def ok(self):
        """True when ``rel_l2`` is within tolerance. Compared with ``<=``."""
        return self.rel_l2 <= self.tolerance

    def line(self):
        """One aligned report row."""
        return ("  [%s] tier %s %-28s rel_l2 %10.3e  max_abs %10.3e  "
                "(tol %8.1e)  %s"
                % ("ok" if self.ok else "FAIL", self.tier, "%s/%s"
                   % (self.key, self.output), self.rel_l2, self.max_abs,
                   self.tolerance, ""))


@dataclass
class ParityReport:
    """Every comparison from one run, and whether the set passed."""

    results: List[ParityResult] = field(default_factory=list)

    @property
    def ok(self):
        """True when every comparison is within tolerance."""
        return all(r.ok for r in self.results)

    def failures(self):
        """The results that exceeded their tolerance."""
        return [r for r in self.results if not r.ok]

    def render(self):
        """The whole run as text, one line per comparison."""
        return "\n".join(r.line() for r in self.results)


def rel_l2(candidate, reference):
    """Relative L2 distance in float64.

    Args:
        candidate: array under test.
        reference: array to measure against.

    Returns:
        float: ``||candidate - reference|| / (||reference|| + 1e-12)``.
    """
    import numpy as np

    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def max_abs(candidate, reference):
    """Largest absolute elementwise difference. Reported, never gated."""
    import numpy as np

    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _to_numpy(value):
    """Detach any torch tensor to a float32 numpy array; pass arrays through."""
    import numpy as np

    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _as_tuple(outputs):
    """Normalise a module's return value to a tuple of tensors."""
    if isinstance(outputs, (list, tuple)):
        return tuple(outputs)
    if isinstance(outputs, dict):
        return tuple(outputs.values())
    return (outputs,)


def session(onnx_path):
    """Build a CPU-only onnxruntime session for ``onnx_path``.

    Raises:
        RuntimeError: when the CPU execution provider is unavailable, which
            would otherwise silently move the reference onto the GPU under test.
    """
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if "CPUExecutionProvider" not in providers:
        raise RuntimeError(
            "parity needs CPUExecutionProvider; onnxruntime offers %s. A GPU "
            "provider would let the reference diverge on the silicon under test."
            % (providers,))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(str(onnx_path), options,
                                providers=["CPUExecutionProvider"])


def compare_module(spec, wrapper, reference_fn, inputs, tolerance=DEFAULT_TOLERANCE):
    """Tier (a): the export wrapper against the unpatched original.

    Args:
        spec: the :class:`..spec.GraphSpec`.
        wrapper: the export wrapper module.
        reference_fn: callable taking the same positional inputs and returning
            the unpatched result.
        inputs: tuple of torch tensors, FP32, on CPU.
        tolerance: relative-L2 ceiling.

    Returns:
        List[ParityResult], one per output.
    """
    import torch

    with torch.no_grad():
        got = _as_tuple(wrapper(*inputs))
        want = _as_tuple(reference_fn(*inputs))
    return _pair_up(spec, "a", got, want, tolerance)


def compare_onnx(spec, wrapper, onnx_path, inputs, tolerance=DEFAULT_TOLERANCE):
    """Tier (b): the exported ONNX against the wrapper that produced it.

    Args:
        spec: the :class:`..spec.GraphSpec`; supplies the input names.
        wrapper: the export wrapper module, FP32 on CPU.
        onnx_path: the exported graph.
        inputs: tuple of torch tensors in the spec's declared order.
        tolerance: relative-L2 ceiling.

    Returns:
        List[ParityResult], one per output.
    """
    import torch

    with torch.no_grad():
        want = _as_tuple(wrapper(*inputs))
    feed = dict(zip(spec.input_names(), (_to_numpy(t) for t in inputs)))
    got = tuple(session(onnx_path).run(None, feed))
    return _pair_up(spec, "b", got, want, tolerance)


def _pair_up(spec, tier, got, want, tolerance):
    """Zip two output tuples into :class:`ParityResult` rows.

    Raises:
        ValueError: on an output-count mismatch, which is an export bug rather
            than a numeric one and must not be reported as a tolerance failure.
    """
    if len(got) != len(want):
        raise ValueError(
            "graph %r tier %s produced %d output(s) against %d in the "
            "reference; that is a structural mismatch, not a tolerance one"
            % (spec.key, tier, len(got), len(want)))
    names = list(spec.outputs) or ["out%d" % i for i in range(len(got))]
    rows = []
    for name, a, b in zip(names, got, want):
        a, b = _to_numpy(a), _to_numpy(b)
        rows.append(ParityResult(key=spec.key, tier=tier, output=name,
                                 rel_l2=rel_l2(a, b), max_abs=max_abs(a, b),
                                 tolerance=float(tolerance)))
    return rows


def validate(specs, wrappers, onnx_dir, example_inputs, tolerances=None,
             unpatched=None):
    """Run every applicable tier over every graph and return the report.

    Every comparison runs even after one fails, so a single invocation shows
    the whole picture rather than the first problem.

    Args:
        specs: the :class:`..spec.GraphSpec` objects.
        wrappers: engine key -> export wrapper module (FP32, CPU).
        onnx_dir: directory holding ``<key>.onnx``.
        example_inputs: callable ``spec -> tuple`` of torch CPU tensors.
        tolerances: optional ``{key: rel_l2_ceiling}``; :data:`DEFAULT_TOLERANCE`
            elsewhere. Read from config rather than hardcoded, so a build policy
            can tighten a graph without editing this module.
        unpatched: optional ``{key: callable}`` enabling tier (a). A key with no
            entry runs tier (b) only, which is reported rather than implied.

    Returns:
        A :class:`ParityReport`.
    """
    tolerances = dict(tolerances or {})
    unpatched = dict(unpatched or {})
    report = ParityReport()
    for spec in specs:
        tolerance = float(tolerances.get(spec.key, DEFAULT_TOLERANCE))
        inputs = example_inputs(spec)
        wrapper = wrappers[spec.key]
        reference_fn = unpatched.get(spec.key)
        if reference_fn is not None:
            report.results.extend(
                compare_module(spec, wrapper, reference_fn, inputs, tolerance))
        report.results.extend(
            compare_onnx(spec, wrapper, Path(onnx_dir) / (spec.key + ".onnx"),
                         inputs, tolerance))
    return report


def enforce(report):
    """Print every line, then raise if any comparison failed.

    Args:
        report: a :class:`ParityReport`.

    Returns:
        The same report, when everything passed.

    Raises:
        RuntimeError: naming every failure. Printing first matters: one run
            should show all the problems, not just the first.
    """
    print(report.render())
    if report.ok:
        return report
    raise RuntimeError(
        "parity failed for %s; the export does not reproduce the reference, so "
        "nothing downstream may be believed"
        % ", ".join("%s/%s tier %s (rel_l2 %.3e > %.1e)"
                    % (r.key, r.output, r.tier, r.rel_l2, r.tolerance)
                    for r in report.failures()))
