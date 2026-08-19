"""The vocabulary of a TensorRT optimization plan: components, graphs, cadence.

Every decision this package makes is expressed in these types, and every
network-specific adapter speaks them. The distinction that carries the most
weight is **component vs graph**:

  * A :class:`Component` is a named part of the model that appears in the
    *latency inventory*. Everything the model spends time in is a component,
    including parts that will never become a TensorRT engine.
  * A :class:`GraphSpec` is a component that has been judged worth exporting,
    and describes the static IO contract of the engine it becomes.

The inventory comes first and the graphs are derived from it. That ordering is
the whole point: a component is converted because it was *measured* to dominate
the per-decision budget, never because it happened to be easy to export.

:class:`Cadence` is the second load-bearing idea. A component that runs once per
episode contributes ~nothing to steady-state frame rate no matter how slow it
is, so converting it buys nothing and costs a maintenance burden and a numerical
risk. ``calls_per_decision`` turns cadence into the arithmetic that
:mod:`..trt_optimizer.amdahl` uses to bound the achievable speedup.

Pure standard library plus dataclasses; importable anywhere, no torch, no
TensorRT. Python-3.8-compatible syntax so it can be imported on a Jetson's
system interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class Cadence(object):
    """How often a component runs, relative to one control decision.

    These are strings rather than an ``enum.Enum`` so that a plan round-trips
    through JSON without a custom encoder, and so an adapter can be written
    without importing this module.
    """

    ONCE_PER_PROCESS = "once_per_process"
    """Runs at load time (weight loading, warmup). Never on the hot path."""

    ONCE_PER_EPISODE = "once_per_episode"
    """Runs once per mission/instruction (a genuinely cached text encoder)."""

    PER_PLAN = "per_plan"
    """Runs on the slow reasoning cadence -- a System-2 replan every N frames."""

    PER_FRAME = "per_frame"
    """Runs once for every control decision. The hot path."""

    PER_STEP = "per_step"
    """Runs several times *inside* one decision (a diffusion/flow denoise loop)."""

    ON_DEMAND = "on_demand"
    """Runs only when a condition fires; amortized share must be measured."""

    ALL = (ONCE_PER_PROCESS, ONCE_PER_EPISODE, PER_PLAN, PER_FRAME, PER_STEP,
           ON_DEMAND)

    #: Cadences that cannot repay a TensorRT conversion on their own. A
    #: component at one of these is excluded unless an explicit override says
    #: otherwise -- see :mod:`..trt_optimizer.decide`.
    COLD = (ONCE_PER_PROCESS, ONCE_PER_EPISODE)


class Exportability(object):
    """How hostile a component is to ``torch.onnx.export``."""

    CLEAN = "clean"
    """Static shapes, no data-dependent control flow. Exports as-is."""

    NEEDS_PATCH = "needs_patch"
    """Exports only after a known monkey-patch (SDPA math, baked pos-embed)."""

    HOSTILE = "hostile"
    """Autoregressive loop, KV cache, sampling, or a custom CUDA op. Not ONNX."""

    ALL = (CLEAN, NEEDS_PATCH, HOSTILE)


@dataclass
class Component:
    """One named part of the model in the latency inventory.

    Args:
        name: dotted module path (``model.vision_tower``) or a logical name.
        params: parameter count under this component.
        cadence: one of :class:`Cadence`.
        calls_per_decision: how many times it executes per control decision.
            A 20-step denoiser is ``20.0``; a System-2 backbone that replans
            every 8 frames is ``0.125``.
        exportability: one of :class:`Exportability`.
        reason: why ``exportability`` is what it is -- the specific blocker.
        latency_ms: measured mean wall time for ONE call, filled by profiling.
        dtype: parameter dtype as a string (``float32``/``bfloat16``).
    """

    name: str
    params: int = 0
    cadence: str = Cadence.PER_FRAME
    calls_per_decision: float = 1.0
    exportability: str = Exportability.CLEAN
    reason: str = ""
    latency_ms: Optional[float] = None
    dtype: str = "float32"

    @property
    def decision_ms(self):
        """Total wall time this component contributes to ONE decision.

        Returns:
            ``latency_ms * calls_per_decision``, or None if unmeasured.
        """
        if self.latency_ms is None:
            return None
        return self.latency_ms * self.calls_per_decision

    def weight_bytes(self, bytes_per_param=None):
        """Parameter storage in bytes at ``dtype`` (or an override width).

        Args:
            bytes_per_param: override the width implied by ``dtype``.

        Returns:
            Bytes of parameter storage.

        Raises:
            ValueError: on a ``dtype`` this does not know. Guessing 4 bytes for
                an unrecognised dtype would silently understate a bf16 model's
                footprint by half and let a memory budget pass that should have
                failed.
        """
        if bytes_per_param is None:
            key = str(self.dtype).lower().replace("torch.", "")
            if key not in _DTYPE_BYTES:
                raise ValueError(
                    "unknown dtype %r for component %r; known: %s"
                    % (self.dtype, self.name, ", ".join(sorted(_DTYPE_BYTES))))
            bytes_per_param = _DTYPE_BYTES[key]
        return int(self.params * bytes_per_param)


_DTYPE_BYTES = {
    "float64": 8, "float32": 4, "float": 4, "bfloat16": 2, "float16": 2,
    "half": 2, "int8": 1, "uint8": 1, "fp8": 1, "int4": 0.5, "nvfp4": 0.5,
}


@dataclass
class ShapeProfile:
    """The min / opt / max shapes of one input whose size is not fixed.

    TensorRT builds a dynamic engine against an *optimization profile*: it tunes
    tactics for ``opt`` and guarantees correctness between ``min`` and ``max``.
    Both halves of that sentence matter -- a profile whose ``opt`` is far from
    the size you actually run is slower than a static engine, and a range wider
    than you need costs tactic quality across the whole range.

    Prefer a static graph. A dynamic dimension is the right answer only when the
    input genuinely varies at run time (a detector fed whatever resolution the
    camera produced, a batch sized by how many candidates survived a filter);
    when the size is known and fixed, declaring it static gives TensorRT more to
    work with and removes a profile-switch cost from the tail latency.

    Args:
        min: smallest shape the engine must accept.
        opt: the shape tactics are tuned for. Make this the common case.
        max: largest shape the engine must accept; it also sizes the buffers.
    """

    min: Tuple[int, ...]
    opt: Tuple[int, ...]
    max: Tuple[int, ...]

    def validate(self, name):
        """Raise ValueError unless min <= opt <= max elementwise, same rank.

        Args:
            name: input tensor name, for the message.
        """
        ranks = {len(self.min), len(self.opt), len(self.max)}
        if len(ranks) != 1:
            raise ValueError(
                "ShapeProfile for %r has mismatched ranks: min=%r opt=%r max=%r"
                % (name, self.min, self.opt, self.max))
        for i, (lo, mid, hi) in enumerate(zip(self.min, self.opt, self.max)):
            if not (1 <= int(lo) <= int(mid) <= int(hi)):
                raise ValueError(
                    "ShapeProfile for %r is not ordered at axis %d: "
                    "min=%s opt=%s max=%s (need 1 <= min <= opt <= max)"
                    % (name, i, lo, mid, hi))


@dataclass
class GraphSpec:
    """The static IO contract of one exportable sub-graph.

    ``key`` is the single identifier that ties the whole pipeline together: it
    is the ONNX stem, the engine stem, the manifest key and the report row. The
    repo's existing NavDP/FlowNav trees use the same convention.

    Shapes are fully static on purpose. The shared runtime
    (``core.planning.vlas.common.trt.engine_runner.TRTEngineRunner``) rejects
    any engine with a dynamic dimension, because a profile switch costs a
    one-time ``enqueue`` penalty that lands directly in control-loop tail
    latency.

    Args:
        key: engine key, e.g. ``"navdp_encoder"``.
        inputs: ordered mapping of input tensor name -> static shape.
        outputs: ordered output tensor names.
        component: name of the :class:`Component` this graph came from.
        cadence: inherited from the component.
        calls_per_decision: inherited from the component.
        precision_sensitive: True for deep residual stacks (ViT trunks) that
            drift when the whole graph is forced to FP16 with no per-layer
            fallback. Such graphs are built FP32 on a strongly-typed TensorRT.
        opset: ONNX opset. 17 is the floor -- it is what emits a single
            ``LayerNormalization`` node instead of a decomposed
            ReduceMean/Sub/Pow/Div chain, which is the fix for most transformer
            FP16 accuracy regressions.
    """

    key: str
    inputs: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    component: str = ""
    cadence: str = Cadence.PER_FRAME
    calls_per_decision: float = 1.0
    precision_sensitive: bool = False
    opset: int = 17
    notes: str = ""
    profiles: Dict[str, ShapeProfile] = field(default_factory=dict)

    def input_names(self):
        """Ordered input tensor names."""
        return list(self.inputs.keys())

    @property
    def is_dynamic(self):
        """True when any input declares a profile (i.e. a non-fixed axis)."""
        return bool(self.profiles)

    def dynamic_axes(self):
        """``{input_name: [axis, ...]}`` for the axes that vary, for the exporter.

        An axis varies when its ``min`` and ``max`` differ in that input's
        :class:`ShapeProfile`. Axes that happen to be equal across the profile
        are left static, so the graph declares only the freedom it needs.
        """
        axes = {}
        for name, profile in self.profiles.items():
            varying = [i for i, (lo, hi) in
                       enumerate(zip(profile.min, profile.max)) if int(lo) != int(hi)]
            if varying:
                axes[name] = varying
        return axes

    def volume(self, name):
        """Element count of one *input* tensor, at its ``opt`` shape if dynamic.

        Raises:
            KeyError: if ``name`` is not a declared input. Outputs have no
                declared shape here, so they cannot be measured.
        """
        profile = self.profiles.get(name)
        shape = profile.opt if profile is not None else self.inputs[name]
        n = 1
        for d in shape:
            n *= int(d)
        return n

    def validate(self):
        """Raise ValueError unless every shape is static and positive.

        Raises:
            ValueError: on an empty IO list or any non-positive dimension.
        """
        if not self.inputs:
            raise ValueError("GraphSpec %r has no inputs" % self.key)
        if not self.outputs:
            raise ValueError("GraphSpec %r has no outputs" % self.key)
        for name, profile in self.profiles.items():
            if name not in self.inputs:
                raise ValueError(
                    "GraphSpec %r declares a ShapeProfile for %r, which is not "
                    "one of its inputs (%s)"
                    % (self.key, name, ", ".join(self.input_names())))
            profile.validate(name)
        for name, shape in self.inputs.items():
            if not shape:
                raise ValueError("GraphSpec %r input %r has an empty shape"
                                 % (self.key, name))
            for axis, d in enumerate(shape):
                if int(d) > 0:
                    continue
                profile = self.profiles.get(name)
                if profile is None:
                    raise ValueError(
                        "GraphSpec %r input %r axis %d is %r but declares no "
                        "ShapeProfile. Either give it a fixed size, or add a "
                        "ShapeProfile(min, opt, max) so TensorRT can build an "
                        "optimization profile for it."
                        % (self.key, name, axis, d))
                if int(profile.min[axis]) == int(profile.max[axis]):
                    raise ValueError(
                        "GraphSpec %r input %r axis %d is marked dynamic but its "
                        "ShapeProfile pins it to %d; give it a fixed size instead."
                        % (self.key, name, axis, int(profile.min[axis])))


@dataclass
class Verdict:
    """What to do with one component, and why.

    ``action`` is one of :data:`ACTIONS`. ``why`` is written for a human reading
    the final report -- it must name the rule that fired, not restate the action.
    """

    component: str
    action: str
    why: str
    expected_speedup: float = 1.0
    confidence: str = "medium"


#: Every action :mod:`..trt_optimizer.decide` may return.
ACTIONS = (
    "trt_fp16",        # export -> FP16 engine (the default for a hot graph)
    "trt_fp32",        # export -> FP32 engine (precision-sensitive trunk)
    "trt_int8",        # export -> Q/DQ INT8 engine (needs calibration)
    "llm_runtime",     # autoregressive: TensorRT-LLM / llama.cpp, never ONNX
    "leave_in_torch",  # measured share too small, or export too hostile
    "cache_output",    # cadence lets the output be reused across frames
    "reduce_calls",    # the lever is fewer calls (denoise steps, candidates)
)


@dataclass
class Plan:
    """The complete optimization plan for one model on one target.

    This is what the skill's Stage 3 produces and every later stage consumes.
    It serializes to JSON so a plan built on the workstation can be reviewed,
    edited by hand, and replayed on the target device.
    """

    model: str
    target_tag: str = "unknown"
    components: List[Component] = field(default_factory=list)
    graphs: List[GraphSpec] = field(default_factory=list)
    verdicts: List[Verdict] = field(default_factory=list)
    baseline_hz: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def component(self, name):
        """Look up one component by name, or None."""
        for c in self.components:
            if c.name == name:
                return c
        return None

    def graph(self, key):
        """Look up one graph by engine key, or None."""
        for g in self.graphs:
            if g.key == key:
                return g
        return None

    def decision_ms(self):
        """Summed measured wall time of one control decision, or None.

        Returns None unless every component has been profiled -- a partial sum
        would silently understate the denominator that every share and every
        Amdahl bound is computed against.
        """
        if not self.components:
            return None
        total = 0.0
        for c in self.components:
            ms = c.decision_ms
            if ms is None:
                return None
            total += ms
        return total
