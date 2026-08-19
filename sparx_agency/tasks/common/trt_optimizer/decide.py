"""Decide what to convert, what to leave alone, and why -- from measurements.

This is the module the whole toolkit exists for. Everything upstream of it
gathers evidence; everything downstream executes what it decides.

The one rule that outranks every other: **a component is converted because it
was measured to dominate the per-decision budget, never because it was easy to
export.** :func:`decide` therefore refuses to run on an unprofiled plan. That
refusal is deliberate -- the common and expensive failure mode in a TensorRT
project is converting the graph that was convenient rather than the graph that
was slow, and discovering after a week that the end-to-end rate barely moved
because that graph was 6% of the budget.

Cadence is the second gate, and it is checked before any timing. A component
that runs once per episode can be arbitrarily slow without touching the
steady-state frame rate; converting it buys nothing and costs a permanent
numerical risk and a maintenance burden. This is checked by cadence rather than
by a share threshold because a share threshold on something that runs once is
not meaningful in the first place.

What this module never does is guess a speedup it has not seen. Every
``expected_speedup`` it emits is a stated assumption carried into the report so
the final measurement can contradict it in public.
"""
from __future__ import annotations

from sparx_agency.tasks.common.trt_optimizer import amdahl
from sparx_agency.tasks.common.trt_optimizer.spec import (
    Cadence, Exportability, Verdict,
)

#: Markers in a component's exportability reason that mean "this is an
#: autoregressive language model", which routes to an LLM runtime rather than
#: to a refusal. ONNX export of a KV-cached sampling loop is not a hard problem
#: to be solved with more effort; it is the wrong tool.
_LLM_MARKERS = ("kv cache", "past_key_values", "dynamiccache", "generate(",
                "autoregressive", "sampling loop", "use_cache")

#: Assumed engine speedup when a graph is converted, used only to project a
#: gain before anything is built. Deliberately conservative: the repo's own
#: measured NavDP result was well above this, and an optimistic assumption here
#: would let a marginal component pass the worth-it gate on paper.
DEFAULT_ASSUMED_SPEEDUP = 3.0


def precision_ladder(target):
    """Precisions to try, in order, stopping at the first that passes the gate.

    Ordering is by *measured* payoff on the target's silicon, not by datasheet
    width. On GeForce Blackwell (sm_120) a 2048-cube GEMM measured on this
    machine gave FP16 75.5 TFLOP/s, FP8 82.8 (1.10x), INT8 149.5 (1.98x) and
    NVFP4 68.6 -- **NVFP4 was slower than FP16**. So INT8 is the only quantized
    format that reliably pays here, and FP8/NVFP4 are demoted to "benchmark
    before believing" rather than offered as an obvious next step.

    Args:
        target: a :class:`..trt_optimizer.target.Target`.

    Returns:
        List of precision names, cheapest-risk first, filtered to what this
        toolchain can actually produce.
    """
    supported = target.supported_precisions()
    preferred = ["fp16", "int8", "bf16", "fp8", "nvfp4"]
    ladder = [p for p in preferred if p in supported]
    # FP32 is always the last rung, never omitted. A reduced-precision graph can
    # be genuinely unreachable for a particular network -- the FP16 converter
    # fails outright on some graphs -- and an FP32 TensorRT engine still wins on
    # kernel fusion and on not being an eager Python loop. Dropping it would
    # turn "we could not get FP16" into "we shipped nothing".
    if "fp32" not in ladder:
        ladder.append("fp32")
    return ladder


def _is_llm(component):
    """True when the exportability reason describes autoregressive decoding."""
    reason = (component.reason or "").lower()
    return any(marker in reason for marker in _LLM_MARKERS)


def _cold(component, share_pct):
    """Verdict for a component whose cadence cannot repay a conversion."""
    return Verdict(
        component=component.name,
        action="cache_output" if component.cadence == Cadence.ONCE_PER_EPISODE
        else "leave_in_torch",
        why=("runs %s (%.3g calls per decision, %.1f%% of the budget); "
             "converting it cannot move the steady-state rate"
             % (component.cadence, component.calls_per_decision, share_pct)),
        expected_speedup=1.0,
        confidence="high",
    )


def _hostile(component):
    """Verdict for a component that cannot be exported to ONNX at all."""
    if _is_llm(component):
        return Verdict(
            component=component.name,
            action="llm_runtime",
            why=("not an ONNX graph: %s. Route it to a runtime that owns its own "
                 "KV cache and sampling loop (TensorRT-LLM PyTorch backend on a "
                 "discrete GPU, llama.cpp/GGUF on Orin) rather than trying to "
                 "trace it." % (component.reason or "autoregressive decoding")),
            expected_speedup=1.0,
            confidence="high",
        )
    return Verdict(
        component=component.name,
        action="leave_in_torch",
        why="cannot be exported: %s" % (component.reason or "hostile to tracing"),
        expected_speedup=1.0,
        confidence="high",
    )


def _hot(component, plan, target, share_pct, assumed_speedup):
    """Verdict for a component that passed cadence, export and share gates."""
    action = "trt_fp32" if _precision_sensitive(component, plan) else "trt_fp16"
    gain = share_pct / 100.0 * (1.0 - 1.0 / assumed_speedup)
    why = ("%.1f%% of the per-decision budget (%.2f ms x %.3g calls); at an "
           "assumed %.1fx engine speedup that is %.1f%% end to end"
           % (share_pct, component.latency_ms or 0.0,
              component.calls_per_decision, assumed_speedup, gain * 100.0))
    if action == "trt_fp32":
        why += (". Built FP32, not FP16: this graph is marked precision-sensitive "
                "and TensorRT %s is strongly typed, so a forced-FP16 graph has no "
                "per-layer FP32 fallback to rescue a deep residual stream"
                % (target.trt_version or "11+"))
    return Verdict(component=component.name, action=action, why=why,
                   expected_speedup=assumed_speedup, confidence="medium")


def _precision_sensitive(component, plan):
    """True when any graph derived from this component is precision-sensitive."""
    for g in plan.graphs:
        if g.component == component.name and g.precision_sensitive:
            return True
    return False


def _loop_lever(component, share_pct):
    """Extra verdict when the real lever is fewer calls, not a faster call.

    For a diffusion or flow-matching head the step count is the dominant term
    and it is free to change -- no rebuild, no numerics work. Attacking kernels
    before attacking the step count is the classic misallocation.
    """
    if component.cadence != Cadence.PER_STEP or component.calls_per_decision < 4:
        return None
    return Verdict(
        component=component.name,
        action="reduce_calls",
        why=("runs %.0f times per decision and is %.1f%% of the budget; halving "
             "the step count is a ~%.0f%% end-to-end win with no rebuild and no "
             "numerics risk. Attack the step count BEFORE the kernels, and "
             "truncate late/low-noise steps before early ones."
             % (component.calls_per_decision, share_pct, share_pct / 2.0)),
        expected_speedup=1.0,
        confidence="high",
    )


def _async_lever(component, share_pct):
    """Extra verdict when a slow component can be hidden behind a fast one."""
    if component.cadence != Cadence.PER_PLAN or share_pct < 20.0:
        return None
    return Verdict(
        component=component.name,
        action="cache_output",
        why=("runs on the slow planning cadence but still costs %.1f%% of the "
             "averaged budget; run it on its own thread and let the fast path "
             "consume the last completed output with an explicit staleness "
             "policy, so the control loop never blocks on it." % share_pct),
        expected_speedup=1.0,
        confidence="medium",
    )


def decide(plan, target, min_share=0.05, min_end_to_end_gain=0.02,
           assumed_speedup=DEFAULT_ASSUMED_SPEEDUP):
    """Produce a :class:`Verdict` for every component in a profiled plan.

    Args:
        plan: a :class:`..trt_optimizer.spec.Plan` whose components have all
            been profiled (``latency_ms`` set).
        target: the :class:`..trt_optimizer.target.Target` that will build.
        min_share: a component below this fraction of the per-decision budget is
            not worth the conversion, however easy it looks.
        min_end_to_end_gain: minimum projected end-to-end gain to bother.
        assumed_speedup: the engine speedup assumed when projecting a gain.

    Returns:
        List of :class:`Verdict`, in the plan's component order, with any extra
        lever verdicts (``reduce_calls``, ``cache_output``) appended after the
        component's primary verdict.

    Raises:
        ValueError: if the plan has not been profiled. Deciding what to convert
            from parameter counts instead of measurements is the mistake this
            package exists to prevent, so it is refused rather than approximated.
    """
    total = plan.decision_ms()
    if total is None:
        unprofiled = [c.name for c in plan.components if c.latency_ms is None]
        raise ValueError(
            "Cannot decide from an unprofiled plan: %s %s no measured "
            "latency. Run the baseline profile (Stage 2) first -- shares "
            "derived from parameter counts are not a substitute."
            % (", ".join(unprofiled) or "the plan",
               "have" if len(unprofiled) != 1 else "has"))
    if total <= 0:
        raise ValueError("Plan %r has a zero-length decision; nothing to "
                         "apportion." % plan.model)

    verdicts = []
    for component in plan.components:
        share_pct = amdahl.share(component, plan) * 100.0
        verdicts.append(_verdict_for(component, plan, target, share_pct,
                                     min_share, min_end_to_end_gain,
                                     assumed_speedup))
        for extra in (_loop_lever(component, share_pct),
                      _async_lever(component, share_pct)):
            if extra is not None and extra.action != verdicts[-1].action:
                verdicts.append(extra)
    return verdicts


def _verdict_for(component, plan, target, share_pct, min_share,
                 min_end_to_end_gain, assumed_speedup):
    """The single primary verdict for one component (rules in priority order)."""
    if component.cadence in Cadence.COLD:
        return _cold(component, share_pct)
    if component.exportability == Exportability.HOSTILE:
        return _hostile(component)
    worth, reason = amdahl.worth_converting(
        component, plan, min_share=min_share,
        min_end_to_end_gain=min_end_to_end_gain,
        assumed_speedup=assumed_speedup)
    if not worth:
        return Verdict(component=component.name, action="leave_in_torch",
                       why=reason, expected_speedup=1.0, confidence="high")
    return _hot(component, plan, target, share_pct, assumed_speedup)


def convertible(verdicts):
    """Component names whose verdict is an actual engine build."""
    return [v.component for v in verdicts
            if v.action in ("trt_fp16", "trt_fp32", "trt_int8")]


def ceiling(plan, verdicts, assumed_speedup=DEFAULT_ASSUMED_SPEEDUP):
    """Projected end-to-end speedup if every conversion lands, and its ceiling.

    Returns:
        ``(projected, ceiling)`` -- the speedup at ``assumed_speedup`` per
        converted component, and the Amdahl bound if they became free. Reading
        both together is the honest way to present a plan: the ceiling says
        whether the work can ever be worth it.
    """
    names = convertible(verdicts)
    speedups = {}
    for name in names:
        speedups[name] = assumed_speedup
    return (amdahl.speedup_with(plan, speedups),
            amdahl.max_speedup(plan, names))


def coverage(plan, verdicts):
    """Check that what was judged worth converting is actually exported.

    The plan reasons about *components* (what the profiler measured) while the
    adapter exports *graphs* (what becomes an engine), and nothing forces those
    two to line up. When they do not, the report reads strangely: components
    marked ``trt_fp16`` whose time never moves, because no engine was ever built
    for them.

    Both directions are worth knowing:

    * a component judged worth converting with no graph covering it -- the
      speedup projected in the plan cannot be realised;
    * a graph exported for a component judged not worth converting -- effort and
      numerical risk spent where the measurement said not to.

    Args:
        plan: the profiled :class:`..spec.Plan`.
        verdicts: the verdicts from :func:`decide`.

    Returns:
        ``(uncovered, unjustified)`` -- lists of component names.
    """
    covered = set()
    for graph in plan.graphs:
        if graph.component:
            covered.add(graph.component)
    wanted = set(convertible(verdicts))
    uncovered = sorted(wanted - covered)
    unjustified = sorted(covered - wanted)
    return uncovered, unjustified


def coverage_notes(plan, verdicts):
    """Human-readable warnings from :func:`coverage`, or an empty list."""
    uncovered, unjustified = coverage(plan, verdicts)
    notes = []
    if uncovered:
        notes.append(
            "%d component(s) were judged worth converting but no exported graph "
            "covers them (%s). Either add a GraphSpec whose `component` names "
            "them, or accept that the projected speedup will not appear."
            % (len(uncovered), ", ".join(uncovered)))
    if unjustified:
        notes.append(
            "%d exported graph(s) cover components the measurement said to leave "
            "alone (%s). That is effort and numerical risk spent where it does "
            "not pay -- unless one graph deliberately spans several components."
            % (len(unjustified), ", ".join(unjustified)))
    return notes


def explain(verdicts):
    """Group verdicts by action for the report, preserving order within a group."""
    grouped = {}
    for v in verdicts:
        grouped.setdefault(v.action, []).append(v)
    return grouped
