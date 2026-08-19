"""Amdahl arithmetic: the gate that decides whether a conversion can ever pay.

A TensorRT engine is never free. It costs an export patch, a build recipe, a
numerical-drift risk and a second code path that has to be kept alive on every
target device. That price is only worth paying when the component being
converted actually owns a meaningful slice of one control decision -- and the
ceiling on what any conversion can buy is fixed by Amdahl's law long before a
single kernel is tuned. This module is where that ceiling is computed, so the
rest of the toolkit never has to argue about it.

Two decisions in here are deliberate and non-obvious:

**Cadence is checked before share, never after.** A text encoder that runs once
per episode may be profiled at 400 ms and still contribute nothing to the
steady-state rate, and it is usually *unprofiled* on the hot path -- so asking
"what is its share?" is not merely wrong, it is unanswerable. Cold components
are excluded by :data:`Cadence.COLD`, and only then is any arithmetic
attempted.
Doing it in the other order would either divide by an inventory that never
measured the component or, worse, admit it on a share computed from a one-shot
warm-up cost.

**The denominator is all-or-nothing.** :meth:`Plan.decision_ms` returns None
unless every component in the inventory has been profiled, and this module
propagates that as a ``ValueError`` rather than summing what it happens to
have.
A partial denominator understates the decision budget, which inflates every
share and every projected gain -- the exact failure mode that gets a component
converted for a speedup that never appears end to end. The repo rule applies:
a wrong number that flies is worse than a crash on the ground.

The one piece of arithmetic worth writing down, because everything here is a
rearrangement of it: speeding a component of share ``s`` by a factor ``f``
leaves a decision of length ``1 - s + s/f``, so the end-to-end *gain* is
``s * (1 - 1/f)`` and the end-to-end *speedup* is ``1 / (1 - s + s/f)``.
Letting ``f`` go to infinity gives the ceiling ``1 / (1 - s)``.

Pure standard library; no numpy, no torch, no TensorRT. Python-3.8-compatible
syntax so it can be imported on a Jetson's system interpreter.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from sparx_agency.tasks.common.trt_optimizer.spec import (
    Cadence,
    Component,
    Exportability,
    Plan,
)


def share(component: Component, plan: Plan) -> float:
    """Fraction of one decision's wall time spent in this component.

    Args:
        component: the :class:`Component` to weigh. It must be part of
            ``plan`` -- identity is by ``name``, because a share computed
            against a foreign inventory is meaningless.
        plan: the profiled :class:`Plan` that supplies the denominator.

    Returns:
        float: ``component.decision_ms / plan.decision_ms()``, in ``0..1``.

    Raises:
        ValueError: if the component is not in the plan, if the component is
            unprofiled, if the plan is unprofiled (any component missing a
            ``latency_ms``), or if the profiled decision has zero length.
    """
    if plan.component(component.name) is None:
        raise ValueError(
            "component %r is not in plan %r; its share would be measured "
            "against a decision budget that never included it"
            % (component.name, plan.model))
    total = _decision_total(plan)
    ms = component.decision_ms
    if ms is None:
        raise ValueError(
            "component %r is unprofiled (latency_ms is None); profile the "
            "inventory before asking for its share" % component.name)
    return ms / total


def max_speedup(plan: Plan, component_names: Sequence[str]) -> float:
    """Amdahl upper bound on end-to-end speedup if the named parts were free.

    This is the ceiling no amount of kernel work can beat: it assumes the named
    components drop to *zero* wall time. Read it before starting an export --
    if the bound is 1.05x, the conversion is not worth the maintenance burden
    no matter how good the engine turns out to be.

    Args:
        plan: the profiled :class:`Plan`.
        component_names: names of the components assumed to become free.
            Duplicates are ignored; an unknown name is an error, not a no-op.

    Returns:
        float: ``1 / (1 - s)`` where ``s`` is the summed share of the named
        components. Returns ``float('inf')`` when they are the entire decision.

    Raises:
        ValueError: if the plan is unprofiled or a name is not in the plan.
    """
    total_share = 0.0
    for name in _unique(component_names):
        component = plan.component(name)
        if component is None:
            raise ValueError(
                "component %r is not in plan %r (have: %s)"
                % (name, plan.model,
                   ", ".join(c.name for c in plan.components) or "nothing"))
        total_share += share(component, plan)
    remainder = 1.0 - total_share
    if remainder <= 0.0:
        return float("inf")
    return 1.0 / remainder


def speedup_with(plan: Plan, speedups: Dict[str, float]) -> float:
    """Realized end-to-end speedup for a per-component speedup factor mapping.

    Unlike :func:`max_speedup` this takes the factors you actually expect to
    measure, so it is the number that goes in a report next to a built engine.

    Args:
        plan: the profiled :class:`Plan`.
        speedups: mapping of component name -> speedup factor, where ``>1`` is
            faster and ``<1`` is a regression. Components absent from the
            mapping keep a factor of ``1.0``. An unknown name is an error.

    Returns:
        float: ``old_decision_ms / new_decision_ms``. ``float('inf')`` if the
        accelerated decision collapses to zero length.

    Raises:
        ValueError: if the plan is unprofiled, a name is not in the plan, or a
            factor is not strictly positive.
    """
    total = _decision_total(plan)
    for name, factor in speedups.items():
        if plan.component(name) is None:
            raise ValueError(
                "component %r is not in plan %r" % (name, plan.model))
        if factor <= 0.0:
            raise ValueError(
                "speedup factor for %r is %r; a factor must be strictly "
                "positive (1.0 means unchanged)" % (name, factor))
    accelerated = 0.0
    for component in plan.components:
        factor = speedups.get(component.name, 1.0)
        accelerated += component.decision_ms / factor
    if accelerated <= 0.0:
        return float("inf")
    return total / accelerated


def worth_converting(component: Component, plan: Plan,
                     min_share: float = 0.05,
                     min_end_to_end_gain: float = 0.02,
                     assumed_speedup: float = 3.0) -> Tuple[bool, str]:
    """Decide whether converting a component is worth the complexity it costs.

    The rules are applied in order and the first match wins. Cadence and
    exportability come first precisely because they need no profiling: a cold
    or ONNX-hostile component is rejected even in an unprofiled plan, where a
    share threshold would have nothing to divide by.

    Args:
        component: the :class:`Component` under consideration.
        plan: the :class:`Plan` it belongs to.
        min_share: smallest fraction of a decision worth attacking at all.
        min_end_to_end_gain: smallest end-to-end fraction of wall time the
            conversion must return, at ``assumed_speedup``.
        assumed_speedup: the component-level speedup a conversion is assumed to
            deliver. 3.0 is the conservative default for an FP16 engine
            replacing eager PyTorch.

    Returns:
        tuple: ``(bool, str)`` -- the verdict and the reason a human will read
        in the final report. The reason always names the number that decided
        it, never just the rule.

    Raises:
        ValueError: if ``assumed_speedup`` is not strictly positive, or (for
            rules 3 onward) the plan or component is unprofiled.
    """
    if assumed_speedup <= 0.0:
        raise ValueError(
            "assumed_speedup must be > 0, got %r" % assumed_speedup)
    if component.cadence in Cadence.COLD:
        return False, ("runs %s, contributes nothing to steady-state rate"
                       % component.cadence)
    if component.exportability == Exportability.HOSTILE:
        return False, _hostile_reason(component)
    measured = share(component, plan)
    total = plan.decision_ms()
    if measured < min_share:
        return False, ("owns %s of the %.2f ms decision (%.2f ms), under the "
                       "%s floor worth attacking"
                       % (_pct(measured), total, component.decision_ms,
                          _pct(min_share)))
    gain = measured * (1.0 - 1.0 / assumed_speedup)
    if gain < min_end_to_end_gain:
        return False, ("owns %s of the %.2f ms decision, but at an assumed "
                       "%.1fx that returns only %s end to end, under the %s "
                       "floor" % (_pct(measured), total, assumed_speedup,
                                  _pct(gain), _pct(min_end_to_end_gain)))
    return True, ("owns %s of the %.2f ms decision (%.2f ms); at an assumed "
                  "%.1fx that returns %s end to end, %.2fx overall"
                  % (_pct(measured), total, component.decision_ms,
                     assumed_speedup, _pct(gain), _overall(gain)))


def rank(plan: Plan) -> List[Component]:
    """Components ordered by the wall time they cost one decision, worst first.

    This is the list a human reads top-down when deciding what to attack:
    ``decision_ms`` already folds in ``calls_per_decision``, so a 3 ms kernel
    called twenty times per decision correctly outranks a 20 ms kernel called
    once. Unprofiled components sort last -- they are not "fast", they are
    unknown, and pretending a missing measurement is a zero would hide the
    thing most worth measuring next.

    Args:
        plan: the :class:`Plan` whose inventory to order. It need not be fully
            profiled.

    Returns:
        list: a new list of :class:`Component`, profiled ones sorted by
        ``decision_ms`` descending, then the unprofiled ones in inventory
        order. Ties keep inventory order (the sort is stable).
    """
    profiled = [c for c in plan.components if c.decision_ms is not None]
    unprofiled = [c for c in plan.components if c.decision_ms is None]
    profiled.sort(key=lambda c: c.decision_ms, reverse=True)
    return profiled + unprofiled


def _decision_total(plan):
    """Profiled length of one decision, or raise naming what is missing."""
    total = plan.decision_ms()
    if total is None:
        raise ValueError(
            "plan %r is unprofiled: %s. Every share and every Amdahl bound is "
            "divided by the decision budget, and a partial sum would silently "
            "inflate all of them." % (plan.model, _missing(plan)))
    if total <= 0.0:
        raise ValueError(
            "plan %r profiles to a %.6f ms decision; nothing can be a "
            "fraction of a zero-length decision" % (plan.model, total))
    return total


def _missing(plan):
    """Human-readable note about why a plan has no decision time."""
    if not plan.components:
        return "it has no components"
    names = [c.name for c in plan.components if c.decision_ms is None]
    return "unprofiled components: %s" % ", ".join(names)


def _hostile_reason(component):
    """Reason line for a hostile component, quoting its recorded blocker."""
    if component.reason:
        return "hostile to ONNX export: %s" % component.reason
    return ("hostile to ONNX export and no blocker was recorded on the "
            "component; record one before revisiting this")


def _overall(gain):
    """End-to-end speedup implied by an end-to-end gain fraction.

    Guards the degenerate case where the component is the entire decision and
    the assumed speedup is infinite -- the honest answer there is an infinite
    end-to-end speedup, not a ZeroDivisionError.
    """
    if gain >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - gain)


def _pct(fraction):
    """Format a 0..1 fraction as a percentage string."""
    return "%.1f%%" % (100.0 * fraction)


def _unique(names):
    """De-duplicated names, in first-seen order."""
    seen = set()
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out
