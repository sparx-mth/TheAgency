"""Stage 2 of the plan: measure the untouched torch model, before optimizing it.

Nothing downstream in this package may claim a speedup without a number that
came from here. :mod:`..trt_optimizer.amdahl` bounds the achievable gain from
each component's *share* of the per-decision budget, and a share only means
something if its denominator was measured on the real model, on the real
device, with the real inputs. Parameter counts are not a proxy for it: a 300M
ViT that runs once per episode costs nothing per frame, while a 2M denoiser
called twenty times inside one decision can own half the budget.

Four decisions in here are not obvious, and each of them is a way to produce
numbers that look plausible and are wrong.

**CUDA is asynchronous, so an unsynchronized hook times kernel launches.**
``forward`` returns as soon as the work has been *queued*. Without a
``torch.cuda.synchronize()`` on both edges of the timed region every component
comes out microscopic and suspiciously uniform -- the shape of a launch-cost
measurement, not of a workload. Both the pre-hook and the forward hook
therefore synchronize whenever the profiled modules hold CUDA tensors. It costs
a few microseconds per call, and it is the difference between an inventory and
a fiction.

**A submodule is not called once per forward.** A flow-matching or diffusion
head runs its denoiser K times inside one decision; a System-2 trunk may run
once every N frames. The hooks therefore *count* invocations and report a
per-call mean alongside the observed calls per run. Dividing total time by
``iters`` alone would bill a 20-step denoiser as a single monstrous call and
hide the real lever, which is K rather than the kernel.

**Nested components double-count.** Naming both ``model.head`` and
``model.head.denoiser`` measures the inner block inside the outer one, so
summing that inventory overstates the decision budget. This module refuses to
guess which one was meant: :func:`detect_overlap` reports the
ancestor/descendant pairs and the caller decides. It reports instead of raising
because an overlapping inventory is a legitimate thing to want -- it is how you
find out what fraction of a block its inner loop owns.

**A component that never ran is unmeasured, not free.** Zero observed calls is
what an ``ON_DEMAND`` branch, a wrong module name, or a submodule invoked as
``mod.forward(x)`` (which bypasses hooks entirely) all look like.
:func:`fill_latencies` leaves such a component's ``latency_ms`` at ``None``,
which makes :meth:`..spec.Plan.decision_ms` return ``None`` and stops a partial
denominator from silently inflating every share computed against it.

torch is imported lazily inside the functions that need it, so the pure logic --
:func:`detect_overlap` and :func:`fill_latencies` -- stays importable and
testable in the torch-free venv. Python-3.8-compatible syntax throughout.
"""
from __future__ import annotations

import time
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple,
                    TYPE_CHECKING)

from sparx_agency.tasks.common.trt_optimizer.spec import Component

if TYPE_CHECKING:  # pragma: no cover - import for annotations only
    from sparx_agency.tasks.common.trt_optimizer.bench.latency import (
        LatencyStats)

#: Relative tolerance when comparing a declared cadence against a measured one.
#: Both sides are exact rationals (integer counts over integer iterations), so
#: anything past this is a real disagreement rather than timing noise.
_CADENCE_RTOL = 1e-6


def profile_components(
        model: Any,
        run_fn: Callable[[], Any],
        component_names: Sequence[str],
        warmup: int = 3,
        iters: int = 10,
        device: Any = None,
) -> Dict[str, Dict[str, float]]:
    """Time the named submodules of ``model`` while ``run_fn`` drives it.

    Forward hooks are attached to each named submodule, ``run_fn()`` is called
    ``warmup`` times untimed and then ``iters`` times timed, and every
    invocation of every hooked submodule is bracketed by a
    :func:`time.perf_counter` pair. The hooks are removed in a ``finally``
    block, so a ``run_fn`` that raises leaves the model exactly as it was found.

    On CUDA the brackets synchronize on both edges (see the module docstring);
    the decision is taken once, before the run, from ``device`` when it is given
    and otherwise from the dtype/device of the profiled modules' own tensors. A
    model with no parameters or buffers at all is synchronized whenever CUDA is
    available, because syncing a CPU workload costs microseconds while not
    syncing a GPU one invalidates the whole run.

    Args:
        model: the ``torch.nn.Module`` that owns the named submodules.
        run_fn: zero-argument callable performing ONE full decision (one
            end-to-end inference). Its return value is discarded.
        component_names: dotted submodule paths, as they appear in
            ``model.named_modules()``. Duplicates are collapsed. ``""`` denotes
            the root model itself.
        warmup: untimed iterations run first, to pay for lazy CUDA context
            creation, cuDNN autotuning and allocator growth.
        iters: timed iterations. Must be >= 1.
        device: optional explicit device (``"cuda"``, ``"cuda:1"``, ``"cpu"``
            or a ``torch.device``) forcing the synchronization decision.

    Returns:
        Dict[str, Dict[str, float]]: ``{name: {"ms_per_call": mean wall time of
        ONE invocation, "calls_per_run": observed invocations divided by
        ``iters``}}``. ``calls_per_run == 0.0`` means the submodule was never
        observed -- ``ms_per_call`` is then 0.0 and carries no information.

    Raises:
        ImportError: if torch is not importable in this interpreter.
        ValueError: if ``component_names`` is empty, ``iters < 1`` or
            ``warmup < 0``.
        KeyError: if a name is not a submodule of ``model``. A silent skip here
            would report the component as free.
        RuntimeError: if ``device`` names CUDA and CUDA is unavailable.

    Note:
        Hooks fire on ``module(x)``, not on ``module.forward(x)``. Code that
        calls ``forward`` directly is invisible to this profiler and shows up
        as zero calls, which :func:`fill_latencies` refuses to read as zero
        cost.

    Note:
        A self-recursive module accumulates every frame of its own recursion,
        so its total counts the nesting twice. This is the same double-count
        :func:`detect_overlap` reports across components, and no model in this
        repo does it.
    """
    names = _unique(component_names)
    if not names:
        raise ValueError("profile_components() needs at least one component name")
    if int(iters) < 1:
        raise ValueError("iters must be >= 1, got %r" % (iters,))
    if int(warmup) < 0:
        raise ValueError("warmup must be >= 0, got %r" % (warmup,))
    torch_mod = _import_torch()

    modules = [(name, _resolve_submodule(model, name)) for name in names]
    sync = _make_sync(torch_mod, model, [m for _, m in modules], device)
    watches = dict((name, _Stopwatch()) for name in names)
    gate = _Gate()
    handles = []
    try:
        for name, module in modules:
            watch = watches[name]
            handles.append(
                module.register_forward_pre_hook(_pre_hook(watch, gate, sync)))
            handles.append(
                module.register_forward_hook(_post_hook(watch, gate, sync)))
        for _ in range(int(warmup)):
            run_fn()
        gate.recording = True
        for _ in range(int(iters)):
            run_fn()
    finally:
        gate.recording = False
        for handle in handles:
            handle.remove()

    return dict(
        (name, {"ms_per_call": watches[name].mean_ms(),
                "calls_per_run": watches[name].calls / float(int(iters))})
        for name in names)


def detect_overlap(model: Any,
                   component_names: Sequence[str]) -> List[Tuple[str, str]]:
    """Report which named components contain which others.

    Two components overlap when one is a submodule of the other, in which case
    the inner one's time is already inside the outer one's and the two
    measurements must never be summed. Containment is judged both by dotted
    name (``a`` contains ``a.b``) and structurally, by walking each component's
    own module tree -- the structural pass catches a component reached through
    a second attribute path, which the name test alone would miss.

    This function never raises. A name that cannot be resolved on ``model``
    degrades to the name-prefix test rather than failing, because overlap
    detection is a reporting aid the caller consults, never a gate that decides
    whether the pipeline may continue; the resolution errors that matter are
    raised by :func:`profile_components`, which does gate on them.

    Args:
        model: the module the names are relative to. Any object exposing
            ``get_submodule`` or ``named_modules`` works, which keeps this
            testable without torch.
        component_names: the names in the inventory. Duplicates are collapsed.

    Returns:
        List[Tuple[str, str]]: ``(ancestor, descendant)`` pairs, in the order
        the names were given. Empty when the inventory is a clean partition.

    Note:
        Two names bound to the *same* module object are aliases rather than
        nesting and are not reported here; that inventory is malformed in a
        different way and shows up as two identical latencies.
    """
    names = _unique(component_names)
    resolved = dict((name, _try_resolve(model, name)) for name in names)
    descendants = dict((name, _descendant_ids(resolved[name])) for name in names)
    pairs = []
    for outer in names:
        for inner in names:
            if outer == inner:
                continue
            if _contains(outer, inner, resolved, descendants):
                pairs.append((outer, inner))
    return pairs


def profile_end_to_end(
        run_fn: Callable[[], Any],
        warmup: int = 5,
        iters: int = 20,
        sync: Optional[Callable[[], None]] = None,
) -> "LatencyStats":
    """Measure the whole decision, which is the denominator of every share.

    Delegates to :func:`..bench.latency.measure` so the end-to-end baseline and
    every later engine benchmark are produced by one timing implementation and
    stay comparable.

    Args:
        run_fn: zero-argument callable performing ONE full decision.
        warmup: untimed iterations run first.
        iters: timed iterations.
        sync: zero-argument synchronization callable. ``None`` means *auto*:
            :func:`..bench.latency.cuda_sync` is used, which is the GPU
            synchronize on a CUDA process and ``None`` on a CPU-only one. Pass
            an explicit no-op to time launches deliberately.

    Returns:
        bench.latency.LatencyStats: mean/p50/p90/p99 over the timed iterations.
    """
    from sparx_agency.tasks.common.trt_optimizer.bench.latency import (
        cuda_sync, measure)

    if sync is None:
        sync = cuda_sync()
    return measure(run_fn, warmup=warmup, iters=iters, sync=sync)


def peak_memory_bytes(run_fn: Callable[[], Any], device: Any = None) -> int:
    """Run ``run_fn`` once and report the peak CUDA allocation it reached.

    The number bounds what an engine may cost at runtime: a workspace plus
    weights that do not fit beside the rest of the stack is a plan that cannot
    be deployed, and that is cheaper to learn here than on the aircraft.

    Args:
        run_fn: zero-argument callable performing ONE full decision.
        device: optional CUDA device to query; ``None`` is the current device.

    Returns:
        int: ``torch.cuda.max_memory_allocated`` in bytes, or 0 when CUDA is
        unavailable. ``run_fn`` is executed either way, so the caller's side
        effects do not depend on the presence of a GPU.

    Raises:
        ImportError: if torch is not importable in this interpreter.
    """
    torch_mod = _import_torch()
    if not torch_mod.cuda.is_available():
        run_fn()
        return 0
    torch_mod.cuda.reset_peak_memory_stats(device)
    run_fn()
    torch_mod.cuda.synchronize(device)
    return int(torch_mod.cuda.max_memory_allocated(device))


def fill_latencies(
        components: List[Component],
        measured: Dict[str, Any],
        calls_per_run: Optional[Dict[str, float]] = None,
) -> Tuple[List[Component], List[str]]:
    """Write measured timings into an inventory, letting measurement win.

    An adapter declares ``calls_per_decision`` from its reading of the model.
    When the profiler observed a different number, the observation is the truth
    and the declaration was a guess: this is the guard against an adapter that
    hard-coded ``1.0`` for a denoiser that actually runs twenty times, which
    would understate that component's share of the decision by 20x and route
    the whole optimization at the wrong block.

    Args:
        components: the inventory from ``dissect.inventory`` -- any objects
            carrying ``name``, ``latency_ms`` and ``calls_per_decision``.
            Mutated in place.
        measured: the mapping from :func:`profile_components`. Each value may
            be its ``{"ms_per_call": ..., "calls_per_run": ...}`` dict or a
            bare float of milliseconds per call.
        calls_per_run: optional ``{name: calls}`` overriding the call counts in
            ``measured``, for a caller who knows the cadence from a source the
            hooks cannot see.

    Returns:
        Tuple[List[Component], List[str]]: the same component objects, and the
        human-readable notes for :attr:`..spec.Plan.notes`. Every override and
        every gap in the measurement produces exactly one note -- an inventory
        that was silently patched is one nobody can review.

    Raises:
        KeyError: if a measurement dict has no ``ms_per_call`` entry.
    """
    notes = []
    for component in components:
        entry = measured.get(component.name)
        if entry is None:
            notes.append(
                "profile: %r was not measured; its latency stays unset and the "
                "decision budget stays undefined." % (component.name,))
            continue
        ms_per_call, calls = _unpack_measurement(entry)
        if calls_per_run is not None and component.name in calls_per_run:
            calls = float(calls_per_run[component.name])
        notes.extend(_apply_measurement(component, ms_per_call, calls))
    return components, notes


def _apply_measurement(component, ms_per_call, calls):
    """Apply one measurement to one component and return the notes it earned.

    Args:
        component: the component to mutate.
        ms_per_call: measured mean wall time of one call, in milliseconds.
        calls: measured calls per run, or None when unknown.

    Returns:
        List[str]: zero, one or two notes describing what was changed.
    """
    if calls is not None and calls <= 0.0:
        return ["profile: %r was never called during profiling -- left "
                "unmeasured rather than free (a wrong module name, a "
                "direct .forward() call, or an on-demand branch that did not "
                "fire)." % (component.name,)]
    component.latency_ms = float(ms_per_call)
    if calls is None:
        return []
    declared = float(component.calls_per_decision)
    if _close(calls, declared):
        return []
    component.calls_per_decision = float(calls)
    return ["profile: %r declared calls_per_decision=%g but was measured at %g "
            "call(s) per run; the measured cadence wins."
            % (component.name, declared, calls)]


class _Gate(object):
    """Shared on/off switch telling the hooks whether this run is being timed."""

    def __init__(self):
        self.recording = False


class _Stopwatch(object):
    """Accumulated wall time and invocation count for one hooked submodule.

    The open brackets are held on a stack so a re-entrant module cannot pair a
    ``forward`` exit with the wrong ``forward`` entry.
    """

    def __init__(self):
        self.total_ms = 0.0
        self.calls = 0
        self._open = []

    def enter(self, t0):
        """Record the start of one invocation."""
        self._open.append(t0)

    def exit(self, t1):
        """Close the innermost open invocation and bank its duration."""
        if not self._open:
            return
        self.total_ms += (t1 - self._open.pop()) * 1000.0
        self.calls += 1

    def mean_ms(self):
        """Mean milliseconds per invocation, or 0.0 if never invoked."""
        if self.calls == 0:
            return 0.0
        return self.total_ms / float(self.calls)


def _pre_hook(watch, gate, sync):
    """Build the forward-pre-hook that opens ``watch``'s timing bracket."""

    def _hook(module, inputs):
        if gate.recording:
            sync()
            watch.enter(time.perf_counter())

    return _hook


def _post_hook(watch, gate, sync):
    """Build the forward hook that closes ``watch``'s timing bracket."""

    def _hook(module, inputs, output):
        if gate.recording:
            sync()
            watch.exit(time.perf_counter())

    return _hook


def _make_sync(torch_mod, model, modules, device):
    """Return the zero-argument synchronize to call on both timing edges.

    Reuses :func:`..bench.latency.cuda_sync`, which pins the device index once
    so a later ``set_device`` elsewhere in the process cannot move what is being
    synchronized mid-run.

    Args:
        torch_mod: the imported torch module.
        model: the profiled model, consulted when the components hold no
            tensors of their own.
        modules: the resolved component modules.
        device: explicit device override, or None to infer.

    Returns:
        Callable[[], None]: the synchronize, or a no-op on CPU-only work.

    Raises:
        RuntimeError: if ``device`` names CUDA and CUDA is unavailable.
    """
    if not _sync_needed(torch_mod, model, modules, device):
        return _noop
    from sparx_agency.tasks.common.trt_optimizer.bench.latency import cuda_sync

    sync = cuda_sync()
    return sync if sync is not None else _noop


def _sync_needed(torch_mod, model, modules, device):
    """Decide once, before the run, whether the timing brackets must sync."""
    if device is not None:
        wants_cuda = "cuda" in str(device)
        if wants_cuda and not torch_mod.cuda.is_available():
            raise RuntimeError(
                "profile_components(device=%r) asks for CUDA timing but "
                "torch.cuda.is_available() is False" % (device,))
        return wants_cuda
    if not torch_mod.cuda.is_available():
        return False
    seen_any = False
    for module in list(modules) + [model]:
        for tensor in _module_tensors(module):
            seen_any = True
            if bool(getattr(tensor, "is_cuda", False)):
                return True
    # No tensors anywhere: a purely functional model whose device we cannot
    # read. Sync -- microseconds wasted on CPU beats a meaningless GPU run.
    return not seen_any


def _module_tensors(module):
    """Yield every parameter and buffer of ``module``, tolerating duck types."""
    for getter_name in ("parameters", "buffers"):
        getter = getattr(module, getter_name, None)
        if getter is None:
            continue
        for tensor in getter():
            yield tensor


def _noop():
    """Do nothing -- the synchronize used when no CUDA work is in flight."""


def _import_torch():
    """Import torch lazily, with a message naming the interpreter to use.

    Raises:
        ImportError: if torch is not installed in this interpreter.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "profile_torch needs torch; run it in an interpreter that has one "
            "(the repo's .venv deliberately does not). Original error: %s"
            % (exc,))
    return torch


def _resolve_submodule(model, name):
    """Resolve a dotted submodule name against ``model``.

    Raises:
        KeyError: if the name is not a submodule.
        TypeError: if ``model`` exposes neither ``get_submodule`` nor
            ``named_modules``.
    """
    getter = getattr(model, "get_submodule", None)
    if getter is not None:
        try:
            return getter(name)
        except AttributeError as exc:
            raise KeyError(
                "%r is not a submodule of the model: %s" % (name, exc))
    named = getattr(model, "named_modules", None)
    if named is None:
        raise TypeError(
            "model %r exposes neither get_submodule() nor named_modules(); it "
            "cannot be profiled by name" % (type(model).__name__,))
    table = dict(named())
    if name not in table:
        raise KeyError("%r is not a submodule of the model" % (name,))
    return table[name]


def _try_resolve(model, name):
    """Resolve a submodule for reporting, or None if it cannot be resolved."""
    try:
        return _resolve_submodule(model, name)
    except (KeyError, TypeError, AttributeError):
        return None


def _descendant_ids(module):
    """Return the ids of every module strictly below ``module``."""
    if module is None:
        return frozenset()
    named = getattr(module, "named_modules", None)
    if named is None:
        return frozenset()
    return frozenset(id(sub) for _, sub in named() if sub is not module)


def _contains(outer, inner, resolved, descendants):
    """True when component ``outer`` contains component ``inner``."""
    if outer == "" and inner != "":
        return True
    if inner.startswith(outer + "."):
        return True
    outer_module = resolved.get(outer)
    inner_module = resolved.get(inner)
    if outer_module is None or inner_module is None:
        return False
    if outer_module is inner_module:
        return False
    return id(inner_module) in descendants[outer]


def _unpack_measurement(entry):
    """Split one measurement into ``(ms_per_call, calls_per_run_or_None)``.

    Raises:
        KeyError: if a mapping entry carries no ``ms_per_call``.
    """
    getter = getattr(entry, "get", None)
    if getter is None:
        return float(entry), None
    if "ms_per_call" not in entry:
        raise KeyError(
            "measurement %r has no 'ms_per_call'; profile_components() "
            "produces one for every component" % (entry,))
    calls = getter("calls_per_run")
    return float(entry["ms_per_call"]), None if calls is None else float(calls)


def _close(a, b):
    """True when two cadences agree to within :data:`_CADENCE_RTOL`."""
    return abs(a - b) <= _CADENCE_RTOL * max(1.0, abs(a), abs(b))


def _unique(names):
    """Collapse duplicates in ``names``, preserving first-seen order."""
    out = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out
