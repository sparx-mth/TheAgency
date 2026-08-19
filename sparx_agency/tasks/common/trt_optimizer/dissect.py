"""Stage 1 of the optimizer: turn a live model into a :class:`Component` inventory.

Every later stage -- the Amdahl bound, the export verdicts, the report -- reasons
over a list of :class:`~sparx_agency.tasks.common.trt_optimizer.spec.Component`.
This module is the only one that touches a *model object*, and it has to work on
an architecture the toolkit has never seen. So it knows nothing about
transformers, ViTs or diffusion heads: it walks ``named_children()`` and counts
parameters, and that is all.

Four decisions in here are load-bearing.

**The frontier, not the leaves.** The walk stops at ``max_depth`` and emits one
component per node on that frontier. A 300-block ViT has to appear as a single
``vision_tower`` row, because the inventory is a *latency* document -- nobody
profiles, exports or converts one transformer block, and 300 rows of 0.3% each
hide the fact that the trunk is 90% of the frame budget.

**Accounting is a hard invariant.** ``sum(c.params) == total_params(model)``,
always. A subtree dropped by ``min_params`` is not discarded, it is folded into a
synthetic ``<parent>.other`` component. The failure this prevents is specific and
nasty: a truncated inventory produces a *confident* Amdahl analysis that is
quietly wrong, because the denominator it divides by is missing weight it never
knew about. :func:`check_accounting` exists to make that crash instead.

**Parameters only; buffers counted apart.** ``Component.params`` counts what
``named_parameters()`` yields. Running statistics, position-embedding tables and
attention masks are buffers -- they ride along in the checkpoint and cost VRAM,
but mixing them into the parameter count breaks the identity above and inflates
the weight footprint of the modules that happen to hold them. :func:`total_buffers`
reports them separately so they stay visible.

**Duck-typed, torch optional.** Nothing here imports torch at module scope; torch
is imported lazily and only to sharpen an error message. Every function works
against any object exposing ``named_children()``, ``named_parameters()`` and
``parameters()`` (plus ``named_buffers()`` for :func:`total_buffers`), where a
parameter is anything with ``.numel()`` and ``.dtype``. That keeps the accounting
logic unit-testable against a hand-built fake tree in the pure-numpy venv, which
is where the invariant above actually gets its coverage.

Python-3.8-compatible syntax: this may be imported on a Jetson's system
interpreter alongside the deployed runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from sparx_agency.tasks.common.trt_optimizer.spec import (
    Cadence, Component, Exportability)

_LOG = logging.getLogger(__name__)

#: Dtype reported for a component that holds no parameters at all. It matches
#: :class:`Component`'s own default; a zero-parameter row has no dtype to report.
_DEFAULT_DTYPE = "float32"

#: Suffix of the synthetic bucket that keeps the parameter accounting exact.
_OTHER_SUFFIX = "other"

#: Opening words of a synthetic bucket's ``reason``. Together with the name
#: suffix this is what :func:`is_synthetic` matches on, so a real submodule
#: that happens to be called ``other`` is not mistaken for one.
_SYNTHETIC_REASON = "synthetic accounting bucket"


# --------------------------------------------------------------------------
# duck-typing guards
# --------------------------------------------------------------------------

def _lazy_torch():
    """Return the ``torch`` module, or None when it is not installed."""
    try:
        import torch  # noqa: F401  (optional; only used to explain an error)
        return torch
    except Exception:  # pragma: no cover - depends on the interpreter
        return None


def _check_module_api(obj, required):
    """Raise TypeError unless ``obj`` exposes every method name in ``required``."""
    missing = [name for name in required if not callable(getattr(obj, name, None))]
    if not missing:
        return
    hint = "" if _lazy_torch() is not None else (
        " (torch is not importable in this interpreter, so a real nn.Module "
        "cannot be passed here either)")
    raise TypeError(
        "%r does not expose the module API this walker needs: missing %s%s"
        % (type(obj).__name__, ", ".join("%s()" % m for m in missing), hint))


def _root_label(model):
    """Logical name of the root, used to prefix its synthetic bucket."""
    return type(model).__name__


def _join(prefix, name):
    """Join a dotted module path, tolerating an empty root prefix."""
    return "%s.%s" % (prefix, name) if prefix else name


# --------------------------------------------------------------------------
# parameter bookkeeping
# --------------------------------------------------------------------------

def _subtree_params(module):
    """Map ``id(param) -> param`` for every parameter under ``module``.

    Keyed by identity so a weight tied across two subtrees (a shared embedding)
    is counted once, exactly as ``torch``'s own ``parameters()`` de-duplicates.
    """
    out = {}
    for _name, param in module.named_parameters():
        out.setdefault(id(param), param)
    return out


def _unseen_params(module, seen):
    """Subtree parameters of ``module`` that no earlier component has claimed."""
    return {pid: p for pid, p in _subtree_params(module).items() if pid not in seen}


def _direct_params(module, children, seen):
    """Unclaimed parameters held on ``module`` itself rather than on a child."""
    owned = _unseen_params(module, seen)
    for _name, child in children:
        for pid in _subtree_params(child):
            owned.pop(pid, None)
    return owned


def _numel_total(params):
    """Element count of a ``{id: param}`` mapping."""
    return int(sum(int(p.numel()) for p in params.values()))


def _dtype_name(param):
    """Plain dtype string of one parameter (``'float32'``, ``'bfloat16'``).

    Raises:
        TypeError: if the parameter has no ``dtype`` -- a broken duck-type is a
            bug in the caller, not something to paper over with a default.
    """
    dtype = getattr(param, "dtype", None)
    if dtype is None:
        raise TypeError("parameter %r has no .dtype" % (param,))
    text = str(dtype)
    return text[len("torch."):] if text.startswith("torch.") else text


def _dominant_dtype(params):
    """Dtype holding the most *elements* here; ties go to the first seen."""
    counts = {}
    for param in params.values():
        name = _dtype_name(param)
        counts[name] = counts.get(name, 0) + int(param.numel())
    best = _DEFAULT_DTYPE
    best_count = -1
    for name, count in counts.items():  # insertion-ordered, so ties are stable
        if count > best_count:
            best, best_count = name, count
    return best


# --------------------------------------------------------------------------
# cadence / exportability resolution
# --------------------------------------------------------------------------

def _validated_cadences(cadences):
    """Copy ``cadences``, raising on any value outside :class:`Cadence`."""
    if not cadences:
        return {}
    for key, value in cadences.items():
        if value not in Cadence.ALL:
            raise ValueError(
                "cadences[%r] = %r is not a Cadence; expected one of %s"
                % (key, value, ", ".join(Cadence.ALL)))
    return dict(cadences)


def _resolve_cadence(name, cadences):
    """Return ``(cadence, matched)``: exact key wins, then longest prefix."""
    if name in cadences:
        return cadences[name], True
    best = None
    for key in cadences:
        if name.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is not None:
        return cadences[best], True
    return Cadence.PER_FRAME, False


def _resolve_exportability(fn, module, name):
    """Call ``fn(module)`` and validate its ``(exportability, reason)`` answer."""
    if fn is None:
        return Exportability.CLEAN, ""
    result = fn(module)
    try:
        exportability, reason = result
    except (TypeError, ValueError):
        raise ValueError(
            "exportability_fn(%s) returned %r; expected an "
            "(exportability, reason) pair" % (name, result))
    if exportability not in Exportability.ALL:
        raise ValueError(
            "exportability_fn(%s) returned exportability %r; expected one of %s"
            % (name, exportability, ", ".join(Exportability.ALL)))
    return exportability, str(reason)


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

@dataclass
class _Walk:
    """Mutable state of one :func:`inventory` walk."""

    max_depth: int
    min_params: int
    cadences: Dict[str, str]
    exportability_fn: Optional[Callable]
    root_label: str
    seen: set = field(default_factory=set)
    out: List[Component] = field(default_factory=list)


def _emit(walk, name, module, params):
    """Append one real component, claiming its parameters."""
    walk.seen.update(params.keys())
    exportability, reason = _resolve_exportability(
        walk.exportability_fn, module, name)
    cadence, matched = _resolve_cadence(name, walk.cadences)
    if not matched:
        _LOG.info("dissect: no cadence given for %r, assuming %s (hot path); "
                  "pass cadences={%r: ...} to override", name, cadence, name)
    walk.out.append(Component(
        name=name, params=_numel_total(params), cadence=cadence,
        exportability=exportability, reason=reason,
        dtype=_dominant_dtype(params)))


def _emit_other(walk, prefix, params):
    """Append the synthetic ``<parent>.other`` bucket that preserves the total.

    It is marked HOSTILE deliberately: it is a sum over several places in the
    tree, not an ``nn.Module``, so there is nothing for ``torch.onnx.export`` to
    point at and no verdict may ever try.
    """
    walk.seen.update(params.keys())
    name = "%s.%s" % (prefix or walk.root_label, _OTHER_SUFFIX)
    cadence, _matched = _resolve_cadence(name, walk.cadences)
    walk.out.append(Component(
        name=name, params=_numel_total(params), cadence=cadence,
        exportability=Exportability.HOSTILE,
        reason=("%s: parameters held directly on %r plus subtrees below "
                "min_params; not a module, not exportable"
                % (_SYNTHETIC_REASON, prefix or walk.root_label)),
        dtype=_dominant_dtype(params)))


def _walk(module, prefix, depth, walk):
    """Expand ``module``: emit each child at the frontier, recurse otherwise."""
    children = list(module.named_children())
    leftover = _direct_params(module, children, walk.seen)
    for child_name, child in children:
        path = _join(prefix, child_name)
        params = _unseen_params(child, walk.seen)
        if _numel_total(params) < walk.min_params:
            leftover.update(params)
            continue
        if depth + 1 >= walk.max_depth or not list(child.named_children()):
            _emit(walk, path, child, params)
        else:
            _walk(child, path, depth + 1, walk)
    if leftover:
        _emit_other(walk, prefix, leftover)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def inventory(model, max_depth=2, min_params=0, cadences=None,
              exportability_fn=None):
    """Walk a model and return the :class:`Component` inventory.

    Components come back in structural (definition) order, not sorted by size,
    so the table reads like the model reads. ``calls_per_decision`` is left at
    the neutral ``1.0``: dissection sees structure, never call counts, and
    inventing one would be exactly the "wrong number that flies".

    Args:
        model: any object exposing ``named_children()`` and ``named_parameters()``
            -- a ``torch.nn.Module`` in production, a fake tree in the tests.
        max_depth: how many levels of ``named_children()`` to descend. ``1``
            emits only the top-level children; ``2`` (the default) emits their
            children. A node shallower than the frontier with no children of its
            own is emitted where it sits.
        min_params: subtrees with fewer parameters than this are not emitted;
            their parameters fold into ``<parent>.other`` so the total is kept.
        cadences: optional mapping of component name -- or a name prefix -- to a
            :class:`Cadence` value. An exact name wins; otherwise the longest
            matching prefix wins, so a specific override beats a general one.
            Anything unmatched gets :data:`Cadence.PER_FRAME` (the pessimistic
            assumption -- an unclassified component stays in the hot-path
            denominator) and logs a note naming it.
        exportability_fn: optional ``callable(module) -> (exportability, reason)``.
            When None every real component is :data:`Exportability.CLEAN` with an
            empty reason. It is never called for a synthetic ``.other`` bucket,
            which has no module.

    Returns:
        List[Component]: one component per frontier node, plus any ``.other``
        buckets, whose ``params`` sum to :func:`total_params`.

    Raises:
        TypeError: if ``model`` does not expose the walk API.
        ValueError: if ``max_depth`` is below 1, if a cadence value is not a
            :class:`Cadence`, or if ``exportability_fn`` returns something other
            than a valid ``(exportability, reason)`` pair.
    """
    _check_module_api(model, ("named_children", "named_parameters"))
    max_depth = int(max_depth)
    if max_depth < 1:
        raise ValueError(
            "max_depth must be >= 1 (got %d); the root of an unnamed model has "
            "no name to emit a component under" % max_depth)
    walk = _Walk(max_depth=max_depth, min_params=int(min_params),
                 cadences=_validated_cadences(cadences),
                 exportability_fn=exportability_fn,
                 root_label=_root_label(model))
    if list(model.named_children()):
        _walk(model, "", 0, walk)
    else:
        _emit(walk, walk.root_label, model, _unseen_params(model, walk.seen))
    return walk.out


def total_params(model):
    """Total parameter count of ``model``, de-duplicated by identity.

    Args:
        model: object exposing ``parameters()``.

    Returns:
        int: the number this module's whole accounting story is measured against.
    """
    _check_module_api(model, ("parameters",))
    unique = {id(p): p for p in model.parameters()}
    return _numel_total(unique)


def total_buffers(model):
    """Total *buffer* element count, reported apart from the parameters.

    Buffers are excluded from :class:`Component`. ``params`` on purpose (see the
    module docstring); this is where they stay visible, so a large
    position-embedding table is not simply missing from the VRAM story.

    Args:
        model: object exposing ``named_buffers()``.

    Returns:
        int: buffer elements under ``model``, de-duplicated by identity.
    """
    _check_module_api(model, ("named_buffers",))
    unique = {id(b): b for _name, b in model.named_buffers()}
    return _numel_total(unique)


def dtype_histogram(model):
    """Map parameter dtype -> number of parameter *elements* at that dtype.

    Args:
        model: object exposing ``parameters()``.

    Returns:
        Dict[str, int]: e.g. ``{'bfloat16': 85_000_000, 'float32': 1_200}``. A
        mixed-dtype answer is the signal that a checkpoint was partially cast,
        which changes both the export path and the expected TensorRT speedup.
    """
    _check_module_api(model, ("parameters",))
    counts = {}
    for param in {id(p): p for p in model.parameters()}.values():
        name = _dtype_name(param)
        counts[name] = counts.get(name, 0) + int(param.numel())
    return counts


def is_synthetic(component):
    """True when ``component`` is an accounting bucket rather than a module.

    :func:`inventory` emits a ``<parent>.other`` row for parameters held
    directly on a frontier node plus any subtree below ``min_params``, so the
    parameter total stays exact. That row is a *sum over several places in the
    tree*: there is no ``nn.Module`` to resolve it against, no forward hook to
    attach to it, and nothing for ``torch.onnx.export`` to point at.

    Callers that resolve component names against the model must skip these --
    :func:`..profile_torch.profile_components` raises ``KeyError`` on a name it
    cannot resolve, and it is right to, because for a *real* component that is
    an adapter bug rather than a soft condition.

    Args:
        component: a :class:`..spec.Component`.

    Returns:
        bool: True for a synthetic bucket.
    """
    return (str(component.name).endswith("." + _OTHER_SUFFIX)
            and str(component.reason).startswith(_SYNTHETIC_REASON))


def check_accounting(components, model):
    """Raise unless the inventory accounts for every parameter in ``model``.

    Args:
        components: the list returned by :func:`inventory`.
        model: the model it was built from.

    Raises:
        ValueError: if the sums disagree. This is the guard that stops a
            silently truncated inventory from producing a confident, wrong
            Amdahl analysis.
    """
    counted = int(sum(int(c.params) for c in components))
    expected = total_params(model)
    if counted != expected:
        raise ValueError(
            "inventory accounts for %d parameters but %s has %d (%+d); a "
            "component was dropped without folding it into a '.other' bucket, "
            "so every share and speedup computed from this inventory would be "
            "wrong" % (counted, _root_label(model), expected,
                       counted - expected))


def describe(components):
    """Render the inventory as a plain-text table for a terminal.

    Args:
        components: the list returned by :func:`inventory`.

    Returns:
        str: aligned ``name / params / dtype / cadence`` rows plus a TOTAL line.
    """
    if not components:
        return "(no components)"
    headers = ("component", "params", "dtype", "cadence")
    rows = [(c.name, _human(c.params), c.dtype, c.cadence) for c in components]
    total = ("TOTAL", _human(sum(int(c.params) for c in components)), "", "")
    widths = [max(len(r[i]) for r in list(rows) + [headers, total])
              for i in range(4)]
    line = "  ".join("-" * w for w in widths)

    def _fmt(row):
        return "  ".join(row[i].ljust(widths[i]) for i in range(4)).rstrip()

    return "\n".join([_fmt(headers), line]
                     + [_fmt(r) for r in rows] + [line, _fmt(total)])


def _human(count):
    """Format a parameter count in human units (``'11.69M'``, ``'512'``)."""
    count = int(count)
    for suffix, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(count) >= scale:
            return "%.2f%s" % (count / scale, suffix)
    return str(count)
