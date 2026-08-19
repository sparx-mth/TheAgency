"""Judge ONNX-exportability from source, before anyone spends a day on export.

Misreading exportability is the single most expensive mistake in a TensorRT
project: the failure mode is not a fast error, it is an afternoon of chasing
``torch.onnx.export`` tracebacks for a component that was never convertible in
principle. This module front-runs that by reading the *source* -- the model's
own code -- and matching it against markers that are each grounded in a real,
observed export failure.

The verdicts are :class:`..spec.Exportability`'s three, and every marker below
is grounded in a failure that was actually hit, not in a guess about ONNX.

Non-obvious decisions in here:

**Hostile wins.** A component that is both patchable and hostile is hostile;
patching it buys a graph that still cannot be traced. The precedence is not a
severity ranking, it is a claim about wasted effort.

**Every match is reported, not the first.** :func:`scan` returns the whole list,
because the question a human actually asks is "how deep is this hole" and one
marker never answers it. :func:`classify_source` collapses the list into the
single verdict a plan needs, naming at most three markers and pointing back at
``scan()`` for the rest.

**Torch's own source is never read.** :func:`classify_module` reads the source
of the *user's* classes but skips any class defined under the installed torch
package: ``torch.nn``'s own ``MultiheadAttention`` mentions
``scaled_dot_product_attention`` and every fast path it has, so reading it would
flag every model ever built. Stock torch layers are judged by class *name*
instead -- the only signal there that describes the model and not torch.

**Markers are heuristics over text, and the module says so.** Source scanning
cannot be sound: a comment mentioning ``past_key_values`` trips the KV-cache
marker, and CLEAN means only that nothing known matched. The bias is chosen --
a false HOSTILE costs a five-minute read, a false CLEAN costs the afternoon --
but nothing here fails quietly: bad input raises and an unknown marker raises.

Pure standard library. torch is imported lazily and only to locate the
installed package root; the module is fully usable, and fully tested, in an
interpreter that has no torch at all. Python-3.8-compatible syntax.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Dict, Tuple

from sparx_agency.tasks.common.trt_optimizer.spec import Exportability

_KIND_HOSTILE = "hostile"
_KIND_PATCHABLE = "patchable"

#: Default depth of the ``named_modules()`` walk. Three levels reach the block
#: and the attention layer inside it, which is where the markers live, without
#: paying for the leaf Linear/LayerNorm of every one of 24 blocks.
_DEFAULT_MAX_DEPTH = 3

#: Markers that make a component un-exportable, mapped to why. Each is a real
#: failure seen in this repo's VLA work or in the upstream models it wraps.
HOSTILE_MARKERS: Dict[str, str] = {
    "generate(": (
        "autoregressive decode: a Python sampling loop that tracing unrolls "
        "into one fixed-length graph. Route it to an LLM runtime."
    ),
    "past_key_values": (
        "a KV cache grows by one token per step, so the attention inputs "
        "change shape on every call; a static engine cannot hold it."
    ),
    "DynamicCache": (
        "HuggingFace's growing-shape cache is a Python container, not a "
        "tensor, and does not survive tracing."
    ),
    "use_cache": (
        "selects the cached autoregressive path, whose shapes follow the step "
        "index; only a model with the decode loop removed can export."
    ),
    "flash_attn": (
        "a custom CUDA kernel with no ONNX symbolic: the export dies on the "
        "op and there is nothing inside the graph to patch."
    ),
    "flash_attention_2": (
        "the same custom kernel selected by name; the module must be rebuilt "
        "with eager or math-SDPA attention before any export attempt."
    ),
    "attn_implementation": (
        "the attention backend is chosen at load time, and anything but eager "
        "or math-SDPA leaves a fused, unexportable op in the graph. Read the "
        "call site: it names the backend this checkpoint will really use."
    ),
    "xformers": "xFormers kernels are not traceable and carry no ONNX symbolic.",
    "memory_efficient_attention": (
        "the fused xFormers entry point collapses into one opaque CUDA call "
        "that ONNX cannot represent."
    ),
    ".item()": (
        "moves a tensor value into Python, so every shape derived from it is "
        "data-dependent and freezes to whatever the trace happened to see."
    ),
    "torch.nonzero": (
        "the output shape depends on the data, which a fully static engine "
        "cannot express."
    ),
    "masked_scatter": (
        "the Qwen2.5-VL image-token splice: a masked_scatter guarded by an "
        ".item() token count, so the graph shape follows the image content."
    ),
    "cu_seqlens": (
        "varlen packed attention: sequences are concatenated and delimited by "
        "a cumulative-length tensor, so batch and sequence dims are "
        "data-dependent by construction."
    ),
    "tensor_dependent_branch": (
        "a Python `if` on a tensor value: tracing keeps only the branch taken "
        "at export time and silently discards the other."
    ),
    "tensor_length_loop": (
        "a Python `for` over a tensor-derived length -- mRoPE index building "
        "over image_grid_thw is the canonical case -- unrolled by tracing to "
        "the trace-time count."
    ),
}

#: Markers that block export until a known patch is applied, mapped to the
#: patch. The value is the instruction, specific enough to act on.
PATCHABLE_MARKERS: Dict[str, str] = {
    "gradient_checkpointing": (
        "call .disable_gradient_checkpointing() before exporting: the "
        "checkpoint wrapper re-enters the forward and derails the tracer."
    ),
    "guidance_scale_one": (
        "classifier-free guidance at scale 1.0 is a provable no-op that still "
        "doubles the batch; delete the null-conditioning branch before export."
    ),
    "MemoryEfficientSwish": (
        "swap EfficientNet's MemoryEfficientSwish for a plain nn.SiLU: the "
        "custom autograd Function has no symbolic, the arithmetic is identical."
    ),
    "AdaptiveAvgPool1d": (
        "replace it with a fixed matrix multiply at the known input length."
    ),
    "view_as_complex": (
        "complex rotary embeddings: rewrite them as real sin/cos pairs, "
        "because ONNX has no complex tensor type."
    ),
    "interpolate_pos_encoding": (
        "pre-bake the positional embedding at the fixed export resolution so "
        "no Resize node survives (the DINOv2 patch)."
    ),
    "bicubic": (
        "a bicubic positional-embedding resize: bake it at the fixed input "
        "size, since a traced Resize pins the mode and drags a dynamic shape "
        "in with it."
    ),
    "scaled_dot_product_attention": (
        "force the SDPA MATH backend around the export call, or a fused "
        "*Attention op leaks into the graph and the op gate rejects it."
    ),
    "MultiheadAttention": (
        "disable nn.MultiheadAttention's fast path (need_weights=True, or the "
        "math-only SDPA context) so it decomposes into MatMul/Softmax."
    ),
}

#: The order patches must be applied in: structural edits to the module first,
#: then the shape-baking rewrites, and last the backend switches, which are
#: context managers wrapping the export call itself rather than edits to the
#: model. Applying a backend switch first and then editing the module silently
#: drops the switch, which is why this is an explicit order and not a sort.
_PATCH_ORDER: Tuple[str, ...] = (
    "gradient_checkpointing",
    "guidance_scale_one",
    "MemoryEfficientSwish",
    "AdaptiveAvgPool1d",
    "view_as_complex",
    "interpolate_pos_encoding",
    "bicubic",
    "scaled_dot_product_attention",
    "MultiheadAttention",
)

#: Markers that are shapes of code rather than tokens. Everything not listed
#: here is matched as a literal substring of the source.
_PATTERN_RULES = {
    "tensor_dependent_branch": re.compile(
        r"(?m)^[ \t]*(?:el)?if\s+[^\n:]*?\."
        r"(?:item|any|all|numel|nonzero|count_nonzero)\s*\("),
    "tensor_length_loop": re.compile(
        r"(?m)^[ \t]*for\s+[^\n]*?\bin\b[^\n]*?"
        r"(?:image_grid_thw|\.shape\[|\.size\(|\.tolist\(\)|\.item\(\))"),
    # The trailing lookahead is the whole point: `guidance_scale = 1.5` is a
    # LIVE guidance scale, and deleting its null branch would change the
    # model's output. Only an exact 1 / 1.0 makes that branch a no-op.
    "guidance_scale_one": re.compile(
        r"guidance_scale\s*(?::\s*\w+\s*)?(?:==|=)\s*1(?:\.0+)?(?![\d.])"),
    # The window crosses newlines on purpose: a real pos-embed resize is a
    # wrapped multi-line interpolate() call, and a line-bound window misses it.
    "bicubic": re.compile(
        r"bicubic[\s\S]{0,160}(?:pos|embed)"
        r"|(?:pos|embed)[\s\S]{0,160}bicubic"),
}


def _pattern_for(marker):
    """Compiled matcher for one marker: its own pattern, else a literal."""
    pattern = _PATTERN_RULES.get(marker)
    if pattern is not None:
        return pattern
    return re.compile(re.escape(marker))


def _build_rules():
    """Ordered (marker, kind, pattern) triples -- hostile rules first.

    The ordering is load-bearing: :func:`scan` reports findings in rule order,
    so a reader sees the blockers before the chores.
    """
    rules = []
    for marker in HOSTILE_MARKERS:
        rules.append((marker, _KIND_HOSTILE, _pattern_for(marker)))
    for marker in PATCHABLE_MARKERS:
        rules.append((marker, _KIND_PATCHABLE, _pattern_for(marker)))
    return tuple(rules)


_RULES = _build_rules()


def _check_invariants():
    """Raise at import if the marker tables and the patch order disagree.

    Raises:
        RuntimeError: if :data:`_PATCH_ORDER` does not cover exactly the keys
            of :data:`PATCHABLE_MARKERS`, which would make :func:`patch_plan`
            drop or crash on a legitimate finding.
    """
    missing = set(PATCHABLE_MARKERS) - set(_PATCH_ORDER)
    extra = set(_PATCH_ORDER) - set(PATCHABLE_MARKERS)
    if missing or extra:
        raise RuntimeError(
            "_PATCH_ORDER is out of sync with PATCHABLE_MARKERS: missing %r, "
            "unknown %r" % (sorted(missing), sorted(extra)))


_check_invariants()


def _scan_text(text):
    """Every marker matching ``text``, in rule order."""
    findings = []
    for marker, kind, pattern in _RULES:
        if pattern.search(text) is None:
            continue
        table = HOSTILE_MARKERS if kind == _KIND_HOSTILE else PATCHABLE_MARKERS
        findings.append({"marker": marker, "kind": kind, "why": table[marker]})
    return findings


def _summarize(findings, limit=3):
    """One reason line naming at most ``limit`` markers."""
    shown = findings[:limit]
    text = "; ".join("%s -- %s" % (f["marker"], f["why"]) for f in shown)
    if len(findings) > limit:
        text += (" (+%d further marker(s); call scan() for the full picture)"
                 % (len(findings) - limit))
    return text


def _verdict(findings, n_chars):
    """Collapse findings into ``(exportability, reason)``. Hostile wins."""
    hostile = [f for f in findings if f["kind"] == _KIND_HOSTILE]
    if hostile:
        return Exportability.HOSTILE, "%d hostile marker(s): %s" % (
            len(hostile), _summarize(hostile))
    patchable = [f for f in findings if f["kind"] == _KIND_PATCHABLE]
    if patchable:
        return Exportability.NEEDS_PATCH, "%d patchable marker(s): %s" % (
            len(patchable), _summarize(patchable))
    return Exportability.CLEAN, (
        "no hostile or patchable marker in %d characters of source; this is "
        "absence of evidence, so still confirm with one export dry run"
        % n_chars)


def classify_source(source_text):
    """Judge one module's source text.

    Args:
        source_text: the Python source of the component, as read from disk or
            from ``inspect.getsource``.

    Returns:
        ``(exportability, reason)`` where exportability is one of
        :attr:`..spec.Exportability.CLEAN` / ``NEEDS_PATCH`` / ``HOSTILE`` and
        reason names the markers that decided it.

    Raises:
        TypeError: if ``source_text`` is not a string. Passing a live module
            here is the likely mistake -- use :func:`classify_module`.
    """
    if not isinstance(source_text, str):
        raise TypeError(
            "classify_source() takes source text, got %r; use "
            "classify_module() for a live nn.Module"
            % type(source_text).__name__)
    return _verdict(_scan_text(source_text), len(source_text))


def classify_module(module, max_depth=_DEFAULT_MAX_DEPTH):
    """Judge a live module by its own source and its submodules' class names.

    Args:
        module: anything exposing ``named_modules()`` -- an ``nn.Module`` in
            practice, but the walk is duck-typed so the judgement can be tested
            without torch installed.
        max_depth: how deep into ``named_modules()`` to walk, counting the root
            as depth 0. Deeper costs time and adds no markers.

    Returns:
        ``(exportability, reason)``, same contract as :func:`classify_source`.

    Raises:
        TypeError: if ``module`` has no callable ``named_modules()``.
        ValueError: if ``max_depth`` is negative.
    """
    text = _module_text(module, max_depth)
    return _verdict(_scan_text(text), len(text))


def scan(module_or_source):
    """Every marker found, so a human sees the whole hole and not its first inch.

    Args:
        module_or_source: source text, or a live module with
            ``named_modules()``.

    Returns:
        A list of ``{'marker', 'kind', 'why'}`` dicts, hostile findings first
        and each kind in the declaration order of its marker table. Empty when
        nothing matched.

    Raises:
        TypeError: if the argument is neither text nor a module-like object.
    """
    if isinstance(module_or_source, str):
        return _scan_text(module_or_source)
    return _scan_text(_module_text(module_or_source, _DEFAULT_MAX_DEPTH))


def patch_plan(findings):
    """Turn patchable findings into ordered, human-readable patch steps.

    Hostile findings are dropped rather than reported as work: no step in this
    list makes them exportable, and mixing them in would suggest otherwise.

    Args:
        findings: the list returned by :func:`scan` (or any subset of it).

    Returns:
        Numbered patch steps in :data:`_PATCH_ORDER` order, one per distinct
        patchable marker. Empty when there is nothing to patch.

    Raises:
        ValueError: on a malformed finding, an unrecognised ``kind``, or a
            marker that is not in :data:`PATCHABLE_MARKERS` -- a caller that
            invented a marker gets told, not quietly ignored.
    """
    markers = []
    for finding in findings:
        marker = _patchable_marker(finding)
        if marker is not None and marker not in markers:
            markers.append(marker)
    markers.sort(key=_PATCH_ORDER.index)
    return ["%d. [%s] %s" % (i, m, PATCHABLE_MARKERS[m])
            for i, m in enumerate(markers, 1)]


def _patchable_marker(finding):
    """Validate one finding; return its marker if patchable, else None."""
    if not isinstance(finding, dict) or "marker" not in finding \
            or "kind" not in finding:
        raise ValueError(
            "each finding must be a dict with 'marker' and 'kind' keys, got %r"
            % (finding,))
    kind = finding["kind"]
    if kind not in (_KIND_HOSTILE, _KIND_PATCHABLE):
        raise ValueError("unknown finding kind %r" % (kind,))
    if kind == _KIND_HOSTILE:
        return None
    marker = finding["marker"]
    if marker not in PATCHABLE_MARKERS:
        raise ValueError(
            "unknown patchable marker %r; it has no patch to apply"
            % (marker,))
    return marker


def _module_text(module, max_depth):
    """The searchable text of a live module: class names plus own source.

    One ``name -> ClassName`` line per visited submodule, plus the class source
    of every distinct non-torch class among them.
    """
    parts = []
    seen = set()
    for name, sub in _iter_named_modules(module, max_depth):
        cls = type(sub)
        parts.append("# %s -> %s" % (name or "<root>", cls.__name__))
        key = (getattr(cls, "__module__", ""), getattr(cls, "__qualname__", ""))
        if key in seen:
            continue
        seen.add(key)
        if _is_vendor_class(cls):
            continue
        source = _class_source(cls)
        if source:
            parts.append(source)
    return "\n".join(parts)


def _iter_named_modules(module, max_depth):
    """``(name, submodule)`` pairs no deeper than ``max_depth``.

    Raises:
        TypeError: if ``module`` exposes no callable ``named_modules()``.
        ValueError: if ``max_depth`` is negative.
    """
    if int(max_depth) < 0:
        raise ValueError("max_depth must be >= 0, got %r" % (max_depth,))
    named = getattr(module, "named_modules", None)
    if not callable(named):
        raise TypeError(
            "classify_module()/scan() need an object with named_modules(), got "
            "%r; pass source text to classify_source() instead"
            % type(module).__name__)
    out = []
    for name, sub in named():
        depth = 0 if not name else name.count(".") + 1
        if depth <= int(max_depth):
            out.append((name, sub))
    return out


def _class_source(cls):
    """Source of one class, or None when it cannot be read.

    Best-effort by necessity, and the one place this module tolerates a miss:
    classes built at runtime, defined in a REPL, or shipped only as bytecode
    have no retrievable source. A miss narrows the evidence -- it never turns
    into a verdict on its own, because the class *name* was already recorded.
    """
    try:
        return inspect.getsource(cls)
    except (OSError, TypeError, IndentationError):
        return None


_TORCH_ROOT_PROBED = False
_TORCH_ROOT = None


def _torch_root():
    """Directory of the installed torch package, or None if torch is absent.

    torch is imported lazily and exactly once: this module must stay usable in
    the pure-numpy venv, where importing torch would be an ImportError rather
    than an answer.
    """
    global _TORCH_ROOT_PROBED, _TORCH_ROOT
    if not _TORCH_ROOT_PROBED:
        _TORCH_ROOT_PROBED = True
        try:
            import torch
            torch_file = getattr(torch, "__file__", None)
            if torch_file:
                _TORCH_ROOT = str(Path(torch_file).resolve().parent)
        except Exception:  # pragma: no cover - probe, absence is the answer
            _TORCH_ROOT = None
    return _TORCH_ROOT


def _is_vendor_class(cls):
    """True for classes defined inside torch itself.

    Reading ``torch.nn.modules.activation`` would report every fast-path branch
    torch has, for every model, which says nothing about the model. Stock torch
    layers are judged by class name instead.
    """
    module_name = getattr(cls, "__module__", "") or ""
    if module_name == "torch" or module_name.startswith("torch."):
        return True
    root = _torch_root()
    if not root:
        return False
    source_file = getattr(inspect.getmodule(cls), "__file__", None)
    return bool(source_file) and str(Path(source_file)).startswith(root)
