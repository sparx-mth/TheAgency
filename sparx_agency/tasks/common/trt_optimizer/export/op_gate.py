"""Reject an ONNX graph that will disappoint TensorRT, at export time.

An ONNX file that loads fine and even runs under onnxruntime can still be a bad
TensorRT input: a fused attention node the parser cannot decompose, a ``Resize``
left behind by a positional-embedding interpolation that should have been baked,
a data-dependent ``If``/``Loop``, or a dynamic input dimension in a pipeline
whose runtime requires fully static engines.

Catching those here is worth a lot. The alternative is discovering them an hour
later as an opaque builder failure, or -- far worse -- not discovering them at
all because the build quietly succeeded with a slow fallback.

The gate is a policy object rather than a fixed list because "forbidden" is
genuinely model-dependent. ``Resize`` is a bug in a ViT export and completely
normal in a segmentation decoder. What is *never* acceptable is a fused
``*Attention`` node, a non-standard operator domain, or a dynamic dimension in a
graph destined for a static engine -- those are the hard rules.

Imports ``onnx`` lazily, and every function also accepts an already-loaded model
so a caller can gate a graph it built in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence

#: Operator suffix that always means a fused attention kernel leaked into the
#: graph. TensorRT wants MatMul/Softmax/MatMul so that *it* can choose the fused
#: tactic for the target; an opaque node denies it that choice.
FUSED_ATTENTION_SUFFIX = "Attention"

#: Operator domains outside the ONNX standard. The TensorRT parser has no
#: symbolic for these, so their presence is a hard failure rather than a warning.
NONSTANDARD_DOMAINS = ("com.microsoft", "org.pytorch", "ai.onnx.contrib")

#: Ops whose presence means the graph makes a decision at runtime. They are not
#: automatically fatal, but every one of them must be justified before parity is
#: believed, because a traced graph froze whichever branch the example input took.
DATA_DEPENDENT_OPS = frozenset({
    "If", "Loop", "Scan", "NonZero", "NonMaxSuppression", "TopK", "Where",
    "Unique", "Compress",
})


@dataclass
class OpGatePolicy:
    """What this particular graph is and is not allowed to contain.

    Args:
        forbidden: op types that are fatal for this model.
        forbidden_suffixes: op-type suffixes that are fatal.
        suspect: op types that are reported but not fatal.
        forbidden_domains: operator domains that are fatal.
        require_static: fail when any graph input or output has a dynamic
            dimension. True by default because the shared engine runtime in this
            repo rejects dynamic engines outright.
    """

    forbidden: FrozenSet[str] = frozenset()
    forbidden_suffixes: Sequence[str] = (FUSED_ATTENTION_SUFFIX,)
    suspect: FrozenSet[str] = DATA_DEPENDENT_OPS
    forbidden_domains: Sequence[str] = NONSTANDARD_DOMAINS
    require_static: bool = True


def dynamic_policy(base=None):
    """A policy that permits dynamic shapes, for a graph that declares profiles.

    ``require_static`` is on by default because a static engine is faster and
    has no profile-switch cost in its tail. A graph whose
    :class:`..spec.GraphSpec` declares :class:`..spec.ShapeProfile` entries has
    made that trade deliberately, so the gate must not then fail it for the very
    freedom it asked for.
    """
    policy = base or OpGatePolicy()
    return OpGatePolicy(forbidden=policy.forbidden,
                        forbidden_suffixes=policy.forbidden_suffixes,
                        suspect=policy.suspect,
                        forbidden_domains=policy.forbidden_domains,
                        require_static=False)


def vit_policy():
    """The policy for a vision-transformer graph.

    Adds ``Resize`` to the forbidden set: in a ViT it is the bicubic positional
    embedding interpolation, which has exactly one answer at a fixed input size
    and must have been pre-baked (see
    :func:`..export.patches.bake_pos_embed`).
    """
    return OpGatePolicy(forbidden=frozenset({"Resize"}))


@dataclass
class GateResult:
    """What the gate found."""

    key: str
    ok: bool
    op_counts: Dict[str, int] = field(default_factory=dict)
    forbidden_found: List[str] = field(default_factory=list)
    suspect_found: List[str] = field(default_factory=list)
    dynamic_tensors: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    @property
    def node_count(self):
        """Total nodes in the graph."""
        return sum(self.op_counts.values())

    def summary(self):
        """One human line."""
        return ("%s: %d nodes, %d op types, %s"
                % (self.key, self.node_count, len(self.op_counts),
                   "PASS" if self.ok else "FAIL"))


def _load(onnx_path_or_model):
    """Return an onnx ModelProto from a path or pass a model through."""
    if hasattr(onnx_path_or_model, "graph"):
        return onnx_path_or_model
    import onnx
    return onnx.load(str(onnx_path_or_model))


def _dynamic_tensors(graph):
    """Names of graph inputs/outputs carrying a non-fixed dimension."""
    out = []
    for value in list(graph.input) + list(graph.output):
        ttype = getattr(getattr(value, "type", None), "tensor_type", None)
        if ttype is None:
            continue
        for dim in ttype.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                out.append(value.name)
                break
    return out


def gate(onnx_path_or_model, policy=None, key=None, check_model=True):
    """Inspect a graph against ``policy`` and report, without raising.

    Args:
        onnx_path_or_model: path to an ``.onnx`` file, or a loaded ModelProto.
        policy: an :class:`OpGatePolicy`; the default policy when omitted.
        key: name used in messages; derived from the path when omitted.
        check_model: run ``onnx.checker.check_model`` as well. Only applies
            when a *path* was given: an already-loaded object is taken on
            trust, because it may be an in-memory stand-in with no ``onnx``
            behind it. Gating a ModelProto therefore never runs the checker,
            whatever this is set to.

    Returns:
        A :class:`GateResult`.
    """
    policy = policy or OpGatePolicy()
    model = _load(onnx_path_or_model)
    if key is None:
        key = (Path(str(onnx_path_or_model)).stem
               if not hasattr(onnx_path_or_model, "graph") else "graph")

    result = GateResult(key=key, ok=True)
    if check_model and not hasattr(onnx_path_or_model, "graph"):
        import onnx
        onnx.checker.check_model(model)

    for node in model.graph.node:
        result.op_counts[node.op_type] = result.op_counts.get(node.op_type, 0) + 1
        domain = getattr(node, "domain", "") or ""
        if domain and domain in policy.forbidden_domains:
            result.forbidden_found.append("%s (domain %s)" % (node.op_type, domain))
        elif node.op_type in policy.forbidden:
            result.forbidden_found.append(node.op_type)
        elif any(node.op_type.endswith(s) for s in policy.forbidden_suffixes):
            result.forbidden_found.append(node.op_type)
        elif node.op_type in policy.suspect:
            result.suspect_found.append(node.op_type)

    result.forbidden_found = sorted(set(result.forbidden_found))
    result.suspect_found = sorted(set(result.suspect_found))

    if policy.require_static:
        result.dynamic_tensors = sorted(set(_dynamic_tensors(model.graph)))

    if result.forbidden_found:
        result.ok = False
        result.messages.append(
            "forbidden ops survived: %s. A fused *Attention node means the SDPA "
            "math backend / MHA fast-path patch did not apply; a Resize in a ViT "
            "means the positional embedding was not pre-baked."
            % ", ".join(result.forbidden_found))
    if result.dynamic_tensors:
        result.ok = False
        result.messages.append(
            "dynamic shapes on %s -- engines here must be fully static (the "
            "shared runtime rejects a dynamic dimension, and a profile switch "
            "costs tail latency in a control loop)."
            % ", ".join(result.dynamic_tensors))
    if result.suspect_found:
        result.messages.append(
            "data-dependent ops present: %s. Not fatal, but tracing froze "
            "whichever branch the example input took -- verify parity on inputs "
            "that exercise the other branch." % ", ".join(result.suspect_found))
    return result


def enforce(onnx_path_or_model, policy=None, key=None, check_model=True):
    """Gate a graph and raise on failure.

    Returns:
        The passing :class:`GateResult`.

    Raises:
        RuntimeError: with every failure message, when the gate fails.
    """
    result = gate(onnx_path_or_model, policy=policy, key=key,
                  check_model=check_model)
    if not result.ok:
        raise RuntimeError("ONNX op gate failed for %s:\n  - %s"
                           % (result.key, "\n  - ".join(result.messages)))
    return result
