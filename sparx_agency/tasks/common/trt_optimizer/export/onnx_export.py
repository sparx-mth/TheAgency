"""Export a :class:`GraphSpec` to a static-shape FP32 ONNX, and prove it is clean.

FP32 is exported deliberately. Precision is applied later, per target, by
:mod:`..engine.precision` -- the ONNX is the *portable* artifact and the engine
is the per-device one, so baking FP16 into the export would force a re-export
for every device that wanted something else.

Two settings here are not stylistic and should not be changed casually:

``dynamo=False``
    torch 2.9+ made the ``torch.export``-based exporter the default. It produces
    a different graph for the same model, and this pipeline's op gate, parity
    tolerances and TensorRT parser behaviour are all calibrated against the
    TorchScript exporter. Pinning it means a torch upgrade cannot silently swap
    the graph-producing backend underneath a working build.

``opset_version=17``
    17 is the floor, not a preference: it is the first opset that emits a single
    ``LayerNormalization`` node instead of a decomposed
    ReduceMean/Sub/Pow/Div chain. TensorRT maps that node onto its own
    normalization layer, and doing so alone fixes most transformer FP16 accuracy
    regressions -- a much cheaper fix than pinning layers by hand afterwards.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sparx_agency.tasks.common.trt_optimizer.export import op_gate, patches

#: The minimum ONNX opset this pipeline will export. See the module docstring.
MIN_OPSET = 17


def sha256(path):
    """Hex SHA-256 of a file, used to tie an engine back to its exact graph."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def example_inputs(spec, seed=0, device="cpu"):
    """Deterministic dummy inputs matching a spec's static shapes.

    Values are irrelevant to a static trace, but determinism is not: the same
    seed must produce the same graph so an ONNX hash is meaningful.

    Args:
        spec: the :class:`..spec.GraphSpec` being exported.
        seed: RNG seed.
        device: torch device string.

    Returns:
        A tuple of tensors in the spec's declared input order.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shapes = []
    for name in spec.input_names():
        profile = spec.profiles.get(name)
        # Trace at ``opt``: it is a legal concrete shape, and it is the one
        # TensorRT will tune tactics for.
        shapes.append(profile.opt if profile is not None else spec.inputs[name])
    return tuple(
        torch.randn(*shape, generator=generator).to(device) for shape in shapes
    )


def export_graph(spec, module, out_dir, inputs=None, policy=None, slim=True,
                 seed=0):
    """Export one graph to ONNX, gate it, and return the written path.

    Args:
        spec: the :class:`..spec.GraphSpec`; supplies the engine key, the input
            and output tensor names, and the static shapes.
        module: an ``nn.Module`` whose ``forward`` takes the spec's inputs in
            order. This is the *export wrapper*, not the original model.
        out_dir: directory to write ``<key>.onnx`` into.
        inputs: example inputs; generated from the spec when omitted.
        policy: an :class:`..export.op_gate.OpGatePolicy`; use
            :func:`..export.op_gate.vit_policy` for a vision transformer.
        slim: run the optional ``onnxslim`` simplification pass. Must be False
            on Jetson/aarch64 -- onnxslim invokes onnxruntime, whose CPU-feature
            detection aborts there, and a native abort cannot be caught.
        seed: seed for generated example inputs.

    Returns:
        Path to the written ``.onnx``.

    Raises:
        ValueError: if the spec is not fully static, or the opset is too low.
        RuntimeError: if the op gate rejects the exported graph.
    """
    spec.validate()
    if spec.opset < MIN_OPSET:
        raise ValueError(
            "GraphSpec %r asks for opset %d; %d is the floor because it is the "
            "first opset emitting a single LayerNormalization node, which is "
            "the cheapest fix for transformer FP16 drift."
            % (spec.key, spec.opset, MIN_OPSET))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (spec.key + ".onnx")
    args = inputs if inputs is not None else example_inputs(spec, seed=seed)

    with patches.export_context(module):
        _export(module, args, out_path, spec)

    # Simplify BEFORE gating, so the gate validates the artifact that actually
    # ships rather than an intermediate the builder will never see.
    if slim:
        _maybe_slim(out_path)
    if policy is None and spec.is_dynamic:
        policy = op_gate.dynamic_policy()
    op_gate.enforce(out_path, policy=policy, key=spec.key)
    return out_path


def _export(module, args, out_path, spec):
    """Call torch.onnx.export with the pinned exporter and settings."""
    import torch

    kwargs = dict(input_names=spec.input_names(), output_names=list(spec.outputs),
                  opset_version=spec.opset, do_constant_folding=True)
    axes = spec.dynamic_axes()
    if axes:
        # torch wants {tensor: {axis: label}}. The label only has to be stable
        # within the graph; TensorRT reads the range from the build profile, not
        # from these names.
        kwargs["dynamic_axes"] = dict(
            (name, dict((axis, "%s_%d" % (name, axis)) for axis in axis_list))
            for name, axis_list in axes.items())
    try:
        torch.onnx.export(module, args, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        # torch old enough that TorchScript is the only exporter and there is no
        # dynamo kwarg to pin. Same graph, so nothing else changes.
        torch.onnx.export(module, args, str(out_path), **kwargs)


def _maybe_slim(onnx_path):
    """Simplify the graph with onnxslim when it is installed."""
    try:
        import onnxslim
    except ImportError:
        return
    onnxslim.slim(str(onnx_path), str(onnx_path))



def count_parameters(module):
    """Parameters reachable from ``module``, counted once each.

    Deduplicated by tensor identity: an export wrapper commonly holds the same
    submodule the model does, and a tied embedding appears under two names. The
    number feeds :func:`..engine.precision.verify_engine_precision`, which
    divides engine weight bytes by it -- so a double count silently halves the
    measured bytes-per-element and turns a failed precision check into a passing
    one.

    Args:
        module: an ``nn.Module``, or anything without ``parameters()``.

    Returns:
        int: the parameter count, or 0 when ``module`` exposes no parameters.
    """
    parameters = getattr(module, "parameters", None)
    if parameters is None:
        return 0
    seen = {}
    for tensor in parameters():
        seen[id(tensor)] = int(tensor.numel())
    return int(sum(seen.values()))


def export_all(specs, modules, out_dir, policy_for=None, slim=True, seed=0,
               extra=None):
    """Export every graph and write the portable manifest beside them.

    Args:
        specs: the :class:`..spec.GraphSpec` objects to export.
        modules: mapping of engine key -> export wrapper module.
        out_dir: the ``engines/onnx`` directory.
        policy_for: optional callable ``key -> OpGatePolicy``.
        slim: see :func:`export_graph`.
        seed: seed for example inputs.
        extra: extra keys folded into the manifest (checkpoint hash, model name).

    Returns:
        The manifest dict, also written to ``<out_dir>/manifest.json``.
        ``opset_floor`` is the pipeline-wide minimum (:data:`MIN_OPSET`); the
        opset each graph was *actually* exported at is recorded per graph,
        because a spec may legitimately ask for a higher one and a single
        top-level number would then misreport every file beside it.

    Raises:
        KeyError: if a spec has no matching module.
    """
    out_dir = Path(out_dir)
    # Created here rather than only inside export_graph: an empty or fully
    # filtered spec list still has to leave a readable manifest behind.
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"opset_floor": MIN_OPSET, "graphs": {}}
    manifest.update(extra or {})
    for spec in specs:
        if spec.key not in modules:
            raise KeyError("no export wrapper supplied for graph %r" % spec.key)
        policy = policy_for(spec.key) if policy_for else None
        if policy is None and spec.is_dynamic:
            policy = op_gate.dynamic_policy()
        path = export_graph(spec, modules[spec.key], out_dir, policy=policy,
                            slim=slim, seed=seed)
        manifest["graphs"][spec.key] = {
            "onnx": path.name,
            "onnx_sha256": sha256(path),
            "inputs": spec.input_names(),
            "outputs": list(spec.outputs),
            "shapes": {k: list(v) for k, v in spec.inputs.items()},
            "opset": spec.opset,
            "cadence": spec.cadence,
            "calls_per_decision": spec.calls_per_decision,
            "precision_sensitive": spec.precision_sensitive,
            # Recorded so the builder can verify afterwards that the precision
            # it asked for survived into the engine. A build that quietly widens
            # a format reports no error and only an unchanged latency.
            "params": count_parameters(modules[spec.key]),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(onnx_dir):
    """Load the manifest written by :func:`export_all`.

    Raises:
        FileNotFoundError: with the export command to run, when it is absent.
    """
    path = Path(onnx_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            "%s has no manifest.json; run the ONNX export stage first (the "
            "engines/onnx directory is build output and is gitignored, so a "
            "fresh clone never has it)." % onnx_dir)
    return json.loads(path.read_text())
