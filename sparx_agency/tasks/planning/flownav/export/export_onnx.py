"""Export the three FlowNav graphs to static-shape FP32 ONNX.

Builds the trained model, wraps the encoder / velocity-field / distance head, and
exports each to ONNX with the settings that make an EfficientNet + DINOv2 +
ConditionalUnet1D graph trace cleanly for TensorRT:

  * TorchScript exporter (``dynamo=False``), ``opset_version=17``,
    ``do_constant_folding=True``.
  * SDPA forced to the MATH backend and ``nn.MultiheadAttention`` fast-path
    disabled, so the 4-layer self-attention traces to plain matmul/softmax.
  * The wrappers pre-bake the DINOv2 positional embedding (no ``Resize``), switch
    EfficientNet to exportable swish, and bake the navigation goal-mask.

A hard op-type gate then fails the export if a forbidden node survives -- in
particular ``Resize`` (pos-embed not pre-baked) or any fused ``*Attention`` op.
FP32 is exported; TensorRT applies FP16 later. Writes ``flownav_head_params.npz``
and a ``manifest.json`` (io specs, opset, ckpt sha256) alongside the ``.onnx``.

Run (FlowNav build env; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.flownav.export.export_onnx \
        --ckpt ~/PycharmProjects/flownav/flownav/checkpoints/flownav_weights.pth \
        --flownav-repo ~/PycharmProjects/flownav \
        --out-dir sparx_agency/tasks/planning/flownav/engines/onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from sparx_agency.core.planning.flownav.trt.policy import VF_IN_TIME
from sparx_agency.tasks.planning.flownav.export import io_spec
from sparx_agency.tasks.planning.flownav.export.build_model import (
    build_flownav_model, dump_head_params, resolve_flownav_repo,
)
from sparx_agency.tasks.planning.flownav.export.wrappers import (
    DistWrapper, EncoderWrapper, VFieldWrapper,
)

OPSET = 17
# Op types that must NOT survive a clean static export.
FORBIDDEN_OPS = {"Resize"}
FORBIDDEN_SUFFIX = "Attention"
# Data-dependent ops worth warning about (shouldn't appear, but flag if they do).
SUSPECT_OPS = {"If", "Loop", "NonZero", "Scan"}


def _sdpa_math():
    """Context manager forcing the SDPA math backend (version-robust)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    try:
        return sdpa_kernel(SDPBackend.MATH)
    except TypeError:
        return sdpa_kernel([SDPBackend.MATH])


def _disable_mha_fastpath():
    """Best-effort disable of nn.MultiheadAttention's fused fast path."""
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:  # noqa: BLE001 (older torch: no such knob)
        pass


def build_wrappers(model):
    """Return ``[(engine_key, module)]`` for the three exportable graphs."""
    return [
        (io_spec.ENCODER, EncoderWrapper(model.vision_encoder)),
        (io_spec.VFIELD, VFieldWrapper(model.noise_pred_net)),
        (io_spec.DIST, DistWrapper(model.dist_pred_net)),
    ]


def _dummy_inputs(engine_key):
    """Example inputs (values irrelevant for static tracing).

    ``timestep`` is fixed to a mid-interval value in [0,1] for realism; all other
    inputs are random (the graph structure does not depend on the values).
    """
    sh = io_spec.shapes(engine_key)
    inputs = []
    for name in io_spec.input_names(engine_key):
        if name == VF_IN_TIME:
            inputs.append(torch.full(sh[name], 0.5, dtype=torch.float32))
        else:
            inputs.append(torch.randn(*sh[name]))
    return tuple(inputs)


def export_one(engine_key, module, out_dir, slim=True):
    """Export one wrapper to ONNX and run the op-type gate. Returns the path.

    ``slim`` enables the optional ``onnxslim`` graph simplification. Disable it on
    Jetson/aarch64: onnxslim invokes onnxruntime, whose CPU-feature detection
    SIGABRTs there ("Unknown CPU vendor"), and a native abort cannot be caught.
    """
    out_path = Path(out_dir) / (engine_key + ".onnx")
    inputs = io_spec.input_names(engine_key)
    outputs = io_spec.output_names(engine_key)
    module.eval()
    _disable_mha_fastpath()
    with torch.no_grad(), _sdpa_math():
        try:
            torch.onnx.export(
                module, _dummy_inputs(engine_key), str(out_path),
                input_names=inputs, output_names=outputs, opset_version=OPSET,
                do_constant_folding=True, dynamo=False)
        except TypeError:        # torch without the dynamo kwarg (TS is default)
            torch.onnx.export(
                module, _dummy_inputs(engine_key), str(out_path),
                input_names=inputs, output_names=outputs, opset_version=OPSET,
                do_constant_folding=True)
    _gate_ops(engine_key, out_path)
    if slim:
        _maybe_slim(out_path)
    return out_path


def _gate_ops(engine_key, onnx_path):
    """Fail loud if a forbidden op survived; warn on suspect ops."""
    import onnx
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    ops = [n.op_type for n in model.graph.node]
    bad = [o for o in ops if o in FORBIDDEN_OPS or o.endswith(FORBIDDEN_SUFFIX)]
    if bad:
        raise RuntimeError(
            "%s.onnx contains forbidden ops %r -- the DINOv2 pos-embed bicubic "
            "must be pre-baked and attention must decompose to matmul/softmax."
            % (engine_key, sorted(set(bad))))
    suspect = sorted({o for o in ops if o in SUSPECT_OPS})
    if suspect:
        print("[warn] %s.onnx has data-dependent ops %r; verify parity carefully"
              % (engine_key, suspect))


def _maybe_slim(onnx_path):
    """Simplify the graph with onnxslim if it is installed (optional)."""
    try:
        import onnxslim
    except ImportError:
        return
    onnxslim.slim(str(onnx_path), str(onnx_path))


def _sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--flownav-repo", default=None)
    ap.add_argument("--config", default=None, help="model config yaml (default: <repo>/flownav/config/flownav.yaml)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-slim", action="store_true",
                    help="skip the optional onnxslim pass (required on Jetson/aarch64)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = resolve_flownav_repo(args.flownav_repo)

    model = build_flownav_model(args.ckpt, flownav_repo=repo, config_path=args.config,
                                device="cpu")
    npz = dump_head_params(repo, out_dir / "flownav_head_params.npz")

    manifest = {"opset": OPSET, "ckpt_sha256": _sha256(args.ckpt),
                "head_params": npz.name, "num_samples": io_spec.N,
                "horizon": io_spec.HORIZON, "action_dim": io_spec.ACT_DIM,
                "engines": {}}
    for key, module in build_wrappers(model):
        path = export_one(key, module, out_dir, slim=not args.no_slim)
        manifest["engines"][key] = {
            "onnx": path.name, "onnx_sha256": _sha256(path),
            "inputs": io_spec.input_names(key), "outputs": io_spec.output_names(key),
            "shapes": {k: list(v) for k, v in io_spec.shapes(key).items()},
        }
        print("[ok] exported", path.name)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[done] manifest:", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
