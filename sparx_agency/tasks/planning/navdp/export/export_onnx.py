"""Export the three NavDP point-goal graphs to static-shape FP32 ONNX.

Builds the trained policy, wraps the encoder / denoiser / critic, and exports
each to ONNX with the settings that make a DINOv2 + nn.TransformerDecoder graph
trace cleanly for TensorRT 10.x:

  * TorchScript exporter (``dynamo=False``) -- far better trodden for this code
    than the dynamo exporter (asserts, shape arithmetic, classic MHA).
  * ``opset_version=17``, ``do_constant_folding=True`` -- decomposes SDPA into
    MatMul/Softmax/Add and folds the pos-embed/mask/arange constants.
  * SDPA forced to the MATH backend and ``nn.MultiheadAttention`` fast-path
    disabled, so attention traces to plain matmul/softmax (no fused op leaks).

A hard op-type gate then fails the export if a forbidden node survives -- in
particular ``Resize`` (the pos-embed bicubic that must have been pre-baked) or
any fused ``*Attention`` op. FP32 is exported; TensorRT applies FP16/INT8 later.
Writes ``navdp_head_params.npz`` and a ``manifest.json`` (io specs, opset, ckpt
sha256) alongside the ``.onnx`` files.

Run (navdp conda env, with ``onnx`` installed; ``PYTHONPATH`` = repo root):
    python -m sparx_agency.tasks.planning.navdp.export.export_onnx \
        --ckpt ~/PycharmProjects/NavDP/baselines/navdp/checkpoints/navdp-cross-modal.ckpt \
        --navdp-repo ~/PycharmProjects/NavDP/baselines/navdp \
        --out-dir sparx_agency/tasks/planning/navdp/engines/onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from sparx_agency.tasks.planning.navdp.export import io_spec
from sparx_agency.tasks.planning.navdp.export.build_policy import (
    build_navdp_policy, dump_head_params,
)
from sparx_agency.tasks.planning.navdp.export.wrappers import (
    CriticWrapper, DenoiseStepWrapper, EncoderWrapper,
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


def build_wrappers(policy):
    """Return ``[(engine_key, module)]`` for the three exportable graphs."""
    return [
        (io_spec.ENCODER, EncoderWrapper(policy.rgbd_encoder)),
        (io_spec.DENOISE, DenoiseStepWrapper(policy)),
        (io_spec.CRITIC, CriticWrapper(policy)),
    ]


def _dummy_inputs(engine_key):
    """Random fp32 example inputs (values irrelevant for static tracing)."""
    sh = io_spec.shapes(engine_key)
    return tuple(torch.randn(*sh[name]) for name in io_spec.input_names(engine_key))


def export_one(engine_key, module, out_dir):
    """Export one wrapper to ONNX and run the op-type gate. Returns the path."""
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
            "%s.onnx contains forbidden ops %r -- the pos-embed bicubic must be "
            "pre-baked and attention must decompose to matmul/softmax."
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
    ap.add_argument("--navdp-repo", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = build_navdp_policy(args.ckpt, navdp_repo=args.navdp_repo, device="cpu")
    npz = dump_head_params(policy, out_dir / "navdp_head_params.npz")

    manifest = {"opset": OPSET, "ckpt_sha256": _sha256(args.ckpt),
                "head_params": npz.name, "engines": {}}
    for key, module in build_wrappers(policy):
        path = export_one(key, module, out_dir)
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
