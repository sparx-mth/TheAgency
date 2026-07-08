"""Export the open-set YOLO-World split to two ONNX graphs (backbone + head).

Open-set is preserved: the text embeddings are a **runtime input** to the head,
never baked. This exports:

  * ``yolo_world_<variant>.backbone.onnx`` -- ``image[1,3,H,W] -> feature maps``.
    Fully static and text-free -> the DLA target.
  * ``yolo_world_<variant>.head.onnx`` -- ``(feature maps, txt_feats[...,N,...])
    -> raw detections [1, 4+N, anchors]``. ``N`` (the prompt count) is a **dynamic
    axis**, so any number of prompts runs without a rebuild -> the GPU target.

Before trusting the cut, a **parity gate** runs the library's own
``WorldModel.predict`` and compares it to ``head(backbone(image), txt)`` on random
inputs; a mismatch aborts the export (this is the one thing that can silently
break across ultralytics versions). A ``<variant>.io.json`` sidecar records the
feature-map names/shapes, the txt axis, and the dynamic-N profile bounds that
``build_engine`` and the runtime consume.

Run (any box with ultralytics + torch + onnx; CPU is fine):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.export_onnx \\
        --weights /path/to/yolov8s-worldv2.pt --variant s --imgsz 288x512
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sparx_agency.tasks.mapping.yolo_world_trt import wrappers
from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import (
    load_config, parse_imgsz, variant_weights,
)
from sparx_agency.tasks.mapping.yolo_world_trt.text_embed import TextEmbedder, txt_n_axis

OPSET = 17
# A distinctive example prompt count so the N axis is unambiguous in txt_feats
# (avoid 1 and the 512 embed dim). Only used to trace the head.
_EXAMPLE_N = 7
_EXAMPLE_PROMPTS = ["object%d" % i for i in range(_EXAMPLE_N)]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _onnx_export(module, args, path, input_names, output_names, dynamic_axes=None):
    """torch.onnx.export pinned to the TorchScript exporter (robust across versions).

    ``dynamo=False`` forces the well-trodden TorchScript tracer (the dynamo exporter
    mishandles YOLO's shape arithmetic); it is retried without the kwarg on older
    torch that predates it.
    """
    import torch

    kw = dict(input_names=input_names, output_names=output_names,
              opset_version=OPSET, do_constant_folding=True)
    if dynamic_axes:
        kw["dynamic_axes"] = dynamic_axes
    try:
        torch.onnx.export(module, args, path, dynamo=False, **kw)
    except TypeError:                       # torch too old for the dynamo kwarg
        torch.onnx.export(module, args, path, **kw)


def _set_export_mode(world_model):
    """Put the WorldDetect head in export mode so it returns the raw tensor."""
    detect = world_model.model[-1]
    detect.export = True
    detect.format = "onnx"
    return detect


def _parity(world_model, backbone, head, image, txt, atol=1e-3):
    """Assert head(backbone(image), txt) matches WorldModel.predict(image, txt)."""
    import torch

    with torch.no_grad():
        ref = world_model.predict(image, txt_feats=txt)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        got = head(backbone(image), txt)
    err = (ref - got).abs().max().item()
    if err > atol:
        raise RuntimeError(
            "backbone/head split does NOT match the full model (max abs err %.3e > "
            "%.1e). The ultralytics graph cut is wrong for this version -- inspect "
            "wrappers.find_cut / the layer routing." % (err, atol))
    return err


def export_one(weights, variant, imgsz, out_dir, n_min=1, n_opt=8, n_max=256):
    """Export the backbone + head ONNX for one checkpoint. Returns their paths."""
    import torch

    from ultralytics import YOLOWorld  # lazy: heavy torch dep

    h, w = imgsz
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLOWorld(str(weights))
    world = yolo.model.float().eval()
    _set_export_mode(world)
    backbone, head, out_indices, cut = wrappers.build_split(world)

    # Example inputs: image + the text embeddings for a distinctive prompt count.
    image = torch.zeros(1, 3, h, w, dtype=torch.float32)
    txt_np = TextEmbedder(weights).embed(_EXAMPLE_PROMPTS)
    txt = torch.from_numpy(txt_np).float()
    n_axis = txt_n_axis(tuple(txt.shape), _EXAMPLE_N)
    embed_dim = int(txt.shape[-1])

    with torch.no_grad():
        feats = backbone(image)
    feat_names = ["feat%d" % i for i in range(len(feats))]
    feat_shapes = [list(f.shape) for f in feats]

    err = _parity(world, backbone, head, image, txt)
    print("[parity] split matches full model (max abs err %.2e, cut=%d, "
          "backbone outs=%s)" % (err, cut, out_indices))

    backbone_path = out_dir / ("yolo_world_%s.backbone.onnx" % variant)
    head_path = out_dir / ("yolo_world_%s.head.onnx" % variant)

    # Backbone: fully static (no dynamic_axes).
    _onnx_export(
        backbone, (image,), str(backbone_path),
        input_names=["image"], output_names=feat_names)

    # Head: N (prompt count) is dynamic on the txt input and the output class dim.
    _onnx_export(
        head, (tuple(feats), txt), str(head_path),
        input_names=feat_names + ["txt_feats"], output_names=["output"],
        dynamic_axes={"txt_feats": {n_axis: "N"}, "output": {1: "C"}})

    with torch.no_grad():
        raw = head(tuple(feats), txt)
    io = {
        "variant": variant,
        "weights": str(weights),
        "weights_sha256": _sha256(weights),
        "imgsz_hw": [h, w],
        "opset": OPSET,
        "embed_dim": embed_dim,
        "txt_n_axis": n_axis,
        "txt_example_shape": list(txt.shape),
        "cut_index": cut,
        "backbone": {
            "input": "image",
            "feat_names": feat_names,
            "feat_shapes": feat_shapes,
            "onnx_sha256": _sha256(backbone_path),
        },
        "head": {
            "feat_inputs": feat_names,
            "txt_input": "txt_feats",
            "output": "output",
            "output_example_shape": list(raw.shape),
            "n_min": n_min, "n_opt": n_opt, "n_max": n_max,
            "onnx_sha256": _sha256(head_path),
        },
    }
    (out_dir / ("yolo_world_%s.io.json" % variant)).write_text(json.dumps(io, indent=2))
    return backbone_path, head_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None,
                    help="path to the .pt checkpoint (default: config filename)")
    ap.add_argument("--variant", required=True, choices=["s", "m", "l", "x"])
    ap.add_argument("--imgsz", default=None, help="HxW (default: config imgsz)")
    ap.add_argument("--n-max", type=int, default=None,
                    help="max prompt count the head profile allows (default: config)")
    ap.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parent / "engines" / "onnx"))
    args = ap.parse_args()

    cfg = load_config()
    weights = args.weights or variant_weights(cfg)[args.variant]
    imgsz = parse_imgsz(args.imgsz if args.imgsz else cfg.get("imgsz", "288x512"))
    prof = cfg.get("head_prompt_profile", {})
    n_min = int(prof.get("n_min", 1))
    n_opt = int(prof.get("n_opt", 8))
    n_max = int(args.n_max if args.n_max else prof.get("n_max", 256))

    print("[export] %s  weights=%s  imgsz=%dx%d  dynamic-N in [%d,%d,%d]"
          % (args.variant, weights, imgsz[0], imgsz[1], n_min, n_opt, n_max))
    bpath, hpath = export_one(weights, args.variant, imgsz, args.out_dir,
                              n_min=n_min, n_opt=n_opt, n_max=n_max)
    print("[ok] backbone ->", bpath.name)
    print("[ok] head     ->", hpath.name)


if __name__ == "__main__":
    main()
