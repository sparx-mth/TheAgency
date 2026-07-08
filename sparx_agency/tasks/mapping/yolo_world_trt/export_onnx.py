"""Export an ultralytics YOLO-World checkpoint to a static-shape ONNX graph.

The open-vocabulary class list is **baked in** at export time via
:meth:`YOLOWorld.set_classes`: this freezes the CLIP text embeddings into the
head, so the exported graph is a pure-vision CNN (backbone + RepVL-PAN neck +
detection head) with a *fixed* ``nc = len(prompts)``. That is the whole reason a
YOLO-World engine can run efficiently on TensorRT (and on the Orin DLA): the heavy
text transformer never enters the graph.

Trade-off you are opting into: after baking, the engine detects ONLY the baked
prompts. Re-prompting to a class outside that set means re-exporting + re-building.
So bake the *full mission vocabulary* you expect to target (e.g. every object the
mission might approach), not just one class -- ``--prompts`` takes a list.

The export is static-shape (batch 1, fixed ``HxW``) and NMS-free (``nms=False``):
the raw head output ``[1, 4 + nc, num_anchors]`` is decoded + NMS'd at runtime in
:mod:`postprocess` (torch-free numpy) so the engine stays a clean CNN. A sidecar
``<stem>.classes.json`` records the baked prompt order so the runtime can map a
class index back to its label.

Run (any box with ultralytics + torch + onnx; a GPU is NOT required to export):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.export_onnx \\
        --weights /path/to/yolov8s-worldv2.pt \\
        --variant s --imgsz 288x512 \\
        --prompts refrigerator chair door person \\
        --out-dir sparx_agency/tasks/mapping/yolo_world_trt/engines/onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import (
    load_config, parse_imgsz, variant_weights,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_one(weights, variant, prompts, imgsz, out_dir, opset=17, simplify=True):
    """Export one checkpoint to ``<out_dir>/yolo_world_<variant>.onnx``.

    Args:
        weights: path to the ``.pt`` YOLO-World checkpoint.
        variant: size tag (``s``/``m``/``l``/``x``) used to name the output.
        prompts: baked open-vocabulary class list (>= 1 non-empty string).
        imgsz: ``(H, W)`` engine input, stride-32 multiples.
        out_dir: directory to write the ONNX + sidecar into.
        opset: ONNX opset (17 traces YOLOv8 cleanly for TensorRT 10.x).
        simplify: run onnx-simplifier to fold shape math (DLA-friendlier graph).

    Returns:
        Path to the written ``.onnx``.
    """
    from ultralytics import YOLOWorld           # lazy: heavy torch dep

    prompts = [str(p).strip() for p in prompts if str(p).strip()]
    if not prompts:
        raise ValueError("export needs at least one non-empty prompt to bake.")
    h, w = imgsz
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOWorld(str(weights))
    model.set_classes(prompts)                  # bakes text embeddings into the head

    # Ultralytics writes <weights_stem>.onnx next to the source; export on CPU so
    # this step is portable (no GPU needed) and deterministic.
    exported = model.export(
        format="onnx",
        imgsz=(h, w),
        opset=opset,
        simplify=simplify,
        dynamic=False,                          # static shape -> DLA + no opt profile
        nms=False,                              # decode + NMS live in postprocess.py
        half=False,                             # FP32 ONNX; TRT applies FP16/INT8
        device="cpu",
    )

    dst = out_dir / ("yolo_world_%s.onnx" % variant)
    src = Path(exported)
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())

    sidecar = {
        "variant": variant,
        "weights": str(weights),
        "weights_sha256": _sha256(weights),
        "prompts": prompts,
        "nc": len(prompts),
        "imgsz_hw": [h, w],
        "opset": opset,
        "simplify": bool(simplify),
        "onnx_sha256": _sha256(dst),
        "nms": False,
        "layout": "NCHW",
    }
    (out_dir / ("yolo_world_%s.classes.json" % variant)).write_text(
        json.dumps(sidecar, indent=2))
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None,
                    help="path to the .pt checkpoint (default: config filename)")
    ap.add_argument("--variant", required=True, choices=["s", "m", "l", "x"])
    ap.add_argument("--prompts", nargs="+", required=True,
                    help="baked open-vocab class list, e.g. refrigerator chair door")
    ap.add_argument("--imgsz", default=None, help="HxW (default: config imgsz)")
    ap.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parent / "engines" / "onnx"))
    args = ap.parse_args()

    cfg = load_config()
    weights = args.weights or variant_weights(cfg)[args.variant]
    imgsz = parse_imgsz(args.imgsz if args.imgsz else cfg.get("imgsz", "288x512"))
    opset = int(cfg.get("opset", 17))
    simplify = bool(cfg.get("simplify", True))

    print("[export] %s  weights=%s  imgsz=%dx%d  prompts=%s"
          % (args.variant, weights, imgsz[0], imgsz[1], args.prompts))
    path = export_one(weights, args.variant, args.prompts, imgsz, args.out_dir,
                      opset=opset, simplify=simplify)
    print("[ok] wrote", path)


if __name__ == "__main__":
    main()
