"""Compare PyTorch YOLO-World against *every* TRT variant (s/m/l/x) in one run.

This is the N-way generalisation of :mod:`compare_folder` (which is fixed to one
torch model vs one TRT engine). It runs the same open-vocab prompts over the same
folder of frames through the PyTorch baseline and each TRT engine you pass, and
emits, for a single command:

  1. **Identifications for each model** -- ``<out>/pytorch/`` and ``<out>/trt_<label>/``
     each get annotated ``.jpg``s, a ``detections.csv`` and a ``detections.jsonl``
     (same layout as :mod:`detect_folder`).
  2. **Speed of each model** -- one latency / FPS table across PyTorch + all TRT
     variants, ranked, plus each variant's speed-up over PyTorch.
  3. **How much accuracy each TRT engine costs vs PyTorch** -- per frame the two
     detection sets are matched by class + box IoU, and the run reports, per
     variant, matched / missed-by-TRT / extra-in-TRT counts, recall vs PyTorch,
     mean IoU of matches, and mean confidence drift.

To bound GPU memory on the Orin, engines are run **one at a time**: the PyTorch
pass runs first and its detections are held in memory, then each TRT engine is
loaded, run, compared to the stored PyTorch detections, and released before the
next. PyTorch runs at a single input size (the first engine's, or ``--imgsz``);
each TRT engine runs at its own built size -- so it is the honest "is there a
drop?" check. The single ``--torch-weights`` doubles as the frozen CLIP text
encoder for every engine (YOLO-World v2 shares it across sizes).

Run (target Orin, venv with ultralytics+torch AND tensorrt+pycuda, PYTHONPATH=repo):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.compare_all \\
        --torch-weights /path/to/yolov8s-worldv2.pt \\
        --pair s:.../orin_sm87/yolo_world_s.backbone.fp16.dla0.engine,.../orin_sm87/yolo_world_s.head.fp16.gpu.engine \\
        --pair m:.../orin_sm87/yolo_world_m.backbone.fp16.dla0.engine,.../orin_sm87/yolo_world_m.head.fp16.gpu.engine \\
        --pair l:.../orin_sm87/yolo_world_l.backbone.fp16.dla0.engine,.../orin_sm87/yolo_world_l.head.fp16.gpu.engine \\
        --pair x:.../orin_sm87/yolo_world_x.backbone.fp16.dla0.engine,.../orin_sm87/yolo_world_x.head.fp16.gpu.engine \\
        --images /path/to/rgb --out /path/to/identifications \\
        --labels "chair, bottle, table" --conf 0.4
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np

from sparx_agency.tasks.mapping.yolo_world_trt.benchmark import (
    Stat, find_images, parse_pair, stat)
from sparx_agency.tasks.mapping.yolo_world_trt.compare_folder import _Writer, match
from sparx_agency.tasks.mapping.yolo_world_trt.compare_torch_vs_trt import TorchYoloWorld
from sparx_agency.tasks.mapping.yolo_world_trt.detect_folder import parse_labels
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector


def detect_pass(detect_fn: Callable[[np.ndarray], list], files: List[str],
                warmup: int, writer: _Writer, labels: List[str]) -> tuple:
    """Run ``detect_fn`` over every frame; time each call and stream identifications.

    Returns ``(Stat, dets_by_name)`` where ``dets_by_name`` maps each frame's
    filename to its list of detections (for later cross-model matching). Only the
    ``detect_fn`` call is timed; image I/O and drawing are excluded.
    """
    warm = cv2.cvtColor(cv2.imread(files[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    for _ in range(max(0, warmup)):
        detect_fn(warm)

    ms: List[float] = []
    dets_by_name: Dict[str, list] = {}
    for path in files:
        name = Path(path).name
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("  skipped unreadable image:", name)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        dets = detect_fn(rgb)
        ms.append((time.perf_counter() - t0) * 1000.0)
        writer.add(name, bgr, dets, labels)
        dets_by_name[name] = dets
    return stat(ms), dets_by_name


def agreement(ref_by_name: Dict[str, list], trt_by_name: Dict[str, list],
              labels: List[str], match_iou: float) -> dict:
    """Match each frame's TRT detections against the PyTorch baseline detections.

    ``ref`` is the PyTorch (baseline) set, ``trt`` is the variant under test.
    Returns aggregate counts, mean matched IoU, mean absolute confidence delta,
    and a per-class ``{pytorch, tensorrt, matched}`` breakdown.
    """
    agg = {"pt": 0, "trt": 0, "matched": 0, "pt_only": 0, "trt_only": 0,
           "iou_sum": 0.0, "score_abs_sum": 0.0}
    per_class = {name.strip().lower(): {"pytorch": 0, "tensorrt": 0, "matched": 0}
                 for name in labels}
    for name, trt_dets in trt_by_name.items():
        pt_dets = ref_by_name.get(name, [])
        matched, pt_only, trt_only = match(pt_dets, trt_dets, match_iou)
        agg["pt"] += len(pt_dets)
        agg["trt"] += len(trt_dets)
        agg["matched"] += len(matched)
        agg["pt_only"] += len(pt_only)
        agg["trt_only"] += len(trt_only)
        for i, j, v in matched:
            agg["iou_sum"] += v
            agg["score_abs_sum"] += abs(pt_dets[i].score - trt_dets[j].score)
            if pt_dets[i].label in per_class:
                per_class[pt_dets[i].label]["matched"] += 1
        for d in pt_dets:
            if d.label in per_class:
                per_class[d.label]["pytorch"] += 1
        for d in trt_dets:
            if d.label in per_class:
                per_class[d.label]["tensorrt"] += 1
    m = max(1, agg["matched"])
    agg["mean_matched_iou"] = agg["iou_sum"] / m
    agg["mean_abs_conf_delta"] = agg["score_abs_sum"] / m
    agg["recall_vs_pytorch"] = (agg["matched"] / agg["pt"]) if agg["pt"] else None
    agg["per_class"] = per_class
    return agg


def print_speed(rows: List[tuple], pt_stat: Stat) -> None:
    """Print one ranked latency/FPS table over PyTorch + every TRT variant."""
    print("\n%s\nSPEED (per-frame detect(), ranked by FPS)\n%s" % ("=" * 66, "=" * 66))
    print("%-12s | %9s %8s %8s %8s | %8s | %8s"
          % ("model", "mean ms", "std", "min", "max", "FPS", "speedup"))
    print("-" * 74)
    ordered = sorted(rows, key=lambda r: r[1].fps, reverse=True)
    for name, s in ordered:
        speedup = ("%.2fx" % (pt_stat.mean_ms / s.mean_ms)
                   if s.mean_ms > 0 and name != "pytorch" else "-")
        print("%-12s | %9.2f %8.2f %8.2f %8.2f | %8.2f | %8s"
              % (name, s.mean_ms, s.std_ms, s.min_ms, s.max_ms, s.fps, speedup))


def print_agreement(label: str, a: dict, match_iou: float) -> None:
    """Print the identification agreement of one TRT variant vs PyTorch."""
    print("\n%s\nID AGREEMENT  trt_%s vs pytorch  (match IoU >= %.2f)\n%s"
          % ("-" * 66, label, match_iou, "-" * 66))
    print("  PyTorch detections : %d" % a["pt"])
    print("  TensorRT detections: %d" % a["trt"])
    print("  matched            : %d  (mean IoU %.3f, mean |Δconf| %.3f)"
          % (a["matched"], a["mean_matched_iou"], a["mean_abs_conf_delta"]))
    print("  missed by TRT      : %d  (PyTorch found, TRT did not)" % a["pt_only"])
    print("  extra in TRT       : %d  (TRT found, PyTorch did not)" % a["trt_only"])
    if a["recall_vs_pytorch"] is not None:
        print("  recall vs PyTorch  : %.1f%%" % (100.0 * a["recall_vs_pytorch"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torch-weights", required=True,
                    help=".pt YOLO-World checkpoint: baseline + text encoder for all engines")
    ap.add_argument("--pair", action="append", required=True, type=parse_pair,
                    help="label:/backbone.engine,/head.engine (repeatable, e.g. s:.../s.backbone,.../s.head)")
    ap.add_argument("--images", required=True, help="folder of RGB frames")
    ap.add_argument("--out", required=True,
                    help="output folder (pytorch/ + trt_<label>/ subdirs)")
    ap.add_argument("--labels", required=True, type=parse_labels,
                    help="comma-separated open-vocab class prompts")
    ap.add_argument("--conf", type=float, default=0.40,
                    help="min confidence, shared by all (runtime NMS; default 0.40)")
    ap.add_argument("--iou", type=float, default=0.50, help="NMS IoU, shared by all")
    ap.add_argument("--match-iou", type=float, default=0.50,
                    help="IoU to count a PyTorch and TRT box as the same object")
    ap.add_argument("--max-det", type=int, default=None, help="TRT max detections/frame")
    ap.add_argument("--device", default="cuda:0",
                    help="torch device for the baseline and text encode")
    ap.add_argument("--imgsz", default=None,
                    help="HxW for the PyTorch baseline (default: first engine's size)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--no-vis", action="store_true", help="skip annotated images")
    args = ap.parse_args()

    labels = args.labels
    files = find_images(args.images, args.max_images)
    out_dir = Path(args.out)

    # PyTorch input size: explicit --imgsz, else read the first engine's built size.
    if args.imgsz:
        from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import parse_imgsz
        imgsz = parse_imgsz(args.imgsz)
    else:
        probe = YoloTRTDetector(args.pair[0][1], args.pair[0][2])
        imgsz = tuple(probe.stage.imgsz)
        del probe
        gc.collect()

    print("[compare-all] %d frames | %d prompts=%s | %d TRT variants | conf=%.2f iou=%.2f"
          % (len(files), len(labels), labels, len(args.pair), args.conf, args.iou))
    print("[compare-all] PyTorch imgsz=%dx%d (each TRT runs at its own engine size)"
          % (imgsz[0], imgsz[1]))

    # --- PyTorch baseline pass (held in memory for every variant to match against).
    print("\n[compare-all] running PyTorch baseline ...")
    torch_model = TorchYoloWorld(args.torch_weights, labels, imgsz, args.device,
                                 args.conf, args.iou)
    pt_writer = _Writer(out_dir / "pytorch", args.no_vis)
    pt_stat, pt_by_name = detect_pass(torch_model.detect, files, args.warmup,
                                      pt_writer, labels)
    pt_writer.close()
    del torch_model
    gc.collect()

    # --- Each TRT variant, one at a time, compared to the stored PyTorch set.
    speed_rows = [("pytorch", pt_stat)]
    variants = []
    for label, backbone, head in args.pair:
        print("\n[compare-all] running TRT variant '%s' ..." % label)
        trt = YoloTRTDetector(backbone, head, text_weights=args.torch_weights,
                              text_device=args.device, conf_thresh=args.conf,
                              iou_thresh=args.iou, max_det=args.max_det)
        trt.set_prompts(labels)
        writer = _Writer(out_dir / ("trt_" + label), args.no_vis)
        s, trt_by_name = detect_pass(trt.detect, files, args.warmup, writer, labels)
        writer.close()
        del trt
        gc.collect()

        a = agreement(pt_by_name, trt_by_name, labels, args.match_iou)
        speed_rows.append((label, s))
        variants.append({"label": label, "backbone": Path(backbone).name,
                         "head": Path(head).name, "stat": s, "agreement": a})

    # --- Combined report.
    print_speed(speed_rows, pt_stat)
    for v in variants:
        print_agreement(v["label"], v["agreement"], args.match_iou)

    print("\n%-16s | %8s | %-6s variants (detections)"
          % ("class", "pytorch", str(len(variants))))
    print("-" * 66)
    header = "%-16s | %8s | " % ("class", "pytorch") + " ".join(
        "%6s" % v["label"] for v in variants)
    print(header)
    for name in [l.strip().lower() for l in labels]:
        pt_c = variants[0]["agreement"]["per_class"][name]["pytorch"] if variants else 0
        cells = " ".join("%6d" % v["agreement"]["per_class"][name]["tensorrt"]
                         for v in variants)
        print("%-16s | %8d | %s" % (name, pt_c, cells))

    summary = {
        "images": len(files), "prompts": labels, "pytorch_imgsz": list(imgsz),
        "conf_thresh": args.conf, "iou_thresh": args.iou, "match_iou": args.match_iou,
        "pytorch_speed_ms": {"mean": pt_stat.mean_ms, "fps": pt_stat.fps},
        "variants": [{
            "label": v["label"], "backbone": v["backbone"], "head": v["head"],
            "speed_ms": {"mean": v["stat"].mean_ms, "std": v["stat"].std_ms,
                         "min": v["stat"].min_ms, "max": v["stat"].max_ms,
                         "fps": v["stat"].fps,
                         "speedup": (pt_stat.mean_ms / v["stat"].mean_ms
                                     if v["stat"].mean_ms > 0 else None)},
            "agreement": {k: v["agreement"][k] for k in
                          ("pt", "trt", "matched", "pt_only", "trt_only",
                           "mean_matched_iou", "mean_abs_conf_delta",
                           "recall_vs_pytorch", "per_class")},
        } for v in variants],
    }
    (out_dir / "comparison_all.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "comparison_all.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "mean_ms", "fps", "speedup", "pytorch_dets", "trt_dets",
                    "matched", "missed_by_trt", "extra_in_trt", "recall_vs_pytorch",
                    "mean_matched_iou", "mean_abs_conf_delta"])
        w.writerow(["pytorch", "%.4f" % pt_stat.mean_ms, "%.4f" % pt_stat.fps,
                    "-", "", "", "", "", "", "", "", ""])
        for v in variants:
            a, s = v["agreement"], v["stat"]
            w.writerow([
                "trt_" + v["label"], "%.4f" % s.mean_ms, "%.4f" % s.fps,
                ("%.4f" % (pt_stat.mean_ms / s.mean_ms)) if s.mean_ms > 0 else "",
                a["pt"], a["trt"], a["matched"], a["pt_only"], a["trt_only"],
                ("%.4f" % a["recall_vs_pytorch"]) if a["recall_vs_pytorch"] is not None
                else "", "%.4f" % a["mean_matched_iou"],
                "%.4f" % a["mean_abs_conf_delta"]])

    print("\n[done] wrote %s/{pytorch,trt_*}/ + comparison_all.{json,csv}" % out_dir)


if __name__ == "__main__":
    main()