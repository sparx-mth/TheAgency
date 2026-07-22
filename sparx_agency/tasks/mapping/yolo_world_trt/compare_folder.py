"""Run PyTorch YOLO-World *and* the TensorRT split over one folder, then compare.

For every frame in ``--images`` this runs both detectors with the same prompts,
input size, and thresholds, and produces three things:

  1. **Identifications for each** -- ``<out>/pytorch/`` and ``<out>/tensorrt/``
     each get annotated ``.jpg``s, a ``detections.csv`` and a ``detections.jsonl``
     (same layout as :mod:`detect_folder`).
  2. **Speed of each** -- a latency / FPS table (per-frame ``detect()`` call) and
     the TensorRT speed-up.
  3. **How much accuracy the TRT engine costs** -- per frame the two detection sets
     are matched by class + box IoU, and the run reports matched / missed-by-TRT /
     extra-in-TRT counts, mean IoU of matches, and mean confidence drift. A
     ``comparison.json`` + ``comparison_per_frame.csv`` capture the detail.

The TRT engine runs FP16 (and part of the backbone on the DLA) at the engine's
built size, while PyTorch runs its own fp32 decode -- so this is the honest
"is there a drop?" check. Confidence (``--conf``) is a runtime NMS knob shared by
both; it does not touch the engines.

Run (target Orin, venv with ultralytics+torch AND tensorrt+pycuda, PYTHONPATH=repo):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.compare_folder \\
        --torch-weights /path/to/yolov8s-worldv2.pt \\
        --backbone .../orin_sm87/yolo_world_s.backbone.fp16.dla0.engine \\
        --head     .../orin_sm87/yolo_world_s.head.fp16.gpu.engine \\
        --images /path/to/rgb --out /path/to/identifications \\
        --labels "chair, bottle, table" --conf 0.4
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.mapping.yolo_world_trt.benchmark import Stat, find_images, stat
from sparx_agency.tasks.mapping.yolo_world_trt.compare_torch_vs_trt import TorchYoloWorld
from sparx_agency.tasks.mapping.yolo_world_trt.detect_folder import draw, parse_labels
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector


def iou_xyxy(a, b) -> float:
    """IoU of two ``(x1,y1,x2,y2)`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(pt_dets, trt_dets, iou_thr: float) -> Tuple[list, list, list]:
    """Greedy match by identical label + IoU >= ``iou_thr`` (highest IoU first).

    Returns ``(matched, pt_only, trt_only)`` where ``matched`` is a list of
    ``(pt_index, trt_index, iou)`` and the ``*_only`` lists are unmatched indices.
    """
    pairs = []
    for i, p in enumerate(pt_dets):
        for j, t in enumerate(trt_dets):
            if p.label == t.label:
                v = iou_xyxy(p.bbox_xyxy, t.bbox_xyxy)
                if v >= iou_thr:
                    pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_t, matched = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        matched.append((i, j, v))
    pt_only = [i for i in range(len(pt_dets)) if i not in used_p]
    trt_only = [j for j in range(len(trt_dets)) if j not in used_t]
    return matched, pt_only, trt_only


class _Writer:
    """Streams annotated images + detections.csv + detections.jsonl for one backend."""

    def __init__(self, out_dir: Path, no_vis: bool):
        self.vis_dir = out_dir / "annotated"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.no_vis = no_vis
        self._cf = open(out_dir / "detections.csv", "w", newline="", encoding="utf-8")
        self._jf = open(out_dir / "detections.jsonl", "w", encoding="utf-8")
        self._w = csv.writer(self._cf)
        self._w.writerow(["image", "label", "score", "x1", "y1", "x2", "y2"])

    def add(self, name: str, bgr: np.ndarray, dets, labels: List[str]) -> None:
        records = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox_xyxy
            self._w.writerow([name, d.label, "%.4f" % d.score, x1, y1, x2, y2])
            records.append({"label": d.label, "score": round(float(d.score), 4),
                            "bbox_xyxy": [x1, y1, x2, y2]})
        self._jf.write(json.dumps({"image": name, "detections": records}) + "\n")
        if not self.no_vis:
            cv2.imwrite(str(self.vis_dir / (Path(name).stem + ".jpg")),
                        draw(bgr, dets, labels))

    def close(self) -> None:
        self._cf.close()
        self._jf.close()


def _print_speed(pt: Stat, trt: Stat) -> None:
    print("\n%-10s | %9s %8s %8s %8s | %8s"
          % ("model", "mean ms", "std", "min", "max", "FPS"))
    print("-" * 62)
    for name, s in (("pytorch", pt), ("tensorrt", trt)):
        print("%-10s | %9.2f %8.2f %8.2f %8.2f | %8.2f"
              % (name, s.mean_ms, s.std_ms, s.min_ms, s.max_ms, s.fps))
    if pt.mean_ms > 0 and trt.mean_ms > 0:
        print("-" * 62)
        print("TensorRT speed-up: %.2fx  (%.1f -> %.1f ms/frame, %.1f -> %.1f FPS)"
              % (pt.mean_ms / trt.mean_ms, pt.mean_ms, trt.mean_ms, pt.fps, trt.fps))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torch-weights", required=True, help=".pt YOLO-World checkpoint")
    ap.add_argument("--backbone", required=True, help="TRT backbone engine")
    ap.add_argument("--head", required=True, help="TRT head engine")
    ap.add_argument("--images", required=True, help="folder of RGB frames")
    ap.add_argument("--out", required=True, help="output folder (pytorch/ + tensorrt/)")
    ap.add_argument("--labels", required=True, type=parse_labels,
                    help="comma-separated open-vocab class prompts")
    ap.add_argument("--conf", type=float, default=0.40,
                    help="min confidence, shared by both (runtime NMS; default 0.40)")
    ap.add_argument("--iou", type=float, default=0.50, help="NMS IoU, shared by both")
    ap.add_argument("--match-iou", type=float, default=0.50,
                    help="IoU to count a PyTorch and TRT box as the same object")
    ap.add_argument("--max-det", type=int, default=None, help="TRT max detections/frame")
    ap.add_argument("--device", default="cuda:0", help="torch device for the baseline")
    ap.add_argument("--imgsz", default=None, help="HxW (default: from engine manifest)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--no-vis", action="store_true", help="skip annotated images")
    args = ap.parse_args()

    labels = args.labels
    files = find_images(args.images, args.max_images)

    trt = YoloTRTDetector(args.backbone, args.head, text_weights=args.torch_weights,
                          text_device=args.device, conf_thresh=args.conf,
                          iou_thresh=args.iou, max_det=args.max_det)
    trt.set_prompts(labels)
    if args.imgsz:
        from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import parse_imgsz
        imgsz = parse_imgsz(args.imgsz)
    else:
        imgsz = tuple(trt.stage.imgsz)
    torch_model = TorchYoloWorld(args.torch_weights, labels, imgsz, args.device,
                                 args.conf, args.iou)

    print("[compare] %d frames | %d prompts=%s | imgsz=%dx%d conf=%.2f iou=%.2f"
          % (len(files), len(labels), labels, imgsz[0], imgsz[1], args.conf, args.iou))

    warm = cv2.cvtColor(cv2.imread(files[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    for _ in range(max(0, args.warmup)):
        torch_model.detect(warm)
        trt.detect(warm)

    out_dir = Path(args.out)
    pt_writer = _Writer(out_dir / "pytorch", args.no_vis)
    trt_writer = _Writer(out_dir / "tensorrt", args.no_vis)
    pt_ms, trt_ms = [], []
    agg = {"pt": 0, "trt": 0, "matched": 0, "pt_only": 0, "trt_only": 0,
           "iou_sum": 0.0, "score_abs_sum": 0.0}
    per_class = {name: {"pytorch": 0, "tensorrt": 0, "matched": 0} for name in
                 [l.strip().lower() for l in labels]}
    per_frame_rows = []

    for path in files:
        name = Path(path).name
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("  skipped unreadable image:", name)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        pt_dets = torch_model.detect(rgb)
        t1 = time.perf_counter()
        trt_dets = trt.detect(rgb)
        t2 = time.perf_counter()
        pt_ms.append((t1 - t0) * 1000.0)
        trt_ms.append((t2 - t1) * 1000.0)

        pt_writer.add(name, bgr, pt_dets, labels)
        trt_writer.add(name, bgr, trt_dets, labels)

        matched, pt_only, trt_only = match(pt_dets, trt_dets, args.match_iou)
        agg["pt"] += len(pt_dets)
        agg["trt"] += len(trt_dets)
        agg["matched"] += len(matched)
        agg["pt_only"] += len(pt_only)
        agg["trt_only"] += len(trt_only)
        for i, j, v in matched:
            agg["iou_sum"] += v
            agg["score_abs_sum"] += abs(pt_dets[i].score - trt_dets[j].score)
        for d in pt_dets:
            if d.label in per_class:
                per_class[d.label]["pytorch"] += 1
        for d in trt_dets:
            if d.label in per_class:
                per_class[d.label]["tensorrt"] += 1
        for i, j, v in matched:
            if pt_dets[i].label in per_class:
                per_class[pt_dets[i].label]["matched"] += 1
        per_frame_rows.append((name, len(pt_dets), len(trt_dets), len(matched),
                               len(pt_only), len(trt_only)))

    pt_writer.close()
    trt_writer.close()

    pt_stat, trt_stat = stat(pt_ms), stat(trt_ms)
    _print_speed(pt_stat, trt_stat)

    m = max(1, agg["matched"])
    print("\n%s\nID AGREEMENT (match IoU >= %.2f)\n%s"
          % ("=" * 62, args.match_iou, "=" * 62))
    print("  PyTorch detections : %d" % agg["pt"])
    print("  TensorRT detections: %d" % agg["trt"])
    print("  matched            : %d  (mean IoU %.3f, mean |Δconf| %.3f)"
          % (agg["matched"], agg["iou_sum"] / m, agg["score_abs_sum"] / m))
    print("  missed by TRT      : %d  (PyTorch found, TRT did not)" % agg["pt_only"])
    print("  extra in TRT       : %d  (TRT found, PyTorch did not)" % agg["trt_only"])
    if agg["pt"]:
        print("  recall vs PyTorch  : %.1f%%  (matched / PyTorch dets)"
              % (100.0 * agg["matched"] / agg["pt"]))
    print("\n%-16s | %8s %8s %8s" % ("class", "pytorch", "tensorrt", "matched"))
    print("-" * 48)
    for name, c in per_class.items():
        print("%-16s | %8d %8d %8d"
              % (name, c["pytorch"], c["tensorrt"], c["matched"]))

    summary = {
        "images": len(files), "prompts": labels, "imgsz": list(imgsz),
        "conf_thresh": args.conf, "iou_thresh": args.iou, "match_iou": args.match_iou,
        "speed_ms": {"pytorch_mean": pt_stat.mean_ms, "tensorrt_mean": trt_stat.mean_ms,
                     "pytorch_fps": pt_stat.fps, "tensorrt_fps": trt_stat.fps,
                     "speedup": (pt_stat.mean_ms / trt_stat.mean_ms
                                 if trt_stat.mean_ms > 0 else None)},
        "agreement": {
            "pytorch_dets": agg["pt"], "tensorrt_dets": agg["trt"],
            "matched": agg["matched"], "missed_by_trt": agg["pt_only"],
            "extra_in_trt": agg["trt_only"],
            "mean_matched_iou": agg["iou_sum"] / m,
            "mean_abs_conf_delta": agg["score_abs_sum"] / m,
            "recall_vs_pytorch": (agg["matched"] / agg["pt"]) if agg["pt"] else None,
        },
        "per_class": per_class,
    }
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "comparison_per_frame.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "pytorch", "tensorrt", "matched", "missed_by_trt",
                    "extra_in_trt"])
        w.writerows(per_frame_rows)
    print("\n[done] wrote %s/{pytorch,tensorrt}/ + comparison.json" % out_dir)


if __name__ == "__main__":
    main()
