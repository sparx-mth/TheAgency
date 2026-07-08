"""Compare per-frame runtime: PyTorch YOLO-World vs. the TensorRT split.

Runs both detectors over the same folder of images with the same open-vocab
labels and the same input size, and prints a side-by-side latency / FPS table plus
the TRT speed-up. Both timings are the *full* per-frame path (preprocess ->
inference -> NMS/decode), which is what the drone actually pays. It also reports
mean detections/frame so you can sanity-check that the two agree.

Both are run at the TRT engine's input size (from its manifest) so the comparison
is apples-to-apples; override with ``--imgsz`` if you want.

Run (target device, TRT venv with ultralytics+torch AND tensorrt+pycuda):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.compare_torch_vs_trt \\
        --torch-weights /path/to/yolov8s-world.pt \\
        --backbone .../yolo_world_s.backbone.fp16.dla0.engine \\
        --head     .../yolo_world_s.head.fp16.gpu.engine \\
        --images   /path/to/frames \\
        --labels   "weapon, chair, refrigerator"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from sparx_agency.tasks.mapping.yolo_world_trt.benchmark import Stat, find_images, stat
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector


def parse_labels(text: str) -> List[str]:
    """`"weapon, chair, refrigerator"` -> `["weapon", "chair", "refrigerator"]`."""
    labels = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    if not labels:
        raise ValueError("--labels is empty; give e.g. \"weapon, chair, fridge\".")
    return labels


class TorchYoloWorld:
    """The PyTorch YOLO-World baseline (ultralytics), matched to the engine's size."""

    def __init__(self, weights, labels, imgsz, device, conf, iou):
        from ultralytics import YOLOWorld  # lazy: heavy torch dep

        self.model = YOLOWorld(str(weights))
        self.model.to(device)
        self.model.set_classes(labels)
        self.imgsz = (int(imgsz[0]), int(imgsz[1]))
        self.device = device
        self.conf = conf
        self.iou = iou

    def detect_count(self, rgb: np.ndarray) -> int:
        """Run one frame; return the detection count (times the full call)."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])   # ultralytics expects BGR
        res = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)[0]
        boxes = getattr(res, "boxes", None)
        return 0 if boxes is None else len(boxes)


def _time_loop(run_one, files, repeat) -> tuple:
    """Time ``run_one(rgb)`` over the files; return (Stat, mean detections)."""
    times, counts = [], []
    for path in files:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("  skipped unreadable image:", path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            n = run_one(rgb)
            times.append((time.perf_counter() - t0) * 1000.0)
            counts.append(int(n))
    return stat(times), (float(np.mean(counts)) if counts else 0.0)


def _print_row(name: str, s: Stat, dets: float):
    print("%-10s | %9.2f %8.2f %8.2f %8.2f | %8.2f | %8.2f"
          % (name, s.mean_ms, s.std_ms, s.min_ms, s.max_ms, s.fps, dets))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torch-weights", required=True, help="the .pt YOLO-World checkpoint")
    ap.add_argument("--backbone", required=True, help="TRT backbone engine")
    ap.add_argument("--head", required=True, help="TRT head engine")
    ap.add_argument("--images", required=True, help="folder of RGB frames")
    ap.add_argument("--labels", required=True,
                    help='comma-separated, e.g. "weapon, chair, refrigerator"')
    ap.add_argument("--imgsz", default=None, help="HxW (default: from engine manifest)")
    ap.add_argument("--device", default="cuda:0", help="torch device for the baseline")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    labels = parse_labels(args.labels)
    files = find_images(args.images, args.max_images)

    # TensorRT split (its manifest sets the input size both models run at).
    trt = YoloTRTDetector(args.backbone, args.head, text_weights=args.torch_weights,
                          text_device=args.device, conf_thresh=args.conf,
                          iou_thresh=args.iou)
    trt.set_prompts(labels)
    if args.imgsz:
        from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import parse_imgsz
        imgsz = parse_imgsz(args.imgsz)
    else:
        imgsz = tuple(trt.stage.imgsz)

    # PyTorch baseline at the same size + thresholds.
    torch_model = TorchYoloWorld(args.torch_weights, labels, imgsz, args.device,
                                 args.conf, args.iou)

    print("labels=%s  imgsz=%dx%d  images=%d  repeat=%d  device=%s"
          % (labels, imgsz[0], imgsz[1], len(files), args.repeat, args.device))

    warm = cv2.cvtColor(cv2.imread(files[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    for _ in range(max(0, args.warmup)):
        torch_model.detect_count(warm)
        trt.detect(warm)

    torch_stat, torch_dets = _time_loop(torch_model.detect_count, files, args.repeat)
    trt_stat, trt_dets = _time_loop(lambda rgb: len(trt.detect(rgb)), files, args.repeat)

    print("\n%-10s | %9s %8s %8s %8s | %8s | %8s"
          % ("model", "mean ms", "std", "min", "max", "FPS", "dets/img"))
    print("-" * 74)
    _print_row("pytorch", torch_stat, torch_dets)
    _print_row("tensorrt", trt_stat, trt_dets)
    if trt_stat.mean_ms > 0 and torch_stat.mean_ms > 0:
        print("-" * 74)
        print("TensorRT speed-up: %.2fx  (%.1f -> %.1f ms/frame, %.1f -> %.1f FPS)"
              % (torch_stat.mean_ms / trt_stat.mean_ms, torch_stat.mean_ms,
                 trt_stat.mean_ms, torch_stat.fps, trt_stat.fps))


if __name__ == "__main__":
    main()
