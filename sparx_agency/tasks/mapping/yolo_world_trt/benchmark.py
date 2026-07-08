"""Benchmark & compare the YOLO-World TensorRT engines (s / m / l / x).

Runs each engine over a folder of RGB frames and reports, per variant, the mean /
std / min / max latency and FPS for the three stages -- letterbox preprocess,
TensorRT inference (H2D + execute + D2H + sync), numpy decode + NMS -- plus the
end-to-end total. It also pulls ``device`` and ``dla_eligible_layers`` from each
engine's manifest so the table shows, at a glance, how much of each variant landed
on the DLA. A side-by-side summary ranks the variants so you can pick the
accuracy/latency trade for the 15 W target.

Run (target device, TRT venv, PYTHONPATH = repo root):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.benchmark \\
        --images /path/to/frames \\
        --engine s:.../yolo_world_s.fp16.dla0.engine \\
        --engine m:.../yolo_world_m.fp16.dla0.engine \\
        --engine l:.../yolo_world_l.fp16.dla0.engine \\
        --engine x:.../yolo_world_x.fp16.dla0.engine \\
        --out /tmp/yolo_world_trt_compare
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.mapping.yolo_world_trt import postprocess, preprocess
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import (
    YoloWorldTRTEngine, load_manifest,
)


@dataclass
class Stat:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    fps: float
    n: int


def stat(times_ms: List[float]) -> Stat:
    if not times_ms:
        return Stat(0, 0, 0, 0, 0, 0)
    a = np.asarray(times_ms, dtype=np.float64)
    mean = float(a.mean())
    return Stat(mean, float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                float(a.min()), float(a.max()),
                1000.0 / mean if mean > 0 else 0.0, len(a))


def parse_engine_arg(value: str) -> Tuple[str, str]:
    if ":" not in value:
        return Path(value).stem, value
    label, path = value.split(":", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("engine must be label:/path, got %r" % value)
    return label, path


def find_images(images_dir: str, max_images: Optional[int]) -> List[str]:
    exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    files = sorted(p for p in glob.glob(os.path.join(images_dir, "*"))
                   if p.rsplit(".", 1)[-1].lower() in exts)
    if not files:
        raise RuntimeError("No images found in: %s" % images_dir)
    if max_images and len(files) > max_images:
        files = files[::max(1, len(files) // max_images)][:max_images]
    return files


def benchmark_engine(label, engine_path, files, warmup, repeat) -> dict:
    print("\n" + "=" * 78)
    print("VARIANT: %s   (%s)" % (label, Path(engine_path).name))
    print("=" * 78)
    man = load_manifest(engine_path)
    engine = YoloWorldTRTEngine(engine_path)
    labels = [str(p).strip().lower() for p in man.get("prompts", [])] or ["object"]
    imgsz = tuple(man.get("imgsz_hw", [engine.input_h, engine.input_w]))
    conf = float(man.get("conf_thresh", 0.25))
    iou = float(man.get("iou_thresh", 0.5))
    print("device=%s  dla_eligible=%s/%s layers  imgsz=%s  nc=%d"
          % (man.get("device"), man.get("dla_eligible_layers"),
             man.get("total_layers"), imgsz, len(labels)))

    warm = cv2.imread(files[0], cv2.IMREAD_COLOR)
    if warm is None:
        raise RuntimeError("Failed to read warmup image: %s" % files[0])
    warm_rgb = cv2.cvtColor(warm, cv2.COLOR_BGR2RGB)
    for _ in range(max(0, warmup)):
        padded, _t = preprocess.letterbox(warm_rgb, imgsz)
        engine.infer(preprocess.to_engine_tensor(padded))

    t_pre, t_inf, t_post, t_tot = [], [], [], []
    for path in files:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("  skipped unreadable image:", path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            padded, tr = preprocess.letterbox(rgb, imgsz)
            tensor = preprocess.to_engine_tensor(padded)
            t1 = time.perf_counter()
            raw = engine.infer(tensor)
            t2 = time.perf_counter()
            postprocess.decode(raw, labels, tr, conf_thresh=conf, iou_thresh=iou)
            t3 = time.perf_counter()
            t_pre.append((t1 - t0) * 1000.0)
            t_inf.append((t2 - t1) * 1000.0)
            t_post.append((t3 - t2) * 1000.0)
            t_tot.append((t3 - t0) * 1000.0)

    result = {
        "label": label, "engine_path": engine_path,
        "device": man.get("device"), "imgsz": imgsz,
        "dla_eligible": man.get("dla_eligible_layers"),
        "total_layers": man.get("total_layers"),
        "preprocess": stat(t_pre), "infer": stat(t_inf),
        "postprocess": stat(t_post), "total": stat(t_tot),
    }
    _print_one(result)
    return result


def _print_one(r: dict):
    print("%-14s | %9s %8s %8s %8s | %8s | %6s"
          % ("phase", "mean ms", "std", "min", "max", "FPS", "n"))
    print("-" * 74)
    for name, key in (("preprocess", "preprocess"), ("inference TRT", "infer"),
                      ("decode+NMS", "postprocess"), ("TOTAL", "total")):
        s = r[key]
        print("%-14s | %9.2f %8.2f %8.2f %8.2f | %8.2f | %6d"
              % (name, s.mean_ms, s.std_ms, s.min_ms, s.max_ms, s.fps, s.n))


def _print_summary(results: List[dict]):
    print("\n" + "=" * 78)
    print("SUMMARY (end-to-end, ranked by FPS)")
    print("=" * 78)
    print("%-8s | %-6s | %10s | %9s | %8s | %12s"
          % ("variant", "device", "infer ms", "total ms", "FPS", "DLA layers"))
    print("-" * 74)
    for r in sorted(results, key=lambda r: r["total"].fps, reverse=True):
        print("%-8s | %-6s | %10.2f | %9.2f | %8.2f | %6s/%-6s"
              % (r["label"], r["device"], r["infer"].mean_ms, r["total"].mean_ms,
                 r["total"].fps, r["dla_eligible"], r["total_layers"]))


def write_csv(results: List[dict], out_prefix: str):
    path = out_prefix + ".csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "device", "dla_eligible", "total_layers", "imgsz",
                    "phase", "mean_ms", "std_ms", "min_ms", "max_ms", "fps", "n"])
        for r in results:
            for phase in ("preprocess", "infer", "postprocess", "total"):
                s = r[phase]
                w.writerow([r["label"], r["device"], r["dla_eligible"],
                            r["total_layers"], "x".join(map(str, r["imgsz"])), phase,
                            "%.6f" % s.mean_ms, "%.6f" % s.std_ms, "%.6f" % s.min_ms,
                            "%.6f" % s.max_ms, "%.6f" % s.fps, s.n])
    print("\nSaved:", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of RGB frames")
    ap.add_argument("--engine", action="append", required=True, type=parse_engine_arg,
                    help="label:/path/to/engine (repeatable, e.g. s:.../..engine)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="/tmp/yolo_world_trt_compare")
    args = ap.parse_args()

    files = find_images(args.images, args.max_images)
    print("Images: %s (%d frames)" % (args.images, len(files)))
    results = [benchmark_engine(lbl, path, files, args.warmup, args.repeat)
               for lbl, path in args.engine]
    _print_summary(results)
    write_csv(results, args.out)


if __name__ == "__main__":
    main()
