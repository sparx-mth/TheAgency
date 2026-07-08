"""Benchmark & compare the open-set YOLO-World TRT splits (s / m / l / x).

For each variant it runs the full per-frame path -- letterbox preprocess ->
backbone(DLA)+head(GPU) inference -> numpy decode+NMS -> over a folder of frames,
and reports mean/std/min/max latency and FPS per stage plus the end-to-end total.
It reads ``device`` and ``dla_eligible_layers`` from the backbone manifest so the
table shows how much of each variant landed on the DLA, and ranks the variants.

The text branch is NOT exercised here (it runs only on re-prompt): the benchmark
uploads random text embeddings for ``--num-prompts`` classes, which drives the
head's dynamic-N cost realistically without needing torch on the target.

Run (target device, TRT venv, PYTHONPATH = repo root):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.benchmark \\
        --images /path/to/frames --num-prompts 4 \\
        --pair s:.../yolo_world_s.backbone.fp16.dla0.engine,.../yolo_world_s.head.fp16.gpu.engine \\
        --pair m:.../yolo_world_m.backbone.fp16.dla0.engine,.../yolo_world_m.head.fp16.gpu.engine
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
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import TwoStageYoloTRT


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


def parse_pair(value: str) -> Tuple[str, str, str]:
    """``label:/backbone.engine,/head.engine`` -> (label, backbone, head)."""
    if ":" not in value or "," not in value:
        raise argparse.ArgumentTypeError(
            "pair must be label:/backbone.engine,/head.engine, got %r" % value)
    label, paths = value.split(":", 1)
    back, head = paths.split(",", 1)
    return label.strip(), back.strip(), head.strip()


def find_images(images_dir: str, max_images: Optional[int]) -> List[str]:
    exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    files = sorted(p for p in glob.glob(os.path.join(images_dir, "*"))
                   if p.rsplit(".", 1)[-1].lower() in exts)
    if not files:
        raise RuntimeError("No images found in: %s" % images_dir)
    if max_images and len(files) > max_images:
        files = files[::max(1, len(files) // max_images)][:max_images]
    return files


def _random_text_features(stage: TwoStageYoloTRT, n: int):
    """Upload n random unit-norm embeddings so the head runs at prompt count n."""
    axis = stage._txt_axis
    shape = list(stage._txt_base)
    shape[axis] = n
    emb = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-9)
    stage.set_text_features(emb, ["c%d" % i for i in range(n)])


def benchmark_pair(label, backbone, head, files, num_prompts, warmup, repeat) -> dict:
    print("\n" + "=" * 78)
    print("VARIANT: %s" % label)
    print("  backbone: %s" % Path(backbone).name)
    print("  head    : %s" % Path(head).name)
    print("=" * 78)
    stage = TwoStageYoloTRT(backbone, head)
    _random_text_features(stage, num_prompts)
    labels = list(stage._labels)
    bman = stage.bman
    print("backbone device=%s  dla_eligible=%s/%s layers  imgsz=%s  N=%d"
          % (bman.get("device"), bman.get("dla_eligible_layers"),
             bman.get("total_layers"), stage.imgsz, num_prompts))

    warm = cv2.imread(files[0], cv2.IMREAD_COLOR)
    if warm is None:
        raise RuntimeError("Failed to read warmup image: %s" % files[0])
    warm_rgb = cv2.cvtColor(warm, cv2.COLOR_BGR2RGB)
    for _ in range(max(0, warmup)):
        padded, _t = preprocess.letterbox(warm_rgb, stage.imgsz)
        stage.infer(preprocess.to_engine_tensor(padded))

    t_pre, t_inf, t_post, t_tot = [], [], [], []
    for path in files:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("  skipped unreadable image:", path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            padded, tr = preprocess.letterbox(rgb, stage.imgsz)
            tensor = preprocess.to_engine_tensor(padded)
            t1 = time.perf_counter()
            raw = stage.infer(tensor)
            t2 = time.perf_counter()
            postprocess.decode(raw, labels, tr)
            t3 = time.perf_counter()
            t_pre.append((t1 - t0) * 1000.0)
            t_inf.append((t2 - t1) * 1000.0)
            t_post.append((t3 - t2) * 1000.0)
            t_tot.append((t3 - t0) * 1000.0)

    result = {
        "label": label, "device": bman.get("device"), "imgsz": stage.imgsz,
        "num_prompts": num_prompts,
        "dla_eligible": bman.get("dla_eligible_layers"),
        "total_layers": bman.get("total_layers"),
        "preprocess": stat(t_pre), "infer": stat(t_inf),
        "postprocess": stat(t_post), "total": stat(t_tot),
    }
    _print_one(result)
    return result


def _print_one(r: dict):
    print("%-14s | %9s %8s %8s %8s | %8s | %6s"
          % ("phase", "mean ms", "std", "min", "max", "FPS", "n"))
    print("-" * 74)
    for name, key in (("preprocess", "preprocess"), ("backbone+head", "infer"),
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
                    "num_prompts", "phase", "mean_ms", "std_ms", "min_ms", "max_ms",
                    "fps", "n"])
        for r in results:
            for phase in ("preprocess", "infer", "postprocess", "total"):
                s = r[phase]
                w.writerow([r["label"], r["device"], r["dla_eligible"],
                            r["total_layers"], "x".join(map(str, r["imgsz"])),
                            r["num_prompts"], phase, "%.6f" % s.mean_ms,
                            "%.6f" % s.std_ms, "%.6f" % s.min_ms, "%.6f" % s.max_ms,
                            "%.6f" % s.fps, s.n])
    print("\nSaved:", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of RGB frames")
    ap.add_argument("--pair", action="append", required=True, type=parse_pair,
                    help="label:/backbone.engine,/head.engine (repeatable)")
    ap.add_argument("--num-prompts", type=int, default=4,
                    help="prompt count to drive the head's dynamic-N cost")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default="/tmp/yolo_world_trt_compare")
    args = ap.parse_args()

    files = find_images(args.images, args.max_images)
    print("Images: %s (%d frames)  num_prompts=%d" % (args.images, len(files),
                                                       args.num_prompts))
    results = [benchmark_pair(lbl, b, h, files, args.num_prompts, args.warmup,
                              args.repeat) for lbl, b, h in args.pair]
    _print_summary(results)
    write_csv(results, args.out)


if __name__ == "__main__":
    main()
