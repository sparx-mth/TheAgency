#!/usr/bin/env python3
"""DA3 TensorRT acceptance smoke test for a new target device (e.g. a
freshly-delivered Robotican Orin-NX unit).

Standalone on purpose: this file has no dependency on our own repo/package
(no `sparx_agency` import) -- it's handed to Robotican on its own, not
alongside our whole codebase, so it inlines the minimal TensorRT
engine-load-and-infer logic itself rather than importing our
DA3TensorRTModel wrapper. Only needs: cv2, numpy, tensorrt, pycuda.

Builds the engine from ONNX itself if --engine-path doesn't exist yet
(pass --onnx-path). TensorRT engines are hardware/version-locked (a
recurring, already-documented failure mode in our own stack: engines
built on one TensorRT build fail to deserialize on another), so "can this
device build our engine from ONNX" is itself part of the acceptance
check, not a prerequisite to skip past by handing over a pre-built engine.

No camera intrinsics/calibration YAML is needed here -- this check only
asks "does the engine load and produce sane depth values," it does not
project depth into a point cloud, so there's nothing for intrinsics to do.

Usage (engine already built):
    python3 da3_acceptance_summary.py \\
        --rgb-dir /path/to/sample_images \\
        --engine-path /path/to/DA3-METRIC.engine

Usage (build the engine from ONNX first, then run the same check):
    python3 da3_acceptance_summary.py \\
        --rgb-dir /path/to/sample_images \\
        --onnx-path /path/to/DA3-METRIC.onnx \\
        --engine-path /path/to/DA3-METRIC.engine
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401 -- initializes CUDA context before any pycuda.driver calls
import pycuda.driver as cuda

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_engine(
    onnx_path: Path, engine_path: Path, trtexec_bin: str, fp16: bool
) -> tuple[bool, float, str]:
    """Build a TensorRT engine from ONNX via trtexec. Returns
    (success, build_sec, log_tail) -- log_tail is trtexec's last ~40 lines,
    useful on failure without dumping the whole (often very long) build log."""
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [trtexec_bin, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if fp16:
        cmd.append("--fp16")

    print(f"[da3-acceptance] building engine: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    build_sec = time.perf_counter() - t0

    log = (proc.stdout or "") + (proc.stderr or "")
    log_tail = "\n".join(log.splitlines()[-40:])
    success = proc.returncode == 0 and engine_path.exists()
    return success, build_sec, log_tail


class DA3Engine:
    """Minimal TensorRT engine wrapper: load a DA3 .engine file, run
    inference on a BGR frame, return a (H, W) float32 depth map. Resizes
    internally to whatever input shape the engine reports -- no
    camera-specific preprocessing."""

    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        self.input_name = self.engine.get_tensor_name(0)
        input_shape = self.engine.get_tensor_shape(self.input_name)
        self.input_h = int(input_shape[2])
        self.input_w = int(input_shape[3])

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        self.depth_output = next(
            out for out in self.outputs if "depth" in out["name"].lower()
        ) if any("depth" in o["name"].lower() for o in self.outputs) else self.outputs[0]

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for tensor_name in self.engine:
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem, "name": tensor_name})
            else:
                outputs.append({"host": host_mem, "device": device_mem, "name": tensor_name})
        return inputs, outputs, bindings, stream

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) * (1.0 / 255.0)
        img = np.transpose(img, (2, 0, 1)).ravel()

        np.copyto(self.inputs[0]["host"], img)
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.depth_output["host"], self.depth_output["device"], self.stream)
        self.stream.synchronize()

        out_shape = tuple(self.engine.get_tensor_shape(self.depth_output["name"]))
        if len(out_shape) == 4:
            _, _, h, w = out_shape
        elif len(out_shape) == 3:
            _, h, w = out_shape
        elif len(out_shape) == 2:
            h, w = out_shape
        else:
            raise ValueError(f"Unexpected depth output shape: {out_shape}")

        return self.depth_output["host"].reshape(h, w).astype(np.float32, copy=True)


def list_images(rgb_dir: Path, image_glob: str, max_images: int | None) -> list[Path]:
    images = sorted(
        p for p in rgb_dir.glob(image_glob)
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_images is not None:
        images = images[:max_images]
    return images


def finite_ratio(depth: np.ndarray) -> float:
    if depth.size == 0:
        return 0.0
    return float(np.isfinite(depth).sum()) / float(depth.size)


def run(args: argparse.Namespace) -> int:
    rgb_dir = Path(args.rgb_dir).expanduser().resolve()
    if not rgb_dir.exists():
        print(f"[da3-acceptance] FAIL: rgb-dir does not exist: {rgb_dir}")
        return 1

    images = list_images(rgb_dir, args.image_glob, args.max_images)
    if not images:
        print(f"[da3-acceptance] FAIL: no images found in {rgb_dir} (glob={args.image_glob})")
        return 1

    print(f"[da3-acceptance] rgb_dir      : {rgb_dir}")
    print(f"[da3-acceptance] engine_path  : {args.engine_path}")
    print(f"[da3-acceptance] num_images   : {len(images)}")

    engine_path = Path(args.engine_path).expanduser().resolve()
    need_build = args.force_rebuild or not engine_path.exists()
    if need_build:
        if not args.onnx_path:
            print(
                f"[da3-acceptance] FAIL: engine does not exist at {engine_path} "
                f"and no --onnx-path given to build it from."
            )
            return 1
        onnx_path = Path(args.onnx_path).expanduser().resolve()
        if not onnx_path.exists():
            print(f"[da3-acceptance] FAIL: --onnx-path does not exist: {onnx_path}")
            return 1
        success, build_sec, log_tail = build_engine(
            onnx_path, engine_path, args.trtexec_bin, fp16=not args.fp32
        )
        print(f"[da3-acceptance] engine build time: {build_sec:.1f}s")
        if not success:
            print(f"[da3-acceptance] FAIL: trtexec did not produce an engine. Last output:\n{log_tail}")
            return 1
        print(f"[da3-acceptance] engine built successfully: {engine_path}")

    t_load0 = time.perf_counter()
    try:
        model = DA3Engine(str(engine_path))
    except Exception as e:
        print(f"[da3-acceptance] FAIL: engine failed to load: {e}")
        return 1
    load_sec = time.perf_counter() - t_load0
    print(f"[da3-acceptance] engine load time: {load_sec:.2f}s")

    latencies_sec = []
    finite_ratios = []
    errors = 0

    for idx, image_path in enumerate(images):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[da3-acceptance][WARN] failed to read image: {image_path}")
            errors += 1
            continue
        try:
            t0 = time.perf_counter()
            depth = model.infer(bgr)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"[da3-acceptance][WARN] inference failed on {image_path.name}: {e}")
            errors += 1
            continue

        latencies_sec.append(elapsed)
        ratio = finite_ratio(depth)
        finite_ratios.append(ratio)
        print(
            f"[da3-acceptance] {idx + 1:04d}/{len(images):04d} {image_path.name} "
            f"shape={depth.shape[1]}x{depth.shape[0]} "
            f"finite_ratio={ratio:.3f} elapsed={elapsed * 1000:.1f}ms"
        )

    if not latencies_sec:
        print("[da3-acceptance] FAIL: every image failed to read or infer.")
        return 1

    mean_latency = statistics.mean(latencies_sec)
    p95_latency = sorted(latencies_sec)[int(0.95 * (len(latencies_sec) - 1))]
    mean_fps = 1.0 / mean_latency if mean_latency > 0 else 0.0
    mean_finite_ratio = statistics.mean(finite_ratios)
    error_ratio = errors / len(images)

    print("\n[da3-acceptance] ---- summary ----")
    print(f"[da3-acceptance] images processed : {len(latencies_sec)}/{len(images)} "
          f"(errors: {errors}, error_ratio={error_ratio:.3f})")
    print(f"[da3-acceptance] mean latency     : {mean_latency * 1000:.1f}ms")
    print(f"[da3-acceptance] p95 latency      : {p95_latency * 1000:.1f}ms")
    print(f"[da3-acceptance] mean FPS         : {mean_fps:.2f}")
    print(f"[da3-acceptance] mean finite ratio: {mean_finite_ratio:.3f}")

    passed = (
        error_ratio <= args.max_error_ratio
        and mean_fps >= args.min_fps
        and mean_finite_ratio >= args.min_finite_ratio
    )
    print(f"[da3-acceptance] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick pass/fail gate for a DA3 TensorRT engine on a new device: "
        "does it build, does it load, how fast is inference, are the depth values sane."
    )
    p.add_argument("--rgb-dir", required=True)
    p.add_argument("--engine-path", required=True,
                    help="Where the built engine lives (or should be built to).")
    p.add_argument("--onnx-path", default=None,
                    help="DA3 ONNX model to build the engine from, if --engine-path doesn't exist yet.")
    p.add_argument("--trtexec-bin", default="/usr/src/tensorrt/bin/trtexec")
    p.add_argument("--fp32", action="store_true",
                    help="Build in FP32 instead of the default FP16.")
    p.add_argument("--force-rebuild", action="store_true",
                    help="Rebuild the engine even if --engine-path already exists.")
    p.add_argument("--image-glob", default="*")
    p.add_argument("--max-images", type=int, default=None,
                    help="Cap the number of images processed (default: all matching).")
    p.add_argument("--min-fps", type=float, default=5.0,
                    help="Pass threshold for mean inference FPS.")
    p.add_argument("--max-error-ratio", type=float, default=0.05,
                    help="Pass threshold for the fraction of images that failed to read/infer.")
    p.add_argument("--min-finite-ratio", type=float, default=0.9,
                    help="Pass threshold for the mean fraction of finite (non-NaN/inf) depth pixels.")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
