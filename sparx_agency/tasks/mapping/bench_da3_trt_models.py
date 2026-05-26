#!/usr/bin/env python3
"""
Benchmark DA3 TensorRT engines without ROS transport.

Measures:
  - image read time
  - TensorRT inference time:
      preprocess + H2D + execute + D2H + sync
  - ROS2-style depth Image packing time:
      depth float32 -> bytes / sensor_msgs.Image-like payload

Example:
  python3 bench_da3_trt_models.py \
    --images /home/user/Pictures/bench_frames \
    --normalize zero_one \
    --warmup 5 \
    --repeat 3 \
    --out /tmp/da3_metric_compare \
    --model 294x504:/path/to/DA3METRIC-LARGE.fp16-294x504.depth_only.engine \
    --model 392x504:/path/to/DA3METRIC-LARGE.fp16-392x504.depth_only.engine \
    --model 420x728:/path/to/DA3METRIC-LARGE.fp16-420x728.engine
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

try:
    from sensor_msgs.msg import Image as RosImage
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class TimingStats:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    fps_from_mean: float
    n: int


def calc_stats(times_ms: list[float]) -> TimingStats:
    if not times_ms:
        return TimingStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    std_ms = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    min_ms = float(arr.min())
    max_ms = float(arr.max())
    fps_from_mean = 1000.0 / mean_ms if mean_ms > 0.0 else 0.0

    return TimingStats(
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        fps_from_mean=fps_from_mean,
        n=len(arr),
    )


class DA3Engine:
    def __init__(
        self,
        engine_path: str,
        *,
        depth_name: str | None = None,
        normalize: str = "zero_one",
    ):
        self.engine_path = engine_path
        self.normalize = normalize

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs: dict[str, dict] = {}
        self.outputs: dict[str, dict] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.context.set_tensor_address(name, int(device_mem))

            slot = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem,
            }

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = slot
            else:
                self.outputs[name] = slot

        if len(self.inputs) != 1:
            raise RuntimeError(f"Expected exactly 1 input, found: {list(self.inputs)}")

        self.input_name = next(iter(self.inputs))
        self.input_shape = self.inputs[self.input_name]["shape"]

        if len(self.input_shape) != 4:
            raise RuntimeError(f"Expected NCHW input shape, got: {self.input_shape}")

        _, _, self.input_h, self.input_w = self.input_shape

        if depth_name is not None:
            if depth_name not in self.outputs:
                raise RuntimeError(
                    f"Requested depth output {depth_name!r}, "
                    f"but outputs are {list(self.outputs)}"
                )
            self.depth_name = depth_name
        elif "depth" in self.outputs:
            self.depth_name = "depth"
        else:
            matches = [name for name in self.outputs if "depth" in name.lower()]
            self.depth_name = matches[0] if matches else next(iter(self.outputs))

        self.depth_output = self.outputs[self.depth_name]
        self.depth_shape = self.depth_output["shape"]

    def describe(self) -> str:
        lines = [
            f"engine: {self.engine_path}",
            f"input : {self.input_name} {self.input_shape}",
        ]
        for name, slot in self.outputs.items():
            suffix = " <- depth" if name == self.depth_name else ""
            lines.append(f"output: {name} {slot['shape']}{suffix}")
        lines.append(f"normalize: {self.normalize}")
        return "\n".join(lines)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_AREA,
        )

        img = rgb.astype(np.float32) / 255.0

        if self.normalize == "imagenet":
            img = (img - MEAN) / STD
        elif self.normalize == "zero_one":
            pass
        else:
            raise ValueError(f"Unsupported normalize mode: {self.normalize}")

        chw = np.transpose(img, (2, 0, 1))[None]
        return np.ascontiguousarray(chw)

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        img = self.preprocess(bgr)

        input_slot = self.inputs[self.input_name]
        output_slot = self.depth_output

        np.copyto(input_slot["host"], img.ravel())

        cuda.memcpy_htod_async(
            input_slot["device"],
            input_slot["host"],
            self.stream,
        )

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(
            output_slot["host"],
            output_slot["device"],
            self.stream,
        )

        self.stream.synchronize()

        return output_slot["host"].reshape(output_slot["shape"])


def depth_to_ros2_payload(depth: np.ndarray, frame_id: str = "camera_depth", encoding: str = "32FC1"):
    d = np.squeeze(depth).astype(np.float32, copy=False)

    if d.ndim != 2:
        raise ValueError(f"Expected 2D depth after squeeze, got shape {d.shape}")

    h, w = d.shape

    if encoding == "32FC1":
        out = d.astype(np.float32, copy=False)
        data = out.tobytes()
        step = w * 4

    elif encoding == "16UC1":
        out = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out * 1000.0, 0.0, 65535.0).astype(np.uint16)
        data = out.tobytes()
        step = w * 2

    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    if HAS_ROS2:
        msg = RosImage()
        msg.header.frame_id = frame_id
        msg.height = h
        msg.width = w
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = step
        msg.data = data
        return msg

    return {
        "header": {"frame_id": frame_id},
        "height": h,
        "width": w,
        "encoding": encoding,
        "is_bigendian": 0,
        "step": step,
        "data": data,
    }

def parse_model_arg(value: str) -> tuple[str, str]:
    if ":" not in value:
        path = value
        label = Path(value).stem
        return label, path

    label, path = value.split(":", 1)
    label = label.strip()
    path = path.strip()

    if not label:
        raise argparse.ArgumentTypeError(f"Invalid model label in {value!r}")
    if not path:
        raise argparse.ArgumentTypeError(f"Invalid model path in {value!r}")

    return label, path


def find_images(images_dir: str) -> list[str]:
    exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    files = sorted(
        p
        for p in glob.glob(os.path.join(images_dir, "*"))
        if p.rsplit(".", 1)[-1].lower() in exts
    )

    if not files:
        raise RuntimeError(f"No images found in: {images_dir}")

    return files


def benchmark_model(
    *,
    label: str,
    engine_path: str,
    files: list[str],
    normalize: str,
    depth_name: str | None,
    warmup: int,
    repeat: int,
    depth_encoding: str
) -> dict:
    print("\n" + "=" * 80)
    print(f"MODEL: {label}")
    print("=" * 80)

    engine = DA3Engine(
        engine_path,
        depth_name=depth_name,
        normalize=normalize,
    )

    print(engine.describe())

    first = cv2.imread(files[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read warmup image: {files[0]}")

    print(f"Warmup iterations: {warmup}")
    for _ in range(max(0, warmup)):
        _ = engine.infer(first)

    t_read: list[float] = []
    t_infer: list[float] = []
    t_ros: list[float] = []
    t_total_no_read: list[float] = []
    t_total_with_read: list[float] = []

    last_depth_shape = None

    print(f"Benchmark images: {len(files)}, repeat per image: {repeat}")

    for image_path in files:
        read_t0 = time.perf_counter()
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        read_t1 = time.perf_counter()

        if bgr is None:
            print(f"  skipped unreadable image: {image_path}")
            continue

        read_ms = (read_t1 - read_t0) * 1000.0
        t_read.append(read_ms)

        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            depth = engine.infer(bgr)
            t1 = time.perf_counter()

            _ = depth_to_ros2_payload(depth, encoding=depth_encoding)
            t2 = time.perf_counter()

            infer_ms = (t1 - t0) * 1000.0
            ros_ms = (t2 - t1) * 1000.0

            t_infer.append(infer_ms)
            t_ros.append(ros_ms)
            t_total_no_read.append(infer_ms + ros_ms)
            t_total_with_read.append(read_ms + infer_ms + ros_ms)

            last_depth_shape = tuple(np.squeeze(depth).shape)

    result = {
        "label": label,
        "engine_path": engine_path,
        "input_shape": engine.input_shape,
        "depth_shape": last_depth_shape,
        "normalize": normalize,
        "n_images": len(files),
        "repeat": repeat,
        "read": calc_stats(t_read),
        "infer": calc_stats(t_infer),
        "ros_pack": calc_stats(t_ros),
        "total_no_read": calc_stats(t_total_no_read),
        "total_with_read": calc_stats(t_total_with_read),
    }

    print_result_table(result)
    return result


def print_result_table(result: dict):
    rows = [
        ("Read image", result["read"]),
        ("Inference TRT", result["infer"]),
        ("ROS pack", result["ros_pack"]),
        ("Total no read", result["total_no_read"]),
        ("Total with read", result["total_with_read"]),
    ]

    print(f"depth shape: {result['depth_shape']}")
    print(
        f"{'Phase':<16} | {'mean ms':>9} {'std':>8} "
        f"{'min':>8} {'max':>8} | {'FPS':>8} | {'n':>5}"
    )
    print("-" * 78)

    for name, st in rows:
        print(
            f"{name:<16} | "
            f"{st.mean_ms:9.2f} {st.std_ms:8.2f} "
            f"{st.min_ms:8.2f} {st.max_ms:8.2f} | "
            f"{st.fps_from_mean:8.2f} | "
            f"{st.n:5d}"
        )


def write_outputs(results: list[dict], out_prefix: str):
    txt_path = out_prefix + ".txt"
    csv_path = out_prefix + ".csv"

    with open(txt_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write("=" * 80 + "\n")
            f.write(f"MODEL: {result['label']}\n")
            f.write(f"engine_path: {result['engine_path']}\n")
            f.write(f"input_shape: {result['input_shape']}\n")
            f.write(f"depth_shape: {result['depth_shape']}\n")
            f.write(f"normalize: {result['normalize']}\n")
            f.write(f"n_images: {result['n_images']}\n")
            f.write(f"repeat: {result['repeat']}\n\n")

            for phase_key, phase_name in [
                ("read", "Read image"),
                ("infer", "Inference TRT"),
                ("ros_pack", "ROS pack"),
                ("total_no_read", "Total no read"),
                ("total_with_read", "Total with read"),
            ]:
                st = result[phase_key]
                f.write(
                    f"{phase_name:<16} "
                    f"mean_ms={st.mean_ms:.4f}, "
                    f"std_ms={st.std_ms:.4f}, "
                    f"min_ms={st.min_ms:.4f}, "
                    f"max_ms={st.max_ms:.4f}, "
                    f"fps={st.fps_from_mean:.4f}, "
                    f"n={st.n}\n"
                )
            f.write("\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "engine_path",
            "input_shape",
            "depth_shape",
            "normalize",
            "n_images",
            "repeat",
            "phase",
            "mean_ms",
            "std_ms",
            "min_ms",
            "max_ms",
            "fps_from_mean",
            "n_samples",
        ])

        for result in results:
            for phase_key, phase_name in [
                ("read", "read_image"),
                ("infer", "inference_trt"),
                ("ros_pack", "ros_pack"),
                ("total_no_read", "total_no_read"),
                ("total_with_read", "total_with_read"),
            ]:
                st = result[phase_key]
                writer.writerow([
                    result["label"],
                    result["engine_path"],
                    result["input_shape"],
                    result["depth_shape"],
                    result["normalize"],
                    result["n_images"],
                    result["repeat"],
                    phase_name,
                    f"{st.mean_ms:.6f}",
                    f"{st.std_ms:.6f}",
                    f"{st.min_ms:.6f}",
                    f"{st.max_ms:.6f}",
                    f"{st.fps_from_mean:.6f}",
                    st.n,
                ])

    print(f"\nSaved: {txt_path}")
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder with RGB images")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=parse_model_arg,
        help="Model entry as label:/path/to/engine. Can be repeated.",
    )
    parser.add_argument(
        "--normalize",
        choices=["zero_one", "imagenet"],
        default="zero_one",
        help="Use zero_one to match current ROS wrapper, imagenet if ONNX export expects mean/std.",
    )
    parser.add_argument("--depth-name", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", default="/tmp/da3_metric_compare")
    parser.add_argument("--depth-encoding", choices=["32FC1", "16UC1"], default="32FC1")

    args = parser.parse_args()

    files = find_images(args.images)

    print(f"Images: {args.images}")
    print(f"Found: {len(files)}")
    print(f"ROS2 sensor_msgs available: {HAS_ROS2}")
    print(f"normalize: {args.normalize}")

    results = []
    for label, engine_path in args.model:
        results.append(
            benchmark_model(
                label=label,
                engine_path=engine_path,
                files=files,
                normalize=args.normalize,
                depth_name=args.depth_name,
                warmup=args.warmup,
                repeat=args.repeat,
                depth_encoding=args.depth_encoding,
            )
        )

    write_outputs(results, args.out)


if __name__ == "__main__":
    main()