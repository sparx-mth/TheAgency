#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def colorize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    depth_clean = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_clipped = np.clip(depth_clean, 0.0, max_depth_m)
    depth_norm = (depth_clipped / max_depth_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)


def get_matrix(data: dict, key: str, shape: tuple[int, int]) -> np.ndarray:
    values = data[key]["data"]
    return np.array(values, dtype=np.float64).reshape(shape)


def create_rectified_da3_yaml(calib_yaml_path: Path, output_yaml_path: Path) -> dict:
    """
    Create a DA3-friendly YAML using projection_matrix as the rectified pinhole intrinsics.

    This is the YAML that should be passed to DA3 when the image itself is already rectified.
    """
    calib = load_yaml(calib_yaml_path)

    width = int(calib["image_width"])
    height = int(calib["image_height"])

    P = get_matrix(calib, "projection_matrix", (3, 4))

    fx = float(P[0, 0])
    fy = float(P[1, 1])
    cx = float(P[0, 2])
    cy = float(P[1, 2])

    rectified = {
        "image_width": width,
        "image_height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion_model": "none",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0,
            ],
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [
                fx, 0.0, cx, 0.0,
                0.0, fy, cy, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        },
    }

    write_yaml(output_yaml_path, rectified)
    return rectified


def build_rectify_maps(calib_yaml_path: Path):
    calib = load_yaml(calib_yaml_path)

    width = int(calib["image_width"])
    height = int(calib["image_height"])

    K = get_matrix(calib, "camera_matrix", (3, 3))
    D = np.array(
        calib["distortion_coefficients"]["data"],
        dtype=np.float64,
    ).reshape(-1, 1)
    R = get_matrix(calib, "rectification_matrix", (3, 3))
    P = get_matrix(calib, "projection_matrix", (3, 4))
    K_rect = P[:3, :3].copy()

    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=D,
        R=R,
        newCameraMatrix=K_rect,
        size=(width, height),
        m1type=cv2.CV_16SC2,
    )

    return map1, map2, width, height


def rectify_bgr(bgr: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    return cv2.remap(
        bgr,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def find_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(input_dir.glob(f"*.{ext}"))
        paths.extend(input_dir.glob(f"*.{ext.upper()}"))
    return sorted(set(paths))


def finite_depth_stats(depth_m: np.ndarray) -> tuple[float, float, float, float]:
    finite = depth_m[np.isfinite(depth_m)]
    finite = finite[finite > 0.0]

    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    return (
        float(np.min(finite)),
        float(np.median(finite)),
        float(np.mean(finite)),
        float(np.max(finite)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create DA3 depth .npy and depth visualization PNG from an image folder."
    )

    parser.add_argument("--input-images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine-path", required=True)
    parser.add_argument("--calib-yaml", required=True)

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["jpg", "jpeg", "png"],
        help="Image extensions to read.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=15.0,
        help="Max depth used only for visualization.",
    )
    parser.add_argument(
        "--no-rectify",
        action="store_true",
        help="Use input images as-is. Default is to rectify using calibration YAML.",
    )
    parser.add_argument(
        "--resize-to-calib",
        action="store_true",
        help="Resize images to calibration resolution if they do not match.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to process. 0 means all.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_images_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    calib_yaml_path = Path(args.calib_yaml).expanduser()
    engine_path = Path(args.engine_path).expanduser()

    if not input_dir.exists():
        raise RuntimeError(f"Input image directory does not exist: {input_dir}")

    if not calib_yaml_path.exists():
        raise RuntimeError(f"Calibration YAML does not exist: {calib_yaml_path}")

    if not engine_path.exists():
        raise RuntimeError(f"DA3 engine does not exist: {engine_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_rect_dir = output_dir / "rgb_rectified"
    depth_npy_dir = output_dir / "depth_npy"
    depth_vis_dir = output_dir / "depth_vis"

    rgb_rect_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)

    raw_calib_copy = output_dir / "camera_raw_calibration.yaml"
    shutil.copyfile(calib_yaml_path, raw_calib_copy)

    if args.no_rectify:
        da3_yaml_path = output_dir / "camera_da3_input.yaml"
        shutil.copyfile(calib_yaml_path, da3_yaml_path)
        map1 = None
        map2 = None

        calib = load_yaml(calib_yaml_path)
        calib_width = int(calib["image_width"])
        calib_height = int(calib["image_height"])
    else:
        da3_yaml_path = output_dir / "camera_rectified_pinhole_da3.yaml"
        create_rectified_da3_yaml(calib_yaml_path, da3_yaml_path)
        map1, map2, calib_width, calib_height = build_rectify_maps(calib_yaml_path)

    image_paths = find_images(input_dir, args.extensions)

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")

    print(f"[depth] Input images: {input_dir}")
    print(f"[depth] Found images: {len(image_paths)}")
    print(f"[depth] Output dir: {output_dir}")
    print(f"[depth] DA3 engine: {engine_path}")
    print(f"[depth] DA3 YAML: {da3_yaml_path}")
    print(f"[depth] Rectify: {not args.no_rectify}")
    print(f"[depth] Calibration resolution: {calib_width}x{calib_height}")

    depth_model = DA3TensorRTModel(
        engine_path=str(engine_path),
        yaml_path=str(da3_yaml_path),
    )

    metadata_path = output_dir / "metadata_depth.csv"

    with open(metadata_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "frame_idx",
            "input_path",
            "rgb_rectified_path",
            "depth_npy_path",
            "depth_vis_path",
            "input_width",
            "input_height",
            "output_width",
            "output_height",
            "depth_width",
            "depth_height",
            "depth_min_m",
            "depth_median_m",
            "depth_mean_m",
            "depth_max_m",
        ])

        for idx, image_path in enumerate(image_paths):
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if bgr is None:
                print(f"[depth] Skipping unreadable image: {image_path}")
                continue

            input_h, input_w = bgr.shape[:2]

            if (input_w, input_h) != (calib_width, calib_height):
                if args.resize_to_calib:
                    print(
                        f"[depth] Resizing {image_path.name}: "
                        f"{input_w}x{input_h} -> {calib_width}x{calib_height}"
                    )
                    bgr = cv2.resize(
                        bgr,
                        (calib_width, calib_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    raise RuntimeError(
                        f"Image size mismatch for {image_path.name}: "
                        f"image={input_w}x{input_h}, "
                        f"calibration={calib_width}x{calib_height}. "
                        "Use matching images or pass --resize-to-calib."
                    )

            if args.no_rectify:
                bgr_out = bgr
            else:
                bgr_out = rectify_bgr(bgr, map1, map2)

            out_h, out_w = bgr_out.shape[:2]

            frame_name = image_path.stem

            rgb_out_path = rgb_rect_dir / f"{frame_name}.jpg"
            depth_npy_path = depth_npy_dir / f"{frame_name}.npy"
            depth_vis_path = depth_vis_dir / f"{frame_name}.png"

            depth_m = depth_model.infer_depth(bgr_out).astype(np.float32)
            depth_vis = colorize_depth(depth_m, max_depth_m=args.max_depth_m)

            cv2.imwrite(str(rgb_out_path), bgr_out)
            np.save(str(depth_npy_path), depth_m)
            cv2.imwrite(str(depth_vis_path), depth_vis)

            dmin, dmedian, dmean, dmax = finite_depth_stats(depth_m)

            writer.writerow([
                idx,
                str(image_path),
                str(rgb_out_path),
                str(depth_npy_path),
                str(depth_vis_path),
                input_w,
                input_h,
                out_w,
                out_h,
                int(depth_m.shape[1]),
                int(depth_m.shape[0]),
                dmin,
                dmedian,
                dmean,
                dmax,
            ])
            fp.flush()

            print(
                f"[depth] {idx + 1}/{len(image_paths)} {image_path.name} "
                f"rgb={out_w}x{out_h} "
                f"depth={depth_m.shape[1]}x{depth_m.shape[0]} "
                f"median={dmedian:.3f}m mean={dmean:.3f}m"
            )

    print(f"[depth] Done.")
    print(f"[depth] Saved metadata: {metadata_path}")
    print(f"[depth] Saved rectified DA3 YAML: {da3_yaml_path}")


if __name__ == "__main__":
    main()