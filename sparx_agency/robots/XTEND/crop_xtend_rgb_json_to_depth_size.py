#!/usr/bin/env python3
"""
Crop paired XTEND RGB JPG + JSON files to a target image size.

Default target is 504x280, matching the DA3 depth output you are seeing:
  rgb=720x420 depth=504x280

Input:
  input_dir/
    R2_20260504_111538_0.jpg
    R2_20260504_111538_0.json
    ...

Output:
  output_dir/
    R2_20260504_111538_0.jpg
    R2_20260504_111538_0.json
    crop_info.json

The JSON sidecar is copied and its "image" field is updated to the output JPG name.

Optional:
  If you pass --calib-yaml and --output-calib-yaml, the script also writes a
  cropped calibration YAML where cx/cy and image_width/image_height are updated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import yaml


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def center_crop_box(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    if dst_w > src_w or dst_h > src_h:
        raise ValueError(
            f"Target crop {dst_w}x{dst_h} is larger than source image {src_w}x{src_h}"
        )

    x0 = (src_w - dst_w) // 2
    y0 = (src_h - dst_h) // 2
    x1 = x0 + dst_w
    y1 = y0 + dst_h
    return x0, y0, x1, y1


def explicit_crop_box(
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    crop_x: int | None,
    crop_y: int | None,
) -> tuple[int, int, int, int]:
    if crop_x is None or crop_y is None:
        return center_crop_box(src_w, src_h, dst_w, dst_h)

    x0 = int(crop_x)
    y0 = int(crop_y)
    x1 = x0 + dst_w
    y1 = y0 + dst_h

    if x0 < 0 or y0 < 0 or x1 > src_w or y1 > src_h:
        raise ValueError(
            f"Crop box x={x0}:{x1}, y={y0}:{y1} is outside source image {src_w}x{src_h}"
        )

    return x0, y0, x1, y1


def update_calib_for_crop(
    calib: dict[str, Any],
    crop_x: int,
    crop_y: int,
    target_w: int,
    target_h: int,
) -> dict[str, Any]:
    """
    Update a ROS camera calibration YAML after cropping.

    Cropping does not change fx/fy.
    Cropping shifts the principal point:
      cx_new = cx_old - crop_x
      cy_new = cy_old - crop_y
    """
    out = json.loads(json.dumps(calib))

    out["image_width"] = int(target_w)
    out["image_height"] = int(target_h)

    if "camera_matrix" in out and "data" in out["camera_matrix"]:
        k = list(out["camera_matrix"]["data"])
        k[2] = float(k[2]) - float(crop_x)
        k[5] = float(k[5]) - float(crop_y)
        out["camera_matrix"]["data"] = k

    if "projection_matrix" in out and "data" in out["projection_matrix"]:
        p = list(out["projection_matrix"]["data"])
        p[2] = float(p[2]) - float(crop_x)
        p[6] = float(p[6]) - float(crop_y)
        out["projection_matrix"]["data"] = p

    return out


def find_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(input_dir.glob(f"*.{ext}"))
        paths.extend(input_dir.glob(f"*.{ext.upper()}"))
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Center-crop paired XTEND JPG+JSON frames to a fixed size."
    )

    parser.add_argument("--input-dir", required=True, help="Folder with paired JPG/JSON files.")
    parser.add_argument("--output-dir", required=True, help="Output folder for cropped JPG/JSON files.")

    parser.add_argument("--target-width", type=int, default=504)
    parser.add_argument("--target-height", type=int, default=280)

    parser.add_argument(
        "--crop-x",
        type=int,
        default=None,
        help="Optional left crop offset. Default: center crop.",
    )
    parser.add_argument(
        "--crop-y",
        type=int,
        default=None,
        help="Optional top crop offset. Default: center crop.",
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["jpg", "jpeg", "png"],
        help="Image extensions to process.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved cropped images.",
    )

    parser.add_argument(
        "--copy-missing-json",
        action="store_true",
        help="If an image has no JSON sidecar, still save the cropped image and a default JSON.",
    )

    parser.add_argument(
        "--calib-yaml",
        default=None,
        help="Optional original calibration YAML to update for the crop.",
    )
    parser.add_argument(
        "--output-calib-yaml",
        default=None,
        help="Optional output calibration YAML path. Requires --calib-yaml.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(input_dir, args.extensions)
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    src_h, src_w = first_img.shape[:2]
    crop_x0, crop_y0, crop_x1, crop_y1 = explicit_crop_box(
        src_w=src_w,
        src_h=src_h,
        dst_w=args.target_width,
        dst_h=args.target_height,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
    )

    print(f"[crop] Input dir: {input_dir}")
    print(f"[crop] Output dir: {output_dir}")
    print(f"[crop] Source size: {src_w}x{src_h}")
    print(f"[crop] Target size: {args.target_width}x{args.target_height}")
    print(f"[crop] Crop box: x={crop_x0}:{crop_x1}, y={crop_y0}:{crop_y1}")

    saved = 0
    skipped = 0

    for image_path in image_paths:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[crop] skip unreadable: {image_path.name}")
            skipped += 1
            continue

        h, w = bgr.shape[:2]
        if (w, h) != (src_w, src_h):
            print(
                f"[crop] skip size mismatch: {image_path.name} "
                f"got {w}x{h}, expected {src_w}x{src_h}"
            )
            skipped += 1
            continue

        cropped = bgr[crop_y0:crop_y1, crop_x0:crop_x1].copy()

        out_image_path = output_dir / image_path.name
        ok = cv2.imwrite(
            str(out_image_path),
            cropped,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
        )
        if not ok:
            print(f"[crop] failed writing image: {out_image_path}")
            skipped += 1
            continue

        json_path = image_path.with_suffix(".json")
        out_json_path = output_dir / json_path.name

        if json_path.exists():
            data = load_json(json_path)
            data["image"] = out_image_path.name
            write_json(out_json_path, data)
        elif args.copy_missing_json:
            write_json(
                out_json_path,
                {
                    "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "image": out_image_path.name,
                },
            )
        else:
            print(f"[crop] warning: missing JSON for {image_path.name}")

        saved += 1

    crop_info = {
        "source_width": src_w,
        "source_height": src_h,
        "target_width": args.target_width,
        "target_height": args.target_height,
        "crop_x": crop_x0,
        "crop_y": crop_y0,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "saved": saved,
        "skipped": skipped,
    }
    write_json(output_dir / "crop_info.json", crop_info)

    if args.calib_yaml or args.output_calib_yaml:
        if not args.calib_yaml or not args.output_calib_yaml:
            raise RuntimeError("Use both --calib-yaml and --output-calib-yaml together.")

        calib = load_yaml(Path(args.calib_yaml).expanduser().resolve())
        cropped_calib = update_calib_for_crop(
            calib=calib,
            crop_x=crop_x0,
            crop_y=crop_y0,
            target_w=args.target_width,
            target_h=args.target_height,
        )
        write_yaml(Path(args.output_calib_yaml).expanduser().resolve(), cropped_calib)
        print(f"[crop] Wrote cropped calibration YAML: {args.output_calib_yaml}")

    print(f"[crop] Done. saved={saved}, skipped={skipped}")


if __name__ == "__main__":
    main()
