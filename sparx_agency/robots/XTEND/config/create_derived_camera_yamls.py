#!/usr/bin/env python3

import argparse
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def padded_yaml(base, pad_left, pad_top, new_width, new_height):
    out = dict(base)

    out["image_width"] = int(new_width)
    out["image_height"] = int(new_height)

    k = list(base["camera_matrix"]["data"])
    p = list(base["projection_matrix"]["data"])

    k[2] = float(k[2]) + float(pad_left)
    k[5] = float(k[5]) + float(pad_top)

    p[2] = float(p[2]) + float(pad_left)
    p[6] = float(p[6]) + float(pad_top)

    out["camera_matrix"] = dict(base["camera_matrix"])
    out["projection_matrix"] = dict(base["projection_matrix"])

    out["camera_matrix"]["data"] = k
    out["projection_matrix"]["data"] = p

    return out


def crop_resize_yaml(
    base,
    crop_left,
    crop_top,
    crop_width,
    crop_height,
    new_width,
    new_height,
):
    out = dict(base)

    sx = float(new_width) / float(crop_width)
    sy = float(new_height) / float(crop_height)

    out["image_width"] = int(new_width)
    out["image_height"] = int(new_height)

    k = list(base["camera_matrix"]["data"])
    p = list(base["projection_matrix"]["data"])

    # K:
    # [fx 0 cx]
    # [0 fy cy]
    # [0  0  1]
    k[0] = float(k[0]) * sx
    k[2] = (float(k[2]) - float(crop_left)) * sx
    k[4] = float(k[4]) * sy
    k[5] = (float(k[5]) - float(crop_top)) * sy

    # P:
    # [fx 0 cx Tx]
    # [0 fy cy Ty]
    # [0  0  1  0]
    p[0] = float(p[0]) * sx
    p[2] = (float(p[2]) - float(crop_left)) * sx
    p[3] = float(p[3]) * sx

    p[5] = float(p[5]) * sy
    p[6] = (float(p[6]) - float(crop_top)) * sy
    p[7] = float(p[7]) * sy

    out["camera_matrix"] = dict(base["camera_matrix"])
    out["projection_matrix"] = dict(base["projection_matrix"])

    out["camera_matrix"]["data"] = k
    out["projection_matrix"]["data"] = p

    return out


def print_intrinsics(label, data):
    k = data["camera_matrix"]["data"]
    p = data["projection_matrix"]["data"]

    print(f"\n{label}")
    print(f"  size: {data['image_width']}x{data['image_height']}")
    print(f"  K: fx={k[0]:.3f}, fy={k[4]:.3f}, cx={k[2]:.3f}, cy={k[5]:.3f}")
    print(f"  P: fx={p[0]:.3f}, fy={p[5]:.3f}, cx={p[2]:.3f}, cy={p[6]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-yaml", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_yaml).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    base = load_yaml(base_path)

    yaml_728 = padded_yaml(
        base,
        pad_left=4,
        pad_top=0,
        new_width=728,
        new_height=420,
    )

    yaml_504_392 = crop_resize_yaml(
        base,
        crop_left=90,
        crop_top=0,
        crop_width=540,
        crop_height=420,
        new_width=504,
        new_height=392,
    )

    out_728 = out_dir / "camera_xtend_ros_calib_728_420_padded.yaml"
    out_504 = out_dir / "camera_xtend_ros_calib_504_392_crop_resize.yaml"

    save_yaml(yaml_728, out_728)
    save_yaml(yaml_504_392, out_504)

    print_intrinsics("Base", base)
    print_intrinsics("Padded 728x420", yaml_728)
    print_intrinsics("Crop-resize 504x392", yaml_504_392)

    print(f"\nSaved:\n  {out_728}\n  {out_504}")


if __name__ == "__main__":
    main()