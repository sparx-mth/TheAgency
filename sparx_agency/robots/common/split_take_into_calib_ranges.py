#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


# label, start_idx, end_idx inclusive
RANGES = [
    ("4_0", 40, 190),
    ("3_5", 240, 350),
    ("3_0", 420, 510),
    ("2_5", 530, 655),
    ("2_0", 675, 750),
    ("1_5", 765, 850),
    ("1_0", 900, 950),
    ("0_5", 980, 1090),

    ("2_5", 1740, 1850),
    ("2_0", 1890, 1940),
    ("1_5", 1990, 2040),
    ("1_0", 2070, 2130),
    ("0_5", 2155, 2255),

    ("4_0", 2750, 2840),
    ("3_5", 2890, 2940),
    ("3_0", 2990, 3050),
    ("2_5", 3085, 3135),
    ("2_0", 3180, 3220),
    ("1_5", 3280, 3340),
]


def frame_name(idx: int, suffix: str) -> str:
    return f"frame_{idx:06d}{suffix}"


def copy_or_symlink(src: Path, dst: Path, use_symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return

    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--take-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--rgb-ext", default=".jpg")
    parser.add_argument("--depth-ext", default=".npy")
    args = parser.parse_args()

    take_dir = Path(args.take_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    rgb_src_dir = take_dir / "rgb"
    depth_src_dir = take_dir / "depth_npy"

    if not rgb_src_dir.exists():
        raise FileNotFoundError(f"Missing RGB dir: {rgb_src_dir}")

    if not depth_src_dir.exists():
        raise FileNotFoundError(f"Missing depth dir: {depth_src_dir}")

    copied = 0
    missing_rgb = []
    missing_depth = []

    for seg_idx, (label, start_idx, end_idx) in enumerate(RANGES):
        segment_name = f"{label}_seg{seg_idx:02d}"

        rgb_dst_dir = out_dir / "rgb" / segment_name
        depth_dst_dir = out_dir / "depth" / segment_name

        for idx in range(start_idx, end_idx + 1):
            rgb_src = rgb_src_dir / frame_name(idx, args.rgb_ext)
            depth_src = depth_src_dir / frame_name(idx, args.depth_ext)

            rgb_dst = rgb_dst_dir / rgb_src.name
            depth_dst = depth_dst_dir / depth_src.name

            if not rgb_src.exists():
                missing_rgb.append(str(rgb_src))
                continue

            if not depth_src.exists():
                missing_depth.append(str(depth_src))
                continue

            copy_or_symlink(rgb_src, rgb_dst, args.symlink)
            copy_or_symlink(depth_src, depth_dst, args.symlink)
            copied += 1

    print(f"Done. Copied/linked pairs: {copied}")
    print(f"Output: {out_dir}")

    if missing_rgb:
        print(f"\nMissing RGB files: {len(missing_rgb)}")
        for p in missing_rgb[:20]:
            print("  ", p)
        if len(missing_rgb) > 20:
            print("  ...")

    if missing_depth:
        print(f"\nMissing depth files: {len(missing_depth)}")
        for p in missing_depth[:20]:
            print("  ", p)
        if len(missing_depth) > 20:
            print("  ...")


if __name__ == "__main__":
    main()