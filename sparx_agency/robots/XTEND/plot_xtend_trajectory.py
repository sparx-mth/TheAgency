#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_poses(folder: Path):
    poses = []

    for json_path in sorted(folder.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            pose = data.get("pose", {})
            image = data.get("image", json_path.with_suffix(".jpg").name)

            x = float(pose.get("x", 0.0))
            y = float(pose.get("y", 0.0))
            z = float(pose.get("z", 0.0))
            yaw = float(pose.get("yaw", 0.0))

            poses.append({
                "json": json_path.name,
                "image": image,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
            })

        except Exception as exc:
            print(f"[WARN] failed reading {json_path}: {exc}")

    return poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing paired JPG/JSON files")
    parser.add_argument("--save", default=None, help="Optional output PNG path")
    parser.add_argument("--show-index", action="store_true", help="Draw frame index numbers")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    poses = load_poses(folder)

    if not poses:
        raise RuntimeError(f"No poses found in {folder}")

    xs = [p["x"] for p in poses]
    ys = [p["y"] for p in poses]
    zs = [p["z"] for p in poses]

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, marker="o", linewidth=1)

    ax.scatter(xs[0], ys[0], zs[0], marker="o", s=80, label="start")
    ax.scatter(xs[-1], ys[-1], zs[-1], marker="x", s=100, label="end")

    if args.show_index:
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax.text(x, y, z, str(i), fontsize=8)

    ax.set_title(f"XTEND trajectory: {folder.name}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()

    # Keep axes roughly equal scale
    max_range = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        1e-6,
    )
    mid_x = 0.5 * (max(xs) + min(xs))
    mid_y = 0.5 * (max(ys) + min(ys))
    mid_z = 0.5 * (max(zs) + min(zs))

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    plt.tight_layout()

    if args.save:
        out_path = Path(args.save).expanduser().resolve()
        fig.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()