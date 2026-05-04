#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_gt_poses(folder: Path):
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


def load_estimated_poses(filepath: Path):
    poses = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            pose = item.get("pose", {})
            poses.append({
                "x": float(pose.get("x", 0.0)),
                "y": float(pose.get("y", 0.0)),
                "z": float(pose.get("z", 0.0))
            })
    except Exception as exc:
        print(f"[WARN] failed reading estimated JSON {filepath}: {exc}")
        
    return poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing paired JPG/JSON files (Ground Truth)")
    parser.add_argument("--estimated", default=None, help="Path to the estimated trajectory JSON file")
    parser.add_argument("--save", default=None, help="Optional output PNG path")
    parser.add_argument("--show-index", action="store_true", help="Draw frame index numbers")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    gt_poses = load_gt_poses(folder)

    if not gt_poses:
        raise RuntimeError(f"No GT poses found in {folder}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_xs, all_ys, all_zs = [], [], []

    xs = [p["x"] for p in gt_poses]
    ys = [p["y"] for p in gt_poses]
    zs = [p["z"] for p in gt_poses]
    
    all_xs.extend(xs)
    all_ys.extend(ys)
    all_zs.extend(zs)

    ax.plot(xs, ys, zs, marker="o", markersize=3, linewidth=1.5, label="Ground Truth", color="blue", alpha=0.7)
    ax.scatter(xs[0], ys[0], zs[0], color="blue", marker="o", s=80, label="GT Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="blue", marker="x", s=100, label="GT End")

    if args.show_index:
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax.text(x, y, z, str(i), fontsize=8, color="blue")

    if args.estimated:
        est_path = Path(args.estimated).expanduser().resolve()
        if est_path.exists():
            est_poses = load_estimated_poses(est_path)
            if est_poses:
                exs = [p["x"] for p in est_poses]
                eys = [p["y"] for p in est_poses]
                ezs = [p["z"] for p in est_poses]

                all_xs.extend(exs)
                all_ys.extend(eys)
                all_zs.extend(ezs)

                ax.plot(exs, eys, ezs, marker="^", markersize=3, linewidth=1.5, label="Estimated (VO)", color="darkorange", alpha=0.9)
                ax.scatter(exs[0], eys[0], ezs[0], color="darkorange", marker="o", s=80, label="Est Start")
                ax.scatter(exs[-1], eys[-1], ezs[-1], color="darkorange", marker="x", s=100, label="Est End")
        else:
            print(f"[ERROR] Estimated file not found: {est_path}")

    ax.set_title(f"Trajectory Comparison: {folder.name}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()

    max_range = max(
        max(all_xs) - min(all_xs),
        max(all_ys) - min(all_ys),
        max(all_zs) - min(all_zs),
        1e-6,
    )
    mid_x = 0.5 * (max(all_xs) + min(all_xs))
    mid_y = 0.5 * (max(all_ys) + min(all_ys))
    mid_z = 0.5 * (max(all_zs) + min(all_zs))

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