"""Render baseline vs trained routes over the fused judge map.

The paired statistics say *whether* the routes differ; this says *how*. Each
panel is one (frame, goal) sample drawn in the camera's body frame over the
fused multi-frame clearance field, with the obstacle boundary marked so the
reader can see which route hugs a wall.

    python -m sparx_agency.tasks.planning.finetune.eval.routes_figure \
        --rec walk_into --trained ~/Downloads/flight_dataset/run_new/best.pth \
        --out ~/Downloads/flight_dataset/run_new/eval/routes.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..verify import pipeline  # noqa: E402
from ..verify.pixel_goal import pixel_to_goal  # noqa: E402
from .bag_poses import load_frame_poses  # noqa: E402
from .compare import _intrinsics_matrix, _load_models  # noqa: E402
from .judge_map import JudgeMapConfig, body_to_world, build_judge_field  # noqa: E402
from .metrics import densify  # noqa: E402

COLORS = {"baseline": "#eb6834", "trained": "#2a78d6"}
EXTENT_M = 5.0


def _panel(ax, field, pose, traj, goal, d_safe):
    """Draw one sample's two routes over the fused clearance field."""
    cam = pose[:2, 3]
    half = EXTENT_M
    x0, x1 = cam[0] - half, cam[0] + half
    y0, y1 = cam[1] - half, cam[1] + half
    gx0 = int((x0 - field.origin_x) / field.resolution)
    gy0 = int((y0 - field.origin_y) / field.resolution)
    gx1 = int((x1 - field.origin_x) / field.resolution)
    gy1 = int((y1 - field.origin_y) / field.resolution)
    sub = field.clearance[gy0:gy1, gx0:gx1]
    seen = field.observed[gy0:gy1, gx0:gx1]
    shown = np.where(seen, np.clip(sub, 0, 2.0), np.nan)

    ax.imshow(shown, extent=[x0, x1, y0, y1], origin="lower", cmap="RdYlBu",
              vmin=0, vmax=2.0, aspect="equal", zorder=0)
    ax.contour(np.clip(sub, 0, 2.0), levels=[d_safe],
               extent=[x0, x1, y0, y1], origin="lower",
               colors="k", linewidths=0.8, linestyles="--", zorder=2)

    for arm in ("baseline", "trained"):
        w = body_to_world(traj[arm], pose)
        ax.plot(w[:, 0], w[:, 1], "-", color=COLORS[arm], lw=2.4, zorder=4,
                alpha=0.95, label=arm)
    gw = body_to_world(np.asarray(goal, float)[None], pose)[0]
    ax.plot(cam[0], cam[1], "^", color="k", ms=9, zorder=6)
    ax.plot(gw[0], gw[1], "*", color="magenta", ms=16, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])


def render(dataset: Path, bag_root: Path, rec: str, base_ckpt: Path, repo: Path,
           trained: Path, out: Path, n_panels: int, clearance: float,
           max_shift: float, d_safe: float, window: int, device: str,
           exclude_bottom_frac: float, seed: int) -> None:
    """Render a grid of route-comparison panels to ``out``."""
    rec_dir = dataset / rec
    intr = pipeline.load_intrinsics(rec_dir)
    K = _intrinsics_matrix(intr)
    posed_idx, poses = load_frame_poses(bag_root, rec)
    available = set(pipeline.list_frames(rec_dir))
    usable = [i for i, f in enumerate(posed_idx) if int(f) in available]
    picks = usable[:: max(1, len(usable) // n_panels)][:n_panels]

    base, tuned = _load_models(base_ckpt, repo, trained, device)
    judge_cfg = JudgeMapConfig(window=window)

    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for k, pos in enumerate(picks):
        frame = int(posed_idx[pos])
        bgr, depth = pipeline.load_frame(rec_dir, frame)
        field = build_judge_field(rec_dir, K, posed_idx, poses, pos, judge_cfg)
        pix = pipeline.sample_valid_pixels(depth, 1, seed=seed + frame,
                                           exclude_bottom_frac=exclude_bottom_frac)
        if not len(pix):
            continue
        u, v = pix[0]
        try:
            goal = pixel_to_goal(u, v, depth, intr)[:2]
        except ValueError:
            continue
        traj = {"baseline": base.predict(bgr, depth, goal),
                "trained": tuned.predict(bgr, depth, goal)}
        ax = axes[k // cols][k % cols]
        ax.axis("on")
        _panel(ax, field, poses[pos], traj, goal, d_safe)
        ax.set_title(f"frame {frame}", fontsize=9)
        print(f"  panel {k}: frame {frame}", flush=True)

    handles = [plt.Line2D([], [], color=c, lw=2.4, label=a) for a, c in COLORS.items()]
    handles.append(plt.Line2D([], [], color="k", ls="--", lw=0.8,
                              label=f"{d_safe:.2f} m safety contour"))
    handles.append(plt.Line2D([], [], color="magenta", marker="*", ls="",
                              ms=12, label="goal"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{rec} (held out): routes over the fused multi-frame clearance map\n"
                 "warm = near an obstacle, cool = open, grey = never observed",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path.home() / "Downloads/flight_dataset")
    ap.add_argument("--bag-root", type=Path, default=Path.home() / "Videos")
    ap.add_argument("--rec", default="walk_into")
    ap.add_argument("--trained", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=Path.home() / "GIT/NavDP/baselines/navdp/checkpoints/navdp-cross-modal.ckpt")
    ap.add_argument("--navdp-repo", type=Path, default=Path.home() / "GIT/NavDP/baselines/navdp")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-panels", type=int, default=8)
    ap.add_argument("--clearance", type=float, default=0.3)
    ap.add_argument("--max-shift", type=float, default=0.2)
    ap.add_argument("--d-safe", type=float, default=0.30)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--exclude-bottom-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    render(args.dataset.expanduser(), args.bag_root.expanduser(), args.rec,
           args.ckpt.expanduser(), args.navdp_repo.expanduser(),
           args.trained.expanduser(), args.out.expanduser(), args.n_panels,
           args.clearance, args.max_shift, args.d_safe, args.window,
           args.device, args.exclude_bottom_frac, args.seed)


if __name__ == "__main__":
    main()
