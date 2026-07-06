"""Compare a fine-tuned NavDP checkpoint against the untrained baseline.

Runs BOTH models through the *same* torch inference path (loading the base weights,
then the fine-tuned EMA weights into one policy) on a set of (frame, pixel-goal)
samples, and scores each trajectory against the single-frame ESDF:

  * min clearance (m)        -- distance to the nearest wall along the path (higher safer)
  * waypoints in wall (%)    -- fraction with negative ESDF (lower safer)
  * dist to target (m)       -- closeness to the corrected+smoothed target we trained toward
  * smoothness (bending)     -- kinkiness (lower smoother; should not regress)
  * goal gap (m)             -- endpoint distance to the goal (should not regress)

Prints a baseline-vs-trained table and renders side-by-side BEV panels. Run in the
``navdp`` conda env:

    python -m sparx_agency.tasks.planning.finetune.train.evaluate \
        --rec walk_into --finetuned runs/walk_into/ema_latest.pth --out eval.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..verify import bev_render, pipeline  # noqa: E402
from ..verify.correction import correct_navdp_trajectory, make_config  # noqa: E402
from ..verify.navdp_infer import preprocess_depth, preprocess_rgb  # noqa: E402
from ..verify.pixel_goal import pixel_to_goal  # noqa: E402
from sparx_agency.tasks.planning.navdp.export.build_policy import build_navdp_policy  # noqa: E402


class TorchNavDP:
    """Torch NavDP point-goal inference with hot-swappable weights."""

    def __init__(self, ckpt: str, navdp_repo: str, device: str = "cuda", memory: int = 8):
        self.policy = build_navdp_policy(ckpt, navdp_repo, device=device)
        self.device, self.memory = device, memory

    def load_weights(self, pth: Path) -> None:
        state = torch.load(Path(pth).expanduser(), map_location=self.device)
        self.policy.load_state_dict(state, strict=False)

    def predict(self, rgb_bgr: np.ndarray, depth_m: np.ndarray, goal_body) -> np.ndarray:
        frame = preprocess_rgb(rgb_bgr, 224)
        images = np.tile(frame[None, None], (1, self.memory, 1, 1, 1)).astype(np.float32)
        dep = preprocess_depth(depth_m, 224)[None].astype(np.float32)
        goal = np.array([[goal_body[0], goal_body[1], 0.0]], np.float32).clip(-10, 10)
        goal[:, 0] = np.clip(goal[:, 0], 0.0, 10.0)
        np.random.seed(0)
        with torch.no_grad():
            _all, _crit, pos, _neg = self.policy.predict_pointgoal_action(goal, images, dep)
        pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        return pos[0, 0, :, :2].astype(np.float32)


def _sample_esdf(sdf, res, ox, oy, xy):
    gx = np.clip(((xy[:, 0] - ox) / res).astype(int), 0, sdf.shape[1] - 1)
    gy = np.clip(((xy[:, 1] - oy) / res).astype(int), 0, sdf.shape[0] - 1)
    return sdf[gy, gx]


def _bending(xy):
    d2 = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    return float(np.sum(np.linalg.norm(d2, axis=1)))


def _metrics(traj, target_xy, goal, sdf, res, ox, oy):
    clr = _sample_esdf(sdf, res, ox, oy, traj)
    n = min(len(traj), len(target_xy))
    return {
        "min_clear": float(clr.min()),
        "frac_wall": float((clr < 0).mean()),
        "dist_target": float(np.mean(np.linalg.norm(traj[:n] - target_xy[:n], axis=1))),
        "bending": _bending(traj),
        "goal_gap": float(np.hypot(traj[-1, 0] - goal[0], traj[-1, 1] - goal[1])),
    }


def evaluate(dataset: Path, rec: str, base_ckpt: str, navdp_repo: str, finetuned: Path,
             n_frames: int, n_goals: int, out: Path, pitch: float, height: float,
             clearance: float, device: str, exclude_bottom_frac: float = 0.2) -> None:
    rec_dir = dataset / rec
    intr = pipeline.load_intrinsics(rec_dir)
    frames = pipeline.list_frames(rec_dir)
    pick = frames[:: max(1, len(frames) // n_frames)][:n_frames]
    cfg = make_config(corrector="esdf", target_clearance_m=clearance,
                      pitch_deg=pitch, camera_height_m=height)

    model = TorchNavDP(base_ckpt, navdp_repo, device)

    # pass 1: baseline -> also build the corrected+smoothed target per sample
    samples = []
    for frame in pick:
        bgr, depth = pipeline.load_frame(rec_dir, frame)
        for (u, v) in pipeline.sample_valid_pixels(depth, n_goals, seed=777 + frame,
                                                   exclude_bottom_frac=exclude_bottom_frac):
            try:
                goal = pixel_to_goal(u, v, depth, intr)[:2]
            except ValueError:
                continue
            traj_b = model.predict(bgr, depth, goal)
            tgt = correct_navdp_trajectory(traj_b, depth, intr, goal, cfg)
            target_xy = bev_render.path_xy(tgt.corrected_path)[1:]      # drop origin
            samples.append({"frame": frame, "bgr": bgr, "depth": depth, "goal": goal,
                            "traj_b": traj_b, "target": tgt, "target_xy": target_xy})

    # pass 2: fine-tuned weights, same samples
    model.load_weights(finetuned)
    for s in samples:
        s["traj_t"] = model.predict(s["bgr"], s["depth"], s["goal"])

    # metrics
    agg = {"baseline": [], "trained": []}
    for s in samples:
        occ = s["target"].occupancy
        args = (s["target"].sdf_m, occ.resolution, occ.origin_x, occ.origin_y)
        agg["baseline"].append(_metrics(s["traj_b"], s["target_xy"], s["goal"], *args))
        agg["trained"].append(_metrics(s["traj_t"], s["target_xy"], s["goal"], *args))

    keys = ["min_clear", "frac_wall", "dist_target", "bending", "goal_gap"]
    summary = {w: {k: float(np.mean([m[k] for m in agg[w]])) for k in keys} for w in agg}
    better = {"min_clear": "higher", "frac_wall": "lower", "dist_target": "lower",
              "bending": "lower", "goal_gap": "lower"}
    print(f"\n{rec}: {len(samples)} samples  (better = safer/closer)")
    print(f"{'metric':<14}{'baseline':>12}{'trained':>12}{'Δ':>12}   want")
    for k in keys:
        b, t = summary["baseline"][k], summary["trained"][k]
        print(f"{k:<14}{b:>12.3f}{t:>12.3f}{t - b:>+12.3f}   {better[k]}")

    _render_panels(samples[:6], out)
    out.with_suffix(".json").write_text(json.dumps(
        {"rec": rec, "n_samples": len(samples), "summary": summary,
         "finetuned": str(finetuned)}, indent=2))
    print("saved", out, "and", out.with_suffix(".json"))


def _render_panels(samples, out: Path) -> None:
    n = len(samples)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, s in enumerate(samples):
        ax = axes[k // cols][k % cols]
        ax.axis("on")
        bev_render.draw_bev(ax, s["target"], s["goal"])
        ax.plot(s["traj_t"][:, 0], s["traj_t"][:, 1], "o-", color="deepskyblue",
                ms=3, lw=2, label="trained", zorder=5)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("frame %d  goal(%.1f,%.1f)" % (s["frame"], s["goal"][0], s["goal"][1]),
                     fontsize=9)
    fig.suptitle("baseline NavDP (orange) vs trained (blue) vs target (green)", fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path.home() / "flight_dataset")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--finetuned", type=Path, required=True, help="ema_latest.pth")
    ap.add_argument("--ckpt", type=Path, default=Path.home() / "Downloads/navdp-cross-modal.ckpt")
    ap.add_argument("--navdp-repo", type=Path, default=Path.home() / "PycharmProjects/NavDP/baselines/navdp")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--n-goals", type=int, default=6)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--clearance", type=float, default=0.5)
    ap.add_argument("--exclude-bottom-frac", type=float, default=0.2,
                    help="match the training goal region (drop bottom fraction of rows)")
    ap.add_argument("--out", type=Path, default=Path("eval.png"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    evaluate(args.dataset.expanduser(), args.rec, str(args.ckpt.expanduser()),
             str(args.navdp_repo.expanduser()), args.finetuned, args.n_frames,
             args.n_goals, args.out.expanduser(), args.pitch, args.height,
             args.clearance, args.device, args.exclude_bottom_frac)


if __name__ == "__main__":
    main()
