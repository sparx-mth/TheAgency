"""Driver: score baseline / trained / teacher NavDP routes on a held-out recording.

Three arms answer the same (frame, pixel-goal) pairs, and all three are scored
against the fused multi-frame judge field rather than the single-frame ESDF the
teacher optimized against:

* **baseline**  -- the pretrained NavDP checkpoint.
* **trained**   -- your fine-tuned weights.
* **teacher**   -- ``correct_navdp_trajectory`` applied to the baseline route.
  This is the ceiling the fine-tune was imitating. Because the corrector is
  ``lateral_only`` and capped at ``max_total_shift_m``, that ceiling is modest,
  and knowing it tells you what fraction of the available gain the network
  actually captured.

Writes a per-sample CSV, a JSON summary with paired statistics, and prints the
comparison table. Run in the ``navdp`` conda env:

    python -m sparx_agency.tasks.planning.finetune.eval.compare \
        --rec walk_into --trained ~/Downloads/flight_dataset/run_new/best.pth \
        --out-dir ~/Downloads/flight_dataset/run_new/eval
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

from ..train.navdp_torch import TorchNavDP
from ..verify import bev_render, pipeline
from ..verify.correction import correct_navdp_trajectory, make_config
from ..verify.pixel_goal import pixel_to_goal
from . import stats
from .bag_poses import load_frame_poses
from .judge_map import JudgeMapConfig, body_to_world, build_judge_field
from .metrics import HIGHER_IS_BETTER, score

ARMS = ("baseline", "trained", "teacher")
REPORT_METRICS = ("min_clearance_m", "p5_clearance_m", "mean_clearance_m",
                  "frac_below_safe", "goal_gap_m", "bending")


def _intrinsics_matrix(intr) -> np.ndarray:
    """``Intrinsics`` -> the ``(3, 3)`` K matrix ``update_grid_from_depth`` wants."""
    return np.array([[intr.fx, 0.0, intr.cx],
                     [0.0, intr.fy, intr.cy],
                     [0.0, 0.0, 1.0]], dtype=float)


def _load_models(base_ckpt: Path, repo: Path, trained: Path, device: str):
    """Both policies, held in memory simultaneously (they fit on a 24 GB card).

    ``TorchNavDP.load_weights`` uses ``strict=False``, which silently tolerates a
    checkpoint that shares no keys with the model -- that would make the two arms
    identical and the whole comparison vacuous. So verify the load actually
    changed the weights.
    """
    base = TorchNavDP(str(base_ckpt), str(repo), device)
    tuned = TorchNavDP(str(base_ckpt), str(repo), device)
    before = [p.detach().clone() for p in tuned.policy.parameters()]
    tuned.load_weights(trained)
    after = list(tuned.policy.parameters())
    if all(bool((b == a).all()) for b, a in zip(before, after)):
        raise ValueError(
            f"{trained}: loading changed no weights (load_state_dict is strict=False, "
            "so a mismatched checkpoint loads silently). The comparison would be "
            "baseline-vs-baseline.")
    return base, tuned


def run(dataset: Path, bag_root: Path, rec: str, base_ckpt: Path, repo: Path,
        trained: Path, out_dir: Path, n_frames: int, n_goals: int,
        clearance: float, max_shift: float, d_safe: float, window: int,
        device: str, exclude_bottom_frac: float, seed: int,
        occ_prob: float = 0.65) -> dict:
    """Run the three-arm comparison and write CSV + JSON.

    Args:
        dataset: root holding the extracted recordings.
        bag_root: root holding the source rosbags (for poses).
        rec: recording name; must be one the model never trained on.
        base_ckpt: pretrained NavDP checkpoint.
        repo: NavDP baseline repo path.
        trained: fine-tuned checkpoint to evaluate.
        out_dir: directory for ``per_sample.csv`` and ``summary.json``.
        n_frames: how many frames to sample across the recording.
        n_goals: pixel goals sampled per frame.
        clearance: teacher's target clearance (match the value used at label time).
        max_shift: teacher's per-waypoint shift cap (likewise).
        d_safe: clearance threshold defining "too tight".
        window: frames fused on each side for the judge map.
        device: torch device.
        exclude_bottom_frac: drop the bottom image rows when sampling goals,
            matching the training goal region.
        seed: base RNG seed for pixel sampling.
        occ_prob: fused probability above which a judge cell counts as occupied.
            Vary it to confirm the conclusion is not an artefact of the map's
            strictness.

    Returns:
        The summary dict that is also written to ``summary.json``.
    """
    rec_dir = dataset / rec
    intr = pipeline.load_intrinsics(rec_dir)
    K = _intrinsics_matrix(intr)
    posed_idx, poses = load_frame_poses(bag_root, rec)

    available = set(pipeline.list_frames(rec_dir))
    usable = [i for i, f in enumerate(posed_idx) if int(f) in available]
    if not usable:
        raise ValueError(f"{rec}: no frame has both depth on disk and a pose")
    picks = usable[:: max(1, len(usable) // n_frames)][:n_frames]

    base, tuned = _load_models(base_ckpt, repo, trained, device)
    cfg = make_config(corrector="esdf", target_clearance_m=clearance,
                      max_total_shift_m=max_shift)
    judge_cfg = JudgeMapConfig(window=window, occ_prob=occ_prob)

    rows: List[dict] = []
    for pos in picks:
        frame = int(posed_idx[pos])
        bgr, depth = pipeline.load_frame(rec_dir, frame)
        field = build_judge_field(rec_dir, K, posed_idx, poses, pos, judge_cfg)
        pose = poses[pos]

        for (u, v) in pipeline.sample_valid_pixels(
                depth, n_goals, seed=seed + frame,
                exclude_bottom_frac=exclude_bottom_frac):
            try:
                goal = pixel_to_goal(u, v, depth, intr)[:2]
            except ValueError:
                continue
            traj = {"baseline": base.predict(bgr, depth, goal),
                    "trained": tuned.predict(bgr, depth, goal)}
            corrected = correct_navdp_trajectory(traj["baseline"], depth, intr, goal, cfg)
            traj["teacher"] = bev_render.path_xy(corrected.corrected_path)[1:]

            goal_w = body_to_world(np.asarray(goal, float)[None], pose)[0]
            for arm in ARMS:
                m = score(body_to_world(traj[arm], pose), goal_w, field, d_safe_m=d_safe)
                rows.append({"frame": frame, "u": int(u), "v": int(v), "arm": arm,
                             "n_fused": field.n_frames, **m.as_dict()})
        print(f"  frame {frame}: {len(rows)} rows", flush=True)

    return _summarize(rows, rec, trained, d_safe, clearance, max_shift, window,
                      occ_prob, out_dir)


def _summarize(rows: List[dict], rec: str, trained: Path, d_safe: float,
               clearance: float, max_shift: float, window: int, occ_prob: float,
               out_dir: Path) -> dict:
    """Write the CSV, compute paired stats, print the tables, write JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_sample.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    n = len(by_arm["baseline"])
    print(f"\n{rec}: {n} paired samples, judged on a {window * 2 + 1}-frame fused map")

    summary = {"recording": rec, "n_samples": n, "checkpoint": str(trained),
               "d_safe_m": d_safe, "teacher_clearance_m": clearance,
               "teacher_max_shift_m": max_shift, "judge_window": window,
               "judge_occ_prob": occ_prob,
               "collision_rate": {a: stats.collision_rate(by_arm[a]) for a in ARMS},
               "paired": {}}

    for arm in ("trained", "teacher"):
        res = stats.compare_all(by_arm["baseline"], by_arm[arm], REPORT_METRICS)
        print("\n" + stats.format_table(res, "baseline", arm))
        summary["paired"][f"baseline_vs_{arm}"] = {
            m: asdict(r) | {"significant": r.significant, "verdict": r.verdict}
            for m, r in res.items()}

    print("\ncollision rate (any waypoint inside an obstacle):")
    for arm in ARMS:
        print(f"  {arm:<10}{summary['collision_rate'][arm]:.1%}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {csv_path} and {out_dir / 'summary.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path.home() / "Downloads/flight_dataset")
    ap.add_argument("--bag-root", type=Path, default=Path.home() / "Videos",
                    help="directory of source rosbags, for AprilTag poses")
    ap.add_argument("--rec", default="walk_into", help="a HELD-OUT recording")
    ap.add_argument("--trained", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=Path.home() / "GIT/NavDP/baselines/navdp/checkpoints/navdp-cross-modal.ckpt")
    ap.add_argument("--navdp-repo", type=Path, default=Path.home() / "GIT/NavDP/baselines/navdp")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--n-goals", type=int, default=10)
    ap.add_argument("--clearance", type=float, default=0.3,
                    help="teacher target clearance; match the label-time value")
    ap.add_argument("--max-shift", type=float, default=0.2,
                    help="teacher shift cap; match the label-time value")
    ap.add_argument("--d-safe", type=float, default=0.30)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--occ-prob", type=float, default=0.65,
                    help="judge-map occupancy threshold; vary for a robustness check")
    ap.add_argument("--exclude-bottom-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run(args.dataset.expanduser(), args.bag_root.expanduser(), args.rec,
        args.ckpt.expanduser(), args.navdp_repo.expanduser(),
        args.trained.expanduser(), args.out_dir.expanduser(), args.n_frames,
        args.n_goals, args.clearance, args.max_shift, args.d_safe, args.window,
        args.device, args.exclude_bottom_frac, args.seed, args.occ_prob)


if __name__ == "__main__":
    main()
