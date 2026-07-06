"""Offline pixel-goal label generation (pose-free) for NavDP fine-tuning.

For each frame of a recording we sample ``n_per_frame`` pixel goals, run NavDP,
push its own trajectory off the single-frame walls (correct + smooth), and encode
the corrected path as the NavDP action label. The signed ESDF is stored once per
frame (it is goal-independent). No poses are used.

Output: one ``labels.npz`` with, per SAMPLE, ``frame`` / ``goal (fwd,left)`` /
``label (24,3)``; and per unique FRAME, its ``sdf (H,W)`` + grid geometry. The
torch dataset (:mod:`.pixel_dataset`) reads it alongside the recording frames.

    python -m sparx_agency.tasks.planning.finetune.train.pixel_labels \
        --dataset ~/flight_dataset --rec walk_into --n-per-frame 25 --out <rec>/labels.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..common.label_format import to_navdp_label
from ..verify import pipeline
from ..verify.correction import make_config
from ..verify.navdp_infer import NavDPInfer, default_engine_paths


def _goal_clearance(target, goal) -> float:
    """Signed ESDF (m) at the goal point -- <0 means the goal is inside a wall."""
    occ = target.occupancy
    gi = int((goal[0] - occ.origin_x) / occ.resolution)
    gj = int((goal[1] - occ.origin_y) / occ.resolution)
    h, w = target.sdf_m.shape
    return float(target.sdf_m[gj, gi]) if 0 <= gj < h and 0 <= gi < w else 1e3


def generate(dataset: Path, rec: str, out: Path, n_per_frame: int, frame_stride: int,
             corrector: str, clearance: float, max_shift: float, smooth: float,
             pitch: float, height: float, seed: int, min_goal_clear: float = 0.0,
             exclude_bottom_frac: float = 0.0) -> int:
    rec_dir = dataset / rec
    intr = pipeline.load_intrinsics(rec_dir)
    frames = pipeline.list_frames(rec_dir)[::frame_stride]
    engine_dir, head = default_engine_paths()
    infer = NavDPInfer(engine_dir, head)
    cfg = make_config(corrector=corrector, target_clearance_m=clearance,
                      max_total_shift_m=max_shift, pitch_deg=pitch,
                      camera_height_m=height, smooth_strength=smooth)

    s_frame, s_goal, s_label = [], [], []
    u_frames, u_sdf = [], []
    res = ox = oy = None
    skipped = 0
    # oversample pixels so the wall-goal filter still leaves ~n_per_frame goals
    take = n_per_frame * (3 if min_goal_clear > 0 else 1)
    for fi, frame in enumerate(frames):
        bgr, depth = pipeline.load_frame(rec_dir, frame)
        pix = pipeline.sample_valid_pixels(depth, take, seed=seed + frame,
                                           exclude_bottom_frac=exclude_bottom_frac)
        saved_sdf = False
        kept = 0
        for u, v in pix:
            if kept >= n_per_frame:
                break
            try:
                r = pipeline.run_pixel(bgr, depth, intr, u, v, infer, cfg, seed=0)
            except ValueError:
                continue
            t = r["target"]
            if min_goal_clear > 0 and _goal_clearance(t, r["goal"]) < min_goal_clear:
                skipped += 1                       # goal sits on / too near a wall
                continue
            kept += 1
            s_frame.append(frame)
            s_goal.append(r["goal"])
            s_label.append(to_navdp_label(t.corrected_path, horizon=24))
            if not saved_sdf:
                u_frames.append(frame)
                u_sdf.append(t.sdf_m.astype(np.float32))
                res = t.occupancy.resolution
                ox, oy = t.occupancy.origin_x, t.occupancy.origin_y
                saved_sdf = True
        if (fi + 1) % 10 == 0:
            print("  %d/%d frames, %d samples" % (fi + 1, len(frames), len(s_frame)))

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_frame=np.asarray(s_frame, np.int32),
        goal=np.asarray(s_goal, np.float32),
        label=np.asarray(s_label, np.float32),
        sdf_frame=np.asarray(u_frames, np.int32),
        sdf=np.asarray(u_sdf, np.float32),
        resolution=np.float32(res), origin_x=np.float32(ox), origin_y=np.float32(oy),
        recording=str(rec_dir),
    )
    print("wrote %d samples over %d frames (%d wall-goals skipped) -> %s"
          % (len(s_frame), len(u_frames), skipped, out))
    return len(s_frame)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path.home() / "flight_dataset")
    ap.add_argument("--recs", nargs="+", required=True,
                    help="one or more recording names; a labels file is written per recording")
    ap.add_argument("--out-name", default="labels.npz",
                    help="labels filename written inside each <rec>/ directory")
    ap.add_argument("--n-per-frame", type=int, default=25)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--corrector", default="esdf", choices=["esdf", "potential_field"])
    ap.add_argument("--clearance", type=float, default=0.5)
    ap.add_argument("--max-shift", type=float, default=0.8)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--min-goal-clear", type=float, default=0.0,
                    help="skip goals whose ESDF clearance is below this (drops "
                         "goals that land on a wall); 0 keeps all")
    ap.add_argument("--exclude-bottom-frac", type=float, default=0.0,
                    help="drop the bottom fraction of image rows (e.g. 0.2) so goals "
                         "avoid the ground right below the drone")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dataset = args.dataset.expanduser()
    total = 0
    for rec in args.recs:
        print(f"== {rec} ==")
        out = dataset / rec / args.out_name
        total += generate(dataset, rec, out, args.n_per_frame, args.frame_stride,
                          args.corrector, args.clearance, args.max_shift, args.smooth,
                          args.pitch, args.height, args.seed, args.min_goal_clear,
                          args.exclude_bottom_frac)
    print(f"TOTAL {total} samples across {len(args.recs)} recordings")


if __name__ == "__main__":
    main()
