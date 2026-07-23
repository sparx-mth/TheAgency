"""Non-interactive preview: sample several pixel goals on one frame -> a PNG grid.

Runs the full goal -> NavDP -> PF/ESDF-correction pipeline for N sampled pixels
and lays out one BEV panel per goal, so you can eyeball (headless, or to share) how
the corrected target diverges from NavDP across many directions on a single frame.
Also the quickest way to sanity-check the whole stack end-to-end.

    python -m sparx_agency.tasks.planning.vlas.navdp.finetune.verify.batch_preview \
        --dataset ~/flight_dataset --rec walk_into --frame 40 --n 6 --out preview.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import bev_render, pipeline  # noqa: E402
from .correction import make_config  # noqa: E402
from .navdp_infer import NavDPInfer, default_engine_paths  # noqa: E402


def render(dataset: Path, rec: str, frame: int, n: int, out: Path,
           corrector: str, clearance: float, max_shift: float,
           pitch_deg: float, height_m: float, field: str, seed: int) -> None:
    rec_dir = dataset / rec
    intr = pipeline.load_intrinsics(rec_dir)
    bgr, depth = pipeline.load_frame(rec_dir, frame)
    engine_dir, head = default_engine_paths()
    infer = NavDPInfer(engine_dir, head)
    cfg = make_config(corrector=corrector, target_clearance_m=clearance,
                      max_total_shift_m=max_shift, pitch_deg=pitch_deg,
                      camera_height_m=height_m)

    pixels = pipeline.sample_valid_pixels(depth, n, seed=seed)
    cols = 3
    rows = (len(pixels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (u, v) in enumerate(pixels):
        r = pipeline.run_pixel(bgr, depth, intr, u, v, infer, cfg, seed=seed)
        ax = axes[k // cols][k % cols]
        ax.axis("on")
        bev_render.draw_bev(ax, r["target"], r["goal"], field_mode=field, clearance=clearance)
        ax.set_title("px(%d,%d) goal(%.1f,%.1f) moved=%d"
                     % (u, v, r["goal"][0], r["goal"][1], int(r["target"].num_moved)),
                     fontsize=9)
        print("px(%d,%d) goal(fwd=%.2f left=%.2f) moved=%d critic[%.2f,%.2f]"
              % (u, v, r["goal"][0], r["goal"][1], int(r["target"].num_moved),
                 float(r["navdp"].critic.max()), float(r["navdp"].critic.min())))
    fig.suptitle("%s frame %d  [%s corrector, clearance=%.2f, pitch=%.0f, h=%.1f]"
                 % (rec, frame, corrector, clearance, pitch_deg, height_m), fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    print("saved", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path.home() / "flight_dataset")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corrector", default="esdf", choices=["potential_field", "esdf"])
    ap.add_argument("--clearance", type=float, default=0.5)
    ap.add_argument("--max-shift", type=float, default=0.8)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--field", default="esdf", choices=["esdf", "repulsion"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    render(args.dataset.expanduser(), args.rec, args.frame, args.n, args.out.expanduser(),
           args.corrector, args.clearance, args.max_shift, args.pitch, args.height,
           args.field, args.seed)


if __name__ == "__main__":
    main()
