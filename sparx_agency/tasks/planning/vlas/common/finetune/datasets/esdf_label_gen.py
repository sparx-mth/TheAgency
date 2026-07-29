"""Offline label generation: flight recording -> per-frame PF/ESDF labels on disk.

For every frame it builds the single-frame potential-field / ESDF-corrected target
trajectory (:mod:`..common.esdf_target`), encodes it into each model's action
format (:mod:`..common.label_format`), and saves that plus the signed ESDF grid the
differentiable penalty samples at training time. All numpy -- runs in the plain
``.venv`` (install ``numba`` for throughput on the mapping kernels).

The label path is intentionally the *non-differentiable* half of the loss (a fixed
BC target); the SDF grid it also writes is what the *differentiable* penalty reads.

Output per recording::

    labels/
      000000.npz   {navdp (24,3), flownav (8,2), sdf (H,W), resolution,
                    origin_x, origin_y, goal_fwd, goal_left, num_moved}
      ...
      index.json   {frames, config...}
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..common.esdf_target import EsdfTargetConfig, generate_target
from ..common.frames import LocalMapConfig
from ..common.label_format import to_flownav_label, to_navdp_label
from .recording import FlightRecording, load_recording


def generate_labels(
    recording: FlightRecording,
    out_dir,
    *,
    seed_from_flight: bool = True,
    goal_lookahead: int = 24,
    navdp_horizon: int = 24,
    flownav_horizon: int = 8,
    metric_waypoint_spacing: float = 0.25,
    target_config: Optional[EsdfTargetConfig] = None,
) -> int:
    """Write one ``.npz`` label per frame. Returns the number written.

    Args:
        recording: A loaded :class:`FlightRecording`.
        out_dir: Output ``labels/`` directory.
        seed_from_flight: Seed the corrector with the drone's flown-future path
            (else a straight origin->goal line).
        goal_lookahead: Frames ahead used as the auto point-goal.
        navdp_horizon / flownav_horizon: Label horizons.
        metric_waypoint_spacing: FlowNav waypoint-unit scale.
        target_config: Override the ESDF-target config (else built from the
            recording's camera height / pitch).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = target_config or EsdfTargetConfig(
        local_map=LocalMapConfig(
            camera_height_m=recording.camera_height_m,
            pitch_deg=recording.pitch_deg,
        )
    )

    n = 0
    for i in range(recording.num_frames):
        depth = recording.depth(i)
        goal = recording.goal_body(i, goal_lookahead)
        seed = recording.future_path_body(i, navdp_horizon) if seed_from_flight else None
        target = generate_target(depth, recording.intrinsics, goal, cfg, seed_path=seed)

        navdp = to_navdp_label(target.corrected_path, horizon=navdp_horizon)
        flownav = to_flownav_label(target.corrected_path, horizon=flownav_horizon,
                                   metric_waypoint_spacing=metric_waypoint_spacing)
        np.savez_compressed(
            out_dir / f"{i:06d}.npz",
            navdp=navdp.astype(np.float32),
            flownav=flownav.astype(np.float32),
            sdf=target.sdf_m.astype(np.float32),
            resolution=np.float32(target.occupancy.resolution),
            origin_x=np.float32(target.occupancy.origin_x),
            origin_y=np.float32(target.occupancy.origin_y),
            goal_fwd=np.float32(goal[0]),
            goal_left=np.float32(goal[1]),
            num_moved=np.int32(target.num_moved),
        )
        n += 1

    (out_dir / "index.json").write_text(json.dumps({
        "frames": n,
        "navdp_horizon": navdp_horizon,
        "flownav_horizon": flownav_horizon,
        "metric_waypoint_spacing": metric_waypoint_spacing,
        "local_map": asdict(cfg.local_map),
        "corrector": cfg.corrector,
        "target_clearance_m": cfg.target_clearance_m,
    }, indent=2))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PF/ESDF fine-tune labels.")
    ap.add_argument("--recording", required=True, help="recording directory")
    ap.add_argument("--out-dir", required=True, help="output labels/ directory")
    ap.add_argument("--corrector", default="potential_field", choices=["potential_field", "esdf"])
    ap.add_argument("--straight-seed", action="store_true", help="use a straight seed, not the flown future")
    ap.add_argument("--metric-waypoint-spacing", type=float, default=0.25)
    args = ap.parse_args()

    rec = load_recording(args.recording)
    cfg = EsdfTargetConfig(
        local_map=LocalMapConfig(camera_height_m=rec.camera_height_m, pitch_deg=rec.pitch_deg),
        corrector=args.corrector,
    )
    n = generate_labels(rec, args.out_dir, seed_from_flight=not args.straight_seed,
                        metric_waypoint_spacing=args.metric_waypoint_spacing,
                        target_config=cfg)
    print(f"wrote {n} labels to {args.out_dir}")


if __name__ == "__main__":
    main()
