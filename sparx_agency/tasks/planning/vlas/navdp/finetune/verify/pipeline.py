"""One clicked pixel -> goal -> NavDP -> PF/ESDF-corrected target (shared core).

The single per-pixel step both the interactive app and the batch preview run, plus
the small frame-loading and pixel-sampling helpers around it. Keeping it here means
the UI files hold only presentation/interaction, not the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.vlas.common.finetune.common.esdf_target import EsdfTargetConfig

from .correction import correct_navdp_trajectory
from .navdp_infer import NavDPInfer
from .pixel_goal import pixel_to_goal


def load_intrinsics(rec_dir: Path) -> Intrinsics:
    """Read ``intrinsics.json`` from an extracted recording directory."""
    d = json.loads((Path(rec_dir) / "intrinsics.json").read_text())
    return Intrinsics(width=d["width"], height=d["height"],
                      fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"])


def list_frames(rec_dir: Path) -> List[int]:
    """Sorted frame indices present as both an rgb PNG and a depth npy."""
    rec_dir = Path(rec_dir)
    rgb = {int(p.stem) for p in (rec_dir / "rgb").glob("*.png")}
    dep = {int(p.stem) for p in (rec_dir / "depth").glob("*.npy")}
    return sorted(rgb & dep)


def load_frame(rec_dir: Path, frame: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``(rgb_bgr uint8, depth_m float32)`` for one frame index."""
    rec_dir = Path(rec_dir)
    bgr = cv2.imread(str(rec_dir / "rgb" / f"{frame:06d}.png"))
    if bgr is None:
        raise FileNotFoundError(rec_dir / "rgb" / f"{frame:06d}.png")
    depth = np.load(rec_dir / "depth" / f"{frame:06d}.npy").astype(np.float32)
    return bgr, depth


def sample_valid_pixels(depth_m: np.ndarray, n: int, seed: int = 0,
                        margin: int = 20, min_depth_m: float = 0.3,
                        exclude_bottom_frac: float = 0.0) -> List[Tuple[int, int]]:
    """Pick ``n`` random pixels with valid depth.

    Args:
        exclude_bottom_frac: drop the bottom fraction of image rows before
            sampling. For a drone, the lowest rows look at the ground right below
            / just ahead, which is a "land / creep forward" non-goal -- excluding
            them keeps goals to meaningful forward navigation. ``0.2`` drops the
            bottom fifth.
    """
    h, w = depth_m.shape
    v_max = int(round(h * (1.0 - exclude_bottom_frac)))
    ys, xs = np.where(np.isfinite(depth_m) & (depth_m > min_depth_m))
    keep = ((xs >= margin) & (xs < w - margin)
            & (ys >= margin) & (ys < min(v_max, h - margin)))
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(xs.size, size=min(n, xs.size), replace=False)
    return list(zip(xs[idx].tolist(), ys[idx].tolist()))


def run_pixel(rgb_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: Intrinsics,
              u: int, v: int, infer: NavDPInfer, config: EsdfTargetConfig,
              seed: int = 0) -> Dict:
    """Full per-pixel pipeline: goal -> NavDP -> corrected target.

    Returns a dict ``{goal, depth, navdp, target}`` where ``goal`` is
    ``(forward, left)`` meters, ``navdp`` is a :class:`NavDPResult`, and ``target``
    is a :class:`PerFrameTarget` (occupancy + ESDF + seed + corrected + num_moved).
    """
    fwd, left, d = pixel_to_goal(u, v, depth_m, intrinsics)
    navdp = infer.predict(rgb_bgr, depth_m, (fwd, left), seed=seed)
    target = correct_navdp_trajectory(navdp.trajectory, depth_m, intrinsics, (fwd, left), config)
    return {"goal": (fwd, left), "depth": d, "navdp": navdp, "target": target}
