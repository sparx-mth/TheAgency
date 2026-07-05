"""NavDP point-goal inference for a single RGB-D frame (TensorRT, torch-free).

Wraps :class:`NavDPTRTPolicy` (the numpy + TensorRT re-implementation shipped in
``core/planning/navdp/trt``) with the exact host-side preprocessing the external
``NavDP_Agent`` uses -- ``process_image`` / ``process_depth`` / ``process_pointgoal``
-- reimplemented here so the tool needs **no external NavDP repo, no torch model,
and no transformers**, only the built engines. It runs entirely in the ``navdp``
conda env off ``navdp-cross-modal``-derived engines.

The 8-frame RGB memory is filled deterministically by tiling the current frame
(the drone "has been" looking at this scene), and the diffusion noise is seeded,
so a given (frame, goal) always yields the same trajectory -- essential for
reproducible, click-order-independent verification and label generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from sparx_agency.core.planning.navdp.trt.policy import NavDPTRTPolicy


@dataclass
class NavDPResult:
    """One NavDP point-goal inference."""

    trajectory: np.ndarray   # (predict_size, 2) executed [fwd, left] waypoints, meters
    all_traj: np.ndarray     # (sample_num, predict_size, 3) all candidates
    critic: np.ndarray       # (sample_num,) critic value per candidate
    goal_clipped: np.ndarray  # (2,) [fwd, left] after NavDP's goal clipping


class NavDPInfer:
    """Deterministic single-frame NavDP point-goal inference over TensorRT engines.

    Args:
        engine_dir: directory with ``selected.json`` + the ``.engine`` files.
        head_params_npz: exported numpy head params (``navdp_head_params.npz``).
        sample_num: diffusion samples (must equal the built engine N; default 16).
        predict_size: trajectory horizon (24).
        image_size: model input side (224).
        memory_size: RGB context length (8).
        depth_min_m/depth_max_m: depth clip applied before the encoder (0.1 / 5.0).
        device_id: CUDA device index.
    """

    def __init__(self, engine_dir, head_params_npz, sample_num: int = 16,
                 predict_size: int = 24, image_size: int = 224, memory_size: int = 8,
                 depth_min_m: float = 0.1, depth_max_m: float = 5.0, device_id: int = 0):
        self.sample_num = int(sample_num)
        self.image_size = int(image_size)
        self.memory_size = int(memory_size)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self._policy = NavDPTRTPolicy(engine_dir, head_params_npz,
                                      sample_num=sample_num, predict_size=predict_size,
                                      device_id=device_id)

    # ------------------------------------------------------------------
    # Preprocessing (faithful to NavDP_Agent, kept pure / stateless)
    # ------------------------------------------------------------------
    def _resize_pad(self, arr: np.ndarray) -> np.ndarray:
        """Keep-aspect resize to fit ``image_size`` then center-pad to a square."""
        s = self.image_size
        prop = s / max(arr.shape[0], arr.shape[1])
        r = cv2.resize(arr, (-1, -1), fx=prop, fy=prop)
        pw = max((s - r.shape[1]) // 2, 0)
        ph = max((s - r.shape[0]) // 2, 0)
        pad = ((ph, ph), (pw, pw), (0, 0)) if r.ndim == 3 else ((ph, ph), (pw, pw))
        r = np.pad(r, pad, mode="constant", constant_values=0)
        return cv2.resize(r, (s, s))

    def _process_image(self, rgb: np.ndarray) -> np.ndarray:
        return self._resize_pad(rgb).astype(np.float32) / 255.0

    def _process_depth(self, depth_m: np.ndarray) -> np.ndarray:
        d = depth_m.astype(np.float32).copy()
        d[~np.isfinite(d)] = 0.0
        d = self._resize_pad(d)
        d[d > self.depth_max_m] = 0.0
        d[d < self.depth_min_m] = 0.0
        return d[:, :, None]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, rgb_bgr: np.ndarray, depth_m: np.ndarray,
                goal_body: Tuple[float, float], seed: int = 0) -> NavDPResult:
        """Run point-goal inference on one frame.

        Args:
            rgb_bgr: ``(H, W, 3)`` uint8 BGR (as ``cv2.imread`` returns); converted
                to RGB internally to match NavDP's training colour order.
            depth_m: ``(H, W)`` float32 depth in meters.
            goal_body: ``(forward, left)`` body-FLU goal in meters.
            seed: RNG seed for the diffusion noise (reproducibility).

        Returns:
            A :class:`NavDPResult`; ``trajectory`` is the executed ``(24, 2)``
            ``[fwd, left]`` path (meters, body FLU).
        """
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        frame = self._process_image(rgb)                                  # (S,S,3)
        images = np.tile(frame[None, None], (1, self.memory_size, 1, 1, 1))  # (1,8,S,S,3)
        depth = self._process_depth(depth_m)[None]                        # (1,S,S,1)

        goal = np.array([[goal_body[0], goal_body[1], 0.0]], np.float32).clip(-10, 10)
        goal[:, 0] = np.clip(goal[:, 0], 0.0, 10.0)

        np.random.seed(int(seed))
        all_traj, critic, positive, _neg = self._policy.predict_pointgoal_action(
            goal, images.astype(np.float32), depth.astype(np.float32),
            sample_num=self.sample_num)
        return NavDPResult(
            trajectory=np.asarray(positive[0, 0, :, :2], np.float32),
            all_traj=np.asarray(all_traj[0], np.float32),
            critic=np.asarray(critic[0], np.float32),
            goal_clipped=goal[0, :2].copy(),
        )


def default_engine_paths(repo_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """Resolve the fp16 engine dir + head-params for this host (single GPU build)."""
    # parents[4] is the `sparx_agency` package dir (…/verify -> …/sparx_agency).
    pkg = Path(repo_root) / "sparx_agency" if repo_root else Path(__file__).resolve().parents[4]
    eng_root = pkg / "tasks/planning/navdp/engines"
    dirs = sorted(d for d in eng_root.glob("*") if (d / "selected.json").exists())
    if not dirs:
        raise FileNotFoundError(f"no built NavDP engine dir under {eng_root}")
    engine_dir = dirs[0]
    return engine_dir, engine_dir / "navdp_head_params.npz"
