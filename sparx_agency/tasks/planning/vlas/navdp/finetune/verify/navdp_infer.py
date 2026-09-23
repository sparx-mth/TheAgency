"""NavDP point-goal inference for a single RGB-D frame (TensorRT, torch-free).

Wraps :class:`NavDPTRTPolicy` (the numpy + TensorRT re-implementation shipped in
``core/planning/vlas/navdp/trt``) with the exact host-side preprocessing the external
``NavDP_Agent`` uses -- ``process_image`` / ``process_depth`` / ``process_pointgoal``
-- taken from :mod:`sparx_agency.core.planning.vlas.navdp.preprocess`, so the tool
needs **no external NavDP repo, no torch model, and no transformers**, only the
built engines. It runs entirely in the ``navdp`` conda env off
``navdp-cross-modal``-derived engines.

A verification tool that preprocesses differently from the server it verifies is
worse than no tool, so the colour order is the deployed one by default; see
``color_order`` on :class:`NavDPInfer`.

The 8-frame RGB memory is filled deterministically by tiling the current frame
(the drone "has been" looking at this scene), and the diffusion noise is seeded,
so a given (frame, goal) always yields the same trajectory -- essential for
reproducible, click-order-independent verification and label generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.planning.vlas.navdp import preprocess as _core
from sparx_agency.core.planning.vlas.navdp.preprocess import (  # noqa: F401
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    ENCODER_COLOR_ORDER,
    IMAGE_SIZE,
    resize_pad,   # re-exported: this module was where callers found it
)
from sparx_agency.core.planning.vlas.navdp.trt.policy import NavDPTRTPolicy


def preprocess_rgb(rgb_bgr: np.ndarray, size: int = IMAGE_SIZE,
                   color_order: str = ENCODER_COLOR_ORDER) -> np.ndarray:
    """BGR uint8 -> ``(size, size, 3)`` float32 in [0, 1], in the encoder's order.

    ``color_order`` defaults to what the deployed TensorRT server feeds the
    encoder (BGR), so a trajectory produced here is the one the drone would fly.
    This function used to convert to RGB unconditionally, which is one channel
    swap away from deployment; pass ``"rgb"`` to reproduce that older behaviour.
    """
    return _core.preprocess_rgb(rgb_bgr, input_order="bgr", encoder_order=color_order,
                                size=size, layout="hwc")


def preprocess_depth(depth_m: np.ndarray, size: int = IMAGE_SIZE,
                     dmin: float = DEPTH_MIN_M, dmax: float = DEPTH_MAX_M) -> np.ndarray:
    """Metric depth -> ``(size, size, 1)`` float32, out-of-range zeroed (NavDP-style)."""
    return _core.preprocess_depth(depth_m, size=size, depth_min_m=dmin,
                                  depth_max_m=dmax, layout="hwc")


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
        color_order: channel order handed to the encoder. Defaults to the
            deployed server's (BGR); ``"rgb"`` reproduces this tool's pre-2026-08
            behaviour, which was one channel swap away from what flies.
    """

    def __init__(self, engine_dir, head_params_npz, sample_num: int = 16,
                 predict_size: int = 24, image_size: int = IMAGE_SIZE, memory_size: int = 8,
                 depth_min_m: float = DEPTH_MIN_M, depth_max_m: float = DEPTH_MAX_M,
                 device_id: int = 0, color_order: str = ENCODER_COLOR_ORDER):
        self.sample_num = int(sample_num)
        self.image_size = int(image_size)
        self.memory_size = int(memory_size)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.color_order = str(color_order)
        self._policy = NavDPTRTPolicy(engine_dir, head_params_npz,
                                      sample_num=sample_num, predict_size=predict_size,
                                      device_id=device_id)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, rgb_bgr: np.ndarray, depth_m: np.ndarray,
                goal_body: Tuple[float, float], seed: int = 0) -> NavDPResult:
        """Run point-goal inference on one frame.

        Args:
            rgb_bgr: ``(H, W, 3)`` uint8 BGR (as ``cv2.imread`` returns); handed
                to the encoder in ``self.color_order``.
            depth_m: ``(H, W)`` float32 depth in meters.
            goal_body: ``(forward, left)`` body-FLU goal in meters.
            seed: RNG seed for the diffusion noise (reproducibility).

        Returns:
            A :class:`NavDPResult`; ``trajectory`` is the executed ``(24, 2)``
            ``[fwd, left]`` path (meters, body FLU).
        """
        frame = preprocess_rgb(rgb_bgr, self.image_size,
                               self.color_order)                  # (S,S,3) [0,1]
        images = np.tile(frame[None, None], (1, self.memory_size, 1, 1, 1))  # (1,8,S,S,3)
        depth = preprocess_depth(depth_m, self.image_size,
                                 self.depth_min_m, self.depth_max_m)[None]   # (1,S,S,1)

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
    # Anchor on the installed package rather than counting `parents[N]`: this file
    # has already moved once (finetune/verify -> vlas/navdp/finetune/verify) and a
    # depth count fails silently, resolving to a plausible-but-wrong directory.
    if repo_root:
        pkg = Path(repo_root) / "sparx_agency"
    else:
        import sparx_agency
        pkg = Path(sparx_agency.__file__).resolve().parent
    eng_root = pkg / "tasks/planning/vlas/navdp/trt/engines"
    dirs = sorted(d for d in eng_root.glob("*") if (d / "selected.json").exists())
    if not dirs:
        raise FileNotFoundError(f"no built NavDP engine dir under {eng_root}")
    engine_dir = dirs[0]
    return engine_dir, engine_dir / "navdp_head_params.npz"
