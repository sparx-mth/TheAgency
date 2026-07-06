"""Torch NavDP point-goal inference with hot-swappable weights (no matplotlib).

Shared by the batch evaluator and the interactive comparison tool. Kept free of
any matplotlib import so an interactive UI that uses it is not forced onto a
non-interactive backend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..verify.navdp_infer import preprocess_depth, preprocess_rgb
from sparx_agency.tasks.planning.navdp.export.build_policy import build_navdp_policy


class TorchNavDP:
    """Torch NavDP point-goal inference; load base or fine-tuned weights."""

    def __init__(self, ckpt: str, navdp_repo: str, device: str = "cuda", memory: int = 8):
        self.policy = build_navdp_policy(ckpt, navdp_repo, device=device)
        self.device, self.memory = device, memory

    def load_weights(self, pth) -> None:
        """Load a fine-tuned state_dict (e.g. an EMA checkpoint) into the policy."""
        state = torch.load(Path(pth).expanduser(), map_location=self.device)
        self.policy.load_state_dict(state, strict=False)

    def predict(self, rgb_bgr: np.ndarray, depth_m: np.ndarray, goal_body) -> np.ndarray:
        """Run point-goal inference on one frame -> ``(24, 2)`` [fwd, left] waypoints."""
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
