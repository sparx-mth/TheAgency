"""Torch dataset over precomputed pixel-goal labels (:mod:`.pixel_labels`).

Each item is one (frame, sampled-goal) pair: the frame's RGB tiled into the
8-frame memory + its depth (both preprocessed exactly as NavDP inference does, so
the training input matches what the label was generated from), the goal, the
corrected+smoothed NavDP action label, and the frame's signed ESDF grid. No poses.

Batches carry the keys the shared NavDP training loop expects
(``images/depth/goal/label/sdf_grid/resolution/origin_{x,y}``). Torch + cv2 --
runs in the ``navdp`` conda env.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..verify.navdp_infer import preprocess_depth, preprocess_rgb


class PixelGoalDataset(Dataset):
    """Per-(frame, pixel-goal) NavDP fine-tune samples from a ``labels.npz``."""

    def __init__(self, labels_npz, image_size: int = 224, memory_size: int = 8,
                 depth_min_m: float = 0.1, depth_max_m: float = 5.0) -> None:
        z = np.load(labels_npz, allow_pickle=False)
        self.frame = z["sample_frame"].astype(np.int64)
        self.goal = z["goal"].astype(np.float32)
        self.label = z["label"].astype(np.float32)
        self._sdf = {int(f): z["sdf"][i] for i, f in enumerate(z["sdf_frame"])}
        self.res = float(z["resolution"])
        self.ox, self.oy = float(z["origin_x"]), float(z["origin_y"])
        self.rec_dir = Path(str(z["recording"]))
        self.image_size, self.memory_size = image_size, memory_size
        self.depth_min_m, self.depth_max_m = depth_min_m, depth_max_m
        self._cache: Dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _frame_inputs(self, frame: int):
        """Processed ``(images (8,3,S,S), depth (1,S,S))`` for a frame (cached)."""
        if frame not in self._cache:
            bgr = cv2.imread(str(self.rec_dir / "rgb" / f"{frame:06d}.png"))
            depth = np.load(self.rec_dir / "depth" / f"{frame:06d}.npy").astype(np.float32)
            rgb01 = preprocess_rgb(bgr, self.image_size)                 # (S,S,3) RGB [0,1]
            img = np.transpose(rgb01, (2, 0, 1))                         # (3,S,S)
            images = np.repeat(img[None], self.memory_size, axis=0)      # (8,3,S,S)
            dep = preprocess_depth(depth, self.image_size,
                                   self.depth_min_m, self.depth_max_m)   # (S,S,1)
            dep = np.transpose(dep, (2, 0, 1))                           # (1,S,S)
            self._cache[frame] = (images.astype(np.float32), dep.astype(np.float32))
        return self._cache[frame]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame = int(self.frame[idx])
        images, depth = self._frame_inputs(frame)
        g = self.goal[idx]
        return {
            "images": torch.from_numpy(images),
            "depth": torch.from_numpy(depth),
            "goal": torch.tensor([g[0], g[1], 0.0], dtype=torch.float32),
            "label": torch.from_numpy(self.label[idx]),                 # (24,3)
            "sdf_grid": torch.from_numpy(self._sdf[frame])[None],       # (1,H,W)
            "resolution": torch.tensor(self.res),
            "origin_x": torch.tensor(self.ox),
            "origin_y": torch.tensor(self.oy),
        }
