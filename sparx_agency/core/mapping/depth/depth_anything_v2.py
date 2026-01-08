from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from PIL import Image
import numpy as np

from sparx_agency.core.mapping.interfaces.depth_model import DepthModel


@dataclass
class DepthAnythingV2Config:
    # Indoor metric model (best fit for rooms)
    model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    # If you want relative depth instead, examples: "depth-anything/Depth-Anything-V2-Small-hf"
    device: Optional[str] = "cuda"  # "cuda", "cpu", or None=auto
    assume_bgr: bool = False      # set True if your images come as BGR (OpenCV)


class DepthAnythingV2DepthModel(DepthModel):
    """
    Depth Anything V2 via HuggingFace transformers.
    Downloads weights automatically to HF cache (no .pth path needed).
    """

    def __init__(self, cfg: Optional[DepthAnythingV2Config] = None):
        self.cfg = cfg or DepthAnythingV2Config()

        # Lazy imports so core stays lightweight unless used
        import torch
        from transformers import pipeline

        if self.cfg.device is None:
            device_idx = 0 if torch.cuda.is_available() else -1
        else:
            device_idx = 0 if self.cfg.device.startswith("cuda") else -1

        # pipeline(task="depth-estimation", model=...) is the simplest API
        self.pipe = pipeline(
            task="depth-estimation",
            model=self.cfg.model_id,
            device=device_idx,
        )

    def infer_depth(self, rgb: np.ndarray) -> np.ndarray:
        # 1. Get raw disparity from model (0-255 or 0-N)
        out = self.pipe(Image.fromarray(rgb))
        raw_disparity = np.array(out["depth"]).astype(np.float32)

        # 2. Normalize to 0.0 - 1.0
        d_min = raw_disparity.min()
        d_max = raw_disparity.max()
        den = max(d_max - d_min, 1e-6)
        depth_norm = (raw_disparity - d_min) / den

        # 3. THE FIX: The "Inverted Scale"
        # To see a person clearly, the points must have a wide Z-range.
        # Person should be at ~2.0m, Background at ~15.0m.
        max_range = 5.0
        min_range = -1.0

        # This formula flips disparity so high values (close) become small meters
        depth_m = min_range + (max_range - min_range) * (1.0 - depth_norm)

        return depth_m
