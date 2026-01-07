from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        """
        Input: HxWx3 uint8/float, RGB (or BGR if assume_bgr=True)
        Output: HxW float32 depth (metric for *Metric* models; otherwise relative depth)
        """
        from PIL import Image

        if rgb is None or rgb.size == 0:
            raise ValueError("infer_depth got empty image")

        img = rgb
        if img.dtype != np.uint8:
            # common cases: float32 0..1 or 0..255
            img = np.clip(img, 0.0, 1.0) if img.max() <= 1.5 else np.clip(img / 255.0, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)

        if self.cfg.assume_bgr:
            img = img[..., ::-1]  # BGR->RGB

        pil = Image.fromarray(img, mode="RGB")
        out = self.pipe(pil)

        # Transformers depth-estimation pipeline returns a PIL image under key "depth"
        depth_pil = out["depth"]
        depth = np.array(depth_pil).astype(np.float32)

        # Ensure HxW
        if depth.ndim == 3:
            depth = depth[..., 0]

        return depth
