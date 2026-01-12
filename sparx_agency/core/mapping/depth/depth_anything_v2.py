from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Optional
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from sparx_agency.core.mapping.interfaces.depth_model import DepthModel


@dataclass
class DepthAnythingV2Config:
    # Indoor metric model (best fit for rooms)
    model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    # If you want relative depth instead, examples: "depth-anything/Depth-Anything-V2-Small-hf"
    device: Optional[str] = "cuda"  # "cuda", "cpu", or None=auto
    assume_bgr: bool = False      # set True if your images come as BGR (OpenCV)
    max_range_m: float = 15.0


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
        self.frame_count = 0
        self.save_interval = 100

    def infer_depth(self, rgb: np.ndarray) -> np.ndarray:
        # 1. Get raw disparity from model (0-255 or 0-N)
        out = self.pipe(Image.fromarray(rgb))
        raw_disparity = np.array(out["depth"]).astype(np.float32)
        self.visualize_depth_raw_data(raw_disparity)
        # 2. Normalize to 0.0 - 1.0
        d_min = raw_disparity.min()
        d_max = raw_disparity.max()
        den = max(d_max - d_min, 1e-6)

        # depth = (raw_disparity - d_min) / (d_max - d_min) * 255.0
        # depth_m = depth.astype(np.uint8)

        depth_norm = (raw_disparity - d_min) / den

        # 3. To see a person clearly, the points must have a wide Z-range.
        # Person should be at ~2.0m, Background at ~15.0m.
        max_range = 15.0
        min_range = 0.5  # Objects start 0.5m in front of camera

        depth_m = raw_disparity/50
        return depth_m


    def visualize_depth_raw_data(self, raw_depth):
        self.frame_count += 1
        if self.frame_count % self.save_interval != 0:
            return

        # Create a single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(f"Raw Depth Distribution - Frame {self.frame_count}")

        # 1. Image Depth (The 'D' values)
        # This helps see if your depth model is saturated before clipping/scaling
        ax.hist(raw_depth.ravel(), bins=100, color='gray')
        ax.set_title("Raw Model Output (Pre-Clipping)")
        ax.set_xlabel("Intensity / Raw Value")
        ax.set_ylabel("Frequency")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save as PNG
        time_str = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
        filename = f"depth_diagnostic_{self.frame_count:04d}_{time_str}.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Diagnostic saved to {filename}")
