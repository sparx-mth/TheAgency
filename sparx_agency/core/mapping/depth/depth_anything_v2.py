from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Optional

import torch
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
    max_range_m: float = 35.0
    min_range_m: float = 0.3
    debug: bool = False


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

    def infer_raw(self, rgb: np.ndarray) -> np.ndarray:
        """
        Returns raw model output as float32 HxW (no min/max mapping).
        Prefer predicted_depth over depth visualization.
        """
        out = self.pipe(Image.fromarray(rgb))

        if "predicted_depth" in out:
            pred = out["predicted_depth"]
            if torch.is_tensor(pred):
                pred = pred.detach().float().cpu().numpy()
            pred = np.asarray(pred).squeeze().astype(np.float32)
            if self.cfg.debug:
                self.visualize_depth_raw_data(pred)
            return pred

        # Fallback (less ideal): out["depth"] is often a PIL visualization
        raw = np.array(out["depth"]).astype(np.float32)
        if self.cfg.debug:
            self.visualize_depth_raw_data(raw)
        return raw

    def raw_stats(self, raw: np.ndarray) -> dict:
        raw = raw[np.isfinite(raw)]
        if raw.size == 0:
            return {"min": 0.0, "max": 1.0}
        # robust range is better than strict min/max
        lo = float(np.percentile(raw, 1))
        hi = float(np.percentile(raw, 99))
        return {"min": lo, "max": hi}

    def infer_depth(self, rgb: np.ndarray, norm_stats: dict | None = None) -> np.ndarray:
        raw = self.infer_raw(rgb)

        # If this is a metric model and predicted_depth is metric-ish:
        # -> you can just return raw (optionally clamp)
        if "Metric" in self.cfg.model_id or "metric" in self.cfg.model_id:
            depth_m = raw
            depth_m = np.clip(depth_m, self.cfg.min_range_m, self.cfg.max_range_m)
            return depth_m.astype(np.float32)

        # Relative model: normalize using provided global stats if available
        if norm_stats is None:
            norm_stats = self.raw_stats(raw)
        lo = norm_stats["min"]
        hi = norm_stats["max"]
        den = max(hi - lo, 1e-6)

        depth_norm = np.clip((raw - lo) / den, 0.0, 1.0)
        depth_m = depth_norm * (self.cfg.max_range_m - self.cfg.min_range_m) + self.cfg.min_range_m
        return depth_m.astype(np.float32)


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
