from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Optional
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os

from sparx_agency.core.mapping.interfaces.depth_model import DepthModel


@dataclass
class DepthAnythingV2Config:
    # Indoor metric model (best fit for rooms)
    model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    # If you want relative depth instead, examples: "depth-anything/Depth-Anything-V2-Small-hf"
    device: Optional[str] = "cuda"  # "cuda", "cpu", or None=auto
    assume_bgr: bool = False      # set True if your images come as BGR (OpenCV)
    max_range_m: float = 15.0
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
        
        # Create directory for depth frame dumps
        self.depth_dump_dir = "/tmp/depth_dumps"
        os.makedirs(self.depth_dump_dir, exist_ok=True)
        self.dump_every_n_frames = 10  # Save every 10 frames

    def infer_depth(self, rgb: np.ndarray) -> np.ndarray:
        # The model is the metric variant (Depth-Anything-V2-Metric-*).
        # The HuggingFace pipeline returns TWO keys:
        #   out["depth"]            → PIL Image (uint8 visualization, 0-255, NOT metres)
        #   out["predicted_depth"]  → torch.Tensor with actual metric depth in metres
        # We must use "predicted_depth"; "depth" is only for display.
        out = self.pipe(Image.fromarray(rgb))
        depth_m = np.array(out["predicted_depth"]).astype(np.float32)
        
        # DEBUG: Print depth statistics
        valid_depth = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
        if len(valid_depth) > 0:
            print(f"[DepthAnythingV2] Raw depth stats - min: {np.min(valid_depth):.4f}m, max: {np.max(valid_depth):.4f}m, mean: {np.mean(valid_depth):.4f}m, std: {np.std(valid_depth):.4f}m")
        
        # Save depth frame for analysis
        self.frame_count += 1
        if self.frame_count % self.dump_every_n_frames == 0:
            self._save_depth_frame(depth_m, rgb)
        
        if self.cfg.debug:
            self.visualize_depth_raw_data(depth_m)
        depth_m = np.clip(depth_m, 0.0, self.cfg.max_range_m)
        return depth_m

    def _save_depth_frame(self, depth_m: np.ndarray, rgb: np.ndarray):
        """Save depth map and RGB image for later analysis"""
        frame_id = self.frame_count
        
        # Save depth as NPZ (includes metadata)
        depth_file = os.path.join(self.depth_dump_dir, f"depth_frame_{frame_id:06d}.npz")
        np.savez_compressed(depth_file, depth=depth_m, rgb=rgb)
        
        # Also save depth as visualization
        depth_vis = (np.clip(depth_m, 0, 15) / 15.0 * 255).astype(np.uint8)
        from PIL import Image as PILImage
        PILImage.fromarray(depth_vis, mode='L').save(
            os.path.join(self.depth_dump_dir, f"depth_vis_frame_{frame_id:06d}.png")
        )
        
        # Save RGB for reference
        PILImage.fromarray(rgb).save(
            os.path.join(self.depth_dump_dir, f"rgb_frame_{frame_id:06d}.png")
        )
        
        # Save statistics to CSV
        valid_depth = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
        stats_file = os.path.join(self.depth_dump_dir, "depth_statistics.csv")
        with open(stats_file, "a") as f:
            if os.path.getsize(stats_file) == 0:
                f.write("frame_id,min_depth,max_depth,mean_depth,std_depth,valid_pixels\n")
            f.write(f"{frame_id},{np.min(valid_depth):.6f},{np.max(valid_depth):.6f},{np.mean(valid_depth):.6f},{np.std(valid_depth):.6f},{len(valid_depth)}\n")
        
        print(f"[DepthAnythingV2] Saved frame {frame_id} to {self.depth_dump_dir}")


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
