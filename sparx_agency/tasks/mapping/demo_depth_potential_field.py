#!/usr/bin/env python3
"""
demo_depth_potential_field.py
=============================
Standalone visualization demo for the Depth-to-Potential-Field pipeline.

Displays side-by-side:
1. RGB Frame (with optional depth overlay)
2. Probability Map (M_acc)
3. Repulsive Potential Field (U_rep) with Gradient Arrows (quiver)

Supports:
- Webcam (source index)
- Video files
- Directory of images
- TRT Engine or HuggingFace fallback

Usage:
    python demo_depth_potential_field.py --source 0 --engine path/to/model.engine
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
# matplotlib imports moved inside main() / plotting blocks to avoid global import errors

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.mapping.depth.depth_engine_trt import DepthEngineTRT, DepthEngineTRTConfig
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel, DepthAnythingV2Config
from sparx_agency.core.mapping.costmap.potential_mapper import PotentialMapper, PotentialMapperConfig


def parse_args():
    p = argparse.ArgumentParser(description="Depth-to-Potential-Field Demo")
    
    # Input Source
    p.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or path to video/image")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=infinite)")
    
    # Depth Model
    p.add_argument("--engine", type=str, default="/home/daphnaa/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-batch1.engine", 
                   help="Path to TensorRT .engine file (uses TRT if provided)")


    p.add_argument("--hf-model", type=str, default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", 
                   help="HuggingFace model ID (fallback if no engine provided)")
    p.add_argument("--device", type=str, default="cuda", help="Device for HF model (cuda/cpu)")

    # Intrinsics (minimal set for demo)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--hfov-deg", type=float, default=70.0, help="Horizontal FOV for synthetic intrinsics")

    # Mapper Config
    p.add_argument("--alpha", type=float, default=0.3, help="EMA decay factor (0-1)")
    p.add_argument("--sigma-m", type=float, default=0.3, help="Potential field sigma (metres)")
    p.add_argument("--res-m", type=float, default=0.10, help="Grid resolution (metres)")
    p.add_argument("--range-min", type=float, default=0.2, help="Min range (metres)")
    p.add_argument("--range-max", type=float, default=20.0, help="Max range (metres)")

    p.add_argument("--size-m", type=float, default=30.0, help="Grid side length (metres)")
    p.add_argument("--pitch", type=float, default=0.0, help="Camera pitch down (degrees)")
    p.add_argument("--camera-height", type=float, default=1.0, help="Camera height (metres)")
    
    # Navigation Goal
    p.add_argument("--goal-fwd", type=float, default=5.0, help="Goal forward position (meters)")
    p.add_argument("--goal-left", type=float, default=0.0, help="Goal left position (meters)")
    p.add_argument("--zeta", type=float, default=0.05, help="Navigation gain (attractive strength)")



    
    # Visualization
    p.add_argument("--no-show", action="store_true", help="Don't open matplotlib window")
    p.add_argument("--out-dir", type=str, help="Save frames to directory")
    
    return p.parse_args()


def get_intrinsics(args):
    """Compute synthetic intrinsics from HFOV and resolution."""
    w, h = args.width, args.height
    hfov_rad = np.deg2rad(args.hfov_deg)
    fx = w / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return Intrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)


def main():
    args = parse_args()
    intr = get_intrinsics(args)

    # 1. Initialize Depth Model
    depth_model = None
    if args.engine:
        print(f"[demo] Attempting TensorRT Engine: {args.engine}")
        try:
            depth_cfg = DepthEngineTRTConfig(engine_path=args.engine)
            depth_model = DepthEngineTRT(depth_cfg)
            # Force load to check for deserialization errors early
            depth_model.infer_depth(np.zeros((args.height, args.width, 3), dtype=np.uint8))
            print("[demo] TensorRT Engine loaded successfully.")
        except Exception as e:
            print(f"[warning] TensorRT load failed: {e}")
            print(f"[demo] Falling back to HuggingFace model: {args.hf_model}")
            depth_model = None

    if depth_model is None:
        depth_cfg = DepthAnythingV2Config(model_id=args.hf_model, device=args.device)
        depth_model = DepthAnythingV2DepthModel(depth_cfg)


    # 2. Initialize PotentialMapper
    mapper_cfg = PotentialMapperConfig(
        resolution_m=args.res_m,
        size_m=args.size_m,
        alpha=args.alpha,
        sigma_m=args.sigma_m,
        pitch_deg=args.pitch,
        height_m=args.camera_height,
        zeta=args.zeta,
    )



    mapper = PotentialMapper(mapper_cfg)
    mapper.set_goal(args.goal_fwd, args.goal_left)


    # 3. Open Video Source
    try:
        src = int(args.source)
    except ValueError:
        src = args.source
    
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[error] Could not open source: {src}")
        return

    # 4. Setup Visualization
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.canvas.manager.set_window_title("Depth-to-Potential-Field Pipeline (Revision 5)")
        
        im_rgb = axs[0].imshow(np.zeros((args.height, args.width, 3), dtype=np.uint8))
        axs[0].set_title("RGB Input")
        axs[0].axis("off")

        im_depth = axs[1].imshow(np.zeros((args.height, args.width, 3), dtype=np.uint8))
        axs[1].set_title("Metric Depth (m)")
        axs[1].axis("off")
        
        im_prob = axs[2].imshow(np.zeros(mapper.grid_shape), vmin=0, vmax=1.0, cmap="hot", origin="lower")
        axs[2].set_title(f"Prob Map (α={args.alpha})")
        
        im_pot = axs[3].imshow(np.zeros(mapper.grid_shape), vmin=0, vmax=1.0, cmap="viridis", origin="lower")
        axs[3].set_title("Field (Cleaned Walls)")

        
        # Plot Goal as a red X
        # World -> Grid
        gn_fwd = int((args.goal_fwd - mapper._origin) / args.res_m)
        gn_left = int((-mapper._origin - args.goal_left) / args.res_m)
        axs[3].plot(gn_left, gn_fwd, 'rx', markersize=12, markeredgewidth=3, label="Goal")
        axs[3].legend(loc='upper right')

        # Wall visualization storage
        w_lines = []




        
        q_arrows = None # for quiver
    
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    frame_idx = 0
    print("[demo] Starting processing loop. Press Ctrl+C to stop.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            # Pre-process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != args.height or rgb.shape[1] != args.width:
                rgb = cv2.resize(rgb, (args.width, args.height))

            # Step 1: Infer Depth
            t0 = time.perf_counter()
            depth_m = depth_model.infer_depth(rgb)
            dt_depth = (time.perf_counter() - t0) * 1000

            # Step 2: Update Mapper
            t0 = time.perf_counter()
            mapper.step(depth_m, intr)
            dt_map = (time.perf_counter() - t0) * 1000

            if frame_idx % 10 == 0:
                print(f"[frame {frame_idx:04d}] depth: {dt_depth:.1f}ms | map: {dt_map:.1f}ms")

            # Update Viz
            if not args.no_show:
                # RGB
                im_rgb.set_data(rgb)

                # Metric Depth Panel
                depth_norm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                im_depth.set_data(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
                
                # Grid Updates
                im_prob.set_data(mapper.get_prob_map())
                im_pot.set_data(mapper.get_potential_map())
                
                # Update wall lines (Revision 5)
                for l in w_lines:
                    l.remove()
                w_lines = []
                if hasattr(mapper, '_wall_segments') and mapper._wall_segments.size > 0:
                    for (x1, y1, x2, y2) in mapper._wall_segments:
                        l = axs[2].plot([x1, x2], [y1, y2], color='cyan', linewidth=1)[0]
                        w_lines.append(l)

                # Quiver plots (Gradients)

                if frame_idx % 5 == 0:
                    grad = mapper.get_total_gradient() # COMBINED Repulsive + Attractive
                    n = mapper.grid_shape[0]
                    step = max(1, n // 20)
                    X_q, Y_q = np.meshgrid(np.arange(0, n, step), np.arange(0, n, step), indexing='xy')
                    U_q = grad[::step, ::step, 1]
                    V_q = grad[::step, ::step, 0]
                    
                    if q_arrows:
                        q_arrows.remove()
                    q_arrows = axs[3].quiver(X_q, Y_q, U_q, V_q, color="cyan", scale=5, width=0.005)


                plt.pause(0.001)



            if args.out_dir:
                # Save just the depth/map if needed, or the whole figure
                out_path = Path(args.out_dir) / f"frame_{frame_idx:04d}.png"
                if not args.no_show:
                    plt.savefig(str(out_path))
                else:
                    cv2.imwrite(str(out_path), frame)

    except KeyboardInterrupt:
        print("\n[demo] Interrupted by user.")
    finally:
        cap.release()
        if not args.no_show:
            plt.close('all')
        print("[demo] Finished.")


if __name__ == "__main__":
    main()
