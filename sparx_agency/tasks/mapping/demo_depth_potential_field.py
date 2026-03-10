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
    p.add_argument("--hfov-deg", type=float, default=90.0, help="Horizontal FOV for synthetic intrinsics")

    # Mapper Config
    p.add_argument("--alpha", type=float, default=0.3, help="EMA decay factor (0-1)")
    p.add_argument("--sigma-m", type=float, default=0.3, help="Potential field sigma (metres)")
    p.add_argument("--res-m", type=float, default=0.10, help="Grid resolution (metres)")
    p.add_argument("--range-min", type=float, default=0.2, help="Min range (metres)")
    p.add_argument("--range-max", type=float, default=20.0, help="Max range (metres)")

    p.add_argument("--size-m", type=float, default=10.0, help="Grid side length (metres)")
    p.add_argument("--pitch", type=float, default=0.0, help="Camera pitch down (degrees)")
    p.add_argument("--camera-height", type=float, default=1.0, help="Camera height (metres)")
    
    # Navigation Goal
    p.add_argument("--goal-fwd", type=float, default=3.0, help="Goal forward position (meters)")
    p.add_argument("--goal-left", type=float, default=3.0, help="Goal left position (meters)")
    p.add_argument("--zeta", type=float, default=0.3, help="Navigation gain (attractive strength)")
    
    # Temporal Smoothing
    p.add_argument("--alpha-depth", type=float, default=0.5, help="EMA decay factor for depth map (0-1). Lower means smoother but more lag.")



    
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
    
    print(f"[demo] Intrinsics: {intr}")
    print(f"[demo] Mapper Origin: {mapper._origin}m, Resolution: {mapper.cfg.resolution_m}m")


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
        
        # Calculation of plot extent [Lateral(m), Forward(m)]
        # Lateral X: -Left. So -5m (Left=+5) to 5m (Left=-5)
        side = mapper.cfg.size_m
        hs = side / 2.0
        extent = [-hs, hs, -hs, hs] 
        
        im_prob = axs[2].imshow(np.zeros(mapper.grid_shape), vmin=0, vmax=1.0, cmap="hot", origin="lower", extent=extent)
        axs[2].set_title(f"Prob Map (α={args.alpha})")
        axs[2].set_xlabel("Lateral (+Right) (m)")
        axs[2].set_ylabel("Forward (m)")
        
        im_pot = axs[3].imshow(np.zeros(mapper.grid_shape), vmin=0, vmax=1.0, cmap="viridis", origin="lower", extent=extent)
        axs[3].set_title("Potential Field")
        axs[3].set_xlabel("Lateral (+Right) (m)")
        
        # Plot Goal as a red X (-goal_left, goal_fwd)
        axs[3].plot(-args.goal_left, args.goal_fwd, 'rx', markersize=12, markeredgewidth=3, label="Goal")
        
        # Plot Robot Position
        axs[3].plot(0.0, 0.0, 'bo', markersize=8, label="Robot")
        
        axs[3].legend(loc='upper right')

        # Wall visualization storage
        w_lines = []
        
        # Robot vector storage
        r_vector = None
        
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
            depth_raw = depth_model.infer_depth(rgb)
            dt_depth = (time.perf_counter() - t0) * 1000
            
            # Apply Temporal Depth Smoothing (EMA)
            if 'depth_ema' not in locals():
                depth_ema = depth_raw.copy()
            else:
                depth_ema = args.alpha_depth * depth_raw + (1.0 - args.alpha_depth) * depth_ema

            # Step 2: Update Mapper
            t0 = time.perf_counter()
            mapper.step(depth_ema, intr)
            dt_map = (time.perf_counter() - t0) * 1000
            
            if frame_idx % 10 == 0:
                print(f"[frame {frame_idx:04d}] depth: {dt_depth:.1f}ms | map: {dt_map:.1f}ms")

            # Update Viz
            if not args.no_show:
                # RGB
                im_rgb.set_data(rgb)

                # Metric Depth Panel
                depth_norm = cv2.normalize(depth_ema, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                im_depth.set_data(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
                
                # Flip data for Lateral (+Right) display
                im_prob.set_data(np.flip(mapper.get_prob_map(), axis=1))
                
                # 4. Field Panel with combined potential heatmap
                U_rep = mapper.get_potential_map()
                
                # U_att = zeta * dist (Conic Potential) - computed in meters
                rows = np.arange(mapper.grid_shape[0])
                cols = np.arange(mapper.grid_shape[1])
                fwd_coords = rows * args.res_m + mapper._origin
                left_coords = cols * args.res_m + mapper._origin # pos = col*res + origin
                fwd_grid, left_grid = np.meshgrid(fwd_coords, left_coords, indexing='ij')
                
                dist_grid = np.sqrt((args.goal_fwd - fwd_grid)**2 + (args.goal_left - left_grid)**2)
                U_att = args.zeta * dist_grid
                U_total = U_rep + U_att
                
                if np.max(U_total) > 1e-6:
                    im_pot.set_data(np.flip(U_total / np.max(U_total), axis=1))
                else:
                    im_pot.set_data(np.flip(U_total, axis=1))
                
                # 5. Wall Segments (Plotly-style lines in meters)
                for l in w_lines: l.remove()
                # w_lines = []
                # if mapper._wall_segments.size > 0:
                #     for line in mapper._wall_segments:
                #         x1, y1, x2, y2 = line
                #         # Grid -> World
                #         # Lateral X = -Left
                #         lx1 = - (x1 * args.res_m + mapper._origin)
                #         lx2 = - (x2 * args.res_m + mapper._origin)
                #         # Forward Y = Fwd
                #         ly1 = y1 * args.res_m + mapper._origin
                #         ly2 = y2 * args.res_m + mapper._origin
                        
                #         # wl, = axs[3].plot([lx1, lx2], [ly1, ly2], 'y-', linewidth=3)
                #         # w_lines.append(wl)
                #         # Also plot on Prob Map for sync check
                #         # wl2, = axs[2].plot([lx1, lx2], [ly1, ly2], 'c-', linewidth=1)
                #         # w_lines.append(wl2)

                # 6. Gradients & Robot Vector
                if frame_idx % 2 == 0:
                    grad = mapper.get_total_gradient() # Combined [Fwd, Left]
                    n = mapper.grid_shape[0]
                    # Subsample for quiver
                    step = max(2, n // 10)
                    sl = slice(0, n, step)
                    
                    # Lateral screen velocity = -v_left = dPotential/dLeft
                    U_qv = grad[sl, sl, 1]
                    # Forward screen velocity = v_fwd = -dPotential/dFwd
                    V_qv = grad[sl, sl, 0]
                    
                    rows_idx, cols_idx = np.indices(grad.shape[:2])
                    # Cell index -> Left -> Lateral (X)
                    qx = -(cols_idx[sl, sl] * args.res_m + mapper._origin)
                    # Cell index -> Forward (Y)
                    qy = rows_idx[sl, sl] * args.res_m + mapper._origin
                    
                    if q_arrows: q_arrows.remove()
                    q_arrows = axs[3].quiver(qx, qy, U_qv, V_qv, color='cyan', alpha=0.5, scale=5.0, width=0.003)
                    
                    # Robot Navigation Vector at (0,0)
                    if r_vector: r_vector.remove()
                    
                    # Get exact gradient at robot position (approx center cells)
                    cr, cc = n//2, n//2
                    rv = grad[cr, cc] # [Fwd, Left]
                    
                    # Plot vector in meters. Velocity = -grad.
                    # v_lat = - (-dP/dLeft) = dP/dLeft? No.
                    # rv[1] is dP/dLeft. Move away from Left hill: v_left = -rv[1].
                    # Lateral Screen X: Move away from Left hill = move towards smaller Left.
                    # Smaller Left is Larger Lateral. So Lateral Screen U = dP/dLeft?
                    # Let's just trust v_lat = rv[1] and v_fwd = -rv[0].
                    r_vector = axs[3].quiver(0.0, 0.0, rv[1], rv[0], color='red', scale=2.0, width=0.015, zorder=10)

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
