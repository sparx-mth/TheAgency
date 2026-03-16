#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import yaml
from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel
from sparx_agency.core.mapping.costmap.potential_mapper import PotentialMapper, PotentialMapperConfig


def load_intrinsics(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Intrinsics(width=data['image_width'], height=data['image_height'],
                      fx=data['fx'], fy=data['fy'], cx=data['cx'], cy=data['cy'])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")# --source "/home/daphnaa/Videos/pp.mp4"
    p.add_argument("--engine", type=str,
                   default="/home/daphnaa/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine")
    p.add_argument("--yaml", type=str,
                   default="/home/daphnaa/depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_laptop.yaml")
    p.add_argument("--res-m", type=float, default=0.03)
    p.add_argument("--size-m", type=float, default=6.5)
    p.add_argument("--sigma-m", type=float, default=0.02)
    p.add_argument("--zeta", type=float, default=1.5)
    p.add_argument("--goal-fwd", type=float, default=3.0)
    p.add_argument("--goal-left", type=float, default=1.5)
    return p.parse_args()

# Global variables for mouse callback calibration
click_pt = None
PANE_W, PANE_H = 320, 320
MAP_SIZE = 6.5

def mouse_callback(event, x, y, flags, param):
    global click_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check which pane was clicked (bottom-right is potential mapper)
        if x >= PANE_W and y >= PANE_H:
            # Local coords to pane
            px, py = x - PANE_W, y - PANE_H
            # sy = (1.0 - fwd / map_size) * (PANE_H - 1)
            fwd = (1.0 - py / (PANE_H - 1)) * MAP_SIZE
            # sx = (0.5 - left / map_size) * (PANE_W - 1)
            left = (0.5 - px / (PANE_W - 1)) * MAP_SIZE
            click_pt = (fwd, left)
            print(f"Clicked Potential Map at: fwd={fwd:.2f}m, left={left:.2f}m")

def main():
    args = parse_args()
    global MAP_SIZE, PANE_W, PANE_H
    MAP_SIZE = args.size_m

    depth_model = DA3TensorRTModel(args.engine, args.yaml)
    potential_mapper = PotentialMapper(PotentialMapperConfig(
        resolution_m=args.res_m, size_m=args.size_m,
        sigma_m=args.sigma_m, zeta=args.zeta
    ))
    potential_mapper.set_goal(args.goal_fwd, args.goal_left)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    cv2.namedWindow("Sparx Navigator")
    cv2.setMouseCallback("Sparx Navigator", mouse_callback)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_h, frame_w = frame.shape[:2]
        PANE_H = int(PANE_W * frame_h / frame_w)

        # 1. Depth & Mapping
        depth_raw, point_cloud = depth_model.infer_all(frame)
        potential_mapper.update(point_cloud)

        # 2. Potential Calculation
        U_total = potential_mapper.get_total_potential()
        U_rep = potential_mapper.get_potential_map()
        n = potential_mapper.grid_shape[0]

        # 3. Coordinate Mapping
        map_size = potential_mapper.cfg.size_m
        half = 0.5 * map_size

        def world_to_screen(fwd, left):
            fwd = max(0.0, min(map_size, fwd))
            left = max(-half, min(half, left))
            sy = int((1.0 - fwd / map_size) * (PANE_H - 1))
            sx = int((0.5 - left / map_size) * (PANE_W - 1))
            return sx, sy

        # 4. Dashboard Panes
        vis_rgb = cv2.resize(frame, (PANE_W, PANE_H))
        depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_depth = cv2.resize(cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET), (PANE_W, PANE_H))

        # OCCUPANCY: Use flipud so row 0 (robot) is at screen bottom
        prob_map_raw = potential_mapper.get_prob_map()
        prob_map = np.flipud(prob_map_raw)
        p_vis = np.nan_to_num(prob_map, nan=0.0)
        vis_prob = cv2.resize(cv2.applyColorMap((p_vis * 255).astype(np.uint8), cv2.COLORMAP_HOT), (PANE_W, PANE_H))

        # POTENTIAL: use flipud
        U_vis = 0.7 * U_total + 0.3 * U_rep
        U_total_vis = np.flipud(U_vis)
        pot_norm = cv2.normalize(U_total_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_pot = cv2.resize(cv2.applyColorMap(pot_norm, cv2.COLORMAP_VIRIDIS), (PANE_W, PANE_H))

        # 5. Fixed-Length Arrows
        grad = potential_mapper.get_total_gradient()
        res = potential_mapper.cfg.resolution_m
        step = max(1, n // 18)
        ARROW_LEN = 10.0 # Fixed display length (pixels)

        for r in range(0, n, step):
            for c in range(0, n, step):
                f_val = r * res
                l_val = half - c * res # col 0 is far-left
                sx, sy = world_to_screen(f_val, l_val)
                v = grad[r, c]
                mag = np.hypot(v[0], v[1])
                if mag < 0.2: continue # Ignore weak fields for clarity

                # Normalized direction
                dx = int(-v[1] * ARROW_LEN)
                dy = int(-v[0] * ARROW_LEN)
                if 0 <= sx < PANE_W and 0 <= sy < PANE_H:
                    cv2.arrowedLine(vis_pot, (sx, sy), (sx + dx, sy + dy), (255, 255, 0), 1, tipLength=0.3)

        # 6. Navigation Overlays (Alpha-Transparent Yellow)
        gx, gy = world_to_screen(args.goal_fwd, args.goal_left)
        rx, ry = world_to_screen(0.0, 0.0)
        
        occ = np.nan_to_num(prob_map, nan=0.0) > potential_mapper.cfg.occ_thresh
        yellow_layer = np.zeros_like(vis_pot)
        occ_img = cv2.resize(occ.astype(np.uint8) * 255, (PANE_W, PANE_H), interpolation=cv2.INTER_NEAREST)
        yellow_layer[occ_img > 0] = (0, 255, 255)
        vis_pot = cv2.addWeighted(vis_pot, 1.0, yellow_layer, 0.6, 0) # Alpha overlay

        cv2.drawMarker(vis_pot, (rx, ry), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.drawMarker(vis_pot, (gx, gy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
        cv2.circle(vis_pot, (rx, ry), 4, (255, 0, 0), -1)

        # Decision Arrow (Red) - using robot cell (0, n//2)
        rv = grad[0, n // 2]
        dec_dx = int(-rv[1] * 40)
        dec_dy = int(-rv[0] * 40)
        cv2.arrowedLine(vis_pot, (rx, ry), (rx + dec_dx, ry + dec_dy), (0, 0, 255), 2)

        # Calibration Print
        if click_pt:
            cv2.putText(vis_pot, f"Last Click: {click_pt[0]:.2f}m, {click_pt[1]:.2f}m", (10, PANE_H - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Final Stack
        top = np.hstack((vis_rgb, vis_depth))
        bottom = np.hstack((vis_prob, vis_pot))
        cv2.imshow("Sparx Navigator", np.vstack((top, bottom)))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
