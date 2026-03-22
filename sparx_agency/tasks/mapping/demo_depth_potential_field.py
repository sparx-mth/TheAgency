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

# Global variables for mouse callback calibration
click_pt = None
PANE_W, PANE_H = 320, 320
MAP_SIZE = 6.5

def mouse_callback(event, x, y, flags, param):
    global click_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        # Bottom-right pane: nav potential (col >= PANE_W, row >= 2*PANE_H)
        if x >= PANE_W and y >= 2 * PANE_H:
            px = x - PANE_W
            fwd, left = get_click_pt(y, px)
            click_pt = (fwd, left)
            print(f"Clicked Nav Potential at: fwd={fwd:.2f}m, left={left:.2f}m")
        # Bottom-left pane: attractive field
        elif x < PANE_W and y >= 2 * PANE_H:
            px = x
            fwd, left = get_click_pt(y, px)
            click_pt = (fwd, left)
            print(f"Clicked Attractive Field at: fwd={fwd:.2f}m, left={left:.2f}m")

def get_click_pt(y, px):
    py = y - 2 * PANE_H
    fwd = (1.0 - py / (PANE_H - 1)) * MAP_SIZE
    left = (0.5 - px / (PANE_W - 1)) * MAP_SIZE
    return fwd, left

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
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        # Top row (RGB, Depth) uses video aspect ratio
        cam_pane_h = int(PANE_W * frame_h / frame_w)
        # Middle + Bottom rows use SQUARE panes for top-down maps
        PANE_H = PANE_W  # square!

        depth_raw, point_cloud = depth_model.infer_all(frame)
        potential_mapper.update(point_cloud)

        u_total = potential_mapper.get_total_potential()
        m_temp = np.flipud(potential_mapper.get_temp_map())
        m_acc = np.flipud(potential_mapper.get_prob_map())
        m_nav = np.flipud(potential_mapper.get_nav_map())
        n = potential_mapper.grid_shape[0]

        map_size = potential_mapper.cfg.size_m
        half = 0.5 * map_size

        def world_to_screen(fwd, left):
            fwd = max(0.0, min(map_size, fwd))
            left = max(-half, min(half, left))
            y_px = int((1.0 - fwd / map_size) * (PANE_H - 1))
            x_px = int((0.5 - left / map_size) * (PANE_W - 1))
            return x_px, y_px

        # --- Top row: camera views (use video aspect ratio) ---
        vis_rgb = cv2.resize(frame, (PANE_W, cam_pane_h))
        depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_depth = cv2.resize(cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET), (PANE_W, cam_pane_h))

        # --- Middle row: occupancy maps (SQUARE panes) ---
        temp_vis = np.nan_to_num(m_temp, nan=0.0)
        acc_vis = np.nan_to_num(m_acc, nan=0.0)

        vis_temp = cv2.resize(
            cv2.applyColorMap((np.clip(temp_vis, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT),
            (PANE_W, PANE_H))
        vis_acc = cv2.resize(
            cv2.applyColorMap((np.clip(acc_vis, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_BONE),
            (PANE_W, PANE_H))

        # --- Bottom row: potential fields (SQUARE panes) ---
        # Attractive field
        u_att = np.flipud(potential_mapper.get_attractive_potential())
        att_norm = cv2.normalize(u_att, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_att = cv2.resize(cv2.applyColorMap(att_norm, cv2.COLORMAP_COOL), (PANE_W, PANE_H))

        # Nav potential (combined)
        u_total_vis = np.flipud(u_total)
        pot_norm = cv2.normalize(u_total_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_pot = cv2.resize(cv2.applyColorMap(pot_norm, cv2.COLORMAP_VIRIDIS), (PANE_W, PANE_H))

        # Obstacle overlay (yellow)
        nav_vis = np.nan_to_num(m_nav, nan=0.0)
        occ = nav_vis > potential_mapper.cfg.occ_thresh
        yellow_layer = np.zeros_like(vis_pot)
        occ_img = cv2.resize(occ.astype(np.uint8) * 255, (PANE_W, PANE_H), interpolation=cv2.INTER_NEAREST)
        yellow_layer[occ_img > 0] = (0, 255, 255)
        vis_pot = cv2.addWeighted(vis_pot, 1.0, yellow_layer, 0.6, 0)

        # --- Arrows on nav potential pane ---
        grad = potential_mapper.get_total_gradient()
        res = potential_mapper.cfg.resolution_m
        step = max(1, n // 18)
        arrow_len = 10.0

        for r in range(0, n, step):
            for c in range(0, n, step):
                f_val = r * res
                l_val = half - c * res
                sx, sy = world_to_screen(f_val, l_val)
                v = grad[r, c]
                mag = np.hypot(v[0], v[1])
                if mag < 0.01:  # lowered from 0.2 so attraction arrows show everywhere
                    continue
                dx = int(v[1] * arrow_len)
                dy = int(-v[0] * arrow_len)
                if 0 <= sx < PANE_W and 0 <= sy < PANE_H:
                    cv2.arrowedLine(vis_pot, (sx, sy), (sx + dx, sy + dy), (255, 255, 0), 1, tipLength=0.3)

        # --- Markers ---
        gx, gy = world_to_screen(args.goal_fwd, args.goal_left)
        rx, ry = world_to_screen(0.0, 0.0)

        for vis in (vis_att, vis_pot):
            cv2.drawMarker(vis, (rx, ry), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
            cv2.drawMarker(vis, (gx, gy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
            cv2.circle(vis, (rx, ry), 4, (255, 0, 0), -1)

        # Decision arrow (red)
        rv = grad[0, n // 2]
        dec_dx = int(rv[1] * 40)
        dec_dy = int(-rv[0] * 40)
        cv2.arrowedLine(vis_pot, (rx, ry), (rx + dec_dx, ry + dec_dy), (0, 0, 255), 2)

        # --- Labels ---
        cv2.putText(vis_rgb, "RGB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_depth, "Depth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_temp, "Temp Occupancy", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_acc, "Accumulated Occupancy", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_att, "Attractive Field", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_pot, "Nav Potential (att+rep)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if click_pt is not None:
            cv2.putText(
                vis_pot,
                f"Last Click: {click_pt[0]:.2f}m, {click_pt[1]:.2f}m",
                (10, PANE_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

        # --- Assemble: top row may differ in height from middle/bottom ---
        top = np.hstack((vis_rgb, vis_depth))
        middle = np.hstack((vis_temp, vis_acc))
        bottom = np.hstack((vis_att, vis_pot))
        # Resize top row to match width of middle/bottom (2*PANE_W)
        target_w = 2 * PANE_W
        if top.shape[1] != target_w:
            top = cv2.resize(top, (target_w, cam_pane_h))
        cv2.imshow("Sparx Navigator", np.vstack((top, middle, bottom)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()