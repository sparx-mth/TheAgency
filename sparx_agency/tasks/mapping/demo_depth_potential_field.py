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
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--engine", type=str,
                   default="/home/daphnaa/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine")
    p.add_argument("--yaml", type=str,
                   default="/home/daphnaa/depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_laptop.yaml")
    p.add_argument("--res-m", type=float, default=0.05)
    p.add_argument("--size-m", type=float, default=10.0)
    p.add_argument("--sigma-m", type=float, default=0.10)
    p.add_argument("--zeta", type=float, default=0.5)
    p.add_argument("--goal-fwd", type=float, default=6.0)
    p.add_argument("--goal-left", type=float, default=-4.0)
    return p.parse_args()


def main():
    args = parse_args()
    # Explicitly DO NOT import or call matplotlib.pyplot here to avoid extra windows

    depth_model = DA3TensorRTModel(args.engine, args.yaml)
    potential_mapper = PotentialMapper(PotentialMapperConfig(
        resolution_m=args.res_m, size_m=args.size_m,
        sigma_m=args.sigma_m, zeta=args.zeta
    ))
    potential_mapper.set_goal(args.goal_fwd, args.goal_left)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    cv2.namedWindow("Sparx Navigator", cv2.WINDOW_AUTOSIZE)

    PANE = 320  # Keep it compact as requested

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Depth & Mapping
        depth_raw, point_cloud = depth_model.infer_all(frame)
        potential_mapper.update(point_cloud)

        # 2. Potential Calculation
        U_total = potential_mapper.get_potential_map()
        n = potential_mapper.grid_shape[0]


        # 3. Coordinate Mapping
        def world_to_screen(f, l):
            scale = PANE / args.size_m
            # Forward is -Y (Up), Left is -X (Left)
            sy = PANE - int(((f - potential_mapper._origin) / args.size_m) * PANE)
            sx = PANE - int(((l - potential_mapper._origin) / args.size_m) * PANE)
            return (sx, sy)

        # 4. Dashboard Panes
        vis_rgb = cv2.resize(frame, (PANE, PANE))

        depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_depth = cv2.resize(cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET), (PANE, PANE))

        # OCCUPANCY: Removed flip(0) to fix inversion
        prob_map = potential_mapper.get_prob_map()
        vis_prob = cv2.resize(cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_HOT), (PANE, PANE))

        # POTENTIAL: Using Viridis-like logic with clear obstacle contrast
        pot_norm = cv2.normalize(U_total, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_pot = cv2.resize(cv2.applyColorMap(255 - pot_norm, cv2.COLORMAP_VIRIDIS), (PANE, PANE))

        # 5. Field Arrows (Subsampled)
        grad = potential_mapper.get_total_gradient()
        step = max(1, n // 12)
        for r in range(0, n, step):
            for c in range(0, n, step):
                f_val = (r * args.res_m) + potential_mapper._origin
                l_val = (c * args.res_m) + potential_mapper._origin
                sx, sy = world_to_screen(f_val, l_val)
                v = grad[r, c]
                # Negative gradient for "downhill" flow
                dx, dy = int(v[1] * 12), int(v[0] * 12)
                if 0 <= sx < PANE and 0 <= sy < PANE:
                    cv2.arrowedLine(vis_pot, (sx, sy), (sx + dx, sy + dy), (200, 200, 0), 1, tipLength=0.3)

        # 6. Navigation Overlays
        gx, gy = world_to_screen(args.goal_fwd, args.goal_left)
        cv2.drawMarker(vis_pot, (gx, gy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 2)

        rx, ry = world_to_screen(0, 0)
        cv2.circle(vis_pot, (rx, ry), 5, (255, 0, 0), -1)

        # Decision Arrow (Red)
        rv = grad[n // 2, n // 2]
        cv2.arrowedLine(vis_pot, (rx, ry), (rx + int(rv[1] * 40), ry + int(rv[0] * 40)), (0, 0, 255), 2)

        # Final Stack
        top = np.hstack((vis_rgb, vis_depth))
        bottom = np.hstack((vis_prob, vis_pot))
        cv2.imshow("Sparx Navigator", np.vstack((top, bottom)))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()