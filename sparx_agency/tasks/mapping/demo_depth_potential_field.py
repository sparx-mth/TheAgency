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
    p.add_argument("--sigma-m", type=float, default=0.02)
    p.add_argument("--zeta", type=float, default=1.5)
    p.add_argument("--goal-fwd", type=float, default=3.0)
    p.add_argument("--goal-left", type=float, default=-2.0)
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
        U_total = potential_mapper.get_total_potential()
        n = potential_mapper.grid_shape[0]


        # 3. Coordinate Mapping
        map_size = potential_mapper.cfg.size_m
        half = 0.5 * map_size

        def world_to_screen(fwd, left):
            # clamp to map bounds first
            fwd = max(-half, min(half, fwd))
            left = max(-half, min(half, left))

            # normalize to [0,1]
            u = (fwd + half) / map_size  # forward: -half->0, +half->1
            v = (left + half) / map_size  # left:    -half->0, +half->1

            # convert to pixels
            sx = int((1.0 - v) * (PANE - 1))  # left+ goes to screen-left (flip)
            sy = int((1.0 - u) * (PANE - 1))  # forward+ goes up (flip)
            return sx, sy

        print("[dbg] w2s(0,0) =", world_to_screen(0.0, 0.0), "expected ~", (PANE // 2, PANE // 2))
        # 4. Dashboard Panes
        vis_rgb = cv2.resize(frame, (PANE, PANE))

        depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_depth = cv2.resize(cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET), (PANE, PANE))

        # OCCUPANCY: Removed flip(0) to fix inversion
        prob_map = np.flipud(potential_mapper.get_prob_map())
        p_vis = np.nan_to_num(prob_map, nan=0.0)  # unknown -> 0
        print("prob stats:",
              "min", float(np.nanmin(prob_map)),
              "max", float(np.nanmax(prob_map)),
              "p>occ_thresh", int(np.sum(prob_map > potential_mapper.cfg.occ_thresh)))
        vis_prob = cv2.resize(cv2.applyColorMap((p_vis * 255).astype(np.uint8), cv2.COLORMAP_HOT), (PANE, PANE))

        # POTENTIAL: Using Viridis-like logic with clear obstacle contrast
        U_total_vis = np.flipud(U_total)
        pot_norm = cv2.normalize(U_total_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_pot = cv2.resize(cv2.applyColorMap(pot_norm, cv2.COLORMAP_VIRIDIS), (PANE, PANE))

        # 5. Field Arrows (Subsampled)
        grad = potential_mapper.get_total_gradient()

        step = max(1, n // 20)
        scale = 20.0
        min_len = 3.0

        cr, cc = n // 2, n // 2
        v_center = grad[cr, cc]
        print("[dbg] center v", v_center)

        for r in range(0, n, step):
            for c in range(0, n, step):
                res = potential_mapper.cfg.resolution_m
                half = 0.5 * map_size
                f_val = -half + r * res
                right_val = -half + c * res
                l_val = -right_val

                sx, sy = world_to_screen(f_val, l_val)
                v = grad[r, c]

                dx_f = v[1] * scale
                dy_f = -v[0] * scale

                L = float(np.hypot(dx_f, dy_f))
                if L < 1e-6:
                    continue
                if L < min_len:
                    dx_f *= (min_len / (L + 1e-6))
                    dy_f *= (min_len / (L + 1e-6))

                dx = int(round(dx_f))
                dy = int(round(dy_f))

                if 0 <= sx < PANE and 0 <= sy < PANE:
                    cv2.arrowedLine(vis_pot, (sx, sy), (sx + dx, sy + dy),
                                    (200, 200, 0), 1, tipLength=0.25)

        # 6. Navigation Overlays
        gx, gy = world_to_screen(args.goal_fwd, args.goal_left)
        rx, ry = world_to_screen(0.0, 0.0)

        print("[dbg] goal screen", gx, gy, "robot screen", rx, ry)

        def clamp_xy(x, y):
            return max(0, min(PANE - 1, x)), max(0, min(PANE - 1, y))

        gx, gy = clamp_xy(gx, gy)
        rx, ry = clamp_xy(rx, ry)
        cv2.drawMarker(vis_pot, (rx, ry), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)  # robot big cross
        cv2.drawMarker(vis_pot, (gx, gy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 40, 2)  # goal big X

        # cv2.drawMarker(vis_pot, (gx, gy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 2)

        # rx, ry = world_to_screen(0, 0)
        cv2.circle(vis_pot, (rx, ry), 5, (255, 0, 0), -1)

        # Decision Arrow (Red)
        rv = grad[n // 2, n // 2]
        print("rv fwd,left:", rv)
        cv2.arrowedLine(vis_pot, (rx, ry),
                        (rx + int(rv[1] * 200), ry + int(-rv[0] * 200)),
                        (0, 0, 255), 2)

        print(f"[debug]: cv2 arrowed line cv2.arrowedLine(vis_pot, (rx, ry),(rx + int(rv[1] * 200), "
              f"ry + int(-rv[0] * 200)),(0, 0, 255), 2) \n rv[1]*200 = {rv[1]*200}, rv[0]*200 = {rv[0]*200}")

        # Final Stack
        top = np.hstack((vis_rgb, vis_depth))
        bottom = np.hstack((vis_prob, vis_pot))
        cv2.imshow("Sparx Navigator", np.vstack((top, bottom)))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()