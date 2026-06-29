import argparse
import os
import yaml
import cv2
import numpy as np

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel

large_engine = "/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-batch1.engine"
small_engine = "/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3-SMALL/DA3-SMALL.fp16-batch1.engine"

camera_yaml = "/home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_crop_504_280.yaml"

da3_large = DA3TensorRTModel(
    engine_path=large_engine,
    yaml_path=camera_yaml,
)

da3_small = DA3TensorRTModel(
    engine_path=small_engine,
    yaml_path=camera_yaml,
)

from sparx_agency.robots.common.helpers import load_intrinsics_from_yaml as _load_intrinsics

def load_intrinsics_from_yaml(path):
    fx, fy, cx, cy = _load_intrinsics(path, prefer_projection=False)
    return fx, fy, cx, cy, 0.5 * (fx + fy)


def save_depth_outputs(name, depth_m, out_dir,roi):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{name}_depth_m.npy"), depth_m)

    valid = np.isfinite(depth_m)
    vis = np.zeros_like(depth_m, dtype=np.uint8)
    x1, y1, x2, y2 = roi
    if np.any(valid):
        d = depth_m[valid]
        lo, hi = np.percentile(d, [2, 98])
        norm = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = (norm * 255).astype(np.uint8)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite(os.path.join(out_dir, f"{name}_depth_vis.png"), vis)


def compute_metrics(pred_m, gt_m, min_depth=0.2, max_depth=10.0):
    mask = (
        np.isfinite(pred_m)
        & np.isfinite(gt_m)
        & (gt_m > min_depth)
        & (gt_m < max_depth)
        & (pred_m > min_depth)
        & (pred_m < max_depth)
    )

    if mask.sum() == 0:
        return None

    err = pred_m[mask] - gt_m[mask]
    abs_err = np.abs(err)

    return {
        "valid_pixels": int(mask.sum()),
        "mae_m": float(np.mean(abs_err)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "median_abs_err_m": float(np.median(abs_err)),
        "bias_m": float(np.mean(err)),
        "mae_percent_of_gt": float(100.0 * np.mean(abs_err / gt_m[mask])),
    }


def run_da3_metric_large(image_bgr, focal_px):
    """
    Replace this function with your actual DA3METRIC-LARGE inference call.

    Important:
    DA3METRIC-LARGE output is canonical depth.
    Convert to meters using:
        metric_depth = focal * net_output / 300
    """
    raise NotImplementedError


def run_da3_small(image_bgr):
    """
    Replace this function with your actual DA3-SMALL inference call.

    Important:
    DA3-SMALL is usually not directly metric like DA3METRIC-LARGE.
    For fair accuracy comparison, align it to GT by scale/shift or compare relative depth.
    """
    raise NotImplementedError

def robust_depth_from_roi(depth_m, roi, min_depth=0.2, max_depth=20.0):
    x1, y1, x2, y2 = roi
    patch = depth_m[y1:y2, x1:x2]

    valid = (
        np.isfinite(patch)
        & (patch > min_depth)
        & (patch < max_depth)
    )

    vals = patch[valid]

    if vals.size < 20:
        return None

    return {
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "std": float(np.std(vals)),
        "n": int(vals.size),
    }


def align_scale_shift_to_gt(pred, gt, min_depth=0.2, max_depth=10.0):
    mask = (
        np.isfinite(pred)
        & np.isfinite(gt)
        & (gt > min_depth)
        & (gt < max_depth)
    )

    x = pred[mask].reshape(-1)
    y = gt[mask].reshape(-1)

    if x.size < 100:
        return pred, None

    A = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]

    aligned = scale * pred + shift
    return aligned, {"scale": float(scale), "shift": float(shift)}

def print_engine_io(model, label):
    print(f"\n=== {label} engine I/O ===")
    for i in range(model.engine.num_io_tensors):
        name = model.engine.get_tensor_name(i)
        shape = model.engine.get_tensor_shape(name)
        mode = model.engine.get_tensor_mode(name)
        dtype = model.engine.get_tensor_dtype(name)
        print(i, name, shape, mode, dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--camera-yaml", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gt-depth", default=None)
    args = parser.parse_args()

    fx, fy, cx, cy, focal = load_intrinsics_from_yaml(args.camera_yaml)

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    print(f"Image shape: {image_bgr.shape}")
    print(f"fx={fx:.3f}, fy={fy:.3f}, focal_avg={focal:.3f}")

    # Run inference
    depth_large_raw, _ = da3_large.infer_all(image_bgr)
    metric_depth = (focal * depth_large_raw) / 300.0

    depth_small_raw, _ = da3_small.infer_all(image_bgr)

    known_regions = [
        {
            "name": "door",
            "roi": (200, 50, 250, 100),
            "gt_m": 5.5,
        },
        {
            "name": "right_wall",
            "roi": (340, 150, 380, 260),
            "gt_m": 2.10,
        },
    ]

    rows = []

    # Calibrate DA3-SMALL using first known region
    calib_region = known_regions[1]

    large_calib = robust_depth_from_roi(metric_depth, calib_region["roi"])
    small_calib = robust_depth_from_roi(depth_small_raw, calib_region["roi"])

    small_scale = calib_region["gt_m"] / small_calib["median"]

    depth_small_scaled_m = depth_small_raw * small_scale

    print("Large calib:", large_calib)
    print("Small calib raw:", small_calib)
    print("Small scale:", small_scale)

    print_engine_io(da3_large, "DA3METRIC-LARGE")
    print_engine_io(da3_small, "DA3-SMALL")

    save_depth_outputs("da3_metric_large", metric_depth, args.out_dir, calib_region["roi"])
    save_depth_outputs("da3_small_raw", depth_small_raw, args.out_dir, calib_region["roi"])
    save_depth_outputs("da3_small_scaled", depth_small_scaled_m, args.out_dir, calib_region["roi"])

    for region in known_regions:
        name = region["name"]
        roi = region["roi"]
        gt_m = region.get("gt_m", None)

        large_stats = robust_depth_from_roi(metric_depth, roi)
        small_stats_raw = robust_depth_from_roi(depth_small_raw, roi)

        if large_stats is None or small_stats_raw is None:
            print(f"Skipping {name}: invalid ROI/depth")
            continue

        large_pred = large_stats["median"]
        small_raw_pred = small_stats_raw["median"]

        row = {
            "region": name,
            "large_m": large_pred,
            "small_raw": small_raw_pred,
            "small_raw_p10": small_stats_raw["p10"],
            "small_raw_p90": small_stats_raw["p90"],
            "small_raw_std": small_stats_raw["std"],
        }

        if gt_m is not None:
            row.update({
                "gt_m": gt_m,
                "large_err_m": large_pred - gt_m,
                "large_abs_err_m": abs(large_pred - gt_m),
                "large_err_pct": 100.0 * abs(large_pred - gt_m) / gt_m,
            })

        rows.append(row)

    for row in rows:
        print(row)



if __name__ == "__main__":
    main()